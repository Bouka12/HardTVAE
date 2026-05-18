"""
Anlyse the effect of hardness weighting on the gradient decent.
    *   Compare the GD in the baseline and the HardVAE
    *   Check the loss components in the HardVAE (RE and KLD)
    
"""
import matplotlib.pyplot as plt
import os
import sys
# Walk up to the working directory (parent of hardvae_code)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List
import numpy as np
import warnings
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from models.hardness import HardnessCalculator, CVAEHardnessIntegrator
from load_data import load_data
from models.hardtvae import TabularCVAE, prepare_dataloader
from utils import preprocess_data

# device related -> GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def capture_grad_stats(model):
    """Calcule la norme L2 globale des gradients du modèle."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

class HardnessAwareCVAETrainer:
    """Trainer for hardness-aware tabular CVAE."""

    def __init__(self,
                 model:                TabularCVAE,
                 hardness_calculator:  HardnessCalculator,
                 hardness_integrator:  CVAEHardnessIntegrator,
                 device=DEVICE):
        self.model                = model.to(device)
        self.hardness_calculator  = hardness_calculator
        self.hardness_integrator  = hardness_integrator
        self.device               = device
        self.optimizer            = optim.Adam(model.parameters(), lr=1e-3)
        self.hardness_scores      = None

    # ── Hardness (unchanged) ──────────────────────────────────────────────────
    def calculate_hardness_scores(self, X, y, hardness_metrics):
        if hardness_metrics is not None:
            hardness_df = self.hardness_calculator.calculate_hardness_scores(
                X, y, hardness_metrics
            )
            if hardness_df.empty:
                warnings.warn(
                    "No valid hardness scores calculated. "
                    "Skipping hardness-aware training for this iteration."
                )
                self.hardness_scores = None
                return None
            self.hardness_scores = hardness_df.values
            return self.hardness_scores
        else:
            self.hardness_scores = None

    # ── Loss ──────────────────────────────────────────────────────────────────
    def cvae_loss(self,
                  num_out:  torch.Tensor,
                  cat_outs: List[torch.Tensor],
                  x:        torch.Tensor,
                  mu:       torch.Tensor,
                  logvar:   torch.Tensor,
                  weights:  Optional[torch.Tensor] = None,
                  beta:     float = 1.0
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tabular reconstruction loss  +  KL  +  optional hardness weighting.

        Reconstruction
        --------------
        Numerical features  → MSE  (sum over columns, per sample)
                              matches original TVAE paper
        Categorical groups  → Cross-entropy per group (per sample)
                              matches original TVAE paper

        Importance weighting  (with instance hardness score)
        --------------------
        Instance-level `weights` are multiplied into the per-sample combined
        reconstruction loss BEFORE the mean is taken 
        """
        n_num    = self.model.n_numerical
        cat_dims = self.model.cat_dims

        # ── Numerical MSE  (per sample) ───────────────────────────────
        x_num        = x[:, :n_num]                            # (B, n_num)
        recon_loss   = torch.sum((x_num - num_out) ** 2, dim=1)  # (B,)

        # ── Categorical cross-entropy  (per sample, accumulated) ──────
        offset = n_num
        for i, cat_dim in enumerate(cat_dims):
            x_cat_onehot = x[:, offset : offset + cat_dim]    # (B, cat_dim)
            target_idx   = x_cat_onehot.argmax(dim=1)          # (B,)  int
            ce           = F.cross_entropy(
                cat_outs[i], target_idx, reduction='none'
            )                                                   # (B,)
            recon_loss   = recon_loss + ce
            offset      += cat_dim

        # ── Instance-level importance weighting ───────────────────────
        #    Your contribution: down/up-weight each sample before averaging.
        #    Shape of weights must broadcast to (B,).
        if weights is not None:
            recon_loss = recon_loss * weights

        recon_loss = recon_loss.mean()

        # ── KL divergence (unchanged) ─────────────────────────────────
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ── Training epoch ────────────────────────────────────────────────────────
    def train_epoch(self, dataloader: DataLoader,
                    epoch: int, total_epochs: int,
                    beta: float = 1.0) -> dict:
        self.model.train()
        total_loss = total_recon = total_kl = 0.0

        # Initialize the totals
        epoch_grad_norms = []

        for data, conditions, indices in dataloader:
            data       = data.to(self.device)
            conditions = conditions.to(self.device)

            # Hardness weights
            if self.hardness_scores is not None:
                batch_hardness = self.hardness_scores[indices.numpy()]
                weights = torch.tensor(
                    self.hardness_integrator.get_sample_weights(
                        batch_hardness, epoch, total_epochs
                    ),
                    dtype=torch.float32, device=self.device
                )
            else:
                weights = None

            # Forward  ← unpacks the new 4-tuple
            num_out, cat_outs, mu, logvar = self.model(data, conditions)

            # Loss  ← now tabular-aware
            loss, recon_loss, kl_loss = self.cvae_loss(
                num_out, cat_outs, data, mu, logvar, weights, beta
            )

            self.optimizer.zero_grad()
            loss.backward()

            # Capture the gradient before the step
            epoch_grad_norms.append(capture_grad_stats(self.model))

            self.optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()

        n = len(dataloader)
        return {
            'total_loss': total_loss  / n,
            'grad_norm':  np.mean(epoch_grad_norms),
            'recon_loss': total_recon / n,
            'kl_loss':    total_kl    / n,
        }

    # ── Generation ────────────────────────────────────────────────────────────
    def generate_samples(self,
                         conditions: torch.Tensor,
                         n_samples:  int) -> torch.Tensor:
        """
        Generate synthetic samples and return a full feature vector:
          [ numerical_cols  |  one-hot group 0  |  one-hot group 1  | ... ]

        Categorical groups are sampled from the softmax distribution over
        the decoder logits (rather than argmax) to preserve diversity.
        """
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)

            if conditions.dim() == 1:
                conditions = conditions.unsqueeze(0)
            conditions = conditions.repeat(n_samples, 1).to(self.device)

            num_out, cat_outs = self.model.decode(z, conditions)

            # Sample each categorical group from its softmax distribution
            cat_samples = []
            for i, cat_dim in enumerate(self.model.cat_dims):
                probs   = F.softmax(cat_outs[i], dim=1)          # (N, cat_dim)
                indices = torch.multinomial(probs, num_samples=1).squeeze(1)
                one_hot = F.one_hot(indices, num_classes=cat_dim).float()
                cat_samples.append(one_hot)

            # Reconstruct the full feature vector in the same column order
            # as the preprocessor output: [numerical | cat_group_0 | cat_group_1 | ...]
            parts = [num_out] + cat_samples
            synthetic_data = torch.cat(parts, dim=1)

        return synthetic_data


def save_run_metrics(dataset_name: str, training_loss: list, combination_name: str, seed: int, results_dir: str):
    """
    Save per-epoch metrics for one run to a CSV.
    combination_name format: "dataset,metric,seed,strategy"
    """
    # Replace the broken split approach
    if combination_name == "TVAE":
        model, metric, strategy = "TVAE", "None", "None"
    else:
        # combination_name = "HardTVAE_{metric}_{strategy}"
        # Strip the prefix and use known strategies as anchor
        rest = combination_name[len("HardTVAE_"):]          # e.g. "TD_P_self_paced"
        strategy = next(s for s in ['self_paced', 'curriculum', 'static'] 
                        if rest.endswith(s))
        metric = rest[:-(len(strategy) + 1)]                # strip "_{strategy}"
        model = "HardTVAE"

    df = pd.DataFrame(training_loss)  # already has: epoch, total_loss, recon_loss, kl_loss, grad_norm
    df['model']   = model
    df['dataset']  = dataset_name
    df['metric']   = metric
    df['seed']     = seed
    df['strategy'] = strategy

    save_dir = os.path.join(results_dir, "run_metrics")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{dataset_name}_{combination_name}_{str(seed)}.csv"
    df.to_csv(os.path.join(save_dir, filename), index=False)
    print(f"  Saved metrics → {filename}")

# from load_data import load_data
import random
N_EPOCHS = 150  # Total number of epochs for training
CURRICULUM_EPOCHS = (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)  # Epochs for each hardness strategy
MASTER_SEED = 42  # Master seed for reproducibility
random.seed(MASTER_SEED)  # Random state for reproducibility
random_seeds = random.sample(range(1, 10**6), 10) # -> change to 10 later  # Random seeds for different runs FIX IT TO 5 RANDOM SEEDS
## RANDOM SEEDS AUGMENTATION TO 10 FOR STABILITY

def main():

    datasets = ["BCWDD", "ILPD"]   #[ 'Hypothyroid', 'NewThyroid1', 'Vertebral']
    hardness_metrics = [None, 'DCP', 'TD_P', 'CLD', 'MV', 'CB', 'N2', 'LSC', 
                   'LSR', 'Harmfulness', 'F1', 'F4']
    seeds = list(random_seeds)  # Different seeds for reproducibility CHANGE THIS 
    print(f"seeds = {seeds}")

    # Create the csv files of RESULTS_ALL and KS_RESULTS and update them with the results of each dataset and hardness metric on the airflow
    # Store results for all datasets and hardness metrics
    RESULTS_DIR = "RESULTS-GRADIENT-ANALYSIS" # -> Results directory path -> (23/04/2026) for test change the path
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hardness_dir = f"{RESULTS_DIR}/hardness_scores"
    os.makedirs(hardness_dir, exist_ok=True)

    for dataset_name, seed in itertools.product(datasets, seeds):
        # set_seed(seed)
        DATA_PATH = f'data/processed/{dataset_name}.csv'
        (X_train, y_train, X_val, y_val, X_test, y_test) = load_data(
            DATA_PATH, random_state=seed
        )

        # ── Preprocessing  ──────────────────────────────────
        print("Preprocessing data...")
        preprocessor = preprocess_data(random_state=seed)
        X_train_proc, df_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc, df_val_proc   = preprocessor.transform(X_val)
        X_test_proc, df_test_proc  = preprocessor.transform(X_test)
        data_info = preprocessor.data_info          # ← {'n_numerical', 'cat_dims'}
        input_dim = X_train_proc.shape[1]           # numerical + sum(cat_dims)
        print(f"data_info of {dataset_name}:\n {data_info}")
        # print(f"data_info of the df_train_proc:\n {df_train_proc.info()}")
        # Identify labels
        label_counts = pd.Series(y_train).value_counts()
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        n_samples_needed = label_counts[majority_label] - label_counts[minority_label]

        # CTVAE the modified version of the CVAE 

        for hardness_metric in hardness_metrics:
            # Save the distribution of hardness scores for the current metric and dataset and training seed and save it as csv, in a separate script we plot the average distribution of hardness scores for each metric and dataset over the different seeds (mean and std shaded plot)
            if hardness_metric is not None:
                hardness_calculator = HardnessCalculator(random_state=seed)
                hardness_scores = hardness_calculator.calculate_hardness_scores(X_train_proc, y_train, [hardness_metric])
                hardness_df = pd.DataFrame(hardness_scores, columns=[hardness_metric])
                hardness_df.to_csv(os.path.join(hardness_dir, f"{dataset_name}_{hardness_metric}_seed{seed}_hardness_scores.csv"), index=False)
                print(f"  Saved hardness scores → {dataset_name}_{hardness_metric}_seed{seed}_hardness_scores.csv")

            strategies = ['static'] if hardness_metric is None else ['curriculum', 'static', 'self_paced']
            for strategy in strategies:
                model_name = "TVAE" if hardness_metric is None else f"HardTVAE_{hardness_metric}_{strategy}"
                print(f"\n--- Training {model_name} for {dataset_name} ---")

                model = TabularCVAE(
                    input_dim=X_train_proc.shape[1], 
                    latent_dim=5, condition_dim=1,
                    hidden_dims=[128, 64, 32],
                    data_info=data_info)
                trainer = HardnessAwareCVAETrainer(
                    model= model,
                    hardness_calculator=HardnessCalculator(random_state=seed),
                    hardness_integrator=CVAEHardnessIntegrator(hardness_strategy=strategy),
                    device=DEVICE
                )

                trainer.calculate_hardness_scores(X_train_proc, y_train, [hardness_metric])
                dataloader = prepare_dataloader(X_train_proc, y_train)
                if trainer.hardness_scores is None and hardness_metric is not None:
                    print(f"Skipping: invalid hardness scores for {dataset_name}")
                    continue

                # Store the the values of the loss components during training and plot them for each combination and save them in subfolder in hardvae_code
                training_loss = []
                # grad_norms = []
                for epoch in range(N_EPOCHS):
                    metrics = trainer.train_epoch(dataloader, epoch, N_EPOCHS)
                    metrics['epoch'] = epoch
                    training_loss.append(metrics)
                    # grad_norms.append(metrics)
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch:2d}: Loss={metrics['total_loss']:.4f}, "
                            f"Recon={metrics['recon_loss']:.4f}, KL={metrics['kl_loss']:.4f}")

                # Save raw run data (we plot them in a separate script, averaged over the runs)
                save_run_metrics(dataset_name=dataset_name, training_loss=training_loss, combination_name=model_name, seed=seed, results_dir=RESULTS_DIR)
                



if __name__ == "__main__":
    main()

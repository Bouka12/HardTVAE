"""
TabularCVAE  —  TVAE-style Conditional VAE for tabular data
============================================================

KEY CHANGES vs original TabularCVAE:
  1. __init__ accepts `data_info` dict  {'n_numerical': int, 'cat_dims': [int,...]}
  2. Decoder split into:
       - shared trunk  (same hidden layers as before, minus final projection)
       - self.num_head : Linear → one output per numerical feature  (no activation)
       - self.cat_heads: ModuleList of Linears, one per categorical group (logits)
  3. forward() / decode() now return  (num_out, cat_outs, mu, logvar)
  4. generate_samples() samples from softmax of cat logits → one-hot → concat

KEY CHANGES vs original HardnessAwareCVAETrainer:
  1. cvae_loss()  uses
       - MSE  per numerical column  (same as original TVAE paper)
       - CrossEntropy per categorical group  (same as original TVAE paper)
       - instance-level importance weights applied to combined per-sample loss
         BEFORE the mean  (your contribution, preserved exactly)
  2. train_epoch() unpacks the new (num_out, cat_outs, mu, logvar) forward signature
  3. generate_samples() reconstructs full feature vector (numerical + one-hot)

Everything else (encoder, reparameterize, hardness score calculation,
optimizer, DataLoader helpers, main() wiring) is unchanged.
"""

import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List, Dict
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ── project imports (unchanged) ───────────────────────────────────────────────
from hardness_ import HardnessCalculator, CVAEHardnessIntegrator
from load_data import load_data
from gradients_stability_analysis.loss_viz import loss_plots
# from classifier_eval import evaluate_classification_model
from utils import preprocess_data          # ← use the new version

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class TabularCVAE(nn.Module):
    """
    Tabular-aware Conditional VAE.

    Parameters
    ----------
    input_dim     : total width of the preprocessed feature matrix
                    (= n_numerical + sum(cat_dims))
    latent_dim    : size of latent space
    condition_dim : size of the condition vector (1 for binary label)
    hidden_dims   : hidden layer widths for encoder and decoder trunk
    data_info     : dict with keys
                      'n_numerical' – int, number of numerical output cols
                      'cat_dims'    – list[int], one-hot group sizes in order
    """

    def __init__(self,
                 input_dim:    int,
                 latent_dim:   int,
                 condition_dim: int,
                 hidden_dims:  List[int],
                 data_info:    Dict):
        super().__init__()

        self.input_dim     = input_dim
        self.latent_dim    = latent_dim
        self.condition_dim = condition_dim
        self.n_numerical   = data_info['n_numerical']
        self.cat_dims      = data_info['cat_dims']        # e.g. [3, 2, 5]

        # ── Encoder (unchanged) ───────────────────────────────────────
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2),
            ]
            prev_dim = h
        self.encoder   = nn.Sequential(*encoder_layers)
        self.fc_mu     = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # ── Decoder trunk  (hidden layers only — no final projection) ─
        trunk_layers = []
        prev_dim = latent_dim + condition_dim
        for h in reversed(hidden_dims):
            trunk_layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2),
            ]
            prev_dim = h
        self.decoder_trunk = nn.Sequential(*trunk_layers)
        self._trunk_out_dim = prev_dim            # remember for the heads

        # ── Output heads ──────────────────────────────────────────────
        # Numerical head: raw linear output (QuantileTransformer → ~Normal)
        self.num_head = nn.Linear(self._trunk_out_dim, self.n_numerical)

        # Categorical heads: one linear per group, outputs are raw logits
        # fed into cross-entropy loss during training, softmax during generation
        self.cat_heads = nn.ModuleList([
            nn.Linear(self._trunk_out_dim, cat_dim)
            for cat_dim in self.cat_dims
        ])

    # ── Encoder helpers ───────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor, c: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor
                       ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Decoder ───────────────────────────────────────────────────────────────
    def decode(self, z: torch.Tensor, c: torch.Tensor
               ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        num_out  : (batch, n_numerical)   – numerical reconstruction
        cat_outs : list of (batch, cat_dim_i) – logits per categorical group
        """
        h       = self.decoder_trunk(torch.cat([z, c], dim=1))
        num_out  = self.num_head(h)
        cat_outs = [head(h) for head in self.cat_heads]
        return num_out, cat_outs

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, c: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor],
                           torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        num_out, cat_outs = self.decode(z, c)
        return num_out, cat_outs, mu, logvar


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════════

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
        # epoch_grad_norms = []

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
            # epoch_grad_norms.append(capture_grad_stats(self.model))

            self.optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()

        n = len(dataloader)
        return {
            'total_loss': total_loss  / n,
            # 'grad_norm':  np.mean(epoch_grad_norms),
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


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader helper 
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataloader(X: np.ndarray, y: np.ndarray,
                       batch_size: int = 32) -> DataLoader:
    X_tensor       = torch.tensor(X, dtype=torch.float32)
    y_tensor       = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    indices_tensor = torch.arange(len(X))
    dataset        = TensorDataset(X_tensor, y_tensor, indices_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main  — 
# ══════════════════════════════════════════════════════════════════════════════

N_EPOCHS          = 150
CURRICULUM_EPOCHS = (N_EPOCHS * 0.3, N_EPOCHS * 0.3, N_EPOCHS * 0.4)
MASTER_SEED       = 42
random.seed(MASTER_SEED)
random_seeds      = random.sample(range(1, 10**6), 1)


def main():
    datasets = ['Hypothyroid']
    # datasets = [
    #     'BCWDD', 'HeartCleveland', 'Hepatitis', 'Hypothyroid',
    #     'ILPD', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral'
    # ]
    hardness_metrics = [None, 'kDN']
    # hardness_metrics = [None, 'kDN', 'DS', 'DCP', 'TD_P',
    #                'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 
    #                'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    RESULTS_DIR = "RESULTS-TEST2304"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plots_dir = f"{RESULTS_DIR}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    all_results_path = os.path.join(RESULTS_DIR, 'all_classification_results.csv')
    ks_results_path  = os.path.join(RESULTS_DIR, 'all_ks_results.csv')
    if not os.path.exists(all_results_path):
        pd.DataFrame().to_csv(all_results_path, index=False)
    if not os.path.exists(ks_results_path):
        pd.DataFrame().to_csv(ks_results_path, index=False)

    fid_save_dir = "fidelity_results"
    os.makedirs(fid_save_dir, exist_ok=True)

    # Build valid combinations (same logic as original)
    combinations = []
    for dataset_name, hardness_metric, seed in itertools.product(
        datasets, hardness_metrics, seeds
    ):
        strategies = ['static'] if hardness_metric is None \
                     else ['curriculum', 'static', 'self_paced']
        for strategy in strategies:
            combinations.append((dataset_name, hardness_metric, seed, strategy))

    for dataset_name, hardness_metric, seed, weighting_strategy in combinations:
        print(f"\n=== Dataset: {dataset_name} | metric: {hardness_metric} | "
              f"strategy: {weighting_strategy} | seed: {seed} ===")

        plot_dir = f"{plots_dir}/plots_{hardness_metric}_{weighting_strategy}_{seed}"
        os.makedirs(plot_dir, exist_ok=True)

        DATA_PATH = f'data/processed/{dataset_name}.csv'
        (X_train, y_train, X_val, y_val, X_test, y_test) = load_data(
            DATA_PATH, random_state=seed
        )

        # ── Preprocessing  ← CHANGED ──────────────────────────────────
        print("Preprocessing data...")
        preprocessor = preprocess_data(random_state=seed)
        X_train_proc, df_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc, df_val_proc   = preprocessor.transform(X_val)
        X_test_proc, df_test_proc  = preprocessor.transform(X_test)

        ## ---- DEBUG ---- ##
        print(f"TRAINING DATA AFTER PREPROCESSING WITH DATA PREPROCESS for \n {dataset_name}")
        print(f"X_train_proc\n {X_test_proc[:4, :]}")

        data_info = preprocessor.data_info          # ← {'n_numerical', 'cat_dims'}
        input_dim = X_train_proc.shape[1]           # numerical + sum(cat_dims)

        # ── Hardness on raw (unscaled) X_train  (unchanged) ───────────
        hardness_calc = HardnessCalculator(random_state=seed)

        # ── Model  ← CHANGED: pass data_info ──────────────────────────
        print("Initializing TabularCVAE model...")
        model = TabularCVAE(
            input_dim     = input_dim,
            latent_dim    = 5,
            condition_dim = 1,
            hidden_dims   = [128, 64, 32],
            data_info     = data_info,              # ← NEW argument
        )

        hardness_integrator = CVAEHardnessIntegrator(
            hardness_strategy=weighting_strategy
        )
        trainer = HardnessAwareCVAETrainer(
            model                = model,
            hardness_calculator  = hardness_calc,
            hardness_integrator  = hardness_integrator,
            device               = DEVICE,
        )


        # Hardness scores computed on raw X_train (before scaling) — unchanged
        trainer.calculate_hardness_scores(X_train_proc, y_train, [hardness_metric])
        if trainer.hardness_scores is None and hardness_metric is not None:
            print(f"Skipping: invalid hardness scores for {dataset_name}")
            continue

        # DataLoader now uses preprocessed data  ← CHANGED
        dataloader = prepare_dataloader(X_train_proc, y_train, batch_size=32)

        # ── Training loop (unchanged) ──────────────────────────────────
        print("Training TabularCVAE...")
        training_loss = []
        for epoch in range(N_EPOCHS):
            metrics          = trainer.train_epoch(dataloader, epoch, N_EPOCHS, beta=1.0)
            metrics['epoch'] = epoch
            training_loss.append(metrics)
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={metrics['total_loss']:.4f}  "
                      f"Recon={metrics['recon_loss']:.4f}  "
                      f"KL={metrics['kl_loss']:.4f}")

        combination_name = f"{dataset_name},{hardness_metric},{seed},{weighting_strategy}"
        loss_plots(training_loss, combination_name, "training_loss_plots_HardCTVAE-2")

        # ── Generation (unchanged call signature) ─────────────────────
        print("Generating synthetic samples...")
        majority_class_label = int(np.argmax(np.bincount(y_train)))
        minority_class_label = int(np.argmin(np.bincount(y_train)))
        N_SAMPLES = (np.bincount(y_train)[majority_class_label]
                     - np.bincount(y_train)[minority_class_label])

        minority_condition = torch.tensor([minority_class_label])
        synthetic_samples  = trainer.generate_samples(minority_condition, n_samples=N_SAMPLES)
        synthetic_samp     = synthetic_samples.cpu().numpy()
        synthetic_labels   = np.full((synthetic_samp.shape[0], 1), minority_class_label)


        # ── Utility evaluation (unchanged) ────────────────────────────
        print("Evaluating utility of synthetic samples...")
        synthetic_df = pd.DataFrame(
            np.concatenate([synthetic_samp, synthetic_labels], axis=1),
            columns=[f'feature_{i}' for i in range(X_train_proc.shape[1])] + ['label']
        )
        print(f"generated samples for {dataset_name}\n {synthetic_df.head()}")
    #     original_df = pd.DataFrame(
    #         X_train_proc,
    #         columns=[f'feature_{i}' for i in range(X_train_proc.shape[1])]
    #     )
    #     original_df['label'] = y_train
    #     combined_data = pd.concat([original_df, synthetic_df], ignore_index=True)

    #     results_dir = (f"{RESULTS_DIR}/{dataset_name}_{hardness_metric}"
    #                    f"_seed{seed}_weighting_{weighting_strategy}")
    #     os.makedirs(results_dir, exist_ok=True)

    #     X_aug = np.array(combined_data.iloc[:, :-1])
    #     y_aug = np.array(combined_data.iloc[:, -1])
    #     result = evaluate_classification_model( # <- TO CHANGE FOR mle_xgboost.py
    #         X_aug, y_aug, X_test_proc, y_test,       # ← use preprocessed test
    #         k_folds=3, hardness_metric=hardness_metric, random_state=seed
    #     )
    #     for r in result:
    #         r.update({
    #             'dataset':           dataset_name,
    #             'hardness_metric':   hardness_metric,
    #             'seed':              seed,
    #             'weighting_strategy': weighting_strategy,
    #         })

    #     result_cols = list(result[0].keys())
    #     pd.DataFrame(result, columns=result_cols).to_csv(
    #         all_results_path, mode='a',
    #         header=not os.path.exists(all_results_path), index=False
    #     )
    #     experiment_dir = f"{results_dir}/{dataset_name}_{hardness_metric}_seed{seed}"
    #     os.makedirs(experiment_dir, exist_ok=True)
    #     pd.DataFrame(result).to_csv(
    #         os.path.join(experiment_dir, 'classification_results.csv'), index=False
    #     )

    # print(f"\n✅ Done. Results saved in {all_results_path}")


if __name__ == "__main__":
    main()
"""
visualization ofthe loss of the hardvae for each configuration 
    * store the value of the loss for each component in each training epoch
    * plot them together using line plot and store
    * store the line plot witht the name of the combination in a loss plots path directory
    * Analysis -> Decision on the hyperparameters of the HardVAE 
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.data import DataLoader, TensorDataset
# from classifier_eval import evaluate_classification_model
from typing import Tuple, Optional
from models.hardness import HardnessCalculator, CVAEHardnessIntegrator
# import the load_data from load_data.py
from load_data import load_data


def loss_plots(training_loss:list,combination_name:str, plots_path:str):
    """ the fucntion plot the loss components values during the training per epoch and save the plots
    Inputs:
        - training_loss (list): a list of dictionaries with keys ("epoch", "recon_loss", "total_loss", "kl_loss")
        - combination_name (str): the combination (datasetname, hardness_metric, seed, weighting_strategy)
        - plots_path (str): the saving path
    Function:
        - plot the components in one plot using a line plot (for now)
        - save the plot using the combination_name in the plots_path
    """

    # Extract the data for the plot
    epochs = [d['epoch'] for d in training_loss]
    recon_loss = [d['recon_loss'] for d in training_loss]
    total_loss = [d['total_loss'] for d in training_loss]
    kl_loss = [d['kl_loss'] for d in training_loss]

    # create the figure
    plt.figure(figsize=(10, 6))

    # Trace the different losses
    plt.plot(epochs, total_loss, label="Total Loss", marker='o', linestyle="-")
    plt.plot(epochs, recon_loss, label="Reconstruction Loss", marker='s', linestyle="--")
    plt.plot(epochs, kl_loss, label="KL loss", marker='^', linestyle=":")

    # Add  labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title(f"Training Loss Components: {combination_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure the plots path exists
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Save the figure
    save_filename = f"{combination_name.replace(' ', '_').replace(',', "_")}_loss.png"
    save_full_path = os.path.join(plots_path, save_filename)
    plt.savefig(save_full_path)
    plt.close('all')

    print(f"Plot saved in {save_full_path}")

    return save_full_path




# # device related -> GPU or CPU
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# class TabularCVAE(nn.Module):
#     """
#     Conditional Variational Autoencoder for tabular data.
#     """
    
#     def __init__(self, input_dim: int, latent_dim: int, condition_dim: int, 
#                  hidden_dims: list = [128, 64]):
#         super(TabularCVAE, self).__init__()
        
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.condition_dim = condition_dim
        
#         # Encoder
#         encoder_layers = []
#         prev_dim = input_dim + condition_dim
        
#         for hidden_dim in hidden_dims:
#             encoder_layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(0.2)
#             ])
#             prev_dim = hidden_dim
        
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # Latent space
#         self.fc_mu = nn.Linear(prev_dim, latent_dim)
#         self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
#         # Decoder
#         decoder_layers = []
#         prev_dim = latent_dim + condition_dim
        
#         for hidden_dim in reversed(hidden_dims):
#             decoder_layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.Dropout(0.2)
#             ])
#             prev_dim = hidden_dim
        
#         decoder_layers.append(nn.Linear(prev_dim, input_dim))
#         self.decoder = nn.Sequential(*decoder_layers)
    
#     def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Encode input with condition to latent parameters."""
#         x_c = torch.cat([x, c], dim=1)
#         h = self.encoder(x_c)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar
    
#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """Reparameterization trick."""
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         """Decode latent representation with condition."""
#         z_c = torch.cat([z, c], dim=1)
#         return self.decoder(z_c)
    
#     def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Forward pass through CVAE."""
#         mu, logvar = self.encode(x, c)
#         z = self.reparameterize(mu, logvar)
#         x_recon = self.decode(z, c)
#         return x_recon, mu, logvar


# class HardnessAwareCVAETrainer:
#     """
#     Trainer for hardness-aware CVAE with integrated hardness scoring.
#     """
    
#     def __init__(self, model: TabularCVAE, hardness_calculator: HardnessCalculator,
#                  hardness_integrator: CVAEHardnessIntegrator, device: str = DEVICE):
#         self.model = model.to(device)
#         self.hardness_calculator = hardness_calculator
#         self.hardness_integrator = hardness_integrator
#         self.device = device
        
#         self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
#         self.hardness_scores = None
        
#     # MOVE THIS FUNCTION IN THE HEAD OF THE HARDVAE  NOT INSIDE
#     def calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, 
#                                   hardness_metrics: list) -> np.ndarray:
#         """Calculate and store hardness scores for the dataset."""
#         if hardness_metrics is not None:
#             print("hardness metrics is not None 'calculate_hardness_scores")
#             hardness_df = self.hardness_calculator.calculate_hardness_scores(X, y, hardness_metrics)
#             # print(f" Number of NaN values in hardness_df: {hardness_df.isnull().sum().sum()}")
#             # Check emptiness of hardness_df
#             if hardness_df.empty:
#                 warnings.warn("No valid hardness scores calculated by HardnessCalculator. Skipping hardness-aware training for this iteration.")
#                 self.hardness_scores = None
#                 return None
            
#             # Use the first metric as primary hardness score (can be modified)
#             # primary_metric = hardness_metrics[metric_index] # later we want to loop over the hardness metrics to get results with each hardness metric
#             # print(f"Using hardness metric in HardnessAwareCVATrainer: {primary_metric}")
#             self.hardness_scores = hardness_df.values #hardness_df[primary_metric].values
#             # We can return the whole dataset and later for the weight we 
#             return self.hardness_scores
#         else:
#             self.hardness_scores = None
    
#     def cvae_loss(self, x_recon: torch.Tensor, x: torch.Tensor, 
#                   mu: torch.Tensor, logvar: torch.Tensor, 
#                   weights: Optional[torch.Tensor] = None, 
#                   beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Calculate CVAE loss with optional hardness weighting.
        
#         Args:
#             x_recon: Reconstructed input
#             x: Original input
#             mu: Latent mean
#             logvar: Latent log variance
#             weights: Sample weights based on hardness
#             beta: Beta parameter for beta-VAE
#         """
#         # Reconstruction loss (MSE)
#         recon_loss = torch.sum((x - x_recon) ** 2, dim=1)
        
#         # Apply hardness weights if provided
#         if weights is not None:
#             recon_loss = recon_loss * weights
        
#         recon_loss = torch.mean(recon_loss)
        
#         # KL divergence loss
#         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
#         kl_loss = torch.mean(kl_loss)
        
#         # Total loss
#         total_loss = recon_loss + beta * kl_loss
        
#         return total_loss, recon_loss, kl_loss
    
#     def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int, 
#                     beta: float = 1.0) -> dict:
#         """Train for one epoch with hardness-aware weighting."""
#         self.model.train()
#         total_loss = 0
#         total_recon_loss = 0
#         total_kl_loss = 0
        
#         for batch_idx, (data, conditions, indices) in enumerate(dataloader):
#             data = data.to(self.device)
#             conditions = conditions.to(self.device)
            
#             # Get hardness weights for this batch
#             if self.hardness_scores is not None:
#                 batch_hardness = self.hardness_scores[indices.numpy()]
#                 weights = self.hardness_integrator.get_sample_weights(
#                     batch_hardness, epoch, total_epochs
#                 )
#                 weights = torch.tensor(weights, dtype=torch.float32, device = self.device).to(self.device)
#             else:
#                 weights = None
            
#             # Forward pass
#             x_recon, mu, logvar = self.model(data, conditions)
            
#             # Calculate loss
#             loss, recon_loss, kl_loss = self.cvae_loss(
#                 x_recon, data, mu, logvar, weights, beta
#             )
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
#             total_recon_loss += recon_loss.item()
#             total_kl_loss += kl_loss.item()
        
#         n_batches = len(dataloader)
#         return {
#             'total_loss': total_loss / n_batches,
#             'recon_loss': total_recon_loss / n_batches,
#             'kl_loss': total_kl_loss / n_batches
#         }
    
#     def generate_samples(self, conditions: torch.Tensor, n_samples: int) -> torch.Tensor:
#         """Generate synthetic samples given conditions."""
#         self.model.eval()
        
#         with torch.no_grad():
#             # Sample from latent space
#             z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            
#             # Repeat conditions for all samples
#             if conditions.dim() == 1:
#                 conditions = conditions.unsqueeze(0)
#             conditions = conditions.repeat(n_samples, 1).to(self.device)
            
#             # Generate samples
#             synthetic_data = self.model.decode(z, conditions)
        
#         return synthetic_data


# def prepare_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> DataLoader:
#     """Prepare DataLoader with indices for hardness tracking."""
#     # Convert to tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     print(f"shape of X_tensor: {X_tensor.shape}")
#     y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Condition
#     print(f"shape of y_tensor: {y_tensor.shape}")
#     indices_tensor = torch.arange(len(X))
    
#     # Create dataset and dataloader
#     dataset = TensorDataset(X_tensor, y_tensor, indices_tensor)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     return dataloader

# from load_data import load_data
# import random
# N_EPOCHS = 150  # Total number of epochs for training
# CURRICULUM_EPOCHS = (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)  # Epochs for each hardness strategy
# MASTER_SEED = 42  # Master seed for reproducibility
# random.seed(MASTER_SEED)  # Random state for reproducibility
# random_seeds = random.sample(range(1, 10**6), 1)  # Random seeds for different runs FIX IT TO 5 RANDOM SEEDS
# ## RANDOM SEEDS AUGMENTATION TO 10 FOR STABILITY

# def main():

#     datasets = ['BCWDD', 'HeartCleveland', 'Hepatitis', 'Hypothyroid', 'ILPD', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral']

#     # hardness_metrics = [None, 'kDN']
#     hardness_metrics = [None, 'kDN', 'DS', 'DCP', 'TD_P',
#                    'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 
#                    'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
#     seeds = list(random_seeds)  # Different seeds for reproducibility CHANGE THIS 
#     print(f"seeds = {seeds}")


#     # Generate valid combinations
#     combinations = []
#     for dataset_name, hardness_metric, seed in itertools.product(datasets, hardness_metrics, seeds):
#         if hardness_metric is None:
#             strategies = ['static']
#         else:
#             strategies = ['curriculum', 'static', 'self_paced']
#         for strategy in strategies:
#             combinations.append((dataset_name, hardness_metric, seed, strategy))

#     # Loop through valid combinations
#     for dataset_name, hardness_metric, seed, weighting_strategy in combinations: #itertools.product(datasets, hardness_metrics, seeds, weighting_strategies):
#         print(f"\n=== Dataset: {dataset_name} | hardness metric: {hardness_metric} | Weighting strategy: {weighting_strategy} | Seed: {seed} ===")
#         print("="*50)
#         print("\n=== Step 1: Loading dataset ===")
#         # # Path of the plots directory for the synthetic data evaluator: plots are specific to each combination and dataset so needs to make specific directory for each combination plots:
#         # plot_dir = f"{plots_dir}/plots_{hardness_metric}_{weighting_strategy}_{seed}"
#         # os.makedirs(plot_dir, exist_ok=True)
        
#         # # Load dataset
#         # Train-test split -> change path 
#         DATA_PATH = f'data/processed/{dataset_name}.csv'
        
#         # DO TRAIN-VAL-TEST SPLIT: VAL FOR MLE (TUNING) -> (23/04/2026) DONE 
#         X_train, y_train, X_val, y_val, X_test, y_test, majority_data, minority_data, imbalance_ratio = load_data(DATA_PATH, random_state=seed)

#         print("2. Initializingg the HardnessCalculator with seed...")        
#         hardness_calc = HardnessCalculator(random_state=seed)

    
#         # Initialize CVAE model
#         print("3. Initializing CVAE model...")
#         input_dim = X_train.shape[1]
#         latent_dim = 5
#         condition_dim = 1  # Binary condition
        
#         model = TabularCVAE(
#             input_dim=input_dim,
#             latent_dim=latent_dim,
#             condition_dim=condition_dim,
#             hidden_dims=[128, 64, 32]
#         )
#         # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\\n")
        
#         # Initialize hardness integrator
#         print("4. Initializing the VAE HardnessIntegrator...")
#         hardness_integrator = CVAEHardnessIntegrator(
#             hardness_strategy= weighting_strategy 
#         )
        
#         # Initialize trainer
#         print("Initializing the HardnessAware VAE Trainer...")
#         trainer = HardnessAwareCVAETrainer(
#             model=model,
#             hardness_calculator=hardness_calc,
#             hardness_integrator=hardness_integrator,
#             device=DEVICE
#         )
        
#         print("Calculating the hardness score for training data...")
#         # Calculate hardness scores for training data
#         trainer.calculate_hardness_scores(X_train, y_train, [hardness_metric])

#         if trainer.hardness_scores is None:
#             if  hardness_metric is not None:
#                 print(f"Skipping combination due to invalid hardness scores: Dataset={dataset_name}, Metric={hardness_metric}, Seed={seed}, Strategy={weighting_strategy}")
#                 continue # skip to the next iteration of the loop


#         print("Preparing the dataloader...")
#         # Prepare data
#         dataloader = prepare_dataloader(X_train, y_train, batch_size=32)
        
#         # Training loop
#         print("5. Training CVAE with hardness awareness...")
#         n_epochs = N_EPOCHS
        
#         # Store the the values of the loss components during training and plot them for each combination and save them in subfolder in hardvae_code
#         training_loss = []
#         for epoch in range(n_epochs):
#             metrics = trainer.train_epoch(dataloader, epoch, n_epochs, beta=1.0)
#             metrics['epoch'] = epoch
#             training_loss.append(metrics)            
#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch:2d}: Loss={metrics['total_loss']:.4f}, "
#                     f"Recon={metrics['recon_loss']:.4f}, KL={metrics['kl_loss']:.4f}")
        
#         # Plot the loss componenets
#         print("Visualizing the loss componenets...")
        
#         # from loss_viz import loss_plots
#         combination_name = str(f"{dataset_name},{hardness_metric},{seed},{weighting_strategy}")
#         loss_plots(training_loss, combination_name,"training_loss_plots")
        

# if __name__ == "__main__":
#     main()



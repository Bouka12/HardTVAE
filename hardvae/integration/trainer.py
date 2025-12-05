"""
CVAE Trainer for HardVAE

This module provides the HardnessAwareCVAETrainer class for training the CVAE with hardness-aware weighting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

from hardvae.core.hardness import HardnessCalculator, CVAEHardnessIntegrator
from .cvae import TabularCVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HardnessAwareCVAETrainer:
    """
    Trainer for hardness-aware CVAE with integrated hardness scoring.
    """

    def __init__(self, model: TabularCVAE, hardness_calculator: HardnessCalculator, hardness_integrator: CVAEHardnessIntegrator, device: str = DEVICE):
        self.model = model.to(device)
        self.hardness_calculator = hardness_calculator
        self.hardness_integrator = hardness_integrator
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.hardness_scores = None

    def _calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, hardness_metric: str) -> Optional[np.ndarray]:
        if hardness_metric is not None:
            hardness_df = self.hardness_calculator.calculate_hardness_scores(X, y, [hardness_metric])
            if hardness_df.empty:
                return None
            return hardness_df[hardness_metric].values
        return None

    def _cvae_loss(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, weights: Optional[torch.Tensor] = None, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='none').mean(dim=1)
        if weights is not None:
            recon_loss = (recon_loss * weights).mean()
        else:
            recon_loss = recon_loss.mean()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, hardness_metric: Optional[str] = None, beta: float = 1.0):
        self.hardness_scores = self._calculate_hardness_scores(X_train, y_train, hardness_metric)
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x_batch, y_batch)
                weights = None
                if self.hardness_scores is not None:
                    indices = (dataset.tensors[0][:, None] == x_batch).all(-1).any(-1)
                    batch_hardness = self.hardness_scores[indices]
                    weights = self.hardness_integrator.calculate_weights(batch_hardness, epoch, epochs)
                loss, recon_loss, kld_loss = self._cvae_loss(x_recon, x_batch, mu, logvar, weights, beta)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")

    def generate(self, n_samples: int, condition: int) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            c = torch.full((n_samples, 1), float(condition)).to(self.device)
            synthetic_data = self.model.decode(z, c)
        return synthetic_data.cpu().numpy()

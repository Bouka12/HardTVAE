"""
Hardness-Aware Module for CVAE Training

This module provides comprehensive hardness calculation functionality for tabular medical data,
designed to integrate with Conditional Variational Autoencoders (CVAEs) for improved synthetic
data generation in imbalanced datasets.

Features:
- PyHard integration for standard hardness metrics
- Flexible class-based architecture
- CVAE integration utilities
- Comprehensive error handling and validation

Updates:
21/04/2026 : 
    - the pyhard.measures is in the hardvae_code (locally) and works fine in python 3.13
    - the calculate_all() is now calculate selected metrics without errors
    - the hardness metrics are now updated to eliminate the prefix "feature_"

22/04/2026 to-do:
    - The weights now are in penalization form (min_weight + h_i)
    * the hardness score could lead the NN to neglect easy instances completely-> think how to fix it!
    *** -> penalized (1 + alpha*h_i), or root squared (h_i^gamma) gamma=0.5
    * Check the weighting logic accordingly (curriculum, self-paced, static)

"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from imblearn.datasets import fetch_datasets
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# device related -> GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    from pyhard.measures import ClassificationMeasures
    PYHARD_AVAILABLE = True
except ImportError:
    PYHARD_AVAILABLE = False
    warnings.warn("PyHard package not available. Standard hardness metrics will be disabled.")
    
HardnessMetrics = ['kDN', 'DS', 'DCP', 'TD_P',
                   'TD_U', 'CL', 'CLD', 'MV', 
                   'CB', 'N1', 'N2', 'LSC', 
                   'LSR', 'Harmfulness', 
                   'F1', 'F2', 'F3', 'F4']

class HardnessCalculator:
    """
    Comprehensive hardness calculator for tabular data with support for multiple hardness metrics.
    """
    
    # Standard PyHard metrics
    PYHARD_METRICS = [
        'kDN', 'DS', 'DCP', 'TD_P',
        'TD_U', 'CL', 'CLD', 'MV', 
        'CB', 'N1', 'N2', 'LSC', 
        'LSR', 'Harmfulness',  
        'F1', 'F2', 'F3', 'F4'
    ]
    
    
    NO_WEIGHT_METRICS = [None]
    # Metric groups for analysis
    METRIC_GROUPS = {
        "linear": ['kDN', 'DS', 'DCP', 'TD_P', 'TD_U'],
        "neighborhood_based": ['CL', 'CLD', 'MV', 'CB'],
        "network_based": ['N1', 'N2'],
        "feature_based": ['LSC', 'LSR'],
        "other": ['Harmfulness', 'F1', 'F2', 'F3', 'F4']
    }
    
    def __init__(self, random_state: int = 42, n_classifiers: int = 5, min_hardness_value: float = 1e-6):
        """
        Initialize the HardnessCalculator.
        
        Args:
            random_state: Random seed for reproducibility
            n_classifiers: Number of classifiers for ensemble-based metrics
            min_hardness_value: Minimum value to replace zeros after scaling
        """
        self.random_state = random_state
        self.n_classifiers = n_classifiers
        self.min_hardness_value = min_hardness_value
        self.scaler = MinMaxScaler()
        # self.pca = None
        self.label_encoder = None
        
        # # Initialize probabilistic classifiers for relative entropy
        # self.classifiers = [
        #     # LogisticRegression(random_state=random_state),
        #     RandomForestClassifier(random_state=random_state),
        #     # GaussianNB(),
        #     SVC(probability=True, random_state=random_state),
        #     KNeighborsClassifier(n_neighbors=5)
        # ]
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input data."""
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X and y should not be empty.")
        
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError("y should be a 1D array of labels.")
        
        # Ensure y is 1D
        y = y.ravel()
        
        # Handle categorical labels
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
            else:
                y = self.label_encoder.transform(y)
        
        return X, y
    
    def _calculate_pyhard_metrics(self, X: np.ndarray, y: np.ndarray, 
                                  metrics: List[str]) -> pd.DataFrame:
        """Calculate PyHard-based hardness metrics."""
        # print("USING _calculate_pyhard_metrics FUNCTION")
        if not PYHARD_AVAILABLE:
            raise ImportError("PyHard package is required for standard hardness metrics.")
        
        # Create DataFrame for PyHard
        data = pd.DataFrame(X)
        data['target'] = y
        column_names = [f"feature_{i}" for i in range(X.shape[1])] + ['target']
        data.columns = column_names
        
        # Calculate hardness metrics
        hm = ClassificationMeasures(data, target_col='target')
        # HERE!!! HOW TO GET ONLY THE METRIC REQUIRED
        hardness_df = hm.calculate_all(measures_list=metrics)

        
        # hardness_df = pd.DataFrame(hardness_scores)
        if hardness_df is None or hardness_df.empty:
            raise ValueError("No valid PyHard metrics were calculated")
        print(f"Number of NaN values in hardness_df in _calculate_pyhard_metrics: {hardness_df.isnull().sum().sum()}")

        
        # Handle null values
        if hardness_df.isnull().values.any():
            null_metrics = hardness_df.columns[hardness_df.isnull().any()].tolist()
            warnings.warn(f"Dropping metrics with null values: {null_metrics}")
            hardness_df = hardness_df.dropna(axis=1, how='any')
        
        # New Check after dropping NaNs, if the DataFrame is empty--
        if hardness_df.empty:
            warnings.warn(
                f"All calculated hardness metrics {metrics} resulted in NaN values or an empty DataFrame after dropping columns with nulls. Returning empty DataFrame."
            )
            return pd.DataFrame() # Return an empty DataFrame
        
        
        return hardness_df
    
    
    def calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, 
                                  hardness_metrics: List[str]) -> pd.DataFrame:
        """
        Calculate hardness scores for each instance using specified metrics.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            hardness_metrics: List of hardness metrics to calculate
            
        Returns:
            DataFrame with hardness scores (scaled to [0,1])
        """
        # Validate inputs
        X, y = self._validate_inputs(X, y)
        
        if not hardness_metrics:
            raise ValueError("No hardness metrics specified.")
        
        #  PyHard  metrics
        pyhard_metrics = [m for m in hardness_metrics if m in self.PYHARD_METRICS]
        no_metrics = [m for m in hardness_metrics if m in self.NO_WEIGHT_METRICS]
        
        # Check for invalid metrics
        invalid_metrics = [m for m in hardness_metrics 
                          if m not in self.PYHARD_METRICS and  m not in self.NO_WEIGHT_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid hardness metrics: {invalid_metrics}")
        
        # Calculate metrics
        hardness_dfs = []
        
        if pyhard_metrics:
            pyhard_df = self._calculate_pyhard_metrics(X, y, pyhard_metrics)
            hardness_dfs.append(pyhard_df)
        
        if no_metrics:
            # Create dataframe with ones for no-weight metrics
            no_weight_df = pd.DataFrame(np.ones((X.shape[0], len(no_metrics))),
                                        columns=no_metrics)
            hardness_dfs.append(no_weight_df)
        
        # Combine all hardness scores
        if len(hardness_dfs) == 1:
            hardness_df = hardness_dfs[0]
        else:
            hardness_df = pd.concat(hardness_dfs, axis=1)
        
        if hardness_df.empty:
            warnings.warn("Combined hardness DataFrame is empty after processing. No valid scores to scale.")
            return pd.DataFrame() # Return an empty DataFrame


        # Scale to [0, 1] and avoid zeros
        scaled_hardness = pd.DataFrame(
            self.scaler.fit_transform(hardness_df), 
            columns=hardness_df.columns
        )
        scaled_hardness = scaled_hardness.replace(0, self.min_hardness_value)
        
        return scaled_hardness
    
    def get_metric_info(self, metric: str) -> Dict[str, str]:
        """Get information about a specific hardness metric."""
        metric_definitions = {
            'None': 'No hardness metric - used for uniform weighting',
            'kDN': 'k-Disagreeing Neighbors - measures local class disagreement',
            'DS': 'Decision Surface - complexity of decision boundary',
            'DCP': 'Disjunct Class Percentage - class distribution complexity',
            'TD_P': 'Tree Depth Pruned - decision tree complexity (pruned)',
            'TD_U': 'Tree Depth Unpruned - decision tree complexity (unpruned)',
            'CL': 'Class Likelihood - probability of correct classification',
            'CLD': 'Class Likelihood Difference - margin of classification',
            'MV': 'Minority Value - rarity in feature space',
            'CB': 'Class Balance - local class distribution',
            'N1': 'Borderline Points - fraction of different class neighbors',
            'N2': 'Intra-Extra Ratio - within vs between class distances',
            'LSC': 'Locality Sensitive Complexity - local complexity measure',
            'LSR': 'Locality Sensitive Radius - local neighborhood size',
            'Harmfulness': 'Harmfulness - negative impact on learning',
            'F1': 'Feature Overlap F1 - overlapping feature regions',
            'F2': 'Feature Overlap F2 - non-overlapping feature regions',
            'F3': 'Feature F3 - minority class feature characteristics',
            'F4': 'Feature F4 - majority class feature characteristics',
            # 'relative_entropy': 'Relative Entropy - classifier disagreement measure',
            # 'pca_contribution': 'PCA Contribution - instance importance in principal components',
            #'pca_reconstruction_error': 'PCA Reconstruction Error - information loss in dimensionality reduction'
        }
        
        group = None
        for group_name, metrics in self.METRIC_GROUPS.items():
            if metric in metrics:
                group = group_name
                break
        
        return {
            'name': metric,
            'definition': metric_definitions.get(metric, 'Definition not available'),
            'group': group or 'unknown'
        }
    
    def get_summary_statistics(self, hardness_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for calculated hardness scores."""
        return hardness_df.describe()

def project_to_min1(w_raw, min_w):
    """ map range from [min_weight, max_weight] to [min_weight, 1]"""
    w_clamped = np.clip(w_raw, min_w, min_w + 1.0)
    return min_w + (w_clamped - min_w) * (1.0 - min_w)

class CVAEHardnessIntegrator:
    """
    Utility class for integrating hardness scores with CVAE training.
    """
    
    def __init__(self, hardness_strategy: str = 'static', 
                 curriculum_epochs: Optional[Tuple[int, int, int]] = None, min_weight = 0.1, gamma = 1.0):
        """
        Initialize the CVAE hardness integrator.
        
        Args:
            hardness_strategy: 'static', 'curriculum', or 'self_paced'
            curriculum_epochs: Tuple of (easy_epochs, mixed_epochs, hard_epochs) for curriculum learning
            min_weight: minimal weight to define a functional floor so gradients never vanish. By default min_weight=0.5 (weights will range from 0.5 to 1.5)
            gamma: self-paced learning hyperparameter to control the rate of inclusion of harder samples (gamma > 1 slows down, gamma < 1 speeds up)
        """
        self.hardness_strategy = hardness_strategy
        self.curriculum_epochs = curriculum_epochs or (10, 10, 10)
        self.min_weight = min_weight # in the sensitivity analysis we experiment with {0.1, 0.2, 0.3, 0.4, 0.5}
        # self.annealing_weight = annealing_weight # in the sensitivity analysis annealing will be in {0.1, 0.2, 0.3, 0.4, 0.5}
        self.gamma = gamma # self-paced learning hyperparameter to control the rate of inclusion of harder samples (gamma > 1 slows down, gamma < 1 speeds up)
    def get_sample_weights(self, hardness_scores: np.ndarray, 
                          epoch: int = 0, total_epochs: int = 100) -> np.ndarray:
        """
        Get sample weights based on hardness strategy.
        
        Args:
            hardness_scores: Array of hardness scores for each sample
            epoch: Current training epoch
            total_epochs: Total number of training epochs
            
        Returns:
            Array of weights for each sample
        """
        if self.hardness_strategy == 'static':
            # Static weighting: higher hardness = higher weight
            
            ## weight_i = min_weight + alpha * h_i # h_i is the hardness score of instance (i)
            ## for now alpha is 0 and min_weight = 0.5 or 1 as will be seen in the sensitivity analysis
            # Static weighting L Sift the [0, 1] range to [min_weight, min_weight + 1.0]
            # SOLUTION:  MAP FROM [min_weight, 1+min_weight] (penalization) -> [min_weight, 1]
            ## w_final = min_weight + (weight_raw -min_weight)/(max_weight-min_weight)  * (1-min_weight)
            # w_raw = self.min_weight + hardness_scores
            return self.min_weight + hardness_scores
        
        elif self.hardness_strategy == 'curriculum':
            # Curriculum learning: easy -> mixed -> hard self.curriculum_epochs is a tuple of (easy_epochs, mixed_epochs, hard_epochs) defining the epoch thresholds for switching strategies
            easy_epochs, mixed_epochs, hard_epochs = (int(total_epochs*self.curriculum_epochs[0]), int(total_epochs*self.curriculum_epochs[1]), int(total_epochs*self.curriculum_epochs[2])) if self.curriculum_epochs else (int(total_epochs * 0.3), int(total_epochs * 0.3), int(total_epochs * 0.4))
            
            if epoch < easy_epochs:
                # Focus on easy samples (low hardness)
                # Hard samples are suppressed but NOT zeroed out
                
                return self.min_weight + (1.0 - hardness_scores)
            elif epoch < easy_epochs + mixed_epochs:
                # Uniform weighting
                # Smooth Loss transition: the average weight of a batch is roughly 1.0
                # keeping the uniform phase at 1.0 prevents sudden spikes or drops in the toal loss value when the curriculum shifts phases
                return np.ones_like(hardness_scores) * (self.min_weight + 0.5)
            else:
                # Focus on hard samples (high hardness)
                # Hard get max weight, easy get min_weight
                return self.min_weight + hardness_scores
        
        elif self.hardness_strategy == 'self_paced':
            # Self-paced: gradually include harder samples
            progress = epoch / total_epochs
            threshold = np.percentile(hardness_scores, (progress ** self.gamma) * 100)
            weights = np.where(hardness_scores <= threshold, 1.0 + self.min_weight, self.min_weight) # Annealing weight is fixed to min_weight to match other strategies
         
            return weights
        
        else:
            raise ValueError(f"Unknown hardness strategy: {self.hardness_strategy}")
    
    def weighted_reconstruction_loss(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                                   weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted reconstruction loss for CVAE.
        
        Args:
            x_true: True input data
            x_pred: Reconstructed data
            weights: Sample weights based on hardness
            
        Returns:
            Weighted reconstruction loss
        """
        # Calculate per-sample reconstruction loss (MSE)
        recon_loss = torch.sum((x_true - x_pred) ** 2, dim=1)
        
        # Apply weights
        weighted_loss = recon_loss * weights
        
        return torch.mean(weighted_loss)


# # Example usage and testing
# if __name__ == "__main__":
#     # Create sample data
#     np.random.seed(42)
#     data = fetch_datasets()['ecoli']
#     X, y  = data.data, data.target
#     # y values are in -1 and 1, convert to 0 and 1 for binary classification
#     y = np.where(y == -1, 0, 1)  # Convert to binary labels
#     # check the count of the classes
#     print(f"Class distribution: {np.bincount(y)}")
#     # X = np.random.randn(200, 10)
#     # y = np.random.choice([0, 1], size=200, p=[0.8, 0.2])  # Imbalanced
    
#     # Initialize hardness calculator
#     calc = HardnessCalculator()
    
#     # Calculate hardness metrics
#     metrics = ['TD_P']  
    
#     try:
#         hardness_df = calc.calculate_hardness_scores(X, y, metrics)
#         print("Hardness scores calculated successfully!")
#         print(f"Shape: {hardness_df.shape}")
#         print("\nSummary statistics:")
#         print(calc.get_summary_statistics(hardness_df))
        
#         # Test CVAE integration
#         integrator = CVAEHardnessIntegrator(hardness_strategy='curriculum')
        
#         # Example of getting weights for different epochs
#         hardness_values = hardness_df['TD_P'].values
        
#         for epoch in [5, 15, 25]:
#             weights = integrator.get_sample_weights(hardness_values, epoch, 30)
#             print(f"\nEpoch {epoch} - Weight statistics:")
#             print(f"Mean: {np.mean(weights):.3f}, Std: {np.std(weights):.3f}")
#             print(f"Min: {np.min(weights):.3f}, Max: {np.max(weights):.3f}")
        
#     except Exception as e:
#         print(f"Error: {e}")


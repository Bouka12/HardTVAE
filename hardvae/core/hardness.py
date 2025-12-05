"""
Hardness Calculator for HardVAE

This module provides the HardnessCalculator class for comprehensive hardness calculation on tabular data.
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from scipy.stats import entropy
import torch
import torch.nn as nn

from .metrics import PYHARD_METRICS, CUSTOM_METRICS, METRIC_GROUPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from pyhard.measures import ClassificationMeasures
    PYHARD_AVAILABLE = True
except ImportError:
    PYHARD_AVAILABLE = False
    warnings.warn("PyHard package not available. Standard hardness metrics will be disabled.")

class HardnessCalculator:
    """
    Comprehensive hardness calculator for tabular data with support for multiple hardness metrics.
    """

    def __init__(self, random_state: int = 42, n_classifiers: int = 5, min_hardness_value: float = 1e-6):
        self.random_state = random_state
        self.n_classifiers = n_classifiers
        self.min_hardness_value = min_hardness_value
        self.scaler = MinMaxScaler()
        self.pca = None
        self.label_encoder = None
        self.classifiers = [
            RandomForestClassifier(random_state=random_state),
            SVC(probability=True, random_state=random_state),
            KNeighborsClassifier(n_neighbors=5)
        ]

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X and y should not be empty.")
        y = y.ravel()
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
            else:
                y = self.label_encoder.transform(y)
        return X, y

    def _calculate_pyhard_metrics(self, X: np.ndarray, y: np.ndarray, metrics: List[str]) -> pd.DataFrame:
        if not PYHARD_AVAILABLE:
            raise ImportError("PyHard package is required for standard hardness metrics.")
        data = pd.DataFrame(X)
        data['target'] = y
        column_names = [f"feature_{i}" for i in range(X.shape[1])] + ['target']
        data.columns = column_names
        hm = ClassificationMeasures(data)
        data_hm = hm.calculate_all()
        hardness_scores = {metric: data_hm[metric] for metric in metrics if metric in data_hm}
        if not hardness_scores:
            raise ValueError("No valid PyHard metrics were calculated.")
        hardness_df = pd.DataFrame(hardness_scores)
        if hardness_df.isnull().values.any():
            hardness_df = hardness_df.dropna(axis=1, how='any')
        if hardness_df.empty:
            warnings.warn("All calculated hardness metrics resulted in NaN values. Returning empty DataFrame.")
            return pd.DataFrame()
        return hardness_df

    def _calculate_relative_entropy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_classes = X.shape[0], len(np.unique(y))
        selected_classifiers = self.classifiers[:self.n_classifiers]
        all_probabilities = []
        for clf in selected_classifiers:
            try:
                probas = cross_val_predict(clf, X, y, cv=3, method='predict_proba')
                all_probabilities.append(probas)
            except Exception as e:
                warnings.warn(f"Classifier {type(clf).__name__} failed: {e}")
        if not all_probabilities:
            raise ValueError("No classifiers could generate probability predictions.")
        relative_entropies = []
        for i in range(n_samples):
            instance_probas = [proba[i] for proba in all_probabilities]
            avg_proba = np.mean(instance_probas, axis=0)
            uniform_dist = np.ones(n_classes) / n_classes
            avg_proba = np.clip(avg_proba, 1e-10, 1.0)
            uniform_dist = np.clip(uniform_dist, 1e-10, 1.0)
            rel_entropy = entropy(avg_proba, uniform_dist)
            relative_entropies.append(rel_entropy)
        return np.array(relative_entropies)

    def _calculate_pca_metrics(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            n_components = min(X.shape[1], 50)
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.pca.fit(X)
        X_transformed = self.pca.transform(X)
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        selected_components = np.where(cumulative_variance < 0.80)[0]
        contributions = X_transformed[:, selected_components] / self.pca.singular_values_[selected_components]
        total_contributions = np.sum(np.abs(contributions), axis=1)
        return total_contributions

    def calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, metrics: List[str]) -> pd.DataFrame:
        X, y = self._validate_inputs(X, y)
        all_hardness_scores = pd.DataFrame(index=range(len(X)))
        pyhard_metrics_to_calc = [m for m in metrics if m in PYHARD_METRICS]
        if pyhard_metrics_to_calc:
            pyhard_scores = self._calculate_pyhard_metrics(X, y, pyhard_metrics_to_calc)
            all_hardness_scores = pd.concat([all_hardness_scores, pyhard_scores], axis=1)
        if 'relative_entropy' in metrics:
            all_hardness_scores['relative_entropy'] = self._calculate_relative_entropy(X, y)
        if 'pca_contribution' in metrics:
            all_hardness_scores['pca_contribution'] = self._calculate_pca_metrics(X)
        all_hardness_scores = self.scaler.fit_transform(all_hardness_scores)
        all_hardness_scores[all_hardness_scores == 0] = self.min_hardness_value
        return pd.DataFrame(all_hardness_scores, columns=metrics)

class CVAEHardnessIntegrator:
    """
    Integrates hardness scores with CVAE training.
    """

    def __init__(self, weighting_strategy: str = 'static', min_weight: float = 0.1, max_weight: float = 10.0):
        self.weighting_strategy = weighting_strategy
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_weights(self, hardness_scores: np.ndarray, epoch: int, total_epochs: int) -> torch.Tensor:
        if self.weighting_strategy == 'static':
            weights = torch.tensor(hardness_scores, dtype=torch.float32)
        elif self.weighting_strategy == 'curriculum':
            progress = epoch / total_epochs
            weights = torch.tensor(hardness_scores ** (1 + progress), dtype=torch.float32)
        elif self.weighting_strategy == 'self_paced':
            # Placeholder for a more sophisticated self-paced strategy
            progress = epoch / total_epochs
            threshold = np.percentile(hardness_scores, progress * 100)
            weights = torch.tensor(np.where(hardness_scores >= threshold, hardness_scores, 0.1), dtype=torch.float32)
        else:
            weights = torch.ones(len(hardness_scores), dtype=torch.float32)
        
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        return weights.to(device)

"""
Synthetic Data Evaluator for HardVAE

This module provides the SyntheticDataEvaluator class for comprehensive quality assessment of synthetic data.
"""

import numpy as np
from typing import Dict

from .metrics import statistical_evaluation, complexity_evaluation, hardness_evaluation, topological_evaluation, utility_evaluation
from .visualizer import EvaluationVisualizer

class SyntheticDataEvaluator:
    """
    Comprehensive evaluator for synthetic minority data quality assessment.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.visualizer = EvaluationVisualizer()

    def evaluate_all(self, X_real: np.ndarray, y_real: np.ndarray, X_synth: np.ndarray, y_synth: np.ndarray, save_path: str, dataset_name: str) -> Dict:
        print(f"Starting comprehensive evaluation for {dataset_name}...")

        # Statistical Evaluation
        stat_results = statistical_evaluation(X_real, y_real, X_synth, y_synth, self.random_state)
        self.results["statistical"] = stat_results

        # Complexity Analysis
        complexity_results = complexity_evaluation(X_real, y_real, X_synth, y_synth, self.random_state)
        self.results["complexity"] = complexity_results

        # Instance Hardness Analysis
        hardness_results = hardness_evaluation(X_real, y_real, X_synth, y_synth, self.random_state)
        self.results["hardness"] = hardness_results

        # Topological Data Analysis
        tda_results = topological_evaluation(X_real, y_real, X_synth, y_synth)
        self.results["topological"] = tda_results

        # Utility Evaluation
        utility_results = utility_evaluation(X_real, y_real, X_synth, y_synth, self.random_state)
        self.results["utility"] = utility_results

        # Generate visualizations
        self.visualizer.create_comprehensive_dashboard(self.results, save_path, dataset_name)

        return self.results

"""
Evaluation Visualizer for HardVAE

This module provides the EvaluationVisualizer class for creating visualizations of evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class EvaluationVisualizer:
    """
    Creates visualizations for synthetic data evaluation results.
    """

    def create_comprehensive_dashboard(self, results: Dict, save_path: str, dataset_name: str):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Synthetic Data Quality Dashboard for {dataset_name}", fontsize=16)

        # Statistical Similarity
        if "statistical" in results:
            self._plot_radar(axes[0, 0], "Statistical Similarity", list(results["statistical"].keys()), list(results["statistical"].values()))

        # Complexity Similarity
        if "complexity" in results:
            self._plot_radar(axes[0, 1], "Complexity Similarity", list(results["complexity"].keys()), list(results["complexity"].values()))

        # Hardness Similarity
        if "hardness" in results:
            self._plot_radar(axes[0, 2], "Hardness Similarity", list(results["hardness"].keys()), list(results["hardness"].values()))

        # Topological Similarity
        if "topological" in results:
            self._plot_bar(axes[1, 0], "Topological Similarity", list(results["topological"].keys()), list(results["topological"].values()))

        # Utility Score
        if "utility" in results:
            self._plot_bar(axes[1, 1], "Utility Score", list(results["utility"].keys()), list(results["utility"].values()))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_path}/{dataset_name}_dashboard.png")
        plt.close()

    def _plot_radar(self, ax, title, labels, values):
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        ax.polar(angles, values, marker=".")
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(title)

    def _plot_bar(self, ax, title, labels, values):
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylim(0, 1)

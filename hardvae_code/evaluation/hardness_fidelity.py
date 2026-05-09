"""
hardness_fidelity.py
====================
Standalone hardness fidelity view — measures how well the synthetic minority
samples preserve the instance-level hardness distribution of the real data.

Interface mirrors distributional_fidelity and complexity_fidelity:
    keys, values = hardness_fidelity(X_real, y_real, X_synth, y_synth, ...)

Uses the `pyhard` package (ClassificationMeasures) for hardness metrics
(kDN, DS, DCP, TD_P, TD_U, CL, CLD, MV, CB, N1, N2, LSC, LSR,
 Harmfulness, F1, F2, F3, F4).

KS test (scipy.stats.ks_2samp) is used to compare hardness distributions
between real and synthetic data — giving a per-metric similarity score of
(1 - KS_statistic), averaged to an overall hardness fidelity score.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — avoids Tkinter errors
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle as sk_shuffle


# =============================================================================
# Valid pyhard metric names (used for validation in _calculate_hardness_scores)
# =============================================================================

PYHARD_METRICS = [
    'kDN', 'DS', 'DCP', 'TD_P', 'TD_U', 'CL', 'CLD',
    'MV', 'CB', 'N1', 'N2', 'LSC', 'LSR', 'Harmfulness',
    'F1', 'F2', 'F3', 'F4',
]

DEFAULT_METRICS = PYHARD_METRICS   # use all by default


# =============================================================================
# Internal helpers
# =============================================================================

def _to_array(data) -> np.ndarray:
    """Convert DataFrame / Series / list to float64 ndarray."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values.astype(np.float64)
    return np.array(data, dtype=np.float64)


def _cluster_sampling(X, y, k=4, sampling_ratio=None, random_state=42):
    """
    Cluster-based sampling for large datasets.
    sampling_ratio is auto-derived from dataset size if not provided:
      >= 100,000 samples → 1 %
      >= 10,000  samples → 10 %
      <  10,000  samples → returned unchanged
    """
    if sampling_ratio is None:
        if X.shape[0] >= 100_000:
            sampling_ratio = 0.01
        elif X.shape[0] >= 10_000:
            sampling_ratio = 0.1
        else:
            return X, y

    X_shuf, y_shuf = sk_shuffle(X, y, random_state=random_state)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state).fit(X_shuf)
    cluster_labels = kmeans.predict(X_shuf)

    X_sampled, y_sampled = [], []
    cluster_counts = Counter(cluster_labels)
    for i in range(k):
        sample_size = int(sampling_ratio * cluster_counts[i])
        if sample_size > 0:
            mask = cluster_labels == i
            X_sampled.append(X_shuf[mask][:sample_size])
            y_sampled.append(y_shuf[mask][:sample_size])

    return np.concatenate(X_sampled), np.concatenate(y_sampled)


def _calculate_hardness_scores(X: np.ndarray, y: np.ndarray,
                                hardness_metrics: list) -> pd.DataFrame:
    """
    Calculate per-instance hardness scores using pyhard.ClassificationMeasures.

    Returns a DataFrame of shape (n_samples, n_valid_metrics) with MinMax-scaled
    scores in [0, 1].  Returns an empty DataFrame on failure.
    """
    try:
        from pyhard.measures import ClassificationMeasures
    except ImportError as e:
        print(f"[hardness_fidelity] ImportError: {e}")
        return pd.DataFrame()

    try:
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")

        # Build DataFrame expected by pyhard
        col_names = [f"feature_{i}" for i in range(X.shape[1])] + ["target"]
        data = pd.DataFrame(
            np.column_stack([X, y.reshape(-1, 1)]),
            columns=col_names,
        )

        # Validate and filter metrics
        invalid = [m for m in hardness_metrics if m not in PYHARD_METRICS]
        if invalid:
            print(f"[hardness_fidelity] Skipping invalid metrics: {invalid}")
        valid_metrics = [m for m in hardness_metrics if m in PYHARD_METRICS]

        if not valid_metrics:
            raise ValueError("No valid pyhard metrics after filtering.")

        hm = ClassificationMeasures(data)
        data_hm = hm.calculate_all()

        scores = {m: data_hm[m] for m in valid_metrics if m in data_hm}
        hardness_df = pd.DataFrame(scores)

        if hardness_df.empty:
            raise ValueError("pyhard returned no metrics.")

        # Drop columns with any NaN
        null_cols = hardness_df.columns[hardness_df.isnull().any()].tolist()
        if null_cols:
            print(f"[hardness_fidelity] Dropping NaN metric columns: {null_cols}")
            hardness_df = hardness_df.dropna(axis=1)

        if hardness_df.empty:
            raise ValueError("All metric columns contained NaN values.")

        # MinMax scale to [0, 1]; replace exact 0s with small epsilon
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(hardness_df),
            columns=hardness_df.columns,
        )
        scaled = scaled.replace(0, 1e-6)
        return scaled

    except Exception as e:
        print(f"[hardness_fidelity] _calculate_hardness_scores failed: {e}")
        return pd.DataFrame()


def _save_hardness_plots(metric_statistics: dict, iteration_results: dict,
                         mean_ks_statistic: float, overall_similarity: float,
                         save_path: str, dataset_name: str, random_state: int) -> None:
    """Save 6-panel hardness evaluation figure to save_path."""
    try:
        os.makedirs(save_path, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = list(metric_statistics.keys())
        x = np.arange(len(metrics))
        width = 0.35

        # ── Plot 1: KS statistics ───────────────────────────────────────────
        ks_stats = [metric_statistics[m]['ks_statistic'] for m in metrics]
        axes[0, 0].bar(x, ks_stats, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('KS Statistics by Hardness Metric')
        axes[0, 0].set_xlabel('Metric'); axes[0, 0].set_ylabel('KS Statistic')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)

        # ── Plot 2: Similarity scores ───────────────────────────────────────
        sim_scores = [metric_statistics[m]['similarity_score'] for m in metrics]
        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in sim_scores]
        axes[0, 1].bar(x, sim_scores, alpha=0.7, color=colors)
        axes[0, 1].set_title('Similarity Scores by Hardness Metric')
        axes[0, 1].set_xlabel('Metric'); axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        # ── Plot 3: Mean comparison real vs synth ───────────────────────────
        real_means  = [metric_statistics[m]['real_mean']  for m in metrics]
        synth_means = [metric_statistics[m]['synth_mean'] for m in metrics]
        axes[0, 2].bar(x - width/2, real_means,  width, label='Real',      alpha=0.7, color='blue')
        axes[0, 2].bar(x + width/2, synth_means, width, label='Synthetic', alpha=0.7, color='orange')
        axes[0, 2].set_title('Mean Hardness Comparison')
        axes[0, 2].set_xlabel('Metric'); axes[0, 2].set_ylabel('Mean Score')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

        # ── Plot 4: p-values ────────────────────────────────────────────────
        p_values = [metric_statistics[m]['ks_pvalue'] for m in metrics]
        axes[1, 0].bar(x, p_values, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('KS Test P-values')
        axes[1, 0].set_xlabel('Metric'); axes[1, 0].set_ylabel('p-value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # ── Plot 5: Std comparison ──────────────────────────────────────────
        real_stds  = [metric_statistics[m]['real_std']  for m in metrics]
        synth_stds = [metric_statistics[m]['synth_std'] for m in metrics]
        axes[1, 1].bar(x - width/2, real_stds,  width, label='Real',      alpha=0.7, color='lightblue')
        axes[1, 1].bar(x + width/2, synth_stds, width, label='Synthetic', alpha=0.7, color='lightsalmon')
        axes[1, 1].set_title('Hardness Variability Comparison')
        axes[1, 1].set_xlabel('Metric'); axes[1, 1].set_ylabel('Std Dev')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        # ── Plot 6: Summary ─────────────────────────────────────────────────
        summary = {
            'Successful\nIterations': iteration_results['successful_iterations'],
            'Total\nMetrics':         len(metric_statistics),
            'Avg KS\nStatistic':      mean_ks_statistic,
            'Overall\nSimilarity':    overall_similarity,
        }
        bar_colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightpink']
        bars = axes[1, 2].bar(summary.keys(), summary.values(), alpha=0.7, color=bar_colors)
        axes[1, 2].set_title('Evaluation Summary')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(True, alpha=0.3)
        for bar, val in zip(bars, summary.values()):
            h = bar.get_height()
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2., h + 0.01,
                f'{val:.2f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=10,
            )

        plt.suptitle(f'Hardness Fidelity Results — {dataset_name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(save_path, f'hardness_fidelity_{random_state}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[hardness_fidelity] Plot saved to: {plot_path}")

    except Exception as e:
        print(f"[hardness_fidelity] Could not save plots: {e}")


# =============================================================================
# Public API
# =============================================================================

def hardness_fidelity_calculate(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    k: int = 3,
    minority_class=None,
    hardness_metrics: list = None,
    cluster_threshold: int = 10_000,
    save_path: str = None,
    dataset_name: str = "dataset",
    return_detailed: bool = False,
    random_state: int = 42,
) -> tuple:
    """
    Hardness fidelity view: compares the per-instance hardness distributions
    of synthetic minority data against real data using pyhard metrics and
    KS tests.

    Parameters
    ----------
    X_real : array-like, shape (n_real_total, n_features)
        Complete original training features — both minority AND majority.
        Must be fully numerical (pass preprocessed data).

    y_real : array-like, shape (n_real_total,)
        Labels for X_real.

    X_synth : array-like, shape (n_synth, n_features)
        Synthetic minority class features only.

    y_synth : array-like, shape (n_synth,)
        Labels for X_synth (all minority class).

    k : int, default=3
        Number of sampling iterations; scores averaged across iterations.

    minority_class : int/str, optional
        Minority class label. Auto-detected as least frequent if None.

    hardness_metrics : list[str], optional
        Subset of PYHARD_METRICS to compute. Defaults to all 18 metrics.

    cluster_threshold : int, default=10,000
        Apply cluster sub-sampling when a combined dataset exceeds this size.

    save_path : str, optional
        Directory to save the 6-panel diagnostic plot. No plot if None.

    dataset_name : str, default='dataset'
        Used in plot titles and filenames.

    return_detailed : bool, default=False
        If True, returns a third element with per-metric statistics and
        iteration details.

    random_state : int, default=42

    Returns
    -------
    keys   : list[str]
        'hardness_overall_similarity'  — mean (1 - KS) across all metrics
        'hardness_mean_ks_statistic'   — mean KS statistic (lower = more similar)
        'hardness_std_ks_statistic'    — std of KS statistics across metrics
        'hardness_mean_ks_pvalue'      — mean p-value across metrics
        'hardness_successful_iters'    — how many of k iterations succeeded
        'hardness_metrics_analyzed'    — number of metrics successfully computed

    values : list[float]

    detailed : dict  (only when return_detailed=True)
        Keys: 'metric_statistics', 'hardness_metrics_used',
              'individual_ks_statistics', 'individual_ks_pvalues',
              'sampling_strategy', 'class_imbalance_ratio',
              'iteration_details'
    """
    np.random.seed(random_state)

    if hardness_metrics is None:
        hardness_metrics = DEFAULT_METRICS

    # ── Type conversion ───────────────────────────────────────────────────────
    X_real  = _to_array(X_real)
    y_real  = np.array(y_real).ravel()
    X_synth = _to_array(X_synth)
    y_synth = np.array(y_synth).ravel()

    # ── Auto-detect minority class ────────────────────────────────────────────
    unique_classes, class_counts = np.unique(y_real, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected binary classification, got {len(unique_classes)} "
            f"classes: {unique_classes}"
        )
    if minority_class is None:
        minority_class = unique_classes[np.argmin(class_counts)]
        print(f"[hardness_fidelity] Auto-detected minority class: {minority_class}")

    # ── Split real data ───────────────────────────────────────────────────────
    min_mask = y_real == minority_class
    X_real_min, y_real_min = X_real[min_mask],  y_real[min_mask]
    X_real_maj, y_real_maj = X_real[~min_mask], y_real[~min_mask]

    n_real_min = len(X_real_min)
    n_synth    = len(X_synth)
    n_real_maj = len(X_real_maj)
    class_imbalance_ratio = n_real_min / n_real_maj

    # ── Sampling strategy ─────────────────────────────────────────────────────
    if n_synth >= n_real_min:
        sampling_strategy = "synth_larger"
        sample_size_min   = n_real_min
        print(f"[hardness_fidelity] Strategy: synth_larger "
              f"(sampling {sample_size_min} from synthetic)")
    else:
        sampling_strategy = "real_larger"
        sample_size_min   = n_synth
        print(f"[hardness_fidelity] Strategy: real_larger "
              f"(sampling {sample_size_min} from real minority)")

    # ── k iterations ─────────────────────────────────────────────────────────
    iter_real_hardness  = []
    iter_synth_hardness = []
    iteration_details   = []
    successful          = 0

    for it in range(k):
        print(f"  Iteration {it + 1}/{k}")
        try:
            # Sample minority
            if sampling_strategy == "synth_larger":
                s_idx = np.random.choice(n_synth,    size=sample_size_min, replace=False)
                X_synth_s, y_synth_s = X_synth[s_idx], y_synth[s_idx]
                X_real_min_it = X_real_min.copy()
                y_real_min_it = y_real_min.copy()
            else:
                r_idx = np.random.choice(n_real_min, size=sample_size_min, replace=False)
                X_real_min_it, y_real_min_it = X_real_min[r_idx], y_real_min[r_idx]
                X_synth_s, y_synth_s         = X_synth.copy(), y_synth.copy()

            # Sample majority to maintain imbalance ratio
            maj_size = min(int(sample_size_min / class_imbalance_ratio), n_real_maj)
            m_idx = np.random.choice(n_real_maj, size=maj_size, replace=False)
            X_maj_s, y_maj_s = X_real_maj[m_idx], y_real_maj[m_idx]

            # Complete datasets
            X_real_it  = np.vstack([X_real_min_it, X_maj_s])
            y_real_it  = np.hstack([y_real_min_it, y_maj_s])
            X_synth_it = np.vstack([X_synth_s, X_maj_s])
            y_synth_it = np.hstack([y_synth_s, y_maj_s])

            # Cluster sub-sampling for large datasets
            clustering_applied = False
            if len(X_real_it) > cluster_threshold:
                print(f"    Applying cluster sub-sampling ({len(X_real_it)} samples)")
                X_real_it,  y_real_it  = _cluster_sampling(X_real_it,  y_real_it,
                                                            random_state=random_state)
                X_synth_it, y_synth_it = _cluster_sampling(X_synth_it, y_synth_it,
                                                            random_state=random_state)
                clustering_applied = True

            # Compute hardness scores
            print(f"    Calculating hardness scores...")
            real_hdf  = _calculate_hardness_scores(X_real_it,  y_real_it,  hardness_metrics)
            synth_hdf = _calculate_hardness_scores(X_synth_it, y_synth_it, hardness_metrics)

            if real_hdf.empty or synth_hdf.empty:
                raise ValueError("Hardness calculation returned empty DataFrame.")

            iter_real_hardness.append(real_hdf)
            iter_synth_hardness.append(synth_hdf)
            successful += 1

            iteration_details.append({
                'iteration':            it + 1,
                'real_dataset_size':    len(X_real_it),
                'synth_dataset_size':   len(X_synth_it),
                'minority_sample_size': sample_size_min,
                'majority_sample_size': maj_size,
                'clustering_applied':   clustering_applied,
                'real_metrics':         list(real_hdf.columns),
                'synth_metrics':        list(synth_hdf.columns),
                'real_mean_hardness':   real_hdf.mean().to_dict(),
                'synth_mean_hardness':  synth_hdf.mean().to_dict(),
                'failed':               False,
            })
            print(f"    Computed {len(real_hdf.columns)} hardness metrics")

        except Exception as e:
            print(f"    Warning: iteration {it + 1} failed — {e}")
            iteration_details.append({'iteration': it + 1, 'failed': True, 'error': str(e)})

    # ── Statistical analysis across iterations ────────────────────────────────
    metric_statistics  = {}
    ks_statistics      = []
    ks_pvalues         = []
    overall_similarity = 0.0
    mean_ks_stat       = 1.0
    std_ks_stat        = 0.0
    mean_ks_pval       = 0.0

    if successful > 0:
        print(f"[hardness_fidelity] Statistical analysis across {successful} iterations...")

        # Common metrics across all successful iterations
        common = set(iter_real_hardness[0].columns)
        for df in iter_real_hardness[1:] + iter_synth_hardness:
            common &= set(df.columns)
        common = sorted(common)
        print(f"  Common metrics: {common}")

        for metric in common:
            real_vals  = np.concatenate([df[metric].values for df in iter_real_hardness])
            synth_vals = np.concatenate([df[metric].values for df in iter_synth_hardness])

            real_clean  = real_vals[~np.isnan(real_vals)]
            synth_clean = synth_vals[~np.isnan(synth_vals)]

            if len(real_clean) > 0 and len(synth_clean) > 0:
                ks_stat, ks_pval = ks_2samp(real_clean, synth_clean)
                metric_statistics[metric] = {
                    'real_mean':       float(np.mean(real_clean)),
                    'real_std':        float(np.std(real_clean)),
                    'synth_mean':      float(np.mean(synth_clean)),
                    'synth_std':       float(np.std(synth_clean)),
                    'ks_statistic':    float(ks_stat),
                    'ks_pvalue':       float(ks_pval),
                    'similarity_score': float(1.0 - ks_stat),
                    'n_real_samples':  int(len(real_clean)),
                    'n_synth_samples': int(len(synth_clean)),
                }
                ks_statistics.append(ks_stat)
                ks_pvalues.append(ks_pval)
                print(f"  {metric}: real={np.mean(real_clean):.3f}±{np.std(real_clean):.3f}  "
                      f"synth={np.mean(synth_clean):.3f}±{np.std(synth_clean):.3f}  "
                      f"KS={ks_stat:.3f} (p={ks_pval:.3f})")
            else:
                print(f"  Warning: insufficient data for {metric}")
                metric_statistics[metric] = {
                    'real_mean': 0.0, 'real_std': 0.0,
                    'synth_mean': 0.0, 'synth_std': 0.0,
                    'ks_statistic': 1.0, 'ks_pvalue': 0.0,
                    'similarity_score': 0.0,
                    'n_real_samples': 0, 'n_synth_samples': 0,
                }

        if ks_statistics:
            overall_similarity = float(np.mean(
                [metric_statistics[m]['similarity_score'] for m in metric_statistics]
            ))
            mean_ks_stat = float(np.mean(ks_statistics))
            std_ks_stat  = float(np.std(ks_statistics))
            mean_ks_pval = float(np.mean(ks_pvalues))

    # ── Optional plot ─────────────────────────────────────────────────────────
    if save_path and successful > 0 and metric_statistics:
        _save_hardness_plots(
            metric_statistics,
            {'successful_iterations': successful},
            mean_ks_stat, overall_similarity,
            save_path, dataset_name, random_state=random_state
        )

    print(f"[hardness_fidelity] Done — "
          f"similarity={overall_similarity:.3f}  "
          f"KS={mean_ks_stat:.3f}±{std_ks_stat:.3f}  "
          f"({successful}/{k} iterations OK)")

    # ── Flat output ───────────────────────────────────────────────────────────
    keys = [
        'hardness_overall_similarity',
        'hardness_mean_ks_statistic',
        'hardness_std_ks_statistic',
        'hardness_mean_ks_pvalue',
        'hardness_successful_iters',
        'hardness_metrics_analyzed',
    ]
    values = [
        overall_similarity,
        mean_ks_stat,
        std_ks_stat,
        mean_ks_pval,
        float(successful),
        float(len(metric_statistics)),
    ]

    if not return_detailed:
        return keys, values

    # ── Detailed output ───────────────────────────────────────────────────────
    detailed = {
        'metric_statistics':        metric_statistics,
        'hardness_metrics_used':    list(metric_statistics.keys()),
        'individual_ks_statistics': ks_statistics,
        'individual_ks_pvalues':    ks_pvalues,
        'sampling_strategy':        sampling_strategy,
        'class_imbalance_ratio':    class_imbalance_ratio,
        'iteration_details':        iteration_details,
    }
    return keys, values, detailed
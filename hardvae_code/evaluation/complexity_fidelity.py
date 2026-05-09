"""
complexity_fidelity.py
======================
Standalone complexity fidelity view — measures how well the synthetic minority
samples preserve the geometric / structural complexity of the real data.

Interface mirrors distributional_fidelity:
    keys, values = complexity_fidelity(X_real, y_real, X_synth, y_synth, ...)

Uses the `problexity` package for complexity metrics (F1, F2, N1, N2, …).
"""

import numpy as np
import pandas as pd
import problexity as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle as sk_shuffle
from collections import Counter


# =============================================================================
# Internal helpers (previously class methods)
# =============================================================================

def _cluster_sampling(X, y, k=4, sampling_ratio=None, random_state=42) -> tuple:
    """
    Cluster-based sampling for large datasets.
    sampling_ratio is auto-derived from dataset size if not provided:
      >= 100,000 samples → 1%
      >= 10,000  samples → 10%
      <  10,000  samples → no sampling (returned as-is)
    """
    from sklearn.utils import shuffle as sk_shuffle
    from collections import Counter

    if sampling_ratio is None:
        if X.shape[0] >= 100_000:
            sampling_ratio = 0.01
        elif X.shape[0] >= 10_000:
            sampling_ratio = 0.1
        else:
            return X, y   # dataset small enough — skip sampling

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


def _to_array(data) -> np.ndarray:
    """Convert DataFrame or any array-like to float64 ndarray."""
    if isinstance(data, pd.DataFrame):
        return data.values.astype(np.float64)
    return np.array(data, dtype=np.float64)


# =============================================================================
# Public API
# =============================================================================

def complexity_fidelity_calculate(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    k: int = 3,
    minority_class=None,
    cluster_threshold: int = 10_000,
    return_detailed: bool = False,
    random_state: int = 42,
) -> tuple:
    """
    Complexity fidelity view: compares the geometric/structural complexity of
    synthetic minority data against real data using the `problexity` package.

    Parameters
    ----------
    X_real : array-like, shape (n_real_total, n_features)
        Complete original training features — both minority AND majority classes.
        Must be fully numerical (pass preprocessed data, same as distributional_fidelity).

    y_real : array-like, shape (n_real_total,)
        Labels for X_real.

    X_synth : array-like, shape (n_synth, n_features)
        Synthetic minority class features only.

    y_synth : array-like, shape (n_synth,)
        Labels for X_synth (all should be the minority class label).

    k : int, default=3
        Number of sampling iterations. Scores are averaged across iterations
        to reduce variance from random sampling.

    minority_class : int/str, optional
        Label of the minority class. Auto-detected as least frequent if None.

    cluster_threshold : int, default=10_000
        If a combined dataset exceeds this size, cluster-based sub-sampling is
        applied before calling problexity (to keep runtimes tractable).

    return_detailed : bool, default=False
        If True, a third element is added to the return tuple containing the
        full per-metric arrays and iteration details.
        If False, only (keys, values) are returned — CSV-friendly flat output.

    random_state : int, default=42
        Seed for reproducibility of sampling steps.

    Returns
    -------
    keys   : list[str]
        Metric names:
            'complexity_real_score'  — mean problexity score on real datasets
            'complexity_synth_score' — mean problexity score on synthetic datasets
            'complexity_score_diff'  — |real - synth| (lower = more similar)
            'complexity_mean_sim'    — mean per-metric similarity in [0,1]
                                       (higher = more similar)
            'complexity_successful_iters' — how many of k iterations succeeded

    values : list[float]
        Corresponding numeric values.

    detailed : dict  (only when return_detailed=True)
        Keys: 'metrics_names', 'mean_real_complexity', 'mean_synth_complexity',
              'std_real_complexity', 'std_synth_complexity', 'similarity_scores',
              'sampling_strategy', 'iteration_details', 'class_imbalance_ratio'

    Notes
    -----
    Algorithm
    ~~~~~~~~~
    1. Separate X_real into minority (X_min) and majority (X_maj).
    2. Decide sampling strategy:
         • synth_larger (n_synth ≥ n_real_min):
             sample n_real_min from synthetic; use all real minority
         • real_larger (n_synth < n_real_min):
             sample n_synth from real minority; use all synthetic
    3. For each of k iterations:
         a. Sample minority portion (per strategy) and a proportional majority slice
         b. Construct real_dataset  = sampled_real_min   + sampled_majority
                        synth_dataset = sampled_synth_min  + sampled_majority
         c. Apply cluster sub-sampling if either exceeds cluster_threshold
         d. Fit problexity.ComplexityCalculator on both; store scores
    4. Average metrics across successful iterations.
    5. Return flat (keys, values) and optionally the detailed dict.
    """
    np.random.seed(random_state)

    # ── Type conversion: accept DataFrame, ndarray, list ─────────────────────
    X_real  = _to_array(X_real)
    y_real  = np.array(y_real).ravel()
    X_synth = _to_array(X_synth)
    y_synth = np.array(y_synth).ravel()

    # ── Auto-detect minority class ────────────────────────────────────────────
    unique_classes, class_counts = np.unique(y_real, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected binary classification, got {len(unique_classes)} classes: {unique_classes}"
        )
    if minority_class is None:
        minority_class = unique_classes[np.argmin(class_counts)]
        print(f"[complexity_fidelity] Auto-detected minority class: {minority_class}")

    # ── Split real data into minority / majority ───────────────────────────────
    min_mask = y_real == minority_class
    X_real_min, y_real_min = X_real[min_mask],  y_real[min_mask]
    X_real_maj, y_real_maj = X_real[~min_mask], y_real[~min_mask]

    n_real_min = len(X_real_min)
    n_synth    = len(X_synth)
    n_real_maj = len(X_real_maj)

    class_imbalance_ratio = n_real_min / n_real_maj   # < 1 for imbalanced data

    # ── Sampling strategy ─────────────────────────────────────────────────────
    if n_synth >= n_real_min:
        sampling_strategy = "synth_larger"
        sample_size_min   = n_real_min
        print(f"[complexity_fidelity] Strategy: synth_larger "
              f"(sampling {sample_size_min} from synthetic)")
    else:
        sampling_strategy = "real_larger"
        sample_size_min   = n_synth
        print(f"[complexity_fidelity] Strategy: real_larger "
              f"(sampling {sample_size_min} from real minority)")

    # ── k iterations ─────────────────────────────────────────────────────────
    iter_real_complexities = []
    iter_synth_complexities = []
    iter_real_scores  = []
    iter_synth_scores = []
    metrics_names     = None
    iteration_details = []

    for it in range(k):
        print(f"  Iteration {it + 1}/{k}")
        try:
            # Sample minority portion
            if sampling_strategy == "synth_larger":
                s_idx = np.random.choice(n_synth,    size=sample_size_min, replace=False)
                X_min_s, y_min_s     = X_synth[s_idx],    y_synth[s_idx]
                X_real_min_it = X_real_min.copy()
                y_real_min_it = y_real_min.copy()
            else:
                r_idx = np.random.choice(n_real_min, size=sample_size_min, replace=False)
                X_real_min_it, y_real_min_it = X_real_min[r_idx], y_real_min[r_idx]
                X_min_s, y_min_s             = X_synth.copy(), y_synth.copy()

            # Sample majority to maintain original imbalance ratio
            maj_size = int(sample_size_min / class_imbalance_ratio)
            maj_size = min(maj_size, n_real_maj)
            m_idx = np.random.choice(n_real_maj, size=maj_size, replace=False)
            X_maj_s, y_maj_s = X_real_maj[m_idx], y_real_maj[m_idx]

            # Build complete datasets for this iteration
            X_real_it  = np.vstack([X_real_min_it, X_maj_s])
            y_real_it  = np.hstack([y_real_min_it, y_maj_s])
            X_synth_it = np.vstack([X_min_s, X_maj_s])
            y_synth_it = np.hstack([y_min_s, y_maj_s])

            # Cluster sub-sampling for large datasets
            clustering_applied = False
            if len(X_real_it) > cluster_threshold:
                print(f"    Applying cluster sub-sampling ({len(X_real_it)} samples)")
                X_real_it,  y_real_it  = _cluster_sampling(X_real_it,  y_real_it,  random_state=random_state)
                X_synth_it, y_synth_it = _cluster_sampling(X_synth_it, y_synth_it, random_state=random_state)
                clustering_applied = True

            # Compute complexity
            cc_real  = px.ComplexityCalculator().fit(X_real_it,  y_real_it)
            cc_synth = px.ComplexityCalculator().fit(X_synth_it, y_synth_it)

            iter_real_complexities.append(cc_real.complexity)
            iter_synth_complexities.append(cc_synth.complexity)
            iter_real_scores.append(cc_real.score())
            iter_synth_scores.append(cc_synth.score())

            if metrics_names is None:
                metrics_names = cc_real._metrics()

            iteration_details.append({
                'iteration':             it + 1,
                'real_dataset_size':     len(X_real_it),
                'synth_dataset_size':    len(X_synth_it),
                'minority_sample_size':  sample_size_min,
                'majority_sample_size':  maj_size,
                'imbalance_ratio':       sample_size_min / maj_size,
                'clustering_applied':    clustering_applied,
                'real_score':            cc_real.score(),
                'synth_score':           cc_synth.score(),
                'failed':                False,
            })
            print(f"    real={cc_real.score():.3f}  synth={cc_synth.score():.3f}")

        except Exception as e:
            print(f"    Warning: iteration {it + 1} failed — {e}")
            if metrics_names is not None:
                n_m = len(metrics_names)
                iter_real_complexities.append([0.0] * n_m)
                iter_synth_complexities.append([0.0] * n_m)
            iter_real_scores.append(0.0)
            iter_synth_scores.append(0.0)
            iteration_details.append({'iteration': it + 1, 'failed': True, 'error': str(e)})

    # ── Aggregate across iterations ───────────────────────────────────────────
    successful = sum(1 for d in iteration_details if not d.get('failed', False))

    if iter_real_complexities and metrics_names:
        real_arr  = np.array(iter_real_complexities)
        synth_arr = np.array(iter_synth_complexities)

        mean_real  = np.nanmean(real_arr,  axis=0)
        mean_synth = np.nanmean(synth_arr, axis=0)
        std_real   = np.nanstd(real_arr,   axis=0)
        std_synth  = np.nanstd(synth_arr,  axis=0)

        valid = ~(np.isnan(mean_real) | np.isnan(mean_synth))
        mean_real_c  = mean_real[valid]
        mean_synth_c = mean_synth[valid]
        std_real_c   = std_real[valid]
        std_synth_c  = std_synth[valid]
        names_clean  = [metrics_names[i] for i in range(len(metrics_names)) if valid[i]]

        rel_diff        = np.abs(mean_real_c - mean_synth_c) / (np.abs(mean_real_c) + 1e-8)
        similarity_scores = 1.0 / (1.0 + rel_diff)
        mean_similarity   = float(np.mean(similarity_scores)) if len(similarity_scores) else 0.0

        # Overall problexity scores (exclude failed iterations = 0.0 score)
        real_sc  = np.array(iter_real_scores);  real_sc  = real_sc[real_sc  > 0]
        synth_sc = np.array(iter_synth_scores); synth_sc = synth_sc[synth_sc > 0]

        mean_real_score  = float(np.mean(real_sc))  if len(real_sc)  else 0.0
        mean_synth_score = float(np.mean(synth_sc)) if len(synth_sc) else 0.0

    else:
        names_clean       = []
        mean_real_c       = np.array([])
        mean_synth_c      = np.array([])
        std_real_c        = np.array([])
        std_synth_c       = np.array([])
        similarity_scores = np.array([])
        mean_similarity   = 0.0
        mean_real_score   = 0.0
        mean_synth_score  = 0.0

    # ── Flat output (keys, values) ────────────────────────────────────────────
    score_diff = abs(mean_real_score - mean_synth_score)

    keys = [
        'complexity_real_score',
        'complexity_synth_score',
        'complexity_score_diff',
        'complexity_mean_sim',
        'complexity_successful_iters',
    ]
    values = [
        mean_real_score,
        mean_synth_score,
        score_diff,
        mean_similarity,
        float(successful),
    ]

    print(f"[complexity_fidelity] Done — "
          f"real={mean_real_score:.3f}  synth={mean_synth_score:.3f}  "
          f"sim={mean_similarity:.3f}  ({successful}/{k} iterations OK)")

    if not return_detailed:
        return keys, values

    # ── Detailed output ───────────────────────────────────────────────────────
    detailed = {
        'metrics_names':          names_clean,
        # use of .tolist() to make them JSON serializable
        'mean_real_complexity':   mean_real_c.tolist(),
        'mean_synth_complexity':  mean_synth_c.tolist(),
        'std_real_complexity':    std_real_c.tolist(),
        'std_synth_complexity':   std_synth_c.tolist(),
        'similarity_scores':      similarity_scores.tolist(),
        'sampling_strategy':      sampling_strategy,
        'class_imbalance_ratio':  float(class_imbalance_ratio),
        'iteration_details':      iteration_details,
    }
    return keys, values, detailed
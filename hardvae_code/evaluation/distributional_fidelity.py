"""
 distributional_fidelity
    This module implements various SDV (Synthetic Data Vault) metrics for evaluating the statistical properties of synthetic data.
    
    Tests used:
        - Kolmogorv Smirnov test statistic for comparing pairs of numerical cols (real vs. synthetic)
        - Total Variation Distance (TVD) test statistic for comparing pairs of categorical pairs (real vs. synthetic)
 
"""

# import sdmetrics
from sdmetrics.single_column import  KSComplement, TVComplement
import numpy as np
import pandas as pd

def avgKSComplement(real_data_num: pd.DataFrame, synthetic_data_num: pd.DataFrame) -> float:
    """
    Average KSComplement score over all numerical columns.
    KSComplement = 1 - KS_statistic → higher is better (1.0 = identical distributions).
    """
    scores = {}
    for col in real_data_num.columns:
        scores[col] = KSComplement.compute(real_data_num[col], synthetic_data_num[col])
    return float(np.mean(list(scores.values()))) if scores else np.nan
 

def avgTVComplement(real_data_cat: pd.DataFrame, synthetic_data_cat: pd.DataFrame) -> float:
    """
    Average TVComplement score over all categorical (one-hot) columns.
    TVComplement = 1 - TVD → higher is better (1.0 = identical distributions).
    """
    scores = {}
    for col in real_data_cat.columns:
        scores[col] = TVComplement.compute(
            real_data=real_data_cat[col],
            synthetic_data=synthetic_data_cat[col]
        )
    return float(np.mean(list(scores.values()))) if scores else np.nan


def distributional_fidelity_calculate(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    data_info: dict,
    feature_names: list = None,
) -> tuple:
    """
    Calculate distributional fidelity using KSComplement (numerical) and
    TVComplement (categorical one-hot columns), both from sdmetrics.
 
    Parameters
    ----------
    real_data       : np.ndarray  shape (n_real, total_cols)
                      Real minority samples from the training set.
    synthetic_data  : np.ndarray  shape (n_synth, total_cols)
                      Synthetic minority samples from the generator.
    data_info       : dict  {'n_numerical': int, 'cat_dims': list[int]}
                      Produced by preprocess_data after fit_transform().
                      The first n_numerical columns are numerical (QuantileTransformer
                      output, range [0,1]); the remaining columns are one-hot blocks
                      whose sizes are given by cat_dims.
                      Examples:
                        BCWDD:       {'n_numerical': 30, 'cat_dims': []}
                        Hypothyroid: {'n_numerical': 6,
                                      'cat_dims': [2,2,1,2,2,2,2,2,2,2,2,5,3,2,2,2]}
    feature_names   : list[str], optional
                      Column names in the same order as the array columns
                      (i.e. preprocessor.feature_names_out). If None, columns
                      are named  num_0 … num_k  and  cat_0 … cat_m.
 
    Returns
    -------
    keys   : list[str]   metric names
    values : list[float] corresponding scores
                         KSComplement and TVComplement are both in [0, 1],
                         higher = more similar to real data.
                         np.nan is returned when a split has zero columns.
    """
    n_num = data_info['n_numerical']
    cat_dims = data_info.get('cat_dims', [])
    n_cat_total = sum(cat_dims)
 
    # ── Convert ndarrays → DataFrames with named columns ──────────────────
    # sdmetrics operates on pd.Series; DataFrames let us iterate by column name
    if feature_names is not None:
        num_names = feature_names[:n_num]
        cat_names = feature_names[n_num:n_num + n_cat_total]
    else:
        num_names = [f"num_{i}" for i in range(n_num)]
        cat_names = [f"cat_{i}" for i in range(n_cat_total)]
 
    # ── Numerical slice ────────────────────────────────────────────────────
    if n_num > 0:
        real_num = pd.DataFrame(
            real_data[:, :n_num].astype(np.float64),
            columns=num_names
        )
        synth_num = pd.DataFrame(
            synthetic_data[:, :n_num].astype(np.float64),
            columns=num_names
        )
        ks_score = avgKSComplement(real_num, synth_num)
    else:
        ks_score = np.nan
 
    # ── Categorical slice ──────────────────────────────────────────────────
    # One-hot columns are binary {0, 1} floats. TVComplement treats each
    # column as a discrete distribution over {0.0, 1.0} — this is correct
    # because TVD on a Bernoulli(p) variable reduces to |p_real - p_synth|,
    # which is exactly the per-column proportion difference.
    if n_cat_total > 0:
        real_cat = pd.DataFrame(
            real_data[:, n_num: n_num + n_cat_total].astype(np.float64),
            columns=cat_names
        )
        synth_cat = pd.DataFrame(
            synthetic_data[:, n_num: n_num + n_cat_total].astype(np.float64),
            columns=cat_names
        )
        tv_score = avgTVComplement(real_cat, synth_cat)
    else:
        tv_score = np.nan
 
    # ── Aggregate ──────────────────────────────────────────────────────────
    # Average only over the splits that exist (ignore nan)
    valid = [s for s in [ks_score, tv_score] if not np.isnan(s)]
    avg_dist_fid = float(np.mean(valid)) if valid else np.nan
 
    keys   = ['avgKSComplement', 'avgTVComplement', 'DistFidScore']
    values = [ks_score,           tv_score,          avg_dist_fid]

 
    return keys, values  





# from scipy.stats import ks_2samp

# def evaluate_synthetic_data(X_real, y_real, X_synth, y_synth, verbose=True):
#     """
#     Comprehensive evaluation of synthetic data quality.
    
#     Args:
#         X_real: Real feature data
#         y_real: Real labels
#         X_synth: Synthetic feature data
#         y_synth: Synthetic labels
#         verbose: Whether to print results
    
#     Returns:
#         Dictionary of evaluation metrics
#     """
#     results = {}
    
#     # 1. Statistical Similarity (Kolmogorov-Smirnov test)
#     ks_scores = []
#     for i in range(X_real.shape[1]):
#         ks_stat, p_value = ks_2samp(X_real[:, i], X_synth[:, i])
#         ks_scores.append(ks_stat)
    
#     results['ks_mean'] = np.mean(ks_scores)
#     results['ks_std'] = np.std(ks_scores)
    

    
#     if verbose:
#         print("=== Synthetic Data Evaluation ===")
#         print(f"KS Test (lower is better): {results['ks_mean']:.4f} ± {results['ks_std']:.4f}")
    
#     return results
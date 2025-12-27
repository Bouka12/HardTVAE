"""
 sdv_metrics
    This module implements various SDV (Synthetic Data Vault) metrics for evaluating the statistical properties of synthetic data.
"""

# import sdmetrics
from sdmetrics.single_column import RangeCoverage, KSComplement, BoundaryAdherence, StatisticSimilarity
import numpy as np


# sdmetrics: similarity metrics
def avgRangeCoverage(real_data, synthetic_data):
    avgRCov = {}
    for col in real_data.columns:
        avgRCov[col] = RangeCoverage.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgRCov.values()))

def avgKSComplement(real_data, synthetic_data):
    avgKSC = {}
    for col in real_data.columns:
        avgKSC[col] = KSComplement.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgKSC.values()))

def avgBoundaryAdherence(real_data, synthetic_data):
    avgBAdher = {}
    for col in real_data.columns:
        avgBAdher[col] = BoundaryAdherence.compute(real_data[col], synthetic_data[col])
    return np.mean(list(avgBAdher.values()))
# added similarity metric for numerical data
def avgStatisticalSimilarity(real_data, synthetic_data):
    avgStatSim = {}
    for col in real_data.columns:
        avgStatSim[col] = StatisticSimilarity.compute(real_data[col], synthetic_data[col], statistic='mean')
    return np.mean(list(avgStatSim.values()))

def similarity_metrics(real_data, synthetic_data):
    avgRangeCoverage_ = avgRangeCoverage(real_data, synthetic_data)
    avgKSComplement_ = avgKSComplement(real_data, synthetic_data)
    avgBoundaryAdherence_ = avgBoundaryAdherence(real_data, synthetic_data)
    avgStatisticalSimilarity_ = avgStatisticalSimilarity(real_data, synthetic_data)
    return ['avgRangeCoverage', 'avgKSComplement', 'avgBoundaryAdherence', 'avgStatisticalSimilarity'], [avgRangeCoverage_, avgKSComplement_, avgBoundaryAdherence_, avgStatisticalSimilarity_]
    

from scipy.stats import ks_2samp

def evaluate_synthetic_data(X_real, y_real, X_synth, y_synth, verbose=True):
    """
    Comprehensive evaluation of synthetic data quality.
    
    Args:
        X_real: Real feature data
        y_real: Real labels
        X_synth: Synthetic feature data
        y_synth: Synthetic labels
        verbose: Whether to print results
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # 1. Statistical Similarity (Kolmogorov-Smirnov test)
    ks_scores = []
    for i in range(X_real.shape[1]):
        ks_stat, p_value = ks_2samp(X_real[:, i], X_synth[:, i])
        ks_scores.append(ks_stat)
    
    results['ks_mean'] = np.mean(ks_scores)
    results['ks_std'] = np.std(ks_scores)
    

    
    if verbose:
        print("=== Synthetic Data Evaluation ===")
        print(f"KS Test (lower is better): {results['ks_mean']:.4f} ± {results['ks_std']:.4f}")
    
    return results
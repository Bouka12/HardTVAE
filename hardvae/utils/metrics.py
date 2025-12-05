"""
Hardness Metric Definitions for HardVAE

This module contains the definitions and groupings of instance hardness metrics used in the HardVAE framework.
"""

# Standard PyHard metrics
PYHARD_METRICS = [
    'feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
    'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV',
    'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC',
    'feature_LSR', 'feature_Harmfulness', 'feature_Usefulness',
    'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4'
]

# Custom proposed metrics
CUSTOM_METRICS = [
    'relative_entropy', 'pca_contribution'
]

NO_WEIGHT_METRICS = [None]

# Metric groups for analysis
METRIC_GROUPS = {
    "linear": ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P', 'feature_TD_U'],
    "neighborhood_based": ['feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB'],
    "network_based": ['feature_N1', 'feature_N2'],
    "feature_based": ['feature_LSC', 'feature_LSR'],
    "other": ['feature_Harmfulness', 'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4'],
    "custom": ['relative_entropy', 'pca_contribution']
}

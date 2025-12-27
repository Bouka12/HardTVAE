# HardVAE Experimental Results

This directory contains the experimental results from the HardVAE research study on hardness-aware synthetic data generation for imbalanced classification.

## Directory Structure

```
results/
├── data/                                    # Raw experimental data
│   ├── all_ks_results_medical.csv          # Distributional fidelity results
│   └── all_classification_results_medical_complete.csv  # Classification performance
├── analysis/                                # Analysis scripts and outputs
├── README.md                                # This file
└── RESULTS_SUMMARY.md                       # Summary of key findings
```

## Data Files

### 1. `all_ks_results_medical.csv`

**Purpose**: Distributional fidelity evaluation results using Kolmogorov-Smirnov (KS) statistics.

**Columns**:
- `dataset`: Name of the medical dataset (e.g., NewThyroid2)
- `hardness_metric`: Hardness metric used for weighting (blank for baseline, or metric name)
- `random_seed`: Random seed for reproducibility
- `weighting_strategy`: Training strategy (static, curriculum, self_paced)
- `ks_mean`: Mean KS statistic across features (lower is better)
- `ks_std`: Standard deviation of KS statistic

**Data Characteristics**:
- Multiple datasets with varying imbalance ratios
- Baseline configuration (no hardness weighting)
- Multiple hardness metrics: relative_entropy, pca_contribution, feature_kDN, feature_DS, feature_DCP, etc.
- Three weighting strategies: static, curriculum, self_paced
- Multiple random seeds for statistical robustness

**Interpretation**:
- **KS Mean**: Lower values indicate better distribution matching between real and synthetic data
- **KS Std**: Indicates consistency across features
- Baseline provides reference point for comparison

### 2. `all_classification_results_medical_complete.csv`

**Purpose**: Utility evaluation results measuring downstream classification performance.

**Columns**:
- `Classifier`: Classification algorithm (LogisticRegression, RandomForestClassifier, GaussianNB, SVC, KNeighborsClassifier)
- `cv_k_folds`: Number of cross-validation folds (typically 3)
- `Random_State`: Random seed for reproducibility
- `Precision`: Precision score [0, 1]
- `Recall`: Recall score [0, 1]
- `F1 Score`: F1 score [0, 1]
- `Specificity`: Specificity score [0, 1]
- `Balanced Accuracy`: Balanced accuracy [0, 1]
- `dataset`: Dataset name
- `hardness_metric`: Hardness metric used (blank for baseline)
- `seed`: Random seed identifier
- `weighting_strategy`: Training strategy

**Data Characteristics**:
- Evaluation with 5 different classifiers
- Multiple performance metrics per classifier
- Cross-validation for robustness
- Comprehensive coverage of hardness metrics and strategies

**Interpretation**:
- **Higher scores are better** for all metrics
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: Proportion of actual negatives correctly identified
- **Balanced Accuracy**: Average of recall for each class

## Analysis Scripts

The `analysis/` directory contains Python scripts for analyzing and visualizing results:

- `distributional_analysis.py`: Analyzes KS statistics and creates visualizations
- `hardness_analysis.py`: Analyzes hardness fidelity across metrics and strategies
- `topological_analysis.py`: Analyzes topological fidelity results
- `complexity_analysis.py`: Analyzes complexity-based fidelity results
- `utility_analysis.py`: Analyzes classification performance gains


## Experimental Setup

### Datasets
Medical datasets with varying characteristics:
- Different imbalance ratios
- Different feature dimensions
- Different sample sizes

### Hardness Metrics
- **PyHard Metrics** (18+): kDN, DS, DCP, TD_P, TD_U, CL, CLD, MV, CB, N1, N2, LSC, LSR, Harmfulness, F1, F2, F3, F4


### Weighting Strategies
1. **Static**: Constant weights based on hardness scores
2. **Curriculum**: Gradually increase focus on hard instances
3. **Self-Paced**: Dynamically adjust weights based on learning progress

### Evaluation Metrics
- **Distributional Fidelity**: KS statistics
- **Classification Utility**: Precision, Recall, F1, Specificity, Balanced Accuracy
- **Topological Fidelity**: Persistent homology distances (Bottleneck, Wasserstein)
- **Complexity Fidelity**: Data complexity metric preservation
- **Hardness Fidelity**: Instance difficulty pattern similarity

## Reproducibility

To reproduce these results:

1. **Install HardVAE**:
   ```bash
   pip install -e .
   ```

2. **Prepare datasets**: Ensure medical datasets are available

3. **Run experiments**:
   ```bash
   python examples/full_pipeline.py --datasets medical --output results/
   ```

4. **Analyze results**:
   ```bash
   python results/analysis/distributional_analysis.py
   python results/analysis/utility_analysis.py
   ```



## Data Availability

All experimental data is provided in CSV format for:
- Easy integration with statistical software (R, Python, MATLAB)
- Transparency and reproducibility
- Community verification and extension

## Citation

If you use these experimental results in your research, please cite (to be updated):

```bibtex
@article{hardvae2024,
  title={HardVAE: Hardness-Aware Synthetic Data Generation for Imbalanced Classification },
  author={},
  journal={},
  year={},
  volume={XX},
  pages={XX--XX},
  doi={XX.XXXX/XXXXX}
}
```


---

**Data Version**: 1.0  
**Last Updated**: December 2025

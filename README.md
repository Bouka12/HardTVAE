# HardVAE: Hardness-Aware Synthetic Data Generation for Imbalanced Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


A comprehensive Python framework for evaluating and generating high-quality synthetic minority data in imbalanced classification scenarios. HardVAE integrates **instance hardness metrics** with **Conditional Variational Autoencoders (CVAEs)** to produce synthetic data that preserves the complexity characteristics of the original minority class.

## 🎯 Overview

Class imbalance is a fundamental challenge in machine learning, where minority class samples are underrepresented. Traditional oversampling methods like SMOTE generate synthetic data but often fail to preserve the **distribution** of minority instances. HardVAE addresses this by:

1. **Quantifying instance hardness** using multiple metrics (PyHard)
2. **Integrating hardness into CVAE training** through weighted loss functions
3. **Comprehensively evaluating** synthetic data quality across 5 dimensions:
   - Statistical fidelity (distribution matching)
   - Topological fidelity (shape preservation)
   - Instance-level fidelity (hardness preservation)
   - Complexity fidelity (data complexity metrics)
   - Utility evaluation (downstream task performance)

## 🚀 Key Features

### 📊 Multi-Dimensional Evaluation Framework
- **Statistical Evaluation**: Feature-wise distribution similarity using meta-features and Kolmogorov-Smirnov tests
- **Topological Data Analysis**: Persistent homology for shape and structure preservation
- **Instance Hardness Analysis**: Meta-features from six groups: feature-base, geometry-based. etc
- **Complexity Analysis**: Data complexity patterns using problexity package
- **Utility Assessment**: Train-on-balanced data (mixed) → test-on-real evaluation with multiple classifiers

### 🔧 Hardness Calculation
- **18+ PyHard Metrics**: Linear, neighborhood-based, network-based, and feature-based hardness measures
- **Flexible Architecture**: Easy integration with custom hardness metrics
- **Metric Grouping**: Organized by category for targeted analysis

### 🧠 CVAE Integration
- **Hardness-Aware Training**: Weighted loss functions based on instance difficulty
- **Flexible Weighting Strategies**: Static, dynamic, and adaptive weighting schemes
- **Conditional Generation**: Generate synthetic data conditioned on class labels, adaptable for multi-class data
- **Imbalanced Dataset Support**: Native handling of class imbalance

### 📈 Comprehensive Visualization & Reporting
- **CSV Exports**: Detailed metrics for further analysis
- **Visualizations**: Plots for utility/fidelity analysis

## 📋 Requirements

### Core Dependencies
```
numpy>=1.23.5,<2.0.0
pandas>=1.5.3,<2.0.0
torch>=2.6.0
scikit-learn>=1.5.2,<2.0.0
scipy>=1.13.0,<2.0.0
matplotlib>=3.9.4,<4.0.0
seaborn>=0.13.2,<1.0.0
```

### Evaluation & Hardness Metrics
```
pymfe>=0.4.0          # Meta-feature extraction
problexity>=1.0.0     # Data complexity metrics
pyhard>=2.2.4         # Instance hardness metrics (optional but recommended)
ripser>=0.6.12        # Topological data analysis 
persim>=0.3.8         # Persistence diagram comparison 
```

### Utility Evaluation
```
imblearn>=0.0      # Imbalanced learning utilities
```

## 🔧 Installation

### Clone and Install from Source
```bash
git clone https://github.com/Bouka12/HardVAE.git
cd HardVAE
pip install -r requirements.txt
```


## 📖 Quick Start

### Basic Usage: Evaluate Synthetic Data Quality

```python
from synthetic_data_evaluator import SyntheticDataEvaluator
import numpy as np

# Prepare your data
X_real = np.random.randn(100, 10)      # Real minority data
y_real = np.ones(100)                   # Labels
X_synth = np.random.randn(100, 10)     # Synthetic minority data
y_synth = np.ones(100)

# Initialize evaluator
evaluator = SyntheticDataEvaluator(random_state=42)

# Run comprehensive evaluation
results = evaluator.evaluate_all(
    X_real, y_real,
    X_synth, y_synth,
    save_path="./results",
    dataset_name="my_dataset"
)
```


### Calculate Instance Hardness

```python
from hardness_module import HardnessCalculator

# Initialize calculator
hardness_calc = HardnessCalculator(random_state=42)

# Calculate hardness scores
hardness_scores = hardness_calc.calculate_hardness_scores(
    X_data, y_data,
    metrics=['feature_kDN', 'feature_DS', 'relative_entropy']
)

print(hardness_scores.head())
```


## 📊 Evaluation Results Interpretation


### Component Scores
Each evaluation component returns 
- **Statistical**: Distribution matching quality
- **Topological**: Shape and structure preservation
- **Hardness**: Instance difficulty pattern similarity
- **Complexity**: Data complexity metric preservation




## 🎓 Citation

If you use HardVAE in your research, please cite:

```bibtex
@article{hardvae,
  title={HardVAE: Hardness-Aware Synthetic Data Generation for Imbalanced Classification},
  author={X,Y,Z},
  journal={Journal-X},
  year={202X},
  volume={XX},
  pages={XX--XX},
  doi={XX.XXXX/XXXXX}
}
```



## 📖 References

1. Lorena, A. C., et al. (2019). "Data complexity meta-features for regression problems." *Machine Learning*, 108(12), 2209-2246.
2. Chazal, F., & Michel, B. (2016). "An introduction to topological data analysis." arXiv preprint arXiv:1710.04019.
3. Smith, M. R., et al. (2014). "Instance hardness levels for instance filtering and weighting." *Journal of Machine Learning Research*, 15(1), 2049-2080.
4. Jordon, J., et al. (2022). "Synthetic data generation and adaptation for machine learning." *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*.



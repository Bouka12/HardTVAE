# HardVAE: Hardness-Aware Synthetic Data Generation for Imbalanced Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A comprehensive Python framework for evaluating and generating high-quality synthetic minority data in imbalanced classification scenarios. HardVAE integrates **instance hardness metrics** with **Conditional Variational Autoencoders (CVAEs)** to produce synthetic data that preserves the complexity characteristics of the original minority class.

## 🎯 Overview

Class imbalance is a fundamental challenge in machine learning, where minority class samples are underrepresented. Traditional oversampling methods like SMOTE generate synthetic data but often fail to preserve the **difficulty distribution** of minority instances. HardVAE addresses this by:

1. **Quantifying instance hardness** using multiple metrics (PyHard + custom metrics)
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
- **Instance Hardness Analysis**: Nearest neighbor distance analysis and PyHard metrics
- **Complexity Analysis**: Data complexity patterns using problexity package
- **Utility Assessment**: Train-on-synthetic → test-on-real evaluation with multiple classifiers

### 🔧 Hardness Calculation
- **18+ PyHard Metrics**: Linear, neighborhood-based, network-based, and feature-based hardness measures
- **Flexible Architecture**: Easy integration with custom hardness metrics
- **Metric Grouping**: Organized by category for targeted analysis

### 🧠 CVAE Integration
- **Hardness-Aware Training**: Weighted loss functions based on instance difficulty
- **Flexible Weighting Strategies**: Static, dynamic, and adaptive weighting schemes
- **Conditional Generation**: Generate synthetic data conditioned on class labels
- **Imbalanced Dataset Support**: Native handling of class imbalance

### 📈 Comprehensive Visualization & Reporting
- **CSV Exports**: Detailed metrics for further analysis

## 📋 Requirements

### Core Dependencies
```
numpy>=1.19.0
pandas>=1.1.0
torch>=1.9.0
scikit-learn>=0.24.0
scipy>=1.5.0
matplotlib>=3.3.0
```

### Evaluation & Hardness Metrics
```
pymfe>=0.4.0          # Meta-feature extraction
problexity>=1.0.0     # Data complexity metrics
pyhard>=0.2.0         # Instance hardness metrics (optional but recommended)
ripser>=0.0.16        # Topological data analysis (optional)
persim>=0.2.0         # Persistence diagram comparison (optional)
```

### Utility Evaluation
```
imblearn>=0.8.0       # Imbalanced learning utilities
```

## 🔧 Installation

### Option 1: Clone and Install from Source
```bash
git clone https://github.com/Bouka12/HardVAE.git
cd HardVAE
pip install -r requirements.txt
```

### Option 2: Install with Optional Dependencies
```bash
# For full functionality including topological analysis
pip install -r requirements-full.txt
```

### Option 3: Development Installation
```bash
git clone https://github.com/Bouka12/HardVAE.git
cd HardVAE
pip install -e .
```

## 📖 Quick Start

### Basic Usage: Evaluate Synthetic Data Quality

```python
from hardvae.evaluation import SyntheticDataEvaluator
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

# Access results
print(f"Overall Quality Score: {results['summary']['overall_quality_score']:.3f}")
print(f"Assessment: {results['summary']['assessment']}")
```

### Calculate Instance Hardness

```python
from hardvae.core import HardnessCalculator

# Initialize calculator
hardness_calc = HardnessCalculator(random_state=42)

# Calculate hardness scores
hardness_scores = hardness_calc.calculate_hardness_scores(
    X_data, y_data,
    metrics=['feature_kDN', 'feature_DS', 'relative_entropy']
)

print(hardness_scores.head())
```

### Train Hardness-Aware CVAE

```python
from hardvae.integration import HardnessAwareCVAETrainer, TabularCVAE
from hardvae.core import HardnessCalculator

# Initialize CVAE
cvae = TabularCVAE(
    input_dim=X_train.shape[1],
    latent_dim=10,
    condition_dim=1,
    hidden_dims=[128, 64]
)

# Initialize hardness calculator
hardness_calc = HardnessCalculator()

# Initialize trainer
trainer = HardnessAwareCVAETrainer(cvae, hardness_calc)

# Train with hardness weighting
trainer.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    hardness_metric='feature_kDN'
)

# Generate synthetic data
X_synthetic = trainer.generate(n_samples=500, condition=1)
```

## 📊 Evaluation Results Interpretation


### Component Scores
Each evaluation component returns a similarity score [0, 1]:
- **Statistical**: Distribution matching quality
- **Topological**: Shape and structure preservation
- **Hardness**: Instance difficulty pattern similarity
- **Complexity**: Data complexity metric preservation
- **Utility**: Downstream task performance

## 📁 Project Structure

```
HardVAE/
├── hardvae/                          # Main package
│   ├── __init__.py
│   ├── core/                         # Core hardness functionality
│   │   ├── __init__.py
│   │   ├── hardness.py              # HardnessCalculator class
│   │   └── metrics.py               # Hardness metric definitions
│   ├── evaluation/                   # Evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py             # SyntheticDataEvaluator class
│   │   ├── visualizer.py            # Visualization utilities
│   │   └── metrics.py               # Evaluation metrics
│   ├── integration/                  # CVAE integration
│   │   ├── __init__.py
│   │   ├── cvae.py                  # TabularCVAE architecture
│   │   ├── trainer.py               # HardnessAwareCVAETrainer
│   │   └── utils.py                 # Integration utilities
│   └── utils/                        # General utilities
│       ├── __init__.py
│       ├── data.py                  # Data loading and preprocessing
│       └── config.py                # Configuration management
├── examples/                         # Example scripts
│   ├── basic_evaluation.py
│   ├── hardness_calculation.py
│   ├── cvae_training.py
│   └── full_pipeline.py
├── tests/                            # Unit tests
│   ├── test_hardness.py
│   ├── test_evaluation.py
│   └── test_cvae.py
├── docs/                             # Documentation
│   ├── api_reference.md
│   ├── tutorials.md
│   └── methodology.md
├── requirements.txt                  # Core dependencies
├── requirements-full.txt             # All dependencies
├── setup.py                          # Package setup
├── LICENSE
└── README.md
```

## 🔬 Methodology

### Instance Hardness Metrics

HardVAE implements multiple hardness metrics organized into categories:

#### PyHard Metrics (18 metrics)
- **Linear Separability**: kDN, DS, DCP, TD_P, TD_U
- **Neighborhood-Based**: CL, CLD, MV, CB
- **Network-Based**: N1, N2
- **Feature-Based**: LSC, LSR
- **Other**: Harmfulness, F1, F2, F3, F4



### Evaluation Framework

The evaluation framework assesses synthetic data quality across five complementary dimensions:

| View | Goal | Method | Reference |
|------|------|--------|-----------|
| **Statistical Fidelity** | Match feature-wise distributions | Statistical meta-features, KS test | Lorena et al. (2019) |
| **Topological Fidelity** | Preserve shape & structure | Persistent homology | Chazal & Michel (2016) |
| **Instance-Level Fidelity** | Preserve instance difficulty | Instance hardness metrics | Smith et al. (2014) |
| **Complexity Fidelity** | Preserve classification complexity | Meta-feature complexity measures | Lorena et al. (2019) |
| **Utility Evaluation** | Support downstream modeling | Train-on-synth → test-on-real | Jordon et al. (2022) |

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

## 📚 Documentation

- **[Examples](examples/)**: Complete working examples


## 🙏 Acknowledgments

- PyHard library for hardness metric calculations
- Problexity for data complexity analysis
- Ripser and Persim for topological data analysis
- Scikit-learn and PyTorch communities

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on [GitHub Issues](https://github.com/Bouka12/HardVAE/issues)
- Contact: anonymous

## 📖 References

1. Lorena, A. C., et al. (2019). "Data complexity meta-features for regression problems." *Machine Learning*, 108(12), 2209-2246.
2. Chazal, F., & Michel, B. (2016). "An introduction to topological data analysis." arXiv preprint arXiv:1710.04019.
3. Smith, M. R., et al. (2014). "Instance hardness levels for instance filtering and weighting." *Journal of Machine Learning Research*, 15(1), 2049-2080.
4. Jordon, J., et al. (2022). "Synthetic data generation and adaptation for machine learning." *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*.

---

**Last Updated**: December 2025  
**Version**: 1.0.0

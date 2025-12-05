# HardVAE Project Structure Guide

This document provides a detailed overview of the HardVAE project organization and the purpose of each module.

## Directory Organization

```
HardVAE/
├── hardvae/                              # Main package directory
│   ├── __init__.py                       # Package initialization
│   │
│   ├── core/                             # Core hardness calculation functionality
│   │   ├── __init__.py
│   │   ├── hardness.py                   # HardnessCalculator class (from hardness_module_improved.py)
│   │   │   ├── HardnessCalculator        # Main class for hardness calculation
│   │   │   ├── CVAEHardnessIntegrator    # Integration utilities for CVAE
│   │   │   └── Metrics definitions       # PyHard metrics and custom metrics
│   │   │
│   │   └── metrics.py                    # Hardness metric definitions and groupings
│   │       ├── PYHARD_METRICS            # 18+ PyHard metrics
│   │       ├── CUSTOM_METRICS            # Relative entropy, PCA-based
│   │       └── METRIC_GROUPS             # Organized metric categories
│   │
│   ├── evaluation/                       # Comprehensive evaluation framework
│   │   ├── __init__.py
│   │   │
│   │   ├── evaluator.py                  # Main evaluation class (from synthetic_data_evaluator.py)
│   │   │   ├── SyntheticDataEvaluator    # Primary evaluation interface
│   │   │   ├── statistical_evaluation()  # Distribution similarity
│   │   │   ├── complexity_evaluation()   # Data complexity metrics
│   │   │   ├── hardness_evaluation()     # Instance hardness analysis
│   │   │   ├── topological_evaluation()  # Persistent homology
│   │   │   ├── utility_evaluation()      # Model-based assessment
│   │   │   └── clustering_evaluation()   # Cluster structure analysis
│   │   │
│   │   ├── visualizer.py                 # Visualization utilities (from synthetic_data_evaluator.py)
│   │   │   ├── create_comprehensive_dashboard()
│   │   │   ├── create_interactive_dashboard()
│   │   │   └── create_comparison_matrix()
│   │   │
│   │   ├── metrics.py                    # Evaluation metrics and similarity calculations
│   │   │   ├── SimilarityCalculator      # Multiple similarity metrics
│   │   │   └── DataQualityMetrics        # Quality assessment utilities
│   │   │
│   │   ├── distributional_analysis.py    # Fidelity analysis for distributional metrics
│   │   │   ├── KS statistic processing
│   │   │   ├── Boxplot generation
│   │   │   └── Heatmap visualization
│   │   │
│   │   ├── hardness_analysis.py          # Fidelity analysis for hardness metrics
│   │   │   ├── Hardness fidelity processing
│   │   │   ├── Weighting strategy comparison
│   │   │   └── Per-metric analysis
│   │   │
│   │   └── topological_analysis.py       # Fidelity analysis for topological metrics
│   │       ├── Persistent homology processing
│   │       ├── Bottleneck/Wasserstein distance analysis
│   │       └── Topology preservation visualization
│   │
│   ├── integration/                      # CVAE integration and training
│   │   ├── __init__.py
│   │   │
│   │   ├── cvae.py                       # CVAE architecture (from cvae_hardness_integration.py)
│   │   │   ├── TabularCVAE               # Conditional VAE for tabular data
│   │   │   ├── encode()                  # Encoder forward pass
│   │   │   ├── decode()                  # Decoder forward pass
│   │   │   └── forward()                 # Full CVAE forward pass
│   │   │
│   │   ├── trainer.py                    # Hardness-aware training (from cvae_hardness_integration.py)
│   │   │   ├── HardnessAwareCVAETrainer  # Main trainer class
│   │   │   ├── train()                   # Training loop
│   │   │   ├── calculate_hardness_scores()
│   │   │   ├── cvae_loss()               # Weighted CVAE loss
│   │   │   └── generate()                # Synthetic data generation
│   │   │
│   │   └── utils.py                      # Integration utilities
│   │       ├── Weighting strategies      # Static, curriculum, self-paced
│   │       └── Loss computation helpers
│   │
│   └── utils/                            # General utilities
│       ├── __init__.py
│       │
│       ├── data.py                       # Data loading and preprocessing (from load_data.py)
│       │   ├── load_data()               # Load and preprocess datasets
│       │   ├── split_train_test()        # Train-test splitting
│       │   └── identify_imbalance()      # Imbalance ratio calculation
│       │
│       ├── config.py                     # Configuration management
│       │   ├── Default parameters
│       │   ├── Hardness metric configs
│       │   └── Training hyperparameters
│       │
│       └── metrics.py                    # Utility metrics (from utility_evaluator.py, sdv_metrics.py, classifier_eval.py)
│           ├── UtilityEvaluator          # Utility gain analysis
│           ├── evaluate_classification_model()
│           └── SDMetrics integration
│
├── examples/                             # Example scripts and tutorials
│   ├── basic_evaluation.py               # Quick start evaluation example
│   ├── hardness_calculation.py           # Hardness metric calculation example
│   ├── cvae_training.py                  # CVAE training example
│   ├── full_pipeline.py                  # Complete pipeline example
│   └── README.md                         # Examples documentation
│
├── tests/                                # Unit tests
│   ├── test_hardness.py                  # Hardness calculator tests
│   ├── test_evaluation.py                # Evaluation framework tests
│   ├── test_cvae.py                      # CVAE and trainer tests
│   └── test_utils.py                     # Utility function tests
│
├── docs/                                 # Documentation
│   ├── api_reference.md                  # Complete API documentation
│   ├── tutorials.md                      # Step-by-step tutorials
│   ├── methodology.md                    # Detailed methodology explanation
│   └── fidelity_views.md                 # Explanation of 6 evaluation dimensions
│
├── README.md                             # Project overview and quick start
├── PROJECT_STRUCTURE.md                  # This file
├── setup.py                              # Package installation configuration
├── requirements.txt                      # Core dependencies
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore rules
```

## Module Descriptions

### Core Module (`hardvae/core/`)

**Purpose**: Implements instance hardness calculation and metrics.

**Key Classes**:
- `HardnessCalculator`: Computes hardness scores using PyHard metrics and custom metrics
- `CVAEHardnessIntegrator`: Utilities for integrating hardness into CVAE training

**Key Features**:
- 18+ PyHard metrics (linear, neighborhood-based, network-based, feature-based)
- Custom metrics: relative entropy (ensemble disagreement), PCA-based contributions
- Flexible metric grouping for targeted analysis
- Support for multiple weighting strategies

### Evaluation Module (`hardvae/evaluation/`)

**Purpose**: Comprehensive synthetic data quality assessment across 6 dimensions.

**Key Classes**:
- `SyntheticDataEvaluator`: Main evaluation interface
- `SimilarityCalculator`: Multiple similarity metrics
- `DataQualityMetrics`: Advanced quality assessments

**Evaluation Dimensions**:
1. **Statistical Fidelity**: Feature-wise distribution matching (21 meta-features, KS tests)
2. **Topological Fidelity**: Shape and structure preservation (persistent homology)
3. **Instance-Level Fidelity**: Hardness pattern similarity (nearest neighbor analysis)
4. **Complexity Fidelity**: Data complexity metric preservation (problexity)
5. **Utility Evaluation**: Downstream task performance (train-on-synth → test-on-real)
6. **Clustering Fidelity**: Cluster structure preservation

**Analysis Scripts**:
- `distributional_analysis.py`: Processes and visualizes distributional fidelity metrics
- `hardness_analysis.py`: Analyzes hardness fidelity across metrics and strategies
- `topological_analysis.py`: Visualizes topological fidelity results

### Integration Module (`hardvae/integration/`)

**Purpose**: CVAE architecture and hardness-aware training.

**Key Classes**:
- `TabularCVAE`: Conditional VAE for tabular data synthesis
- `HardnessAwareCVAETrainer`: Training with hardness weighting

**Key Features**:
- Conditional generation based on class labels
- Hardness-weighted loss functions
- Multiple weighting strategies (static, curriculum, self-paced)
- Support for GPU/CPU training
- Synthetic data generation

### Utils Module (`hardvae/utils/`)

**Purpose**: Data loading, preprocessing, and utility functions.

**Key Functions**:
- `load_data()`: Load and preprocess datasets with stratified splitting
- `evaluate_classification_model()`: Multi-classifier evaluation
- `UtilityEvaluator`: Utility gain analysis and statistical testing

## Data Flow

```
Raw Data
    ↓
[load_data.py] → Preprocessing & Stratification
    ↓
Minority Class Data
    ↓
┌─────────────────────────────────────────────┐
│  [hardness_module_improved.py]              │
│  Calculate Instance Hardness Scores         │
│  - PyHard metrics (18+)                     │
│  - Custom metrics (entropy, PCA)            │
└─────────────────────────────────────────────┘
    ↓
Hardness-Weighted Data
    ↓
┌─────────────────────────────────────────────┐
│  [cvae_hardness_integration.py]             │
│  Train Hardness-Aware CVAE                  │
│  - Weighted loss functions                  │
│  - Conditional generation                   │
└─────────────────────────────────────────────┘
    ↓
Synthetic Minority Data
    ↓
┌─────────────────────────────────────────────┐
│  [synthetic_data_evaluator.py]              │
│  Comprehensive Quality Evaluation           │
│  - 6 evaluation dimensions                  │
│  - Statistical, topological, hardness, etc. │
└─────────────────────────────────────────────┘
    ↓
Evaluation Results & Visualizations
    ↓
┌─────────────────────────────────────────────┐
│  [fidelity_analysis scripts]                │
│  Generate Analysis Plots & Tables           │
│  - Distributional analysis                  │
│  - Hardness analysis                        │
│  - Topological analysis                     │
└─────────────────────────────────────────────┘
    ↓
Publication-Ready Results
```

## Key Concepts

### Instance Hardness
Measures the difficulty of classifying individual instances. Higher hardness indicates more challenging samples.

**Metrics Used**:
- **kDN (k-Disagreement Neighbors)**: Disagreement among k-nearest neighbors
- **DS (Disjunct Size)**: Minimum number of samples needed to cover an instance
- **DCP (Density-Certainty Percentage)**: Density and certainty of instance neighborhood
- **Relative Entropy**: Ensemble classifier disagreement
- **PCA Contribution**: Instance contribution to principal components

### Fidelity Views
Six complementary perspectives on synthetic data quality:

1. **Statistical**: Do distributions match?
2. **Topological**: Is the shape preserved?
3. **Hardness**: Are difficulty patterns preserved?
4. **Complexity**: Are complexity metrics preserved?
5. **Utility**: Does it help downstream tasks?
6. **Clustering**: Is cluster structure preserved?

### Weighting Strategies
- **Static**: Constant hardness weights throughout training
- **Curriculum**: Gradually increase focus on hard instances
- **Self-Paced**: Dynamically adjust weights based on learning progress

## Integration with Research Articles

This structure is designed for:
1. **Reproducibility**: Clear module organization with documented functions
2. **Citation**: Each module can be cited with specific DOI/reference
3. **Extensibility**: Easy to add new metrics or evaluation dimensions
4. **Transparency**: Full source code available for peer review
5. **Usability**: Well-documented APIs for researchers to build upon

## Usage Patterns

### Basic Evaluation
```python
from hardvae.evaluation import SyntheticDataEvaluator
evaluator = SyntheticDataEvaluator()
results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)
```

### Hardness Calculation
```python
from hardvae.core import HardnessCalculator
calc = HardnessCalculator()
scores = calc.calculate_hardness_scores(X, y, metrics=['feature_kDN', 'relative_entropy'])
```

### CVAE Training
```python
from hardvae.integration import TabularCVAE, HardnessAwareCVAETrainer
cvae = TabularCVAE(input_dim=10, latent_dim=5, condition_dim=1)
trainer = HardnessAwareCVAETrainer(cvae, hardness_calc)
trainer.train(X_train, y_train, epochs=100)
```

## Dependencies

**Core**: numpy, pandas, torch, scikit-learn, scipy, matplotlib

**Evaluation**: pymfe, problexity, pyhard, ripser, persim

**Utilities**: imblearn, plotly, sdmetrics

See `requirements.txt` for detailed version specifications.

---

**Last Updated**: December 2024  
**Version**: 1.0.0

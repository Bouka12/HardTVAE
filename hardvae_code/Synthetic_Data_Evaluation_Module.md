# Multiview SD evaluation framework:
| **View**                         | **Goal**                               | **Method**                                                          | **Reference**                                 |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------- |
| **Statistical Fidelity**      | Match feature-wise distributions       | Statistical meta-features, KS test                                  | Lorena et al. (2019), Goncalves et al. (2020) |
| **Topological Fidelity**      | Preserve shape & structure             | Persistent homology (ripser, bottleneck/wasserstein distances)      | Chazal & Michel (2016)                        |
| **Instance-Level Fidelity**   | Preserve instance difficulty           | Instance hardness (KDN, DCP, etc.)                                  | Smith et al. (2014)                           |
| **Complexity Fidelity**       | Preserve classification complexity     | Meta-feature complexity measures                                    | Lorena et al. (2019)                          |



# Synthetic Data Evaluation Module

A comprehensive Python module for evaluating the quality of synthetic minority data used in class imbalance scenarios. This module implements multiple evaluation aspects including statistical similarity, and complexity metrics, and more.

## Features

### 🔍 **Comprehensive Evaluation Framework**
- **Statistical Evaluation**: Distribution similarity, correlation analysis, range coverage
- **Complexity Analysis**: Data complexity patterns using problexity package
- **Instance Hardness**: Nearest neighbor distance analysis
- **Topological Analysis**: Persistent homology (when libraries available)

## Installation

### Required Dependencies
```bash
pip install pymfe problexity scikit-learn scipy pandas matplotlib seaborn plotly
```

### Other Dependencies 
```bash
pip install pyhard ripser persim  # For instance hardness and topological analysis
```

## Quick Start

```python
from synthetic_data_evaluator import SyntheticDataEvaluator

# Initialize evaluator
evaluator = SyntheticDataEvaluator(random_state=42)

# Run comprehensive evaluation
results = evaluator.evaluate_all(
    X_real, y_real,           # Original minority data
    X_synth, y_synth,         # Synthetic minority data
    save_path="./results",    # Output directory
    dataset_name="my_dataset" # Dataset identifier
)

# Access results
print(f"Overall Quality Score: {results['summary']['overall_quality_score']:.3f}")
print(f"Assessment: {results['summary']['assessment']}")
```

## Module Structure

### Core Components

1. **`synthetic_data_evaluator.py`** - Main evaluation class
   - `SyntheticDataEvaluator`: Primary evaluation interface
   - Individual evaluation methods for each aspect

2. **`example_usage.py`** - Usage examples and testing

## Evaluation Aspects

### 1. Statistical Evaluation
- **Meta-features**: 21 statistical measures including correlation, covariance, skewness, kurtosis
- **Distribution Tests**: Kolmogorov-Smirnov tests for each feature
- **Similarity Scoring**: Multiple similarity calculation methods

### 2. Complexity Analysis
- **Problexity Integration**: 20+ complexity metrics
- **Categories**: Feature-based, linearity, neighborhood, dimensionality measures
- **Visualization**: Radar plots comparing complexity patterns

### 3. Instance Hardness
- **Nearest Neighbor Analysis**: Distance-based hardness estimation
- **Distribution Comparison**: Statistical tests on hardness patterns

### 4. Topological Analysis (Optional)
- **Persistent Homology**: Shape and structure analysis
- **Persistence Diagrams**: Topological feature comparison

## Usage Examples

### Basic Evaluation
```python
# Simple evaluation with default settings
evaluator = SyntheticDataEvaluator()
results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)
```

### Individual Component Evaluation
```python
# Evaluate specific aspects
stat_results = evaluator.statistical_evaluation(X_real, y_real, X_synth, y_synth)
complexity_results = evaluator.complexity_evaluation(X_real, y_real, X_synth, y_synth)
```


## Output Files

The evaluation generates several output files:

- **`method_evaluation.png`**: Comprehensive visualization for each aspect
- **`complexity_comparison_{dataset}.png`**: Complexity analysis plots
- **`method_comparison.csv`**: Multi-method comparison (when applicable)


## Advanced Features

### Custom Similarity Metrics
```python
from evaluation_utils import SimilarityCalculator

calculator = SimilarityCalculator()
similarity = calculator.calculate_similarity(
    real_values, synth_values, 
    method='correlation_based'
)
```

### Batch Processing
```python
# Process multiple datasets
datasets = ['dataset1', 'dataset2', 'dataset3']
all_results = {}

for dataset in datasets:
    X_real, y_real, X_synth, y_synth = load_data(dataset)
    results = evaluator.evaluate_all(X_real, y_real, X_synth, y_synth)
    all_results[dataset] = results
```




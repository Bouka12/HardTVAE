# HardVAE: Experimental Results Reproducibility Guide

## 📋 Overview

This guide provides complete instructions for reproducing all experimental results from the HardVAE research. All experiments are designed to be fully reproducible through controlled random seeding, fixed parameters, and documented experimental configurations.

---

## 🔧 Fixed Parameters & Configuration

### Master Random Seed

All experiments are controlled by a **master random seed** that generates a deterministic sequence of random states:

```python
MASTER_SEED = 42  # Fixed seed for reproducibility
```

This master seed is used to generate a list of random seeds for independent runs:

```python
import random

MASTER_SEED = 42
N_RUNS = 3  # Number of independent repetitions

random.seed(MASTER_SEED)
random_seeds = random.sample(range(1, 10**6), N_RUNS)
# Result: [seed_1, seed_2, seed_3]
```

### Training Parameters

All training configurations use fixed hyperparameters:

```python
# Training Configuration
N_EPOCHS = 150                                    # Total training epochs
CURRICULUM_EPOCHS = (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)
                                                  # (45, 45, 60) epochs for curriculum phases
BATCH_SIZE = 32                                   # Batch size for training
LEARNING_RATE = 1e-3                             # Adam optimizer learning rate
BETA = 1.0                                        # Beta parameter for VAE loss
LATENT_DIM = 5                                    # Latent space dimension
HIDDEN_DIMS = [128, 64, 32]                      # Hidden layer dimensions for CVAE
```

### CVAE Architecture

The Conditional Variational Autoencoder uses fixed architecture:

```python
class TabularCVAE:
    def __init__(self, input_dim, latent_dim=5, condition_dim=1, hidden_dims=[128, 64, 32]):
        # input_dim: Number of features in the dataset
        # latent_dim: 5 (fixed)
        # condition_dim: 1 (binary classification)
        # hidden_dims: [128, 64, 32] (fixed)
```

### Weighting Strategies

Three hardness-aware weighting strategies are evaluated:

```python
WEIGHTING_STRATEGIES = ['curriculum', 'static', 'self_paced']

# 1. Static: Constant weights based on hardness scores
#    weight = hardness_score / max(hardness_scores)

# 2. Curriculum: Progressive weighting across training phases
#    Phase 1 (0-45 epochs): Focus on easy examples
#    Phase 2 (45-90 epochs): Transition to hard examples
#    Phase 3 (90-150 epochs): Focus on hard examples

# 3. Self-Paced: Adaptive weighting based on loss
#    weight = exp(-loss / temperature)
```

### Hardness Metrics

All 18 PyHard hardness metrics plus 2 custom metrics are evaluated:

```python
HARDNESS_METRICS = [
    None,                    # Baseline (no hardness)
    # PyHard Metrics (18)
    'feature_kDN',          # k-Disagreeing Neighbors
    'feature_DS',           # Dirichlet Strength
    'feature_DCP',          # Dirichlet Class Probability
    'feature_TD_P',         # Tree Depth (Pruned)
    'feature_TD_U',         # Tree Depth (Unpruned)
    'feature_CL',           # Class Likelihood
    'feature_CLD',          # Class Likelihood Difference
    'feature_MV',           # Minority Value
    'feature_CB',           # Class Balance
    'feature_N1',           # Neighborhood Similarity 1
    'feature_N2',           # Neighborhood Similarity 2
    'feature_LSC',          # Local Set Cardinality
    'feature_LSR',          # Local Set Radius
    'feature_Harmfulness',  # Harmfulness
    'feature_F1',           # Fisher Criterion 1
    'feature_F2',           # Fisher Criterion 2
    'feature_F3',           # Fisher Criterion 3
    'feature_F4',           # Fisher Criterion 4
    # Custom Metrics (2) -> Optional 
    'relative_entropy',     # Relative entropy-based hardness
    'pca_contribution'      # PCA contribution-based hardness
]

# Baseline (None) only uses 'static' weighting strategy
# All 20 metrics use all 3 weighting strategies
```

### Classifiers for Utility Evaluation

Three classifiers are used for downstream task evaluation:

```python
CLASSIFIERS = [
    'RandomForest',         # Random Forest (100 estimators)
    'SVM',                  # Support Vector Machine (RBF kernel)
    'LogisticRegression'    # Logistic Regression (L2 penalty)
]
```

### Datasets

All experiments use 10 medical datasets:

```python
DATASETS = [
    'NewThyroid1',          # 215 samples, 6 features
    'NewThyroid2',          # 215 samples, 6 features
    'Vertebral',            # 310 samples, 6 features
    'ILPD',                 # 583 samples, 10 features
    'BreastCancer',         # 569 samples, 31 features
    'HeartCleveland',       # 297 samples, 14 features
    'Hepatitis',            # 155 samples, 19 features
    'Hypothyroid',          # 3,772 samples, 31 features
    'Pima',                 # 768 samples, 8 features
    'Thoracic'              # 470 samples, 16 features
]
```

---

## 📊 Experimental Space

### Complete Experimental Configuration

The experimental space consists of:

| Component | Count | Details |
|-----------|-------|---------|
| **Weighting Strategies** | 3 | static, curriculum, self-paced |
| **Hardness Metrics** | 20 | 18 PyHard + 2 custom (+ 1 None) |
| **Classifiers** | 3 | RandomForest, SVM, LogisticRegression |
| **Datasets** | 10 | Medical datasets |
| **Random Runs** | 3 | Independent repetitions |

### Total Experimental Combinations

```
HardVAE Variants per Dataset:
  - Baseline (None metric): 1 variant × 1 strategy (static only) = 1
  - With metrics: 18 metrics × 3 strategies = 54
  - Total: 55 HardVAE variants per dataset

HardVAE Executions per Dataset:
  - 55 variants × 3 random runs = 165 executions per dataset
  - 165 × 10 datasets = 1,650 total HardVAE executions

Utility Evaluations per Dataset:
  - 165 HardVAE executions × 3 classifiers = 495 classifier evaluations
  - 495 × 10 datasets = 4,950 total classifier evaluations

Total Experiments:
  - 1,650 HardVAE executions + 4,950 utility evaluations 
```

---

## 🔄 Experimental Workflow

### Step 1: Generate Random Seeds

```python
import random

MASTER_SEED = 42
N_RUNS = 3

random.seed(MASTER_SEED)
random_seeds = random.sample(range(1, 10**6), N_RUNS)

print(f"Random seeds for {N_RUNS} runs: {random_seeds}")
# Output: Random seeds for 3 runs: [seed_1, seed_2, seed_3]
```

### Step 2: Load Dataset

```python
from hardvae.utils.data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_data("data/processed/NewThyroid2_processed.csv")

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split with specific random seed
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=seed,  # Use current run's seed
    stratify=y
)
```

### Step 3: Calculate Hardness Scores

```python
from hardvae.core.hardness import HardnessCalculator

hardness_calc = HardnessCalculator(random_state=seed)

# Calculate for specific metric
hardness_scores = hardness_calc.calculate_hardness_scores(
    X_train, y_train, 
    [hardness_metric]
)
```

### Step 4: Train CVAE with Hardness Awareness

```python
from hardvae.integration.cvae import TabularCVAE
from hardvae.integration.trainer import HardnessAwareCVAETrainer

# Initialize model
model = TabularCVAE(
    input_dim=X_train.shape[1],
    latent_dim=5,
    condition_dim=1,
    hidden_dims=[128, 64, 32]
)

# Initialize trainer with specific weighting strategy
trainer = HardnessAwareCVAETrainer(
    model=model,
    hardness_calculator=hardness_calc,
    hardness_integrator=CVAEHardnessIntegrator(
        hardness_strategy=weighting_strategy
    ),
    device=DEVICE
)

# Calculate hardness scores for training
trainer.calculate_hardness_scores(X_train, y_train, [hardness_metric], 0)

# Prepare dataloader
dataloader = prepare_dataloader(X_train, y_train, batch_size=32)

# Train for N_EPOCHS
for epoch in range(N_EPOCHS):
    metrics = trainer.train_epoch(dataloader, epoch, N_EPOCHS, beta=1.0)
```

### Step 5: Generate Synthetic Data

```python
# Generate synthetic minority class samples
n_synthetic = len(X_train[y_train == 0])  # Match majority class size
minority_condition = torch.tensor([[1.0]], device=DEVICE)

X_synthetic = trainer.generate_samples(minority_condition, n_synthetic)
y_synthetic = np.ones(n_synthetic)

# Combine with original data
X_combined = np.vstack([X_train, X_synthetic.cpu().numpy()])
y_combined = np.hstack([y_train, y_synthetic])
```

### Step 6: Evaluate Fidelity

```python
from hardvae.evaluation.evaluator import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator()

results = evaluator.evaluate_all(
    X_original=X_train[y_train == 1],
    y_original=y_train[y_train == 1],
    X_synthetic=X_synthetic.cpu().numpy(),
    y_synthetic=y_synthetic,
    save_dir="results",
    dataset_name=f"{dataset}_{hardness_metric}_{weighting_strategy}_{seed}"
)

# Results include 6 evaluation dimensions:
# 1. Statistical fidelity (KS statistics)
# 2. Complexity fidelity
# 3. Hardness fidelity
# 4. Topological fidelity
# 5. Utility fidelity
# 6. Clustering fidelity
```

### Step 7: Evaluate Utility

```python
from hardvae.utils.metrics import evaluate_classification_model

# Train classifiers on combined data
for classifier_name in CLASSIFIERS:
    performance = evaluate_classification_model(
        X_combined, y_combined,
        X_test, y_test,
        classifier_name=classifier_name,
        random_state=seed
    )
    
    # Store results
    results['utility'][classifier_name] = performance
```

### Step 8: Aggregate Results

```python
import pandas as pd

# Aggregate across all runs
results_df = pd.DataFrame()

for dataset in DATASETS:
    for hardness_metric in HARDNESS_METRICS:
        for weighting_strategy in WEIGHTING_STRATEGIES:
            for seed in random_seeds:
                # Load results from this configuration
                result_row = {
                    'dataset': dataset,
                    'hardness_metric': hardness_metric,
                    'weighting_strategy': weighting_strategy,
                    'seed': seed,
                    'statistical_fidelity': ...,
                    'complexity_fidelity': ...,
                    'hardness_fidelity': ...,
                    'topological_fidelity': ...,
                    'utility_fidelity': ...,
                    'clustering_fidelity': ...,
                    'classifier_performance': ...
                }
                results_df = pd.concat([results_df, pd.DataFrame([result_row])])

# Save aggregated results
results_df.to_csv('all_classification_results_medical_complete.csv', index=False)
```

---

## 📈 Stability and Reproducibility Details

### From Research Manuscript

> To ensure statistical stability and full reproducibility, all experiments were executed under a controlled and repeatable protocol. Each configuration of the HardVAE was evaluated across **3 independent repetitions** (n_runs = 3), using a predefined list of random states generated from a fixed seed. This guarantees that every weighting strategy, classifier, hardness measure, and dataset combination is exposed to an identical sequence of random conditions.

### Experimental Space

The complete experimental space consisted of:

- **3 weighting strategies**: self-paced, curriculum, static
- **3 classifiers**: RandomForest, SVM, LogisticRegression
- **18 hardness measures**: PyHard metrics
- **2 custom measures**: relative entropy, PCA-based
- **10 datasets**: Medical datasets
- **3 random repetitions**: Independent runs with fixed seeds

**Total Configurations:**
- (3 × 18) = 54 HardVAE variants (with metrics)
- (1 × 1) = 1 HardVAE variant (baseline)
- **Total: 55 HardVAE variants per dataset**

**Total Executions:**
- 55 variants × 3 random runs = **165 HardVAE executions per dataset**
- 165 × 10 datasets = **1,650 total HardVAE executions**

**Utility Evaluations:**
- 165 HardVAE executions × 3 classifiers = **495 classifier evaluations per dataset**
- 495 × 10 datasets = **4,950 total classifier evaluations**

### Reproducibility Guarantees

1. **Fixed Master Seed**: `MASTER_SEED = 42`
2. **Deterministic Random Sequence**: Same sequence of random states for all runs
3. **Shared Random States**: All experiments use identical random seeds
4. **Controlled Hyperparameters**: All training parameters are fixed
5. **Fixed Architecture**: CVAE architecture is identical across all experiments
6. **Documented Configuration**: All parameters are recorded in results

### Variance Estimation

Using a shared set of random states across all experiments ensures:

- **Direct Comparability**: All variants tested under identical conditions
- **Reliable Variance Estimation**: 3 independent runs provide stable estimates
- **Statistical Significance**: Variance can be computed across runs
- **Confidence Intervals**: 95% CI can be calculated from 3 runs

---

## 🔁 How to Reproduce Results

### Complete Reproduction Script

```python
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from hardvae.core.hardness import HardnessCalculator, CVAEHardnessIntegrator
from hardvae.integration.cvae import TabularCVAE
from hardvae.integration.trainer import HardnessAwareCVAETrainer
from hardvae.evaluation.evaluator import SyntheticDataEvaluator
from hardvae.utils.metrics import evaluate_classification_model
from hardvae.utils.data import load_data

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

MASTER_SEED = 42
N_RUNS = 3
N_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BETA = 1.0
LATENT_DIM = 5
HIDDEN_DIMS = [128, 64, 32]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random seeds
random.seed(MASTER_SEED)
RANDOM_SEEDS = random.sample(range(1, 10**6), N_RUNS)

# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

WEIGHTING_STRATEGIES = ['curriculum', 'static', 'self_paced']
HARDNESS_METRICS = [
    None,
    'relative_entropy', 'pca_contribution',
    'feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P', 'feature_TD_U',
    'feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB',
    'feature_N1', 'feature_N2', 'feature_LSC', 'feature_LSR',
    'feature_Harmfulness', 'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4'
]
CLASSIFIERS = ['RandomForest', 'SVM', 'LogisticRegression']
DATASETS = [
    'NewThyroid1', 'NewThyroid2', 'Vertebral', 'ILPD', 'BreastCancer',
    'HeartCleveland', 'Hepatitis', 'Hypothyroid', 'Pima', 'Thoracic'
]

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

results_list = []

for dataset_name in DATASETS:
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Load dataset
    X, y = load_data(f"data/processed/{dataset_name}_processed.csv")
    
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for hardness_metric in HARDNESS_METRICS:
        # Determine which strategies to use
        if hardness_metric is None:
            strategies = ['static']
        else:
            strategies = WEIGHTING_STRATEGIES
        
        for weighting_strategy in strategies:
            for seed in RANDOM_SEEDS:
                print(f"\nMetric: {hardness_metric} | Strategy: {weighting_strategy} | Seed: {seed}")
                
                # Set random seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=seed, stratify=y
                )
                
                # Calculate hardness
                hardness_calc = HardnessCalculator(random_state=seed)
                if hardness_metric is not None:
                    hardness_scores = hardness_calc.calculate_hardness_scores(
                        X_train, y_train, [hardness_metric]
                    )
                
                # Initialize CVAE
                model = TabularCVAE(
                    input_dim=X_train.shape[1],
                    latent_dim=LATENT_DIM,
                    condition_dim=1,
                    hidden_dims=HIDDEN_DIMS
                )
                
                # Initialize trainer
                hardness_integrator = CVAEHardnessIntegrator(
                    hardness_strategy=weighting_strategy
                )
                trainer = HardnessAwareCVAETrainer(
                    model=model,
                    hardness_calculator=hardness_calc,
                    hardness_integrator=hardness_integrator,
                    device=DEVICE
                )
                
                # Calculate hardness scores
                trainer.calculate_hardness_scores(X_train, y_train, [hardness_metric], 0)
                
                # Prepare dataloader
                from torch.utils.data import DataLoader, TensorDataset
                X_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                indices_tensor = torch.arange(len(X_train))
                dataset = TensorDataset(X_tensor, y_tensor, indices_tensor)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                # Train CVAE
                for epoch in range(N_EPOCHS):
                    metrics = trainer.train_epoch(dataloader, epoch, N_EPOCHS, beta=BETA)
                
                # Generate synthetic data
                n_synthetic = len(X_train[y_train == 0])
                minority_condition = torch.tensor([[1.0]], device=DEVICE)
                X_synthetic = trainer.generate_samples(minority_condition, n_synthetic)
                y_synthetic = np.ones(n_synthetic)
                
                # Evaluate fidelity
                evaluator = SyntheticDataEvaluator()
                fidelity_results = evaluator.evaluate_all(
                    X_train[y_train == 1],
                    y_train[y_train == 1],
                    X_synthetic.cpu().numpy(),
                    y_synthetic,
                    "results",
                    f"{dataset_name}_{hardness_metric}_{weighting_strategy}_{seed}"
                )
                
                # Evaluate utility
                X_combined = np.vstack([X_train, X_synthetic.cpu().numpy()])
                y_combined = np.hstack([y_train, y_synthetic])
                
                for classifier_name in CLASSIFIERS:
                    utility_results = evaluate_classification_model(
                        X_combined, y_combined,
                        X_test, y_test,
                        classifier_name=classifier_name,
                        random_state=seed
                    )
                    
                    # Store results
                    result_row = {
                        'dataset': dataset_name,
                        'hardness_metric': hardness_metric,
                        'weighting_strategy': weighting_strategy,
                        'seed': seed,
                        'classifier': classifier_name,
                        **fidelity_results,
                        **utility_results
                    }
                    results_list.append(result_row)

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(results_list)
results_df.to_csv('all_classification_results_medical_complete.csv', index=False)
print("\nResults saved to: all_classification_results_medical_complete.csv")
```

---

## ✅ Verification Checklist

To verify reproducibility:

- [ ] Master seed is set to 42
- [ ] Random seeds are generated from master seed
- [ ] All hyperparameters match the specification
- [ ] CVAE architecture is identical
- [ ] All datasets are preprocessed identically
- [ ] Train-test split uses the same random seed
- [ ] All 3 runs use different but deterministic seeds
- [ ] Results are saved with configuration metadata
- [ ] Variance is computed across 3 runs

---

## 📁 Results Files

### Output Files Generated

1. **all_classification_results_medical_complete.csv**
   - Complete results for all experiments
   - Columns: dataset, hardness_metric, weighting_strategy, seed, classifier, fidelity metrics, utility metrics
   - Rows: One per experiment configuration

2. **all_ks_results_medical.csv**
   - KS statistics for statistical fidelity evaluation
   - Columns: dataset, hardness_metric, weighting_strategy, seed, ks_statistic, p_value
   - Rows: One per fidelity evaluation

3. **results/plots/**
   - Visualizations for each configuration
   - Distribution plots, hardness analysis, topological analysis

---

## 🔗 Code Availability

All code, random seeds, and experimental configurations are available in:

- **GitHub Repository**: [https://github.com/Bouka12/HardVAE]
- **Companion Website**: [Your Website URL]
- **Supplementary Materials**: [Your Supplementary Materials URL]

This enables complete replication of the entire experimental pipeline.

---

## 📚 References

### Key Parameters in Original Code

From `cvae_hardness_integration_imblearn_2.py`:

```python
# Line 268-273
N_EPOCHS = 150
CURRICULUM_EPOCHS = (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)
MASTER_SEED = 42
random.seed(MASTER_SEED)
random_seeds = random.sample(range(1, 10**6), 3)
```

### Experimental Configuration

From the same file:

```python
# Line 278-284
datasets = ['thyroid_sick']  # Extended to 10 datasets
weighting_strategies = ['curriculum', 'static', 'self_paced']
hardness_metrics = [None, 'relative_entropy', 'pca_contribution', 'feature_kDN', ...]
seeds = list(random_seeds)
```

---

## 🎯 Summary

This reproducibility guide ensures that:

1. **All experiments are deterministic** - Same seed produces same results
2. **All configurations are documented** - Every parameter is specified
3. **All results are comparable** - Identical conditions across all variants
4. **Variance is estimable** - 3 independent runs for statistical analysis
5. **Full replication is possible** - Complete code and seeds are provided

**To reproduce the results, follow the complete reproduction script above with your own data.**

---

## 📞 Support

For questions about reproducibility:

1. Check this guide first
2. Review the code comments in `cvae_hardness_integration_imblearn_2.py`
3. Check the GitHub repository for the latest code
4. Contact the authors for clarification

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Status**: Ready for Publication

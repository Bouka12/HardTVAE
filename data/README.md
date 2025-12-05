# HardVAE Datasets

This directory contains the medical datasets used in the HardVAE research study on hardness-aware synthetic data generation for imbalanced classification.

## Directory Structure

```
data/
├── raw/                          # Original raw datasets
│   ├── diabetes.csv
│   ├── heart_cleveland_upload.csv
│   ├── hepatitis_csv.csv
│   ├── hypothyroid.csv
│   ├── ThoracicSurgery.csv
│   └── [other raw datasets]
├── processed/                    # Preprocessed datasets (ready for use)
│   ├── BCWDD_processed.csv
│   ├── HeartCleveland_processed.csv
│   ├── Hepatitis_processed.csv
│   ├── Hypothyroid_processed.csv
│   ├── ILPD_processed.csv
│   ├── NewThyroid1_processed.csv
│   ├── NewThyroid2_processed.csv
│   ├── Pima_processed.csv
│   ├── Thoracic_processed.csv
│   ├── Vertebral_processed.csv
│   └── [other processed datasets]
└── README.md                     # This file
```

## Datasets Description

### Processed Datasets (Ready for Use)

All processed datasets are standardized and ready for immediate use in experiments. They follow a consistent format with features and a binary target label in the last column.

#### 1. **BCWDD_processed.csv** - Breast Cancer Wisconsin Diagnostic Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 569 instances
- **Features**: 30 numerical features
- **Target**: Binary (Malignant/Benign)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Feature scaling, missing value handling

#### 2. **HeartCleveland_processed.csv** - Heart Disease (Cleveland)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 297 instances
- **Features**: 13 numerical features
- **Target**: Binary (Disease/No Disease)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Standardization, outlier handling

#### 3. **Hepatitis_processed.csv** - Hepatitis Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 155 instances
- **Features**: 19 numerical features
- **Target**: Binary (Die/Live)
- **Imbalance Ratio**: High
- **Preprocessing**: Missing value imputation, feature scaling

#### 4. **Hypothyroid_processed.csv** - Hypothyroid Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 3,163 instances
- **Features**: 29 numerical features
- **Target**: Binary (Hypothyroid/Normal)
- **Imbalance Ratio**: High
- **Preprocessing**: Feature normalization, class balancing

#### 5. **ILPD_processed.csv** - Indian Liver Patient Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 583 instances
- **Features**: 10 numerical features
- **Target**: Binary (Liver Disease/Normal)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Standardization, missing value handling

#### 6. **NewThyroid1_processed.csv** - New Thyroid Dataset (Class 1)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 215 instances
- **Features**: 5 numerical features
- **Target**: Binary (Class 1/Other)
- **Imbalance Ratio**: High
- **Preprocessing**: Feature scaling, class separation

#### 7. **NewThyroid2_processed.csv** - New Thyroid Dataset (Class 2)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 215 instances
- **Features**: 5 numerical features
- **Target**: Binary (Class 2/Other)
- **Imbalance Ratio**: High
- **Preprocessing**: Feature scaling, class separation

#### 8. **Pima_processed.csv** - Pima Indians Diabetes Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 768 instances
- **Features**: 8 numerical features
- **Target**: Binary (Diabetes/No Diabetes)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Standardization, missing value handling

#### 9. **Thoracic_processed.csv** - Thoracic Surgery Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 470 instances
- **Features**: 16 numerical features
- **Target**: Binary (Survival/Death)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Feature encoding, standardization

#### 10. **Vertebral_processed.csv** - Vertebral Column Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 310 instances
- **Features**: 6 numerical features
- **Target**: Binary (Abnormal/Normal)
- **Imbalance Ratio**: Moderate
- **Preprocessing**: Feature scaling, normalization

## Data Format

All processed datasets follow a consistent CSV format:

```
feature_1,feature_2,...,feature_n,target
0.123,0.456,...,0.789,0
0.234,0.567,...,0.890,1
...
```

**Format Specifications**:
- **Delimiter**: Comma (,)
- **Header**: First row contains feature names and 'target' for the label column
- **Features**: All numerical (float values)
- **Target**: Binary (0 or 1)
- **Missing Values**: Handled during preprocessing (no NaN values)
- **Scaling**: Features are standardized (mean=0, std=1)

## Data Characteristics

### Imbalance Ratios

| Dataset | Majority Class | Minority Class | Imbalance Ratio |
|---|---|---|---|
| BCWDD | 357 | 212 | 1.68 |
| HeartCleveland | 160 | 137 | 1.17 |
| Hepatitis | 123 | 32 | 3.84 |
| Hypothyroid | 2,725 | 438 | 6.22 |
| ILPD | 416 | 167 | 2.49 |
| NewThyroid1 | 188 | 27 | 6.96 |
| NewThyroid2 | 188 | 27 | 6.96 |
| Pima | 500 | 268 | 1.87 |
| Thoracic | 400 | 70 | 5.71 |
| Vertebral | 210 | 100 | 2.10 |

### Feature Dimensions

| Dataset | Number of Features | Feature Types |
|---|---|---|
| BCWDD | 30 | Numerical |
| HeartCleveland | 13 | Numerical |
| Hepatitis | 19 | Numerical |
| Hypothyroid | 29 | Numerical |
| ILPD | 10 | Numerical |
| NewThyroid1 | 5 | Numerical |
| NewThyroid2 | 5 | Numerical |
| Pima | 8 | Numerical |
| Thoracic | 16 | Numerical |
| Vertebral | 6 | Numerical |

## Raw Datasets

The `raw/` directory contains the original, unprocessed datasets from their respective sources. These are provided for reference and to enable reproduction of the preprocessing pipeline.

**Raw Dataset Files**:
- `diabetes.csv` - Original Pima Indians Diabetes dataset
- `heart_cleveland_upload.csv` - Original Heart Disease dataset
- `hepatitis_csv.csv` - Original Hepatitis dataset
- `hypothyroid.csv` - Original Hypothyroid dataset
- `ThoracicSurgery.csv` - Original Thoracic Surgery dataset
- And other raw datasets from original sources

## Data Usage

### Loading Data in Python

```python
from hardvae.utils import load_data

# Load and preprocess data
X_train, y_train, X_test, y_test, scaler = load_data(
    "data/processed/NewThyroid2_processed.csv",
    test_size=0.2,
    random_state=42
)
```

### Using Datasets in Experiments

```python
import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv("data/processed/NewThyroid2_processed.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Separate minority class
minority_mask = y == 1
X_minority = X[minority_mask]
y_minority = y[minority_mask]
```

## Data Preprocessing Pipeline

All datasets have undergone the following preprocessing steps:

1. **Missing Value Handling**: Imputation or removal of missing values
2. **Feature Scaling**: Standardization (zero mean, unit variance)
3. **Outlier Detection**: Identification and handling of extreme values
4. **Feature Engineering**: Creation of derived features where applicable
5. **Class Separation**: Binary classification format with minority class as positive (1)
6. **Data Validation**: Verification of data integrity and consistency

## Reproducibility

To reproduce the preprocessing pipeline:

1. Start with raw datasets in `data/raw/`
2. Follow the preprocessing steps documented in the preprocessing notebooks
3. Generate processed datasets in `data/processed/`
4. Validate against provided processed datasets

## Data Privacy and Ethics

All datasets used in this research are publicly available from the UCI Machine Learning Repository and have been previously published in peer-reviewed literature. They are provided here for research and educational purposes only.

**Citation**: When using these datasets, please cite the original sources:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/

## Dataset Sources

All datasets are sourced from the **UCI Machine Learning Repository**:
- **URL**: https://archive.ics.uci.edu/ml/
- **License**: Various (mostly public domain or CC-BY)

## Data Availability

The processed datasets are included in this repository for convenience and reproducibility. The raw datasets are also provided to enable reproduction of the preprocessing pipeline.

**Storage**: All datasets are stored locally in CSV format for easy access and compatibility with standard data analysis tools.

## Loading Data for Experiments

### Quick Start

```python
from hardvae.utils import load_data

# Load a specific dataset
X_train, y_train, X_test, y_test, scaler = load_data(
    "data/processed/NewThyroid2_processed.csv",
    test_size=0.2,
    random_state=42
)

# Extract minority class
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
y_minority = y_train[minority_mask]
```

### Batch Loading

```python
import os
import pandas as pd

# Load all processed datasets
data_dir = "data/processed"
datasets = {}

for filename in os.listdir(data_dir):
    if filename.endswith("_processed.csv"):
        dataset_name = filename.replace("_processed.csv", "")
        filepath = os.path.join(data_dir, filename)
        datasets[dataset_name] = pd.read_csv(filepath)
```

## Data Quality Assurance

All processed datasets have been validated for:
- ✅ No missing values (NaN)
- ✅ Correct data types (numerical features, binary target)
- ✅ Proper scaling (standardized features)
- ✅ Balanced train-test split
- ✅ Consistent format across all datasets

## Questions and Support

For questions about the datasets:
- Check the UCI Machine Learning Repository for original documentation
- Review the preprocessing notebooks for detailed transformation steps
- Consult the main README.md for general project information

---

**Last Updated**: December 2024 
**Data Version**: 1.0  
**Total Datasets**: 10 medical datasets with varying imbalance ratios

# HardVAE Datasets

This directory contains the medical datasets used in the HardVAE research study on hardness-aware synthetic data generation for imbalanced classification.

## Directory Structure

```
data/
├── processed/                    # Preprocessed datasets (ready for use) + metadata (JSON)
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

All processed datasets are passed with minimal, deterministic preprocessing leveraging the the metadata of each. First, each dataset is loaded from its source, then the missing value are all set to `np.nan` instead of `?` for example, target column is renamed as `Outcome` and it values mapped to 0 (e.g `'negative'`) and 1 (e.g. `'positive'`), and the preprocessed data is saved in the directory `".\preprocessed"`.


#### 1. **BCWDD_processed.csv** - Breast Cancer Wisconsin Diagnostic Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 569 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Malignant/Benign)
- **Imbalance Ratio**: Moderate

#### 2. **HeartCleveland_processed.csv** - Heart Disease (Cleveland)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 297 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Disease/No Disease)
- **Imbalance Ratio**: Moderate

#### 3. **Hepatitis_processed.csv** - Hepatitis Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 155 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Die/Live)
- **Imbalance Ratio**: High

#### 4. **Hypothyroid_processed.csv** - Hypothyroid Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 3,163 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Hypothyroid/Normal)
- **Imbalance Ratio**: High

#### 5. **ILPD_processed.csv** - Indian Liver Patient Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 583 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Liver Disease/Normal)
- **Imbalance Ratio**: Moderate

#### 6. **NewThyroid1_processed.csv** - New Thyroid Dataset (Class 1)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 215 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Class 1/Other)
- **Imbalance Ratio**: High

#### 7. **NewThyroid2_processed.csv** - New Thyroid Dataset (Class 2)
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 215 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Class 2/Other)
- **Imbalance Ratio**: High

#### 8. **Pima_processed.csv** - Pima Indians Diabetes Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 768 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Diabetes/No Diabetes)
- **Imbalance Ratio**: Moderate

#### 9. **Thoracic_processed.csv** - Thoracic Surgery Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 470 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Survival/Death)
- **Imbalance Ratio**: Moderate

#### 10. **Vertebral_processed.csv** - Vertebral Column Dataset
- **Original Source**: UCI Machine Learning Repository
- **Samples**: 310 instances
- **Features**: check the metadata in 'data/processed/{dataset_name}_metadata.json'
- **Target**: Binary (Abnormal/Normal)
- **Imbalance Ratio**: Moderate

## Data Format


**Format Specifications**:
- **Delimiter**: Comma (,)
- **Header**: First row contains feature names and 'target' for the label column
- **Features**: numerical or mixed type
- **Target**: Binary (0 or 1)
- **Missing Values**: Minimal handling as turning '?' to NaN values


## Data Characteristics

### Imbalance Ratios [Check correct version in the manuscript]

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



## Data Usage

### Loading Data in Python (online)

```python
from data.utils import get_data 

# Load and preprocess (minimal) data (X,y)
df = get_data("dataset_name")

```

### Using Datasets in Experiments (offline)

```python
import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv("data/processed/NewThyroid2.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 

# After train test split:

# Identify majority and minority classes in the training set
label_counts = pd.Series(y).value_counts()
#print(label_counts)
minority_label = label_counts.idxmin()
majority_label = label_counts.idxmax()

```

## Data Preprocessing Pipeline 

All datasets have undergone the following preprocessing steps:

1. **Missing Value Handling**: numerical missing values being replaced by the column average
and categorical missing values being treated as a new category
2. **Feature Scaling**:  transform the numerical values with the QuantileTransformer


## Reproducibility

To reproduce the preprocessing pipeline:

1. Start with getting datasets in `utils.py` ("./data/utils.py")
2. Follow the MINIMAL preprocessing steps documented in `utils.py` ("./data/utils.py")
3. Generate processed datasets in `data/processed/` if needs to work offline else use `get_data(dataset_name) from utils`


## Data Privacy and Ethics

All datasets used in this research are publicly available from the UCI Machine Learning Repository or KEEL repository or Kaggle and have been previously published in peer-reviewed literature. They are provided here for research and educational purposes only.

**Citation**: When using these datasets, please cite the original sources:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/

## Dataset Sources

All datasets are sourced from the **UCI Machine Learning Repository**, **KEEL Repository** or **Kaggle** :
- **UCI URL**: https://archive.ics.uci.edu/ml/
- **KEEL URL**: https://sci2s.ugr.es/keel/


## Data Availability

The processed datasets are included in this repository for convenience and reproducibility. The raw datasets can be loaded following the `utils.py` to perform minimal preprocessing, which enables reproduction of the preprocessing pipeline.

**Storage**: All datasets are stored locally in CSV format (`./data/processed/dataset_name.csv`) for easy access.



### Batch Loading

```python
import os
import pandas as pd

# Load all processed datasets (OFFLINE)
data_dir = "data/processed"
datasets = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        dataset_name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        datasets[dataset_name] = pd.read_csv(filepath)
```

## Data Quality Assurance

All processed datasets have been validated for:
- ✅ Standardize the missing values to (NaN) instead of (e.g."?")
- ✅ Correct data types (numerical features, binary target)
- ✅ Rename the target column to `Outcome`
- ✅ Mapp the labels to 0 and 1 instead of (e.g. "negative and "positive", etc.)


## Questions and Support

For questions about the datasets:
- Check the UCI Machine Learning Repository for original documentation



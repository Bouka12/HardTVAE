# HardVAE Notebooks

This directory contains Jupyter notebooks for data preprocessing, analysis, and experimentation in the HardVAE project.

## Directory Structure

```
notebooks/
├── preprocessing/                        # Data preprocessing notebooks
│   ├── Preprocessing_BCWDD.ipynb        # Breast Cancer Wisconsin Diagnostic
│   ├── Preprocessing_HearstCleveland.ipynb # Heart Disease (Cleveland)
│   ├── Preprocessing_Hepatitis.ipynb    # Hepatitis Dataset
│   ├── Preprocessing_Hypothyroid.ipynb  # Hypothyroid Dataset
│   ├── Preprocessing_ILPD.ipynb         # Indian Liver Patient Dataset
│   ├── Preprocessing_Pima.ipynb         # Pima Indians Diabetes
│   ├── Preprocessing_Thoracic.ipynb     # Thoracic Surgery
│   ├── Preprocessing_newthyroid1.ipynb  # New Thyroid (Class 1)
│   ├── Preprocessing_newthyroid2.ipynb  # New Thyroid (Class 2)
│   └── DataPrepration_Vertebral.ipynb   # Vertebral Column
└── README.md                            # This file
```

## Preprocessing Notebooks

All preprocessing notebooks follow a consistent structure and produce standardized output datasets suitable for use in the HardVAE framework.

### Dataset-Specific Notebooks

#### 1. **Preprocessing_BCWDD.ipynb**
**Dataset**: Breast Cancer Wisconsin Diagnostic Dataset

**Steps**:
- Data loading and exploration
- Feature analysis and statistics
- Missing value handling
- Feature scaling and normalization
- Class imbalance analysis
- Train-test split
- Output: `BCWDD_processed.csv`

**Key Characteristics**:
- 569 samples, 30 features
- Binary classification (Malignant/Benign)
- Moderate imbalance ratio (1.68)

#### 2. **Preprocessing_HearstCleveland.ipynb**
**Dataset**: Heart Disease (Cleveland)

**Steps**:
- Data loading and initial exploration
- Handling missing values
- Feature engineering and selection
- Standardization of features
- Class distribution analysis
- Data validation
- Output: `HeartCleveland_processed.csv`

**Key Characteristics**:
- 297 samples, 13 features
- Binary classification (Disease/No Disease)
- Moderate imbalance ratio (1.17)

#### 3. **Preprocessing_Hepatitis.ipynb**
**Dataset**: Hepatitis Dataset

**Steps**:
- Loading and initial data inspection
- Missing value imputation strategies
- Feature scaling and transformation
- Outlier detection and handling
- Class balance analysis
- Final validation
- Output: `Hepatitis_processed.csv`

**Key Characteristics**:
- 155 samples, 19 features
- Binary classification (Die/Live)
- High imbalance ratio (3.84)

#### 4. **Preprocessing_Hypothyroid.ipynb**
**Dataset**: Hypothyroid Dataset

**Steps**:
- Data loading and exploration
- Handling categorical and numerical features
- Feature encoding and scaling
- Missing value treatment
- Class imbalance analysis
- Stratified train-test split
- Output: `Hypothyroid_processed.csv`

**Key Characteristics**:
- 3,163 samples, 29 features
- Binary classification (Hypothyroid/Normal)
- High imbalance ratio (6.22)

#### 5. **Preprocessing_ILPD.ipynb**
**Dataset**: Indian Liver Patient Dataset

**Steps**:
- Initial data loading and inspection
- Feature analysis and statistics
- Missing value handling
- Feature normalization
- Outlier detection
- Class distribution analysis
- Output: `ILPD_processed.csv`

**Key Characteristics**:
- 583 samples, 10 features
- Binary classification (Liver Disease/Normal)
- Moderate imbalance ratio (2.49)

#### 6. **Preprocessing_Pima.ipynb**
**Dataset**: Pima Indians Diabetes Dataset

**Steps**:
- Data loading and exploration
- Feature analysis and statistics
- Missing value imputation
- Feature scaling
- Class imbalance analysis
- Data validation
- Output: `Pima_processed.csv`

**Key Characteristics**:
- 768 samples, 8 features
- Binary classification (Diabetes/No Diabetes)
- Moderate imbalance ratio (1.87)

#### 7. **Preprocessing_Thoracic.ipynb**
**Dataset**: Thoracic Surgery Dataset

**Steps**:
- Data loading and initial exploration
- Categorical feature encoding
- Numerical feature scaling
- Missing value handling
- Feature selection
- Class distribution analysis
- Output: `Thoracic_processed.csv`

**Key Characteristics**:
- 470 samples, 16 features
- Binary classification (Survival/Death)
- Moderate imbalance ratio (5.71)

#### 8. **Preprocessing_newthyroid1.ipynb**
**Dataset**: New Thyroid Dataset (Class 1)

**Steps**:
- Data loading and exploration
- Feature scaling and normalization
- Class separation (Class 1 vs Others)
- Imbalance analysis
- Data validation
- Output: `NewThyroid1_processed.csv`

**Key Characteristics**:
- 215 samples, 5 features
- Binary classification (Class 1/Other)
- High imbalance ratio (6.96)

#### 9. **Preprocessing_newthyroid2.ipynb**
**Dataset**: New Thyroid Dataset (Class 2)

**Steps**:
- Data loading and exploration
- Feature scaling and normalization
- Class separation (Class 2 vs Others)
- Imbalance analysis
- Data validation
- Output: `NewThyroid2_processed.csv`

**Key Characteristics**:
- 215 samples, 5 features
- Binary classification (Class 2/Other)
- High imbalance ratio (6.96)

#### 10. **DataPrepration_Vertebral.ipynb**
**Dataset**: Vertebral Column Dataset

**Steps**:
- Data loading and exploration
- Feature analysis and statistics
- Feature scaling and normalization
- Outlier detection and handling
- Class imbalance analysis
- Data validation
- Output: `Vertebral_processed.csv`

**Key Characteristics**:
- 310 samples, 6 features
- Binary classification (Abnormal/Normal)
- Moderate imbalance ratio (2.10)

## Common Preprocessing Steps

All notebooks follow a standardized preprocessing pipeline:

### 1. Data Loading
```python
import pandas as pd
df = pd.read_csv('path/to/raw/data.csv')
```

### 2. Exploratory Data Analysis
- Dataset shape and size
- Feature statistics (mean, std, min, max)
- Missing value analysis
- Class distribution
- Imbalance ratio calculation

### 3. Missing Value Handling
**Strategies Used**:
- **Deletion**: Remove rows with missing values (if < 5% of data)
- **Imputation**: Fill with mean/median for numerical features
- **Forward Fill**: Use previous values for time-series data
- **Domain-Specific**: Use domain knowledge for specific features

### 4. Feature Scaling and Normalization
**Methods Applied**:
- **StandardScaler**: Zero mean, unit variance (most common)
- **MinMaxScaler**: Scale to [0, 1] range
- **RobustScaler**: Resistant to outliers
- **Log Transformation**: For skewed distributions

### 5. Outlier Detection and Handling
**Techniques**:
- **IQR Method**: Remove values beyond 1.5 × IQR
- **Z-Score**: Remove values with |z| > 3
- **Visual Inspection**: Identify and handle domain-specific outliers
- **Domain Knowledge**: Apply expert judgment

### 6. Feature Engineering
**Operations**:
- Feature selection (remove irrelevant features)
- Feature creation (derived features if needed)
- Feature interaction (if applicable)
- Dimensionality reduction (if needed)

### 7. Class Imbalance Analysis
**Analysis**:
- Calculate imbalance ratio
- Visualize class distribution
- Document minority class characteristics
- Prepare for synthetic data generation

### 8. Data Validation
**Checks**:
- No missing values (NaN)
- Correct data types
- Proper scaling (mean ≈ 0, std ≈ 1)
- Correct target labels (0/1)
- No duplicate rows

### 9. Output Generation
**Format**:
- CSV file with all features + target label
- Last column contains binary target (0/1)
- Standardized naming convention
- Ready for immediate use in experiments

## Running the Notebooks

### Prerequisites
```bash
pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

### Execution Steps

1. **Navigate to the notebooks directory**:
   ```bash
   cd notebooks/preprocessing
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open a specific notebook**:
   - Click on the notebook file (e.g., `Preprocessing_BCWDD.ipynb`)
   - Run all cells using `Kernel → Restart & Run All`
   - Or run cells individually using `Shift + Enter`

4. **Verify output**:
   - Check that processed CSV is generated in `data/processed/`
   - Verify no errors in data validation cells

## Notebook Structure

Each preprocessing notebook follows this structure:

```
1. Setup and Imports
   - Import required libraries
   - Set visualization parameters
   - Define utility functions

2. Data Loading
   - Load raw dataset
   - Display basic information

3. Exploratory Data Analysis (EDA)
   - Dataset overview
   - Feature statistics
   - Missing value analysis
   - Class distribution

4. Data Preprocessing
   - Handle missing values
   - Scale features
   - Handle outliers
   - Feature engineering

5. Class Imbalance Analysis
   - Calculate imbalance ratio
   - Visualize distribution
   - Document characteristics

6. Data Validation
   - Check for issues
   - Verify transformations
   - Ensure quality

7. Output Generation
   - Save processed dataset
   - Create summary statistics
   - Document preprocessing steps
```

## Reproducibility

To reproduce the preprocessing:

1. **Use the same raw data**: Ensure raw datasets are in `data/raw/`
2. **Run notebooks in order**: Execute each notebook completely
3. **Verify outputs**: Check that processed files match expected outputs
4. **Document changes**: If modifying preprocessing, document all changes

## Customization

To customize preprocessing for your needs:

1. **Open the notebook** in Jupyter
2. **Modify preprocessing steps** as needed
3. **Run the modified notebook**
4. **Verify the output** matches your requirements
5. **Save the modified notebook** with a new name

## Common Issues and Solutions

### Issue: Missing Values Not Handled
**Solution**: Check the imputation strategy in the "Missing Value Handling" section

### Issue: Features Not Scaled Properly
**Solution**: Verify the scaler type and parameters in the "Feature Scaling" section

### Issue: Outliers Causing Issues
**Solution**: Adjust outlier detection thresholds in the "Outlier Detection" section

### Issue: Class Imbalance Not Addressed
**Solution**: This is intentional - imbalance is handled by HardVAE synthetic generation

## Output Verification

After running a preprocessing notebook, verify:

- ✅ Output file exists in `data/processed/`
- ✅ File size is reasonable (not empty or corrupted)
- ✅ No missing values (NaN) in the output
- ✅ Features are properly scaled (mean ≈ 0, std ≈ 1)
- ✅ Target column contains only 0 and 1
- ✅ Number of samples matches expected count

## Integration with HardVAE

The preprocessed datasets are automatically used by:

1. **Hardness Calculation**: `HardnessCalculator` class
2. **CVAE Training**: `HardnessAwareCVAETrainer` class
3. **Evaluation**: `SyntheticDataEvaluator` class
4. **Utility Assessment**: Classification metrics

## References

For detailed information about preprocessing techniques:

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Pandas Documentation**: https://pandas.pydata.org/
- **Data Preprocessing Guide**: See `docs/preprocessing_guide.md`

## Questions and Support

For questions about preprocessing:

1. Check the specific notebook for detailed comments
2. Review the "Common Issues" section above
3. Consult the main README.md for general information
4. Open an issue on GitHub for specific problems

---

**Last Updated**: December 2024  
**Notebook Version**: 1.0  
**Total Notebooks**: 10 preprocessing notebooks  
**Total Dataset Coverage**: 10 medical datasets with varying characteristics

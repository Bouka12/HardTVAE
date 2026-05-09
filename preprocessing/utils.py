""" This Class for preprocessing of data in training

Following SOTA we apply the procedures:
`TabDiff`:
    1. Replace Missing Numerical values by the column average (TabDDPM)
    2. Treat Categorical Missing values as a new category (TabDDPM)
    3. Transform all categorical feature to numerical either label encoding/one-hot encoding (OURS, STaSy)
    -> this step is necessary as the evaluation framework is adapted to numerical features (KS Similarity test only for distribution, TDA process)
    -> STasy treats the one-hot encoding of categorical columns as continuous features which are then processed together with the numerical columns
    |-|-> Stasy proposed several training strategies including self-paced learning to stabilize the training process.
    4. Transform the ranges of features using QuantileTransformer  (TabDDPM)

`TabSyn`:
    1. Replace Missing Numerical values by the column average (TabDDPM)
    2. Treat Categorical Missing values as a new category (TabDDPM)
    3. each numerical/categorical column is transformed using the QuantileTransformer11/OneHotEncoder12 from scikit-learn,
    respectively

"SOLUTION FOR MIXED TYPE PROBLEM IN EVALUATION"
    * Just for training the generators TVAE and CTGAN use the mixed type Training data
    * After sampling minority data apply necessary numerical transformations for the evaluation (Distributional & TDA)
    * Get HardVAE and HardTVAE! -> are not similar as the data for their training is different in trasnformation

**Note:** if TVAE and CTGAN data will maintain its mixed type nature
    else HardVAE and VAE will transform data to numerical
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split

class preprocess_data():
    """
    Input: receives the x_train, x_test, y_train, y_test
    
    Procedures:
    1. missing value imputation with the column average for numerical features
    2. treat missing values in categorical features as a new category
    3. use one-hot encoding for categorical features
    4. use quantile-transformer for scaling treating all feature as numeric here
    
    Functions:
    1. apply the procedure to train and test without data leakage
    
    Output:
    preprocessed data: X_train, X_test, y_train, y_test
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None

    def fit_transform(self, X_train):
        # Ensure inputs are DataFrames so we can extract column data types
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            
        # Dynamically separate numerical and categorical columns
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns

        # 1. Numerical Pipeline: Impute missing values with mean
        numeric_transformer = SimpleImputer(strategy='mean')

        # 2 & 3. Categorical Pipeline: Impute with 'missing' category, then One-Hot Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_category')),
            # handle_unknown='ignore' prevents errors if test set has unseen categories
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])

        # Combine numerical and categorical pipelines
        column_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # 4. Final Pipeline: Apply ColumnTransformer, then QuantileTransformer to everything
        self.preprocessor = Pipeline(steps=[
            ('col_trans', column_transformer),
            ('quantile', QuantileTransformer(random_state=self.random_state))
        ])

        # 5. Fit on Train, Transform Train and Test (Prevents Data Leakage)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        return X_train_processed

    def transform(self, X_test):
        # Ensure inputs are DataFrames so we can extract column data types
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        X_test_processed = self.preprocessor.transform(X_test)

        return X_test_processed

df =pd.read_csv("./data/processed/Hepatitis.csv")
print(df.head())
X, y = df.iloc[:,:-1], df.iloc[:,-1]
print(X.head())
print(y.head())
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
preprocessor = preprocess_data(random_state=42)
X_train_processed = preprocessor.fit_transform(X_train=X_train)
X_test_processed = preprocessor.transform(X_test=X_test)
print(X_train_processed)



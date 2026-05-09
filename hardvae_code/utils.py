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


**Changes for HardTVAE: >** 
    - QuantileTransformer applied only to numerical features (not one-hot columns)
    - After fitting, returns:
        self.n_numerical    : int       - number of numerical output columns
        self.cat_dims       : list[int] - size of each one-hot group, in order
        self.data_info      : dict      - convenience bundle passed to TabularTVAE
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
    Input: receives the X_train (DataFrame or array), X_test
    
    Procedures:
        1. missing value imputation with the column average for numerical features
        2. treat missing values in categorical features as a new category
        3. One-hot encoding for categorical features
        4. Quantile-transformer scaling for numerical features
    
    Functions:
        1. apply the procedure to train and test without data leakage
    
    Output:
        - X_array : nd.ndarray      -> for TVAE (model input)
        - X_df    : pd.DataFrame    -> for SDV/CTGAN (needs named columns 
                                        to infer metadata)


    Attributes exposed after fit_transform():
        n_numerical : int       - number of numerical comuns in output
        cat_dims    : list[int] - number of one-hot cols per categorical feature
        data_info   : dict      - {'n_numerical': ..., 'cat_dims': [...]}
                                passed directly to TabularCVAE.__init__()

    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.categorical_features = None


        
        # Populated after fir_transform - consumed b TabularCVAE
        self.n_numerical = None
        self.cat_dims = None
        self.data_info = None
        self.feature_names_out = None # <- column names for the output array

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Returns:
            X_array     : nd.ndarray shape (n, total_output_dim)    -> TVAE input
            X_df        : pd.DataFrame shape (n, total_output_dim)  -> SDV/ CTGAN input
        """
        # Ensure inputs are DataFrames so we can extract column data types
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            
        # Dynamically separate numerical and categorical columns
        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()     #['int64', 'float64']).columns
        categorical_features = X_train.columns.difference(numeric_features).tolist() #select_dtypes(include=['object', 'category', 'bool']).columns
        
        self.categorical_features = categorical_features

        # Cast categorical to str to avoid mixed-type issues
        X_train_copy = X_train.copy()
        for col in categorical_features:
            X_train_copy[col] = X_train_copy[col].astype(str)

        # _____ Numerical Pipeline: Impute missing values with mean -> QuantileTrabsformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('quantile', QuantileTransformer(
                output_distribution='uniform', # matches original TVAE paper
                n_quantiles = min(len(X_train), 5000),    # < the number of quantiles should be less than the number of instances
                random_state=self.random_state
            ))
            ])

        # _____ Categorical Pipeline: Impute with 'missing' category, then One-Hot Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy='constant', fill_value='missing_category'
                )),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore', sparse_output=False #,drop='first'
                )) 
        ])

        # ______ Combine numerical and categorical pipelines
        #   Numerical columns come FIRST, then categorical one-hot blocks
        #   This ordering is assumed by TabularCVAE and the loss function
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer,    numeric_features),
                ('cat', categorical_transformer,    categorical_features),
            ],
            remainder='drop'
        )

        X_array = self.preprocessor.fit_transform(X_train_copy)

        # _____ Extract metadata ______
        self.n_numerical = len(numeric_features)

        if categorical_features:
            ohe = (self.preprocessor
                        .named_transformers_['cat']
                        .named_steps['onehot'])
            self.cat_dims = [len(cats) for cats in ohe.categories_]
        else:
            self.cat_dims = []

        self.data_info = {
            'n_numerical': self.n_numerical,
            'cat_dims': self.cat_dims,
        }

        # Build output column names
        ## Numerical: keep oriiginal name
        num_names = list(numeric_features)

        ## Categorical: "<original_col>_<category_value>"
        cat_names = []
        if categorical_features:
            ohe = (self.preprocessor
                       .named_transformers_['cat']
                       .named_steps['onehot'])
            for col, cats in zip(categorical_features, ohe.categories_):
                for cat_val in cats:
                    cat_names.append(f"{col}_{cat_val}")

        self.feature_names_out = num_names + cat_names 

        
        # print(f"[preprocess_data] numerical cols : {self.n_numerical}")
        # print(f"[preprocess_data] categorical groups: {len(self.cat_dims)} "
        #       f"with dims {self.cat_dims}")
        # print(f"[preprocess_data] total output dim: "
        #       f"{self.n_numerical + sum(self.cat_dims)}")
        
        #------ Return both array and named DataFrame ---------
        X_df = pd.DataFrame(X_array, columns=self.feature_names_out,
                            index=X_train.index)
        
        return X_array, X_df


    def transform(self, X_test: pd.DataFrame) -> np.ndarray:
        # Ensure inputs are DataFrames so we can extract column data types
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # Mirror the str cast for categorical columns
        cat_transformer = self.preprocessor.named_transformers_.get('cat')
        if cat_transformer is not None:
            ohe = cat_transformer.named_steps['onehot']
            cat_cols = []
            # Recover original categorical feature names from ColumnTransformer
            for name, _, cols in self.preprocessor.transformers_:
                if name == 'cat':
                    cat_cols = list(cols)
            for col in cat_cols:
                if col in X_test.columns:
                    X_test = X_test.copy()
                    X_test[col] = X_test[col].astype(str)
        
        X_array = self.preprocessor.transform(X_test)

        X_df = pd.DataFrame(X_array, columns=self.feature_names_out,
                            index=X_test.index)
        
        return X_array, X_df


# # TESTING THE `preprocess_data` CLASS

# df =pd.read_csv(r"C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\data\processed\Hypothyroid.csv")
# print(df.head())
# X, y = df.iloc[:,:-1], df.iloc[:,-1]
# print(X.head())
# print(y.head())
# X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
# preprocessor = preprocess_data(random_state=42)
# X_train_array,df_train  = preprocessor.fit_transform(X_train=X_train)
# X_test_array,df_test = preprocessor.transform(X_test=X_test)
# print(f"X_train_array head: {X_train_array[:4, :]}")
# print(f"columns of train df: {df_train.columns}")

# # NOTE: the preprocess_data won't return df after preprocessing which is not adequte for the TVAE an CTGAN from sdc
# # as `sdv` get the metada from the train data, by that we can use back the previous preprocess_data for only ctgan and tvae!
# # now we continue with the HardTVAE.



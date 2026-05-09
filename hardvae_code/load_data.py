
"""
load_data.py
This module provides functionality to load and preprocess data from a CSV file used for the input data

1. Use from preprocesing import utils
2. Handle missing values function
3. Transform Categorical vals to numerical function
4. train, validation, test split (0.8/0.1/0.1) 
"""

from sklearn.model_selection import train_test_split
import pandas as pd
# from utils import preprocess_data

def load_data(path_processed, test_size = 0.2, random_state=None):
    # Read the data from a CSV file
    df = pd.read_csv(path_processed, sep=',')
    
    # Separate features and labels
    df_base = df.iloc[:, :-1]  # Features
    df_labels = df.iloc[:, -1]  # Labels

    # Split data into train and test sets using stratified random sampling
    X_train, X_temp,y_train, y_temp = train_test_split(
        df_base,
        df_labels,
        test_size=test_size,
        stratify=df_labels,
        random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )
    
    
    # # Preprocess the data:
    # preprocessor = preprocess_data(random_state=random_state)
    # X_train_pro = preprocessor.fit_transform(X_train=X_train)
    # X_val_pro = preprocessor.transform(X_val)
    # X_test_pro = preprocessor.transform(X_test)

    # # Identify majority and minority classes in the training set
    # label_counts = pd.Series(y_train).value_counts()
    # #print(label_counts)
    # minority_label = label_counts.idxmin()
    # majority_label = label_counts.idxmax()

    # # Calculate imbalance ratio
    # minority_count = label_counts[minority_label]
    # majority_count = label_counts[majority_label]
    # imbalance_ratio = minority_count / majority_count
    
    # # Separate the training data into majority and minority classes
    # minority_data = X_train_pro[y_train == minority_label]
    # majority_data = X_train_pro[y_train == majority_label]
    
    return X_train, y_train, X_val, y_val, X_test, y_test




# # CHECK THE FUNCTIO `load_data`: ISSUE w/ Hypothyroid, ILPD,
# path = "./data/processed/Hypothyroid.csv"
# X_train_pro, y_train, X_val_pro, y_val, X_test_pro, y_test, majority_data, minority_data, imbalance_ratio = load_data(path_processed=path,random_state=42)

# # check the shape of train, val, test
# print(f"train shape: {X_train_pro.shape}")
# print(f"X_val shape : {X_val_pro.shape}")
# print(f"X_test shape : {X_test_pro.shape}")
# print(f"imbalance ratio : {imbalance_ratio}")

# print(f"minority data shape : {minority_data.shape}")
# print(f"majority data shape : {majority_data.shape}")
# # Print imbalance ratio
# # print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f} (Majority: {majority_count}, Minority: {minority_count})")
# print(f"type of minority data: {type(minority_data)}")
"""
Data loading and preprocessing utilities for HardVAE.
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path_processed, test_size=0.2, random_state=None):
    df = pd.read_csv(path_processed, sep=",")
    df_base = df.iloc[:, :-1]
    df_labels = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        df_base.values,
        df_labels,
        test_size=test_size,
        stratify=df_labels,
        random_state=random_state
    )
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)
    return X_train, y_train, X_test, y_test, standardizer

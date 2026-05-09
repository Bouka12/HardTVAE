"""
    utility_evaluation.py used to evaluate the classification performance on the balanced data
    usign certain classification algorithms
"""
"""
Evaluate the utility of the synthetic data using the XGBoost
    *   Use XGBoost for classification
    *   Perform Hyperparameter tuning for each dataset (combination)
    *   Use a grid search for tuning

-   XGBoost params grid
-   Other Classifiers (params)
-   Weighted F1    
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, classification_report
)
from imblearn.metrics import specificity_score
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class ClassificationEvaluator:
    """
    Evaluates multiple classifiers with hyperparameter tuning using Grid Search.
    Supports train, validation, and test sets.
    """
    
    def __init__(self, dataset_name,combination_name, random_state=42, results_path="./xgboost_results"):
        self.dataset_name = dataset_name
        self.config = combination_name
        self.random_state = random_state
        self.results_path = results_path
        # self.scaler = StandardScaler()
        
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
        # # filename = f"{self.dataset_name}_{classifier_name}_best_params.csv"
        # file_path = os.path.join(self.results_path, self.dataset_name)
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # self.file_path = file_path
    
            
        # Define hyperparameter grids for each classifier
        self.param_grids = {
            # 'LogisticRegression': {
            #     'C': [0.1, 1, 10],
            #     'solver': ['liblinear']
            # },
            # 'RandomForestClassifier': {
            #     'n_estimators': [50, 100, 200, 500, 1000],
            #     'max_depth': [None, 5, 10, 20],
            #     'min_samples_split': [2, 5, 10]
            # },
            # 'SVC': {
            #     'C': [0.1, 1, 10],
            #     'kernel': ['rbf', 'linear'],
            #     'gamma': ['scale', 'auto']
            # },
            # 'KNeighborsClassifier': {
            #     'n_neighbors': [3, 5, 7],
            #     'weights': ['uniform', 'distance']
            # },
            'XGBClassifier': {
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_weight': [1, 3, 5],
                'max_depth': [3, 4, 5],
                'gamma' : [0.0, 0.1, 0.5],
                'subsample': [0.7, 0.8],   # introduce randomness to fight overfitting
                'colsample_bytree' : [0.7, 0.8]
            }
        }

    def _get_classifiers(self):
        return {
            # 'LogisticRegression': LogisticRegression(random_state=self.random_state),
            # 'RandomForestClassifier': RandomForestClassifier(random_state=self.random_state),
            # 'SVC': SVC(probability=True, random_state=self.random_state),
            # 'KNeighborsClassifier': KNeighborsClassifier(),
            'XGBClassifier': XGBClassifier(
                n_estimators = 500,
                early_stopping_rounds = 15,
                eval_metric='aucpr',           # <--- ADDED: Tells XGBoost what to optimize during early stopping
                random_state=self.random_state,
                device = DEVICE.type
                )
        }

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Performs tuning using X_train and X_val via PredefinedSplit,
        then evaluates the best model on X_test.
        """
        
        # Combine train and validation for GridSearchCV with PredefinedSplit
        X_tuning = np.vstack((X_train, X_val))
        y_tuning = np.concatenate((y_train, y_val))
        
        # Create a list where -1 indicates training and 0 indicates validation
        test_fold = np.concatenate([
            -1 * np.ones(X_train.shape[0]),
             0 * np.ones(X_val.shape[0])
        ])
        ps = PredefinedSplit(test_fold)
        
        classifiers = self._get_classifiers()
        all_results = []
        
        for name, clf in classifiers.items():
            print(f"\n--- Tuning and Evaluating {name} for dataset: {self.dataset_name} ---")
            
            grid_search = GridSearchCV(
                estimator=clf,
                param_grid=self.param_grids.get(name, {}),
                cv=ps, # Use the PredefinedSplit here
                scoring='f1_weighted',
                n_jobs=1
            )
            
            # <--- ADDED: This block safely passes the validation set strictly to XGBoost
            fit_params = {}
            if name == 'XGBClassifier':
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False
            # -------------------------------------------------------------------------
            
            # <--- MODIFIED: Notice the **fit_params passed into the fit method
            grid_search.fit(X_tuning, y_tuning, **fit_params) 
            
            best_clf = grid_search.best_estimator_
            best_params = grid_search.best_params_

            params_df = pd.DataFrame([best_params])
            params_df['classifier'] = name
            params_df['dataset'] = self.dataset_name
            params_df['seed'] = self.random_state
            
            # Save best hyperparameters to CSV
            # self._save_best_params(name, best_params)
            
            # Predict on test set
            y_pred = best_clf.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(name, y_test, y_pred)
            all_results.append(metrics)
            
            # Print report
            print(f"Best Params: {best_params}")
            
            # <--- OPTIONAL ADDITION: This proves early stopping worked by printing the final tree count
            if name == 'XGBClassifier':
                print(f"Optimal trees used (early stopping): {best_clf.best_iteration}")
                
            print(classification_report(y_test, y_pred))
        
            
        return pd.DataFrame(all_results), params_df

    def _calculate_metrics(self, name, y_true, y_pred):
        return {
            'Dataset': self.dataset_name,
            'Classifier': name,
            # 'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0, average='weighted'),
            'Recall': recall_score(y_true, y_pred, zero_division=0, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0, average='weighted'),
            'Specificity': specificity_score(y_true, y_pred, average='weighted'),
            # 'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred)
        }

    # def _save_best_params(self, classifier_name, params):
    #     params_df = pd.DataFrame([params])
    #     params_df['dataset'] = self.dataset_name
        
    #     filename = f"{self.config}_best_params.csv"
    #     filepath = os.path.join(self.file_path, filename)
        
    #     params_df.to_csv(filepath, index=False)
    #     print(f"Saved best parameters for {classifier_name} to {filepath}")

# # # Example usage (for testing)
# if __name__ == "__main__":
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split
    
#     # Generate dummy data
#     X, y = make_classification(n_samples=200, n_features=10, random_state=42)
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
#     evaluator = ClassificationEvaluator(dataset_name="DummyDataset", combination_name="basic")
#     results = evaluator.evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
#     print("\nFinal Results Summary:")
#     print(results)

"""
Utility metrics for HardVAE, including classification evaluation.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.metrics import specificity_score

def evaluate_classification_model(X_train, y_train, X_test, y_test, random_state=42):
    classifiers = [
        LogisticRegression(random_state=random_state, solver='liblinear'),
        RandomForestClassifier(random_state=random_state),
        SVC(probability=True, random_state=random_state),
        KNeighborsClassifier(n_neighbors=5)
    ]
    results = []
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = {
            'Classifier': clf.__class__.__name__,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'Specificity': specificity_score(y_test, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred)
        }
        results.append(metrics)
    return results

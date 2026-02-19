import os
import sys
import pickle
import numpy as np
from src.exception import CustomException
from src.logger import logger
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             precision_score, recall_score, confusion_matrix, roc_curve)

def save_object(file_path: str, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads a Python object from a pickle file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: dict) -> dict:
    """
    Trains each model and returns a report dict with Accuracy, F1, AUC-ROC, Precision, Recall,
    and data for Confusion Matrix and ROC Curves.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            
            try:
                y_prob  = model.predict_proba(X_test)[:, 1]
            except (AttributeError, IndexError):
                # Fallback for models without predict_proba
                y_prob = np.zeros(len(y_test)) 

            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # AUC-ROC
            try:
                auc_roc = roc_auc_score(y_test, y_prob) if np.max(y_prob) > 0 else 0.5
            except ValueError:
                auc_roc = 0.5

            # ROC Curve context
            fpr, tpr, _ = roc_curve(y_test, y_prob)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            report[name] = {
                "Accuracy":  round(accuracy, 4),
                "F1-Score":  round(f1, 4),
                "Precision": round(precision, 4),
                "Recall":    round(recall, 4),
                "AUC-ROC":   round(auc_roc, 4),
                "confusion_matrix": cm,
                "roc_data": {"fpr": fpr, "tpr": tpr, "auc": auc_roc}
            }
            logger.info(f"{name}: F1={report[name]['F1-Score']}, AUC={report[name]['AUC-ROC']}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

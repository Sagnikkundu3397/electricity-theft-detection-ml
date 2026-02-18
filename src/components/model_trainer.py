import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_models

# Optional imports for advanced models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("models", "model.pkl")


class ModelTrainer:
    """
    Trains multiple machine learning models and selects the best one.
    """
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array: np.ndarray,
                                test_array: np.ndarray) -> tuple:
        """
        Trains models, evaluates them, and saves the best model.
        """
        logger.info("Model Training started")
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced'),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=42,
                    class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=8, random_state=42, class_weight='balanced'),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            }

            if XGBClassifier:
                models["XGBoost"] = XGBClassifier(
                    use_label_encoder=False, eval_metric='logloss',
                    random_state=42)

            if CatBoostClassifier:
                models["CatBoost"] = CatBoostClassifier(
                    verbose=False, random_seed=42,
                    auto_class_weights='Balanced')

            logger.info(f"Evaluating models: {list(models.keys())}")
            report = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_name = max(report, key=lambda k: report[k]["F1-Score"])
            best_metrics = report[best_name]

            logger.info(f"Best: {best_name} | F1={best_metrics['F1-Score']}")

            best_model = models[best_name]
            best_model.fit(X_train, y_train)

            save_object(self.config.trained_model_path, best_model)

            # Export Feature Importance
            try:
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    feature_names = ["mean", "std", "min", "max", "zeros"]
                    imp_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)

                    imp_path = os.path.join("models", "feature_importance.csv")
                    imp_df.to_csv(imp_path, index=False)
                    logger.info(f"Feature importance saved to {imp_path}")
            except Exception as fe:
                logger.warning(f"Could not export feature importance: {fe}")

            return best_name, best_metrics, report

        except Exception as e:
            raise CustomException(e, sys)

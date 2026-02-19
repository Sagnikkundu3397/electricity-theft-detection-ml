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
                    n_estimators=50, random_state=42,
                    class_weight='balanced', n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=5, random_state=42, class_weight='balanced'),
            }

            if XGBClassifier:
                models["XGBoost"] = XGBClassifier(
                    n_estimators=50, use_label_encoder=False, 
                    eval_metric='logloss', random_state=42, n_jobs=-1)

            logger.info(f"Evaluating models: {list(models.keys())}")
            report = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_name = max(report, key=lambda k: report[k]["F1-Score"])
            best_metrics = report[best_name]

            logger.info(f"Best: {best_name} | F1={best_metrics['F1-Score']}")

            best_model = models[best_name]
            best_model.fit(X_train, y_train)

            save_object(self.config.trained_model_path, best_model)

            # Generate Enhanced Visualizations
            self.generate_visualizations(report, best_name, best_model)

            return best_name, best_metrics, report

        except Exception as e:
            raise CustomException(e, sys)

    def generate_visualizations(self, report, best_name, best_model):
        """
        Generates and saves IEEE-style visualizations for results evaluation.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # --- 1. Feature Importance Plot (XGBoost / Ensemble fallback) ---
            try:
                importance_model = best_model
                # Fallback to an ensemble model if best_model doesn't have features (e.g., Logistic Regression)
                if not hasattr(importance_model, "feature_importances_"):
                    for name in ["XGBoost", "Random Forest", "Decision Tree"]:
                        if name in report:
                            # Re-fit or use the model from trainer if we had saved them
                            # Since we don't save all fitted models in memory, we might need a different approach
                            # but we know they were fitted in evaluate_models. 
                            # Actually, evaluate_models doesn't return the fitted objects.
                            pass
                
                # If best_model has it, use it. If not, try to get it from others or use coefficients for IR
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    title = f"Feature Importance: {best_name} Model"
                elif hasattr(best_model, "coef_"):
                    importances = np.abs(best_model.coef_[0])
                    title = f"Feature Importance (Coefficients): {best_name} Model"
                else:
                    importances = None
                
                if importances is not None:
                    feature_names = ["mean", "std", "min", "max", "zeros"]
                    imp_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="magma", legend=False)
                    plt.title(title, fontsize=14)
                    plt.xlabel("Absolute Score")
                    plt.grid(axis='x', linestyle='--', alpha=0.6)
                    plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Feature Importance plot saved.")
            except Exception as e:
                logger.warning(f"Could not generate feature importance plot: {e}")

            # --- 2. ROC Curve Comparison (All Models) ---
            plt.figure(figsize=(10, 8))
            for name, metrics in report.items():
                roc_data = metrics.get("roc_data")
                if roc_data:
                    plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"{name} (AUC = {roc_data['auc']:.2f})")
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison: Effectiveness of Models', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(reports_dir, "roc_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("ROC Curve Comparison plot saved.")

            # --- 3. Confusion Matrix for the Best Model ---
            cm = report[best_name]["confusion_matrix"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Theft'], yticklabels=['Normal', 'Theft'])
            plt.title(f"Confusion Matrix: Best Selected Model ({best_name})", fontsize=14)
            plt.ylabel('Ground Truth')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion Matrix plot saved.")

            # --- 4. Model Comparison Bar Graph ---
            comparison_metrics = ["Accuracy", "F1-Score", "Recall", "AUC-ROC"]
            df_comp = []
            for name, metrics in report.items():
                for m in comparison_metrics:
                    df_comp.append({"Model": name, "Metric": m, "Score": metrics[m]})
            
            df_comp = pd.DataFrame(df_comp)
            plt.figure(figsize=(12, 7))
            sns.barplot(data=df_comp, x="Metric", y="Score", hue="Model", palette="coolwarm")
            plt.title("Comparative Performance Analysis (All Models)", fontsize=14)
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(os.path.join(reports_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Model Comparison Bar Graph saved.")

        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")

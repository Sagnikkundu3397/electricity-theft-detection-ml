import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("models", "preprocessor.pkl")


class DataTransformation:
    """
    Handles feature engineering, scaling, and handling class imbalance (SMOTE).
    """
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        """Returns a sklearn Pipeline with StandardScaler."""
        try:
            pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def extract_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extracts statistical features (mean, std, min, max) from time-series data.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            tuple: (features_dataframe, target_series)
        """
        try:
            # Separate features and target if exists
            target = None
            if 'FLAG' in df.columns:
                target = df['FLAG']
                df_features = df.drop(columns=['FLAG'])
            else:
                df_features = df

            # Force numeric conversion, coercive errors to NaN
            df_numeric = df_features.apply(pd.to_numeric, errors='coerce')
            
            # Fill NaNs with 0
            df_numeric.fillna(0, inplace=True)

            # Feature Engineering
            features = pd.DataFrame(index=df_numeric.index)
            features['mean']  = df_numeric.mean(axis=1)
            features['std']   = df_numeric.std(axis=1)
            features['min']   = df_numeric.min(axis=1)
            features['max']   = df_numeric.max(axis=1)
            features['zeros'] = (df_numeric == 0).sum(axis=1)  # Count suspicious 0 readings

            return features, target

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Reads pre-engineered datasets, applies scaling and SMOTE.
        
        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        logger.info("Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logger.info("Splitting Features and Target")
            target_column_name = "FLAG"
            
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]
            
            X_test  = test_df.drop(columns=[target_column_name])
            y_test  = test_df[target_column_name]

            logger.info(f"Features Shape: {X_train.shape}")

            preprocessor = self.get_data_transformer_object()
            
            # Scale features
            logger.info("Scaling features")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled  = preprocessor.transform(X_test)

            # Apply SMOTE to Training Data ONLY
            logger.info("Applying SMOTE to balance training data")
            try:
                from imblearn.over_sampling import SMOTE
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Capture distribution BEFORE SMOTE
                df_before = pd.DataFrame({'Target': y_train, 'Status': 'Before SMOTE'})
                
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"SMOTE applied. New Train Shape: {X_train_resampled.shape}")

                # Capture distribution AFTER SMOTE
                df_after = pd.DataFrame({'Target': y_train_resampled, 'Status': 'After SMOTE'})
                
                # Plotting SMOTE Effect
                plt.figure(figsize=(10, 5))
                df_plot = pd.concat([df_before, df_after])
                sns.countplot(data=df_plot, x='Target', hue='Status', palette='viridis')
                plt.title("SMOTE Effect: Impact of Synthetic Over-sampling")
                plt.xlabel("Class (0: Normal, 1: Theft)")
                plt.ylabel("Count")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                reports_dir = "reports"
                os.makedirs(reports_dir, exist_ok=True)
                plt.savefig(os.path.join(reports_dir, "smote_effect.png"), dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("SMOTE effect visualization saved to reports/smote_effect.png")

            except ImportError:
                logger.warning("imbalanced-learn or matplotlib/seaborn not found. Skipping SMOTE visualization.")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train

            # Combine with target
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr  = np.c_[X_test_scaled,  np.array(y_test)]

            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logger.info("Preprocessor saved to models/preprocessor.pkl")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

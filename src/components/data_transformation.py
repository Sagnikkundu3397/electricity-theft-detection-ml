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
        Reads train/test splits, applies feature engineering, SMOTE, and scaling.
        
        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        logger.info("Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logger.info("Extracting features from Train Data")
            X_train_new, y_train = self.extract_features(train_df)
            
            logger.info("Extracting features from Test Data")
            X_test_new, y_test   = self.extract_features(test_df)

            logger.info(f"Engineered Features Shape: {X_train_new.shape}")

            preprocessor = self.get_data_transformer_object()
            
            # Scale features
            logger.info("Scaling features")
            X_train_scaled = preprocessor.fit_transform(X_train_new)
            X_test_scaled  = preprocessor.transform(X_test_new)

            # Apply SMOTE to Training Data ONLY
            logger.info("Applying SMOTE to balance training data")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                logger.info(f"SMOTE applied. New Train Shape: {X_train_resampled.shape}")
            except ImportError:
                logger.warning("imbalanced-learn not found. Skipping SMOTE.")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train

            # Combine with target
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr  = np.c_[X_test_scaled,  np.array(y_test)]

            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logger.info("Preprocessor saved to models/preprocessor.pkl")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

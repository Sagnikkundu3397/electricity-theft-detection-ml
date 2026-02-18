import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""
    # Use absolute path to the user's dataset
    source_data_path: str = os.path.join("data", "raw", "electricity_theft_data.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path:  str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    """
    Handles reading the dataset from the source and splitting it into 
    training and testing sets.
    """
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Reads the dataset, extracts features, splits into train/test, and saves them.
        
        Returns:
            tuple[str, str]: Paths to the train and test CSV files.
        """
        logger.info("Data Ingestion started (with Feature Engineering)")
        try:
            logger.info(f"Reading dataset from: {self.config.source_data_path}")
            
            df = pd.read_csv(self.config.source_data_path)
            
            logger.info(f"Original dataset shape: {df.shape}")

            # Basic cleanup: Ensure target 'FLAG' exists
            if 'FLAG' not in df.columns:
                raise CustomException("Dataset missing 'FLAG' column (Target Variable)", sys)

            # --- Unified Feature Extraction ---
            logger.info("Extracting features during ingestion...")
            target = df['FLAG']
            df_features = df.drop(columns=['FLAG', 'CONS_NO'], errors='ignore')
            
            df_numeric = df_features.apply(pd.to_numeric, errors='coerce')
            df_numeric.fillna(0, inplace=True)

            # Feature Engineering (Engineered features only)
            processed_df = pd.DataFrame(index=df.index)
            processed_df['mean']  = df_numeric.mean(axis=1)
            processed_df['std']   = df_numeric.std(axis=1)
            processed_df['min']   = df_numeric.min(axis=1)
            processed_df['max']   = df_numeric.max(axis=1)
            processed_df['zeros'] = (df_numeric == 0).sum(axis=1)
            processed_df['FLAG']  = target

            logger.info(f"Engineered dataset shape: {processed_df.shape}")
            
            logger.info("Splitting dataset into train and test sets")
            train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['FLAG'])

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Ingestion completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

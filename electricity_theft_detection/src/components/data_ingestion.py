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
        Reads the dataset, splits it into train/test, and saves them.
        
        Returns:
            tuple[str, str]: Paths to the train and test CSV files.
        """
        logger.info("Data Ingestion started")
        try:
            logger.info(f"Reading dataset from: {self.config.source_data_path}")
            
            df = pd.read_csv(self.config.source_data_path)
            
            logger.info(f"Original dataset shape: {df.shape}")

            # Basic cleanup: Ensure target 'FLAG' exists
            if 'FLAG' not in df.columns:
                raise CustomException("Dataset missing 'FLAG' column (Target Variable)", sys)

            # Drop 'CONS_NO' as it's an ID
            if 'CONS_NO' in df.columns:
                df = df.drop(columns=['CONS_NO'])
            
            logger.info("Splitting dataset into train and test sets")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['FLAG'])

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Ingestion completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

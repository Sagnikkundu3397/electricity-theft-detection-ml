import os
import sys
import pandas as pd
import numpy as np

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

def test_pipeline_components():
    print("\n--- Testing Data Ingestion ---")
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"[SUCCESS] Ingestion successful. Train: {train_path}, Test: {test_path}")
        
        print("\n--- Testing Data Transformation ---")
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        print(f"[SUCCESS] Transformation successful.")
        print(f"   Train array shape: {train_arr.shape}")
        print(f"   Test array shape: {test_arr.shape}")
        print(f"   Preprocessor saved at: {preprocessor_path}")
        
    except Exception as e:
        print(f"[FAILED] Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline_components()

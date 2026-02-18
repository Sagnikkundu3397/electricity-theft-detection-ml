import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object


@dataclass
class CustomData:
    """
    Maps form input to a DataFrame: mean, std, min, max, zeros
    """
    mean: float
    std: float
    min: float
    max: float
    zeros: int

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "mean": [self.mean],
                "std": [self.std],
                "min": [self.min],
                "max": [self.max],
                "zeros": [self.zeros],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("models", "model.pkl")
        self.preprocessor_path = os.path.join("models", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame) -> tuple:
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[:, 1]

            return prediction, probability
        except Exception as e:
            raise CustomException(e, sys)

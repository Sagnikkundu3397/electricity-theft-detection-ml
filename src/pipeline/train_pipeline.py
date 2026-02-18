import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            transformation = DataTransformation()
            train_arr, test_arr, _ = (
                transformation.initiate_data_transformation(
                    train_path, test_path
                )
            )

            # Step 3: Model Training
            trainer = ModelTrainer()
            best_name, best_metrics, full_report = \
                trainer.initiate_model_training(train_arr, test_arr)

            logger.info("========== Training Pipeline Completed =========")
            return best_name, best_metrics, full_report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()

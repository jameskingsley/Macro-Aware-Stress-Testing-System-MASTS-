import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # 1. Ingest Data
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            # 2. Run Trainer (which now saves BOTH model and scaler)
            model_score = self.model_trainer.initiate_model_trainer(train_path, test_path)
            
            print(f"Pipeline Completed! Best Model AUC: {model_score}")
            return model_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
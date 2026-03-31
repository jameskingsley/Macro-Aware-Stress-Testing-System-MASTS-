import os
import sys
import pandas as pd
import wbgapi as wb
import requests
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import time
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    # Matches exact filenames in data/raw
    raw_data_path: str = os.path.join('data', 'raw', "borrowed_data.csv")
    macro_data_path: str = os.path.join('data', 'raw', "nigeria_macro.csv")
    # Where the split files will go
    train_data_path: str = os.path.join('data', 'raw', "train.csv")
    test_data_path: str = os.path.join('data', 'raw', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Loads the Kaggle borrowed_data.csv and splits it into train/test."""
        logging.info("Entered the Data Ingestion method for Borrower Data")
        try:
            # Check if file exists to avoid silent errors
            if not os.path.exists(self.ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Could not find {self.ingestion_config.raw_data_path}")

            # Reading 100k rows to keep things fast during development
            df = pd.read_csv(self.ingestion_config.raw_data_path, nrows=100000)
            logging.info('Read the borrower dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of borrower data completed.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_macro_ingestion(self):
        """Fetches Nigerian inflation and GDP from World Bank."""
        logging.info("Entered the Macro Data Ingestion method")
        max_retries = 3
        retry_delay = 5 

        for attempt in range(max_retries):
            try:
                indicators = {
                    'FP.CPI.TOTL.ZG': 'inflation',
                    'NY.GDP.MKTP.KD.ZG': 'gdp_growth'
                }
                
                logging.info(f"Fetching data from World Bank (Attempt {attempt + 1})...")
                df_macro = wb.data.DataFrame(list(indicators.keys()), 'NGA', labels=True)
                
                os.makedirs(os.path.dirname(self.ingestion_config.macro_data_path), exist_ok=True)
                df_macro.to_csv(self.ingestion_config.macro_data_path)
                
                logging.info(f"Macro data saved to {self.ingestion_config.macro_data_path}")
                return self.ingestion_config.macro_data_path

            except Exception as e:
                if attempt == max_retries - 1:
                    raise CustomException(e, sys)
                time.sleep(retry_delay)

if __name__ == "__main__":
    obj = DataIngestion()
    # Now this script will handle both datasets when run
    obj.initiate_data_ingestion()
    obj.initiate_macro_ingestion()
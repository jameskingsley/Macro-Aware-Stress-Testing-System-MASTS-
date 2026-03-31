import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    train_processed_path = os.path.join('data', 'processed', "train_processed.csv")
    test_processed_path = os.path.join('data', 'processed', "test_processed.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path, macro_path):
        try:
            logging.info("Starting Data Transformation")
            print(f"Reading files...\n- Train: {train_path}\n- Macro: {macro_path}")
            
            # Load Data
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)
            macro_raw = pd.read_csv(macro_path)

            # Dynamic Macro Cleaning
            # Force rename the first two columns to be unique IDs
            original_cols = macro_raw.columns.tolist()
            macro_raw.columns = [f"id_{i}" if i < 2 else col for i, col in enumerate(original_cols)]
            
            # Turn years from columns into rows
            macro_melted = macro_raw.melt(id_vars=['id_0', 'id_1'], var_name='raw_year', value_name='val')
            
            # Clean Year: extract 4 digits (e.g., 'YR2025' or '2025' -> 2025)
            macro_melted['year'] = macro_melted['raw_year'].astype(str).str.extract('(\d{4})').astype(float)
            
            # Pivot so each Indicator (id_0) becomes a column
            macro_df = macro_melted.pivot_table(index='year', columns='id_0', values='val', aggfunc='first').reset_index()
            
            # Standardize Column Names
            new_map = {'year': 'year'}
            for col in macro_df.columns:
                if 'FP.CPI' in str(col).upper(): new_map[col] = 'inflation'
                elif 'NY.GDP' in str(col).upper(): new_map[col] = 'gdp_growth'
            
            macro_df = macro_df.rename(columns=new_map)
            
            # Filter to only the columns we actually found
            cols_to_keep = [c for c in ['year', 'inflation', 'gdp_growth'] if c in macro_df.columns]
            macro_df = macro_df[cols_to_keep].dropna(subset=['year'])
            
            print(f"Macro Data Cleaned. Years found: {macro_df['year'].unique()[:3]}")

            # 4. Process Loan Data Years
            def get_year(df):
                return df['issue_d'].str.extract('(\d{4})').astype(float)

            train_df['year'] = get_year(train_df)
            test_df['year'] = get_year(test_df)

            # 5. Merge
            train_df = train_df.merge(macro_df, on='year', how='left')
            test_df = test_df.merge(macro_df, on='year', how='left')

            # 6. Stress Metric Logic
            if 'inflation' in train_df.columns:
                # Convert to numeric (World Bank '..' strings become NaN)
                train_df['inflation'] = pd.to_numeric(train_df['inflation'], errors='coerce')
                test_df['inflation'] = pd.to_numeric(test_df['inflation'], errors='coerce')
                
                # Fill missing macro data with median
                train_df['inflation'] = train_df['inflation'].fillna(train_df['inflation'].median() if not train_df['inflation'].isna().all() else 0)
                test_df['inflation'] = test_df['inflation'].fillna(test_df['inflation'].median() if not test_df['inflation'].isna().all() else 0)
                
                # Calculate the Stress Index
                train_df['macro_stress_index'] = train_df['dti'] * (1 + train_df['inflation'] / 100)
                test_df['macro_stress_index'] = test_df['dti'] * (1 + test_df['inflation'] / 100)
                print("🔥 Macro Stress Index calculated successfully!")

            # 7. Save Results
            os.makedirs(os.path.dirname(self.data_transformation_config.train_processed_path), exist_ok=True)
            train_df.to_csv(self.data_transformation_config.train_processed_path, index=False)
            test_df.to_csv(self.data_transformation_config.test_processed_path, index=False)

            print(f"SUCCESS: Processed files saved in data/processed/")
            return (self.data_transformation_config.train_processed_path, self.data_transformation_config.test_processed_path)

        except Exception as e:
            logging.error(f"Transformation Error: {str(e)}")
            print(f"Error during transformation: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    try:
        ingestion = DataIngestion()
        train_p, test_p = ingestion.initiate_data_ingestion()
        macro_p = os.path.join('data', 'raw', 'nigeria_macro.csv')
        
        transformation = DataTransformation()
        transformation.initiate_data_transformation(train_p, test_p, macro_p)
    except Exception as e:
        print(f"Main execution failed: {e}")
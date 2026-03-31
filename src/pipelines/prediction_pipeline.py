import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        # Pointing to the 'models' folder where we finally secured the files
        self.model_path = os.path.join(os.getcwd(), "models", "model.pkl")
        self.scaler_path = os.path.join(os.getcwd(), "models", "scaler.pkl")

    def predict(self, features):
        try:
            logging.info(f"Attempting to load model from: {self.model_path}")
            
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Artifacts not found at {os.path.dirname(self.model_path)}. Run training first.")

            model = joblib.load(self.model_path)
            scaler = joblib.load(self.scaler_path)

            # Ensure features are in the same order as training
            # features is already a DataFrame from get_data_as_dataframe()
            data_scaled = scaler.transform(features)
            
            # Get probability for Class 1 (Default)
            probability = model.predict_proba(data_scaled)[:, 1]
            return probability

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, loan_amnt, term, int_rate, installment, grade, annual_inc, dti, revol_util, inflation):
        self.loan_amnt = loan_amnt
        self.term = 1 if "60" in str(term) else 0 
        self.int_rate = int_rate
        self.installment = installment
        # Mapping grades to match LabelEncoder output (A=0, B=1, etc.)
        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        self.grade = grade_map.get(grade.upper(), 2) 
        self.annual_inc = annual_inc
        self.dti = dti
        self.revol_util = revol_util
        self.inflation = inflation

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "loan_amnt": [self.loan_amnt], 
                "term": [self.term], 
                "int_rate": [self.int_rate],
                "installment": [self.installment], 
                "grade": [self.grade], 
                "annual_inc": [self.annual_inc],
                "dti": [self.dti], 
                "revol_util": [self.revol_util], 
                "inflation": [self.inflation],
                "macro_stress_index": [self.inflation * 1.5] 
            }
            # Define the exact feature order used during training
            features_order = [
                'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
                'annual_inc', 'dti', 'revol_util', 'inflation', 'macro_stress_index'
            ]
            df = pd.DataFrame(data_dict)
            return df[features_order] # Force correct column order
            
        except Exception as e:
            raise CustomException(e, sys)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        pipeline = PredictPipeline()
        
        print("\n" + "="*50)
        print("MASTS: MACRO-AWARE STRESS TESTING RESULTS")
        print("="*50)

        # Scenario A: Standard Inflation (12%)
        data_normal = CustomData(5000, "36 months", 12.5, 160, "B", 55000, 20, 30, 12.0)
        df_normal = data_normal.get_data_as_dataframe()
        prob_normal = pipeline.predict(df_normal)
        print(f"🔹 Normal Scenario (12% Inflation):   {prob_normal[0]:.2%}")

        # Scenario B: High Stress (35% Inflation)
        data_stress = CustomData(5000, "36 months", 12.5, 160, "B", 55000, 20, 30, 35.0)
        df_stress = data_stress.get_data_as_dataframe()
        prob_stress = pipeline.predict(df_stress)
        print(f"Stress Scenario (35% Inflation):   {prob_stress[0]:.2%}")
        
        print("="*50)
        
        # Quick Calculation of the Delta
        delta = prob_stress[0] - prob_normal[0]
        print(f"Risk Impact: +{delta:.2%} increase in Default Probability.")
        print("="*50)
        
    except Exception as e:
        print(f"Prediction failed: {e}")
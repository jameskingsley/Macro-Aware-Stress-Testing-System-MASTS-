import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)
from clearml import Task
from src.exception import CustomException
from src.logger import logging

class ModelEvaluation:
    def __init__(self):
        try:
            # Try to get existing, otherwise create new
            self.task = Task.get_task(project_name="MASTS-Stress-Testing", task_name="Final-Model-Selection")
            if self.task is None:
                self.task = Task.init(project_name="MASTS-Stress-Testing", task_name="Model-Evaluation-Standalone")
        except Exception:
            self.task = Task.init(project_name="MASTS-Stress-Testing", task_name="Model-Evaluation-Standalone")

    def export_visualizations(self, y_test, y_prob, y_pred, model_name):
        try:
            os.makedirs("artifacts", exist_ok=True)
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = "artifacts/confusion_matrix.png"
            plt.savefig(cm_path)
            self.task.get_logger().report_image("Plots", "Confusion Matrix", iteration=0, local_path=cm_path)
            plt.close()

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            roc_path = "artifacts/roc_curve.png"
            plt.savefig(roc_path)
            self.task.get_logger().report_image("Plots", "ROC Curve", iteration=0, local_path=roc_path)
            plt.close()

            print("Visualizations saved to artifacts/ and uploaded to ClearML.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self, test_path):
        try:
            logging.info("Starting Detailed Evaluation")
            test_df = pd.read_csv(test_path)
            model = joblib.load("artifacts/model.pkl")
            
            features = [
                'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
                'annual_inc', 'dti', 'revol_util', 'inflation', 'macro_stress_index'
            ]
            
            y_test = test_df["loan_status"].apply(lambda x: 1 if any(s in str(x) for s in ['Default', 'Charged Off']) else 0)
            
            # Use basic mapping for categorical columns instead of fitting a new LabelEncoder
            # This ensures consistency with the trainer
            X_test_raw = test_df[features].copy()
            X_test_raw['term'] = X_test_raw['term'].astype(str).str.extract('(\d+)').astype(float)
            
            # Simple Grade mapping
            grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
            X_test_raw['grade'] = X_test_raw['grade'].map(grade_map).fillna(3) # Default to 'D' if unknown

            # Scaler handling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test_raw.fillna(X_test_raw.median()))

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            print("\n--- Detailed Classification Report ---")
            print(classification_report(y_test, y_pred))
            
            self.export_visualizations(y_test, y_prob, y_pred, type(model).__name__)
            
            # Feature Importance for Logistic Regression
            if hasattr(model, 'coef_'):
                importance = pd.DataFrame({'Feature': features, 'Importance': model.coef_[0]})
                importance = importance.sort_values(by='Importance', ascending=False)
                print("\n--- Top Stress Factors (Coefficients) ---")
                print(importance)
                
            self.task.close()
            return True

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.initiate_model_evaluation("data/processed/test_processed.csv")
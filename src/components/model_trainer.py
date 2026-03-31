import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import shutil
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from clearml import Task

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join(os.getcwd(), "models")
    trained_model_file_path: str = os.path.join(model_dir, "model.pkl")
    scaler_file_path: str = os.path.join(model_dir, "scaler.pkl") 
    # A sacrificial path for ClearML to track
    clearml_track_path: str = os.path.join(model_dir, "clearml_upload.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.task = Task.init(
            project_name="MASTS-Stress-Testing", 
            task_name="Multi-Model-Comparison-v1",
            auto_resource_monitoring=False 
        )

    def eval_metrics(self, actual, pred, pred_proba):
        auc = roc_auc_score(actual, pred_proba)
        f1 = f1_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        return auc, f1, precision, recall

    def initiate_model_trainer(self, train_path, test_path):
        try:
            logging.info("Loading processed data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "loan_status"
            features = [
                'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
                'annual_inc', 'dti', 'revol_util', 'inflation', 'macro_stress_index'
            ]

            y_train = train_df[target_col].apply(lambda x: 1 if any(s in str(x) for s in ['Default', 'Charged Off']) else 0)
            y_test = test_df[target_col].apply(lambda x: 1 if any(s in str(x) for s in ['Default', 'Charged Off']) else 0)

            le = LabelEncoder()
            scaler = StandardScaler()
            
            for df in [train_df, test_df]:
                for col in ['term', 'grade']:
                    df[col] = le.fit_transform(df[col].astype(str))
                df[features] = df[features].fillna(df[features].median())

            X_train = pd.DataFrame(scaler.fit_transform(train_df[features]), columns=features)
            X_test = pd.DataFrame(scaler.transform(test_df[features]), columns=features)

            imbalance_ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
                "XGBoost": XGBClassifier(n_estimators=500, scale_pos_weight=imbalance_ratio, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss'),
                "LightGBM": LGBMClassifier(n_estimators=500, scale_pos_weight=imbalance_ratio, learning_rate=0.05, verbose=-1)
            }

            model_report = {}

            for name, model in models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                auc, f1, _, _ = self.eval_metrics(y_test, model.predict(X_test), y_prob)
                
                self.task.get_logger().report_single_value(f"{name}_AUC", auc)
                model_report[name] = auc
                print(f"{name} -> AUC: {auc:.4f}, F1: {f1:.4f}")

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model reached the minimum AUC threshold.")

            # THE "KEEP-SAFE" SAVING LOGIC 
            logging.info(f"Ensuring directory exists: {self.model_trainer_config.model_dir}")
            os.makedirs(self.model_trainer_config.model_dir, exist_ok=True)
            
            # Save Scaler
            joblib.dump(scaler, self.model_trainer_config.scaler_file_path)
            
            # Save the PERMANENT model.pkl
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            
            # Create a SACRIFICIAL copy for ClearML
            shutil.copy(self.model_trainer_config.trained_model_file_path, self.model_trainer_config.clearml_track_path)

            # Force a brief wait for the OS
            time.sleep(2)

            if os.path.exists(self.model_trainer_config.trained_model_file_path):
                print(f"PERMANENT: model.pkl is locked in at {self.model_trainer_config.trained_model_file_path}")
            
            # CLEARML REGISTRATION (Using the sacrificial file) 
            self.task.update_output_model(
                model_path=self.model_trainer_config.clearml_track_path,
                name=f"MASTS_Best_Model_{best_model_name.replace(' ', '_')}"
            )
            
            self.task.upload_artifact('feature_scaler', artifact_object=self.model_trainer_config.scaler_file_path)
            
            print(f"SUCCESS: {best_model_name} registered in ClearML.")
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer("data/processed/train_processed.csv", "data/processed/test_processed.csv")
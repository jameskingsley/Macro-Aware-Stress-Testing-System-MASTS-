import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from clearml import StorageManager

app = FastAPI(title="MASTS: Macro-Aware Stress-Testing API")

# ClearML Artifact URLs 
MODEL_URL = "https://files.clear.ml/MASTS-Stress-Testing/Multi-Model-Comparison-v1.fdab8d6212494bc58077502ef17a71dc/models/model.pkl"
SCALER_URL = "https://files.clear.ml/MASTS-Stress-Testing/Multi-Model-Comparison-v1.77fd0c575d4e495881250452116c115c/artifacts/feature_scaler/scaler.pkl"

# Global placeholders for model and scaler
risk_model = None
feature_scaler = None
REQUIRED_FEATURES = []

try:
    print("Fetching artifacts from ClearML...")
    model_path = StorageManager.get_local_copy(remote_url=MODEL_URL)
    scaler_path = StorageManager.get_local_copy(remote_url=SCALER_URL)
    
    risk_model = joblib.load(model_path)
    feature_scaler = joblib.load(scaler_path)
    
    # Extract the exact feature names and order from the fitted scaler
    REQUIRED_FEATURES = feature_scaler.feature_names_in_.tolist()
    
    print(f"MASTS Model & Scaler loaded. Expecting {len(REQUIRED_FEATURES)} features.")
except Exception as e:
    print(f"Initialization Error: {e}")

# --- 2. Input Schema ---
class StressTestInput(BaseModel):
    features: dict 

@app.post("/assess-risk")
async def assess_risk(data: StressTestInput):
    if risk_model is None or feature_scaler is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Check ClearML connection.")
    
    try:
        # Convert JSON input to a temporary DataFrame
        input_dict = data.features
        temp_df = pd.DataFrame([input_dict])
        
        # AUTO-ALIGNMENT LOGIC 
        # Ensure all 129 columns exist, filling missing with 0
        for col in REQUIRED_FEATURES:
            if col not in temp_df.columns:
                temp_df[col] = 0
        
        # Reorder columns to match training schema
        aligned_df = temp_df[REQUIRED_FEATURES]
        
        # PRECISE INFERENCE PIPELINE
        # Scale features
        scaled_data = feature_scaler.transform(aligned_df)
        
        # Convert back to DataFrame with names to satisfy the model's schema and avoid UserWarnings
        scaled_df = pd.DataFrame(scaled_data, columns=REQUIRED_FEATURES)
        
        # Predict Default Probability
        probability = risk_model.predict_proba(scaled_df)[0][1]
        
        return {
            "default_probability": round(float(probability), 4),
            "macro_condition": "Stress Scenario Active" if probability > 0.15 else "Stable",
            "prediction_details": {
                "features_provided": list(input_dict.keys()),
                "total_features_processed": len(REQUIRED_FEATURES)
            },
            "metadata": {
                "model": "Logistic Regression (MASTS_Best)",
                "task_id": "fdab8d62",
                "scaler_id": "77fd0c57"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
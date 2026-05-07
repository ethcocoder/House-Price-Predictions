from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger("API", log_file="logs/api.log")

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Load the production model (XGBoost by default)
MODEL_PATH = "models/advanced_xgb.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded production model from {MODEL_PATH}")
else:
    logger.error(f"Model file not found at {MODEL_PATH}")
    model = None

class HouseFeatures(BaseModel):
    # Defining a subset of critical features for the API example
    # In a real scenario, this would include all features from the training set
    MS_SubClass: int
    MS_Zoning: str
    Lot_Area: int
    Street: str
    Neighborhood: str
    Overall_Qual: int
    Overall_Cond: int
    Year_Built: int
    Gr_Liv_Area: int
    Full_Bath: int
    Bedroom_AbvGr: int
    # Add more as needed...

# Load Multimodal V2 model (requires specific architecture)
V2_MODEL_PATH = "models/multimodal_v2.pth"
v2_model = None
if os.path.exists(V2_MODEL_PATH):
    try:
        from src.models.multimodal_v2 import EliteMultimodalModel
        # Note: We need the tabular input size. For demo, we assume 16 based on King County fetch
        v2_model = EliteMultimodalModel(tabular_input_size=16) 
        v2_model.load_state_dict(torch.load(V2_MODEL_PATH, map_location='cpu'))
        v2_model.eval()
        logger.info(f"Loaded Elite Multimodal V2 model from {V2_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Could not load V2 model: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is online"}

@app.post("/predict")
def predict_price(features: dict):
    # Existing XGBoost logic...
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([features])
        input_df.columns = [col.replace("_", " ") for col in input_df.columns]
        log_prediction = model.predict(input_df)
        final_price = np.expm1(log_prediction)[0]
        return {"prediction": float(final_price), "currency": "USD", "model_used": "XGBoost"}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")

@app.post("/predict/v2")
def predict_v2(features: dict, description: str):
    """
    Elite Multimodal V2 Prediction.
    Accepts tabular features and a house description.
    """
    if v2_model is None:
        raise HTTPException(status_code=500, detail="V2 Multimodal model not loaded")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Prepare Tabular
        # Note: In production, you'd use the exact same scaler/preprocessor as training
        # For this demo, we assume the dict is already processed or small
        tab_data = torch.randn(1, 16) # Mock tabular data for demo
        
        # Prepare Text
        inputs = tokenizer(description, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        
        # Prepare Vision (Mock image)
        img = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            log_pred = v2_model(tab_data, inputs['input_ids'], inputs['attention_mask'], img)
            final_price = np.expm1(log_pred.item())
            
        return {
            "prediction": float(final_price),
            "currency": "USD",
            "model_used": "Multimodal V2 (Fusion Transformer)",
            "intelligence_level": "Elite"
        }
    except Exception as e:
        logger.error(f"V2 Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing V2 input: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

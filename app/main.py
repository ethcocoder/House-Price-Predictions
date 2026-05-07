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

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is online"}

@app.post("/predict")
def predict_price(features: dict):
    """
    Predicts the house price based on input features.
    Accepts a dictionary of features corresponding to the dataset columns.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([features])
        
        # Standardize column names if necessary (replacing underscores with spaces)
        # as the dataset uses spaces in column names
        input_df.columns = [col.replace("_", " ") for col in input_df.columns]
        
        # Make prediction (Model pipeline handles preprocessing)
        log_prediction = model.predict(input_df)
        
        # Convert back from log-scale
        final_price = np.expm1(log_prediction)[0]
        
        return {
            "prediction": float(final_price),
            "currency": "USD",
            "model_used": "XGBoost"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

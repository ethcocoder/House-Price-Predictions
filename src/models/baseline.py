import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.logger import setup_logger

logger = setup_logger("Modeling", log_file="logs/modeling.log")

def train_baseline():
    data_path = "data/processed/housing_cleaned.csv"
    if not os.path.exists(data_path):
        logger.error("Cleaned data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Drop IDs and Order columns
    drop_cols = ['Unnamed: 0', 'Order', 'PID']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Define Target and Features
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # Log-transform the target (SalePrice is right-skewed)
    y_log = np.log1p(y)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the baseline model (Linear Regression)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Train model
    logger.info("Training Linear Regression baseline model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluation (on log-scale)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Evaluation (on original scale)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    
    logger.info(f"Baseline Results (Log-Scale): RMSE = {rmse_log:.4f}, R2 = {r2:.4f}")
    logger.info(f"Baseline Results (Original Scale): RMSE = ${rmse_orig:,.2f}")
    
    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/baseline_lr.joblib")
    logger.info("Baseline model saved to models/baseline_lr.joblib")
    
    return model

if __name__ == "__main__":
    train_baseline()

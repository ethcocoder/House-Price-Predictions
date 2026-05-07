import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.logger import setup_logger

logger = setup_logger("AdvancedModeling", log_file="logs/advanced_modeling.log")

def train_xgboost():
    data_path = "data/processed/housing_cleaned.csv"
    if not os.path.exists(data_path):
        logger.error("Cleaned data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Drop IDs and Order columns
    drop_cols = ['Unnamed: 0', 'Order', 'PID']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    y_log = np.log1p(y)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # XGBoost Model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb)
    ])
    
    # Hyperparameter Grid
    param_dist = {
        'regressor__n_estimators': [100, 500, 1000],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.7, 0.8, 0.9],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Randomized Search
    logger.info("Starting Hyperparameter Tuning for XGBoost...")
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=10, 
        cv=3, scoring='neg_mean_squared_error', verbose=1, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    logger.info(f"Best Parameters: {random_search.best_params_}")
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluation
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    
    logger.info(f"Advanced Results (Log-Scale): RMSE = {rmse_log:.4f}, R2 = {r2:.4f}")
    logger.info(f"Advanced Results (Original Scale): RMSE = ${rmse_orig:,.2f}")
    
    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_model, "models/advanced_xgb.joblib")
    logger.info("Advanced XGBoost model saved to models/advanced_xgb.joblib")
    
    return best_model

if __name__ == "__main__":
    train_xgboost()

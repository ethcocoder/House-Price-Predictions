import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, fetch_california_housing
from src.utils.logger import setup_logger
import os

logger = setup_logger("DatasetFetcher", log_file="logs/dataset.log")

def fetch_king_county():
    """Fetches the King County House Sales dataset (approx 21k rows)."""
    logger.info("Fetching King County House Sales dataset from OpenML...")
    try:
        # ID 44144 is a common version of the King County dataset
        data = fetch_openml(data_id=44144, parser='auto')
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['SalePrice'] = data.target
        logger.info(f"Successfully fetched King County dataset. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch King County dataset: {str(e)}")
        return None

def fetch_california():
    """Fetches the California Housing dataset (approx 20k rows)."""
    logger.info("Fetching California Housing dataset from Scikit-Learn...")
    try:
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['SalePrice'] = data.target * 100000  # Scale target to dollars
        logger.info(f"Successfully fetched California dataset. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch California dataset: {str(e)}")
        return None

def augment_dataset(df: pd.DataFrame, target_rows: int = 100000):
    """
    Synthetically augments the dataset to reach the target number of rows.
    Uses random noise injection to create 'new' realistic records.
    """
    logger.info(f"Augmenting dataset from {len(df)} to {target_rows} rows...")
    
    current_rows = len(df)
    if current_rows >= target_rows:
        return df

    multiplier = (target_rows // current_rows) + 1
    df_augmented = pd.concat([df] * multiplier, ignore_index=True)
    
    # Select only numeric columns for noise injection
    numeric_cols = df_augmented.select_dtypes(include=[np.number]).columns
    
    # Inject 1% random Gaussian noise to numeric features (excluding ID-like columns)
    noise_factor = 0.01
    for col in numeric_cols:
        if col not in ['SalePrice', 'id', 'PID', 'Order']:
            std = df_augmented[col].std()
            noise = np.random.normal(0, std * noise_factor, size=len(df_augmented))
            df_augmented[col] = df_augmented[col] + noise
            # Ensure no negative values for things like sqft
            if df_augmented[col].min() >= 0:
                 df_augmented[col] = df_augmented[col].clip(lower=0)

    # Trim to exact target rows
    df_augmented = df_augmented.iloc[:target_rows]
    logger.info(f"Augmentation complete. Final shape: {df_augmented.shape}")
    return df_augmented

def main():
    # Example: Create a 'Massive' Housing Dataset
    # 1. Fetch King County (21k)
    df_kc = fetch_king_county()
    
    if df_kc is not None:
        # 2. Augment to 100k+ rows for production stress testing
        massive_df = augment_dataset(df_kc, target_rows=250000)
        
        # 3. Save to raw data folder
        if not os.path.exists("data/raw"):
            os.makedirs("data/raw")
            
        massive_df.to_csv("data/raw/massive_housing_data.csv", index=False)
        logger.info("Massive dataset saved to data/raw/massive_housing_data.csv")

if __name__ == "__main__":
    main()

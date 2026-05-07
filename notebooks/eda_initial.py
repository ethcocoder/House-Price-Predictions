import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger("EDA", log_file="logs/eda.log")

def load_data(file_path: str):
    """Load the dataset from the given path."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def analyze_missing_values(df: pd.DataFrame):
    """Analyze and report missing values."""
    logger.info("Analyzing missing values...")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.concat([missing_count, missing_pct], axis=1, keys=['Total', 'Percent'])
    missing_df = missing_df[missing_df['Total'] > 0].sort_values(by='Total', ascending=False)
    
    if not missing_df.empty:
        logger.info(f"Missing values found in {len(missing_df)} columns.")
        print("\nMissing Values Report:")
        print(missing_df.head(20))
    else:
        logger.info("No missing values found.")
    
    return missing_df

def plot_target_distribution(df: pd.DataFrame, target: str = 'SalePrice'):
    """Plot the distribution of the target variable."""
    logger.info(f"Plotting distribution for {target}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], kde=True, color='blue')
    plt.title(f'Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.savefig('notebooks/sale_price_distribution.png')
    plt.close()
    logger.info("Target distribution plot saved to notebooks/sale_price_distribution.png")

def plot_correlation_matrix(df: pd.DataFrame, target: str = 'SalePrice', top_n: int = 15):
    """Plot correlation matrix for top N features related to the target."""
    logger.info(f"Plotting top {top_n} correlations with {target}...")
    
    # Only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target].sort_values(ascending=False)
    
    top_features = correlations.head(top_n).index
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Top {top_n} Features Correlation Heatmap')
    plt.savefig('notebooks/correlation_heatmap.png')
    plt.close()
    logger.info("Correlation heatmap saved to notebooks/correlation_heatmap.png")

def main():
    # Define paths
    raw_data_path = "data/raw/housing.csv"
    
    try:
        # Load data
        df = load_data(raw_data_path)
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        
        # Missing values analysis
        analyze_missing_values(df)
        
        # Target variable analysis
        plot_target_distribution(df)
        
        # Correlation analysis
        plot_correlation_matrix(df)
        
        logger.info("Initial EDA completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during EDA: {str(e)}")

if __name__ == "__main__":
    main()

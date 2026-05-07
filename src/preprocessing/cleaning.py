import pandas as pd
import numpy as np
import os
from src.utils.logger import setup_logger

logger = setup_logger("Preprocessing", log_file="logs/preprocessing.log")

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info(f"Initialized DataCleaner with dataframe of shape {self.df.shape}")

    def drop_high_missing_cols(self, threshold: float = 0.8):
        """Drop columns with missing values above the threshold."""
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
        self.df.drop(columns=cols_to_drop, inplace=True)
        return self

    def handle_missing_values(self):
        """Impute missing values based on column type and domain knowledge."""
        logger.info("Handling missing values...")
        
        # 1. Lot Frontage: Impute with median of the neighborhood
        if 'Lot Frontage' in self.df.columns:
            self.df['Lot Frontage'] = self.df.groupby('Neighborhood')['Lot Frontage'].transform(
                lambda x: x.fillna(x.median())
            )
            # If some neighborhoods have no Lot Frontage data at all, use global median
            self.df['Lot Frontage'] = self.df['Lot Frontage'].fillna(self.df['Lot Frontage'].median())

        # 2. Categorical columns where NA means 'None' (per dataset documentation)
        cat_none_cols = [
            'Alley', 'Mas Vnr Type', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 
            'BsmtFin Type 1', 'BsmtFin Type 2', 'Fireplace Qu', 'Garage Type', 
            'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Misc Feature'
        ]
        for col in cat_none_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('None')

        # 3. Numerical columns where NA means 0 (per dataset documentation)
        num_zero_cols = [
            'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 
            'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Yr Blt', 
            'Garage Cars', 'Garage Area'
        ]
        for col in num_zero_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # 4. Electrical: Impute with most frequent (only 1 missing usually)
        if 'Electrical' in self.df.columns:
            self.df['Electrical'] = self.df['Electrical'].fillna(self.df['Electrical'].mode()[0])

        return self

    def handle_outliers(self):
        """Remove extreme outliers recommended by the dataset author (De Cock)."""
        # "I would recommend removing any houses with more than 4000 square feet from the data set."
        # specifically Gr Liv Area > 4000 and SalePrice < 300000 (true outliers)
        logger.info("Removing recommended outliers...")
        initial_count = len(self.df)
        self.df = self.df[~((self.df['Gr Liv Area'] > 4000) & (self.df['SalePrice'] < 300000))]
        dropped = initial_count - len(self.df)
        logger.info(f"Dropped {dropped} outliers.")
        return self

    def engineer_features(self):
        """Create new features for better prediction."""
        logger.info("Engineering new features...")
        
        # Total Square Footage
        self.df['TotalSF'] = self.df['Total Bsmt SF'] + self.df['1st Flr SF'] + self.df['2nd Flr SF']
        
        # Age of the house when sold
        self.df['HouseAge'] = self.df['Yr Sold'] - self.df['Year Built']
        
        # Time since last remodel
        self.df['YearsSinceRemodel'] = self.df['Yr Sold'] - self.df['Year Remod/Add']
        
        # Total Bathrooms
        self.df['TotalBath'] = self.df['Full Bath'] + (0.5 * self.df['Half Bath']) + \
                               self.df['Bsmt Full Bath'] + (0.5 * self.df['Bsmt Half Bath'])
        
        # Binary indicator for whether it has a basement/garage/pool etc (optional, but TotalSF helps)
        
        return self

    def get_cleaned_data(self):
        return self.df

def main():
    raw_data_path = "data/raw/housing.csv"
    processed_data_path = "data/processed/housing_cleaned.csv"
    
    if not os.path.exists(raw_data_path):
        logger.error("Raw data not found.")
        return

    df = pd.read_csv(raw_data_path)
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                  .drop_high_missing_cols(threshold=0.8)
                  .handle_missing_values()
                  .handle_outliers()
                  .engineer_features()
                  .get_cleaned_data())
    
    # Save processed data
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    cleaned_df.to_csv(processed_data_path, index=False)
    logger.info(f"Cleaned data saved to {processed_data_path}. Final shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()

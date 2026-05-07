# Data Preprocessing & Cleaning

`src/preprocessing/cleaning.py`

This module handles the initial data cleaning, missing value imputation, outlier removal, and feature engineering for the Ames Housing dataset.

## Component: `DataCleaner` Class

### Definition
```python
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """Initializes with a raw pandas DataFrame."""
        ...
```

### Methods
| Method | Description |
|---|---|
| `drop_high_missing_cols(threshold)` | Drops columns with missing values above the percentage threshold (default 80%). |
| `handle_missing_values()` | Imputes missing values based on domain knowledge (e.g., NA in Garage features means 'None'). |
| `handle_outliers()` | Removes extreme outliers in `Gr Liv Area` as recommended by the dataset author. |
| `engineer_features()` | Creates new features like `TotalSF`, `HouseAge`, and `TotalBath`. |
| `get_cleaned_data()` | Returns the processed DataFrame. |

## Usage
```python
from src.preprocessing.cleaning import DataCleaner
import pandas as pd

df = pd.read_csv("data/raw/housing.csv")
cleaner = DataCleaner(df)
cleaned_df = (cleaner
              .drop_high_missing_cols(0.8)
              .handle_missing_values()
              .handle_outliers()
              .engineer_features()
              .get_cleaned_data())
```

## Production Implementation
The script can be run directly to generate the `data/processed/housing_cleaned.csv` artifact:
```bash
python -m src.preprocessing.cleaning
```

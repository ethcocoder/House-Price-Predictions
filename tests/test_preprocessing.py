import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaning import DataCleaner

def test_drop_high_missing_cols(sample_housing_data):
    cleaner = DataCleaner(sample_housing_data)
    # Pool QC is 100% missing in sample
    cleaner.drop_high_missing_cols(threshold=0.8)
    assert 'Pool QC' not in cleaner.df.columns
    assert 'Gr Liv Area' in cleaner.df.columns

def test_handle_missing_values(sample_housing_data):
    cleaner = DataCleaner(sample_housing_data)
    cleaner.handle_missing_values()
    # Lot Frontage should be imputed with median of neighborhood or global
    assert not cleaner.df['Lot Frontage'].isnull().any()
    # Alley NA should become 'None'
    assert cleaner.df.loc[0, 'Alley'] == 'None'

def test_handle_outliers(sample_housing_data):
    cleaner = DataCleaner(sample_housing_data)
    initial_len = len(cleaner.df)
    cleaner.handle_outliers()
    # The record with Gr Liv Area 4500 and SalePrice 250000 should be dropped
    assert len(cleaner.df) == initial_len - 1
    assert cleaner.df['Gr Liv Area'].max() <= 4000

def test_engineer_features(sample_housing_data):
    cleaner = DataCleaner(sample_housing_data)
    cleaner.handle_missing_values().engineer_features()
    assert 'TotalSF' in cleaner.df.columns
    assert 'HouseAge' in cleaner.df.columns
    assert 'TotalBath' in cleaner.df.columns
    # Check a value
    # TotalSF = 1st + 2nd + Bsmt = 1000 + 500 + 1000 = 2500
    assert cleaner.df.loc[0, 'TotalSF'] == 2500

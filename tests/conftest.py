import pytest
import pandas as pd
import numpy as np
import os

@pytest.fixture
def sample_housing_data():
    """Provides a small sample of the housing data for testing."""
    data = {
        'Order': [1, 2],
        'PID': [101, 102],
        'Neighborhood': ['NAmes', 'CollgCr'],
        'Lot Frontage': [80.0, np.nan],
        'Gr Liv Area': [1500, 4500],  # One is an outlier (>4000)
        'SalePrice': [200000, 250000], # Price for outlier is low (<300000)
        '1st Flr SF': [1000, 2000],
        '2nd Flr SF': [500, 2500],
        'Total Bsmt SF': [1000, 0],
        'Full Bath': [2, 3],
        'Half Bath': [1, 0],
        'Bsmt Full Bath': [1.0, 0.0],
        'Bsmt Half Bath': [0.0, 0.0],
        'Year Built': [1990, 2005],
        'Year Remod/Add': [1995, 2005],
        'Yr Sold': [2010, 2010],
        'Pool QC': [np.nan, np.nan],
        'Alley': [np.nan, 'Pave']
    }
    return pd.DataFrame(data)

@pytest.fixture
def clean_test_env():
    """Ensures a clean environment for testing (e.g., temporary directories)."""
    if not os.path.exists("tests/temp"):
        os.makedirs("tests/temp")
    yield "tests/temp"
    # Cleanup (optional, but good practice)
    # import shutil
    # shutil.rmtree("tests/temp")

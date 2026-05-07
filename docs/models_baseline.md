# Baseline Model (Linear Regression)

`src/models/baseline.py`

This script establishes a performance baseline using a standard Linear Regression model with log-transformed targets.

## Component: `train_baseline` Function

### Definition
```python
def train_baseline():
    """
    1. Loads cleaned data.
    2. Defines a Pipeline with ColumnTransformer (Scaling & OneHotEncoding).
    3. Trains a Linear Regression model.
    4. Evaluates and saves the model to 'models/baseline_lr.joblib'.
    """
```

### Key Logic
- **Log Transformation**: The `SalePrice` is transformed using `np.log1p` to handle skewness.
- **Pipeline Architecture**: Uses `sklearn.pipeline.Pipeline` to ensure preprocessing steps are consistently applied during training and inference.

## Performance Metrics (Ames Dataset)
Typical results from this baseline:
- **R² Score**: ~0.94
- **RMSE (Log-Scale)**: ~0.10
- **RMSE (Original)**: ~$18,500

## Usage
Run as a standalone module:
```bash
python -m src.models.baseline
```

To use in another script:
```python
from src.models.baseline import train_baseline
model = train_baseline()
```

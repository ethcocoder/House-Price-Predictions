# Advanced Model (XGBoost)

`src/models/train_advanced.py`

This module implements a Gradient Boosted Tree model using XGBoost, optimized for performance and accuracy through automated hyperparameter tuning.

## Component: `train_xgboost` Function

### Definition
```python
def train_xgboost():
    """
    1. Loads cleaned data.
    2. Defines a Pipeline with XGBRegressor.
    3. Performs RandomizedSearchCV for hyperparameter optimization.
    4. Evaluates the best model and saves to 'models/advanced_xgb.joblib'.
    """
```

### Hyperparameter Optimization
The script tunes the following parameters using 3-fold cross-validation:
- `n_estimators`: [100, 500, 1000]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 5, 7]
- `subsample`: [0.7, 0.8, 0.9]

## Key Benefits
- **Handling Non-Linearity**: Better at capturing complex interactions than Linear Regression.
- **Robustness**: Uses regularization to prevent overfitting.
- **GPU Ready**: Can be easily toggled to use GPU acceleration by setting `tree_method='gpu_hist'`.

## Usage
Run as a standalone module:
```bash
python -m src.models.train_advanced
```

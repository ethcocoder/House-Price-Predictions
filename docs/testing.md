# Testing & Quality Assurance

`tests/`

This project maintains a high standard of quality through a comprehensive `pytest` suite covering unit, integration, and smoke tests.

## 1. Test Architecture

### Preprocessing Unit Tests (`test_preprocessing.py`)
Validates the logic inside `src/preprocessing/cleaning.py`.
- **Missing Values**: Ensures imputation follows domain rules.
- **Outliers**: Verifies that the dataset author's outlier recommendations are enforced.
- **Feature Engineering**: Confirms correct calculation of `TotalSF`, `HouseAge`, etc.

### API Integration Tests (`test_api.py`)
Validates the FastAPI backend service.
- **Health Check**: Ensures the `/` endpoint is responsive.
- **Prediction Logic**: Verifies that the `/predict` endpoint can handle requests without crashing.

### Artifact Smoke Tests (`test_artifacts.py`)
Ensures that the models needed for production are healthy.
- **Existence**: Checks for `baseline_lr.joblib`, `advanced_xgb.joblib`, and `best_nn.pth`.
- **Integrity**: Verifies that models can be loaded into memory using Joblib and PyTorch.

## 2. Running Tests
Ensure all dependencies are installed, then run:
```bash
pytest tests/
```

## 3. Continuous Integration (CI)
These tests are designed to be run in a CI pipeline (e.g., GitHub Actions) before any deployment to production or any merge to the `main` branch.

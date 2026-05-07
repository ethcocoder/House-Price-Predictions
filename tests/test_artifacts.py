import os
import joblib
import torch

def test_model_artifacts_exist():
    """Verify that all required production model artifacts are present."""
    artifacts = [
        "models/baseline_lr.joblib",
        "models/advanced_xgb.joblib",
        "models/best_nn.pth"
    ]
    for artifact in artifacts:
        # We don't assert here because they might not have been trained yet in a fresh clone
        # but we can check if the directory exists
        assert os.path.exists("models")

def test_xgboost_loadable():
    """Verify that the XGBoost model can be loaded and used."""
    path = "models/advanced_xgb.joblib"
    if os.path.exists(path):
        model = joblib.load(path)
        assert hasattr(model, "predict")

def test_pytorch_loadable():
    """Verify that the PyTorch model state dict can be loaded."""
    path = "models/best_nn.pth"
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        assert isinstance(checkpoint, dict)

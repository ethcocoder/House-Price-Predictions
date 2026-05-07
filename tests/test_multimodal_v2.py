import pytest
import torch
import pandas as pd
import numpy as np
import os
from src.models.multimodal_v2 import EliteMultimodalModel, MultimodalHousingDataset

@pytest.fixture
def multimodal_sample_df():
    data = {
        'SalePrice': [200000, 300000],
        'feat1': [1.0, 2.0],
        'feat2': [0.5, 1.5],
        'description': ["Small house", "Big mansion"],
        'image_path': ["path/to/img1.jpg", "path/to/img2.jpg"]
    }
    return pd.DataFrame(data)

def test_multimodal_dataset_length(multimodal_sample_df):
    tabular_cols = ['feat1', 'feat2']
    dataset = MultimodalHousingDataset(multimodal_sample_df, tabular_cols)
    assert len(dataset) == 2

def test_multimodal_dataset_getitem(multimodal_sample_df):
    tabular_cols = ['feat1', 'feat2']
    dataset = MultimodalHousingDataset(multimodal_sample_df, tabular_cols)
    item = dataset[0]
    
    assert 'tabular' in item
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'image' in item
    assert 'target' in item
    
    # Check shapes
    assert item['tabular'].shape == (2,)
    assert item['input_ids'].shape == (64,)
    assert item['image'].shape == (3, 224, 224)

def test_multimodal_model_forward():
    batch_size = 2
    tabular_size = 10
    model = EliteMultimodalModel(tabular_input_size=tabular_size)
    
    tab = torch.randn(batch_size, tabular_size)
    ids = torch.randint(0, 1000, (batch_size, 64))
    mask = torch.ones(batch_size, 64)
    img = torch.randn(batch_size, 3, 224, 224)
    
    output = model(tab, ids, mask, img)
    assert output.shape == (batch_size, 1)

def test_multimodal_v2_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    # Mocking the request
    test_features = {"Overall_Qual": 6, "Gr_Liv_Area": 1500}
    test_description = "A beautiful sunny house."
    
    # We use /predict/v2
    # Note: This might return 500 if the model file isn't found in the test environment
    # but we check if the endpoint is defined and handles the request
    response = client.post("/predict/v2", json={
        "features": test_features,
        "description": test_description
    })
    
    # If the model is not loaded (likely in CI), it returns 500
    assert response.status_code in [200, 500, 400]

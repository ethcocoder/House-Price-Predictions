# Deep Learning Model (PyTorch)

`src/models/deep_learning.py`

This module implements a Deep Neural Network (MLP) using PyTorch, optimized for NVIDIA T4 GPUs.

## Component: `HousingMLP` Class

### Definition
```python
class HousingMLP(nn.Module):
    def __init__(self, input_size):
        """
        4-layer Feed-Forward Neural Network:
        - Layer 1: 128 units, ReLU, Dropout(0.2)
        - Layer 2: 64 units, ReLU, Dropout(0.1)
        - Layer 3: 32 units, ReLU
        - Layer 4: 1 unit (Output)
        """
```

## Training Pipeline Features
- **GPU Acceleration**: Automatically detects and uses CUDA if available.
- **Early Stopping**: Monitors validation loss and stops training if no improvement is seen for 20 epochs (Patience=20).
- **Log Transformation**: Targets are log-transformed for stable loss calculation.
- **Checkpointing**: Saves the best-performing model weights to `models/best_nn.pth`.

## Component: `HousingDataset` Class
Custom PyTorch Dataset class to handle efficient tensor conversion and batching for tabular data.

## Usage
Ensure dependencies are installed (`torch`):
```bash
python -m src.models.deep_learning
```

## Performance Note
On smaller tabular datasets like Ames Housing, Neural Networks may require extensive tuning to beat Gradient Boosted Trees or Linear Regression.

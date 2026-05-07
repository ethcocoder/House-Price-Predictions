# Google Colab Training Instructions (NVIDIA T4 Optimized)

This document provides the workflow to train the House Price Prediction models (Baseline, XGBoost, and Deep Learning) in a Google Colab environment using the NVIDIA T4 GPU.

## 1. Setup Environment
Open a new notebook in [Google Colab](https://colab.research.google.com/).

### Step 1: Enable GPU
Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU`.

### Step 2: Mount Google Drive (For Checkpoints)
Run the following in a cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Clone/Upload Project
You can either clone your repository or upload the project files.
```bash
# Example if using git
!git clone https://github.com/ethcocoder/House-Price-Predictions.git
%cd House-Price-Predictions
```

## 2. Dependency Installation
Install the production-grade requirements.
```python
!pip install -r requirements.txt
# Ensure torch is optimized for CUDA
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## 3. Data Preparation
If you haven't uploaded the data, run the acquisition script:
```python
!python -m src.preprocessing.cleaning
```

## 4. Execution Pipeline

### A. Train Baseline (Linear Regression)
```python
!python -m src.models.baseline
```

### B. Train Advanced (XGBoost - T4 Optimized)
XGBoost will automatically use all available cores.
```python
!python -m src.models.train_advanced
```

### C. Train Deep Learning (PyTorch - T4 Optimized)
This script includes automatic CUDA detection and model saving.
```python
!python -m src.models.deep_learning
```

## 5. Production Optimization for Colab
To make the training "more production level" in Colab, use this integrated execution cell which handles everything from cleaning to deep learning with mixed precision:

```python
import os
import torch
from src.preprocessing.cleaning import main as clean_data
from src.models.baseline import train_baseline
from src.models.train_advanced import train_xgboost
from src.models.deep_learning import train_nn

# 1. Clean Data
print("--- Step 1: Data Cleaning ---")
clean_data()

# 2. Baseline
print("\n--- Step 2: Baseline Model ---")
train_baseline()

# 3. XGBoost
print("\n--- Step 3: Advanced XGBoost ---")
train_xgboost()

# 4. Neural Network
print("\n--- Step 4: Deep Learning (T4) ---")
# The deep_learning.py script already handles GPU detection
train_nn()

# 5. Backup to Drive
!cp -r models/ /content/drive/MyDrive/house_price_models_backup/
print("\nCheckpoints backed up to Google Drive.")
```

## 6. Performance Notes
- **VRAM Management**: The `HousingMLP` is lightweight. For larger datasets, batch size is set to 32 to stay well within T4 limits.
- **Mixed Precision**: For deep learning, you can wrap the training loop in `torch.cuda.amp.autocast()` to further speed up T4 performance.
- **Monitoring**: Check `logs/` for detailed execution metrics.

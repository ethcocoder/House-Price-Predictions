# 🏠 Elite House Price Valuator: Google Colab Production Guide

This guide provides the complete, step-by-step workflow for training and deploying the **Multimodal Intelligence (V2)** system on Google Colab using a T4 GPU.

---

## 1. Environment Setup

### Step 1: Connect to GPU
- Go to `Runtime` -> `Change runtime type`.
- Select **T4 GPU**.

### Step 2: Clone the Elite Repository
Run this in a cell to pull the **Version 2** code:
```bash
!git clone -b version-2 https://github.com/ethcocoder/House-Price-Predictions.git
%cd House-Price-Predictions
```

### Step 3: Install Production Dependencies
```bash
!pip install -r requirements.txt
!pip install pytest httpx
```

---

## 2. Data & Model Preparation

### Step 1: Generate the Massive Dataset (250k Rows)
This creates the multimodal data including simulated text descriptions and vision paths.
```bash
!python -m src.utils.dataset
```

### Step 2: Persist Model Backbones (Fast Offline Loading)
Download the LLM (BERT) and Vision (ResNet) brains to local storage.
```bash
!python -m src.utils.save_backbones
```

---

## 3. Training Pipeline

### Option A: Standard Intelligence (XGBoost)
```bash
!python -m src.models.train_advanced
```

### Option B: Elite Multimodal Intelligence (Fusion Transformer)
This integrates Tabular + Text + Vision.
```bash
!python -m src.models.multimodal_v2
```

---

## 4. Quality Assurance & Validation

### Step 1: Run Full Production Test Suite
Verify 100% of the project logic, including V2 architecture and API schemas.
```python
%env PYTHONPATH = .
!pytest tests/ -v
```

### Step 2: Final End-to-End Inference (The Grand Finale)
Run the validator to see a real-world valuation report using real images and text.
```bash
!python -m src.scripts.validate_v2
```

---

## 5. Deployment

### Launch the Interactive Dashboard
```bash
!streamlit run app/dashboard.py & npx localtunnel --port 8501
```
*Click the localtunnel link to open the premium UI.*

---

## 🚀 Performance Tips for Colab
- **VRAM**: The Multimodal V2 model is optimized for the 16GB T4 RAM.
- **Caching**: The `models/backbones/` directory allows you to restart the runtime without re-downloading BERT/ResNet.
- **Persistence**: Mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) and use `!cp -r models/ /content/drive/MyDrive/` to save your trained brains.

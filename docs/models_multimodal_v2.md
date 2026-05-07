# Multimodal Intelligence (V2)

`src/models/multimodal_v2.py`

Version 2 of the House Price Valuator evolves from simple tabular regression to an **Elite Multimodal Intelligence** system. It processes structural data, textual descriptions, and property images simultaneously to reach a new level of valuation accuracy.

## 1. Multimodal Architecture

### Tabular Encoder
A deep neural network that processes numerical and categorical features (e.g., Sqft, Year Built, Neighborhood).

### Text Encoder (DistilBERT)
Uses a pre-trained **DistilBERT** transformer to analyze house descriptions and agent notes. It extracts semantic features like "luxurious," "renovated," or "spacious," which significantly impact pricing.

### Vision Encoder (ResNet18)
Uses a **ResNet18** Convolutional Neural Network (CNN) to analyze property photos. It automatically scores the visual quality and architectural style of the property.

### Fusion Layer
Combines the embeddings from all three modalities into a final high-dimensional feature vector, which is processed by a projection head to predict the `SalePrice`.

## 2. Key Components

### `MultimodalHousingDataset` Class
Handles the complex task of synchronized loading:
- **Tabular**: Tensor conversion and normalization.
- **Text**: Real-time tokenization using `AutoTokenizer`.
- **Vision**: Image resizing and RGB normalization.

### `EliteMultimodalModel` Class
The core fusion engine. It features frozen backbones for the heavy encoders (BERT/ResNet) to allow for efficient training on T4 GPUs while fine-tuning the fusion projections.

## 3. Usage
Ensure all multimodal dependencies are installed:
```bash
pip install transformers torchvision Pillow
```

Run the training pipeline:
```bash
python -m src.models.multimodal_v2
```

## 4. Production Significance
This architecture represents the current state-of-the-art in real estate tech (PropTech). By "seeing" the property and "reading" its description, the model can detect value that is invisible to traditional tabular models.

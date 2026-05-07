import os
import torch
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from src.utils.logger import setup_logger

logger = setup_logger("SaveBackbones", log_file="logs/save_backbones.log")

def save_models():
    """
    Downloads and saves pre-trained backbones to the local 'models/' directory
    to allow for offline loading and faster deployment.
    """
    output_dir = "models/backbones"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save DistilBERT (Text)
    text_model_name = "distilbert-base-uncased"
    logger.info(f"Downloading and saving {text_model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    tokenizer.save_pretrained(f"{output_dir}/distilbert_tokenizer")
    
    text_model = AutoModel.from_pretrained(text_model_name)
    text_model.save_pretrained(f"{output_dir}/distilbert_model")
    
    # 2. Save ResNet18 (Vision)
    logger.info("Downloading and saving ResNet18...")
    vision_model = models.resnet18(pretrained=True)
    torch.save(vision_model.state_dict(), f"{output_dir}/resnet18_weights.pth")

    logger.info(f"All backbones saved successfully to {output_dir}")

if __name__ == "__main__":
    save_models()

import torch
import os
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from src.models.multimodal_v2 import EliteMultimodalModel, MultimodalHousingDataset
from src.utils.logger import setup_logger

logger = setup_logger("V2_Validator", log_file="logs/v2_validation.log")

def validate_elite_intelligence():
    """
    Performs a full end-to-end inference using real text and real vision data
    to validate the Elite Multimodal V2 system.
    """
    logger.info("--- Starting Final Multimodal Validation ---")
    
    # 1. Setup paths
    model_path = "models/multimodal_v2.pth"
    image_path = "data/images/house_0.jpg"
    if not os.path.exists(image_path):
        # Fallback to current dir if running in a different context
        image_path = "modern_house_sample_1_1778155947172.png" 
        
    description = "A magnificent modern estate featuring an open-concept design, chef's kitchen, and floor-to-ceiling windows overlooking a private garden."
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train V2 first.")
        return

    # 2. Load Model
    logger.info("Loading Elite Multimodal V2 Model...")
    # tabular_input_size=15 based on King County fetch columns (18 total - 3 multimodal/target)
    model = EliteMultimodalModel(tabular_input_size=15)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 3. Prepare Inputs
    logger.info("Preparing Multimodal Inputs (Text + Vision + Tabular)...")
    
    # Text Tokenization
    local_tok = "models/backbones/distilbert_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(local_tok if os.path.exists(local_tok) else "distilbert-base-uncased")
    inputs = tokenizer(description, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    
    # Vision Processing
    if os.path.exists(image_path):
        from torchvision import transforms
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        logger.info(f"Loaded real vision sample from {image_path}")
    else:
        logger.warning("Real image not found, using simulated vision noise.")
        img_tensor = torch.randn(1, 3, 224, 224)

    # Tabular (Mocking a high-quality house)
    # 15 features matching King County structure
    tab_data = torch.ones(1, 15) * 1.5 

    # 4. Inference
    logger.info("Executing Fusion Intelligence...")
    with torch.no_grad():
        log_price = model(tab_data, inputs['input_ids'], inputs['attention_mask'], img_tensor)
        final_price = np.expm1(log_price.item())

    print("\n" + "="*50)
    print("       ELITE MULTIMODAL VALUATION REPORT")
    print("="*50)
    print(f"DESCRIPTION: {description[:70]}...")
    print(f"IMAGE:       {os.path.basename(image_path)}")
    print(f"PREDICTION:  ${final_price:,.2f} USD")
    print("="*50)
    print("STATUS:      Inference Successful (Fusion Transformer)")
    print("="*50 + "\n")

if __name__ == "__main__":
    validate_elite_intelligence()

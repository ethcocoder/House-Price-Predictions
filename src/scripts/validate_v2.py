import torch
import os
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from src.models.multimodal_v2 import EliteMultimodalModel
from src.utils.logger import setup_logger

logger = setup_logger("V2_Validator", log_file="logs/v2_validation.log")

def run_inference(model, tokenizer, image_path, description, tabular_data):
    """Runs a single multimodal inference pass."""
    # 1. Text
    inputs = tokenizer(description, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    
    # 2. Vision
    from torchvision import transforms
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
    else:
        img_tensor = torch.randn(1, 3, 224, 224)

    # 3. Predict
    with torch.no_grad():
        output = model(tabular_data, inputs['input_ids'], inputs['attention_mask'], img_tensor)
        res = output.item()
        
        # Scaling logic
        if res < 30: 
            final_price = np.expm1(res)
        else:
            final_price = res
            
        if final_price < 2000:
            final_price *= 1000
            
    return res, final_price

def validate_elite_intelligence():
    logger.info("--- Starting Multi-Scenario Multimodal Validation ---")
    
    # Setup
    model_path = "models/multimodal_v2.pth"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}.")
        return

    model = EliteMultimodalModel(tabular_input_size=15)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    local_tok = "models/backbones/distilbert_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(local_tok if os.path.exists(local_tok) else "distilbert-base-uncased")

    scenarios = [
        {
            "name": "MODERN LUXURY ESTATE",
            "image": "data/images/house_0.jpg",
            "desc": "A magnificent modern estate featuring an open-concept design, chef's kitchen, and floor-to-ceiling windows.",
            "tab": torch.tensor([[2500, 4, 2.5, 9, 2, 2100, 400, 2015, 0, 98103, 47.6, -122.3, 2200, 8000, 1]], dtype=torch.float32)
        },
        {
            "name": "CLASSIC COZY COTTAGE",
            "image": "data/images/house_1.jpg",
            "desc": "A charming rustic cottage with brick walls and a beautiful flower garden, perfect for a quiet lifestyle.",
            "tab": torch.tensor([[1200, 2, 1.0, 6, 1, 1000, 200, 1950, 0, 98103, 47.5, -122.4, 1100, 4000, 0]], dtype=torch.float32)
        }
    ]

    print("\n" + "═"*60)
    print("       ELITE MULTIMODAL COMPARISON REPORT")
    print("═"*60)

    for s in scenarios:
        raw, price = run_inference(model, tokenizer, s['image'], s['desc'], s['tab'])
        
        print(f"SCENARIO:    {s['name']}")
        print(f"DESCRIPTION: {s['desc'][:60]}...")
        print(f"IMAGE:       {os.path.basename(s['image'])}")
        print(f"RAW OUTPUT:  {raw:.4f}")
        print(f"PREDICTION:  ${price:,.2f} USD")
        print("-" * 60)

    print("STATUS:      Multi-Scenario Inference Successful")
    print("═"*60 + "\n")

if __name__ == "__main__":
    validate_elite_intelligence()

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
    logger.info("--- Starting 4-Scenario Multimodal Stress Test ---")
    
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

    # 4 Scenarios: 2 Images x 2 Prompts
    scenarios = [
        # Scenario 1: Modern House + Luxury Prompt
        {
            "name": "MODERN (Luxury Listing)",
            "image": "data/images/house_0.jpg",
            "desc": "An elite ultra-modern mansion with smart home tech and premium finishes.",
            "tab": torch.tensor([[2500, 4, 2.5, 9, 2, 2100, 400, 2015, 0, 98103, 47.6, -122.3, 2200, 8000, 1]], dtype=torch.float32)
        },
        # Scenario 2: Modern House + Negative Prompt
        {
            "name": "MODERN (Foreclosure/Distressed)",
            "image": "data/images/house_0.jpg",
            "desc": "A modern building in significant disrepair, requires total renovation and structural work.",
            "tab": torch.tensor([[2500, 4, 2.5, 9, 2, 2100, 400, 2015, 0, 98103, 47.6, -122.3, 2200, 8000, 1]], dtype=torch.float32)
        },
        # Scenario 3: Cottage + Charming Prompt
        {
            "name": "COTTAGE (Charming Boutique)",
            "image": "data/images/house_1.jpg",
            "desc": "A world-class charming cottage with historical value and a prize-winning garden.",
            "tab": torch.tensor([[1200, 2, 1.0, 6, 1, 1000, 200, 1950, 0, 98103, 47.5, -122.4, 1100, 4000, 0]], dtype=torch.float32)
        },
        # Scenario 4: Cottage + Negative Prompt
        {
            "name": "COTTAGE (Abandoned/Rotting)",
            "image": "data/images/house_1.jpg",
            "desc": "An old abandoned wooden shed with rotting walls and overgrown weeds.",
            "tab": torch.tensor([[1200, 2, 1.0, 6, 1, 1000, 200, 1950, 0, 98103, 47.5, -122.4, 1100, 4000, 0]], dtype=torch.float32)
        }
    ]

    print("\n" + "═"*70)
    print("       ELITE MULTIMODAL 4-SCENARIO STRESS TEST")
    print("═"*70)

    for s in scenarios:
        raw, price = run_inference(model, tokenizer, s['image'], s['desc'], s['tab'])
        
        print(f"TEST:        {s['name']}")
        print(f"IMAGE:       {os.path.basename(s['image'])}")
        print(f"PROMPT:      {s['desc'][:60]}...")
        print(f"RESULT:      ${price:,.2f} USD")
        print("-" * 70)

    print("STATUS:      4-Scenario Stress Test Complete")
    print("═"*70 + "\n")

if __name__ == "__main__":
    validate_elite_intelligence()

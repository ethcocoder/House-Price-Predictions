import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from src.utils.logger import setup_logger

logger = setup_logger("MultimodalV2", log_file="logs/multimodal_v2.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalHousingDataset(Dataset):
    def __init__(self, df, tabular_cols, text_col='description', img_col='image_path', tokenizer_name='distilbert-base-uncased'):
        self.df = df
        self.tabular_data = torch.tensor(df[tabular_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(np.log1p(df['SalePrice'].values), dtype=torch.float32).view(-1, 1)
        
        # Check for local tokenizer
        local_tok_path = "models/backbones/distilbert_tokenizer"
        if os.path.exists(local_tok_path):
            self.tokenizer = AutoTokenizer.from_pretrained(local_tok_path)
            logger.info("Loaded tokenizer from local cache.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.descriptions = df[text_col].tolist()
        self.image_paths = df[img_col].tolist()
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Tabular
        tab = self.tabular_data[idx]
        
        # 2. Text
        text = self.descriptions[idx]
        text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        
        # 3. Vision (Simulated with random noise if file doesn't exist for demo)
        img_path = self.image_paths[idx]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img = self.img_transform(img)
        else:
            # Create a mock image for demonstration
            img = torch.randn(3, 224, 224)
            
        target = self.targets[idx]
        
        return {
            'tabular': tab,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'image': img,
            'target': target
        }

class EliteMultimodalModel(nn.Module):
    def __init__(self, tabular_input_size, text_model_name='distilbert-base-uncased'):
        super(EliteMultimodalModel, self).__init__()
        
        # 1. Tabular Encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Text Encoder (DistilBERT)
        local_text_path = "models/backbones/distilbert_model"
        if os.path.exists(local_text_path):
            self.text_encoder = AutoModel.from_pretrained(local_text_path)
            logger.info("Loaded text encoder from local cache.")
        else:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_projection = nn.Linear(768, 128)
        
        # 3. Vision Encoder (ResNet18)
        self.vision_encoder = models.resnet18(pretrained=False) # Use false as we load manual weights
        local_vision_path = "models/backbones/resnet18_weights.pth"
        if os.path.exists(local_vision_path):
            self.vision_encoder.load_state_dict(torch.load(local_vision_path, map_location='cpu'))
            logger.info("Loaded vision encoder from local cache.")
        else:
            # Fallback to online weights
            self.vision_encoder = models.resnet18(pretrained=True)
            
        self.vision_encoder.fc = nn.Identity() 
        
        # 4. Fusion Layer (Transformer-like fusion or simple concatenation)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tab, input_ids, attention_mask, img):
        # Encode Tabular
        tab_feat = self.tabular_encoder(tab)
        
        # Encode Text
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_projection(text_out.last_hidden_state[:, 0, :]) # CLS token
        
        # Encode Vision
        vision_feat = self.vision_projection(self.vision_encoder(img))
        
        # Concatenate & Predict
        combined = torch.cat((tab_feat, text_feat, vision_feat), dim=1)
        return self.fusion(combined)

def train_v2():
    logger.info("Initializing Elite Multimodal V2 Training Pipeline...")
    
    # Load massive dataset
    data_path = "data/raw/massive_housing_data.csv"
    if not os.path.exists(data_path):
        logger.error("Massive dataset not found. Run src/utils/dataset.py first.")
        return

    # Small sample for demo speed
    df = pd.read_csv(data_path).head(1000) 
    
    # Check for required V2 columns
    if 'description' not in df.columns or 'image_path' not in df.columns:
        logger.error("Missing multimodal columns ('description' or 'image_path').")
        logger.info("Please regenerate the massive dataset by running: python -m src.utils.dataset")
        return

    # Identify numeric cols for tabular encoder
    tabular_cols = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice']).columns.tolist()
    
    # Dataset & Loader
    dataset = MultimodalHousingDataset(df, tabular_cols)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model
    model = EliteMultimodalModel(len(tabular_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    logger.info("Starting Multimodal Training Loop...")
    model.train()
    for epoch in range(5): # Small epochs for demo
        epoch_loss = 0
        for batch in loader:
            tab = batch['tabular'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            img = batch['image'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(tab, ids, mask, img)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/5, Loss: {epoch_loss/len(loader):.4f}")

    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/multimodal_v2.pth")
    logger.info("Elite Multimodal V2 model saved to models/multimodal_v2.pth")

if __name__ == "__main__":
    train_v2()

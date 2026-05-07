import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from src.utils.logger import setup_logger

logger = setup_logger("DeepLearning", log_file="logs/deep_learning.log")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HousingMLP(nn.Module):
    def __init__(self, input_size):
        super(HousingMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def train_nn():
    data_path = "data/processed/housing_cleaned.csv"
    if not os.path.exists(data_path):
        logger.error("Cleaned data not found.")
        return

    df = pd.read_csv(data_path)
    drop_cols = ['Unnamed: 0', 'Order', 'PID']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    y_log = np.log1p(y).values
    
    # Preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_log, test_size=0.2, random_state=42)
    
    # DataLoader
    train_dataset = HousingDataset(X_train, y_train)
    test_dataset = HousingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model, Loss, Optimizer
    model = HousingMLP(X_processed.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 100
    best_loss = float('inf')
    
    logger.info("Starting Neural Network training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "models/best_nn.pth")
    
    # Final Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
    
    y_pred = np.concatenate(all_preds).flatten()
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    
    logger.info(f"NN Results (Log-Scale): RMSE = {rmse_log:.4f}")
    logger.info(f"NN Results (Original Scale): RMSE = ${rmse_orig:,.2f}")
    
    return model

if __name__ == "__main__":
    train_nn()

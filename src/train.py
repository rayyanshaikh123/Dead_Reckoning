import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import IOVNBDDataset
from src.model import LightweightIDNN

def run_training():
    print("Loading extracted IO-VNBD dataset...")
    dataset = IOVNBDDataset(data_dir="data/raw/IO-VNBD", window_size=20)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LightweightIDNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    print("Starting baseline training run...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss (MSE): {running_loss/len(dataloader):.4f}")
        
    os.makedirs("results/weights", exist_ok=True)
    torch.save(model.state_dict(), "results/weights/baseline_idnn.pth")
    print("Weights saved successfully to results/weights/")

if __name__ == "__main__":
    run_training()
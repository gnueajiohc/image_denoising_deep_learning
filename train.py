import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_train_loader
from models import DenoisingCNN

def print_model_info(model_name, dataset):
    print("[MODEL INFO]".center(30, '-'))
    print(f"\nModel: {model_name.upper()}")
    print(f"Dataset: {dataset}\n")

def train_model(model, train_loader, epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("[TRAINING]".center(30, '-'))
    print("")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            noisy_images = images + 0.2 * torch.rand_like(images)
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1} of {epochs}] Loss: {avg_loss:.4f}")
    print("")

def main(model_name, dataset, epochs=10, batch_size=64, lr=1e-3):
    train_loader = get_train_loader(dataset=dataset, batch_size=batch_size)
    
    models = {
        "cnn": DenoisingCNN(hidden_channels=[64, 128, 64])
    }
    
    if model_name not in models:
        raise ValueError(f"Not available model")
    
    model = models[model_name]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_model_info(model_name, dataset)
    
    train_model(model, train_loader, epochs=epochs, lr=lr, device=device)
    
if __name__=="__main__":
    main(model_name="cnn", dataset="CIFAR10", epochs=10, batch_size=64, lr=1e-3)
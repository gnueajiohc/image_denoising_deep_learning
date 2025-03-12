import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_train_loader
from utils import add_noise
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
            noisy_images = add_noise(images)
            
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1} of {epochs}] Loss: {avg_loss:.4f}")
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = f"results/{model.__class__.__name__}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[INFO] Model saved at {model_path}\n")
    print("")

def main(model_name, dataset, epochs, batch_size, lr):
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
    parser = argparse.ArgumentParser(description="Denoising model trainer")
    
    parser.add_argument("--model_name", type=str, default="cnn", help="Name of the model (default: cnn)")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Name of the dataset (default: CIFAR10)")
    parser.add_argument("--epochs", type=int, default=10, help="Num of Epochs for training (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training (default: 1e-3)")
    
    args = parser.parse_args()
    
    main(model_name=args.model_name, dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
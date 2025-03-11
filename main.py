import torch
from train import train_model
from utils import get_dataloaders
from models import DenoisingCNN

def main(model_name="cnn", dataset="CIFAR10", epochs=10, batch_size=64, lr=1e-3):
    train_loader, test_loader = get_dataloaders(dataset=dataset, batch_size=batch_size)
    
    models = {
        "cnn": DenoisingCNN(hidden_channels=[64, 128, 64])
    }
    
    if model_name not in models:
        raise ValueError(f"Not available model")
    
    model = models[model_name]
    
    device = "cuda" if torch.cuda.is_available else "cpu"
    
    train_model(model, train_loader, epochs=epochs, lr=lr, device=device)

if __name__=="__main__":
    main(model_name="cnn", dataset="CIFAR10", epochs=10, batch_size=64, lr=1e-3)
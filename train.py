import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, epochs=10, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
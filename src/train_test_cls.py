import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_train_loader, get_test_loader
from utils import add_noise
from utils import print_model_info
from models import ClassifyingResNet, ClassifyingCNN
from models import select_classifier_model

def train_cls_model(model, train_loader, test_loader, save_name, epochs=10, lr=1e-3, device="cpu"):
    """
    Train classifier 'model' with given dataloader and save it in "results/weights/{save_name}.pth"
    
    Args:
        model (nn.Module): image classifier model class
        train_loader (DataLoader): dataloader from dataset
        save_name (str): file name to be saved
        epochs (int): num of epochs
        lr (float): learning rate
        device (str): cpu or cuda
    """
    # use Adam optimizer and MSE loss function
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("[TRAINING]".center(50, '-'))
    print("")
    
    # record training start time
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            images = add_noise(images)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) # forward propagation
            
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # backward propagation
            optimizer.step() # update model parameters
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1} of {epochs}] Loss: {avg_loss:.5f}")
        
    # calculating training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[INFO] Total training time: {elapsed_time:.2f} seconds")
    
    # save trained model parameters
    model_path = f"results/weights/{save_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved at {model_path}\n")
    
    print("[TESTING]".center(50, '-'))
    print("")
    
    correct_answers = 0
    num_samples = 0
    
    # record test start time
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images = add_noise(images)
            labels = labels.to(device)
            
            outputs = model(images) # forward propagation
            predicts = torch.argmax(outputs, dim=1)

            correct_answers += (predicts == labels).sum().item()
            num_samples += images.shape[0]
    
    # calculating testing time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total test time: {elapsed_time:.2f} seconds")
    
    correct_ratio = correct_answers / num_samples
    print(f"[INFO] Eval score - Correct: {correct_ratio:.4f}\n")

def main(model_name, dataset, epochs, batch_size, lr, use_batchnorm):
    """
    Args:
        model_name (str): name of image denoising model class
        dataset (str): name of dataset
        epochs (int): num of epochs
        batch_size (int): batch size of datatset
        lr (float): learning rate
        use_batchnorm (bool): use batch normalization or not
    """
    train_loader = get_train_loader(dataset=dataset, batch_size=batch_size)
    test_loader = get_test_loader(dataset=dataset, batch_size=batch_size)
    
    model = select_classifier_model(model_name, use_batchnorm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_model_info(model_name, model, dataset)
    
    save_name = f"{model.__class__.__name__}_{dataset}"
    train_cls_model(model, train_loader, test_loader, save_name=save_name, epochs=epochs, lr=lr, device=device) # training model

  
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Classifier model trainer")

    parser.add_argument("--model_name", type=str, default="cnn", help="Name of the model (default: cnn)")
    parser.add_argument("--dataset", type=str, default="STL10", help="Name of the dataset (default: STL10)")
    parser.add_argument("--epochs", type=int, default=10, help="Num of Epochs for training (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training (default: 1e-3)")
    parser.add_argument("--no_batchnorm", action="store_true", help="Not using batch normalization")
    
    args = parser.parse_args()
    
    main(model_name=args.model_name, dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, use_batchnorm=not args.use_batchnorm)
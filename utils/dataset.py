import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(dataset="CIFAR10", batch_size=64):
    """
    Get train_loader, test_loader from the given dataset
    
    Args:
        dataset (str): dataset name (CIFAR10, MNIST, CelebA)
        batch_size (int): batch size

    Returns:
        train_loader (DataLoader): train data loader
        test_loader (DataLoader): test data loader
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset.upper() == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    # Here, you can add whatever dataset you want
    else:
        raise ValueError(f"Not available dataset.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
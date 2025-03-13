import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloader(dataset="CIFAR10", batch_size=64, train=True):
    """
    Get train_loader, test_loader from the given dataset
    
    Args:
        dataset (str): dataset name (CIFAR10)
        batch_size (int): batch size
        train (bool): train dataset or test dataset

    Returns:
        dataloader (DataLoader): data loader
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset.upper() == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=True)
    elif dataset.upper() == "STL10":
        split = "train" if train else "test"
        dataset = datasets.STL10(root="./data", split=split, transform=transform, download=True)
    # Here, you can add whichever dataset you want
    else:
        raise ValueError(f"Not available dataset.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def get_train_loader(dataset="CIFAR10", batch_size=64):
    """wrapper function for making train dataloader"""
    return get_dataloader(dataset=dataset, batch_size=batch_size, train=True)

def get_test_loader(dataset="CIFAR10", batch_size=64):
    """wrapper function for making test dataloader"""
    return get_dataloader(dataset=dataset, batch_size=batch_size, train=False)
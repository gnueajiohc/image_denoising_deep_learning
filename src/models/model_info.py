import torch
from class_guided_unet import ClassGuidedUNet

def load_model(model, dataset="STL10", device="cuda"):
    """helper function to load model parameters"""
    model_path = f"results/weights/{model.__class__.__name__}_{dataset}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}. You should train the model first.\n")
        return
    
def get_model_name(model, dataset="STL10"):
    save_name = f"{model.__class__.__name__}_{dataset}"
    if isinstance(model, ClassGuidedUNet):
        save_name += f"_{model.classifier.__class__.__name__}"
    
    return save_name
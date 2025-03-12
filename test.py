import argparse
import torch
from utils import psnr, ssim
from utils import get_test_loader
from utils import add_noise
from utils import save_results
from models import DenoisingCNN

def print_model_info(model_name, dataset):
    print("[MODEL INFO]".center(30, '-'))
    print(f"\nModel: {model_name.upper()}")
    print(f"Dataset: {dataset}\n")

def test_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    print("[TESTING]".center(30, '-'))
    print("")
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            noisy_images = add_noise(images)
            
            denoised_images = model(noisy_images)
            
            psnr_value = psnr(denoised_images, images)
            ssim_value = ssim(denoised_images, images)
            
            total_psnr += psnr_value.item()
            total_ssim += ssim_value.item()
            num_samples += 1
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"[INFO] Eval score - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    save_results(images, noisy_images, denoised_images, num_images=3)
    
    return avg_psnr, avg_ssim

def main(model_name, dataset, batch_size):
    test_loader = get_test_loader(dataset=dataset, batch_size=batch_size)
    
    models = {
        "cnn": DenoisingCNN(hidden_channels=[64, 128, 64])
    }
    
    if model_name not in models:
        raise ValueError(f"Not available model")
    
    model = models[model_name]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = f"results/{model.__class__.__name__}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}. You should train the model first.\n")
        return
    
    print_model_info(model_name, dataset)
    
    test_model(model, test_loader, device=device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Denoising model tester")
    
    parser.add_argument("--model_name", type=str, default="cnn", help="Name of the model (default: cnn)")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Name of the dataset (default: CIFAR10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing (default: 64)")
    
    args = parser.parse_args()
    
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)
import argparse
import time
import torch
from utils import psnr, ssim
from utils import get_test_loader
from utils import add_noise
from utils import save_test_figure
from utils import print_model_info
from utils import save_test_score
from models import select_model

def test_model(model, test_loader, dataset, device="cpu"):
    model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    print("[TESTING]".center(50, '-'))
    print("")
    
    start_time = time.time()
    
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total test time: {elapsed_time:.2f} seconds")
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    save_file_name = f"{model.__class__.__name__}_{dataset}"
    save_test_score(avg_psnr, avg_ssim, save_file_name)
    print(f"[INFO] Eval score - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n")
    
    save_path = f"results/figure/{save_file_name}.png"
    save_test_figure(images, noisy_images, denoised_images, save_path=save_path ,num_images=3)
    
    return avg_psnr, avg_ssim

def main(model_name, dataset, batch_size):
    test_loader = get_test_loader(dataset=dataset, batch_size=batch_size)
    
    model = select_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"results/weights/{model.__class__.__name__}_{dataset}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}. You should train the model first.\n")
        return
    
    print_model_info(model_name, model, dataset)
    
    test_model(model, test_loader, dataset=dataset, device=device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Denoising model tester")
    
    parser.add_argument("--model_name", type=str, default="cnn", help="Name of the model (default: cnn)")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Name of the dataset (default: CIFAR10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing (default: 64)")
    
    args = parser.parse_args()
    
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)
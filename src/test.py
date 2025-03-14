import argparse
import time
import torch
from utils import psnr, ssim
from utils import get_test_loader
from utils import add_noise
from utils import save_test_figure
from utils import print_model_info
from utils import save_test_score
from models import load_model, get_model_name
from models import select_model

def test_model(model, test_loader, dataset, device="cpu"):
    """
    Test 'model' with given dataloader and save it"
    
    Args:
        model (nn.Module): image denoising model class
        test_loader (DataLoader): dataloader from dataset
        dataset (str): dataset name
        device (str): cpu or cuda
    """
    model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    print("[TESTING]".center(50, '-'))
    print("")
    
    # record test start time
    start_time = time.time()
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            noisy_images = add_noise(images) # add noise
            
            denoised_images = model(noisy_images) # forward propagation
            
            psnr_value = psnr(denoised_images, images) # compute PSNR
            ssim_value = ssim(denoised_images, images) # compute SSIM
            
            total_psnr += psnr_value.item()
            total_ssim += ssim_value.item()
            num_samples += 1
    
    # calculating testing time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total test time: {elapsed_time:.2f} seconds")
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # save test results
    save_name = get_model_name(model, dataset)
    save_path = f"results/scores/{save_name}.txt"
    save_test_score(avg_psnr, avg_ssim, save_path)
    print(f"[INFO] Eval score - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n")
    
    save_path = f"results/figures/{save_name}.png"
    save_test_figure(images, noisy_images, denoised_images, save_path=save_path ,num_images=3)
    
    return avg_psnr, avg_ssim

def main(model_name, dataset, batch_size, use_batchnorm):
    """
    Args:
        model_name (str): name of image denoising model class
        dataset (str): name of dataset
        batch_size (int): batch size of datatset
        use_batchnorm (bool): use batch normalization or not
    """
    test_loader = get_test_loader(dataset=dataset, batch_size=batch_size)
    
    model = select_model(model_name, use_batchnorm)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(model, dataset, device)
    
    print_model_info(model_name, model, dataset)
    
    test_model(model, test_loader, dataset=dataset, device=device)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Denoising model tester")
    
    parser.add_argument("--model", type=str, default="cnn", help="Name of the model (default: cnn)")
    parser.add_argument("--dataset", type=str, default="STL10", help="Name of the dataset (default: STL10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing (default: 64)")
    parser.add_argument("--no_batchnorm", action="store_true", help="Not using batch normalization")
    
    args = parser.parse_args()
    
    main(model_name=args.model, dataset=args.dataset, batch_size=args.batch_size, use_batchnorm=not args.no_batchnorm)
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def print_model_info(model_name, dataset):
    print("[MODEL INFO]".center(30, '-'))
    print(f"\nModel: {model_name.upper()}")
    print(f"Dataset: {dataset}\n")

def add_noise(image, noise_level=0.2, sp_prob=0.05):
    noise = noise_level * torch.randn_like(image)
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    # noise = torch.rand_like(image)
    # salt = (noise > 1 - sp_prob).float()
    # pepper = (noise < sp_prob).float()
    # noisy_image = noisy_image * (1 - salt - pepper) + salt

    return noisy_image

def save_results(original, noisy, denoised, save_path, num_images=3):
    if original.dim() == 3:
        original = original.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        denoised = denoised.unsqueeze(0)
    
    original = original[:num_images]
    noisy = noisy[:num_images]
    denoised = denoised[:num_images]
    
    images = torch.cat([original, noisy, denoised], dim=1)  # (3*num_images, C, H, W)
    grid = vutils.make_grid(images, nrow=num_images, normalize=True, scale_each=True)

    # 시각화
    plt.figure(figsize=(num_images * 2, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # [C, H, W] → [H, W, C] 변환
    plt.axis("off")
    plt.title("Original → Noisy → Denoised")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
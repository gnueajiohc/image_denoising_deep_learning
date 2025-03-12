import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_results(original, noisy, denoised, save_path, num_images=3):
    if original.dim() == 3:
        original = original.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        denoised = denoised.unsqueeze(0)
    
    original = original[:num_images]
    noisy = noisy[:num_images]
    denoised = denoised[:num_images]
    
    images = torch.cat([original, noisy, denoised], dim=0)  # (3*num_images, C, H, W)
    grid = vutils.make_grid(images, nrow=num_images, normalize=True, scale_each=True)

    # 시각화
    plt.figure(figsize=(num_images * 2, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # [C, H, W] → [H, W, C] 변환
    plt.axis("off")
    plt.title("Original → Noisy → Denoised")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
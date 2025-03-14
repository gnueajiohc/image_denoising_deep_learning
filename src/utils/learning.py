import torch
import matplotlib.pyplot as plt

def print_model_info(model_name, model, dataset):
    """print model name, dataset, number of parameters"""
    print("[MODEL INFO]".center(50, '-'))
    print(f"\nModel: {model_name.upper()}")
    print(f"Dataset: {dataset}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

def add_noise(image, noise_level=0.2, sp_prob=0.05):
    """add Gaussian or Salt & Pepper noise to 'image'"""
    # add Gaussian noise
    noise = noise_level * torch.randn_like(image)
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    # add salt & pepper noise
    # noise = torch.rand_like(image)
    # salt = (noise > 1 - sp_prob).float()
    # pepper = (noise < sp_prob).float()
    # noisy_image = noisy_image * (1 - salt - pepper) + salt

    return noisy_image

def load_weights(model, dataset, device):
    model_path = f"results/weights/{model.__class__.__name__}_{dataset}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}. You should train the model first.\n")
        return

def save_test_figure(original, noisy, denoised, save_path, num_images=3):
    """
    Displays and saves up to 'num_images' rows of images, each row containing
    (Original, Noisy, Denoised) side by side.

    Args:
        original (torch.Tensor): A collection of original images [D, C, H, W]
        noisy (torch.Tensor): A collection of noisy images [D, C, H, W]
        denoised (torch.Tensor): A collection of denoised images [D, C, H, W]
        save_path (str): The path (including filename) where the plot will be saved
        num_images (int): Number of image triplets (Original, Noisy, Denoised) to display and save
    """

    num_images = min(num_images, original.shape[0])

    original = original[:num_images].cpu().detach().numpy()
    noisy = noisy[:num_images].cpu().detach().numpy()
    denoised = denoised[:num_images].cpu().detach().numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        for j, img in enumerate([original[i], noisy[i], denoised[i]]):
            img = img.transpose(1, 2, 0) if img.shape[0] == 3 else img[0]
            axes[i][j].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_test_score(psnr, ssim, save_file_name):
    """save psnr, ssim score in 'save_file_name'"""
    with open(f"results/scores/{save_file_name}", "w") as f:
        f.write(f"[INFO] Eval score - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}\n")

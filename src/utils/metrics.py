import torch
import torch.nn.functional as F

def psnr(image1, image2, max_pixel=1.0):
    """calculate PSNR of 2 images 'image1' and 'image2'"""
    mse = F.mse_loss(image1, image2)
    return 10 * torch.log10(max_pixel ** 2 / mse)

def gaussian_kernel(kernel_size=11, sigma=1.5):
    """
    Args:
        kernel_size (int): size of Gaussian kernel
        sigma (float): standard deviation of Gaussian distribution
    """
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    # 2-dim Gaussian kernel
    g2d = g.unsqueeze(0) * g.unsqueeze(1)  # [kernel_size, kernel_size]
    return g2d

def ssim(image1: torch.Tensor, 
         image2: torch.Tensor, 
         kernel_size: int = 11, 
         sigma: float = 1.5,
         data_range: float = 1.0,
         K: tuple = (0.01, 0.03)) -> torch.Tensor:
    """
    Calculate SSIM of two images 'image1' and 'image2'

    Args:
        image1 (torch.Tensor): whose shape is [N, C, H, W] or [C, H, W], [H, W]
        image2 (torch.Tensor): whose shape is [N, C, H, W] or [C, H, W], [H, W] (same to 'image1')
        kernel_size (int): size of Gaussian kernel
        sigma (float): standard deviation of Gaussian distribution
        data_range (float): range of image value (1.0 or 255)
        K (tuple): contants for SSIM (K1, K2)
        
    Returns:
        torch.Tensor: SSIM value (scalar)
    """
    # image1, image2 4-dim reshape
    if image1.dim() == 2:
        image1 = image1.unsqueeze(0).unsqueeze(0)
        image2 = image2.unsqueeze(0).unsqueeze(0)
    elif image1.dim() == 3:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

    assert image1.shape == image2.shape, "image1 and image2 have different shape."

    # constants C1, C2 used for SSIM
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Gaussian kernel
    window_2d = gaussian_kernel(kernel_size, sigma).to(image1.device)
    # reshape to 4-dim for using conv2d: [out_channels, in_channels, kH, kW]
    # use repeat for applying same Gaussian kernel to each channel
    window_4d = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    window_4d = window_4d.repeat(image1.shape[1], 1, 1, 1)  # [C, 1, kH, kW]

    # groups=image1.shape[1] for handling each channel separately
    mu1 = F.conv2d(image1, window_4d, padding=kernel_size // 2, groups=image1.shape[1])
    mu2 = F.conv2d(image2, window_4d, padding=kernel_size // 2, groups=image2.shape[1])

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window_4d, padding=kernel_size // 2, groups=image1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window_4d, padding=kernel_size // 2, groups=image2.shape[1]) - mu2_sq
    sigma12   = F.conv2d(image1 * image2, window_4d, padding=kernel_size // 2, groups=image1.shape[1]) - mu1_mu2

    # SSIM calculation
    # SSIM(x, y) = [(2µxµy + C1) * (2σxy + C2)] / [(µx^2 + µy^2 + C1) * (σx^2 + σy^2 + C2)]
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # average SSIM of entire image
    return ssim_map.mean()

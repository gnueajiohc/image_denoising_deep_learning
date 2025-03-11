import torch
import torch.nn.functional as F

def psnr(image1, image2, max_pixel=1.0):
    mse = F.mse_loss(image1, image2)
    return 10 * torch.log10(max_pixel ** 2 / mse)

def ssim(image1, image2, max_pixel=1.0):
    mu_x = image1.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = image2.mean(dim=(1, 2, 3), keepdim=True)
    
    sigma_x = image1.std(dim=(1, 2, 3), keepdim=True)
    sigma_y = image2.std(dim=(1, 2, 3), keepdim=True)
    
    sigma_xy = ((image1 - mu_x) * (image2 - mu_y)).mean(dim=(1, 2, 3), keepdim=True)
    
    C1 = (0.01 * max_pixel) ** 2
    C2 = (0.03 * max_pixel) ** 2
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    
    return numerator / denominator
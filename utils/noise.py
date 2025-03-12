import torch

def add_noise(image, noise_level=0.4, sp_prob=0.05):
    noise = noise_level * torch.randn_like(image)
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    noise = torch.rand_like(image)
    salt = (noise > 1 - sp_prob).float()
    pepper = (noise < sp_prob).float()
    noisy_image = noisy_image * (1 - salt - pepper) + salt

    return noisy_image
import torch
import torch.nn.functional as F

def psnr(image1, image2, max_pixel=1.0):
    mse = F.mse_loss(image1, image2)
    return 10 * torch.log10(max_pixel ** 2 / mse)

def gaussian_kernel(kernel_size=11, sigma=1.5):
    """
    kernel_size: 가우시안 커널의 크기
    sigma: 가우시안 분포의 표준편차
    """
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    # 2차원 가우시안 커널로 변환
    g2d = g.unsqueeze(0) * g.unsqueeze(1)  # [kernel_size, kernel_size]
    return g2d

def ssim(image1: torch.Tensor, 
         image2: torch.Tensor, 
         kernel_size: int = 11, 
         sigma: float = 1.5,
         data_range: float = 1.0,
         K: tuple = (0.01, 0.03)) -> torch.Tensor:
    """
    두 2D 이미지(또는 배치+채널이 붙은 4D 텐서)에 대해 SSIM을 계산합니다.

    Args:
        image1 (torch.Tensor): [N, C, H, W] 혹은 [C, H, W], [H, W] 형태
        image2 (torch.Tensor): image1과 같은 크기의 텐서
        kernel_size (int): 가우시안 윈도우 크기
        sigma (float): 가우시안 윈도우의 표준편차
        data_range (float): 이미지 값의 범위 (보통 1.0 혹은 255)
        K (tuple): SSIM 계산 시 사용하는 상수 (K1, K2)
        
    Returns:
        torch.Tensor: SSIM 값 (스칼라)
    """
    # image1, image2가 4차원이 아니면 4차원으로 reshape
    # (배치가 없다면 가상의 배치/채널을 추가)
    if image1.dim() == 2:
        image1 = image1.unsqueeze(0).unsqueeze(0)
        image2 = image2.unsqueeze(0).unsqueeze(0)
    elif image1.dim() == 3:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

    assert image1.shape == image2.shape, "image1과 image2의 크기가 다릅니다."

    # SSIM 공식에 들어갈 상수값 C1, C2
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 가우시안 커널 준비
    window_2d = gaussian_kernel(kernel_size, sigma).to(image1.device)
    # conv2d에 쓰기 위해 4D로 모양 변환: [out_channels, in_channels, kH, kW]
    # 여기서 각 채널에 똑같은 가우시안 커널을 적용하기 위해 repeat 사용
    window_4d = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    window_4d = window_4d.repeat(image1.shape[1], 1, 1, 1)  # [C, 1, kH, kW]

    # 각 채널을 별도로 처리하기 위해 groups=image1.shape[1] 설정
    mu1 = F.conv2d(image1, window_4d, padding=kernel_size // 2, groups=image1.shape[1])
    mu2 = F.conv2d(image2, window_4d, padding=kernel_size // 2, groups=image2.shape[1])

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window_4d, padding=kernel_size // 2, groups=image1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window_4d, padding=kernel_size // 2, groups=image2.shape[1]) - mu2_sq
    sigma12   = F.conv2d(image1 * image2, window_4d, padding=kernel_size // 2, groups=image1.shape[1]) - mu1_mu2

    # SSIM 계산
    # SSIM(x, y) = [(2µxµy + C1) * (2σxy + C2)] / [(µx^2 + µy^2 + C1) * (σx^2 + σy^2 + C2)]
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 이미지 전체의 평균 SSIM
    return ssim_map.mean()

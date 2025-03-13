from .denoising_cnn import DenoisingCNN
from .denoising_cae import DenoisingCAE
from .denoising_unet import DenoisingUNet

model_list = {
    "cnn": DenoisingCNN(hidden_channels=[64, 128, 64]),
    "cae": DenoisingCAE(hidden_channels=[8, 16, 32], use_batchnorm=True),
    "unet": DenoisingUNet(hidden_channels=[16, 32, 64], use_batchnorm=True)
}

def select_model(model_name):
    if model_name not in model_list:
        raise ValueError(f"Not available model")
    
    model = model_list[model_name]
    return model
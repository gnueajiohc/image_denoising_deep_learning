from .denoising_cnn import DenoisingCNN
from .denoising_cae import DenoisingCAE
from .denoising_unet import DenoisingUNet

# dictionary connecting model name to model class
model_list = {
    "cnn": (DenoisingCNN(hidden_channels=[32, 64, 128, 64, 32], use_batchnorm=False),
            DenoisingCNN(hidden_channels=[32, 64, 128, 64, 32], use_batchnorm=True)),
    "cae": (DenoisingCAE(hidden_channels=[16, 32, 64], use_batchnorm=False),
            DenoisingCAE(hidden_channels=[16, 32, 64], use_batchnorm=True)),
    "unet": (DenoisingUNet(hidden_channels=[16, 32, 64, 128], use_batchnorm=False),
             DenoisingUNet(hidden_channels=[16, 32, 64, 128], use_batchnorm=True))
}

def select_model(model_name, use_batchnorm):
    """return model class matching model_name"""
    if model_name not in model_list:
        raise ValueError(f"Not available model")
    
    models = model_list[model_name]
    if use_batchnorm:
        return models[1]
    else:
        return models[0]
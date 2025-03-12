import torch
import torch.nn as nn

# -----------------------------------
# Denoising CNN Autoencoder Model Class
# -----------------------------------
class DenoisingCAE(nn.Module):
    """
    Denoising CAE Model (default dataset is CIFAR10)
    
    Args:
        in_channels (int): the num of input image channels
        hidden_channels (list[int]): the num of hidden layers' channels
        kernel_size (int): kernel size
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=[8, 16, 32],
        kernel_size=3,
        use_batchnorm=False
    ):
        super(DenoisingCAE, self).__init__()
        
        encoders = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            encoders.append(nn.Conv2d(current_in_channels,
                                      current_out_channels,
                                      kernel_size=kernel_size,
                                      stride=2,
                                      padding=kernel_size // 2))
            if use_batchnorm:
                encoders.append(nn.BatchNorm2d(current_out_channels))
            encoders.append(nn.ReLU(inplace=True))
            
            current_in_channels = current_out_channels
        
        self.encoder = nn.Sequential(*encoders)
        decoders = []
        reversed_channels = list(reversed(hidden_channels))
        
        for current_out_channels in reversed_channels[1:]:
            decoders.append(nn.ConvTranspose2d(current_in_channels,
                                               current_out_channels,
                                               kernel_size=kernel_size,
                                               stride=2,
                                               padding=kernel_size // 2,
                                               output_padding=1))
            if use_batchnorm:
                decoders.append(nn.BatchNorm2d(current_out_channels))
            decoders.append(nn.ReLU(inplace=True))
            
            current_in_channels = current_out_channels
        
        decoders.append(nn.ConvTranspose2d(in_channels=current_in_channels,
                                           out_channels=in_channels,
                                           kernel_size=kernel_size,
                                           stride=2,
                                           padding=kernel_size // 2,
                                           output_padding=1))
        decoders.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoders)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
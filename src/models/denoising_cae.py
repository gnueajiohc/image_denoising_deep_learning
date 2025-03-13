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
        hidden_channels=[16, 32, 64],
        kernel_size=3,
        use_batchnorm=False
    ):
        super(DenoisingCAE, self).__init__()
        
        # list for encoding conv blocks
        encoders = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            encoders.append(self.encoding_conv_block(in_channels=current_in_channels,
                                            out_channels=current_out_channels,
                                            kernel_size=kernel_size,
                                            use_batchnorm=use_batchnorm))
            current_in_channels = current_out_channels
        
        self.encoder = nn.Sequential(*encoders)
        
        # list for decoding conv blocks
        decoders = []
        reversed_channels = list(reversed(hidden_channels))
        
        for current_out_channels in reversed_channels[1:]:
            decoders.append(self.decoding_conv_block(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=kernel_size,
                                               use_batchnorm=use_batchnorm))
            
            current_in_channels = current_out_channels
        
        # final layer with sigmoid
        decoders.append(nn.ConvTranspose2d(in_channels=current_in_channels,
                                           out_channels=in_channels,
                                           kernel_size=kernel_size,
                                           stride=2,
                                           padding=kernel_size // 2,
                                           output_padding=1))
        decoders.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoders)
    
    def encoding_conv_block(self, in_channels, out_channels, kernel_size, use_batchnorm):
        """encoding convolution block function from 'in_channels' to 'out_channels'"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=not use_batchnorm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=not use_batchnorm),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1,nn.BatchNorm2d(out_channels))
            layers.insert(-1,nn.BatchNorm2d(out_channels))
            
        return nn.Sequential(*layers)

    def decoding_conv_block(self, in_channels, out_channels, kernel_size, use_batchnorm):
        """decoding convolution block function from 'in_channels' to 'out_channels'"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2,padding=kernel_size // 2, output_padding=1, bias=not use_batchnorm),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2,padding=kernel_size // 2, output_padding=1, bias=not use_batchnorm),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1,nn.BatchNorm2d(out_channels))
            layers.insert(-1,nn.BatchNorm2d(out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """forward propagation function"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
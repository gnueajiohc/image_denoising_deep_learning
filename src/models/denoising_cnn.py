import torch.nn as nn

# -----------------------------------
# Denoising CNN Model Class
# -----------------------------------
class DenoisingCNN(nn.Module):
    """
    Denoising CNN Model (default dataset is STL10)
    
    Args:
        in_channels (int): the num of input image channels
        hidden_channels (list[int]): the num of hidden layers' channels
        kernel_size (int): kernel size
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=[32, 64, 128, 64, 32],
        kernel_size=3,
        use_batchnorm=True
    ):
        super(DenoisingCNN, self).__init__()
        
        # list for hidden convolution layers
        layers = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            layers.append(nn.Conv2d(current_in_channels,
                                    current_out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size // 2,
                                    bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(current_out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            current_in_channels = current_out_channels
        
        # final layer with sigmoid
        layers.append(nn.Conv2d(current_in_channels,
                                in_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size // 2,
                                bias=not use_batchnorm))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """forward propagation function"""
        return self.net(x)
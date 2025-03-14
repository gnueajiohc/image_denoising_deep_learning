import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# Denoising U-Net Model Class
# -----------------------------------
class DenoisingUNet(nn.Module):
    """
    Denoising U-Net Model (default dataset is CIFAR10)
    
    Args:
        in_channels (int): the num of input image channels
        hidden_channels (list[int]): the num of hidden layers' channels
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=[16, 32, 64, 128],
        use_batchnorm=False
    ):
        super(DenoisingUNet, self).__init__()
        
        # list for down blocks
        down_blocks = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            down_blocks.append(self.conv_block(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               use_batchnorm=use_batchnorm))
            current_in_channels = current_out_channels
        self.down_blocks = nn.Sequential(*down_blocks)
        # after encoder, STL10 image (3 x 96 x 96) -> (128 x 6 x 6)
        
        # bottleneck block makes channel size double
        self.bottleneck = self.conv_block(current_in_channels,
                                          2 * current_in_channels,
                                          use_batchnorm=use_batchnorm)
        current_in_channels *= 2
        # after bottleneck, STL10 image (3 x 96 x 96) -> (128 x 6 x 6)
        
        # up_convs for upside convolution
        up_convs = []
        # up_blocks for up blocks
        up_blocks = []
        # reverse hidden_channels to go upside down
        reversed_channels = list(reversed(hidden_channels))
        
        for current_out_channels in reversed_channels:
            up_convs.append(self.up_conv(in_channels=current_in_channels,
                                         out_channels=current_out_channels))
            
            # after concatenating, channel size will be double (check forward function)
            current_in_channels = current_out_channels * 2
            up_blocks.append(self.conv_block(in_channels=current_in_channels,
                                             out_channels=current_out_channels,
                                             use_batchnorm=use_batchnorm))
            current_in_channels = current_out_channels
        self.up_convs = nn.Sequential(*up_convs)
        self.up_blocks = nn.Sequential(*up_blocks)
        
        # final layer with sigmoid
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=current_in_channels, out_channels=in_channels, kernel_size=1, bias=not use_batchnorm),
            nn.Sigmoid()
        )
    
    def up_conv(self, in_channels, out_channels):
        """upside conv layer from 'in_channels' to 'out_channels' with stride=2"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels, use_batchnorm):
        """convultional block from 'in_channels' to 'out_channels'"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(-1, nn.BatchNorm2d(out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """forward propagation function"""
        # down
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            # for concatenate
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # up
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            # concatenate
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.up_blocks[i](x)
        
        x = self.final_conv(x)
        return x
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
        hidden_channels=[16, 32, 64],
        use_batchnorm=False
    ):
        super(DenoisingUNet, self).__init__()
        
        down_blocks = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            down_blocks.append(self.conv_block(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               use_batchnorm=use_batchnorm))
            current_in_channels = current_out_channels
        self.down_blocks = nn.Sequential(*down_blocks)
        
        self.bottleneck = self.conv_block(current_in_channels,
                                          2 * current_in_channels,
                                          use_batchnorm=use_batchnorm)
        current_in_channels *= 2 #128
        up_convs = []
        up_blocks = []
        reversed_channels = list(reversed(hidden_channels))
        
        for current_out_channels in reversed_channels:#[64,32,16]
            up_convs.append(self.up_conv(in_channels=current_in_channels,
                                         out_channels=current_out_channels))
            
            current_in_channels = current_out_channels * 2 # after cocatenate
            up_blocks.append(self.conv_block(in_channels=current_in_channels,
                                             out_channels=current_out_channels,
                                             use_batchnorm=use_batchnorm))
            current_in_channels = current_out_channels
        self.up_convs = nn.Sequential(*up_convs)
        self.up_blocks = nn.Sequential(*up_blocks)
        
        self.final_conv = nn.Conv2d(
            in_channels=current_in_channels,  # 최종 up_block의 출력 채널 (16)
            out_channels=in_channels,         # 3 채널로 복원
            kernel_size=1
        )
        
    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels, use_batchnorm):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(-1, nn.BatchNorm2d(out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.up_blocks[i](x)
        
        x = self.final_conv(x)
        return x
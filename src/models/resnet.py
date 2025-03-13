import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# ResNet Basic Block class
# -----------------------------------
class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet
    
    This block performs 2 consecutive 3x3 convoultional with ReLU activation, plus a skip connection
    
    Args:
        in_channels (int):
        out_channels (int):
        use_batchnorm (bool):
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=False
    ):
        pass
    
    def forward(self, x):
        pass
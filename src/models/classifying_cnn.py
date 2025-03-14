import torch
import torch.nn as nn

# -----------------------------------
# Classifying CNN Model class
# -----------------------------------
class ClassifyingCNN(nn.Module):
    """
    Args:
    
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        hidden_channels=[64, 128, 256],
        kernel_size=3,
        use_batchnorm=True
    ):
        super(ClassifyingCNN, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        layers = []
        current_in_channels = in_channels
        
        for current_out_channels in hidden_channels:
            layers.append(self.conv_block(in_channels=current_in_channels,
                                          out_channels=current_out_channels,
                                          kernel_size=kernel_size))
            current_in_channels = current_out_channels
        
        self.layers = nn.Sequential(*layers)
        
        self.fc_input_dim = self._get_conv_output_size((in_channels, 96, 96)) # 96 is for STL10
        
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
    
    def conv_block(self, in_channels, out_channels, kernel_size):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=not self.use_batchnorm),
            nn.ReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=not self.use_batchnorm),
            nn.ReLU(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        
        if self.use_batchnorm:
            layers.insert(1,nn.BatchNorm2d(out_channels))
            layers.insert(-2,nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)  # dummy tensor
            x = self.layers(x)  # pass conv layers
            return x.view(1, -1).size(1)  # Flatten size return

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)
        return x
        
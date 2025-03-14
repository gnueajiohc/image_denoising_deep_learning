import torch
import torch.nn as nn
from model_select import FEATURE_CHANNELS

class ClassGuidedUNet(nn.Module):
    def __init__(self, classifier, unet, feature_channels=FEATURE_CHANNELS):
        """
        Args:

        """
        super(ClassGuidedUNet, self).__init__()
        self.classifier = classifier
        self.unet = unet
        
        # classifier output (10) -> (FEATURE_CHANNELS * 12 * 12)
        # Unfortunately, 12 looks like a magic number here...
        # We have 12 because STL10 has width 96 and we have 4 layers with stride 2 as a default in UNet class
        self.fc = nn.Linear(10, feature_channels=FEATURE_CHANNELS * 12 * 12)
        
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            class_out = self.classifier(x)
        
        class_out = self.fc(class_out)
        class_out = class_out.view(-1, self.fc.out_features // (12 * 12), 12, 12)
        
        x = self.unet(x, class_out)
        return x
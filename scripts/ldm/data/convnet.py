import torch.nn as nn
import torch 
import numpy as np 
"""
ConvNet to reduce tensor batch channel dimension D to 1 i.e (HxWxD) -> (HxWx1)
"""
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = None
        # Input dims expected: HxWxD
        # Filter dimensions go from D -> 1 implies Output dims: HxWx1
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self,input_data):

        # Apply the 1x1 convolution
        x = self.conv1x1(input_data)
        # Apply sigmoid activation
        x = torch.sigmoid(x)
        # Save the result in projection
        self.projection = x
        return self.projection
        

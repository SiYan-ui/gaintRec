import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, inplace=True)

class VisualBranch(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256):
        super(VisualBranch, self).__init__()
        
        # Simple CNN Backbone (similar to GaitSet's MGE)
        self.layer1 = BasicConv2d(in_channels, 32, 5, padding=2)
        self.layer2 = BasicConv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.layer3 = BasicConv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.layer4 = BasicConv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 64x64 -> 8x8
        
        # Set Pooling: Max Pooling over temporal dimension
        # No parameters needed for this operation in forward
        
        # Horizontal Mapping (HPM) - Simplified
        # We will just use Global Max Pooling for simplicity in this demo
        # In full GaitSet, we would split features horizontally
        self.fc_bin = nn.Linear(128, hidden_dim)

    def forward(self, x):
        # x: [N, T, C, H, W]
        n, t, c, h, w = x.size()
        x = x.view(n * t, c, h, w)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        
        x = self.layer3(x)
        x = self.pool2(x)
        
        x = self.layer4(x)
        x = self.pool3(x) # [N*T, 128, 8, 8]
        
        # Global Max Pooling (Spatial)
        x = F.max_pool2d(x, x.size()[2:]) # [N*T, 128, 1, 1]
        x = x.view(n, t, -1) # [N, T, 128]
        
        # Set Pooling (Temporal)
        x = x.max(1)[0] # [N, 128]
        
        feature = self.fc_bin(x) # [N, hidden_dim]
        
        return feature

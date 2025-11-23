import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (V, V) Adjacency matrix
        
        # Simple GCN implementation: AXW
        # Here we implement as Conv2d(x) * A (simplified)
        # Ideally: sum(A_k * x * W_k)
        
        x = self.conv(x)
        n, c, t, v = x.size()
        
        # Matrix multiplication with Adjacency matrix
        # x: (N, C, T, V) -> (N, C*T, V)
        x = x.view(n, c * t, v)
        x = torch.matmul(x, A) # (N, C*T, V)
        x = x.view(n, c, t, v)
        
        return self.relu(self.bn(x))

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        
        # Temporal Convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.A = A
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A)
        x = self.tcn(x)
        return F.relu(x + res)

class SkeletonBranch(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, num_point=17):
        super(SkeletonBranch, self).__init__()
        
        # Define Adjacency Matrix (Simplified for COCO 17 keypoints)
        # In real implementation, this should be properly defined based on connections
        self.register_buffer('A', torch.ones(num_point, num_point).float() / num_point)
        
        self.layer1 = STGCNBlock(in_channels, 64, self.A)
        self.layer2 = STGCNBlock(64, 64, self.A)
        self.layer3 = STGCNBlock(64, 128, self.A, stride=2)
        self.layer4 = STGCNBlock(128, 128, self.A)
        self.layer5 = STGCNBlock(128, 256, self.A, stride=2)
        
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        # x: [N, T, V, C] -> [N, C, T, V]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x) # [N, 256, T/4, V]
        
        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:]) # [N, 256, 1, 1]
        x = x.view(x.size(0), -1) # [N, 256]
        
        feature = self.fc(x)
        return feature

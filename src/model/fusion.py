import torch
import torch.nn as nn
from .visual import VisualBranch
from .skeleton import SkeletonBranch

class FusionModel(nn.Module):
    def __init__(self, num_classes, visual_dim=256, skel_dim=256, embed_dim=256):
        super(FusionModel, self).__init__()
        
        self.visual_net = VisualBranch(hidden_dim=visual_dim)
        self.skeleton_net = SkeletonBranch(hidden_dim=skel_dim)
        
        # Fusion Layer
        self.fusion_fc = nn.Linear(visual_dim + skel_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        
        # Classifier (for training with Cross Entropy, optional if using only Triplet)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, imgs, skels):
        # imgs: [N, T, 1, H, W]
        # skels: [N, T, V, C]
        
        v_feat = self.visual_net(imgs)
        s_feat = self.skeleton_net(skels)
        
        # Normalize features before fusion (optional but recommended)
        v_feat = torch.nn.functional.normalize(v_feat, p=2, dim=1)
        s_feat = torch.nn.functional.normalize(s_feat, p=2, dim=1)
        
        # Concatenate
        cat_feat = torch.cat([v_feat, s_feat], dim=1)
        
        # Embedding
        embed = self.fusion_fc(cat_feat)
        embed = self.bn(embed)
        
        # Output for Triplet Loss (embedding) and CE Loss (logits)
        logits = self.classifier(embed)
        
        return embed, logits

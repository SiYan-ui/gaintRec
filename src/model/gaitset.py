"""Silhouette-based gait recognition model inspired by GaitSet."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNRelu(nn.Module):
    """Convolutional building block used by the backbone."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple wrapper
        return self.block(x)


class TemporalDifferencer(nn.Module):
    """
    Computes temporal difference between frames.
    Input A: (B, T, C, H, W)
    Output C = A - B, where B is A shifted by 1 frame (with zero padding at start).
    """
    def __init__(self, in_channels: int, diff_size: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.diff_size = diff_size
        self.stride = stride
        self.output_dim = in_channels * ((diff_size - 1) // stride + 1)
        self.block = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.5)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        if x.ndim != 5:
            raise ValueError(f"TemporalDifferencer expects 5D input (B, T, C, H, W), got {x.shape}")
        
        b, t, c, h, w = x.shape
        
        # Create B: Shifted version of A
        # Remove last frame, prepend zero frame
        # x[:, :-1] is (B, T-1, C, H, W)
        # zero_frame is (B, 1, C, H, W)
        for shift_num in range(1, self.diff_size, self.stride):
            zero_frame = torch.zeros((b, shift_num, c, h, w), dtype=x.dtype, device=x.device)
            shifted_x = torch.cat([zero_frame, x[:, :-shift_num]], dim=1)
            # C = A - B
            diff = x - shifted_x
            if shift_num == 1:
                all_diffs = diff
            else:
                all_diffs = torch.cat([all_diffs, diff], dim=2)  # Concatenate along channel dimension

        return all_diffs



class GaitSetBackbone(nn.Module):
    """Frame encoder for silhouette sequences."""

    def __init__(self, in_channels: int = 1, feature_dims: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for dim in feature_dims:
            layers.append(ConvBNRelu(prev, dim, kernel_size=3, stride=1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = dim
        self.body = nn.Sequential(*layers)
        self.output_dim = feature_dims[-1]

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Encode a batch of clips shaped (B, T, C, H, W)."""
        b, t, c, h, w = clip.shape
        x = clip.view(b * t, c, h, w)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1)     # (B*T, D, 1, 1)
        x = x.view(b, t, self.output_dim)   # (B, T, D)
        return x


class SetPooling(nn.Module):
    """Temporal pyramid set pooling that ignores frame order."""

    def __init__(self, feature_dim: int, pyramid_bins: Iterable[int] = (1, 2, 4)) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pyramid_bins = tuple(pyramid_bins)
        self.output_dim = feature_dim * (2 + sum(self.pyramid_bins))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate features shaped (B, T, D) into a single embedding."""
        if features.ndim != 3:
            raise ValueError("SetPooling expects a tensor shaped (B, T, D)")
        pooled = [features.mean(dim=1), features.max(dim=1).values] # (B, D) each，T被pool掉了
        temporal = features.transpose(1, 2)  # (B, D, T)，第 1 维和第 2 维互换
        for bins in self.pyramid_bins:
            pooled_bins = F.adaptive_avg_pool1d(temporal, output_size=bins) # (B, D, bins)
            pooled.append(pooled_bins.transpose(1, 2).reshape(features.size(0), -1))    # (B, D * bins)
        return torch.cat(pooled, dim=1)


class GaitRecognitionModel(nn.Module):
    """End-to-end silhouette-based gait classifier."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        frame_feature_dims: Sequence[int] = (32, 64, 128),
        pyramid_bins: Sequence[int] = (1, 2, 4),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.temporal_diff = TemporalDifferencer(in_channels=in_channels, diff_size=3, stride=1)
        self.backbone = GaitSetBackbone(in_channels=self.temporal_diff.output_dim, feature_dims=frame_feature_dims)
        self.pool = SetPooling(self.backbone.output_dim, pyramid_bins=pyramid_bins)
        self.embedding_dim = self.pool.output_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and pooled embeddings for an input clip shaped (B, T, C, H, W)."""
        diff_clip = self.temporal_diff(clip)
        frame_features = self.backbone(diff_clip)
        embedding = self.pool(frame_features)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding

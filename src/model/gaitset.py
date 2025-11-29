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
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(b, t, self.output_dim)
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
        pooled = [features.mean(dim=1), features.max(dim=1).values]
        temporal = features.transpose(1, 2)  # (B, D, T)
        for bins in self.pyramid_bins:
            pooled_bins = F.adaptive_avg_pool1d(temporal, output_size=bins)
            pooled.append(pooled_bins.transpose(1, 2).reshape(features.size(0), -1))
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
        self.backbone = GaitSetBackbone(in_channels=in_channels, feature_dims=frame_feature_dims)
        self.pool = SetPooling(self.backbone.output_dim, pyramid_bins=pyramid_bins)
        self.embedding_dim = self.pool.output_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and pooled embeddings for an input clip shaped (B, T, C, H, W)."""
        frame_features = self.backbone(clip)
        embedding = self.pool(frame_features)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding

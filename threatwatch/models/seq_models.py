"""
Sequence classifiers for multivariate physiological windows.

All models accept input of shape ``(batch, time, channels)`` and return raw
logits of shape ``(batch, num_classes)`` so they can be used with
``CrossEntropyLoss`` interchangeably.

Implemented:
  * ``cnn``              - 1D convolutional encoder + global pooling (baseline)
  * ``bilstm``           - bidirectional LSTM + attention pooling
  * ``transformer``      - encoder-only temporal transformer
  * ``cnn_transformer``  - convolutional tokenizer + transformer encoder
"""

from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class CNN1DClassifier(nn.Module):
    """Compact 1D CNN. Uses dilations to enlarge the receptive field cheaply."""

    def __init__(self, n_channels: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, kernel_size=3, padding=4, dilation=2),
            nn.BatchNorm1d(width * 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(width * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # (B, C, T)
        return self.head(self.net(x))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with additive attention pooling over time."""

    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_channels)
        self.proj = nn.Linear(n_channels, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(self.input_norm(x)))
        out, _ = self.lstm(h)                      # (B, T, 2H)
        w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        pooled = (out * w).sum(dim=1)              # (B, 2H)
        return self.head(pooled)


class TransformerClassifier(nn.Module):
    """Encoder-only temporal transformer with mean pooling."""

    def __init__(self, n_channels: int, n_classes: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 4, ff: int = 128,
                 dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_channels)
        self.proj = nn.Linear(n_channels, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.input_norm(x))
        h = self.pos(h)
        h = self.norm(self.encoder(h))
        return self.head(h.mean(dim=1))


class CNNTransformerClassifier(nn.Module):
    """
    Convolutional tokenizer + transformer encoder.

    The strided conv front-end downsamples time before attention, which cuts
    the O(T^2) attention cost roughly 16x and gives the model a local
    multi-scale feature extractor before global temporal mixing.
    """

    def __init__(self, n_channels: int, n_classes: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 4, ff: int = 128,
                 dropout: float = 0.1, conv_stride: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, d_model // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 2), nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, stride=conv_stride),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model, max_len=4096, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T/stride, d_model)
        h = self.pos(h)
        h = self.norm(self.encoder(h))
        return self.head(h.mean(dim=1))


class PatchViTClassifier(nn.Module):
    """
    A genuine Vision-Transformer formulation applied to a 1-D physiological
    series: the window is split into non-overlapping patches, each patch is
    embedded with a strided convolution (the standard ViT "patchify"), a learned
    CLS token is prepended, and an encoder-only transformer classifies from the
    CLS representation.

    This exists so the project can evaluate a **ViT** from the task brief on a
    1-D signal honestly, rather than claiming an image ViT applies.
    """

    def __init__(self, n_channels: int, n_classes: int, patch: int = 32,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 ff: int = 128, dropout: float = 0.1, max_patches: int = 512):
        super().__init__()
        self.patch = int(patch)
        self.embed = nn.Conv1d(n_channels, d_model, kernel_size=self.patch, stride=self.patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                      # (B, C, T)
        pad = (-h.size(-1)) % self.patch
        if pad:
            h = F.pad(h, (0, pad))
        h = self.embed(h).transpose(1, 2)          # (B, P, d_model)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, : h.size(1)]
        h = self.norm(self.encoder(h))
        return self.head(h[:, 0])


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn": CNN1DClassifier,
    "bilstm": BiLSTMClassifier,
    "transformer": TransformerClassifier,
    "cnn_transformer": CNNTransformerClassifier,
    "patch_vit": PatchViTClassifier,
}


def build_model(name: str, n_channels: int, n_classes: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_channels, n_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

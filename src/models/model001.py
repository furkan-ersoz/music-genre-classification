"""
model001 — EfficientNet-B0 + Temporal Attention

Architecture:
    - EfficientNet-B0 pretrained on ImageNet (frozen early layers)
    - Global feature map reshaped as time sequence
    - Multi-head self-attention over temporal dimension
    - FC classification head

Rationale:
    Transfer learning gives strong visual features from day one.
    Temporal attention lets the model focus on the most genre-discriminative
    time frames in the spectrogram, rather than averaging everything equally.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class EfficientNetTemporalAttention(nn.Module):
    def __init__(self, num_classes: int, num_heads: int = 4,
                 dropout: float = 0.3, freeze_until: int = 4):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze early layers, fine-tune later ones
        features = list(backbone.features.children())
        for i, layer in enumerate(features):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

        self.backbone = backbone.features   # output: (B, 1280, H', W')
        self.pool_h   = nn.AdaptiveAvgPool2d((7, 1))  # collapse width → (B, 1280, 7, 1)

        # ── Temporal attention ────────────────────────────────────────────────
        embed_dim = 1280
        self.proj    = nn.Linear(embed_dim, 256)   # reduce dim before attention
        self.attn1   = TemporalAttention(256, num_heads=num_heads, dropout=dropout)
        self.attn2   = TemporalAttention(256, num_heads=num_heads, dropout=dropout)

        # ── Head ──────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Backbone
        x = self.backbone(x)              # (B, 1280, H', W')
        x = self.pool_h(x)                # (B, 1280, 7, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, 7, 1280)

        # Project + attend
        x = self.proj(x)                  # (B, 7, 256)
        x = self.attn1(x)
        x = self.attn2(x)

        # Aggregate temporal dimension
        x = x.mean(dim=1)                 # (B, 256)

        return self.classifier(x)


def get_model(num_classes: int, **kwargs) -> nn.Module:
    return EfficientNetTemporalAttention(
        num_classes=num_classes,
        num_heads=kwargs.get("num_heads", 4),
        dropout=kwargs.get("dropout", 0.3),
        freeze_until=kwargs.get("freeze_until", 2),
    )
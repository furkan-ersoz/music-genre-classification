"""
model002 — ResNet18 + Bidirectional GRU Head

Architecture:
    - ResNet18 pretrained on ImageNet (frozen early layers)
    - Feature map columns treated as time steps → sequence
    - Two-layer Bidirectional GRU over the sequence
    - Attention pooling over GRU outputs
    - FC classification head

Rationale:
    ResNet18 extracts rich spatial features from the spectrogram.
    Reshaping the feature map as a time sequence and passing it through
    a BiGRU lets the model capture temporal patterns across frequency bands —
    something a pure CNN averages away. Attention pooling focuses on the
    most informative time steps rather than taking a naive mean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class AttentionPooling(nn.Module):
    """Soft attention over sequence dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        weights = F.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * weights).sum(dim=1)              # (B, H)


class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes: int, gru_hidden: int = 256,
                 gru_layers: int = 2, dropout: float = 0.3,
                 freeze_until: int = 6):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove avgpool and fc — keep only feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Output: (B, 512, H', W')

        # Freeze early layers
        children = list(self.backbone.children())
        for i, layer in enumerate(children):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

        # ── BiGRU ─────────────────────────────────────────────────────────────
        # Collapse height dimension, treat width as time steps
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, 512, 1, W')

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Attention pooling ─────────────────────────────────────────────────
        self.attn_pool = AttentionPooling(gru_hidden * 2)

        # ── Head ──────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # CNN features
        x = self.backbone(x)              # (B, 512, H', W')
        x = self.freq_pool(x)             # (B, 512, 1, W')
        x = x.squeeze(2).permute(0, 2, 1) # (B, W', 512)  — W' as time steps

        # BiGRU
        x, _ = self.gru(x)               # (B, W', gru_hidden*2)

        # Attention pooling
        x = self.attn_pool(x)            # (B, gru_hidden*2)

        return self.classifier(x)


def get_model(num_classes: int, **kwargs) -> nn.Module:
    return ResNetBiGRU(
        num_classes=num_classes,
        gru_hidden=kwargs.get("gru_hidden", 256),
        gru_layers=kwargs.get("gru_layers", 2),
        dropout=kwargs.get("dropout", 0.3),
        freeze_until=kwargs.get("freeze_until", 6),
    )
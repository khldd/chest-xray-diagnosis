"""DenseNet121 factory with a 14-way multi-label head."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """DenseNet121 with classifier = Linear(1024, 14).

    The state_dict layout matches the prototype's best_model.pth.
    freeze_backbone=True trains only the new head (baseline experiment).
    """
    model = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier = nn.Linear(in_features=1024, out_features=NUM_CLASSES)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model

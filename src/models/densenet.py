"""DenseNet121 factory (kept for backward compatibility; see src/models/factory.py)."""
from __future__ import annotations

import torch.nn as nn

from src.models.factory import build_model as _build


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """DenseNet121 with classifier = Linear(1024, 14).

    The state_dict layout matches the prototype's best_model.pth.
    freeze_backbone=True trains only the new head (baseline experiment).
    """
    return _build("densenet121", pretrained, freeze_backbone)

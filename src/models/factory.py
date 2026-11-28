"""Model factory: several ImageNet CNNs with a 14-way multi-label head.

Every architecture is built from torchvision only (no extra dependency). Each entry knows
1. how to replace its classification head with Linear(features, 14),
2. which parameters form the head (everything else is the backbone, frozen on request),
3. which layer Grad-CAM should hook (last spatial feature map before pooling).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


@dataclass(frozen=True)
class Arch:
    build: Callable[[bool], nn.Module]  # pretrained -> model with a 14-way head
    head_prefix: str  # state_dict prefix of the head parameters
    cam_layer: Callable[[nn.Module], nn.Module]
    display: str


def _densenet121(pretrained: bool) -> nn.Module:
    m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)  # 1024 -> 14
    return m


def _resnet50(pretrained: bool) -> nn.Module:
    m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)  # 2048 -> 14
    return m


def _efficientnet_b0(pretrained: bool) -> nn.Module:
    m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)  # 1280 -> 14
    return m


def _convnext_tiny(pretrained: bool) -> nn.Module:
    m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)  # 768 -> 14
    return m


ARCHS: dict[str, Arch] = {
    "densenet121": Arch(_densenet121, "classifier.", lambda m: m.features.norm5, "DenseNet121"),
    "resnet50": Arch(_resnet50, "fc.", lambda m: m.layer4[-1], "ResNet50"),
    "efficientnet_b0": Arch(
        _efficientnet_b0, "classifier.", lambda m: m.features[-1], "EfficientNet-B0"
    ),
    "convnext_tiny": Arch(_convnext_tiny, "classifier.", lambda m: m.features[-1], "ConvNeXt-Tiny"),
}


def get_arch(name: str) -> Arch:
    if name not in ARCHS:
        raise ValueError(f"Unknown architecture {name!r}; choose from {sorted(ARCHS)}")
    return ARCHS[name]


def build_model(
    name: str = "densenet121", pretrained: bool = True, freeze_backbone: bool = False
) -> nn.Module:
    """Build `name` with a 14-way head. freeze_backbone=True leaves only the head trainable."""
    arch = get_arch(name)
    model = arch.build(pretrained)
    if freeze_backbone:
        for pname, p in model.named_parameters():
            p.requires_grad = pname.startswith(arch.head_prefix)
    return model


def cam_target_layer(model: nn.Module, name: str = "densenet121") -> nn.Module:
    return get_arch(name).cam_layer(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""Grad-CAM heatmaps and overlays for any architecture in src/models/factory.py."""
from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import DISEASES, IMAGENET_MEAN, IMAGENET_STD
from src.models.factory import cam_target_layer


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] float image in 0..1."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


def gradcam_heatmaps(
    model, images: torch.Tensor, class_indices: list[int], arch: str = "densenet121"
) -> np.ndarray:
    """Grad-CAM for a batch: images [B,3,H,W], one class per image -> [B,H,W] in 0..1."""
    device = next(model.parameters()).device
    # With a frozen backbone no parameter before the target layer needs a gradient, so autograd
    # would never reach that layer. Asking for the input's gradient keeps the whole path alive.
    images = images.detach().to(device).requires_grad_(True)
    with GradCAM(model=model, target_layers=[cam_target_layer(model, arch)]) as cam:
        return cam(
            input_tensor=images,
            targets=[ClassifierOutputTarget(i) for i in class_indices],
        )


def gradcam_overlay(
    model, image_tensor: torch.Tensor, disease: str, arch: str = "densenet121"
) -> np.ndarray:
    """Return an RGB uint8 heatmap overlay for one disease. image_tensor is [3,H,W]."""
    heat = gradcam_heatmaps(model, image_tensor.unsqueeze(0), [DISEASES.index(disease)], arch)[0]
    return show_cam_on_image(denormalize(image_tensor).astype(np.float32), heat, use_rgb=True)

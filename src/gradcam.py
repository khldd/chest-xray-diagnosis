"""Grad-CAM overlays for the DenseNet121 model (target layer: features.norm5)."""
from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import DISEASES, IMAGENET_MEAN, IMAGENET_STD


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] float image in 0..1."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)


def gradcam_overlay(model, image_tensor: torch.Tensor, disease: str) -> np.ndarray:
    """Return an RGB uint8 heatmap overlay for one disease. image_tensor is [3,H,W]."""
    device = next(model.parameters()).device
    with GradCAM(model=model, target_layers=[model.features.norm5]) as cam:
        heat = cam(
            input_tensor=image_tensor.unsqueeze(0).to(device),
            targets=[ClassifierOutputTarget(DISEASES.index(disease))],
        )[0]
    return show_cam_on_image(denormalize(image_tensor).astype(np.float32), heat, use_rgb=True)

"""NumPy-only preprocessing, equivalent to src.data.transforms.eval_transforms.

Kept free of torch so the ONNX fast path stays lightweight. tests/test_api.py checks that it
matches the torchvision pipeline used at training/evaluation time.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.config import IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """PIL image -> float32 array of shape [1, 3, size, size], ImageNet-normalized."""
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

"""Model metadata and ONNX Runtime inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from api import settings
from api.preprocess import preprocess
from src.config import DISEASES


class ModelUnavailable(RuntimeError):
    """The model files needed for this operation are missing."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(onnx_path: Path, test_metrics_path: Path) -> dict:
    """Static facts about the served model (test_mean_auc is null until the test eval exists)."""
    test_auc = None
    if test_metrics_path.exists():
        test_auc = json.loads(test_metrics_path.read_text(encoding="utf-8")).get("mean_auc")
    return {
        "model_name": "densenet121",
        "task": "multi-label classification, 14 thoracic diseases (NIH ChestX-ray14)",
        "classes": DISEASES,
        "input_size": settings.IMG_SIZE,
        "val_mean_auc": settings.VAL_MEAN_AUC,
        "test_mean_auc": test_auc,
        "onnx_sha256": file_sha256(onnx_path) if onnx_path.exists() else None,
    }


class OnnxService:
    """Fast-path inference with ONNX Runtime (CPU).

    Any object with the same `classes`, `metadata`, `predict` interface can stand in for it,
    which is how the API tests avoid needing the real weights.
    """

    classes = DISEASES

    def __init__(
        self,
        onnx_path: Path = settings.MODEL_ONNX,
        test_metrics_path: Path = settings.TEST_METRICS,
    ):
        if not onnx_path.exists():
            raise ModelUnavailable(f"ONNX model not found: {onnx_path}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.metadata = build_metadata(onnx_path, test_metrics_path)

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return 14 sigmoid probabilities in DISEASES order."""
        logits = self.session.run(None, {"image": preprocess(image, settings.IMG_SIZE)})[0][0]
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

"""Model metadata (and, in later commits, inference)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from api import settings
from src.config import DISEASES


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

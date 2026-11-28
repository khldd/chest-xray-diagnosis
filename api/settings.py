"""API configuration, overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_ONNX = Path(os.getenv("MODEL_ONNX", ROOT / "models" / "best_model.onnx"))
MODEL_PTH = Path(os.getenv("MODEL_PTH", ROOT / "models" / "best_model.pth"))
TEST_METRICS = Path(os.getenv("TEST_METRICS", ROOT / "results" / "test_metrics.json"))
# Several models (architecture benchmark): ids, files, temperatures, metrics. Optional.
REGISTRY = Path(os.getenv("MODEL_REGISTRY", MODEL_ONNX.parent / "registry.json"))

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

IMG_SIZE = 224
# Mean AUC of the original checkpoint on the validation split (used without a registry).
VAL_MEAN_AUC = 0.8309

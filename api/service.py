"""Model registry, ONNX Runtime inference, calibration, ensemble and Grad-CAM.

Models are listed in models/registry.json (see `load_registry`). Each entry names an
architecture, its ONNX file (fast path for /predict), its PyTorch weights (Grad-CAM) and a
temperature T fitted on the validation split: calibrated probability = sigmoid(logit / T).
Without a registry the API serves the single original DenseNet121 (T = 1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from api import settings
from api.preprocess import preprocess
from src.config import DISEASES

ENSEMBLE = "ensemble"


class ModelUnavailable(RuntimeError):
    """The model files needed for this operation are missing."""


class UnknownModel(ValueError):
    """The requested model id is not served."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_entry() -> dict:
    """The original single-model setup, used when there is no registry."""
    test_auc = None
    if settings.TEST_METRICS.exists():
        test_auc = json.loads(settings.TEST_METRICS.read_text(encoding="utf-8")).get("mean_auc")
    return {
        "id": "densenet121",
        "arch": "densenet121",
        "display_name": "DenseNet121",
        "onnx": str(settings.MODEL_ONNX),
        "pth": str(settings.MODEL_PTH),
        "temperature": 1.0,
        "val_mean_auc": settings.VAL_MEAN_AUC,
        "test_mean_auc": test_auc,
    }


def load_registry(path: Path | None = None) -> list[dict]:
    """Entries from registry.json (file paths relative to its folder), or the default model."""
    path = path or settings.REGISTRY
    if not path.exists():
        return [default_entry()]
    entries = json.loads(path.read_text(encoding="utf-8"))["models"]
    for e in entries:
        for key in ("onnx", "pth"):
            if e.get(key) and not Path(e[key]).is_absolute():
                e[key] = str(path.parent / e[key])
        e.setdefault("temperature", 1.0)
        e.setdefault("display_name", e["id"])
    return entries


def build_metadata(entry: dict, ensemble_ids: list[str] | None = None) -> dict:
    """Static facts about one served model (kept compatible with the single-model API)."""
    onnx_path = Path(entry["onnx"])
    return {
        "model_id": entry["id"],
        "model_name": entry["arch"],
        "display_name": entry.get("display_name", entry["id"]),
        "task": "multi-label classification, 14 thoracic diseases (NIH ChestX-ray14)",
        "classes": DISEASES,
        "input_size": settings.IMG_SIZE,
        "val_mean_auc": entry.get("val_mean_auc"),
        "test_mean_auc": entry.get("test_mean_auc"),
        "temperature": entry.get("temperature", 1.0),
        "calibrated": entry.get("temperature", 1.0) != 1.0,
        "n_parameters": entry.get("n_parameters"),
        "onnx_sha256": file_sha256(onnx_path) if onnx_path.exists() else None,
        "ensemble_of": ensemble_ids,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class ServedModel:
    """One model: ONNX session for prediction, PyTorch weights loaded lazily for Grad-CAM."""

    def __init__(self, entry: dict):
        self.entry = entry
        self.id, self.arch = entry["id"], entry["arch"]
        self.temperature = float(entry.get("temperature", 1.0))
        onnx_path = Path(entry["onnx"])
        if not onnx_path.exists():
            raise ModelUnavailable(f"ONNX model not found: {onnx_path}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.metadata = build_metadata(entry)
        self._torch_model = None

    def logits(self, image: Image.Image) -> np.ndarray:
        return self.session.run(None, {"image": preprocess(image, settings.IMG_SIZE)})[0][0]

    def predict(self, image: Image.Image) -> np.ndarray:
        """14 calibrated probabilities in DISEASES order."""
        return sigmoid(self.logits(image) / self.temperature)

    def torch_model(self):
        if self._torch_model is None:
            pth = Path(self.entry.get("pth") or "")
            if not pth.is_file():
                raise ModelUnavailable(f"PyTorch weights not found: {pth}")
            import torch

            from src.models.factory import build_model

            model = build_model(self.arch, pretrained=False)
            model.load_state_dict(torch.load(pth, map_location="cpu"))
            self._torch_model = model.eval()
        return self._torch_model

    def heatmap(self, image: Image.Image, disease: str) -> np.ndarray:
        """Grad-CAM heatmap [224,224] in 0..1. Needs gradients, so it uses PyTorch."""
        from src.data.transforms import eval_transforms
        from src.gradcam import gradcam_heatmaps

        tensor = eval_transforms(settings.IMG_SIZE)(image.convert("RGB"))
        cls = [DISEASES.index(disease)]
        return gradcam_heatmaps(self.torch_model(), tensor.unsqueeze(0), cls, self.arch)[0]


def overlay(image: Image.Image, heat: np.ndarray) -> np.ndarray:
    """RGB uint8 overlay of a heatmap on the (resized) X-ray."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from src.data.transforms import eval_transforms
    from src.gradcam import denormalize

    tensor = eval_transforms(settings.IMG_SIZE)(image.convert("RGB"))
    return show_cam_on_image(denormalize(tensor).astype(np.float32), heat, use_rgb=True)


class ModelHub:
    """Every model in the registry whose files exist, plus their ensemble (mean probability).

    Any object with the same `classes`, `metadata`, `model_ids`, `models_info`, `predict` and
    `visualize` interface can stand in for it, which is how the API tests avoid real weights.
    """

    classes = DISEASES

    def __init__(self, registry: Path | None = None):
        entries = load_registry(registry)
        self.models: dict[str, ServedModel] = {}
        for e in entries:
            try:
                self.models[e["id"]] = ServedModel(e)
            except ModelUnavailable:
                continue  # listed but not downloaded: skip it, serve the rest
        if not self.models:
            raise ModelUnavailable("No model files found (see models/registry.json)")
        defaults = [e["id"] for e in entries if e.get("default") and e["id"] in self.models]
        self.default_id = defaults[0] if defaults else next(iter(self.models))
        chosen = [e["id"] for e in entries if e.get("in_ensemble") and e["id"] in self.models]
        self.ensemble_ids = chosen
        if len(self.ensemble_ids) < 2:
            self.ensemble_ids = list(self.models) if len(self.models) > 1 else []
        self.metadata = self.models[self.default_id].metadata

    @property
    def model_ids(self) -> list[str]:
        return list(self.models) + ([ENSEMBLE] if self.ensemble_ids else [])

    def models_info(self) -> list[dict]:
        info = [m.metadata | {"default": k == self.default_id} for k, m in self.models.items()]
        if self.ensemble_ids:
            info.append({
                "model_id": ENSEMBLE,
                "model_name": ENSEMBLE,
                "display_name": "Ensemble (mean of " + ", ".join(self.ensemble_ids) + ")",
                "ensemble_of": self.ensemble_ids,
                "default": False,
                "calibrated": all(self.models[i].temperature != 1.0 for i in self.ensemble_ids),
            })
        return info

    def _resolve(self, model: str | None) -> list[ServedModel]:
        model = model or self.default_id
        if model == ENSEMBLE and self.ensemble_ids:
            return [self.models[i] for i in self.ensemble_ids]
        if model not in self.models:
            raise UnknownModel(f"Unknown model: {model}. Available: {', '.join(self.model_ids)}")
        return [self.models[model]]

    def predict(self, image: Image.Image, model: str | None = None) -> np.ndarray:
        return np.mean([m.predict(image) for m in self._resolve(model)], axis=0)

    def visualize(self, image: Image.Image, disease: str, model: str | None = None) -> np.ndarray:
        """Grad-CAM overlay; for the ensemble, the mean of the members' heatmaps."""
        heats = [m.heatmap(image, disease) for m in self._resolve(model)]
        return overlay(image, np.mean(heats, axis=0))


# Backward-compatible name: the single-model service used by earlier code and docs.
OnnxService = ModelHub

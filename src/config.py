"""Configuration loading and shared constants."""
from __future__ import annotations

from pathlib import Path

import yaml

# The 14 disease labels, alphabetical, "No Finding" excluded (it is the all-zero vector).
DISEASES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(DISEASES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: str | Path) -> dict:
    """Load a YAML config, filling gaps from configs/base.yaml."""
    base_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if Path(path).resolve() != base_path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value

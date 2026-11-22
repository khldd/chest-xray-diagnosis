"""NIH ChestX-ray14 label preparation and patient-level splitting."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import DISEASES


def load_metadata(root: str) -> pd.DataFrame:
    """Read Data_Entry_2017.csv and tag each row with its official split."""
    df = pd.read_csv(os.path.join(root, "Data_Entry_2017.csv"))
    with open(os.path.join(root, "train_val_list.txt"), encoding="utf-8") as f:
        train_val = set(f.read().splitlines())
    df["split"] = df["Image Index"].map(lambda x: "train_val" if x in train_val else "test")
    return df


def encode_labels(finding_labels: pd.Series) -> np.ndarray:
    """Multi-hot encode 'A|B' strings into [n, 14]. 'No Finding' becomes all zeros."""
    index = {name: i for i, name in enumerate(DISEASES)}
    out = np.zeros((len(finding_labels), len(DISEASES)), dtype=np.float32)
    for row, text in enumerate(finding_labels):
        for name in text.split("|"):
            if name in index:
                out[row, index[name]] = 1.0
    return out


def patient_split(
    df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) with no Patient ID in both sets."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["Patient ID"]))
    return train_idx, val_idx


def build_image_lookup(root: str) -> dict[str, str]:
    """Map image filename -> full path (images are spread over images_*/images/)."""
    paths = glob.glob(os.path.join(root, "images_*", "images", "*.png"))
    return {os.path.basename(p): p for p in paths}


def make_splits(root: str, val_fraction: float = 0.15, seed: int = 42) -> dict:
    """Build train/val/test frames and label matrices as in the prototype notebook."""
    df = load_metadata(root)
    train_val = df[df["split"] == "train_val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    train_idx, val_idx = patient_split(train_val, val_fraction, seed)
    tv_labels = encode_labels(train_val["Finding Labels"])
    return {
        "train": (train_val.iloc[train_idx].reset_index(drop=True), tv_labels[train_idx]),
        "val": (train_val.iloc[val_idx].reset_index(drop=True), tv_labels[val_idx]),
        "test": (test, encode_labels(test["Finding Labels"])),
    }

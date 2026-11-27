"""Measure Grad-CAM localisation against the NIH expert bounding boxes (BBox_List_2017.csv).

The NIH release has 984 boxes drawn by radiologists on 880 images, for 8 findings. For each box
we compute the Grad-CAM heatmap of that finding and report:
  hit      pointing game: the heatmap's maximum falls inside the box
  energy   share of the heatmap's total mass that lies inside the box
  iou      IoU between the box and the region where heat >= 0.5 (heat is scaled to 0..1)
and two references that need no model: `area` (box area / image area, which is the hit rate of
a pointer placed at random) and `center_hit` (a pointer always placed at the image centre).
Only images outside the training split are scored.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from src.config import DISEASES
from src.gradcam import gradcam_heatmaps

BOX_NAMES = {"Infiltrate": "Infiltration"}  # BBox_List spelling -> Data_Entry spelling


def load_boxes(root: str, sizes: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Return one row per box: image, disease, x, y, w, h (original pixels), width, height."""
    raw = pd.read_csv(os.path.join(root, "BBox_List_2017.csv"))
    df = raw.iloc[:, :6].copy()
    df.columns = ["image", "disease", "x", "y", "w", "h"]
    df["disease"] = df["disease"].replace(BOX_NAMES)
    df = df[df["disease"].isin(DISEASES)].reset_index(drop=True)
    sizes = sizes or {}
    df["width"] = [sizes.get(i, (1024, 1024))[0] for i in df["image"]]
    df["height"] = [sizes.get(i, (1024, 1024))[1] for i in df["image"]]
    return df


def box_metrics(heat: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, float]:
    """heat [S,S] in 0..1, box (x0, y0, x1, y1) in heatmap pixels."""
    s_y, s_x = heat.shape
    x0, y0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x1, y1 = min(int(np.ceil(box[2])), s_x), min(int(np.ceil(box[3])), s_y)
    inside = np.zeros_like(heat, dtype=bool)
    inside[y0:y1, x0:x1] = True
    peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    hot = heat >= 0.5
    union = np.logical_or(hot, inside).sum()
    return {
        "hit": float(inside[peak_y, peak_x]),
        "energy": float(heat[inside].sum() / max(heat.sum(), 1e-8)),
        "iou": float(np.logical_and(hot, inside).sum() / max(union, 1)),
        "area": float(inside.mean()),
        "center_hit": float(inside[s_y // 2, s_x // 2]),
    }


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    keys = ["hit", "energy", "iou", "area", "center_hit"]
    per_disease = {
        name: {"n": int(len(g)), **{k: float(g[k].mean()) for k in keys}}
        for name, g in df.groupby("disease")
    }
    overall = {"n": int(len(df)), **{k: float(df[k].mean()) for k in keys}}
    # Mean over diseases too, so frequent findings do not dominate.
    macro = {k: float(np.mean([v[k] for v in per_disease.values()])) for k in keys}
    return {"overall": overall, "macro": macro, "per_disease": per_disease}


def score_localization(
    model,
    arch: str,
    boxes: pd.DataFrame,
    images,
    transform,
    img_size: int,
    allowed: set[str],
    batch_size: int = 16,
) -> dict:
    """images: dict filename -> path, or CachedImages. Returns summary + per-box rows."""
    from PIL import Image

    model.eval()
    keep = boxes[boxes["image"].isin(allowed)].reset_index(drop=True)
    rows: list[dict] = []
    for start in range(0, len(keep), batch_size):
        chunk = keep.iloc[start : start + batch_size]
        tensors = []
        for name in chunk["image"]:
            if hasattr(images, "open"):  # CachedImages
                img = images.open(name)
            else:
                img = Image.open(images[name]).convert("RGB")
            tensors.append(transform(img))
        classes = [DISEASES.index(d) for d in chunk["disease"]]
        heats = gradcam_heatmaps(model, torch.stack(tensors), classes, arch)
        for heat, (_, r) in zip(heats, chunk.iterrows(), strict=True):
            sx, sy = img_size / r["width"], img_size / r["height"]
            box = (r["x"] * sx, r["y"] * sy, (r["x"] + r["w"]) * sx, (r["y"] + r["h"]) * sy)
            rows.append({"image": r["image"], "disease": r["disease"], **box_metrics(heat, box)})
    result = summarize(rows) if rows else {"overall": {"n": 0}}
    result["n_boxes_total"] = int(len(boxes))
    result["n_excluded_in_train"] = int(len(boxes) - len(keep))
    result["boxes"] = rows
    return result

"""Per-class AUC-ROC evaluation, usable as a library or CLI."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import DISEASES


@torch.no_grad()
def predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels), both [n, 14]. Sigmoid is applied here."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def auc_per_class(probs: np.ndarray, labels: np.ndarray) -> tuple[dict[str, float], float]:
    aucs = {}
    for i, name in enumerate(DISEASES):
        try:
            aucs[name] = float(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:  # class has no positives (or no negatives) in this set
            aucs[name] = float("nan")
    return aucs, float(np.nanmean(list(aucs.values())))


def evaluate_auc(model, loader, device: str = "cuda") -> tuple[dict[str, float], float]:
    return auc_per_class(*predict(model, loader, device))


def main() -> None:
    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.data.dataset import ChestXrayDataset
    from src.data.prepare import build_image_lookup, make_splits
    from src.data.transforms import eval_transforms
    from src.models.factory import build_model

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on val or test")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--arch", default=None, help="override model.name from the config")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = make_splits(cfg["data"]["root"], cfg["data"]["val_fraction"], cfg["data"]["seed"])
    df, labels = splits[args.split]
    d = cfg["data"]
    lookup = build_image_lookup(d["root"])
    ds = ChestXrayDataset(df, labels, lookup, eval_transforms(d["img_size"]))
    loader = DataLoader(ds, batch_size=d["batch_size"], num_workers=d["num_workers"])

    arch = args.arch or cfg["model"]["name"]
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    aucs, mean_auc = evaluate_auc(model, loader, device)
    result = {"split": args.split, "n_images": len(ds), "mean_auc": mean_auc, "per_class_auc": aucs}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

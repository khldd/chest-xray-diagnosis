"""Training script: weighted BCE, per-epoch AUC, MLflow tracking, resumable checkpoints."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import ChestXrayDataset
from src.data.prepare import build_image_lookup, make_splits
from src.data.transforms import eval_transforms, train_transforms
from src.evaluate import evaluate_auc
from src.models.densenet import build_model


def make_criterion(kind: str, train_labels: np.ndarray, device: str) -> nn.Module:
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted_bce":
        pos = train_labels.sum(axis=0)
        neg = len(train_labels) - pos
        weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    raise ValueError(f"Unknown loss: {kind}")


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def main() -> None:
    import mlflow

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, t = cfg["data"], cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(t["output_dir"], exist_ok=True)

    splits = make_splits(d["root"], d["val_fraction"], d["seed"])
    lookup = build_image_lookup(d["root"])
    tr_df, tr_y = splits["train"]
    va_df, va_y = splits["val"]
    train_tf = train_transforms(d["img_size"], d["augmentation"])
    train_ds = ChestXrayDataset(tr_df, tr_y, lookup, train_tf)
    val_ds = ChestXrayDataset(va_df, va_y, lookup, eval_transforms(d["img_size"]))
    kw = {"batch_size": d["batch_size"], "num_workers": d["num_workers"]}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = build_model(cfg["model"]["pretrained"], cfg["model"]["freeze_backbone"]).to(device)
    criterion = make_criterion(t["loss"], tr_y, device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=t["lr"])

    last_path = os.path.join(t["output_dir"], "last_checkpoint.pt")
    best_path = os.path.join(t["output_dir"], "best_model.pth")
    start_epoch, best_auc = 0, 0.0
    if os.path.exists(last_path):  # full resume: weights + optimizer + epoch + best AUC
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_auc = ckpt["epoch"] + 1, ckpt["best_auc"]
        print(f"Resumed at epoch {start_epoch}, best AUC {best_auc:.4f}")

    mlflow.set_experiment(t["mlflow_experiment"])
    params = {f"data.{k}": v for k, v in d.items() if k != "root"}
    params |= {f"model.{k}": v for k, v in cfg["model"].items()}
    params |= {f"train.{k}": v for k, v in t.items()}
    aucs: dict[str, float] = {}
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(params)
        for epoch in range(start_epoch, t["epochs"]):
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            aucs, mean_auc = evaluate_auc(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{t['epochs']} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | mean AUC {mean_auc:.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_mean_auc": mean_auc},
                step=epoch,
            )
            if mean_auc > best_auc:
                best_auc = mean_auc
                torch.save(model.state_dict(), best_path)  # plain state_dict, as in the prototype
                mlflow.log_metric("best_val_auc", best_auc, step=epoch)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                },
                last_path,
            )
        for name, score in aucs.items():
            mlflow.log_metric(f"auc_{name}", score)


if __name__ == "__main__":
    main()

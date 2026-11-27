"""Training script: any architecture, per-epoch AUC, MLflow tracking, resumable checkpoints.

Optional extras (all off in configs/base.yaml, on in configs/bench/*):
  data.cache      decode + resize every image once into a memmap (training is decode-bound)
  train.amp       mixed precision on GPU
  train.predict   after training, save val/test predictions of the best checkpoint
  train.localize  after training, score Grad-CAM against the NIH expert boxes
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import ChestXrayDataset
from src.data.prepare import build_image_lookup, make_splits
from src.data.transforms import eval_transforms, train_transforms
from src.evaluate import auc_per_class, evaluate_auc, predict
from src.models.factory import build_model, count_parameters


def make_criterion(kind: str, train_labels: np.ndarray, device: str) -> nn.Module:
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted_bce":
        pos = np.maximum(train_labels.sum(axis=0), 1)  # avoid division by zero for an empty class
        neg = len(train_labels) - pos
        weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    raise ValueError(f"Unknown loss: {kind}")


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch; return a generator for the DataLoader shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> float:
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates.

    scaler (torch.amp.GradScaler) turns on mixed precision; None keeps full precision.
    """
    training = optimizer is not None
    model.train(training)
    total = 0.0
    amp = scaler is not None
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(images).float(), labels)
            if training:
                optimizer.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def _metadata_arrays(df, prefix: str) -> dict[str, np.ndarray]:
    """Patient id, age, sex and view for subgroup analysis (only columns that exist)."""
    cols = {"Patient ID": "patient", "Patient Age": "age", "Patient Gender": "sex",
            "View Position": "view"}
    out = {}
    for col, key in cols.items():
        if col in df.columns:
            values = df[col].to_numpy()
            out[f"{prefix}_{key}"] = values.astype(str) if values.dtype == object else values
    return out


def save_predictions(model, loaders: dict, frames: dict, device: str, out_path: str) -> dict:
    """Predict each split with `model` and save predictions.npz (same keys as the eval kernel)."""
    arrays, summary = {}, {}
    for split, loader in loaders.items():
        probs, labels = predict(model, loader, device)
        arrays[f"{split}_probs"], arrays[f"{split}_labels"] = probs, labels
        arrays[f"{split}_names"] = np.array(frames[split]["Image Index"].tolist())
        arrays |= _metadata_arrays(frames[split], split)
        aucs, mean_auc = auc_per_class(probs, labels)
        summary[split] = {"n_images": len(labels), "mean_auc": mean_auc, "per_class_auc": aucs}
        print(f"{split}: mean AUC {mean_auc:.4f} on {len(labels)} images", flush=True)
    np.savez_compressed(out_path, **arrays)
    return summary


def write_metrics(
    path: str,
    run_name: str | None,
    cfg: dict,
    history: list[dict],
    best_auc: float,
    best_aucs: dict[str, float],
) -> dict:
    """Write a self-contained summary of a run (collected from Kaggle kernel output)."""
    best_epoch = max(history, key=lambda h: h["val_mean_auc"])["epoch"] if history else None
    summary = {
        "run": run_name,
        "config": cfg,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "per_class_val_auc_at_best_epoch": best_aucs,
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    import mlflow

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data-root", default=None, help="override data.root from the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:  # Kaggle mounts the dataset at varying paths
        cfg["data"]["root"] = args.data_root
    d, t = cfg["data"], cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(t["output_dir"], exist_ok=True)
    generator = seed_everything(int(t.get("seed", 0)))
    arch = cfg["model"]["name"]

    splits = make_splits(d["root"], d["val_fraction"], d["seed"])
    lookup = build_image_lookup(d["root"])
    tr_df, tr_y = splits["train"]
    va_df, va_y = splits["val"]
    te_df, te_y = splits["test"]
    images = lookup
    if d.get("cache"):  # decode + resize every image once; tensors are identical (see cache.py)
        from src.data.cache import CachedImages, build_cache

        started = time.time()
        names = [*tr_df["Image Index"], *va_df["Image Index"], *te_df["Image Index"]]
        path, index = build_cache(names, lookup, d["img_size"], d["cache_dir"], d["cache_workers"])
        images = CachedImages(path, index, d["img_size"])
        print(f"cache ready: {len(index)} images in {time.time() - started:.0f}s", flush=True)
    eval_tf = eval_transforms(d["img_size"])
    train_tf = train_transforms(d["img_size"], d["augmentation"])
    train_ds = ChestXrayDataset(tr_df, tr_y, images, train_tf)
    val_ds = ChestXrayDataset(va_df, va_y, images, eval_tf)
    kw = {"batch_size": d["batch_size"], "num_workers": d["num_workers"]}
    if d["num_workers"] > 0:
        kw |= {"persistent_workers": True, "pin_memory": device == "cuda"}
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=generator, worker_init_fn=_seed_worker, **kw
    )
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = build_model(arch, cfg["model"]["pretrained"], cfg["model"]["freeze_backbone"])
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"model: {arch}, {n_params / 1e6:.1f}M parameters", flush=True)
    criterion = make_criterion(t["loss"], tr_y, device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=t["lr"])
    scaler = torch.amp.GradScaler("cuda") if t.get("amp") and device == "cuda" else None

    last_path = os.path.join(t["output_dir"], "last_checkpoint.pt")
    best_path = os.path.join(t["output_dir"], "best_model.pth")
    start_epoch, best_auc = 0, 0.0
    history: list[dict] = []
    best_aucs: dict[str, float] = {}
    if os.path.exists(last_path):  # full resume: weights + optimizer + epoch + best AUC + history
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_auc = ckpt["epoch"] + 1, ckpt["best_auc"]
        history, best_aucs = ckpt.get("history", []), ckpt.get("best_aucs", {})
        print(f"Resumed at epoch {start_epoch}, best AUC {best_auc:.4f}")

    mlflow.set_experiment(t["mlflow_experiment"])
    params = {f"data.{k}": v for k, v in d.items() if k != "root"}
    params |= {f"model.{k}": v for k, v in cfg["model"].items()}
    params |= {f"train.{k}": v for k, v in t.items()}
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(params)
        for epoch in range(start_epoch, t["epochs"]):
            started = time.time()
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss = run_epoch(model, val_loader, criterion, device, scaler=scaler)
            aucs, mean_auc = evaluate_auc(model, val_loader, device)
            seconds = round(time.time() - started)
            print(
                f"Epoch {epoch + 1}/{t['epochs']} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | mean AUC {mean_auc:.4f} | {seconds}s"
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mean_auc": mean_auc,
                    "seconds": seconds,
                }
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_mean_auc": mean_auc},
                step=epoch,
            )
            if mean_auc > best_auc:
                best_auc, best_aucs = mean_auc, aucs
                torch.save(model.state_dict(), best_path)  # plain state_dict, as in the prototype
                mlflow.log_metric("best_val_auc", best_auc, step=epoch)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                    "best_aucs": best_aucs,
                    "history": history,
                },
                last_path,
            )
        # Per-class AUCs of the BEST epoch (same epoch as best_val_auc), not the last one.
        for name, score in best_aucs.items():
            mlflow.log_metric(f"auc_{name}", score)

    metrics_path = os.path.join(t["output_dir"], "metrics.json")
    summary = write_metrics(metrics_path, args.run_name, cfg, history, best_auc, best_aucs)

    if not (t.get("predict") or t.get("localize")):
        return
    model.load_state_dict(torch.load(best_path, map_location=device))
    extra: dict = {"arch": arch, "n_parameters": n_params}
    if t.get("predict"):
        test_ds = ChestXrayDataset(te_df, te_y, images, eval_tf)
        loaders = {"val": val_loader, "test": DataLoader(test_ds, shuffle=False, **kw)}
        frames = {"val": va_df, "test": te_df}
        pred_path = os.path.join(t["output_dir"], "predictions.npz")
        extra["eval"] = save_predictions(model, loaders, frames, device, pred_path)
    if t.get("localize"):
        extra["localization"] = run_localization(model, arch, d, images, eval_tf, tr_df, t)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary | extra, f, indent=2)


def run_localization(model, arch, d, images, eval_tf, tr_df, t) -> dict:
    """Score Grad-CAM against the expert boxes on every image outside the training split."""
    from src.data.prepare import load_metadata
    from src.localize import load_boxes, score_localization

    if not os.path.exists(os.path.join(d["root"], "BBox_List_2017.csv")):
        print("localization skipped: BBox_List_2017.csv not found", flush=True)
        return {"skipped": True}
    meta = load_metadata(d["root"])
    sizes = {}
    if "OriginalImage[Width" in meta.columns:
        sizes = dict(zip(meta["Image Index"], zip(meta["OriginalImage[Width"], meta["Height]"],
                                                     strict=True), strict=True))
    boxes = load_boxes(d["root"], sizes)
    allowed = set(meta["Image Index"]) - set(tr_df["Image Index"])
    result = score_localization(model, arch, boxes, images, eval_tf, d["img_size"], allowed)
    with open(os.path.join(t["output_dir"], "localization.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    overall = result.get("overall", {})
    print(f"localization: {overall}", flush=True)
    return {k: v for k, v in result.items() if k != "boxes"}


if __name__ == "__main__":
    main()

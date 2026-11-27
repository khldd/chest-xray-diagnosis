"""End-to-end smoke test of src.train on a tiny synthetic NIH-style dataset (CPU, seconds)."""
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("mlflow")

from PIL import Image  # noqa: E402

from src import train  # noqa: E402
from src.config import DISEASES  # noqa: E402


def make_fake_nih(root, n_images=48):
    """Create Data_Entry_2017.csv, train_val_list.txt and images_001/images/*.png."""
    img_dir = root / "images_001" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, names = [], []
    for i in range(n_images):
        name = f"{i // 2:08d}_{i % 2:03d}.png"  # 2 images per patient
        names.append(name)
        labels = "No Finding" if i % 7 == 0 else "|".join(
            {DISEASES[i % 14], DISEASES[(i * 3) % 14]}
        )
        rows.append(f"{name},{labels},{i // 2}")
        arr = (rng.random((40, 40)) * 255).astype("uint8")
        Image.fromarray(arr).save(img_dir / name)
    (root / "Data_Entry_2017.csv").write_text(
        "Image Index,Finding Labels,Patient ID\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    (root / "train_val_list.txt").write_text("\n".join(names), encoding="utf-8")


def run_train(monkeypatch, tmp_path, epochs):
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(
        "data:\n  img_size: 64\n  batch_size: 8\n  num_workers: 0\n"
        "model:\n  pretrained: false\n"
        f"train:\n  epochs: {epochs}\n  loss: bce\n  output_dir: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    argv = ["train", "--config", str(cfg), "--data-root", str(tmp_path / "nih")]
    monkeypatch.setattr(sys, "argv", [*argv, "--run-name", "smoke"])
    train.main()
    return json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))


def test_train_writes_metrics_and_resumes(monkeypatch, tmp_path):
    make_fake_nih(tmp_path / "nih")
    monkeypatch.chdir(tmp_path)  # keeps the mlruns/ folder inside tmp

    first = run_train(monkeypatch, tmp_path, epochs=2)
    assert [h["epoch"] for h in first["history"]] == [1, 2]
    assert first["best_epoch"] in (1, 2)
    assert (tmp_path / "out" / "best_model.pth").exists()
    assert first["run"] == "smoke"

    # Resuming with a higher epoch count must continue from epoch 3, not restart.
    second = run_train(monkeypatch, tmp_path, epochs=3)
    assert [h["epoch"] for h in second["history"]] == [1, 2, 3]


def test_weighted_criterion_handles_empty_class():
    labels = np.zeros((10, 14), dtype=np.float32)
    labels[:5, 0] = 1  # classes 1..13 have no positives
    criterion = train.make_criterion("weighted_bce", labels, "cpu")
    assert torch.isfinite(criterion.pos_weight).all()


def test_write_metrics_picks_best_epoch(tmp_path):
    history = [
        {"epoch": 1, "val_mean_auc": 0.7},
        {"epoch": 2, "val_mean_auc": 0.8},
        {"epoch": 3, "val_mean_auc": 0.75},
    ]
    out = train.write_metrics(str(tmp_path / "m.json"), "r", {}, history, 0.8, {"Mass": 0.8})
    assert out["best_epoch"] == 2



def test_train_benchmark_extras(monkeypatch, tmp_path):
    """Cache + seed + predictions + localisation on a fake dataset with a test split and boxes."""
    root = tmp_path / "nih"
    make_fake_nih(root)
    names = (root / "train_val_list.txt").read_text(encoding="utf-8").split()
    (root / "train_val_list.txt").write_text("\n".join(names[:36]), encoding="utf-8")
    rows = ["Image Index,Finding Label,Bbox [x,y,w,h],,,"]
    rows += [f"{n},Infiltrate,5,5,20,20" for n in names[36:40]]
    rows += [f"{n},Mass,1,1,10,10" for n in names[:36]]  # mostly training images: excluded
    (root / "BBox_List_2017.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = tmp_path / "bench.yaml"
    out = tmp_path / "out"
    cfg.write_text(
        "data:\n  img_size: 64\n  batch_size: 8\n  num_workers: 0\n  cache: true\n"
        f"  cache_dir: {(tmp_path / 'cache').as_posix()}\n  cache_workers: 1\n"
        "model:\n  name: resnet50\n  pretrained: false\n"
        "train:\n  epochs: 1\n  loss: bce\n  seed: 3\n  predict: true\n  localize: true\n"
        f"  output_dir: {out.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg), "--data-root", str(root)])
    train.main()

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["arch"] == "resnet50" and metrics["n_parameters"] > 1e7
    preds = np.load(out / "predictions.npz")
    assert preds["test_probs"].shape == (12, 14)
    assert len(preds["val_names"]) == len(preds["val_labels"])
    assert "test_patient" in preds.files
    loc = json.loads((out / "localization.json").read_text(encoding="utf-8"))
    assert loc["n_boxes_total"] == 40 and loc["n_excluded_in_train"] >= 20
    assert loc["overall"]["n"] == 40 - loc["n_excluded_in_train"]


def test_seed_makes_shuffling_reproducible():
    a = torch.randperm(100, generator=train.seed_everything(7))
    b = torch.randperm(100, generator=train.seed_everything(7))
    assert torch.equal(a, b)

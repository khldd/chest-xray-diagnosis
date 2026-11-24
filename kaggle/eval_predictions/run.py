"""Kaggle kernel script: save the final model's probabilities on the val and test splits.

CPU is enough (no GPU quota used). Output: /kaggle/working/predictions.npz with, for each of
val and test: probs [n, 14], labels [n, 14], image names. Used locally to draw ROC / PR curves,
confusion matrices and calibration plots without re-running the model.
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/khldd/chest-xray-diagnosis.git"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.chdir("/kaggle/working")
sh(f"git clone --depth 1 --branch main {REPO} repo")
os.chdir("repo")
sys.path.insert(0, os.getcwd())

csv = glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)
assert csv, "NIH dataset not found"
root = os.path.dirname(csv[0])
weights = glob.glob("/kaggle/input/**/best_model.pth", recursive=True)
assert weights, "best_model.pth not found"
print("data root:", root, "| weights:", weights[0], flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import ChestXrayDataset  # noqa: E402
from src.data.prepare import build_image_lookup, make_splits  # noqa: E402
from src.data.transforms import eval_transforms  # noqa: E402
from src.evaluate import auc_per_class, predict  # noqa: E402
from src.models.densenet import build_model  # noqa: E402

cfg = load_config("configs/base.yaml")
d = cfg["data"]
torch.set_num_threads(os.cpu_count() or 4)
splits = make_splits(root, d["val_fraction"], d["seed"])
lookup = build_image_lookup(root)
model = build_model(pretrained=False)
model.load_state_dict(torch.load(weights[0], map_location="cpu"))

out = {}
for split in ("test", "val"):
    df, labels = splits[split]
    ds = ChestXrayDataset(df, labels, lookup, eval_transforms(d["img_size"]))
    loader = DataLoader(ds, batch_size=64, num_workers=4)
    probs, y = predict(model, loader, "cpu")
    _, mean_auc = auc_per_class(probs, y)
    print(f"{split}: n={len(ds)} mean AUC {mean_auc:.4f}", flush=True)
    names = df["Image Index"].to_numpy().astype(str)
    out.update({f"{split}_probs": probs, f"{split}_labels": y, f"{split}_names": names})
    np.savez_compressed("/kaggle/working/predictions.npz", **out)  # checkpoint after each split

os.chdir("/kaggle/working")
sh("rm -rf repo")
sh("ls -la /kaggle/working")

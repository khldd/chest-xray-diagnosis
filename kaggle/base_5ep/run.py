"""Kaggle kernel script: train ONE ablation config and keep only the small outputs.

This is a template. scripts/make_kaggle_kernels.py fills in RUN and BRANCH and writes one
kernel folder per run under kaggle/. Outputs (downloadable with `kaggle kernels output`):
  <RUN>/metrics.json   per-epoch history, best epoch, per-class val AUC at the best epoch
  <RUN>/best_model.pth best weights (about 28 MB)
"""
import glob
import os
import subprocess

RUN = "base_5ep"
BRANCH = "exp/ablation-runs"
REPO = "https://github.com/khldd/chest-xray-diagnosis.git"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.chdir("/kaggle/working")
sh(f"git clone --depth 1 --branch {BRANCH} {REPO} repo")
os.chdir("repo")
sh("pip install -q mlflow pyyaml")  # torch, torchvision, pandas, sklearn are preinstalled

csv = glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)
assert csv, "NIH dataset not found under /kaggle/input"
data_root = os.path.dirname(csv[0])
print("data root:", data_root, flush=True)

sh(f"python -m src.train --config configs/{RUN}.yaml --run-name {RUN} --data-root {data_root}")

# Keep the outputs small: drop the optimizer checkpoint and the cloned repo (it holds mlruns/).
out = f"/kaggle/working/{RUN}"
sh(f"rm -f {out}/last_checkpoint.pt")
os.chdir("/kaggle/working")
sh("rm -rf repo")
sh(f"ls -la {out}")

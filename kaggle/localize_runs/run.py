"""Kaggle CPU kernel: re-score Grad-CAM localisation for finished benchmark runs.

The first two benchmark runs scaled the NIH boxes by the original scan size instead of the
1024 px PNG size; this kernel recomputes their localization.json from the saved weights.
Outputs: /kaggle/working/<run>/localization.json
"""
import glob
import os
import subprocess

BRANCH = "feat/multi-arch"
REPO = "https://github.com/khldd/chest-xray-diagnosis.git"
RUNS = {"densenet121_bce": "densenet121", "convnext_tiny_bce": "convnext_tiny"}


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.chdir("/kaggle/working")
sh(f"git clone --depth 1 --branch {BRANCH} {REPO} repo")
sh("pip install -q pyyaml grad-cam")
data_root = os.path.dirname(glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)[0])
for run, arch in RUNS.items():
    found = glob.glob(f"/kaggle/input/**/{run}/best_model.pth", recursive=True)
    if not found:
        print("weights not found for", run, flush=True)
        continue
    os.makedirs(f"/kaggle/working/{run}", exist_ok=True)
    sh(f"cd repo && python -m src.localize --weights {found[0]} --arch {arch} "
       f"--data-root {data_root} --out /kaggle/working/{run}/localization.json")
sh("rm -rf /kaggle/working/repo")

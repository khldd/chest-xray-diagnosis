"""Generate kaggle/<run>/ kernel folders (script + metadata) from kaggle/run_template.py.

Usage (repo root):  python scripts/make_kaggle_kernels.py [--branch main] [run ...]
Then launch one:    kaggle kernels push -p kaggle/base_5ep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_USER = "khaledayedi"
DEFAULT_RUNS = ["base_5ep", "baseline_frozen", "unweighted_bce", "strong_aug"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--branch", default="main", help="git branch the kernel clones")
    args = ap.parse_args()

    template = (ROOT / "kaggle" / "run_template.py").read_text(encoding="utf-8")
    for run in args.runs:
        assert (ROOT / "configs" / f"{run}.yaml").exists(), f"no config for {run}"
        slug = "cxr-ablation-" + run.replace("_", "-")
        folder = ROOT / "kaggle" / run
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.py").write_text(
            template.replace("__RUN__", run).replace("__BRANCH__", args.branch), encoding="utf-8"
        )
        metadata = {
            "id": f"{KAGGLE_USER}/{slug}",
            "title": slug,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": ["nih-chest-xrays/data"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        meta_path = folder / "kernel-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("wrote", folder.relative_to(ROOT))


if __name__ == "__main__":
    main()

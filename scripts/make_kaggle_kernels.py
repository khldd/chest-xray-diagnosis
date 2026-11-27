"""Generate kaggle/<run>/ kernel folders (script + metadata) from kaggle/run_template.py.

Usage (repo root):  python scripts/make_kaggle_kernels.py [--branch main] [--owner USER] [run ...]
Then launch one:    kaggle kernels push -p kaggle/base_5ep

A run name is a config in configs/<run>.yaml (ablations, kernel cxr-ablation-*) or in
configs/bench/<run>.yaml (architecture study, kernel cxr-bench-*). --owner puts the kernel
under another Kaggle account (a teammate pushing with their own credentials).
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
    ap.add_argument("--owner", default=KAGGLE_USER, help="Kaggle account that runs the kernel")
    args = ap.parse_args()

    template = (ROOT / "kaggle" / "run_template.py").read_text(encoding="utf-8")
    for run in args.runs:
        if (ROOT / "configs" / f"{run}.yaml").exists():
            config, slug, folder = f"configs/{run}.yaml", "cxr-ablation-", ROOT / "kaggle" / run
        elif (ROOT / "configs" / "bench" / f"{run}.yaml").exists():
            config, slug = f"configs/bench/{run}.yaml", "cxr-bench-"
            folder = ROOT / "kaggle" / "bench" / run
        else:
            raise SystemExit(f"no config for {run}")
        slug += run.replace("_", "-")
        folder.mkdir(parents=True, exist_ok=True)
        script = template.replace("__RUN__", run).replace("__CONFIG__", config)
        (folder / "run.py").write_text(script.replace("__BRANCH__", args.branch), encoding="utf-8")
        metadata = {
            "id": f"{args.owner}/{slug}",
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

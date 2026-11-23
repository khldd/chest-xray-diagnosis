"""Turn the Kaggle ablation kernel outputs into the results table and training-curve figure.

Inputs: one metrics.json per run (written by src.train) under <outputs>/<run>/metrics.json.
Writes:
  results/ablations.csv           one row per run (settings, best epoch, val AUC, per-class AUC)
  report/tables/ablations.tex     booktabs table for the report
  report/figures/ablation_curves.png   validation mean AUC per epoch for every run

Usage (repo root):
  python scripts/collect_ablations.py --download      # fetch kernel outputs (Kaggle CLI) first
  python scripts/collect_ablations.py                 # rebuild from already downloaded outputs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_USER = "khaledayedi"
DEFAULT_RUNS = ["base_5ep", "baseline_frozen", "unweighted_bce", "strong_aug"]
LABELS = {
    "base_5ep": "Fine-tuned, weighted BCE (reference)",
    "baseline_frozen": "Frozen backbone, head only",
    "unweighted_bce": "Fine-tuned, plain BCE",
    "strong_aug": "Fine-tuned, strong augmentation",
}


def download(runs: list[str], outputs: Path) -> None:
    for run in runs:
        slug = "cxr-ablation-" + run.replace("_", "-")
        target = outputs / run
        target.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "kernels", "output", f"{KAGGLE_USER}/{slug}", "-p", str(target)]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=False)


def find_metrics(outputs: Path, run: str) -> Path | None:
    """The kernel writes <run>/metrics.json, so the download nests it one level down."""
    for candidate in (outputs / run / run / "metrics.json", outputs / run / "metrics.json"):
        if candidate.exists():
            return candidate
    return None


def load_runs(outputs: Path, runs: list[str]) -> dict[str, dict]:
    loaded = {}
    for run in runs:
        path = find_metrics(outputs, run)
        if path is None:
            print(f"skipping {run}: no metrics.json found", file=sys.stderr)
            continue
        loaded[run] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def build_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for run, m in runs.items():
        cfg = m["config"]
        seconds = [h["seconds"] for h in m["history"] if "seconds" in h]
        row = {
            "run": run,
            "description": LABELS.get(run, run),
            "freeze_backbone": cfg["model"]["freeze_backbone"],
            "loss": cfg["train"]["loss"],
            "augmentation": cfg["data"]["augmentation"],
            "epochs": cfg["train"]["epochs"],
            "epochs_completed": len(m["history"]),
            "best_epoch": m["best_epoch"],
            "best_val_auc": round(m["best_val_auc"], 4),
            "mean_epoch_seconds": round(sum(seconds) / len(seconds)) if seconds else None,
        }
        row.update(
            {f"auc_{k}": round(v, 4) for k, v in m["per_class_val_auc_at_best_epoch"].items()}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Run & Backbone & Loss & Augmentation & Best epoch & Val.\ mean AUC \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        backbone = "frozen" if r["freeze_backbone"] else "fine-tuned"
        lines.append(
            f"{tex_escape(r['run'])} & {backbone} & {tex_escape(r['loss'])} & "
            f"{r['augmentation']} & {r['best_epoch']}/{r['epochs']} & {r['best_val_auc']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(runs: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run, m in runs.items():
        epochs = [h["epoch"] for h in m["history"]]
        aucs = [h["val_mean_auc"] for h in m["history"]]
        ax.plot(epochs, aucs, marker="o", label=LABELS.get(run, run))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mean AUC (14 classes)")
    ax.set_title("Ablation: validation AUC per epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")  # save before closing
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--outputs", type=Path, default=ROOT / "kaggle_outputs")
    ap.add_argument("--download", action="store_true", help="fetch kernel outputs first")
    args = ap.parse_args()

    if args.download:
        download(args.runs, args.outputs)
    runs = load_runs(args.outputs, args.runs)
    if not runs:
        sys.exit("No metrics.json found: have the Kaggle kernels finished?")

    df = build_table(runs)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "ablations.csv", index=False)
    write_latex(df, ROOT / "report" / "tables" / "ablations.tex")
    plot_curves(runs, ROOT / "report" / "figures" / "ablation_curves.png")
    print(df[["run", "best_epoch", "epochs", "best_val_auc", "mean_epoch_seconds"]].to_string())


if __name__ == "__main__":
    main()

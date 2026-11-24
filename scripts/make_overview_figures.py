"""Overview figures for the report: class imbalance, data split and system architecture.

Numbers come from the verified project data (scripts/make_report_tables.py and the official
split sizes), nothing is typed in twice.

Usage (repo root):  python scripts/make_overview_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_report_tables import N_TRAIN, TRAIN_POSITIVES  # noqa: E402

FIG = ROOT / "report" / "figures"
BLUE, ORANGE, GREEN, GREY = "#2f6fb0", "#d98324", "#3a8f5c", "#555555"


def box(ax, x, y, w, h, text, color, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=GREY, lw=1.5))


def class_imbalance() -> None:
    items = sorted(TRAIN_POSITIVES.items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in items]
    prev = [100 * v / N_TRAIN for _, v in items]
    weight = [(N_TRAIN - v) / v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.barh(names, prev, color=BLUE)
    for b, p, w in zip(bars, prev, weight):
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                f"{p:.2f}%   (loss weight x{w:.0f})", va="center", fontsize=8)
    ax.set(xlabel="Share of training images with the finding (%)", xlim=(0, 24),
           title="Strong class imbalance: rare findings get a larger loss weight")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "class_imbalance.png", dpi=170)
    plt.close(fig)


def split_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, 0.2, 1.8, 2.6, 1.4, "NIH ChestX-ray14\n112,120 images\n30,805 patients", GREY)
    box(ax, 4.0, 3.2, 2.8, 1.3, "train_val list\n86,524 images", BLUE)
    box(ax, 4.0, 0.5, 2.8, 1.3, "TEST (official)\n25,596 images\nused once, at the end", ORANGE)
    box(ax, 8.2, 3.9, 3.4, 1.0, "Train\n73,916 images", BLUE, fs=9)
    box(ax, 8.2, 2.5, 3.4, 1.0, "Validation (15%)\n12,608 images", GREEN, fs=9)
    arrow(ax, 2.9, 2.9, 3.9, 3.7)
    arrow(ax, 2.9, 2.1, 3.9, 1.2)
    arrow(ax, 6.9, 3.9, 8.1, 4.4)
    arrow(ax, 6.9, 3.5, 8.1, 3.0)
    ax.text(9.9, 1.4, "Split by PATIENT (GroupShuffleSplit, seed 42):\nno patient is in both train and validation",
            ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "split_diagram.png", dpi=170)
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.6, "Training (Kaggle GPU)", fontsize=10, weight="bold", color=GREY)
    box(ax, 0.1, 4.2, 2.2, 1.0, "NIH images\n+ patient split", GREY, fs=9)
    box(ax, 3.0, 4.2, 2.8, 1.0, "DenseNet121\n(ImageNet init)", BLUE, fs=9)
    box(ax, 6.5, 4.2, 2.6, 1.0, "Weighted BCE\nAdam, 15 epochs", BLUE, fs=9)
    box(ax, 9.8, 4.2, 2.0, 1.0, "best_model.pth\n(DVC / DagsHub)", ORANGE, fs=9)
    box(ax, 12.3, 4.2, 1.6, 1.0, "ONNX\nexport", ORANGE, fs=9)
    for x1, x2 in [(2.4, 2.9), (5.9, 6.4), (9.2, 9.7), (11.9, 12.2)]:
        arrow(ax, x1, 4.7, x2, 4.7)

    ax.text(0.1, 3.2, "Serving (Docker Compose)", fontsize=10, weight="bold", color=GREY)
    box(ax, 0.1, 1.6, 2.4, 1.1, "Browser\nReact + TypeScript", GREEN, fs=9)
    box(ax, 3.4, 1.6, 2.0, 1.1, "nginx\n(/api proxy)", GREEN, fs=9)
    box(ax, 6.3, 1.6, 3.0, 1.1, "FastAPI\nvalidation, CORS", GREEN, fs=9)
    box(ax, 10.2, 2.3, 3.6, 0.9, "ONNX Runtime -> /predict\n~25 ms, 14 probabilities", BLUE, fs=8.5)
    box(ax, 10.2, 0.9, 3.6, 0.9, "PyTorch -> /predict/visualize\nGrad-CAM heatmap", BLUE, fs=8.5)
    for x1, x2 in [(2.6, 3.3), (5.5, 6.2)]:
        arrow(ax, x1, 2.15, x2, 2.15)
    arrow(ax, 9.4, 2.3, 10.1, 2.75)
    arrow(ax, 9.4, 1.9, 10.1, 1.35)
    ax.text(7, 0.2, "CI (GitHub Actions): ruff, 26 tests, frontend build, Docker build",
            ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    class_imbalance()
    split_diagram()
    architecture()
    print("wrote class_imbalance.png, split_diagram.png, architecture.png")

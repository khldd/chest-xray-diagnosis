"""Generate report/figures/gradcam_<Disease>.png from the frontend gallery images.

Each figure shows the original X-ray next to the Grad-CAM overlay for the labelled disease.
Run from the repo root:  python scripts/make_gradcam_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DISEASES  # noqa: E402
from src.data.transforms import eval_transforms  # noqa: E402
from src.gradcam import denormalize, gradcam_overlay  # noqa: E402
from src.models.densenet import build_model  # noqa: E402

EXAMPLES = ROOT / "frontend" / "public" / "examples"
OUT = ROOT / "report" / "figures"
WEIGHTS = ROOT / "models" / "best_model.pth"


def main() -> None:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    tf = eval_transforms(224)
    OUT.mkdir(parents=True, exist_ok=True)

    for ex in json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8")):
        disease = ex["label"]
        tensor = tf(Image.open(EXAMPLES / ex["file"]).convert("RGB"))
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.unsqueeze(0)))[0, DISEASES.index(disease)].item()
        overlay = gradcam_overlay(model, tensor, disease)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
        axes[0].imshow(denormalize(tensor))
        axes[0].set_title("Original X-ray")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM: {disease.replace('_', ' ')} (p = {prob:.2f})")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{ex['file']}  |  labelled: {disease.replace('_', ' ')}", fontsize=9)
        fig.tight_layout()
        # Save BEFORE any plt.show(): show() clears the figure and a later savefig writes a blank.
        path = OUT / f"gradcam_{disease}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert pixels.std() > 5, f"{path.name} looks blank"
        print(f"{path.name}: p={prob:.2f}, {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()

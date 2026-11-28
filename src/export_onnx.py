"""Export the trained model to ONNX and verify parity against PyTorch."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.models.factory import build_model


def export(
    weights: str, out: str, img_size: int = 224, tol: float = 1e-3, arch: str = "densenet121"
) -> float:
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        expected = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    # Relative to the logit's size: random noise is far outside the training data, and some
    # trained models answer it with logits of -1e4 or more, where float32 rounding alone
    # exceeds an absolute 1e-3 (EfficientNet-B0: 0.02 on -5.6e4). Normal logits are within a
    # few units, so this is the plain absolute check there.
    max_diff = float((np.abs(expected - got) / np.maximum(1.0, np.abs(expected))).max())
    if max_diff > tol:
        raise RuntimeError(f"ONNX parity failed: max relative diff {max_diff:.2e} > {tol}")
    print(f"Exported {out} (max relative diff vs PyTorch: {max_diff:.2e})")
    return max_diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best_model.pth")
    ap.add_argument("--out", default="models/best_model.onnx")
    ap.add_argument("--arch", default="densenet121")
    a = ap.parse_args()
    export(a.weights, a.out, arch=a.arch)

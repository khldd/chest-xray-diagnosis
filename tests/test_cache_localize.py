"""Image cache gives identical tensors; localisation metrics on known boxes."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from src.data.cache import CachedImages, build_cache  # noqa: E402
from src.data.dataset import ChestXrayDataset  # noqa: E402
from src.data.transforms import eval_transforms, train_transforms  # noqa: E402
from src.localize import box_metrics, summarize  # noqa: E402


def test_cache_gives_identical_tensors(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(0)
    lookup = {}
    for i, size in enumerate([(300, 300), (257, 311), (1024, 1024)]):
        path = tmp_path / f"img{i}.png"
        Image.fromarray((rng.random(size[::-1]) * 255).astype("uint8")).save(path)
        lookup[path.name] = str(path)
    df = pd.DataFrame({"Image Index": list(lookup)})
    labels = np.zeros((len(df), 14), dtype=np.float32)
    path, index = build_cache(list(lookup), lookup, 64, str(tmp_path / "cache"), workers=1)
    cached = CachedImages(path, index, 64)
    for tf in (eval_transforms(64), train_transforms(64, "none")):
        plain = ChestXrayDataset(df, labels, lookup, tf)
        fast = ChestXrayDataset(df, labels, cached, tf)
        for i in range(len(df)):
            assert torch.equal(plain[i][0], fast[i][0])
    # A second call reuses the finished cache.
    assert build_cache(list(lookup), lookup, 64, str(tmp_path / "cache"), workers=1)[1] == index


def test_box_metrics_peak_inside_and_outside():
    heat = np.zeros((100, 100))
    heat[20:30, 20:30] = 1.0
    inside = box_metrics(heat, (10, 10, 40, 40))
    assert inside["hit"] == 1.0 and inside["energy"] == pytest.approx(1.0)
    assert inside["iou"] == pytest.approx(100 / 900)
    assert inside["area"] == pytest.approx(0.09) and inside["center_hit"] == 0.0
    outside = box_metrics(heat, (60, 60, 90, 90))
    assert outside["hit"] == 0.0 and outside["energy"] == 0.0 and outside["iou"] == 0.0


def test_summarize_macro_and_overall():
    keys = ["disease", "hit", "energy", "iou", "area", "center_hit"]
    rows = [
        dict(zip(keys, ["Mass", 1.0, 0.5, 0.2, 0.1, 0.0], strict=True)),
        dict(zip(keys, ["Mass", 0.0, 0.1, 0.0, 0.1, 0.0], strict=True)),
        dict(zip(keys, ["Nodule", 1.0, 0.3, 0.1, 0.01, 1.0], strict=True)),
    ]
    out = summarize(rows)
    assert out["overall"]["n"] == 3 and out["overall"]["hit"] == pytest.approx(2 / 3)
    assert out["macro"]["hit"] == pytest.approx(0.75)
    assert out["per_disease"]["Mass"]["n"] == 2

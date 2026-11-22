import numpy as np
import pandas as pd
import pytest

from src.config import DISEASES, NUM_CLASSES, load_config
from src.data.prepare import encode_labels, patient_split


def test_class_list():
    assert NUM_CLASSES == 14
    assert "No Finding" not in DISEASES
    assert DISEASES == sorted(DISEASES)  # checkpoint output order depends on this


def test_encode_labels_multi_hot_and_no_finding():
    out = encode_labels(pd.Series(["Cardiomegaly|Emphysema", "No Finding", "Hernia"]))
    assert out.shape == (3, 14)
    assert out[0].sum() == 2
    assert out[0, DISEASES.index("Cardiomegaly")] == 1
    assert out[1].sum() == 0  # No Finding -> all zeros
    assert out[2, DISEASES.index("Hernia")] == 1


def test_patient_split_has_no_patient_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Patient ID": rng.integers(0, 200, size=1000)})
    train_idx, val_idx = patient_split(df, val_fraction=0.15, seed=42)
    assert set(df.loc[train_idx, "Patient ID"]).isdisjoint(df.loc[val_idx, "Patient ID"])
    assert len(train_idx) + len(val_idx) == len(df)


def test_patient_split_is_deterministic():
    df = pd.DataFrame({"Patient ID": np.repeat(np.arange(100), 3)})
    a = patient_split(df, 0.15, 42)
    b = patient_split(df, 0.15, 42)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_load_config_merges_override(tmp_path):
    override = tmp_path / "exp.yaml"
    override.write_text("train:\n  loss: bce\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(override)
    assert cfg["train"]["loss"] == "bce" and cfg["train"]["epochs"] == 5
    assert cfg["train"]["lr"] == 0.0001  # inherited from base.yaml
    assert cfg["data"]["val_fraction"] == 0.15


def test_unknown_augmentation_rejected():
    pytest.importorskip("torchvision")
    from src.data.transforms import train_transforms

    with pytest.raises(ValueError):
        train_transforms(224, "bogus")

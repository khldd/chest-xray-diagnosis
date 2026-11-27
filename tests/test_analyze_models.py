# ruff: noqa: E501
"""Tests for scripts/analyze_models.py on small synthetic predictions."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts import analyze_models as am
from src.config import DISEASES

C = len(DISEASES)


def _split(rng, n_pat, offset, prefix, with_patient):
    imgs = rng.integers(1, 4, n_pat)
    pats = np.repeat(np.arange(offset, offset + n_pat), imgs)
    names = np.array([f"{p:08d}_{k:03d}.png" for p, k in zip(pats, np.concatenate([np.arange(i) for i in imgs]), strict=True)])
    n = len(names)
    y = (rng.random((n, C)) < 0.2).astype(np.int64)
    y[:3] = 1  # every class has positives and negatives
    y[3:6] = 0
    out = {f"{prefix}_labels": y, f"{prefix}_names": names,
           f"{prefix}_age": rng.integers(18, 90, n), f"{prefix}_sex": rng.choice(["M", "F"], n),
           f"{prefix}_view": rng.choice(["PA", "AP"], n)}
    if with_patient:
        out[f"{prefix}_patient"] = np.array([f"{p:08d}" for p in pats])
    return out


def _write_model(path, base, rng, strength, bias, with_meta):
    d = {}
    for s in ("test", "val"):
        y = base[f"{s}_labels"]
        d[f"{s}_probs"] = am.sigmoid(strength * y + rng.normal(size=y.shape) + bias).astype(np.float32)
        for k in ("labels", "names"):
            d[f"{s}_{k}"] = base[f"{s}_{k}"]
        if with_meta:
            for k in ("age", "sex", "view", "patient"):
                if f"{s}_{k}" in base:
                    d[f"{s}_{k}"] = base[f"{s}_{k}"]
    np.savez(path, **d)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analyze")
    rng = np.random.default_rng(0)
    base = {**_split(rng, 200, 0, "test", with_patient=False), **_split(rng, 120, 10_000, "val", with_patient=True)}
    _write_model(tmp / "a.npz", base, rng, 1.5, 1.0, with_meta=True)  # over-confident, like the real model
    _write_model(tmp / "b.npz", base, rng, 1.0, 0.0, with_meta=False)
    out = tmp / "out"
    res = am.run({"model_a": str(tmp / "a.npz"), "model_b": str(tmp / "b.npz")}, out, n_boot=200, seed=1,
                 latency={"model_a": 10.0}, params={"model_a": 8_000_000})
    return out, res, tmp


def test_all_outputs_created(result):
    out, _, _ = result
    for rel in ("results/benchmark.csv", "results/pairwise.csv", "results/temperatures.json",
                "report/tables/benchmark.tex", "report/tables/subgroups.tex",
                "report/figures/forest_auc.png", "report/figures/calibration_before_after.png",
                "report/figures/subgroups.png"):
        assert (out / rel).is_file() and (out / rel).stat().st_size > 0, rel


def test_ensemble_row_everywhere(result):
    out, res, _ = result
    with open(out / "results" / "benchmark.csv", encoding="utf-8") as f:
        models = [r["model"] for r in csv.DictReader(f)]
    assert models == ["model_a", "model_b", "ensemble"]
    assert "ensemble" in json.loads((out / "results" / "temperatures.json").read_text())
    assert {p["model_b"] for p in res["pairwise"]} >= {"ensemble"}
    assert len(res["pairwise"]) == 3
    assert {r["model"] for r in res["subgroups"]} == {"model_a", "model_b", "ensemble"}
    assert "ensemble" in (out / "report" / "tables" / "benchmark.tex").read_text()


def test_temperature_preserves_auc_and_improves_ece(result):
    _, _, tmp = result
    d = np.load(tmp / "a.npz")
    p, y = d["test_probs"].astype(np.float64), d["test_labels"]
    t = am.fit_temperature(d["val_probs"].astype(np.float64), d["val_labels"])
    assert t != pytest.approx(1.0, abs=1e-3)
    q = am.apply_temperature(p, t)
    for i in range(C):
        assert roc_auc_score(y[:, i], q[:, i]) == pytest.approx(roc_auc_score(y[:, i], p[:, i]), abs=1e-12)
    assert am.bce(am.apply_temperature(d["val_probs"].astype(np.float64), t), d["val_labels"]) <= \
        am.bce(d["val_probs"].astype(np.float64), d["val_labels"]) + 1e-9


def test_platt_preserves_auc_and_fixes_shift(result):
    out, res, tmp = result
    d = np.load(tmp / "a.npz")  # model_a has a +1 logit bias, like class-weighted BCE
    p, y = d["test_probs"].astype(np.float64), d["test_labels"]
    a, b = am.fit_platt(d["val_probs"].astype(np.float64), d["val_labels"])
    q = am.apply_platt(p, a, b)
    for i in range(C):
        assert roc_auc_score(y[:, i], q[:, i]) == pytest.approx(roc_auc_score(y[:, i], p[:, i]), abs=1e-12)
    row = next(r for r in res["benchmark"] if r["model"] == "model_a")
    assert row["ece_platt"] < row["ece_before"]


def test_ci_contains_point_estimate(result):
    _, res, _ = result
    for r in res["benchmark"] + res["subgroups"]:
        assert r["ci_low"] <= r["auc"] <= r["ci_high"], r
    for p in res["pairwise"]:
        assert 0.0 <= p["p_value"] <= 1.0


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(3)
    y = (rng.random((300, C)) < 0.3).astype(int)
    p = np.round(rng.random((300, C)) + 0.3 * y, 2)  # rounding creates ties
    expected = np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(C)])
    assert am.mean_auc_weighted(am.prep_auc(p, y)) == pytest.approx(expected, abs=1e-12)
    # integer weights == duplicating rows
    w = rng.integers(0, 3, 300)
    idx = np.repeat(np.arange(300), w)
    dup = np.mean([roc_auc_score(y[idx, i], p[idx, i]) for i in range(C)])
    assert am.mean_auc_weighted(am.prep_auc(p, y), w.astype(float)) == pytest.approx(dup, abs=1e-12)


def test_patient_from_names():
    assert list(am.patient_from_names(np.array(["00000013_005.png", "00000013_000.png", "00030805_000.png"]))) == \
        ["00000013", "00000013", "00030805"]


def test_ece_perfect_and_bad():
    y = np.array([[0], [1]] * 50)
    assert am.ece(y.astype(float), y) == 0.0
    assert am.ece(1.0 - y.astype(float), y) == pytest.approx(1.0)

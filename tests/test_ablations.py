import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "collect_ablations.py"
spec = importlib.util.spec_from_file_location("collect_ablations", SCRIPT)
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def fake_metrics(freeze, loss, aucs):
    return {
        "run": "x",
        "config": {
            "model": {"freeze_backbone": freeze},
            "train": {"loss": loss, "epochs": len(aucs)},
            "data": {"augmentation": "flip"},
        },
        "best_val_auc": max(aucs),
        "best_epoch": aucs.index(max(aucs)) + 1,
        "per_class_val_auc_at_best_epoch": {"Mass": max(aucs), "Hernia": 0.5},
        "history": [
            {"epoch": i + 1, "val_mean_auc": a, "seconds": 100 + i} for i, a in enumerate(aucs)
        ],
    }


@pytest.fixture
def outputs(tmp_path):
    # Kaggle downloads nest the file one level down: <outputs>/<run>/<run>/metrics.json
    for run, freeze, loss, aucs in [
        ("base_5ep", False, "weighted_bce", [0.70, 0.78, 0.80]),
        ("baseline_frozen", True, "weighted_bce", [0.65, 0.70, 0.71]),
    ]:
        d = tmp_path / run / run
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(fake_metrics(freeze, loss, aucs)))
    return tmp_path


def test_missing_runs_are_skipped(outputs):
    runs = collect.load_runs(outputs, ["base_5ep", "baseline_frozen", "strong_aug"])
    assert list(runs) == ["base_5ep", "baseline_frozen"]


def test_table_has_best_epoch_and_timing(outputs):
    df = collect.build_table(collect.load_runs(outputs, ["base_5ep", "baseline_frozen"]))
    ref = df[df["run"] == "base_5ep"].iloc[0]
    assert ref["best_epoch"] == 3 and ref["best_val_auc"] == 0.80
    assert ref["mean_epoch_seconds"] == 101
    assert df[df["run"] == "baseline_frozen"].iloc[0]["freeze_backbone"]


def test_latex_and_figure_are_written(outputs, tmp_path):
    runs = collect.load_runs(outputs, ["base_5ep", "baseline_frozen"])
    tex = tmp_path / "t" / "ablations.tex"
    collect.write_latex(collect.build_table(runs), tex)
    text = tex.read_text()
    assert r"base\_5ep" in text and "frozen" in text and r"\toprule" in text
    fig = tmp_path / "f" / "curves.png"
    collect.plot_curves(runs, fig)
    assert fig.stat().st_size > 5000  # a real plot, not a blank canvas

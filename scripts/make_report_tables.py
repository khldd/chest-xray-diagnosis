"""Generate report/tables/classes.tex from verified project data (no hand-typed numbers).

Sources:
  - TRAIN_POSITIVES: per-class positive counts in the 73,916-image training split, copied from
    the output of the prototype notebook (cell 'severe class imbalance').
  - results/test_metrics.json: per-class AUC on the official test set.
pos_weight is recomputed as negatives / positives, matching the weights printed in the notebook.

Usage (repo root):  python scripts/make_report_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_TRAIN = 73916
TRAIN_POSITIVES = {
    "Atelectasis": 7003,
    "Cardiomegaly": 1453,
    "Consolidation": 2482,
    "Edema": 1201,
    "Effusion": 7415,
    "Emphysema": 1274,
    "Fibrosis": 1051,
    "Hernia": 113,
    "Infiltration": 11853,
    "Mass": 3452,
    "Nodule": 4042,
    "Pleural_Thickening": 1870,
    "Pneumonia": 742,
    "Pneumothorax": 2287,
}


def build_classes_table(test_auc: dict[str, float]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Disease & Train positives & Prevalence (\%) & \texttt{pos\_weight} & Test AUC \\",
        r"\midrule",
    ]
    for name, pos in TRAIN_POSITIVES.items():
        weight = (N_TRAIN - pos) / pos
        lines.append(
            f"{name.replace('_', ' ')} & {pos:,} & {100 * pos / N_TRAIN:.2f} & "
            f"{weight:.1f} & {test_auc[name]:.3f} \\\\".replace(",", "{,}")
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads((ROOT / "results" / "test_metrics.json").read_text(encoding="utf-8"))
    out = ROOT / "report" / "tables" / "classes.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_classes_table(metrics["per_class_auc"]), encoding="utf-8")
    print("wrote", out.relative_to(ROOT), f"(test mean AUC {metrics['mean_auc']})")


if __name__ == "__main__":
    main()

"""Draw the standard evaluation figures from saved model predictions.

Input: predictions.npz produced by the kaggle/eval_predictions kernel
(test_probs, test_labels, val_probs, val_labels).

Per-class decision thresholds are chosen on the VALIDATION split (Youden's J = sens + spec - 1)
and then applied unchanged to the TEST split, so no test information leaks into the choice.

Writes:
  report/figures/{roc_curves,pr_curves,confusion_matrices,calibration}.png
  report/tables/metrics.tex      per-class AUC, AP, threshold, sensitivity, specificity, PPV, F1
  results/eval_metrics.json

Usage (repo root):  python scripts/make_eval_figures.py --preds path/to/predictions.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import DISEASES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
BLUE, RED, GREY = "#2f6fb0", "#c0392b", "#888888"


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    return float(thr[finite][np.argmax((tpr - fpr)[finite])])


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tn, fp, fn, tp


def per_class_metrics(val_p, val_y, test_p, test_y) -> list[dict]:
    rows = []
    for i, name in enumerate(DISEASES):
        thr = youden_threshold(val_y[:, i], val_p[:, i])
        tn, fp, fn, tp = confusion(test_y[:, i], test_p[:, i], thr)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
        rows.append(
            {
                "disease": name,
                "n_pos": int(test_y[:, i].sum()),
                "auc": float(roc_auc_score(test_y[:, i], test_p[:, i])),
                "ap": float(average_precision_score(test_y[:, i], test_p[:, i])),
                "threshold": thr,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "f1": f1,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }
        )
    return rows


def plot_roc(test_p, test_y, rows):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(rows):
        fpr, tpr, _ = roc_curve(test_y[:, i], test_p[:, i])
        ax.plot(fpr, tpr, lw=1.3, color=cmap(i), label=f"{r['disease'].replace('_', ' ')} ({r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves per class (official test set)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=7, loc="lower right", ncol=1, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=170)
    plt.close(fig)


def plot_pr(test_p, test_y, rows):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(rows):
        prec, rec, _ = precision_recall_curve(test_y[:, i], test_p[:, i])
        ax.plot(rec, prec, lw=1.3, color=cmap(i),
                label=f"{r['disease'].replace('_', ' ')} (AP {r['ap']:.2f}, prev {test_y[:, i].mean():.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves per class (test set)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves.png", dpi=170)
    plt.close(fig)


def plot_confusions(rows):
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for ax, r in zip(axes.ravel(), rows):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)  # row-normalised: spec/sens
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{v:,}\n{norm[a, b]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[a, b] > 0.55 else "black")
        ax.set_xticks([0, 1], ["pred -", "pred +"], fontsize=7)
        ax.set_yticks([0, 1], ["true -", "true +"], fontsize=7)
        ax.set_title(f"{r['disease'].replace('_', ' ')} (thr {r['threshold']:.2f})", fontsize=9)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Per-class confusion matrices on the test set (threshold from validation, Youden J)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_calibration(test_p, test_y):
    bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    p, y = test_p.ravel(), test_y.ravel()
    idx = np.digitize(p, bins) - 1
    mean_p, frac, count = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 20:
            mean_p.append(p[m].mean())
            frac.append(y[m].mean())
            count.append(int(m.sum()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    a1.plot(mean_p, frac, "o-", color=BLUE, label="Model (all classes pooled)")
    a1.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive",
           title="Reliability diagram (test)", xlim=(0, 1), ylim=(0, 1))
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=0.25)
    a2.hist(p, bins=40, color=BLUE, log=True)
    a2.set(xlabel="Predicted probability", ylabel="Count (log)", title="Score distribution")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=170)
    plt.close(fig)
    return float(np.mean(p)), float(np.mean(y))


def write_table(rows):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Disease & AUC & AP & Thr. & Sens. & Spec. & PPV & F1 \\", r"\midrule"]
    for r in rows:
        lines.append(
            f"{r['disease'].replace('_', ' ')} & {r['auc']:.3f} & {r['ap']:.3f} & {r['threshold']:.2f} & "
            f"{r['sensitivity']:.3f} & {r['specificity']:.3f} & {r['ppv']:.3f} & {r['f1']:.3f} \\\\"
        )
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    lines += [r"\midrule",
              f"Mean & {mean('auc'):.3f} & {mean('ap'):.3f} & -- & {mean('sensitivity'):.3f} & "
              f"{mean('specificity'):.3f} & {mean('ppv'):.3f} & {mean('f1'):.3f} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    out = ROOT / "report" / "tables" / "metrics.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    d = np.load(args.preds, allow_pickle=False)
    test_p, test_y, val_p, val_y = (d[k] for k in ("test_probs", "test_labels", "val_probs", "val_labels"))
    FIG.mkdir(parents=True, exist_ok=True)

    rows = per_class_metrics(val_p, val_y, test_p, test_y)
    plot_roc(test_p, test_y, rows)
    plot_pr(test_p, test_y, rows)
    plot_confusions(rows)
    mean_pred, mean_true = plot_calibration(test_p, test_y)
    write_table(rows)

    summary = {
        "n_test": int(len(test_y)),
        "mean_auc": float(np.mean([r["auc"] for r in rows])),
        "mean_ap": float(np.mean([r["ap"] for r in rows])),
        "mean_predicted_probability": mean_pred,
        "mean_label_prevalence": mean_true,
        "per_class": rows,
    }
    (ROOT / "results" / "eval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean AUC {summary['mean_auc']:.4f} | mean AP {summary['mean_ap']:.4f}")
    print(f"mean predicted prob {mean_pred:.3f} vs label prevalence {mean_true:.3f}")


if __name__ == "__main__":
    main()

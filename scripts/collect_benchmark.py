"""Tables and figures for the architecture benchmark (the ablation study on every model).

Inputs (already downloaded, see the scheduler / README):
  kaggle_outputs/bench/<run>/<run>/metrics.json and localization.json   one per run
  kaggle_outputs/localize_runs/<run>/localization.json   re-scored localisation (overrides)
  results/benchmark.csv      test AUC + 95% CI per run (scripts/analyze_models.py)
  results/latency_ms.json    ONNX CPU latency per architecture (batch 1, median of 100)

Run names are <arch>_<variant>[_s<seed>], variant in {bce, weighted, frozen}; no suffix = seed 0.

Writes:
  results/grid.csv                       one row per run: val/test AUC, CI, localisation
  report/tables/grid.tex                 test AUC, architecture x variant (seed 0)
  report/tables/seeds.tex                DenseNet121 plain vs weighted BCE over 3 seeds
  report/tables/localization.tex         pointing game / energy / IoU per architecture
  report/figures/grid_heatmap.png        the grid as a heatmap
  report/figures/pareto.png              test AUC vs CPU latency (marker area ~ parameters)
  report/figures/localization_bars.png   pointing-game hit rate with the two baselines
  report/figures/localization_per_disease.png   hit rate per architecture and finding

Usage (repo root):  PYTHONPATH=. python scripts/collect_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.factory import get_arch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARCHS = ["densenet121", "resnet50", "efficientnet_b0", "convnext_tiny"]
VARIANTS = ["bce", "weighted", "frozen"]
VARIANT_LABEL = {"bce": "Fine-tuned, plain BCE", "weighted": "Fine-tuned, weighted BCE",
                 "frozen": "Frozen backbone, plain BCE"}
# Reference categorical palette, slots 1-4 in fixed order (one colour per architecture).
COLORS = {"convnext_tiny": "#2a78d6", "densenet121": "#eb6834",
          "resnet50": "#1baf7a", "efficientnet_b0": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUN_RE = re.compile(r"^(?P<arch>.+?)_(?P<variant>bce|weighted|frozen)(?:_s(?P<seed>\d+))?$")


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def display(arch: str) -> str:
    return get_arch(arch).display


def load_runs(bench: Path, overrides: Path, benchmark_csv: Path) -> pd.DataFrame:
    tests = pd.read_csv(benchmark_csv).set_index("model")
    rows = []
    for metrics_path in sorted(bench.glob("*/*/metrics.json")):
        run = metrics_path.parent.name
        m = RUN_RE.match(run)
        if not m:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        loc_path = overrides / run / "localization.json"
        if not loc_path.exists():
            loc_path = metrics_path.parent / "localization.json"
        loc = json.loads(loc_path.read_text(encoding="utf-8")) if loc_path.exists() else {}
        overall = loc.get("overall", {})
        row = {
            "run": run, "arch": m["arch"], "variant": m["variant"],
            "seed": int(m["seed"] or 0),
            "n_parameters": metrics.get("n_parameters"),
            "best_epoch": metrics.get("best_epoch"),
            "val_auc": metrics["best_val_auc"],
            "test_auc": tests.loc[run, "auc"] if run in tests.index else np.nan,
            "test_ci_low": tests.loc[run, "ci_low"] if run in tests.index else np.nan,
            "test_ci_high": tests.loc[run, "ci_high"] if run in tests.index else np.nan,
            "loc_hit": overall.get("hit"), "loc_energy": overall.get("energy"),
            "loc_iou": overall.get("iou"), "loc_area": overall.get("area"),
            "loc_center_hit": overall.get("center_hit"),
        }
        for disease, v in loc.get("per_disease", {}).items():
            row[f"hit_{disease}"] = v["hit"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arch", "variant", "seed"]).reset_index(drop=True)


def grid_table(df: pd.DataFrame, out: Path) -> None:
    seed0 = df[df.seed == 0].set_index(["arch", "variant"])
    best = {v: seed0.xs(v, level="variant")["test_auc"].max() for v in VARIANTS
            if v in seed0.index.get_level_values("variant")}
    lines = [r"\begin{tabular}{l" + "c" * len(VARIANTS) + "}", r"\toprule",
             "Architecture & " + " & ".join(VARIANT_LABEL[v] for v in VARIANTS) + r" \\",
             r"\midrule"]
    for arch in ARCHS:
        cells = []
        for v in VARIANTS:
            if (arch, v) not in seed0.index or np.isnan(seed0.loc[(arch, v), "test_auc"]):
                cells.append("--")
                continue
            r = seed0.loc[(arch, v)]
            txt = f"{r.test_auc:.3f} [{r.test_ci_low:.3f}, {r.test_ci_high:.3f}]"
            cells.append(rf"\textbf{{{txt}}}" if np.isclose(r.test_auc, best[v]) else txt)
        lines.append(display(arch) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seeds_table(df: pd.DataFrame, out: Path) -> None:
    d = df[df.arch == "densenet121"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"DenseNet121 & Seeds & Val AUC (mean $\pm$ sd) & Test AUC (mean $\pm$ sd) \\",
             r"\midrule"]
    for v in ("bce", "weighted"):
        g = d[d.variant == v]
        if g.empty:
            continue
        lines.append(
            f"{VARIANT_LABEL[v]} & {len(g)} & "
            f"{g.val_auc.mean():.4f} $\\pm$ {g.val_auc.std(ddof=1):.4f} & "
            f"{g.test_auc.mean():.4f} $\\pm$ {g.test_auc.std(ddof=1):.4f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def localization_table(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.variant == "bce") & (df.seed == 0)].set_index("arch")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & Pointing game (hit) & Energy in box & IoU (heat $\geq$ 0.5) \\",
             r"\midrule"]
    for arch in sorted(ARCHS, key=lambda a: -d.loc[a, "loc_hit"]):
        r = d.loc[arch]
        lines.append(f"{display(arch)} & {r.loc_hit:.3f} & {r.loc_energy:.3f} & "
                     f"{r.loc_iou:.3f}" + r" \\")
    any_row = d.iloc[0]
    lines += [r"\midrule",
              f"Random point (baseline) & {any_row.loc_area:.3f} & -- & --" + r" \\",
              f"Image centre (baseline) & {any_row.loc_center_hit:.3f} & -- & --" + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grid_heatmap(df: pd.DataFrame, out: Path) -> None:
    seed0 = df[df.seed == 0].pivot(index="arch", columns="variant", values="test_auc")
    seed0 = seed0.reindex(index=ARCHS, columns=VARIANTS)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(seed0.values, cmap="Blues", vmin=0.6, vmax=0.82, aspect="auto")
    labels = [VARIANT_LABEL[v].replace(", ", ",\n") for v in VARIANTS]
    ax.set_xticks(range(len(VARIANTS)), labels)
    ax.set_yticks(range(len(ARCHS)), [display(a) for a in ARCHS])
    for i in range(len(ARCHS)):
        for j in range(len(VARIANTS)):
            val = seed0.values[i, j]
            if np.isnan(val):
                ax.text(j, i, "not run", ha="center", va="center", color=INK2, fontsize=9)
                continue
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if val > 0.75 else INK)
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("Test mean AUC", color=INK2)
    cbar.outline.set_visible(False)
    ax.set_title("Same study, every architecture: test mean AUC (5 epochs, seed 0)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def pareto(df: pd.DataFrame, latency: dict, out: Path, ensemble: dict | None) -> None:
    d = df[(df.variant == "bce") & (df.seed == 0)].set_index("arch")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arch in ARCHS:
        r = d.loc[arch]
        x, y = latency[arch], r.test_auc
        ax.errorbar(x, y, yerr=[[y - r.test_ci_low], [r.test_ci_high - y]], fmt="none",
                    ecolor=COLORS[arch], elinewidth=2, capsize=0, zorder=2)
        ax.scatter(x, y, s=r.n_parameters / 1e5, color=COLORS[arch], edgecolor="white",
                   linewidth=2, zorder=3)
        ax.annotate(f"{display(arch)}\n{r.n_parameters / 1e6:.1f}M params", (x, y),
                    xytext=(10, -4), textcoords="offset points", fontsize=8.5, color=INK)
    if ensemble:
        x = sum(latency[a] for a in ARCHS)
        ax.errorbar(x, ensemble["auc"], yerr=[[ensemble["auc"] - ensemble["ci_low"]],
                    [ensemble["ci_high"] - ensemble["auc"]]], fmt="D", color=INK2,
                    markersize=7, elinewidth=2, zorder=3)
        ax.annotate("Ensemble of all 4\n(sum of latencies)", (x, ensemble["auc"]),
                    xytext=(-10, -4), textcoords="offset points", ha="right", fontsize=8.5,
                    color=INK)
    ax.set_xlabel("ONNX Runtime CPU latency per image (ms, batch 1, median)")
    ax.set_ylabel("Test mean AUC (95% CI, patient bootstrap)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)
    style(ax)
    ax.set_title("Accuracy against speed (marker area ~ parameters)", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def localization_bars(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.variant == "bce") & (df.seed == 0)].set_index("arch")
    order = sorted(ARCHS, key=lambda a: d.loc[a, "loc_hit"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    vals = [d.loc[a, "loc_hit"] for a in order]
    ax.barh(range(len(order)), vals, height=0.55, color=[COLORS[a] for a in order])
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.1%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)), [display(a) for a in order])
    base = d.iloc[0]
    for x, label in ((base.loc_area, "random point"), (base.loc_center_hit, "image centre")):
        ax.axvline(x, color=INK2, linestyle="--", linewidth=1)
        ax.text(x, len(order) - 0.45, f" {label} {x:.1%}", fontsize=8.5, color=INK2)
    ax.set_xlabel("Heatmap peak inside the radiologist's box (pointing game, 984 boxes)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def localization_per_disease(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.variant == "bce") & (df.seed == 0)].set_index("arch").reindex(ARCHS)
    cols = sorted(c for c in d.columns if c.startswith("hit_"))
    mat = d[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c[4:] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(ARCHS)), [display(a) for a in ARCHS])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.55 else INK)
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("Pointing-game hit rate", color=INK2)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default=str(ROOT / "kaggle_outputs" / "bench"))
    ap.add_argument("--localize-overrides", default=str(ROOT / "kaggle_outputs" / "localize_runs"))
    ap.add_argument("--benchmark-csv", default=str(ROOT / "results" / "benchmark.csv"))
    ap.add_argument("--latency", default=str(ROOT / "results" / "latency_ms.json"))
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ("results", "report/tables", "report/figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.bench), Path(args.localize_overrides), Path(args.benchmark_csv))
    df.to_csv(out / "results" / "grid.csv", index=False)
    latency = json.loads(Path(args.latency).read_text(encoding="utf-8"))
    tests = pd.read_csv(args.benchmark_csv).set_index("model")
    ens = tests.loc["ensemble"].to_dict() if "ensemble" in tests.index else None

    grid_table(df, out / "report" / "tables" / "grid.tex")
    seeds_table(df, out / "report" / "tables" / "seeds.tex")
    localization_table(df, out / "report" / "tables" / "localization.tex")
    grid_heatmap(df, out / "report" / "figures" / "grid_heatmap.png")
    pareto(df, latency, out / "report" / "figures" / "pareto.png", ens)
    localization_bars(df, out / "report" / "figures" / "localization_bars.png")
    localization_per_disease(df, out / "report" / "figures" / "localization_per_disease.png")
    cols = ["run", "val_auc", "test_auc", "loc_hit", "loc_iou"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Compare several models (and their ensemble) from saved predictions.

Input: one predictions.npz per model (same format as scripts/make_eval_figures.py:
test_probs, test_labels, val_probs, val_labels, test_names, val_names; optional
test_patient/test_age/test_sex/test_view and the same for val). Probabilities are sigmoid
outputs in src.config.DISEASES order.

What it computes, for every model plus the ensemble (mean of the members' probabilities):
  * mean test AUC over the 14 classes with a 95% CI from a PATIENT-level bootstrap
    (patients resampled with replacement; the same resamples are shared by all models, so
    pairwise differences get a paired bootstrap p-value);
  * mean AP;
  * temperature scaling: one scalar T per model fitted on VALIDATION by minimising the BCE of
    sigmoid(logit(p) / T), then applied to test. ECE (15 equal-width bins, all classes pooled)
    is reported before and after. T is monotone, so it leaves AUC/AP unchanged. A global
    Platt fit (logit(p) * a + b, also on val) is reported alongside: a scalar T cannot remove a
    constant upward shift of the scores (from class-weighted BCE), the bias b can;
  * subgroup mean AUC with CI by sex, age band and view, when that metadata is present.

Writes (under --out-dir, default: repo root):
  results/benchmark.csv, results/pairwise.csv, results/temperatures.json
  report/tables/benchmark.tex, report/tables/subgroups.tex
  report/figures/forest_auc.png, report/figures/calibration_before_after.png,
  report/figures/subgroups.png

Usage (repo root):
  PYTHONPATH=. python scripts/analyze_models.py \
      --preds densenet121=path/a.npz --preds resnet50=path/b.npz \
      --params densenet121=6960000 --latency densenet121=12.3
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import DISEASES  # noqa: E402

try:  # scipy is optional; a 1-D grid search is used without it
    from scipy.optimize import minimize_scalar
except ImportError:  # pragma: no cover
    minimize_scalar = None

ROOT = Path(__file__).resolve().parent.parent
BLUE, RED, GREY = "#2f6fb0", "#c0392b", "#888888"
EPS = 1e-12
N_BINS = 15
META_KEYS = ("patient", "age", "sex", "view")
ENSEMBLE = "ensemble"


# --------------------------------------------------------------------------- loading
def _parse_pairs(items: list[str] | None, cast=str) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected name=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = cast(v.strip())
    return out


def patient_from_names(names: np.ndarray) -> np.ndarray:
    """NIH image names look like 00000013_005.png; the patient id is the part before '_'."""
    return np.array([str(n).split("_")[0] for n in names])


def load_preds(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {}
    for split in ("test", "val"):
        s = {
            "probs": d[f"{split}_probs"].astype(np.float64),
            "labels": d[f"{split}_labels"].astype(np.int64),
            "names": np.array([str(n) for n in d[f"{split}_names"]]),
        }
        for k in META_KEYS:
            if f"{split}_{k}" in d.files:
                s[k] = d[f"{split}_{k}"]
        if s["probs"].ndim != 2 or s["probs"].shape[1] != len(DISEASES):
            raise SystemExit(f"{path}: {split}_probs has shape {s['probs'].shape}, expected (N, {len(DISEASES)})")
        out[split] = s
    return out


def _align(ref_names: np.ndarray, s: dict, name: str) -> dict:
    """Reorder split dict s so its rows follow ref_names."""
    if np.array_equal(ref_names, s["names"]):
        return s
    pos = {n: i for i, n in enumerate(s["names"])}
    missing = [n for n in ref_names if n not in pos]
    if missing or len(pos) != len(ref_names):
        raise SystemExit(f"{name}: image set differs from the first model ({len(missing)} missing)")
    idx = np.array([pos[n] for n in ref_names])
    return {k: v[idx] for k, v in s.items()}


def load_all(preds: dict[str, str]) -> tuple[dict, dict]:
    """Returns {model: {'test': probs, 'val': probs}} and shared {'test': {...}, 'val': {...}}."""
    models, shared = {}, {}
    for i, (name, path) in enumerate(preds.items()):
        d = load_preds(path)
        for split in ("test", "val"):
            if i == 0:
                shared[split] = {k: v for k, v in d[split].items() if k != "probs"}
            else:
                d[split] = _align(shared[split]["names"], d[split], name)
                if not np.array_equal(d[split]["labels"], shared[split]["labels"]):
                    raise SystemExit(f"{name}: {split} labels differ from the first model")
                for k in META_KEYS:  # metadata may live in any file
                    if k in d[split] and k not in shared[split]:
                        shared[split][k] = d[split][k]
        models[name] = {"test": d["test"]["probs"], "val": d["val"]["probs"]}
    for split in ("test", "val"):
        if "patient" not in shared[split]:
            shared[split]["patient"] = patient_from_names(shared[split]["names"])
        shared[split]["patient"] = np.array([str(x) for x in shared[split]["patient"]])
    return models, shared


# --------------------------------------------------------------------------- metrics
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free logistic


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return sigmoid(logit(p) / t)


def bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Scalar T minimising BCE of sigmoid(logit(p)/T) on (p, y); searched over log T."""
    z, yy = logit(p).ravel(), y.ravel().astype(np.float64)

    def loss(log_t: float) -> float:
        return bce(sigmoid(z / np.exp(log_t)), yy)

    if minimize_scalar is not None:
        res = minimize_scalar(loss, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-5})
        return float(np.exp(res.x))
    grid = np.linspace(-4, 4, 401)  # coarse grid, then a fine one around the best point
    best = grid[np.argmin([loss(g) for g in grid])]
    fine = np.linspace(best - 0.02, best + 0.02, 201)
    return float(np.exp(fine[np.argmin([loss(g) for g in fine])]))


def fit_platt(p: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Global Platt scaling sigmoid(a * logit(p) + b), fitted by damped Newton on BCE (convex).

    Unlike a bare temperature (a = 1/T, b = 0) the bias b can undo the upward shift that
    class-weighted BCE puts on every score; still monotone, so AUC/AP are unchanged.
    """
    z, yy = logit(p).ravel(), y.ravel().astype(np.float64)
    a, b = 1.0, 0.0

    def loss(a_: float, b_: float) -> float:
        return bce(sigmoid(a_ * z + b_), yy)

    cur = loss(a, b)
    for _ in range(iters):
        q = sigmoid(a * z + b)
        r, h = q - yy, np.maximum(q * (1 - q), 1e-12)
        grad = np.array([np.dot(r, z), r.sum()])
        hess = np.array([[np.dot(h, z * z), np.dot(h, z)], [np.dot(h, z), h.sum()]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        t = 1.0
        while t > 1e-6 and loss(a - t * step[0], b - t * step[1]) > cur:  # backtracking
            t *= 0.5
        a, b = a - t * step[0], b - t * step[1]
        new = loss(a, b)
        if cur - new < 1e-12:
            break
        cur = new
    return float(a), float(b)


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a * logit(p) + b)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error, equal-width bins, all classes pooled."""
    p, y = p.ravel(), y.ravel()
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS, min_count: int = 20):
    p, y = p.ravel(), y.ravel()
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_count:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def mean_ap(p: np.ndarray, y: np.ndarray) -> float:
    vals = [average_precision_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(vals))


# Fast weighted AUC. Patient bootstrap = integer weights per image (how many times its patient
# was drawn), so each resample is just a new weight vector over a fixed sort order.
def prep_auc(p: np.ndarray, y: np.ndarray) -> list:
    prep = []
    for i in range(y.shape[1]):
        order = np.argsort(p[:, i], kind="mergesort")
        _, g = np.unique(p[order, i], return_inverse=True)
        prep.append((order, g, int(g.max()) + 1, y[order, i].astype(np.float64)))
    return prep


def mean_auc_weighted(prep: list, w: np.ndarray | None = None) -> float:
    aucs = []
    for order, g, ng, ys in prep:
        ws = np.ones(len(order)) if w is None else w[order]
        pos = np.bincount(g, ws * ys, ng)
        neg = np.bincount(g, ws * (1 - ys), ng)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            continue
        below = np.cumsum(neg) - neg  # negatives strictly below each score group
        aucs.append(float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def patient_bootstrap(preps: dict[str, list], patients: np.ndarray, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Mean AUC of every model on the SAME n_boot patient resamples."""
    _, inv = np.unique(patients, return_inverse=True)
    n_pat = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in preps}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[inv].astype(np.float64)
        for k, prep in preps.items():
            out[k][b] = mean_auc_weighted(prep, w)
    return out


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def paired_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired bootstrap p-value for H0: AUC_a == AUC_b."""
    d = a - b
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


# --------------------------------------------------------------------------- subgroups
def subgroup_masks(meta: dict) -> list[tuple[str, str, np.ndarray]]:
    groups = []
    if "sex" in meta:
        sex = np.array([str(s).strip().upper()[:1] for s in meta["sex"]])
        groups += [("Sex", lab, sex == k) for k, lab in (("M", "Male"), ("F", "Female"))]
    if "age" in meta:
        age = meta["age"].astype(np.float64)
        groups += [("Age", "<40", age < 40), ("Age", "40-60", (age >= 40) & (age <= 60)), ("Age", ">60", age > 60)]
    if "view" in meta:
        view = np.array([str(v).strip().upper() for v in meta["view"]])
        groups += [("View", v, view == v) for v in ("PA", "AP")]
    return [g for g in groups if g[2].sum() > 0]


def analyze_subgroups(probs: dict[str, np.ndarray], meta: dict, n_boot: int, seed: int) -> list[dict]:
    y, rows = meta["labels"], []
    for kind, label, m in subgroup_masks(meta):
        preps = {k: prep_auc(p[m], y[m]) for k, p in probs.items()}
        boots = patient_bootstrap(preps, meta["patient"][m], n_boot, seed)
        for k in probs:
            lo, hi = ci(boots[k])
            rows.append({"group": kind, "level": label, "model": k, "n": int(m.sum()),
                         "auc": mean_auc_weighted(preps[k]), "ci_low": lo, "ci_high": hi})
    return rows


# --------------------------------------------------------------------------- outputs
def _color(name: str) -> str:
    return RED if name == ENSEMBLE else BLUE


def plot_forest(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.4))
    for i, r in enumerate(rows[::-1]):
        ax.errorbar(r["auc"], i, xerr=[[r["auc"] - r["ci_low"]], [r["ci_high"] - r["auc"]]],
                    fmt="o", color=_color(r["model"]), capsize=4, lw=1.5)
        ax.text(r["ci_high"] + 0.002, i, f"{r['auc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
                va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows[::-1]])
    lo = min(r["ci_low"] for r in rows)
    hi = max(r["ci_high"] for r in rows)
    ax.set_xlim(lo - 0.01, hi + 0.035)
    ax.set(xlabel="Mean test AUC (95% patient-level bootstrap CI)", title="Model comparison (official test set)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_calibration(cal: dict, y: np.ndarray, path: Path) -> None:
    names = list(cal) if len(cal) <= 6 else list(cal)[:1]
    ncol = min(len(names), 3)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        c = cal[name]
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect calibration")
        for p, col, lab in ((c["before"], BLUE, f"Raw (ECE {c['ece_before']:.3f})"),
                            (c["after"], RED, f"T={c['T']:.2f} (ECE {c['ece_after']:.3f})"),
                            (c["platt"], "#27ae60", f"T+bias (ECE {c['ece_platt']:.3f})")):
            xs, ys = reliability(p, y)
            ax.plot(xs, ys, "o-", color=col, ms=3.5, label=lab)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name, xlabel="Mean predicted probability",
               ylabel="Observed fraction positive")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Reliability: raw vs temperature vs temperature+bias (test, classes pooled, 15 bins)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_subgroups(rows: list[dict], models: list[str], path: Path) -> None:
    levels = list(dict.fromkeys((r["group"], r["level"]) for r in rows))
    fig, ax = plt.subplots(figsize=(1.1 * len(levels) + 2.5, 4.4))
    cmap = plt.get_cmap("tab10")
    width = 0.7 / max(len(models), 1)
    for j, m in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for i, lv in enumerate(levels):
            r = next((r for r in rows if (r["group"], r["level"]) == lv and r["model"] == m), None)
            if r:
                xs.append(i - 0.35 + width * (j + 0.5))
                ys.append(r["auc"])
                lo.append(r["auc"] - r["ci_low"])
                hi.append(r["ci_high"] - r["auc"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3, ms=4, label=m,
                    color=RED if m == ENSEMBLE else cmap(j))
    ax.set_xticks(range(len(levels)), [f"{g}\n{lv}" for g, lv in levels], fontsize=8)
    ax.set(ylabel="Mean test AUC (95% CI)", title="Subgroup performance (test set)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _fmt_params(v) -> str:
    return "--" if v is None else f"{v / 1e6:.1f}M"


def write_benchmark(rows: list[dict], out: Path) -> None:
    keys = ["model", "auc", "ci_low", "ci_high", "ap", "ece_before", "ece_after", "temperature",
            "ece_platt", "platt_a", "platt_b", "params", "latency_ms"]
    with open(out / "results" / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})
    best = max(r["auc"] for r in rows)
    lines = [r"\begin{tabular}{lcrrrrrrr}", r"\toprule",
             r"Model & Mean AUC [95\% CI] & AP & ECE & ECE (T) & $T$ & ECE (T+b) & Params & Latency (ms) \\", r"\midrule"]
    for r in rows:
        if r["model"] == ENSEMBLE:
            lines.append(r"\midrule")
        auc = f"{r['auc']:.4f}"
        auc = rf"\textbf{{{auc}}}" if r["auc"] == best else auc
        lat = "--" if r.get("latency_ms") is None else f"{r['latency_ms']:.1f}"
        lines.append(f"{_tex(r['model'])} & {auc} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] & {r['ap']:.3f} & "
                     f"{r['ece_before']:.3f} & {r['ece_after']:.3f} & {r['temperature']:.2f} & {r['ece_platt']:.3f} & "
                     f"{_fmt_params(r.get('params'))} & {lat} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "report" / "tables" / "benchmark.tex").write_text("\n".join(lines), encoding="utf-8")


def write_pairwise(rows: list[dict], out: Path) -> None:
    keys = ["model_a", "model_b", "auc_a", "auc_b", "diff", "diff_ci_low", "diff_ci_high", "p_value"]
    with open(out / "results" / "pairwise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_subgroups_tex(rows: list[dict], models: list[str], out: Path) -> None:
    levels = list(dict.fromkeys((r["group"], r["level"], r["n"]) for r in rows))
    lines = [r"\begin{tabular}{ll r" + "c" * len(models) + "}", r"\toprule",
             "Group & Level & $n$ & " + " & ".join(_tex(m) for m in models) + r" \\", r"\midrule"]
    for g, lv, n in levels:
        cells = []
        for m in models:
            r = next(r for r in rows if (r["group"], r["level"], r["model"]) == (g, lv, m))
            cells.append(f"{r['auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        lines.append(f"{g} & {_tex(lv)} & {n:,} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / "report" / "tables" / "subgroups.tex").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main
def run(preds: dict[str, str], out_dir: Path, n_boot: int = 1000, seed: int = 0,
        ensemble: list[str] | None = None, latency: dict | None = None, params: dict | None = None) -> dict:
    latency, params = latency or {}, params or {}
    models, shared = load_all(preds)
    test, val = shared["test"], shared["val"]

    members = ensemble or (list(models) if len(models) >= 2 else [])
    if members:
        unknown = set(members) - set(models)
        if unknown:
            raise SystemExit(f"--ensemble: unknown model(s) {sorted(unknown)}")
        models[ENSEMBLE] = {s: np.mean([models[m][s] for m in members], axis=0) for s in ("test", "val")}

    fig_dir, tab_dir, res_dir = out_dir / "report" / "figures", out_dir / "report" / "tables", out_dir / "results"
    for d in (fig_dir, tab_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # calibration
    cal = {}
    for name, m in models.items():
        t = fit_temperature(m["val"], val["labels"])
        after = apply_temperature(m["test"], t)
        pa, pb = fit_platt(m["val"], val["labels"])
        platt = apply_platt(m["test"], pa, pb)
        cal[name] = {"T": t, "before": m["test"], "after": after, "platt": platt, "platt_a": pa, "platt_b": pb,
                     "ece_platt": ece(platt, test["labels"]),
                     "ece_before": ece(m["test"], test["labels"]), "ece_after": ece(after, test["labels"])}

    # AUC + patient bootstrap (shared resamples)
    preps = {k: prep_auc(m["test"], test["labels"]) for k, m in models.items()}
    boots = patient_bootstrap(preps, test["patient"], n_boot, seed)

    bench = []
    for name, m in models.items():
        lo, hi = ci(boots[name])
        bench.append({"model": name, "auc": mean_auc_weighted(preps[name]), "ci_low": lo, "ci_high": hi,
                      "ap": mean_ap(m["test"], test["labels"]), "ece_before": cal[name]["ece_before"],
                      "ece_after": cal[name]["ece_after"], "temperature": cal[name]["T"],
                      "ece_platt": cal[name]["ece_platt"], "platt_a": cal[name]["platt_a"], "platt_b": cal[name]["platt_b"],
                      "mean_prob_before": float(m["test"].mean()), "mean_prob_after": float(cal[name]["after"].mean()),
                      "params": params.get(name), "latency_ms": latency.get(name)})

    pairs = []
    for a, b in itertools.combinations(models, 2):
        d = boots[a] - boots[b]
        lo, hi = ci(d)
        ra, rb = (next(r for r in bench if r["model"] == k) for k in (a, b))
        pairs.append({"model_a": a, "model_b": b, "auc_a": ra["auc"], "auc_b": rb["auc"],
                      "diff": ra["auc"] - rb["auc"], "diff_ci_low": lo, "diff_ci_high": hi,
                      "p_value": paired_p(boots[a], boots[b])})

    write_benchmark(bench, out_dir)
    write_pairwise(pairs, out_dir)
    temps = {k: {"T": c["T"], "ece_before": c["ece_before"], "ece_after": c["ece_after"],
                 "platt_a": c["platt_a"], "platt_b": c["platt_b"], "ece_platt": c["ece_platt"],
                 "mean_prob_platt": float(c["platt"].mean()),
                 "mean_prob_before": float(c["before"].mean()), "mean_prob_after": float(c["after"].mean())}
             for k, c in cal.items()}
    temps["_prevalence_test"] = float(test["labels"].mean())
    (res_dir / "temperatures.json").write_text(json.dumps(temps, indent=2), encoding="utf-8")
    plot_forest(bench, fig_dir / "forest_auc.png")
    plot_calibration(cal, test["labels"], fig_dir / "calibration_before_after.png")

    sub_rows = []
    if any(k in test for k in ("sex", "age", "view")):
        sub_rows = analyze_subgroups({k: m["test"] for k, m in models.items()}, test, n_boot, seed)
        if sub_rows:
            plot_subgroups(sub_rows, list(models), fig_dir / "subgroups.png")
            write_subgroups_tex(sub_rows, list(models), out_dir)
    else:
        print("no sex/age/view metadata in the predictions: subgroup analysis skipped")

    return {"benchmark": bench, "pairwise": pairs, "temperatures": temps, "subgroups": sub_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", action="append", required=True, help="name=path/to/predictions.npz (repeatable)")
    ap.add_argument("--latency", action="append", help="name=milliseconds per image (repeatable)")
    ap.add_argument("--params", action="append", help="name=parameter count (repeatable)")
    ap.add_argument("--ensemble", help="comma-separated members (default: all models when >= 2)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    args = ap.parse_args()

    res = run(_parse_pairs(args.preds), args.out_dir, n_boot=args.n_boot, seed=args.seed,
              ensemble=[s.strip() for s in args.ensemble.split(",")] if args.ensemble else None,
              latency=_parse_pairs(args.latency, float), params=_parse_pairs(args.params, lambda v: int(float(v))))
    for r in res["benchmark"]:
        print(f"{r['model']:>16}: AUC {r['auc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | AP {r['ap']:.4f} | "
              f"T {r['temperature']:.3f} | ECE {r['ece_before']:.4f} -> {r['ece_after']:.4f} (T+b {r['ece_platt']:.4f}) | "
              f"mean p {r['mean_prob_before']:.3f} -> {r['mean_prob_after']:.3f}")
    print(f"label prevalence {res['temperatures']['_prevalence_test']:.3f}")
    for p in res["pairwise"]:
        print(f"{p['model_a']} vs {p['model_b']}: dAUC {p['diff']:+.4f} [{p['diff_ci_low']:+.4f}, {p['diff_ci_high']:+.4f}] p={p['p_value']:.3f}")


if __name__ == "__main__":
    main()

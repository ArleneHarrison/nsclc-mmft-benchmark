"""Publication figure for the necessary supplementary experiments.

Panels (all real numbers):
  A. Incremental value forest plot: hold-out test AUC + bootstrap 95% CI for
     clinical -> +blood -> +metabolic(primary) -> +radiomics, with DeLong p.
  B. Honest model benchmark: ridge vs FT-Transformer vs MMFT ensemble vs MMFT
     rank-refit, test AUC + bootstrap 95% CI; DeLong MMFT-vs-ridge annotated n.s.
  C. Unbiased nested 5-fold CV AUC for the primary model (dots + mean).
  D. Calibration reliability curve (primary model, test) + decision-curve inset.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "outputs" / "supplementary_experiments"
BENCH = ROOT / "server_results" / "petct_blood_benchmark"
OUTDIR = ROOT / "outputs" / "supplementary_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"
BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#4d9221"
AMBER = "#d9a441"
GRID = "#d7dbe0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def boot_ci(y, p, n=3000, seed=20260701):
    y = np.asarray(y); p = np.asarray(p); m = len(y)
    r = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        idx = r.integers(0, m, m)
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    a = np.array(aucs)
    return roc_auc_score(y, p), np.percentile(a, 2.5), np.percentile(a, 97.5)


summary = json.loads((SUP / "supplementary_experiments_summary.json").read_text(encoding="utf-8"))
blocks = summary["incremental_blocks"]
delong = {d["comparison"]: d for d in summary["delong"]}
nested = summary["nested_cv_primary"]

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
              left=0.09, right=0.97, top=0.90, bottom=0.09)

# ---------------- Panel A: incremental forest ----------------
axA = fig.add_subplot(gs[0, 0])
labels = ["Clinical\nonly", "+ Blood /\ninflammatory", "+ PET metabolic\n(primary)", "+ Radiomics\n(3874)"]
aucs = [b["holdout_test_auc"] for b in blocks]
los = [b["holdout_test_ci_low"] for b in blocks]
his = [b["holdout_test_ci_high"] for b in blocks]
colors = [INK, INK, RED, "#9aa0a6"]
ys = np.arange(len(labels))[::-1]
for y, a, lo, hi, c in zip(ys, aucs, los, his, colors):
    axA.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=2)
    axA.scatter([a], [y], s=90, color=c, zorder=3, edgecolor="white", linewidth=1)
    axA.text(hi + 0.008, y, f"{a:.3f}", va="center", fontsize=10, color=c)
axA.axvline(0.5, color=GRID, ls="--", lw=1)
axA.set_yticks(ys); axA.set_yticklabels(labels, fontsize=9.5)
axA.set_xlim(0.45, 1.0); axA.set_xlabel("Hold-out test AUC (95% CI)")
axA.set_title("A  Incremental value of modalities", loc="left")
# DeLong annotations
d1 = delong["clinical+blood+metabolic(primary) vs clinical+blood"]["delong_p"]
d2 = delong["+radiomics(all) vs clinical+blood+metabolic(primary)"]["delong_p"]
axA.annotate(f"metabolic adds signal\nDeLong p={d1:.3f} *", xy=(0.78, ys[2]),
             xytext=(0.50, ys[2] + 0.35), fontsize=8.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
axA.annotate(f"radiomics hurts\nDeLong p={d2:.3f} *", xy=(0.72, ys[3]),
             xytext=(0.50, ys[3] + 0.30), fontsize=8.5, color="#616161",
             arrowprops=dict(arrowstyle="->", color="#616161", lw=1.2))
axA.set_ylim(-0.6, len(labels) - 0.2)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ---------------- Panel B: honest model benchmark ----------------
axB = fig.add_subplot(gs[0, 1])
ridge = pd.read_csv(BENCH / "best_model_test_predictions.csv")
mmft_e = pd.read_csv(BENCH / "mmft_cv_best_predictions.csv")
mmft_r = pd.read_csv(BENCH / "mmft_rank_refit_predictions.csv")
models = [
    ("Ridge logistic\n(primary)", ridge, BLUE),
    ("MMFT ensemble\n(5-fold CV)", mmft_e, GREEN),
    ("MMFT rank-refit\n(exploratory)", mmft_r, AMBER),
]
xs = np.arange(len(models))
for x, (name, dfm, c) in zip(xs, models):
    a, lo, hi = boot_ci(dfm["histology"].astype(int), dfm["predicted_probability"])
    axB.bar(x, a, width=0.55, color=c, alpha=0.85, edgecolor=INK, linewidth=0.8, zorder=2)
    axB.errorbar(x, a, yerr=[[a - lo], [hi - a]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=3)
    axB.text(x, a + (hi - a) + 0.012, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
# FT-transformer point (no per-patient preds): annotate as reference line
axB.axhline(0.815, xmin=0.02, xmax=0.98, color="#9aa0a6", ls=":", lw=1.3)
axB.text(len(models) - 1, 0.815 - 0.02, "FT-Transformer 0.815", ha="right", fontsize=8, color="#616161")
axB.set_xticks(xs); axB.set_xticklabels([m[0] for m in models], fontsize=9)
axB.set_ylim(0.5, 1.0); axB.set_ylabel("Hold-out test AUC (95% CI)")
axB.set_title("B  Deep learning is not superior to ridge", loc="left")
dp = delong["MMFT rank-refit vs Ridge (shared IDs)"]["delong_p"]
axB.plot([0, 0, 2, 2], [0.94, 0.955, 0.955, 0.94], color=INK, lw=1.1)
axB.text(1, 0.958, f"DeLong p = {dp:.2f}  (n.s.)", ha="center", fontsize=9)
axB.axhline(0.5, color=GRID, ls="--", lw=1)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

# ---------------- Panel C: nested CV ----------------
axC = fig.add_subplot(gs[1, 0])
folds = nested["nested_cv_folds"]
xs = np.arange(1, len(folds) + 1)
axC.scatter(xs, folds, s=110, color=RED, edgecolor="white", linewidth=1, zorder=3)
mean = nested["nested_cv_auc_mean"]; std = nested["nested_cv_auc_std"]
axC.axhspan(mean - std, mean + std, color=RED, alpha=0.10, zorder=1)
axC.axhline(mean, color=RED, lw=1.8, zorder=2)
axC.axhline(0.854, color=BLUE, ls="--", lw=1.5)
axC.text(len(folds) + 0.05, 0.854, "hold-out 0.854", va="center", fontsize=9, color=BLUE)
axC.text(len(folds) + 0.05, mean, f"nested-CV\n{mean:.3f}±{std:.3f}", va="center", fontsize=9, color=RED)
axC.set_xticks(xs); axC.set_xticklabels([f"Fold {i}" for i in xs], fontsize=9)
axC.set_ylim(0.6, 0.95); axC.set_xlim(0.5, len(folds) + 1.4)
axC.set_ylabel("AUC"); axC.set_title("C  Unbiased nested 5-fold CV (primary)", loc="left")
for s in ["top", "right"]:
    axC.spines[s].set_visible(False)

# ---------------- Panel D: calibration + DCA inset ----------------
axD = fig.add_subplot(gs[1, 1])
yt = ridge["histology"].astype(int).to_numpy()
pt = ridge["predicted_probability"].to_numpy()
frac_pos, mean_pred = calibration_curve(yt, pt, n_bins=5, strategy="quantile")
axD.plot([0, 1], [0, 1], ls="--", color=GRID, lw=1.2)
axD.plot(mean_pred, frac_pos, "o-", color=BLUE, lw=2, markersize=8, label="Primary model")
cal = summary["calibration_primary_test"]
axD.text(0.05, 0.93, f"Brier = {cal['brier']:.3f}\nslope = {cal['calibration_slope']:.2f}\nHL p = {cal['hosmer_lemeshow_p']:.3f}",
         fontsize=9, va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
axD.set_xlim(0, 1); axD.set_ylim(0, 1)
axD.set_xlabel("Predicted probability of squamous")
axD.set_ylabel("Observed frequency")
axD.set_title("D  Calibration (primary, test) + decision curve", loc="left")
for s in ["top", "right"]:
    axD.spines[s].set_visible(False)
# DCA inset
dca = pd.read_csv(SUP / "S3_decision_curve_primary.csv")
axins = axD.inset_axes([0.56, 0.12, 0.40, 0.40])
axins.plot(dca["threshold"], dca["net_benefit_model"], color=RED, lw=1.6, label="Model")
axins.plot(dca["threshold"], dca["net_benefit_all"], color="#9aa0a6", lw=1.0, label="Treat all")
axins.axhline(0, color=INK, lw=0.8)
axins.set_xlim(0, 0.6); axins.set_ylim(-0.1, 0.65)
axins.set_title("Net benefit", fontsize=8)
axins.tick_params(labelsize=6)
axins.legend(fontsize=5.5, loc="upper right", frameon=False)

fig.suptitle("Supplementary experiments: incremental value, honest benchmarking, calibration (PLOS-2024 cohort, n=255)",
             fontsize=13.5, fontweight="bold", y=0.965)
out = OUTDIR / "FigS7_incremental_value_and_honest_benchmark.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

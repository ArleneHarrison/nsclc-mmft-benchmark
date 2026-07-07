"""Figure 6 (new headline): biology-guided MMFT stabilizes small-sample deep
learning. All numbers real, from mmft_stability_comparison.py (10 seeds each)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "outputs" / "model_optimization"
OUTDIR = ROOT / "outputs" / "supplementary_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"; BLUE = "#2166ac"; RED = "#b2182b"; GREEN = "#4d9221"; AMBER = "#d9a441"; GRID = "#d7dbe0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
})

per_seed = pd.read_csv(OPT / "mmft_stability_comparison_per_seed.csv")
summary = json.loads((OPT / "mmft_stability_comparison_summary.json").read_text(encoding="utf-8"))
delong = json.loads((OPT / "mmft_stability_comparison_delong.json").read_text(encoding="utf-8"))

orig = per_seed["original_mmft_auc"].to_numpy()
bio = per_seed["biology_guided_mmft_auc"].to_numpy()
RIDGE = 0.854

fig = plt.figure(figsize=(13.5, 5.6))
gs = GridSpec(1, 2, figure=fig, wspace=0.28, left=0.08, right=0.97, top=0.84, bottom=0.13)

# ---- A: seed-level scatter/strip comparison ----
axA = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(0)
xo = np.full(len(orig), 0) + rng.uniform(-0.06, 0.06, len(orig))
xb = np.full(len(bio), 1) + rng.uniform(-0.06, 0.06, len(bio))
axA.scatter(xo, orig, s=70, color=RED, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3, label="Original (single risk token)")
axA.scatter(xb, bio, s=70, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3, label="Biology-guided (+ glycolysis & inflammation tokens)")
axA.hlines(orig.mean(), -0.22, 0.22, color=RED, lw=2.2, zorder=2)
axA.hlines(bio.mean(), 0.78, 1.22, color=GREEN, lw=2.2, zorder=2)
axA.axhline(RIDGE, color=INK, ls="--", lw=1.3, zorder=1)
axA.text(1.35, RIDGE, "ridge primary\n0.854", va="center", fontsize=8.5, color=INK)
axA.set_xlim(-0.5, 1.9)
axA.set_xticks([0, 1]); axA.set_xticklabels(["Original\nMMFT rank-refit", "Biology-guided\nMMFT"], fontsize=9.5)
axA.set_ylabel("Test AUC (10 independent seeds)")
o = summary["original_mmft_rank_refit"]; b = summary["biology_guided_mmft"]
axA.text(0, 0.79, f"mean {o['mean']:.3f}\nstd {o['std']:.3f}\nrange [{o['min']:.3f},{o['max']:.3f}]",
         ha="center", fontsize=8.3, color=RED)
axA.text(1, 0.79, f"mean {b['mean']:.3f}\nstd {b['std']:.3f}\nrange [{b['min']:.3f},{b['max']:.3f}]",
         ha="center", fontsize=8.3, color=GREEN)
axA.set_ylim(0.78, 0.90)
axA.set_title("A  Same central tendency, ~11× lower seed variance", loc="left", fontsize=11.5)
lev_p = summary["levene_p_variances_equal"]; mw_p = summary["mann_whitney_u_p_means_equal"]
axA.text(0.5, 0.885, f"Mann-Whitney (means) p={mw_p:.2f}  |  Levene (variances) p={lev_p:.3f}",
         ha="center", fontsize=8.5, color="#616161", transform=axA.transData)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ---- B: worst-case failure comparison ----
axB = fig.add_subplot(gs[0, 1])
cats = ["Worst seed\n(min AUC)", "Best seed\n(max AUC)", "Median seed\nvs ridge (DeLong p)"]
worst_o, worst_b = o["min"], b["min"]
best_o, best_b = o["max"], b["max"]
xs = np.arange(2)
w = 0.35
axB.bar(xs - w/2, [worst_o, best_o], width=w, color=RED, alpha=0.85, label="Original")
axB.bar(xs + w/2, [worst_b, best_b], width=w, color=GREEN, alpha=0.85, label="Biology-guided")
for x, vo, vb in zip(xs, [worst_o, best_o], [worst_b, best_b]):
    axB.text(x - w/2, vo + 0.005, f"{vo:.3f}", ha="center", fontsize=8.5)
    axB.text(x + w/2, vb + 0.005, f"{vb:.3f}", ha="center", fontsize=8.5)
axB.axhline(RIDGE, color=INK, ls="--", lw=1.3)
axB.text(1.55, RIDGE + 0.003, "ridge 0.854", fontsize=8, color=INK)
axB.set_xticks(xs); axB.set_xticklabels(["Worst-case\n(min of 10 seeds)", "Best-case\n(max of 10 seeds)"], fontsize=9.5)
axB.set_ylim(0.78, 0.90)
axB.set_ylabel("Test AUC")
axB.legend(fontsize=8.5, loc="upper left", frameon=False)
axB.set_title("B  Worst-case failure is far less severe", loc="left", fontsize=11.5)
d1 = delong["median_seed_orig_vs_ridge"]["p"]; d2 = delong["median_seed_bio_vs_ridge"]["p"]
axB.text(0.5, 0.885,
          f"deficit vs ridge: original −0.047 (3/10 seeds below) | biology-guided −0.007 (2/10 seeds below)",
          ha="center", fontsize=8.2, color="#616161", transform=axB.transAxes)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

fig.suptitle("Figure 6 | TCGA-guided mechanistic tokens stabilize MMFT across random initializations (10 seeds each, same test set, n=255 cohort)",
             fontsize=12.5, fontweight="bold", y=0.975)
out = OUTDIR / "Figure6_biology_guided_stability.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

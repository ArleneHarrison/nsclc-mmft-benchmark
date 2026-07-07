"""Figure 9: cross-cohort, cross-modality synthesis -- deep learning vs simple
ridge, tested across two independent cohorts and three structurally distinct
feature modalities. All numbers real."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "supplementary_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"; BLUE = "#2166ac"; RED = "#b2182b"; GREEN = "#4d9221"; AMBER = "#d9a441"; GRID = "#d7dbe0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
})

fig = plt.figure(figsize=(13.5, 6.2))
gs = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.07, right=0.98, top=0.83, bottom=0.13)

# ---- A: PLOS-2024 cohort, all DL/optimization attempts vs ridge ----
axA = fig.add_subplot(gs[0, 0])
labels_a = ["Ridge\n(primary)", "MMFT\n(single seed)", "Leakage-free\nsearch", "TabPFN\n(Nature 2025)",
            "Bio-token\nsearch"]
vals_a = [0.854, 0.868, 0.828, 0.852, 0.834]
pvals_a = [None, 0.37, 0.22, 0.98, 0.35]
xs = np.arange(len(labels_a))
colors_a = [INK, BLUE, BLUE, GREEN, BLUE]
for x, v, p, c in zip(xs, vals_a, pvals_a, colors_a):
    axA.bar(x, v, width=0.6, color=c, alpha=0.85, edgecolor=INK, linewidth=0.8)
    label = f"{v:.3f}" if p is None else f"{v:.3f}\np={p:.2f}"
    axA.text(x, v + 0.012, label, ha="center", fontsize=8.8)
axA.axhline(0.854, color=INK, ls=":", lw=1.1)
axA.set_xticks(xs); axA.set_xticklabels(labels_a, fontsize=8.8)
axA.set_ylim(0.5, 1.0); axA.set_ylabel("Test AUC")
axA.set_title("A  PLOS-2024 cohort (n=255): 4 methods, none beat ridge", loc="left", fontsize=11)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ---- B: cross-cohort, cross-modality ridge-vs-DL summary ----
axB = fig.add_subplot(gs[0, 1])
groups = ["PLOS-2024\nPET+blood\n(n=255)", "TCIA\nCT semantic\n(n=187)"]
ridge_vals = [0.778, 0.749]   # robust nested-CV estimates for fair cross-cohort comparison
dl_vals = [0.778, 0.713]      # MMFT nested-CV-comparable robust estimates (PLOS: nested-CV proxy via repeated holdout mean; TCIA: nested-CV)
ridge_err = [0.048, 0.105]
dl_err = [0.057, 0.108]
x = np.arange(len(groups))
w = 0.32
axB.bar(x - w/2, ridge_vals, width=w, yerr=ridge_err, color=INK, alpha=0.85, capsize=4, label="Ridge logistic (robust estimate)")
axB.bar(x + w/2, dl_vals, width=w, yerr=dl_err, color=RED, alpha=0.85, capsize=4, label="Best deep-learning attempt (robust estimate)")
for xi, (rv, dv) in enumerate(zip(ridge_vals, dl_vals)):
    axB.text(xi - w/2, rv + ridge_err[xi] + 0.02, f"{rv:.3f}", ha="center", fontsize=9)
    axB.text(xi + w/2, dv + dl_err[xi] + 0.02, f"{dv:.3f}", ha="center", fontsize=9)
axB.set_xticks(x); axB.set_xticklabels(groups, fontsize=9.5)
axB.set_ylim(0.4, 1.0); axB.set_ylabel("Test AUC (robust estimate ± SD)")
axB.legend(fontsize=8.5, loc="upper right", frameon=False)
axB.set_title("B  2 independent cohorts, 3 feature modalities: same answer", loc="left", fontsize=11)
axB.text(0.5, -0.22, "Robust estimates: PLOS-2024 nested-CV/repeated hold-out; TCIA nested-CV (5x5, 3-4 repeats).\nDeep learning never exceeds the simple model once evaluated without leakage.",
          transform=axB.transAxes, ha="center", fontsize=8, color="#616161")
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

fig.suptitle("Figure 9 | Deep learning vs ridge regression: consistent across cohorts, modalities, and methods",
             fontsize=13, fontweight="bold", y=0.975)
out = OUTDIR / "Figure9_cross_cohort_dl_vs_ridge.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

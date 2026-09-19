"""Figure 6: model-optimization attempts and estimate stability (all real numbers)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "outputs" / "model_optimization"
OUTDIR = Path(os.environ.get("PLOS_REVISION_OUT", ROOT / "outputs" / "supplementary_figures"))
OUTDIR.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"; BLUE = "#2166ac"; RED = "#b2182b"; GREEN = "#4d9221"; AMBER = "#d9a441"; GRID = "#d7dbe0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
})

summary = json.loads((OPT / "model_optimization_summary.json").read_text(encoding="utf-8"))
rep = np.loadtxt(OPT / "repeated_holdout_aucs.csv", delimiter=",", skiprows=1)
lc = pd.read_csv(OPT / "learning_curve.csv")
sub = pd.read_csv(OPT / "subgroup_by_stage.csv")
sub["subgroup"] = sub["subgroup"].str.replace(r"\s*\(likely III[A-C]\)", "", regex=True)

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, left=0.08, right=0.97, top=0.90, bottom=0.09)

# ---- A: repeated hold-out distribution vs single split & nested-CV ----
axA = fig.add_subplot(gs[0, 0])
axA.hist(rep, bins=14, color=BLUE, alpha=0.75, edgecolor="white")
rh = summary["repeated_holdout_50splits"]
axA.axvline(summary["primary_reference"]["test_auc"], color=RED, lw=2.2, ls="-",
            label=f"prespecified held-out split = {summary['primary_reference']['test_auc']:.3f}")
axA.axvline(rh["mean"], color=BLUE, lw=2.2, ls="-", label=f"50-split mean = {rh['mean']:.3f}")
axA.axvline(0.778, color=GREEN, lw=1.8, ls="--", label="nested-CV = 0.778 (independent method)")
axA.set_xlabel("Test AUC"); axA.set_ylabel("Count of splits (n=50)")
axA.set_title("A  Repeated hold-out distribution", loc="left")
axA.legend(fontsize=8.2, loc="upper left", frameon=False)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ---- B: optimization attempts vs primary ----
axB = fig.add_subplot(gs[0, 1])
attempts = [
    ("Primary\n(ridge, k=5)", summary["primary_reference"]["test_auc"], None, INK),
    ("Elastic-net", summary["elastic_net"]["test_auc"], summary["elastic_net"]["delong_p_vs_ridge"], BLUE),
    ("Primary +\nradiomics signature", summary["radiomics_signature"]["primary_plus_signature_test_auc_like_for_like"],
     summary["radiomics_signature"]["delong_p_vs_primary"], RED),
    ("Ridge + XGBoost\nstack", summary["stacking_ensemble"]["stack_test_auc"],
     summary["stacking_ensemble"]["delong_p_vs_primary"], GREEN),
]
xs = np.arange(len(attempts))
for x, (name, auc, p, c) in zip(xs, attempts):
    axB.bar(x, auc, width=0.55, color=c, alpha=0.85, edgecolor=INK, linewidth=0.8, zorder=2)
    label = f"{auc:.3f}" if p is None else f"{auc:.3f}\np={p:.2f}"
    axB.text(x, auc + 0.015, label, ha="center", fontsize=8.8,
              fontweight="bold" if p is not None and p < 0.05 else "normal",
              color=RED if (p is not None and p < 0.05) else INK)
axB.axhline(summary["primary_reference"]["test_auc"], color=INK, ls=":", lw=1.1)
axB.set_xticks(xs); axB.set_xticklabels([a[0] for a in attempts], fontsize=8.8)
axB.set_ylim(0.5, 1.0); axB.set_ylabel("Test AUC")
axB.set_title("B  Optimization attempts vs primary (exploratory DeLong p)", loc="left")
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

# ---- C: learning curve ----
axC = fig.add_subplot(gs[1, 0])
axC.errorbar(lc["train_fraction"] * 100, lc["mean_auc"], yerr=lc["std_auc"],
             fmt="o-", color=BLUE, lw=2, markersize=7, capsize=4, ecolor=GRID)
for _, r in lc.iterrows():
    axC.annotate(f"n={int(r['n_train'])}", (r["train_fraction"] * 100, r["mean_auc"]),
                 textcoords="offset points", xytext=(0, 10), fontsize=7.5, ha="center", color="#616161")
axC.set_xlabel("Training-set fraction used (%)"); axC.set_ylabel("Test AUC (mean of 20 resamples)")
axC.set_title("C  Learning curve: more data still helps", loc="left")
axC.set_ylim(0.6, 0.95)
for s in ["top", "right"]:
    axC.spines[s].set_visible(False)

# ---- D: subgroup by stage ----
axD = fig.add_subplot(gs[1, 1])
ys = np.arange(len(sub))[::-1]
for y, (_, r) in zip(ys, sub.iterrows()):
    if pd.isna(r["auc"]):
        continue
    axD.plot([r["ci_low"], r["ci_high"]], [y, y], color=AMBER, lw=2.2, zorder=2)
    axD.scatter([r["auc"]], [y], s=90, color=AMBER, zorder=3, edgecolor="white", linewidth=1)
    axD.text(r["ci_high"] + 0.02, y, f"{r['auc']:.3f} (n={int(r['n'])})", va="center", fontsize=9)
axD.axvline(0.5, color=GRID, ls="--", lw=1)
axD.set_yticks(ys); axD.set_yticklabels(sub["subgroup"], fontsize=9)
axD.set_xlim(0.4, 1.05); axD.set_xlabel("Test AUC (95% bootstrap CI)")
axD.set_title("D  Subgroup stability by stage substratum", loc="left")
axD.text(0.02, -0.22, "All 255 patients are AJCC Stage III per source inclusion criteria;\n"
         "the public files do not document how raw codes 1/2/3 map to substages.",
         transform=axD.transAxes, fontsize=7.3, color="#616161", va="top")
for s in ["top", "right"]:
    axD.spines[s].set_visible(False)

fig.suptitle("Figure 6 | Model-optimization attempts and estimate stability (real, PLOS-2024 cohort, n=255)",
             fontsize=13.5, fontweight="bold", y=0.965)
out = OUTDIR / "Figure6_model_optimization.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

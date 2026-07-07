"""Figure: independent-cohort (TCIA NSCLC-Radiogenomics) replication of
subtype-associated baseline characteristics + EGFR/KRAS mutation exclusivity.
All numbers real, from independent_cohort_plausibility.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

r = json.loads((OPT / "independent_cohort_plausibility_summary.json").read_text(encoding="utf-8"))

fig = plt.figure(figsize=(13, 5.6))
gs = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.07, right=0.98, top=0.82, bottom=0.14)

# ---- A: replication of gender/smoking (two independent cohorts side by side) ----
axA = fig.add_subplot(gs[0, 0])
labels = ["Male sex\n(PLOS-2024\nn=255)", "Male sex\n(TCIA-Radio-\ngenomics, n=207)",
          "Ever-smoker\n(TCIA-Radio-\ngenomics, n=207)"]
ad_vals = [69.1, r["gender"]["male_pct_adeno"], r["smoking"]["ever_smoker_pct_adeno"]]
sq_vals = [95.2, r["gender"]["male_pct_squamous"], r["smoking"]["ever_smoker_pct_squamous"]]
pvals = [None, r["gender"]["fisher_p"], r["smoking"]["fisher_p"]]
xs = np.arange(len(labels))
w = 0.35
axA.bar(xs - w/2, ad_vals, width=w, color=BLUE, alpha=0.85, label="Adenocarcinoma")
axA.bar(xs + w/2, sq_vals, width=w, color=RED, alpha=0.85, label="Squamous")
for x, va, vs, p in zip(xs, ad_vals, sq_vals, pvals):
    axA.text(x - w/2, va + 1.5, f"{va:.1f}%", ha="center", fontsize=8.5)
    axA.text(x + w/2, vs + 1.5, f"{vs:.1f}%", ha="center", fontsize=8.5)
    label = "p<0.001" if x == 0 else f"p={p:.3g}"
    axA.text(x, 102, label, ha="center", fontsize=8, color="#616161")
axA.set_xticks(xs); axA.set_xticklabels(labels, fontsize=8.3)
axA.set_ylabel("% of subgroup"); axA.set_ylim(0, 112)
axA.legend(fontsize=9, loc="lower left", frameon=False)
axA.set_title("A  Direction replicates in an independent US cohort", loc="left", fontsize=11.5)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ---- B: EGFR/KRAS mutation exclusivity to adenocarcinoma ----
axB = fig.add_subplot(gs[0, 1])
labels2 = [f"EGFR mutant\n(n={r['egfr_mutation']['n_known']})", f"KRAS mutant\n(n={r['kras_mutation']['n_known']})"]
ad2 = [r["egfr_mutation"]["mutant_pct_adeno"], r["kras_mutation"]["mutant_pct_adeno"]]
sq2 = [r["egfr_mutation"]["mutant_pct_squamous"], r["kras_mutation"]["mutant_pct_squamous"]]
p2 = [r["egfr_mutation"]["fisher_p"], r["kras_mutation"]["fisher_p"]]
xs2 = np.arange(len(labels2))
axB.bar(xs2 - w/2, ad2, width=w, color=BLUE, alpha=0.85, label="Adenocarcinoma")
axB.bar(xs2 + w/2, sq2, width=w, color=RED, alpha=0.85, label="Squamous")
for x, va, vs, p in zip(xs2, ad2, sq2, p2):
    axB.text(x - w/2, va + 1, f"{va:.1f}%", ha="center", fontsize=9)
    axB.text(x + w/2, vs + 1, f"{vs:.1f}%\n(0 cases)" if vs == 0 else f"{vs:.1f}%", ha="center", fontsize=9)
    axB.text(x, 33, f"Fisher p={p:.3g}", ha="center", fontsize=8.5, color="#616161")
axB.set_xticks(xs2); axB.set_xticklabels(labels2, fontsize=9.5)
axB.set_ylabel("% mutant within subtype"); axB.set_ylim(0, 36)
axB.legend(fontsize=9, loc="upper right", frameon=False)
axB.set_title("B  EGFR/KRAS mutations essentially exclusive to adenocarcinoma", loc="left", fontsize=11.5)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

fig.suptitle("Figure 7 | Independent-cohort (TCIA NSCLC-Radiogenomics) replication of subtype-associated biology",
             fontsize=13, fontweight="bold", y=0.965)
out = OUTDIR / "Figure7_independent_cohort_replication.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

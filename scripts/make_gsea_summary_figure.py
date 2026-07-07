"""Figure 10: unbiased GSEA across all 50 MSigDB Hallmark pathways (LUSC vs
LUAD), highlighting where the paper's a priori glycolysis/hypoxia and
inflammatory axes actually rank -- including the honest finding that several
inflammation-related pathways run in the OPPOSITE direction from the
paper's curated-marker-based claim."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "outputs" / "model_optimization"
OUTDIR = ROOT / "outputs" / "supplementary_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"; BLUE = "#2166ac"; RED = "#b2182b"; GREEN = "#4d9221"; AMBER = "#d9a441"; GRID = "#d7dbe0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK,
    "axes.linewidth": 1.0, "figure.dpi": 150,
})

df = pd.read_csv(OPT / "tcga_hallmark_gsea_all50_results.csv")
df["NES"] = df["NES"].astype(float)
df["FDR q-val"] = df["FDR q-val"].astype(float)
df = df.sort_values("NES", ascending=True).reset_index(drop=True)

GLYC_HYPOXIA = {"Glycolysis", "Hypoxia"}
INFLAM = {"Inflammatory Response", "TNF-alpha Signaling via NF-kB", "IL-6/JAK/STAT3 Signaling",
          "Complement", "Interferon Gamma Response", "Allograft Rejection"}

fig, ax = plt.subplots(figsize=(11, 13))
ys = np.arange(len(df))
colors = []
for term in df["Term"]:
    if term in GLYC_HYPOXIA:
        colors.append(RED)
    elif term in INFLAM:
        colors.append(BLUE)
    else:
        colors.append("#b5bec7")

ax.barh(ys, df["NES"], color=colors, edgecolor=INK, linewidth=0.4, height=0.72)
sig = df["FDR q-val"] < 0.05
for y, nes, q, s in zip(ys, df["NES"], df["FDR q-val"], sig):
    if s:
        ax.text(nes + (0.05 if nes >= 0 else -0.05), y, "*", fontsize=13, fontweight="bold",
                 color=INK, va="center", ha="left" if nes >= 0 else "right")

ax.set_yticks(ys)
ax.set_yticklabels(df["Term"], fontsize=8.3)
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("Normalized Enrichment Score (NES): positive = squamous-high, negative = adenocarcinoma-high")
ax.set_title("Figure 10 | Unbiased GSEA across all 50 MSigDB Hallmark pathways (LUSC vs LUAD, TCGA)\n"
              "* FDR q<0.05.  Red = a priori glycolysis/hypoxia axis.  Blue = a priori inflammatory-family axis.",
              fontsize=11.5, fontweight="bold", loc="left")

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=RED, edgecolor=INK, label="Glycolysis / Hypoxia (a priori)"),
                Patch(facecolor=BLUE, edgecolor=INK, label="Inflammatory-family pathways (a priori)"),
                Patch(facecolor="#b5bec7", edgecolor=INK, label="Other Hallmark pathways")]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9, frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

plt.tight_layout()
out = OUTDIR / "Figure10_gsea_all50_hallmark.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

"""Create the corrected primary-tumor-only Hallmark GSEA figure for revision."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "outputs" / "PONE-D-26-33174_revision_20260919"
ANALYSIS = REV / "analysis" / "tcga_primary_tumor_gsea"
OUT = REV / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2937"
BLUE = "#2166ac"
RED = "#b2182b"
GRAY = "#b5bec7"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 1.0,
        "figure.dpi": 180,
    }
)

df = pd.read_csv(ANALYSIS / "tcga_hallmark_gsea_all50_results.csv")
df["NES"] = df["NES"].astype(float)
df["FDR q-val"] = df["FDR q-val"].astype(float)
df = df.sort_values("NES").reset_index(drop=True)

glycolysis = {"Glycolysis", "Hypoxia"}
inflammation = {
    "Inflammatory Response",
    "TNF-alpha Signaling via NF-kB",
    "IL-6/JAK/STAT3 Signaling",
    "Complement",
    "Interferon Gamma Response",
    "Allograft Rejection",
}
colors = [RED if t in glycolysis else BLUE if t in inflammation else GRAY for t in df["Term"]]

fig, ax = plt.subplots(figsize=(11, 13))
positions = np.arange(len(df))
ax.barh(positions, df["NES"], color=colors, edgecolor=INK, linewidth=0.4, height=0.72)
for y, nes, q_value in zip(positions, df["NES"], df["FDR q-val"]):
    if q_value < 0.05:
        ax.text(
            nes + (0.05 if nes >= 0 else -0.05),
            y,
            "*",
            fontsize=13,
            fontweight="bold",
            color=INK,
            va="center",
            ha="left" if nes >= 0 else "right",
        )

ax.set_yticks(positions)
ax.set_yticklabels(df["Term"], fontsize=8.3)
ax.axvline(0, color=INK, linewidth=1)
ax.set_xlabel("Normalized enrichment score (positive = squamous-high; negative = adenocarcinoma-high)")
ax.set_title(
    "Hallmark GSEA in TCGA primary tumors (LUSC n=501 vs LUAD n=528)\n"
    "* FDR q<0.05; red = a priori glycolysis/hypoxia; blue = inflammatory-family pathways",
    fontsize=11.5,
    fontweight="bold",
    loc="left",
)
ax.legend(
    handles=[
        Patch(facecolor=RED, edgecolor=INK, label="Glycolysis / hypoxia (a priori)"),
        Patch(facecolor=BLUE, edgecolor=INK, label="Inflammatory family (a priori)"),
        Patch(facecolor=GRAY, edgecolor=INK, label="Other Hallmark pathways"),
    ],
    loc="lower right",
    fontsize=9,
    frameon=False,
)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
for suffix in ["png", "tiff"]:
    fig.savefig(OUT / f"Fig10_TCGA_primary_tumor_Hallmark_GSEA.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
print(OUT / "Fig10_TCGA_primary_tumor_Hallmark_GSEA.png")

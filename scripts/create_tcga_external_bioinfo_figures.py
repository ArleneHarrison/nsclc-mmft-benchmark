from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xenaPython as xena
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reference_style_reproduction"
DATA_OUT = ROOT / "public_data" / "tcga_xena_lung_external"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

HOST = xena.PUBLIC_HUBS["gdcHub"]
COHORTS = {
    "LUAD": {
        "expression": "TCGA-LUAD.star_tpm.tsv",
        "survival": "TCGA-LUAD.survival.tsv",
    },
    "LUSC": {
        "expression": "TCGA-LUSC.star_tpm.tsv",
        "survival": "TCGA-LUSC.survival.tsv",
    },
}

GENES = [
    "CEACAM5",
    "SLC2A1",
    "HK2",
    "LDHA",
    "PKM",
    "ENO1",
    "ALDOA",
    "GAPDH",
    "HIF1A",
    "VEGFA",
    "CA9",
    "TP63",
    "KRT5",
    "KRT6A",
    "KRT14",
    "DSG3",
    "SOX2",
    "NKX2-1",
    "NAPSA",
    "SFTPA1",
    "SFTPB",
    "EPCAM",
    "MUC1",
    "CD274",
    "PDCD1",
    "CTLA4",
    "LAG3",
    "HAVCR2",
    "TIGIT",
    "CD8A",
    "CD8B",
    "CD3D",
    "CD3E",
    "FOXP3",
    "MS4A1",
    "CD68",
    "CD163",
    "MRC1",
    "ITGAM",
    "FCGR3B",
    "S100A8",
    "S100A9",
    "CXCL8",
    "CXCL10",
    "IFNG",
    "GZMB",
    "PRF1",
    "MKI67",
    "TOP2A",
    "VIM",
    "CDH1",
    "TWIST1",
    "SNAI1",
    "ZEB1",
    "MMP9",
    "COL1A1",
    "ACTA2",
    "KRT7",
    "KRT18",
    "KRT19",
    "SCGB3A1",
    "ALDH1A1",
    "EGFR",
    "ERBB2",
    "KRAS",
    "BRAF",
    "MET",
    "ALK",
    "RET",
]

PATHWAYS = {
    "Adenocarcinoma marker": ["CEACAM5", "NKX2-1", "NAPSA", "SFTPB", "KRT7", "KRT18", "KRT19"],
    "Squamous marker": ["TP63", "KRT5", "KRT6A", "KRT14", "DSG3", "SOX2"],
    "Glycolysis/hypoxia": ["SLC2A1", "HK2", "LDHA", "PKM", "ENO1", "ALDOA", "GAPDH", "HIF1A", "CA9"],
    "T-cell cytotoxicity": ["CD8A", "CD8B", "CD3D", "CD3E", "GZMB", "PRF1", "IFNG"],
    "Checkpoint": ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"],
    "Myeloid/neutrophil": ["CD68", "CD163", "MRC1", "ITGAM", "FCGR3B", "S100A8", "S100A9", "CXCL8"],
    "EMT/stroma": ["VIM", "TWIST1", "SNAI1", "ZEB1", "MMP9", "COL1A1", "ACTA2"],
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COL = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
    "gold": "#FFE15B",
    "gray": "#C5CAD3",
    "red": "#EF6F6C",
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "grid.color": TOKENS["grid"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        },
    )


def tumor_samples(dataset: str) -> list[str]:
    samples = xena.dataset_samples(HOST, dataset, None)
    return [s for s in samples if len(s) >= 15 and s[13:15] == "01"]


def fetch_expression_for_cohort(cohort: str, refresh: bool = False) -> pd.DataFrame:
    cache = DATA_OUT / f"TCGA_{cohort}_selected_gene_tpm.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    dataset = COHORTS[cohort]["expression"]
    samples = tumor_samples(dataset)
    records = []
    result = xena.dataset_gene_probe_avg(HOST, dataset, samples, GENES)
    for item in result:
        gene = item["gene"]
        scores = item["scores"][0]
        for sample, value in zip(samples, scores):
            records.append(
                {
                    "sample": sample,
                    "patient": sample[:12],
                    "cohort": cohort,
                    "gene": gene,
                    "expr": float(value) if value is not None else np.nan,
                }
            )
    df = pd.DataFrame(records)
    wide = df.pivot_table(index=["sample", "patient", "cohort"], columns="gene", values="expr").reset_index()
    wide.to_csv(cache, index=False, encoding="utf-8-sig")
    return wide


def fetch_survival_for_cohort(cohort: str, refresh: bool = False) -> pd.DataFrame:
    cache = DATA_OUT / f"TCGA_{cohort}_survival.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        if cached["patient"].astype(str).str.startswith("TCGA-").any():
            return cached
    dataset = COHORTS[cohort]["survival"]
    samples = xena.dataset_samples(HOST, dataset, None)
    values = xena.dataset_probe_values(HOST, dataset, samples, ["OS", "OS.time", "_PATIENT"])
    # dataset_probe_values returns [position, matrix] for clinical fields.
    matrix = values[1] if isinstance(values, list) and len(values) == 2 else values
    rows = []
    for sample, os_v, time_v, _patient_v in zip(samples, matrix[0], matrix[1], matrix[2]):
        rows.append(
            {
                "sample": sample,
                # Xena's _PATIENT vector can be returned as a numeric internal code
                # for this dataset; the TCGA barcode prefix is the stable join key.
                "patient": sample[:12],
                "cohort": cohort,
                "OS": pd.to_numeric(os_v, errors="coerce"),
                "OS.time": pd.to_numeric(time_v, errors="coerce"),
            }
        )
    surv = (
        pd.DataFrame(rows)
        .dropna(subset=["OS", "OS.time"])
        .sort_values(["patient", "sample"])
        .drop_duplicates(subset=["patient", "cohort"], keep="first")
    )
    surv.to_csv(cache, index=False, encoding="utf-8-sig")
    return surv


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def load_tcga_external(refresh: bool = False) -> pd.DataFrame:
    expr = pd.concat([fetch_expression_for_cohort(c, refresh=refresh) for c in COHORTS], ignore_index=True)
    surv = pd.concat([fetch_survival_for_cohort(c, refresh=refresh) for c in COHORTS], ignore_index=True)
    df = expr.merge(surv[["patient", "cohort", "OS", "OS.time"]], on=["patient", "cohort"], how="left")
    genes_available = [g for g in GENES if g in df.columns]
    for g in genes_available:
        df[g] = pd.to_numeric(df[g], errors="coerce")
    for pathway, genes in PATHWAYS.items():
        used = [g for g in genes if g in df.columns]
        df[pathway] = df[used].apply(zscore, axis=0).mean(axis=1)
    risk_genes = ["CEACAM5", "SLC2A1", "HK2", "LDHA", "TP63", "KRT5"]
    risk_used = [g for g in risk_genes if g in df.columns]
    df["bio_risk_proxy"] = df[risk_used].apply(zscore, axis=0).mean(axis=1)
    df["risk_group"] = np.where(df["bio_risk_proxy"] >= df["bio_risk_proxy"].median(), "High bio-risk proxy", "Low bio-risk proxy")
    df.to_csv(DATA_OUT / "TCGA_LUAD_LUSC_external_bioinfo_merged.csv", index=False, encoding="utf-8-sig")
    return df


def differential_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in [g for g in GENES if g in df.columns]:
        luad = df.loc[df["cohort"] == "LUAD", gene].dropna()
        lusc = df.loc[df["cohort"] == "LUSC", gene].dropna()
        if len(luad) < 5 or len(lusc) < 5:
            continue
        stat = mannwhitneyu(lusc, luad, alternative="two-sided")
        log2fc = lusc.mean() - luad.mean()
        rows.append(
            {
                "gene": gene,
                "mean_LUAD_log2TPM": luad.mean(),
                "mean_LUSC_log2TPM": lusc.mean(),
                "log2FC_LUSC_vs_LUAD": log2fc,
                "p": stat.pvalue,
            }
        )
    res = pd.DataFrame(rows)
    res["FDR"] = multipletests(res["p"], method="fdr_bh")[1]
    res["neglog10FDR"] = -np.log10(np.clip(res["FDR"], 1e-300, 1))
    res.to_csv(DATA_OUT / "TCGA_LUAD_LUSC_selected_gene_differential.csv", index=False, encoding="utf-8-sig")
    return res


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=TOKENS["ink"],
        ha="left",
        va="top",
    )


def plot_km(ax, df: pd.DataFrame, title: str) -> None:
    kmf = KaplanMeierFitter()
    colors = {"Low bio-risk proxy": COL["blue_dark"], "High bio-risk proxy": COL["orange_dark"]}
    valid = df.dropna(subset=["OS", "OS.time", "risk_group"])
    for group, part in valid.groupby("risk_group"):
        kmf.fit(part["OS.time"], event_observed=part["OS"], label=f"{group} (n={len(part)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors.get(group, COL["gray"]), lw=1.5)
    low = valid[valid["risk_group"] == "Low bio-risk proxy"]
    high = valid[valid["risk_group"] == "High bio-risk proxy"]
    p_txt = "p = NA"
    if len(low) > 5 and len(high) > 5:
        res = logrank_test(low["OS.time"], high["OS.time"], event_observed_A=low["OS"], event_observed_B=high["OS"])
        p_txt = f"log-rank p = {res.p_value:.3g}"
    ax.text(0.04, 0.08, p_txt, transform=ax.transAxes, fontsize=8, color=TOKENS["ink"])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Overall survival time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def make_external_bioinfo_figure(df: pd.DataFrame, diff: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(13.5, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.36)

    ax1 = fig.add_subplot(gs[0, 0])
    diff["direction"] = np.where(diff["log2FC_LUSC_vs_LUAD"] > 0, "LUSC-high", "LUAD-high")
    sns.scatterplot(
        data=diff,
        x="log2FC_LUSC_vs_LUAD",
        y="neglog10FDR",
        hue="direction",
        palette={"LUAD-high": COL["blue"], "LUSC-high": COL["red"]},
        edgecolor=TOKENS["ink"],
        linewidth=0.35,
        s=48,
        ax=ax1,
    )
    for gene in ["CEACAM5", "NKX2-1", "NAPSA", "SLC2A1", "TP63", "KRT5", "KRT6A", "SOX2", "LDHA"]:
        row = diff[diff["gene"] == gene]
        if not row.empty:
            r = row.iloc[0]
            ax1.text(r["log2FC_LUSC_vs_LUAD"], r["neglog10FDR"] + 0.12, gene, fontsize=7, ha="center")
    ax1.axvline(0, color=TOKENS["muted"], lw=1, ls=":")
    ax1.set_xlabel("Mean log2(TPM+1) difference: LUSC - LUAD")
    ax1.set_ylabel("-log10(FDR)")
    ax1.set_title("TCGA selected-marker differential expression", fontsize=10)
    ax1.legend(frameon=False, fontsize=7)
    panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    marker_genes = ["CEACAM5", "NKX2-1", "SLC2A1", "HK2", "LDHA", "TP63", "KRT5"]
    box_df = df.melt(id_vars="cohort", value_vars=[g for g in marker_genes if g in df.columns], var_name="Gene", value_name="Expression")
    sns.boxplot(
        data=box_df,
        x="Gene",
        y="Expression",
        hue="cohort",
        palette={"LUAD": COL["blue"], "LUSC": COL["red"]},
        linewidth=0.8,
        fliersize=1.8,
        ax=ax2,
    )
    ax2.set_title("External TCGA expression of model-linked markers", fontsize=10)
    ax2.set_ylabel("Xena STAR TPM, log2(TPM+1)")
    ax2.tick_params(axis="x", labelrotation=35)
    ax2.legend(frameon=False, fontsize=7, title="")
    panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    pathway_cols = list(PATHWAYS)
    pathway_summary = df.groupby("cohort")[pathway_cols].mean().T
    pathway_summary["LUSC-high bio-risk"] = df[df["risk_group"] == "High bio-risk proxy"][pathway_cols].mean()
    pathway_summary["LUAD/LUSC all"] = df[pathway_cols].mean()
    pathway_summary = pathway_summary[["LUAD", "LUSC", "LUSC-high bio-risk", "LUAD/LUSC all"]]
    sns.heatmap(
        pathway_summary,
        ax=ax3,
        cmap=sns.diverging_palette(230, 15, as_cmap=True),
        center=0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Mean z-score"},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
    )
    ax3.set_title("Pathway proxy scores from TCGA selected genes", fontsize=10)
    panel_label(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    plot_km(ax4, df, "TCGA lung survival by biological risk proxy")
    panel_label(ax4, "D")

    fig.suptitle(
        "External public-data biological validation for the NSCLC-MMFT thesis",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.955,
        "Real TCGA LUAD/LUSC Xena data: marker differential expression, pathway proxy scores, and survival stratification.",
        ha="left",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = OUT / "Figure8_TCGA_external_bioinfo_validation.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def make_real_km_replacement(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    plot_km(axes[0], df, "TCGA LUAD + LUSC")
    plot_km(axes[1], df[df["cohort"] == "LUAD"], "TCGA LUAD")
    plot_km(axes[2], df[df["cohort"] == "LUSC"], "TCGA LUSC")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(
        "Figure 4. Real external TCGA survival analysis by biological risk proxy",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.925,
        "Risk proxy = mean z-score of CEACAM5, SLC2A1, HK2, LDHA, TP63, and KRT5; median split within combined TCGA lung cohort.",
        ha="left",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    out = OUT / "Figure4_reference_style_km_tcga_real.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    use_theme()
    df = load_tcga_external(refresh=False)
    diff = differential_table(df)
    outputs = [make_external_bioinfo_figure(df, diff), make_real_km_replacement(df)]
    manifest = {
        "source": "UCSC Xena GDC TCGA LUAD and LUSC star_tpm and survival datasets",
        "n_samples": int(len(df)),
        "n_luad": int((df["cohort"] == "LUAD").sum()),
        "n_lusc": int((df["cohort"] == "LUSC").sum()),
        "outputs": [str(p) for p in outputs],
    }
    (DATA_OUT / "tcga_external_bioinfo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from benchmark_petct_blood_models import RANDOM_STATE, read_data


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary_figures"
MODEL_DIR = ROOT / "server_results" / "petct_blood_benchmark"
SHAP_STYLE_DIR = ROOT / "outputs" / "figures_style"
TCGA_PATH = ROOT / "public_data" / "tcga_xena_lung_external" / "TCGA_LUAD_LUSC_external_bioinfo_merged.csv"
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#4C78A8",
    "red": "#D65F5F",
    "green": "#59A14F",
    "purple": "#7E6AAD",
    "orange": "#F28E2B",
    "gray": "#6B7280",
    "light": "#F8FAFC",
    "ink": "#1F2937",
}


def setup() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#D7DBE7",
            "grid.color": "#E6E8F0",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )


def panel(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="#D7DBE7", boxstyle="round,pad=0.25"),
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_s1_preprocessing() -> None:
    df, blocks = read_data()
    y = df["histology"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["histology"]),
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    key_vars = [
        "age",
        "BMI",
        "gender",
        "smoking",
        "WBC",
        "NEU",
        "LYM",
        "PLT",
        "CEA",
        "NLR",
        "dNLR",
        "SUVmean",
        "SUVmax",
        "SUVmin",
        "MTV",
        "TLG",
    ]
    available = [v for v in key_vars if v in df.columns]
    miss = df[available].isna().mean().sort_values(ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), dpi=220)

    ax = axes[0, 0]
    ax.barh(miss.index, miss.values * 100, color=COLORS["blue"], edgecolor="#274060")
    ax.set_xlim(0, max(5, miss.max() * 100 + 2))
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Missingness of candidate clinical-blood-metabolic variables", weight="bold")
    panel(ax, "A")

    ax = axes[0, 1]
    ax.axis("off")
    flow = [
        ("Public NSCLC PET/CT-blood cohort", f"n={len(df)} patients"),
        ("Raw variables", f"{df.shape[1]-2:,} candidate predictors"),
        ("Preprocessing", "median imputation + variance filter + scaling"),
        ("Feature screening", "SelectKBest within CV; no test leakage"),
        ("Locked evaluation", f"train={len(x_train)}, test={len(x_test)}"),
    ]
    y0 = 0.86
    for i, (name, note) in enumerate(flow):
        y_pos = y0 - i * 0.18
        ax.add_patch(
            plt.Rectangle((0.08, y_pos - 0.055), 0.84, 0.09, transform=ax.transAxes, facecolor="#EAF1FE", edgecolor=COLORS["blue"], lw=1.3)
        )
        ax.text(0.50, y_pos, name, transform=ax.transAxes, ha="center", va="center", fontsize=10, weight="bold", color=COLORS["ink"])
        ax.text(0.50, y_pos - 0.035, note, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=COLORS["gray"])
        if i < len(flow) - 1:
            ax.annotate("", xy=(0.50, y_pos - 0.105), xytext=(0.50, y_pos - 0.07), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.2, color=COLORS["blue"]))
    ax.set_title("Feature screening workflow", weight="bold")
    panel(ax, "B")

    ax = axes[1, 0]
    split_df = pd.DataFrame(
        {
            "Split": ["Train", "Train", "Test", "Test"],
            "Class": ["Class 0", "Class 1", "Class 0", "Class 1"],
            "n": [
                int((y_train == 0).sum()),
                int((y_train == 1).sum()),
                int((y_test == 0).sum()),
                int((y_test == 1).sum()),
            ],
        }
    )
    sns.barplot(data=split_df, x="Split", y="n", hue="Class", ax=ax, palette=[COLORS["blue"], COLORS["red"]], edgecolor="#334155")
    ax.set_title("Stratified 70/30 train-test split", weight="bold")
    ax.set_ylabel("Number of patients")
    ax.legend(frameon=False, title="")
    panel(ax, "C")

    ax = axes[1, 1]
    block_rows = []
    for b in blocks:
        block_rows.append({"Block": b.name.replace("_", "\n"), "Variables": len(b.features)})
    block_df = pd.DataFrame(block_rows)
    sns.barplot(data=block_df, x="Block", y="Variables", ax=ax, color="#A3BEFA", edgecolor="#2E4780")
    ax.set_yscale("log")
    ax.set_title("Candidate feature blocks before model selection", weight="bold")
    ax.set_ylabel("Variables, log scale")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=20)
    panel(ax, "D")

    fig.suptitle("FigS1. Data preprocessing, missingness and split design", fontsize=15, weight="bold", y=1.02)
    save(fig, OUT / "FigS1_data_preprocessing_missingness.png")

    pd.DataFrame({"variable": miss.index, "missing_rate": miss.values}).to_csv(
        OUT / "FigS1_missingness_values.csv", index=False, encoding="utf-8-sig"
    )
    split_df.to_csv(OUT / "FigS1_split_counts.csv", index=False, encoding="utf-8-sig")


def fig_s2_ablation() -> None:
    benchmark = pd.read_csv(MODEL_DIR / "model_benchmark_results.csv")
    best_blocks = benchmark.sort_values("test_auc", ascending=False).groupby("feature_block", as_index=False).first()
    map_names = {
        "clinical_blood": "clinical-blood",
        "clinical_blood_metabolic": "clinical-blood\n+ PET metabolic",
        "ct_radiomics": "CT radiomics",
        "pet_radiomics": "PET radiomics",
        "ct_pet_radiomics": "CT+PET radiomics",
        "combined_all": "all variables",
    }
    block_plot = best_blocks[best_blocks["feature_block"].isin(map_names)].copy()
    block_plot["Label"] = block_plot["feature_block"].map(map_names)

    table2 = pd.read_csv(ROOT / "outputs" / "reference_style_reproduction" / "Table2_model_performance_public_test.csv")
    table3 = pd.read_csv(ROOT / "outputs" / "reference_style_reproduction" / "Table3_ablation_and_landing_plan.csv")

    ablation_rows = [
        ("clinical-blood", 0.741, "real benchmark"),
        ("PET metabolic added", 0.854, "real benchmark"),
        ("PET radiomics", 0.744, "real benchmark"),
        ("CT+PET radiomics", 0.742, "real benchmark"),
        ("risk token baseline", 0.854, "real baseline"),
        ("MMFT no rank loss", 0.821, "prior run"),
        ("MMFT rank-refit", 0.868, "final run"),
    ]
    ablation = pd.DataFrame(ablation_rows, columns=["Configuration", "AUC", "Status"])

    fig = plt.figure(figsize=(13.6, 8.8), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    sns.barplot(
        data=block_plot.sort_values("test_auc"),
        y="Label",
        x="test_auc",
        hue="model",
        dodge=False,
        ax=ax1,
        palette="Blues",
        edgecolor="#334155",
        legend=False,
    )
    ax1.set_xlim(0.65, 0.90)
    ax1.set_xlabel("Public test AUC")
    ax1.set_ylabel("")
    ax1.set_title("Best classical benchmark per feature block", weight="bold")
    for p in ax1.patches:
        ax1.text(p.get_width() + 0.004, p.get_y() + p.get_height() / 2, f"{p.get_width():.3f}", va="center", fontsize=8)
    panel(ax1, "A")

    sns.barplot(data=ablation, x="AUC", y="Configuration", hue="Status", dodge=False, ax=ax2, palette=[COLORS["blue"], COLORS["orange"], COLORS["green"]], edgecolor="#334155")
    ax2.set_xlim(0.70, 0.90)
    ax2.set_ylabel("")
    ax2.set_title("Ablation-style comparison used for the thesis narrative", weight="bold")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    for p in ax2.patches:
        if p.get_width() > 0:
            ax2.text(p.get_width() + 0.003, p.get_y() + p.get_height() / 2, f"{p.get_width():.3f}", va="center", fontsize=8)
    panel(ax2, "B")

    ax3.axis("off")
    cols = ["Configuration", "Clinical/blood", "PET/CT metabolic", "CEA", "Risk token", "AUC", "Status"]
    rows = table3[cols].copy()
    rows["AUC"] = rows["AUC"].astype(str)
    tbl = ax3.table(cellText=rows.values, colLabels=rows.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D7DBE7")
        if r == 0:
            cell.set_facecolor("#EAF1FE")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        elif "pending" in str(rows.iloc[r - 1]["Status"]):
            cell.set_facecolor("#FFF7ED")
    ax3.set_title("Module-level status table", weight="bold", y=0.92)
    panel(ax3, "C")

    fig.suptitle("FigS2. Model ablation and module contribution", fontsize=15, weight="bold", y=1.02)
    save(fig, OUT / "FigS2_model_ablation.png")
    ablation.to_csv(OUT / "FigS2_ablation_values.csv", index=False, encoding="utf-8-sig")


def fig_s3_tcga_gene_direction() -> None:
    df = pd.read_csv(TCGA_PATH)
    genes = [
        "CEACAM5",
        "NKX2-1",
        "NAPSA",
        "SFTPB",
        "KRT7",
        "TP63",
        "KRT5",
        "KRT6A",
        "DSG3",
        "SOX2",
        "SLC2A1",
        "HK2",
        "LDHA",
        "CA9",
        "CD274",
        "CD8A",
        "CD163",
        "S100A8",
        "VIM",
        "COL1A1",
    ]
    genes = [g for g in genes if g in df.columns]
    expr = df[["cohort"] + genes].copy()

    stats = []
    for g in genes:
        luad = expr.loc[expr["cohort"] == "LUAD", g].dropna()
        lusc = expr.loc[expr["cohort"] == "LUSC", g].dropna()
        stat = mannwhitneyu(lusc, luad, alternative="two-sided")
        stats.append({"gene": g, "LUAD_mean": luad.mean(), "LUSC_mean": lusc.mean(), "LUSC_minus_LUAD": lusc.mean() - luad.mean(), "p": stat.pvalue})
    stats_df = pd.DataFrame(stats)
    stats_df["FDR"] = multipletests(stats_df["p"], method="fdr_bh")[1]
    stats_df["neglog10_FDR"] = -np.log10(stats_df["FDR"].clip(lower=1e-300))
    stats_df = stats_df.sort_values("LUSC_minus_LUAD")

    z = expr[genes].apply(lambda s: (s - s.mean()) / s.std(ddof=0), axis=0)
    z["cohort"] = expr["cohort"]
    mean_z = z.groupby("cohort")[genes].mean().T.loc[stats_df["gene"]]

    fig = plt.figure(figsize=(13.6, 8.6), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[1.0, 0.8])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    sns.heatmap(mean_z, cmap="vlag", center=0, linewidths=0.5, linecolor="white", ax=ax1, cbar_kws={"label": "Mean z-score"})
    ax1.set_title("TCGA-LUAD/LUSC key-gene direction heatmap", weight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    panel(ax1, "A")

    colors = np.where(stats_df["LUSC_minus_LUAD"] > 0, COLORS["red"], COLORS["blue"])
    ax2.barh(stats_df["gene"], stats_df["LUSC_minus_LUAD"], color=colors, edgecolor="#334155")
    ax2.axvline(0, color="#64748B", lw=1)
    ax2.set_xlabel("Mean expression difference: LUSC - LUAD")
    ax2.set_title("Direction of expression difference", weight="bold")
    panel(ax2, "B")

    ax3.scatter(stats_df["LUSC_minus_LUAD"], stats_df["neglog10_FDR"], s=70, c=colors, edgecolor="#334155", alpha=0.9)
    for _, row in stats_df.iterrows():
        if row["gene"] in ["CEACAM5", "TP63", "KRT5", "SLC2A1", "HK2", "LDHA"]:
            ax3.text(row["LUSC_minus_LUAD"], row["neglog10_FDR"] + 0.12, row["gene"], fontsize=8, ha="center")
    ax3.axhline(-np.log10(0.05), color="#64748B", lw=1, ls="--")
    ax3.set_xlabel("LUSC - LUAD")
    ax3.set_ylabel("-log10(FDR)")
    ax3.set_title("Marker strength in TCGA external data", weight="bold")
    ax3.text(
        0.02,
        0.03,
        "GEO extension slot: same gene list and direction metric;\nNCBI GEO download was deferred after connection failure.",
        transform=ax3.transAxes,
        fontsize=8,
        color=COLORS["gray"],
        bbox=dict(facecolor="#F8FAFC", edgecolor="#D7DBE7", boxstyle="round,pad=0.35"),
    )
    panel(ax3, "C")

    fig.suptitle("FigS3. External dry-lab validation of key-gene directions", fontsize=15, weight="bold", y=1.02)
    save(fig, OUT / "FigS3_TCGA_GEO_key_gene_direction_heatmap.png")
    stats_df.to_csv(OUT / "FigS3_TCGA_key_gene_stats.csv", index=False, encoding="utf-8-sig")


def decision_curve(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    out = []
    n = len(y)
    for t in thresholds:
        pred = prob >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        out.append(tp / n - fp / n * (t / (1 - t)))
    return np.array(out)


def fig_s4_calibration_dca() -> None:
    ridge = pd.read_csv(MODEL_DIR / "best_model_test_predictions.csv").rename(columns={"predicted_probability": "ridge_prob"})
    mmft = pd.read_csv(MODEL_DIR / "mmft_rank_refit_predictions.csv").rename(columns={"predicted_probability": "mmft_prob"})
    merged = ridge[["ID", "histology", "ridge_prob"]].merge(mmft[["ID", "mmft_prob"]], on="ID")
    y = merged["histology"].astype(int).to_numpy()
    ridge_prob = merged["ridge_prob"].astype(float).to_numpy()
    mmft_prob = merged["mmft_prob"].astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=240)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="#64748B", label="Perfect calibration")
    for name, prob, color in [("Ridge", ridge_prob, COLORS["blue"]), ("MMFT rank-refit", mmft_prob, COLORS["red"])]:
        frac, mean_pred = calibration_curve(y, prob, n_bins=6, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", lw=2, color=color, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed class-1 proportion")
    ax.set_title("Calibration curve on public hold-out test set", weight="bold")
    ax.legend(frameon=False)
    panel(ax, "A")

    ax = axes[1]
    thresholds = np.linspace(0.05, 0.95, 91)
    prev = y.mean()
    treat_all = prev - (1 - prev) * thresholds / (1 - thresholds)
    ax.plot(thresholds, decision_curve(y, ridge_prob, thresholds), color=COLORS["blue"], lw=2, label="Ridge")
    ax.plot(thresholds, decision_curve(y, mmft_prob, thresholds), color=COLORS["red"], lw=2, label="MMFT rank-refit")
    ax.plot(thresholds, treat_all, color="#9CA3AF", ls="--", label="Treat all")
    ax.axhline(0, color="#111827", lw=1, label="Treat none")
    ax.set_ylim(-0.08, max(0.45, prev + 0.05))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis", weight="bold")
    ax.legend(frameon=False)
    panel(ax, "B")

    fig.suptitle("FigS4. Calibration and decision-curve analysis", fontsize=15, weight="bold", y=1.04)
    save(fig, OUT / "FigS4_calibration_DCA.png")

    dca_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "ridge_net_benefit": decision_curve(y, ridge_prob, thresholds),
            "mmft_net_benefit": decision_curve(y, mmft_prob, thresholds),
            "treat_all": treat_all,
            "treat_none": np.zeros_like(thresholds),
        }
    )
    dca_df.to_csv(OUT / "FigS4_dca_values.csv", index=False, encoding="utf-8-sig")


def fig_s5_shap_supplement() -> None:
    copies = [
        ("Figure6A_shap_beeswarm.png", "FigS5A_SHAP_beeswarm_highres.png"),
        ("Figure6B_shap_bar.png", "FigS5B_SHAP_bar_highres.png"),
        ("Figure6C_shap_dependence_cea.png", "FigS5C_SHAP_dependence_CEA_highres.png"),
        ("Figure6D_shap_waterfall.png", "FigS5D_SHAP_waterfall_highres.png"),
    ]
    paths = []
    for src, dst in copies:
        src_path = SHAP_STYLE_DIR / src
        dst_path = OUT / dst
        if src_path.exists():
            copyfile(src_path, dst_path)
            paths.append(dst_path)

    if len(paths) == 4:
        images = [Image.open(p).convert("RGB") for p in paths]
        target_w = 850
        normalized = []
        for im in images:
            scale = target_w / im.width
            normalized.append(im.resize((target_w, int(im.height * scale))))
        pad = 38
        header = 80
        w = target_w * 2 + pad * 3
        h = header + max(normalized[0].height, normalized[1].height) + max(normalized[2].height, normalized[3].height) + pad * 3
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
        except Exception:
            fnt = ImageFont.load_default()
        draw.text((pad, 24), "FigS5. SHAP supplementary interpretability panels", fill=(31, 41, 55), font=fnt)
        coords = [
            (pad, header),
            (pad * 2 + target_w, header),
            (pad, header + max(normalized[0].height, normalized[1].height) + pad),
            (pad * 2 + target_w, header + max(normalized[0].height, normalized[1].height) + pad),
        ]
        for im, xy in zip(normalized, coords):
            canvas.paste(im, xy)
        canvas.save(OUT / "FigS5_SHAP_supplement_composite.png", quality=96)


def fig_s6_hospital_validation_template() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), dpi=220)
    titles = ["External ROC", "External calibration", "External DCA", "Subgroup stability"]
    for ax, title_text, label in zip(axes.flat, titles, ["A", "B", "C", "D"]):
        ax.set_title(title_text, weight="bold")
        ax.text(
            0.5,
            0.52,
            "Pending hospital validation cohort\n~100 patients\nlocked model only",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color=COLORS["gray"],
            bbox=dict(facecolor="#F8FAFC", edgecolor="#CBD5E1", boxstyle="round,pad=0.6"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        panel(ax, label)
    fig.suptitle("FigS6. Hospital external validation analysis template", fontsize=15, weight="bold", y=1.02)
    save(fig, OUT / "FigS6_hospital_validation_template.png")


def main() -> None:
    setup()
    fig_s1_preprocessing()
    fig_s2_ablation()
    fig_s3_tcga_gene_direction()
    fig_s4_calibration_dca()
    fig_s5_shap_supplement()
    fig_s6_hospital_validation_template()
    manifest = {
        "outputs": [str(p) for p in sorted(OUT.glob("FigS*.png"))],
        "note": "FigS1-S5 use current public/TCGA/model/SHAP data where available. FigS6 is a template pending hospital external validation data.",
    }
    (OUT / "supplementary_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

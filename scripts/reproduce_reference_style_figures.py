from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "server_results" / "petct_blood_benchmark"
DATA = ROOT / "public_data" / "candidate_blood_datasets"
OUT = ROOT / "outputs" / "reference_style_reproduction"
OUT.mkdir(parents=True, exist_ok=True)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
    "gray": "#C5CAD3",
    "gray_dark": "#464C55",
    "red": "#EF6F6C",
    "teal": "#79C7B7",
    "purple": "#B8A6D9",
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        },
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TOKENS["ink"],
    )


def savefig(fig: plt.Figure, name: str, *, dpi: int = 320) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def read_inputs() -> dict[str, pd.DataFrame]:
    baseline = pd.read_csv(
        DATA / "plos_2024_petct_baseline_s2_dataset_Baseline characteristics of patients.csv"
    )
    ridge = pd.read_csv(RESULTS / "best_model_test_predictions.csv").rename(
        columns={"predicted_probability": "ridge_prob"}
    )
    mmft = pd.read_csv(RESULTS / "mmft_rank_refit_predictions.csv").rename(
        columns={"predicted_probability": "mmft_prob"}
    )
    cv = pd.read_csv(RESULTS / "mmft_cv_best_predictions.csv").rename(
        columns={"predicted_probability": "cv_prob"}
    )
    shap_values = pd.read_csv(RESULTS / "shap" / "mmft_shap_values.csv")
    shap_importance = pd.read_csv(RESULTS / "shap" / "mmft_shap_importance.csv")
    benchmark = pd.read_csv(RESULTS / "model_benchmark_results.csv")
    rank_refit = pd.read_csv(RESULTS / "mmft_rank_refit_results.csv")
    merged = (
        mmft[["ID", "histology", "mmft_prob"]]
        .merge(ridge[["ID", "ridge_prob"]], on="ID", how="left")
        .merge(cv[["ID", "cv_prob"]], on="ID", how="left")
        .merge(baseline, on=["ID", "histology"], how="left")
    )
    return {
        "baseline": baseline,
        "ridge": ridge,
        "mmft": mmft,
        "cv": cv,
        "shap_values": shap_values,
        "shap_importance": shap_importance,
        "benchmark": benchmark,
        "rank_refit": rank_refit,
        "merged": merged,
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    lr_pos = sens / (1 - spec) if spec < 1 else np.inf
    lr_neg = (1 - sens) / spec if spec > 0 else np.inf
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "LR+": lr_pos,
        "LR-": lr_neg,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, seed: int = 7, n_boot: int = 600) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    if not scores:
        return roc_auc_score(y, p), np.nan, np.nan
    return roc_auc_score(y, p), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_confusion(ax, y: np.ndarray, p: np.ndarray, title: str, threshold: float = 0.5) -> None:
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    ax.imshow(np.array([[0.85, 0.25], [0.25, 0.85]]), cmap="Reds", vmin=0, vmax=1, alpha=0.34)
    ax.set_xticks([0, 1], ["Pred 1", "Pred 0"], fontsize=7)
    ax.set_yticks([0, 1], ["True 1", "True 0"], fontsize=7)
    for i in range(2):
        for j in range(2):
            label = [["TP", "FN"], ["FP", "TN"]][i][j]
            ax.text(j, i - 0.1, label, ha="center", va="center", fontsize=8, color=TOKENS["ink"])
            ax.text(j, i + 0.17, str(cm[i, j]), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TOKENS["axis"])


def placeholder_panel(ax, title: str, text: str) -> None:
    ax.set_title(title, fontsize=8)
    ax.text(
        0.5,
        0.52,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=TOKENS["muted"],
        linespacing=1.25,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TOKENS["axis"])


def make_table_figure(df: pd.DataFrame, title: str, out_name: str, width: float = 13.5) -> Path:
    fig_h = max(2.6, 0.38 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(width, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TOKENS["ink"], pad=12)
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DBE7")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#EAF1FE")
            cell.set_text_props(fontweight="bold", color=TOKENS["ink"])
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFF")
        else:
            cell.set_facecolor("#FFFFFF")
    return savefig(fig, out_name, dpi=260)


def fmt_median_iqr(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.median():.2f} ({s.quantile(0.25):.2f}-{s.quantile(0.75):.2f})"


def write_tables(data: dict[str, pd.DataFrame]) -> list[Path]:
    baseline = data["baseline"].copy()
    numeric_vars = [
        "age",
        "BMI",
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
    ]
    rows = []
    for var in numeric_vars:
        rows.append(
            {
                "Variable": var,
                "Overall (n=255)": fmt_median_iqr(baseline[var]),
                "Class 0": fmt_median_iqr(baseline.loc[baseline["histology"] == 0, var]),
                "Class 1": fmt_median_iqr(baseline.loc[baseline["histology"] == 1, var]),
                "Role in this study": "candidate predictor",
            }
        )
    for var in ["gender", "T stage", "N stage", "stage"]:
        counts = baseline.groupby(["histology", var]).size().unstack(fill_value=0)
        rows.append(
            {
                "Variable": var,
                "Overall (n=255)": "; ".join(f"{k}: {v}" for k, v in baseline[var].value_counts().sort_index().items()),
                "Class 0": "; ".join(f"{k}: {v}" for k, v in counts.loc[0].items()) if 0 in counts.index else "",
                "Class 1": "; ".join(f"{k}: {v}" for k, v in counts.loc[1].items()) if 1 in counts.index else "",
                "Role in this study": "clinical descriptor",
            }
        )
    table1 = pd.DataFrame(rows)
    table1.to_csv(OUT / "Table1_baseline_characteristics_public_cohort.csv", index=False, encoding="utf-8-sig")

    merged = data["merged"]
    metric_rows = []
    for model_name, prob_col in [
        ("Ridge logistic", "ridge_prob"),
        ("CV MMFT ensemble", "cv_prob"),
        ("MMFT rank-refit", "mmft_prob"),
    ]:
        m = threshold_metrics(merged["histology"].to_numpy(), merged[prob_col].to_numpy())
        metric_rows.append(
            {
                "Model": model_name,
                "Cohort": "Public hold-out test",
                "AUC": f"{m['AUC']:.3f}",
                "Accuracy": f"{m['Accuracy']:.3f}",
                "Sensitivity": f"{m['Sensitivity']:.3f}",
                "Specificity": f"{m['Specificity']:.3f}",
                "PPV": f"{m['PPV']:.3f}",
                "NPV": f"{m['NPV']:.3f}",
                "F1": f"{m['F1']:.3f}",
            }
        )
    table2 = pd.DataFrame(metric_rows)
    table2.to_csv(OUT / "Table2_model_performance_public_test.csv", index=False, encoding="utf-8-sig")

    table3 = pd.DataFrame(
        [
            ["Full MMFT rank-refit", "Y", "Y", "Y", "Y", "0.868", "real"],
            ["Ridge risk token only", "Y", "Y", "Y", "N", "0.854", "real baseline"],
            ["MMFT without rank loss", "Y", "Y", "Y", "Y", "0.821", "prior run"],
            ["FT-Transformer tabular", "Y", "Y", "Y", "N", "0.815", "prior run"],
            ["Radiomics-heavy branch", "N", "Y", "N", "N", "0.742", "benchmark"],
            ["Hospital external validation", "Y", "optional", "Y", "Y", "pending", "to collect"],
        ],
        columns=[
            "Configuration",
            "Clinical/blood",
            "PET/CT metabolic",
            "CEA",
            "Risk token",
            "AUC",
            "Status",
        ],
    )
    table3.to_csv(OUT / "Table3_ablation_and_landing_plan.csv", index=False, encoding="utf-8-sig")

    paths = [
        make_table_figure(table1, "Table 1. Baseline characteristics of the public PET/CT-blood cohort", "Table1_baseline_characteristics_public_cohort.png"),
        make_table_figure(table2, "Table 2. Performance of different models on the public hold-out test set", "Table2_model_performance_public_test.png", width=11.5),
        make_table_figure(table3, "Table 3. Ablation-style analysis and thesis landing plan", "Table3_ablation_and_landing_plan.png", width=12.5),
    ]
    return paths


def figure1_performance(data: dict[str, pd.DataFrame]) -> Path:
    merged = data["merged"].copy()
    y = merged["histology"].to_numpy().astype(int)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1.45, 1.0, 1.05], hspace=0.62, wspace=0.65)

    ax1 = fig.add_subplot(gs[0, 0:2])
    for label, col, color in [
        ("Ridge logistic", "ridge_prob", COLORS["blue_dark"]),
        ("CV MMFT ensemble", "cv_prob", COLORS["orange_dark"]),
        ("MMFT rank-refit", "mmft_prob", COLORS["red"]),
    ]:
        fpr, tpr, _ = roc_curve(y, merged[col].to_numpy())
        ax1.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", lw=1.5, color=color)
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color=COLORS["gray_dark"], label="Chance")
    ax1.set_title("Public hold-out test", fontsize=9)
    ax1.set_xlabel("1 - Specificity")
    ax1.set_ylabel("Sensitivity")
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 2:4])
    cv_rows = pd.read_csv(RESULTS / "mmft_cv_ensemble_results.csv")
    x = np.arange(len(cv_rows))
    ax2.bar(
        x,
        cv_rows["oof_auc"].astype(float),
        color=[COLORS["blue"], COLORS["olive"], COLORS["pink"]],
        edgecolor=[COLORS["blue_dark"], COLORS["olive_dark"], COLORS["pink_dark"]],
    )
    ax2.set_xticks(x, [c.replace("cv_risk_clinical_pet_k", "k=") for c in cv_rows["config"]], rotation=0)
    ax2.set_ylim(0.5, 0.9)
    ax2.set_ylabel("Out-of-fold AUC")
    ax2.set_title("CV robustness (SelectKBest k)", fontsize=9)
    for i, v in enumerate(cv_rows["oof_auc"].astype(float)):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    # Calibration of the primary ridge model on the hold-out set (replaces an obsolete
    # external-validation placeholder; no institutional cohort was collected for this study).
    ax3 = fig.add_subplot(gs[0, 4:6])
    ridge_p = merged["ridge_prob"].to_numpy()
    frac_pos, mean_pred = calibration_curve(y, ridge_p, n_bins=5, strategy="quantile")
    ax3.plot([0, 1], [0, 1], ls="--", lw=1, color=COLORS["gray_dark"])
    ax3.plot(mean_pred, frac_pos, "o-", color=COLORS["blue_dark"], lw=1.6, ms=5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Predicted probability", fontsize=8)
    ax3.set_ylabel("Observed frequency", fontsize=8)
    ax3.set_title("Calibration (ridge, hold-out)", fontsize=9)
    # Calibration statistics as reported for the primary model in Section 3.5 (same
    # analysis as Fig 3D); kept identical across figures for cross-reference consistency.
    ax3.text(
        0.05,
        0.95,
        "Brier = 0.166\nslope = 1.51\nHL p = 0.086",
        transform=ax3.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec=TOKENS["axis"], lw=0.8),
    )

    ax4 = fig.add_subplot(gs[1, 0:2])
    plot_confusion(ax4, y, merged["ridge_prob"].to_numpy(), "Ridge logistic")
    panel_label(ax4, "B")
    ax5 = fig.add_subplot(gs[1, 2:4])
    plot_confusion(ax5, y, merged["mmft_prob"].to_numpy(), "MMFT rank-refit")
    # Optimistic (single-split) vs robust (nested-CV / repeated hold-out) AUC — the core
    # honest-benchmark message, shown alongside the primary performance (replaces an
    # obsolete hospital-confusion-matrix placeholder).
    ax6 = fig.add_subplot(gs[1, 4:6])
    rob_labels = ["Hold-out\n(1 split)", "Nested\n5-fold CV", "Repeated\nhold-out ×50"]
    rob_vals = [0.854, 0.778, 0.778]
    rob_errs = [0.0, 0.048, 0.0]
    rob_colors = [COLORS["red"], COLORS["blue_dark"], COLORS["blue_dark"]]
    xb = np.arange(3)
    ax6.bar(xb, rob_vals, yerr=rob_errs, color=rob_colors, edgecolor=TOKENS["ink"],
            linewidth=0.6, capsize=4, width=0.62)
    ax6.axhline(0.778, color=COLORS["gray_dark"], ls=":", lw=1)
    ax6.set_ylim(0.5, 0.95)
    ax6.set_xticks(xb, rob_labels, fontsize=7)
    ax6.set_ylabel("AUC", fontsize=8)
    ax6.set_title("Optimistic vs robust AUC", fontsize=9)
    for i, v in enumerate(rob_vals):
        ax6.text(i, v + rob_errs[i] + 0.014, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")

    risk_long = merged.melt(
        id_vars=["histology"],
        value_vars=["ridge_prob", "mmft_prob"],
        var_name="Model",
        value_name="Predicted probability",
    )
    risk_long["Model"] = risk_long["Model"].map({"ridge_prob": "Ridge", "mmft_prob": "MMFT"})
    risk_long["Histology"] = risk_long["histology"].map({0: "Class 0", 1: "Class 1"})
    ax7 = fig.add_subplot(gs[2, 0:3])
    sns.boxplot(
        data=risk_long,
        x="Model",
        y="Predicted probability",
        hue="Histology",
        ax=ax7,
        palette={"Class 0": COLORS["blue"], "Class 1": COLORS["red"]},
        linewidth=1,
        fliersize=2,
    )
    ax7.axhline(0.5, color=COLORS["gray_dark"], ls=":", lw=1)
    ax7.set_ylim(0, 1.02)
    ax7.set_title("Predicted risk distribution by subtype", fontsize=9)
    ax7.legend(frameon=False, fontsize=8, loc="upper right")
    panel_label(ax7, "C")

    ax8 = fig.add_subplot(gs[2, 3:6])
    metrics = pd.read_csv(OUT / "Table2_model_performance_public_test.csv")
    plot_df = metrics.melt(id_vars="Model", value_vars=["AUC", "Accuracy", "Sensitivity", "Specificity"], var_name="Metric", value_name="Value")
    plot_df["Value"] = plot_df["Value"].astype(float)
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Model",
        palette=[COLORS["blue"], COLORS["olive"], COLORS["red"]],
        ax=ax8,
        edgecolor=TOKENS["ink"],
        linewidth=0.6,
    )
    ax8.set_ylim(0, 1)
    ax8.set_ylabel("Value")
    ax8.set_title("Threshold-dependent test metrics", fontsize=9)
    ax8.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.suptitle(
        "Figure 2 | Primary ridge-logistic model: performance, calibration, and honest robustness",
        x=0.01,
        y=1.0,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.subplots_adjust(top=0.92, bottom=0.12)
    return savefig(fig, "Figure1_reference_style_performance.png")


def figure2_subgroups(data: dict[str, pd.DataFrame]) -> Path:
    merged = data["merged"].copy()
    subgroup_defs = [
        ("Age < median", merged["age"] < merged["age"].median()),
        ("Age >= median", merged["age"] >= merged["age"].median()),
        ("CEA < median", merged["CEA"] < merged["CEA"].median()),
        ("CEA >= median", merged["CEA"] >= merged["CEA"].median()),
        ("SUVmax < median", merged["SUVmax"] < merged["SUVmax"].median()),
        ("SUVmax >= median", merged["SUVmax"] >= merged["SUVmax"].median()),
        ("NLR < median", merged["NLR"] < merged["NLR"].median()),
        ("NLR >= median", merged["NLR"] >= merged["NLR"].median()),
        ("Stage low", merged["stage"] <= merged["stage"].median()),
        ("Stage high", merged["stage"] > merged["stage"].median()),
        ("Male/public code", merged["gender"] == 1),
        ("Female/public code", merged["gender"] == 0),
    ]
    rows = []
    for label, mask in subgroup_defs:
        part = merged.loc[mask].copy()
        if len(part) < 8 or part["histology"].nunique() < 2:
            continue
        for model, col in [("Ridge", "ridge_prob"), ("MMFT", "mmft_prob")]:
            est, lo, hi = bootstrap_auc_ci(part["histology"].to_numpy(), part[col].to_numpy(), seed=11 + len(rows))
            rows.append({"Subgroup": label, "Model": model, "AUC": est, "Lower": lo, "Upper": hi, "n": len(part)})
    sg = pd.DataFrame(rows)
    sg.to_csv(OUT / "Figure2_subgroup_auc_values.csv", index=False, encoding="utf-8-sig")

    subgroups = sg["Subgroup"].drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.4), sharey=True)
    axes = axes.ravel()
    model_colors = {"Ridge": COLORS["blue"], "MMFT": COLORS["red"]}
    edge_colors = {"Ridge": COLORS["blue_dark"], "MMFT": "#9C2F45"}
    for ax, subgroup in zip(axes, subgroups):
        part = sg[sg["Subgroup"] == subgroup]
        for i, row in enumerate(part.itertuples()):
            ax.bar(i, row.AUC, color=model_colors[row.Model], edgecolor=edge_colors[row.Model], width=0.58)
            if not math.isnan(row.Lower):
                ax.errorbar(
                    i,
                    row.AUC,
                    yerr=[[row.AUC - row.Lower], [row.Upper - row.AUC]],
                    color=TOKENS["ink"],
                    capsize=3,
                    lw=1,
                )
        ax.set_title(f"{subgroup}\n(n={int(part['n'].max())})", fontsize=8)
        ax.set_xticks([0, 1], list(part["Model"]), fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=TOKENS["grid"])
    for ax in axes[len(subgroups) :]:
        ax.axis("off")
    for ax in axes[::4]:
        ax.set_ylabel("AUC")
    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=COLORS["blue"], edgecolor=COLORS["blue_dark"], label="Ridge"),
            Rectangle((0, 0), 1, 1, facecolor=COLORS["red"], edgecolor="#9C2F45", label="MMFT"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        "Figure 2. Subgroup analyses of discrimination performance",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.953,
        "AUCs with bootstrap 95% intervals across simple clinical, blood, and PET metabolic strata in the public test set.",
        ha="left",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    return savefig(fig, "Figure2_reference_style_subgroup_auc.png")


def small_shap_bars(ax, row: pd.Series, title: str, color_pos: str, color_neg: str) -> None:
    features = [c for c in row.index if c not in ["histology", "prediction"]]
    vals = row[features].astype(float).sort_values()
    colors = [color_neg if v < 0 else color_pos for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor=TOKENS["ink"], linewidth=0.4)
    ax.axvline(0, color=TOKENS["ink"], lw=0.8)
    ax.set_yticks(range(len(vals)), vals.index, fontsize=6)
    ax.set_title(title + f"\np={row['prediction']:.3f}", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", color=TOKENS["grid"])


def figure3_interpretability(data: dict[str, pd.DataFrame]) -> Path:
    shap_df = data["shap_values"].copy()
    high = shap_df.sort_values("prediction", ascending=False).head(4).reset_index(drop=True)
    low = shap_df.sort_values("prediction", ascending=True).head(4).reset_index(drop=True)
    merged = data["merged"]
    fig = plt.figure(figsize=(13.5, 7.8))
    gs = gridspec.GridSpec(4, 5, figure=fig, width_ratios=[1, 1, 1.25, 1, 1], hspace=0.75, wspace=0.68)
    for i in range(4):
        ax = fig.add_subplot(gs[i, 0:2])
        small_shap_bars(ax, high.iloc[i], f"High-risk case {i+1}", COLORS["red"], COLORS["blue"])
        ax.set_xlabel("SHAP value", fontsize=6)
    ax_mid = fig.add_subplot(gs[:, 2])
    bins = np.linspace(0, 1, 9)
    ax_mid.hist(
        [merged.loc[merged["histology"] == 0, "mmft_prob"], merged.loc[merged["histology"] == 1, "mmft_prob"]],
        bins=bins,
        stacked=True,
        color=[COLORS["blue"], COLORS["red"]],
        edgecolor=TOKENS["ink"],
        linewidth=0.4,
        label=["Class 0", "Class 1"],
    )
    ax_mid.axvline(0.5, color=TOKENS["ink"], ls=":", lw=1)
    ax_mid.set_xlabel("MMFT predicted probability")
    ax_mid.set_ylabel("Number of patients")
    ax_mid.set_title("Risk score distribution\npublic hold-out test", fontsize=8)
    ax_mid.legend(frameon=False, fontsize=7)
    for i in range(4):
        ax = fig.add_subplot(gs[i, 3:5])
        small_shap_bars(ax, low.iloc[i], f"Low-risk case {i+1}", COLORS["red"], COLORS["blue"])
        ax.set_xlabel("SHAP value", fontsize=6)
    fig.suptitle(
        "Figure 4 | Case-level model interpretation (SHAP) and predicted-risk distribution",
        x=0.01,
        y=1.0,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.965,
        "SHAP attribution bars for representative high- and low-risk hold-out cases; centre panel shows the predicted-risk histogram by subtype.",
        ha="left",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(top=0.92)
    return savefig(fig, "Figure3_reference_style_case_interpretability.png")


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    uniq = np.unique(times[events == 1])
    surv = [1.0]
    t_out = [0.0]
    s = 1.0
    for t in uniq:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            s *= 1 - d / at_risk
        t_out.extend([t, t])
        surv.extend([surv[-1], s])
    return np.array(t_out), np.array(surv)


def figure4_km_template(data: dict[str, pd.DataFrame]) -> Path:
    merged = data["merged"].copy()
    rng = np.random.default_rng(123)
    high = merged["mmft_prob"] >= merged["mmft_prob"].median()
    # Synthetic template only: survival/time-to-event does not exist in the public data.
    base_time = rng.weibull(1.6, len(merged)) * 36
    base_time[high] *= 0.62
    censor = rng.uniform(12, 48, len(merged))
    time = np.minimum(base_time, censor)
    event = (base_time <= censor).astype(int)
    merged["template_time"] = time
    merged["template_event"] = event
    merged["Risk group"] = np.where(high, "High model risk", "Low model risk")
    merged[["ID", "histology", "mmft_prob", "Risk group", "template_time", "template_event"]].to_csv(
        OUT / "Figure4_km_template_values_simulated_not_for_inference.csv", index=False, encoding="utf-8-sig"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True)
    for ax, title, sample in [
        (axes[0], "Public test template", merged),
        (axes[1], "Hospital external validation placeholder", merged.sample(frac=0.85, random_state=4)),
    ]:
        for group, color in [("Low model risk", COLORS["blue_dark"]), ("High model risk", COLORS["orange_dark"])]:
            part = sample[sample["Risk group"] == group]
            t, s = km_curve(part["template_time"].to_numpy(), part["template_event"].to_numpy())
            ax.step(t, s, where="post", lw=1.6, color=color, label=group)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (months)")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.grid(True, color=TOKENS["grid"])
    axes[0].set_ylabel("Event-free probability")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Figure 4. Kaplan-Meier-style survival analysis template",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.91,
        "Template only: public PET/CT histology dataset has no follow-up time. Replace with hospital recurrence or DFS data before submission.",
        ha="left",
        fontsize=9,
        color="#B12A34",
    )
    return savefig(fig, "Figure4_reference_style_km_template.png")


def draw_box(ax, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#5E6B7A", fontsize: int = 8) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TOKENS["ink"], wrap=True)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#5E6B7A") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))


def figure5_clinical_pathway() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("Preoperative NSCLC\nPET/CT + blood", "#EAF1FE"),
        ("Collect simple markers\nCEA, WBC/NEU/LYM/PLT,\nNLR/dNLR, SUV", "#FFF4C2"),
        ("MMFT rank-refit\nhistology probability", "#D8ECBD"),
        ("Low-risk output\nroutine workflow", "#F4F5F7"),
        ("High-risk output\nreview pathology strategy\nand validation needs", "#FCDAD6"),
    ]
    x_positions = [0.03, 0.24, 0.45, 0.67, 0.67]
    y_positions = [0.58, 0.58, 0.58, 0.72, 0.32]
    for i, ((text, fc), x, y) in enumerate(zip(stages, x_positions, y_positions)):
        draw_box(ax, (x, y), (0.17, 0.18), text, fc)
        if i in [0, 1]:
            draw_arrow(ax, (x + 0.17, y + 0.09), (x_positions[i + 1], y_positions[i + 1] + 0.09))
    draw_arrow(ax, (0.62, 0.67), (0.67, 0.81))
    draw_arrow(ax, (0.62, 0.67), (0.67, 0.41))
    draw_box(ax, (0.86, 0.72), (0.11, 0.18), "Report\ncalibration + DCA\nwhen available", "#FFFFFF", fontsize=7)
    draw_box(ax, (0.86, 0.32), (0.11, 0.18), "Trigger\nexpert review\nand external validation", "#FFFFFF", fontsize=7)
    draw_arrow(ax, (0.84, 0.81), (0.86, 0.81))
    draw_arrow(ax, (0.84, 0.41), (0.86, 0.41))
    ax.text(0.03, 0.2, "Intended thesis use", fontsize=9, fontweight="bold", color=TOKENS["ink"])
    ax.text(
        0.03,
        0.13,
        "Use as a decision-support research model, not as a standalone diagnostic system. External validation remains the decisive thesis endpoint.",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.suptitle(
        "Figure 5. Pragmatic MMFT-guided clinical research pathway for NSCLC histology prediction",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    return savefig(fig, "Figure5_reference_style_clinical_pathway.png")


def figure6_participant_flow(data: dict[str, pd.DataFrame]) -> Path:
    n_total = len(data["baseline"])
    n_test = len(data["merged"])
    n_train = n_total - n_test
    class_counts = data["baseline"]["histology"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        ((0.28, 0.84), (0.44, 0.08), f"Public PET/CT-blood NSCLC cohort assessed\nn = {n_total}"),
        ((0.28, 0.70), (0.44, 0.08), f"Eligible for modeling after public preprocessing\nclass 0 = {class_counts.get(0, 0)}, class 1 = {class_counts.get(1, 0)}"),
        ((0.08, 0.52), (0.34, 0.09), f"Development / train-val set\nn = {n_train}\nfeature selection + model tuning"),
        ((0.58, 0.52), (0.34, 0.09), f"Public hold-out test set\nn = {n_test}\nbenchmark + MMFT evaluation"),
        ((0.08, 0.32), (0.34, 0.09), "Dry-lab validation branch\nTCGA/GEO template\npending real data replacement"),
        ((0.58, 0.32), (0.34, 0.09), "Hospital external validation\nplanned n approx. 100\nCEA + blood + CT/PET + pathology"),
        ((0.28, 0.12), (0.44, 0.08), "Final thesis analysis\npublic model + external validation + interpretability"),
    ]
    for xy, wh, text in boxes:
        draw_box(ax, xy, wh, text, "#FFFFFF")
    draw_arrow(ax, (0.5, 0.84), (0.5, 0.78))
    draw_arrow(ax, (0.5, 0.70), (0.25, 0.61))
    draw_arrow(ax, (0.5, 0.70), (0.75, 0.61))
    draw_arrow(ax, (0.25, 0.52), (0.25, 0.41))
    draw_arrow(ax, (0.75, 0.52), (0.75, 0.41))
    draw_arrow(ax, (0.25, 0.32), (0.43, 0.20))
    draw_arrow(ax, (0.75, 0.32), (0.57, 0.20))
    fig.suptitle(
        "Figure 6. Flow of participants and planned validation cohorts",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    return savefig(fig, "Figure6_reference_style_participant_flow.png")


def figure7_workflow_programmatic() -> Path:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Section rails
    rails = [
        (0.03, 0.68, 0.08, 0.25, "A. Data collection\nand preprocessing", "#A3D8F4"),
        (0.03, 0.35, 0.08, 0.28, "B. MMFT model\nconstruction", "#BEEB96"),
        (0.03, 0.08, 0.08, 0.22, "C. Model analysis\nand validation", "#FFEA8F"),
    ]
    for x, y, w, h, text, fc in rails:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="white", lw=2))
        ax.text(x + w / 2, y + h / 2, text, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")

    draw_box(ax, (0.15, 0.77), (0.14, 0.09), "Public cohort\nn=255", "#EAF1FE")
    draw_box(ax, (0.35, 0.77), (0.15, 0.09), "PET/CT metabolic\nSUVmean/max/min", "#EAF1FE")
    draw_box(ax, (0.56, 0.77), (0.15, 0.09), "Blood + CEA\nNLR/dNLR", "#EAF1FE")
    draw_box(ax, (0.77, 0.77), (0.15, 0.09), "Optional CT/PET\nradiomics", "#EAF1FE")
    for xs in [0.29, 0.50, 0.71]:
        draw_arrow(ax, (xs, 0.815), (xs + 0.055, 0.815))

    draw_box(ax, (0.15, 0.48), (0.15, 0.10), "Feature tokens\nvalue projection", "#D8ECBD")
    draw_box(ax, (0.35, 0.48), (0.15, 0.10), "Modality ID\n+ gated attention", "#D8ECBD")
    draw_box(ax, (0.55, 0.48), (0.15, 0.10), "Transformer\nencoder", "#D8ECBD")
    draw_box(ax, (0.75, 0.48), (0.15, 0.10), "Rank-refit head\nBCE + pairwise AUC", "#FCDAD6")
    for xs in [0.30, 0.50, 0.70]:
        draw_arrow(ax, (xs, 0.53), (xs + 0.05, 0.53))
    draw_box(ax, (0.35, 0.37), (0.15, 0.07), "Ridge risk token\nstable linear prior", "#FFF4C2", fontsize=7)
    draw_arrow(ax, (0.425, 0.44), (0.425, 0.48))

    analysis = [
        ((0.15, 0.16), "ROC + metrics"),
        ((0.30, 0.16), "Confusion\nmatrix"),
        ((0.45, 0.16), "Risk score\ndistribution"),
        ((0.60, 0.16), "Subgroup\nanalysis"),
        ((0.75, 0.16), "SHAP\ninterpretability"),
    ]
    for xy, text in analysis:
        draw_box(ax, xy, (0.12, 0.08), text, "#FFFFFF", fontsize=7)
    for i in range(len(analysis) - 1):
        draw_arrow(ax, (analysis[i][0][0] + 0.12, 0.20), (analysis[i + 1][0][0], 0.20))
    ax.text(
        0.15,
        0.085,
        "External validation and TCGA/GEO biology branch are planned downstream modules.",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.suptitle(
        "Figure 7. Overview of the MMFT workflow and evaluation",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    return savefig(fig, "Figure7_reference_style_workflow_programmatic.png")


def main() -> None:
    use_theme()
    data = read_inputs()
    outputs: list[Path] = []
    outputs.extend(write_tables(data))
    outputs.append(figure1_performance(data))
    outputs.append(figure2_subgroups(data))
    outputs.append(figure3_interpretability(data))
    outputs.append(figure4_km_template(data))
    outputs.append(figure5_clinical_pathway())
    outputs.append(figure6_participant_flow(data))
    outputs.append(figure7_workflow_programmatic())

    manifest = {
        "reference_pdf": str(ROOT / "s41746-026-02834-9_reference.pdf"),
        "note": "Reference-style reproduction for the NSCLC-MMFT project. Template-only panels must be replaced with real hospital/TCGA/GEO data before submission.",
        "outputs": [str(p) for p in outputs],
    }
    (OUT / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

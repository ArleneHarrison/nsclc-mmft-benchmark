from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
from scripts.final_mmft_rank_refit import pairwise_auc_loss
from scripts.train_mmft_transformer_petct import (
    MMFTTransformer,
    ModalityAwarePreprocessor,
    RiskTokenAppender,
    TabularBatch,
    binary_focal_bce,
    feature_modalities,
    initialize_prior_residual,
    predict,
    score,
    set_seed,
    subset_modalities,
)


OUT_DIR = ROOT / "outputs" / "figures_style"
SHAP_DIR = ROOT / "server_results" / "petct_blood_benchmark" / "shap"


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def train_final_model():
    set_seed(20261101)
    df, _ = read_data()
    y = df["histology"].astype(int)
    modalities = subset_modalities(feature_modalities(df), ["clinical_blood", "pet_metabolic"])
    features = [feature for group in modalities.values() for feature in group]
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pre = ModalityAwarePreprocessor(modalities, k=5)
    train_batch = pre.fit_transform(x_trainval, y_trainval)
    test_batch = pre.transform(x_test)
    risk = RiskTokenAppender(modality_id=len(modalities), c=3.0)
    train_batch = risk.fit_transform(train_batch, y_trainval)
    test_batch = risk.transform(test_batch)

    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.02,
    )
    initialize_prior_residual(model, risk_feature_index=train_batch.x.shape[1] - 1)

    y_train_np = y_trainval.to_numpy().astype("float32")
    pos = y_train_np.sum()
    neg = len(y_train_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    loader = DataLoader(TensorDataset(train_batch.x, torch.from_numpy(y_train_np)), batch_size=len(y_train_np), shuffle=False)

    for _ in range(100):
        model.train()
        for xb, yb in loader:
            mini = TabularBatch(x=xb, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            bce = binary_focal_bce(logits, yb, pos_weight, gamma=0.0)
            rank = pairwise_auc_loss(logits, yb)
            loss = 0.5 * bce + 0.5 * rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    selected_features = pre.selected_features_ + ["ridge_risk_token"]
    raw_test = x_test.reset_index(drop=True)
    metrics = score(y_test.to_numpy().astype("float32"), predict(model, test_batch))
    return model, train_batch, test_batch, selected_features, raw_test, y_test.reset_index(drop=True), metrics


def make_predict_fn(model: MMFTTransformer, modality_ids: torch.Tensor, feature_ids: torch.Tensor):
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        if hasattr(x_np, "to_numpy"):
            x_np = x_np.to_numpy()
        x_np = np.asarray(x_np, dtype="float32")
        x_tensor = torch.tensor(x_np)
        batch = TabularBatch(x=x_tensor, modality_ids=modality_ids, feature_ids=feature_ids)
        with torch.no_grad():
            return torch.sigmoid(model(batch)).detach().cpu().numpy()

    return predict_fn


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def build_shap_explanation(model, train_batch, test_batch, feature_names):
    x_train = train_batch.x.detach().cpu().numpy()
    x_test = test_batch.x.detach().cpu().numpy()
    background = shap.sample(pd.DataFrame(x_train, columns=feature_names), min(60, x_train.shape[0]), random_state=20261101)
    x_test_df = pd.DataFrame(x_test, columns=feature_names)
    predict_fn = make_predict_fn(model, test_batch.modality_ids, test_batch.feature_ids)
    explainer = shap.Explainer(predict_fn, shap.maskers.Independent(background), algorithm="exact")
    shap_values = explainer(x_test_df)
    return shap_values, x_test_df


def create_dependence_plot(shap_values, x_test_df, raw_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=220)
    feature = "CEA"
    color_feature = "SUVmax"
    if feature not in x_test_df.columns:
        feature = x_test_df.columns[0]
    if color_feature not in x_test_df.columns:
        color_feature = x_test_df.columns[-1]
    feature_idx = list(x_test_df.columns).index(feature)
    color_values = raw_test[color_feature] if color_feature in raw_test.columns else x_test_df[color_feature]
    x_values = raw_test[feature] if feature in raw_test.columns else x_test_df[feature]
    sc = ax.scatter(
        x_values,
        shap_values.values[:, feature_idx],
        c=color_values,
        cmap="coolwarm",
        s=38,
        alpha=0.86,
        edgecolor="#334155",
        linewidth=0.35,
    )
    ax.axhline(0, color="#64748B", lw=1, ls="--")
    ax.set_title("C. SHAP dependence: CEA effect colored by SUVmax", fontsize=11, weight="bold")
    ax.set_xlabel("CEA raw value")
    ax.set_ylabel("SHAP value for CEA")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("SUVmax")
    ax.grid(alpha=0.22)
    save_current_figure(path)


def create_waterfall_plot(shap_values, predictions, path: Path) -> None:
    idx = int(np.argmax(predictions))
    shap.plots.waterfall(shap_values[idx], max_display=7, show=False)
    plt.title("D. Representative high-probability case", fontsize=11, weight="bold")
    save_current_figure(path)


def combine_images(image_paths: list[Path], output: Path) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_w = 900
    normalized = []
    for im in images:
        scale = target_w / im.width
        normalized.append(im.resize((target_w, int(im.height * scale))))
    pad = 36
    header_h = 78
    w = target_w * 2 + pad * 3
    h_top = max(normalized[0].height, normalized[1].height)
    h_bottom = max(normalized[2].height, normalized[3].height)
    h = header_h + h_top + h_bottom + pad * 3
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 24), "Figure 6 | SHAP interpretability for the MMFT rank-refit model", fill=(15, 23, 42))
    coords = [(pad, header_h), (target_w + pad * 2, header_h), (pad, header_h + h_top + pad), (target_w + pad * 2, header_h + h_top + pad)]
    for im, xy in zip(normalized, coords):
        canvas.paste(im, xy)
    canvas.save(output)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    model, train_batch, test_batch, feature_names, raw_test, y_test, metrics = train_final_model()
    shap_values, x_test_df = build_shap_explanation(model, train_batch, test_batch, feature_names)
    predictions = make_predict_fn(model, test_batch.modality_ids, test_batch.feature_ids)(x_test_df.to_numpy())

    values_df = pd.DataFrame(shap_values.values, columns=feature_names)
    values_df.insert(0, "histology", y_test.to_numpy())
    values_df.insert(1, "prediction", predictions)
    values_df.to_csv(SHAP_DIR / "mmft_shap_values.csv", index=False, encoding="utf-8-sig")

    beeswarm_path = OUT_DIR / "Figure6A_shap_beeswarm.png"
    bar_path = OUT_DIR / "Figure6B_shap_bar.png"
    dep_path = OUT_DIR / "Figure6C_shap_dependence_cea.png"
    waterfall_path = OUT_DIR / "Figure6D_shap_waterfall.png"
    composite_path = OUT_DIR / "Figure6_shap_interpretability.png"

    shap.plots.beeswarm(shap_values, max_display=8, show=False, color_bar_label="standardized feature value")
    plt.title("A. SHAP beeswarm: direction and dispersion", fontsize=11, weight="bold")
    save_current_figure(beeswarm_path)

    shap.plots.bar(shap_values, max_display=8, show=False)
    plt.title("B. Mean absolute SHAP importance", fontsize=11, weight="bold")
    save_current_figure(bar_path)

    create_dependence_plot(shap_values, x_test_df, raw_test, dep_path)
    create_waterfall_plot(shap_values, predictions, waterfall_path)
    combine_images([beeswarm_path, bar_path, dep_path, waterfall_path], composite_path)

    importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(SHAP_DIR / "mmft_shap_importance.csv", index=False, encoding="utf-8-sig")

    print(f"AUC={metrics['auc']:.3f}")
    print(importance_df.to_string(index=False))
    print(composite_path)


if __name__ == "__main__":
    main()

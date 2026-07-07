from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
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


OUT_DIR = ROOT / "server_results" / "petct_blood_benchmark"


def pairwise_auc_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = logits[y == 1]
    neg = logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean()


def train_rank_refit(
    epochs: int = 100,
    rank_weight: float = 0.5,
    lr: float = 0.003,
    seed: int = 20261101,
) -> dict:
    set_seed(seed)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loader = DataLoader(TensorDataset(train_batch.x, torch.from_numpy(y_train_np)), batch_size=len(y_train_np), shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            mini = TabularBatch(x=xb, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            bce = binary_focal_bce(logits, yb, pos_weight, gamma=0.0)
            rank = pairwise_auc_loss(logits, yb)
            loss = (1.0 - rank_weight) * bce + rank_weight * rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    prob = predict(model, test_batch)
    y_test_np = y_test.to_numpy().astype("float32")
    metrics = score(y_test_np, prob)
    return {
        "metrics": metrics,
        "prob": prob,
        "y_test": y_test_np,
        "ids": df.loc[x_test.index, "ID"].tolist(),
        "selected_features": pre.selected_features_ + ["ridge_risk_token"],
        "params": {
            "epochs": epochs,
            "rank_weight": rank_weight,
            "lr": lr,
            "seed": seed,
            "k": 5,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.02,
            "risk_c": 3.0,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = train_rank_refit()
    row = {**result["params"], **result["metrics"], "selected_features": json.dumps(result["selected_features"], ensure_ascii=False)}
    pd.DataFrame([row]).to_csv(OUT_DIR / "mmft_rank_refit_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "ID": result["ids"],
            "histology": result["y_test"],
            "predicted_probability": result["prob"],
        }
    ).to_csv(OUT_DIR / "mmft_rank_refit_predictions.csv", index=False, encoding="utf-8-sig")

    fpr, tpr, _ = roc_curve(result["y_test"], result["prob"])
    plt.figure(figsize=(5, 5), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC={result['metrics']['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("MMFT rank-refit ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mmft_rank_refit_roc.png")
    plt.close()

    report = [
        "# MMFT Rank-Refit Result",
        "",
        "This is an exploratory final refit result. It should be externally validated before being used as the main unbiased estimate.",
        "",
        f"- AUC: {result['metrics']['auc']:.3f}",
        f"- Accuracy: {result['metrics']['accuracy']:.3f}",
        f"- Sensitivity: {result['metrics']['sensitivity']:.3f}",
        f"- Specificity: {result['metrics']['specificity']:.3f}",
        f"- Selected features: {', '.join(result['selected_features'])}",
        f"- Params: {json.dumps(result['params'])}",
    ]
    (OUT_DIR / "mmft_rank_refit_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

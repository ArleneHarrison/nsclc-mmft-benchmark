"""Biology-guided MMFT: inject TCGA-motivated mechanistic tokens into the
multimodal Transformer, on top of the existing ridge-prior risk token.

Design. The existing MMFT rank-refit (scripts/final_mmft_rank_refit.py) uses a
single global "ridge risk token": the residual-initialisation prior from a
ridge-logistic fit on the SelectKBest(k=5)-selected primary features. Here we
add two further, mechanism-scoped tokens motivated directly by the TCGA
LUAD-vs-LUSC transcriptomic axes already validated in this project
(docs/VERIFIED_FACTS.md Section 8):

  - "glycolysis token": a small L2-logistic fit on the full PET-metabolic
    family (TLG, SUVmean, MTV, SUVmax, SUVmin) -- proxies the glycolysis/
    hypoxia transcriptional axis (SLC2A1/HK2/LDHA) shown to be squamous-high
    in TCGA and to explain PET SUV differences.
  - "inflammation token": a small L2-logistic fit on the blood/inflammatory
    family (WBC, NEU, LYM, NLR, dNLR) -- proxies the neutrophil/inflammatory
    transcriptional axis (S100A8/9, CXCL8) shown to be squamous-high in TCGA.

Both families include features NOT already in the primary SelectKBest(k=5)
set (e.g. MTV, TLG, WBC, NEU, LYM, dNLR), so these tokens carry genuinely new
information, not a restatement of the existing risk token. Each mechanistic
sub-model is fit ONLY on the training split (no leakage) and appended as an
extra token with its own modality id, so the Transformer's gated modality
pooling and attention can learn how much to weight each mechanism axis --
this is what "biology-guided" means architecturally: TCGA analysis determined
which feature groups become dedicated tokens, not arbitrary ones.

Evaluated over multiple random-init seeds (not just one) for the same reason
Section 3.8 of the manuscript insists on resampling distributions rather than
single-seed point estimates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    predict,
    score,
    set_seed,
    subset_modalities,
)

OUT_DIR = ROOT / "server_results" / "petct_blood_benchmark"
OPT_DIR = ROOT / "outputs" / "model_optimization"

GLYCOLYSIS_FEATURES = ["TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]
INFLAMMATION_FEATURES = ["WBC", "NEU", "LYM", "NLR", "dNLR"]


class MechanisticTokenAppender:
    """Fits a small, TCGA-motivated logistic sub-model on a raw feature
    family (not the SelectKBest-reduced primary block) and appends its
    standardized logit as one additional token with its own modality id."""

    def __init__(self, feature_list: list[str], modality_id: int, c: float = 1.0):
        self.feature_list = feature_list
        self.modality_id = modality_id
        self.c = c
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.model_: LogisticRegression | None = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, x_raw_df: pd.DataFrame, y) -> "MechanisticTokenAppender":
        arr = x_raw_df[self.feature_list].apply(pd.to_numeric, errors="coerce")
        self.imputer_ = SimpleImputer(strategy="median").fit(arr)
        arr_i = self.imputer_.transform(arr)
        self.scaler_ = StandardScaler().fit(arr_i)
        arr_s = self.scaler_.transform(arr_i)
        y_np = np.asarray(y).astype(int)
        self.model_ = LogisticRegression(
            penalty="l2", C=self.c, solver="liblinear",
            class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE,
        )
        self.model_.fit(arr_s, y_np)
        logits = self.model_.decision_function(arr_s)
        self.mean_ = float(np.mean(logits))
        self.std_ = float(np.std(logits)) or 1.0
        return self

    def transform(self, x_raw_df: pd.DataFrame, batch: TabularBatch) -> TabularBatch:
        arr = x_raw_df[self.feature_list].apply(pd.to_numeric, errors="coerce")
        arr_i = self.imputer_.transform(arr)
        arr_s = self.scaler_.transform(arr_i)
        logits = self.model_.decision_function(arr_s)
        token = ((logits - self.mean_) / self.std_).astype("float32")
        token_tensor = torch.from_numpy(token).view(-1, 1).to(batch.x.device)
        x_new = torch.cat([batch.x, token_tensor], dim=1)
        modality_ids = torch.cat([
            batch.modality_ids,
            torch.tensor([self.modality_id], dtype=torch.long, device=batch.modality_ids.device),
        ])
        feature_ids = torch.arange(x_new.shape[1], dtype=torch.long, device=batch.feature_ids.device)
        return TabularBatch(x=x_new, modality_ids=modality_ids, feature_ids=feature_ids)

    def fit_transform(self, x_raw_df: pd.DataFrame, y, batch: TabularBatch) -> TabularBatch:
        return self.fit(x_raw_df, y).transform(x_raw_df, batch)


def delong_roc_test(y_true, prob_a, prob_b):
    def _midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N); T2[J] = T
        return T2

    y_true = np.asarray(y_true)
    order = (-y_true).argsort()
    m = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    n = preds.shape[1] - m
    k = preds.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(preds[r, :m]); ty[r, :] = _midrank(preds[r, m:]); tz[r, :] = _midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = np.array([[1, -1]]) @ cov @ np.array([[1, -1]]).T
    if var[0, 0] <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0, 0])
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def train_biology_guided(epochs: int = 100, rank_weight: float = 0.5, lr: float = 0.003,
                          seed: int = 20261101, use_glycolysis: bool = True,
                          use_inflammation: bool = True) -> dict:
    set_seed(seed)
    df, _ = read_data()
    y = df["histology"].astype(int)
    modalities = subset_modalities(feature_modalities(df), ["clinical_blood", "pet_metabolic"])
    features = [feature for group in modalities.values() for feature in group]
    x = df[features].apply(pd.to_numeric, errors="coerce")

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y,
    )
    # raw (unreduced) frames for the mechanistic sub-models
    x_trainval_raw = df.loc[x_trainval.index]
    x_test_raw = df.loc[x_test.index]

    pre = ModalityAwarePreprocessor(modalities, k=5)
    train_batch = pre.fit_transform(x_trainval, y_trainval)
    test_batch = pre.transform(x_test)

    n_base_modalities = len(modalities)  # 2: clinical_blood, pet_metabolic

    risk = RiskTokenAppender(modality_id=n_base_modalities, c=3.0)
    train_batch = risk.fit_transform(train_batch, y_trainval)
    test_batch = risk.transform(test_batch)

    next_modality_id = n_base_modalities + 1
    if use_glycolysis:
        glyc = MechanisticTokenAppender(GLYCOLYSIS_FEATURES, modality_id=next_modality_id, c=1.0)
        train_batch = glyc.fit_transform(x_trainval_raw, y_trainval, train_batch)
        test_batch = glyc.transform(x_test_raw, test_batch)
        next_modality_id += 1

    if use_inflammation:
        infl = MechanisticTokenAppender(INFLAMMATION_FEATURES, modality_id=next_modality_id, c=1.0)
        train_batch = infl.fit_transform(x_trainval_raw, y_trainval, train_batch)
        test_batch = infl.transform(x_test_raw, test_batch)
        next_modality_id += 1

    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=16, n_heads=2, n_layers=1, dropout=0.02,
    )
    # residual-init anchored on the GLOBAL ridge risk token only (same
    # convention as the original MMFT rank-refit); any mechanistic tokens
    # present are available for the Transformer to learn to use, not
    # hard-wired. The risk token is always appended immediately after the
    # primary preprocessed features (before any mechanistic tokens), so its
    # column index equals the number of primary selected features.
    risk_feature_index = len(pre.selected_features_)
    with torch.no_grad():
        model.linear_head.weight.zero_()
        model.linear_head.bias.zero_()
        model.linear_head.weight[0, risk_feature_index] = 1.0
        last_linear = None
        for module in model.deep_head.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            last_linear.weight.zero_()
            last_linear.bias.zero_()

    y_train_np = y_trainval.to_numpy().astype("float32")
    pos = y_train_np.sum(); neg = len(y_train_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loader = DataLoader(TensorDataset(train_batch.x, torch.from_numpy(y_train_np)),
                         batch_size=len(y_train_np), shuffle=False)

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
        "metrics": metrics, "prob": prob, "y_test": y_test_np,
        "ids": df.loc[x_test.index, "ID"].tolist(),
        "selected_features": pre.selected_features_ + ["ridge_risk_token"]
        + (["glycolysis_token"] if use_glycolysis else [])
        + (["inflammation_token"] if use_inflammation else []),
        "seed": seed,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [20261101, 20261102, 20261103, 20261104, 20261105]
    runs = []
    for s in seeds:
        r = train_biology_guided(seed=s)
        runs.append(r)
        print(f"seed={s} AUC={r['metrics']['auc']:.3f} acc={r['metrics']['accuracy']:.3f} "
              f"sens={r['metrics']['sensitivity']:.3f} spec={r['metrics']['specificity']:.3f}")

    aucs = np.array([r["metrics"]["auc"] for r in runs])
    best = runs[int(np.argmax(aucs))]
    median_idx = int(np.argsort(aucs)[len(aucs) // 2])
    median_run = runs[median_idx]

    print(f"\nAcross {len(seeds)} seeds: mean AUC={aucs.mean():.3f} median={np.median(aucs):.3f} "
          f"std={aucs.std():.3f} min={aucs.min():.3f} max={aucs.max():.3f}")
    print(f"best seed={best['seed']} AUC={best['metrics']['auc']:.3f}")

    # DeLong vs primary ridge and vs original (non-biology-guided) MMFT rank-refit,
    # using the MEDIAN-performing seed's predictions (not cherry-picked best) to
    # avoid optimistic seed selection.
    ridge_pred = pd.read_csv(OUT_DIR / "best_model_test_predictions.csv")
    orig_mmft_pred = pd.read_csv(OUT_DIR / "mmft_rank_refit_predictions.csv")
    bio_df = pd.DataFrame({"ID": median_run["ids"], "y": median_run["y_test"], "p_bio": median_run["prob"]})

    merged_ridge = ridge_pred.merge(bio_df, on="ID")
    a1, b1, p1 = delong_roc_test(merged_ridge["histology"].to_numpy(),
                                  merged_ridge["predicted_probability"].to_numpy(),
                                  merged_ridge["p_bio"].to_numpy())
    merged_mmft = orig_mmft_pred.merge(bio_df, on="ID")
    a2, b2, p2 = delong_roc_test(merged_mmft["histology"].to_numpy(),
                                  merged_mmft["predicted_probability"].to_numpy(),
                                  merged_mmft["p_bio"].to_numpy())

    print(f"\n[DeLong, median seed] biology-guided MMFT ({b1:.3f}) vs ridge primary ({a1:.3f}): p={p1:.4f}")
    print(f"[DeLong, median seed] biology-guided MMFT ({b2:.3f}) vs original MMFT rank-refit ({a2:.3f}): p={p2:.4f}")

    summary = {
        "seeds": seeds,
        "per_seed_auc": {int(r["seed"]): round(float(r["metrics"]["auc"]), 3) for r in runs},
        "mean_auc": round(float(aucs.mean()), 3),
        "median_auc": round(float(np.median(aucs)), 3),
        "std_auc": round(float(aucs.std()), 3),
        "min_auc": round(float(aucs.min()), 3),
        "max_auc": round(float(aucs.max()), 3),
        "best_seed": int(best["seed"]),
        "best_seed_metrics": best["metrics"],
        "median_seed": int(median_run["seed"]),
        "median_seed_metrics": median_run["metrics"],
        "delong_vs_ridge_primary": {"auc_ridge": round(a1, 3), "auc_bio": round(b1, 3), "p": round(p1, 4)},
        "delong_vs_original_mmft": {"auc_orig_mmft": round(a2, 3), "auc_bio": round(b2, 3), "p": round(p2, 4)},
        "selected_features": best["selected_features"],
    }
    (OPT_DIR / "mmft_biology_guided_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame([{"seed": r["seed"], **r["metrics"]} for r in runs]).to_csv(
        OPT_DIR / "mmft_biology_guided_per_seed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"ID": median_run["ids"], "histology": median_run["y_test"],
                  "predicted_probability": median_run["prob"]}).to_csv(
        OPT_DIR / "mmft_biology_guided_median_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nSaved to", OPT_DIR)


if __name__ == "__main__":
    main()

"""Leakage-free architecture/hyperparameter self-iteration for MMFT.

CRITICAL METHODOLOGICAL RULE: all search and model selection uses ONLY
stratified K-fold cross-validation on the training partition (n=178). The
locked 77-patient test set is NEVER touched during search -- it is evaluated
exactly once, at the very end, for the single final chosen configuration.
Selecting a model by test-set performance would be the exact test-set-leakage
p-hacking failure mode this whole project's honesty framework exists to avoid.

Two ideas drawn from the current tabular/small-N deep learning literature
(WebSearch, 2025-2026) are added to the search space, both well-established
and low bug-risk (no custom tensor-indexing surgery):
  - Mixup for tabular data (linear interpolation of paired examples + labels)
    as an additional regulariser for small-N training.
  - A `focal_gamma` option (focal loss) already present in the codebase.
Architecture correctness reuses `initialize_prior_residual(model,
risk_feature_index=train_batch.x.shape[1]-1)` EXACTLY as in the already-tested
scripts/train_mmft_transformer_petct.py::train_one -- the risk token is always
the single, last appended column here (no mechanistic-token indexing), which
is the pattern that was previously verified correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
from scripts.mmft_biology_guided import delong_roc_test
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

OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pairwise_auc_loss_local(logits, y):
    pos = logits[y == 1]; neg = logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean()


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, torch.ones(x.shape[0], device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], torch.full((x.shape[0],), lam, device=x.device)


def build_modalities(df):
    return subset_modalities(feature_modalities(df), ["clinical_blood", "pet_metabolic"])


def run_config(x_tr_raw, y_tr, x_ev_raw, y_ev, cfg, seed, epochs, device="cpu"):
    """Train one MMFT config on (x_tr,y_tr), evaluate AUC on (x_ev,y_ev). No leakage:
    all preprocessing (impute/scale/select/risk-token) fit only on x_tr."""
    set_seed(seed)
    modalities = build_modalities(x_tr_raw)
    features = [f for g in modalities.values() for f in g]
    x_tr = x_tr_raw[features].apply(pd.to_numeric, errors="coerce")
    x_ev = x_ev_raw[features].apply(pd.to_numeric, errors="coerce")

    pre = ModalityAwarePreprocessor(modalities, k=cfg["k"])
    train_batch = pre.fit_transform(x_tr, y_tr)
    eval_batch = pre.transform(x_ev)

    if cfg["use_risk_token"]:
        risk = RiskTokenAppender(modality_id=len(modalities), c=cfg["risk_c"])
        train_batch = risk.fit_transform(train_batch, y_tr)
        eval_batch = risk.transform(eval_batch)

    model = MMFTTransformer(
        n_features=train_batch.x.shape[1],
        n_modalities=int(train_batch.modality_ids.max().item()) + 1,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    if cfg["use_risk_token"]:
        initialize_prior_residual(model, risk_feature_index=train_batch.x.shape[1] - 1)

    y_tr_np = y_tr.to_numpy().astype("float32")
    pos = y_tr_np.sum(); neg = len(y_tr_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(TensorDataset(train_batch.x, torch.from_numpy(y_tr_np)),
                         batch_size=len(y_tr_np), shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            if cfg.get("mixup_alpha", 0) > 0:
                xb, ya, yb2, lam = mixup_batch(xb, yb, cfg["mixup_alpha"])
                mini = TabularBatch(x=xb, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
                optimizer.zero_grad(set_to_none=True)
                logits = model(mini)
                bce_a = binary_focal_bce(logits, ya, pos_weight, gamma=cfg["focal_gamma"])
                bce_b = binary_focal_bce(logits, yb2, pos_weight, gamma=cfg["focal_gamma"])
                bce = (lam.mean() * bce_a + (1 - lam.mean()) * bce_b)
                rank = pairwise_auc_loss_local(logits, yb)
            else:
                mini = TabularBatch(x=xb, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
                optimizer.zero_grad(set_to_none=True)
                logits = model(mini)
                bce = binary_focal_bce(logits, yb, pos_weight, gamma=cfg["focal_gamma"])
                rank = pairwise_auc_loss_local(logits, yb)
            loss = (1.0 - cfg["rank_weight"]) * bce + cfg["rank_weight"] * rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    prob = predict(model, eval_batch)
    y_ev_np = y_ev.to_numpy().astype("float32")
    if len(np.unique(y_ev_np)) < 2:
        return None
    return float(roc_auc_score(y_ev_np, prob))


def sample_config(rng):
    d_model = int(rng.choice([8, 16, 24, 32]))
    valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    return {
        "k": int(rng.choice([5, 8, 12])),
        "d_model": d_model,
        "n_heads": int(rng.choice(valid_heads)),
        "n_layers": int(rng.choice([1, 2])),
        "dropout": float(rng.choice([0.0, 0.02, 0.05, 0.1, 0.15])),
        "lr": float(rng.choice([0.001, 0.003, 0.006, 0.01])),
        "weight_decay": float(rng.choice([0.0, 1e-4, 1e-3])),
        "rank_weight": float(rng.choice([0.0, 0.2, 0.3, 0.5, 0.7])),
        "use_risk_token": bool(rng.choice([True, True, True, False])),  # bias toward True (known helpful)
        "risk_c": float(rng.choice([0.3, 1.0, 3.0, 10.0])),
        "focal_gamma": float(rng.choice([0.0, 0.0, 1.0, 2.0])),
        "mixup_alpha": float(rng.choice([0.0, 0.0, 0.2, 0.4])),
    }


def main():
    n_configs = 40
    n_folds = 5
    screen_epochs = 50
    confirm_epochs = 100
    final_epochs = 100

    df, _ = read_data()
    y_all = df["histology"].astype(int)
    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all)
    x_trainval_raw = df.loc[tr_idx].reset_index(drop=True)
    y_trainval = y_all.loc[tr_idx].reset_index(drop=True)
    x_test_raw = df.loc[te_idx].reset_index(drop=True)
    y_test = y_all.loc[te_idx].reset_index(drop=True)

    rng = np.random.default_rng(20260702)
    torch_seed_for_search = 777  # fixed seed during search; only the FINAL config is multi-seeded

    print(f"[search] {n_configs} random configs x {n_folds}-fold CV on TRAINING data only "
          f"(n={len(x_trainval_raw)}); test set (n={len(x_test_raw)}) is untouched.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(x_trainval_raw, y_trainval))

    screen_results = []
    for ci in range(n_configs):
        cfg = sample_config(rng)
        fold_aucs = []
        for fold_tr, fold_va in fold_indices:
            auc = run_config(
                x_trainval_raw.iloc[fold_tr], y_trainval.iloc[fold_tr],
                x_trainval_raw.iloc[fold_va], y_trainval.iloc[fold_va],
                cfg, seed=torch_seed_for_search, epochs=screen_epochs,
            )
            if auc is not None:
                fold_aucs.append(auc)
        if not fold_aucs:
            continue
        fold_aucs = np.array(fold_aucs)
        mean_auc = float(fold_aucs.mean()); std_auc = float(fold_aucs.std())
        robust_score = mean_auc - 0.5 * std_auc  # penalize unstable configs
        screen_results.append({"config_id": ci, **cfg, "cv_mean_auc": round(mean_auc, 4),
                                "cv_std_auc": round(std_auc, 4), "robust_score": round(robust_score, 4)})
        print(f"  config {ci+1}/{n_configs}: cv_mean={mean_auc:.4f} cv_std={std_auc:.4f} "
              f"robust={robust_score:.4f}  {cfg}")

    screen_df = pd.DataFrame(screen_results).sort_values("robust_score", ascending=False)
    screen_df.to_csv(OUT_DIR / "mmft_architecture_search_screen.csv", index=False, encoding="utf-8-sig")
    print("\nTop 5 by robust_score (mean CV AUC - 0.5*std):")
    print(screen_df.head(5)[["config_id", "cv_mean_auc", "cv_std_auc", "robust_score"]].to_string())

    # ---- confirmatory: repeated CV (3 repeats x 5 folds) on top 3 configs ----
    top3 = screen_df.head(3).to_dict("records")
    confirm_results = []
    for rank_i, row in enumerate(top3):
        cfg = {k: row[k] for k in ["k", "d_model", "n_heads", "n_layers", "dropout", "lr", "weight_decay",
                                    "rank_weight", "use_risk_token", "risk_c", "focal_gamma", "mixup_alpha"]}
        all_aucs = []
        for rep in range(3):
            skf_r = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000 + rep)
            for fold_tr, fold_va in skf_r.split(x_trainval_raw, y_trainval):
                auc = run_config(
                    x_trainval_raw.iloc[fold_tr], y_trainval.iloc[fold_tr],
                    x_trainval_raw.iloc[fold_va], y_trainval.iloc[fold_va],
                    cfg, seed=torch_seed_for_search + rep, epochs=confirm_epochs,
                )
                if auc is not None:
                    all_aucs.append(auc)
        all_aucs = np.array(all_aucs)
        confirm_results.append({"config_id": row["config_id"], "cfg": cfg,
                                 "repeated_cv_mean": float(all_aucs.mean()),
                                 "repeated_cv_std": float(all_aucs.std()),
                                 "n_folds_total": len(all_aucs)})
        print(f"\n[confirm] config {row['config_id']}: repeated-CV mean={all_aucs.mean():.4f} "
              f"std={all_aucs.std():.4f} (n={len(all_aucs)} fold-evals)")

    best = max(confirm_results, key=lambda r: r["repeated_cv_mean"] - 0.5 * r["repeated_cv_std"])
    print(f"\n[selected] config {best['config_id']} with cfg={best['cfg']}")
    (OUT_DIR / "mmft_architecture_search_confirm.json").write_text(
        json.dumps(confirm_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- FINAL: retrain best config on full training set, evaluate ONCE per seed on the LOCKED test set ----
    print("\n[final] retraining selected config on full training set, evaluating on locked test set, 10 seeds...")
    final_cfg = best["cfg"]
    test_aucs = []
    all_probs = None
    for seed in range(20262001, 20262011):
        auc = run_config(x_trainval_raw, y_trainval, x_test_raw, y_test, final_cfg, seed=seed, epochs=final_epochs)
        test_aucs.append(auc)
        print(f"  seed={seed} test AUC={auc:.4f}")
    test_aucs = np.array(test_aucs)

    ridge_pred = pd.read_csv(ROOT / "server_results" / "petct_blood_benchmark" / "best_model_test_predictions.csv")

    # median-seed DeLong vs ridge for an apples-to-apples significance check
    median_seed = 20262001 + int(np.argsort(test_aucs)[len(test_aucs) // 2])
    prob_median = None
    set_seed(median_seed)
    modalities = build_modalities(x_trainval_raw)
    features = [f for g in modalities.values() for f in g]
    x_tr = x_trainval_raw[features].apply(pd.to_numeric, errors="coerce")
    x_te = x_test_raw[features].apply(pd.to_numeric, errors="coerce")
    pre = ModalityAwarePreprocessor(modalities, k=final_cfg["k"])
    train_batch = pre.fit_transform(x_tr, y_trainval)
    test_batch = pre.transform(x_te)
    if final_cfg["use_risk_token"]:
        risk = RiskTokenAppender(modality_id=len(modalities), c=final_cfg["risk_c"])
        train_batch = risk.fit_transform(train_batch, y_trainval)
        test_batch = risk.transform(test_batch)
    model = MMFTTransformer(n_features=train_batch.x.shape[1],
                             n_modalities=int(train_batch.modality_ids.max().item()) + 1,
                             d_model=final_cfg["d_model"], n_heads=final_cfg["n_heads"],
                             n_layers=final_cfg["n_layers"], dropout=final_cfg["dropout"])
    if final_cfg["use_risk_token"]:
        initialize_prior_residual(model, risk_feature_index=train_batch.x.shape[1] - 1)
    y_tr_np = y_trainval.to_numpy().astype("float32")
    pos = y_tr_np.sum(); neg = len(y_tr_np) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=final_cfg["lr"], weight_decay=final_cfg["weight_decay"])
    loader = DataLoader(TensorDataset(train_batch.x, torch.from_numpy(y_tr_np)), batch_size=len(y_tr_np), shuffle=False)
    for _ in range(final_epochs):
        model.train()
        for xb, yb in loader:
            mini = TabularBatch(x=xb, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            bce = binary_focal_bce(logits, yb, pos_weight, gamma=final_cfg["focal_gamma"])
            rank = pairwise_auc_loss_local(logits, yb)
            loss = (1.0 - final_cfg["rank_weight"]) * bce + final_cfg["rank_weight"] * rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    prob_median = predict(model, test_batch)

    m = ridge_pred.copy()
    m["p_searched"] = prob_median
    a, b, p = delong_roc_test(m["histology"].to_numpy(), m["predicted_probability"].to_numpy(), m["p_searched"].to_numpy())

    result = {
        "final_config": final_cfg,
        "test_auc_10seed_mean": round(float(test_aucs.mean()), 4),
        "test_auc_10seed_std": round(float(test_aucs.std()), 4),
        "test_auc_10seed_min": round(float(test_aucs.min()), 4),
        "test_auc_10seed_max": round(float(test_aucs.max()), 4),
        "median_seed": median_seed,
        "delong_vs_ridge_primary": {"ridge_auc": round(a, 4), "searched_mmft_auc": round(b, 4), "p": round(p, 4)},
    }
    print("\n" + json.dumps(result, indent=2))
    (OUT_DIR / "mmft_architecture_search_final.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

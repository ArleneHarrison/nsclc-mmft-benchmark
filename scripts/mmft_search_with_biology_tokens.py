"""Leakage-free architecture search, extended to let TRAINING-ONLY cross-
validation itself decide whether the TCGA-guided glycolysis/inflammation
tokens (scripts/mmft_named_tokens.py, robust name-indexed, assertion-
verified implementation) are worth including -- rather than us declaring a
verdict from a hand-picked configuration or seed.

Same non-negotiable rule as scripts/mmft_architecture_search.py: the locked
77-patient test set is touched exactly once, for the single final selected
configuration, after all search and confirmation is complete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
from scripts.mmft_biology_guided import delong_roc_test
from scripts.mmft_named_tokens import build_named_batches, run_named_config

OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
        "risk_c": float(rng.choice([0.3, 1.0, 3.0, 10.0])),
        "focal_gamma": float(rng.choice([0.0, 0.0, 1.0, 2.0])),
        "use_risk_token": True,  # always keep the (already-established-useful) global prior
        "use_glycolysis": bool(rng.choice([True, False])),
        "use_inflammation": bool(rng.choice([True, False])),
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

    rng = np.random.default_rng(20260703)
    search_seed = 888

    print(f"[search] {n_configs} random configs (incl. glycolysis/inflammation token on/off) x "
          f"{n_folds}-fold CV on TRAINING data only (n={len(x_trainval_raw)}); test set untouched.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(x_trainval_raw, y_trainval))

    screen_results = []
    for ci in range(n_configs):
        cfg = sample_config(rng)
        fold_aucs = []
        for fold_tr, fold_va in fold_indices:
            auc = run_named_config(
                x_trainval_raw.iloc[fold_tr], y_trainval.iloc[fold_tr],
                x_trainval_raw.iloc[fold_va], y_trainval.iloc[fold_va],
                cfg, seed=search_seed, epochs=screen_epochs,
            )
            if auc is not None:
                fold_aucs.append(auc)
        if not fold_aucs:
            continue
        fold_aucs = np.array(fold_aucs)
        mean_auc = float(fold_aucs.mean()); std_auc = float(fold_aucs.std())
        robust_score = mean_auc - 0.5 * std_auc
        screen_results.append({"config_id": ci, **cfg, "cv_mean_auc": round(mean_auc, 4),
                                "cv_std_auc": round(std_auc, 4), "robust_score": round(robust_score, 4)})
        tok = f"glyc={cfg['use_glycolysis']} infl={cfg['use_inflammation']}"
        print(f"  config {ci+1}/{n_configs}: cv_mean={mean_auc:.4f} cv_std={std_auc:.4f} "
              f"robust={robust_score:.4f}  {tok}")

    screen_df = pd.DataFrame(screen_results).sort_values("robust_score", ascending=False)
    screen_df.to_csv(OUT_DIR / "mmft_biology_search_screen.csv", index=False, encoding="utf-8-sig")

    # does the unbiased search prefer biology tokens on average?
    with_glyc = screen_df[screen_df["use_glycolysis"]]["cv_mean_auc"]
    without_glyc = screen_df[~screen_df["use_glycolysis"]]["cv_mean_auc"]
    with_infl = screen_df[screen_df["use_inflammation"]]["cv_mean_auc"]
    without_infl = screen_df[~screen_df["use_inflammation"]]["cv_mean_auc"]
    u_glyc = stats.mannwhitneyu(with_glyc, without_glyc, alternative="two-sided")[1] if len(with_glyc) and len(without_glyc) else None
    u_infl = stats.mannwhitneyu(with_infl, without_infl, alternative="two-sided")[1] if len(with_infl) and len(without_infl) else None
    print(f"\n[unbiased screen check] glycolysis-token configs mean CV AUC={with_glyc.mean():.4f} (n={len(with_glyc)}) "
          f"vs without={without_glyc.mean():.4f} (n={len(without_glyc)}), Mann-Whitney p={u_glyc}")
    print(f"[unbiased screen check] inflammation-token configs mean CV AUC={with_infl.mean():.4f} (n={len(with_infl)}) "
          f"vs without={without_infl.mean():.4f} (n={len(without_infl)}), Mann-Whitney p={u_infl}")

    print("\nTop 5 by robust_score:")
    print(screen_df.head(5)[["config_id", "cv_mean_auc", "cv_std_auc", "robust_score",
                              "use_glycolysis", "use_inflammation"]].to_string())

    top3 = screen_df.head(3).to_dict("records")
    confirm_results = []
    for row in top3:
        cfg = {k: row[k] for k in ["k", "d_model", "n_heads", "n_layers", "dropout", "lr", "weight_decay",
                                    "rank_weight", "risk_c", "focal_gamma", "use_risk_token",
                                    "use_glycolysis", "use_inflammation"]}
        all_aucs = []
        for rep in range(3):
            skf_r = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2000 + rep)
            for fold_tr, fold_va in skf_r.split(x_trainval_raw, y_trainval):
                auc = run_named_config(
                    x_trainval_raw.iloc[fold_tr], y_trainval.iloc[fold_tr],
                    x_trainval_raw.iloc[fold_va], y_trainval.iloc[fold_va],
                    cfg, seed=search_seed + rep, epochs=confirm_epochs,
                )
                if auc is not None:
                    all_aucs.append(auc)
        all_aucs = np.array(all_aucs)
        confirm_results.append({"config_id": row["config_id"], "cfg": cfg,
                                 "repeated_cv_mean": float(all_aucs.mean()),
                                 "repeated_cv_std": float(all_aucs.std()),
                                 "n_folds_total": len(all_aucs)})
        print(f"\n[confirm] config {row['config_id']} (glyc={cfg['use_glycolysis']}, infl={cfg['use_inflammation']}): "
              f"repeated-CV mean={all_aucs.mean():.4f} std={all_aucs.std():.4f}")

    best = max(confirm_results, key=lambda r: r["repeated_cv_mean"] - 0.5 * r["repeated_cv_std"])
    print(f"\n[selected] config {best['config_id']} with cfg={best['cfg']}")

    final_cfg = best["cfg"]
    test_aucs = []
    for seed in range(20263001, 20263011):
        auc = run_named_config(x_trainval_raw, y_trainval, x_test_raw, y_test, final_cfg, seed=seed, epochs=final_epochs)
        test_aucs.append(auc)
        print(f"  seed={seed} test AUC={auc:.4f}")
    test_aucs = np.array(test_aucs)

    ridge_pred = pd.read_csv(ROOT / "server_results" / "petct_blood_benchmark" / "best_model_test_predictions.csv")
    median_seed = 20263001 + int(np.argsort(test_aucs)[len(test_aucs) // 2])
    prob_median = None
    for seed in [median_seed]:
        import torch
        from scripts.train_mmft_transformer_petct import set_seed
        set_seed(seed)
        train_nb, eval_nb, pre = build_named_batches(
            x_trainval_raw, y_trainval, x_test_raw, y_test, k=final_cfg["k"], risk_c=final_cfg["risk_c"],
            use_risk_token=final_cfg["use_risk_token"], use_glycolysis=final_cfg["use_glycolysis"],
            use_inflammation=final_cfg["use_inflammation"],
        )
        from scripts.mmft_named_tokens import pairwise_auc_loss_local
        from scripts.train_mmft_transformer_petct import MMFTTransformer, TabularBatch, binary_focal_bce, predict
        from torch import nn
        train_batch, eval_batch = train_nb.batch, eval_nb.batch
        model = MMFTTransformer(n_features=train_batch.x.shape[1],
                                 n_modalities=int(train_batch.modality_ids.max().item()) + 1,
                                 d_model=final_cfg["d_model"], n_heads=final_cfg["n_heads"],
                                 n_layers=final_cfg["n_layers"], dropout=final_cfg["dropout"])
        if final_cfg["use_risk_token"]:
            risk_idx = train_nb.index_of("ridge_risk_token")
            with torch.no_grad():
                model.linear_head.weight.zero_(); model.linear_head.bias.zero_()
                model.linear_head.weight[0, risk_idx] = 1.0
                last_linear = None
                for module in model.deep_head.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                if last_linear is not None:
                    last_linear.weight.zero_(); last_linear.bias.zero_()
        y_tr_np = y_trainval.to_numpy().astype("float32")
        pos = y_tr_np.sum(); neg = len(y_tr_np) - pos
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=final_cfg["lr"], weight_decay=final_cfg["weight_decay"])
        for _ in range(final_epochs):
            model.train()
            mini = TabularBatch(x=train_batch.x, modality_ids=train_batch.modality_ids, feature_ids=train_batch.feature_ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mini)
            bce = binary_focal_bce(logits, torch.from_numpy(y_tr_np), pos_weight, gamma=final_cfg["focal_gamma"])
            rank = pairwise_auc_loss_local(logits, torch.from_numpy(y_tr_np))
            loss = (1.0 - final_cfg["rank_weight"]) * bce + final_cfg["rank_weight"] * rank
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        prob_median = predict(model, eval_batch)

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
        "unbiased_screen_glycolysis_effect": {"with_mean": round(float(with_glyc.mean()), 4) if len(with_glyc) else None,
                                               "without_mean": round(float(without_glyc.mean()), 4) if len(without_glyc) else None,
                                               "mannwhitney_p": round(float(u_glyc), 4) if u_glyc is not None else None},
        "unbiased_screen_inflammation_effect": {"with_mean": round(float(with_infl.mean()), 4) if len(with_infl) else None,
                                                 "without_mean": round(float(without_infl.mean()), 4) if len(without_infl) else None,
                                                 "mannwhitney_p": round(float(u_infl), 4) if u_infl is not None else None},
    }
    print("\n" + json.dumps(result, indent=2))
    (OUT_DIR / "mmft_biology_search_final.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

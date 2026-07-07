"""TabPFN as a comparator model -- a recent (Hollmann et al., Nature 2025)
tabular foundation model specifically designed for small-N tabular
classification via in-context learning (no gradient-based training per
task). This directly tests the paper's core question ("does a more modern
method beat simple ridge regression at n=255?") against genuinely current,
published, state-of-the-art small-data tabular ML -- not just our own
custom architecture.

Uses the IDENTICAL train/test split and primary feature set as the rest of
the study for a fair, leakage-free, apples-to-apples comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_petct_blood_models import RANDOM_STATE, read_data
from scripts.mmft_biology_guided import delong_roc_test

OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = ["gender", "age", "BMI", "smoking", "T stage", "N stage", "stage",
           "WBC", "NEU", "LYM", "EOS", "BAS", "PLT", "CEA", "LDH", "ALB", "Ca2+", "NLR", "dNLR",
           "TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]


def main():
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

    df, _ = read_data()
    y = df["histology"].astype(int).to_numpy()
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    y_tr, y_te = y[tr_idx], y[te_idx]

    Xp = df[PRIMARY].apply(pd.to_numeric, errors="coerce")

    # Same preprocessing convention as the primary pipeline: impute -> variance filter -> scale -> SelectKBest(k=5)
    imputer = SimpleImputer(strategy="median").fit(Xp.iloc[tr_idx])
    Xi = imputer.transform(Xp)
    vt = VarianceThreshold().fit(Xi[tr_idx])
    Xv = vt.transform(Xi)
    scaler = StandardScaler().fit(Xv[tr_idx])
    Xs = scaler.transform(Xv)
    selector = SelectKBest(score_func=f_classif, k=5).fit(Xs[tr_idx], y_tr)
    Xk = selector.transform(Xs)
    selected_names = np.array(PRIMARY)[vt.get_support()][selector.get_support()]
    print("TabPFN input features (same primary-block SelectKBest convention):", list(selected_names))

    X_tr, X_te = Xk[tr_idx], Xk[te_idx]

    aucs = []
    for seed in range(7001, 7011):  # 10 seeds -- TabPFN is stochastic in its internal ensembling
        clf = TabPFNClassifier(random_state=seed, device="cpu")
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob)
        aucs.append(auc)
        print(f"  seed={seed} TabPFN test AUC={auc:.4f}")
    aucs = np.array(aucs)

    # median-seed predictions for DeLong vs ridge primary
    median_seed = 7001 + int(np.argsort(aucs)[len(aucs) // 2])
    clf = TabPFNClassifier(random_state=median_seed, device="cpu")
    clf.fit(X_tr, y_tr)
    prob_median = clf.predict_proba(X_te)[:, 1]

    ridge_pred = pd.read_csv(ROOT / "server_results" / "petct_blood_benchmark" / "best_model_test_predictions.csv")
    m = ridge_pred.copy()
    m["p_tabpfn"] = prob_median
    a, b, p = delong_roc_test(m["histology"].to_numpy(), m["predicted_probability"].to_numpy(), m["p_tabpfn"].to_numpy())

    result = {
        "selected_features": list(selected_names),
        "test_auc_10seed_mean": round(float(aucs.mean()), 4),
        "test_auc_10seed_std": round(float(aucs.std()), 4),
        "test_auc_10seed_min": round(float(aucs.min()), 4),
        "test_auc_10seed_max": round(float(aucs.max()), 4),
        "median_seed": median_seed,
        "delong_vs_ridge_primary": {"ridge_auc": round(a, 4), "tabpfn_auc": round(b, 4), "p": round(p, 4)},
        "note": "TabPFN v2 (Hollmann et al., Nature 2025), in-context learning, no gradient training per task; "
                "same primary SelectKBest(k=5) feature block and train/test split as the rest of this study.",
    }
    print("\n" + json.dumps(result, indent=2))
    (OUT_DIR / "tabpfn_comparator_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

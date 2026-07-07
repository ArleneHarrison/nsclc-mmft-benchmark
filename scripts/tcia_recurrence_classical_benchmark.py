"""Honest classical-ML benchmark for the new TCIA recurrence-prediction task
(n=189, 47 events). Given the small N, nested cross-validation is the PRIMARY
evaluation (a single 70/30 hold-out would leave only ~14 test-set events,
too noisy to trust as the headline number) -- the same lesson this project
learned the hard way on the PLOS-2024 cohort, applied proactively here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public_data" / "tcia_semantic_features" / "tcia_recurrence_analysis_ready.csv"
OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 20260702

CATEGORICAL = [
    "aim_Anatomic_Location", "aim_Axial_Location", "aim_Nodule_Attenuation",
    "aim_Nodule_Margins-Primary_Pattern", "aim_Nodule_Margins-Secondary_Pattern",
    "aim_Nodule_Shape", "aim_Nodule_Calcification", "aim_Nodule_Periphery",
    "aim_Satellite_Nodules_in_Primary_Lesion_Lobe_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Non-Lesion_Lobe_Same_Lung_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Contralateral_Lung_gretater_than_4mm_noncalcified",
    "aim_Centrilobular_Nodules_-_Diffuse_RB_type_nodules", "aim_Emphysema", "aim_Fibrosis",
]
CONTINUOUS = ["age", "pack_years"]
BINARY = ["gender_male", "ever_smoker", "egfr_mutant", "kras_mutant", "alk_positive"]
ALL_FEATURES = CATEGORICAL + CONTINUOUS + BINARY


def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("cont", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), CONTINUOUS),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
    ])
    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif, k=10)),
        ("model", model),
    ])


def nested_cv_auc(X, y, model_pipeline, param_grid, outer_splits=5, inner_splits=5, n_repeats=4, seed0=RANDOM_STATE):
    """Repeated nested CV: outer loop estimates generalisation, inner loop tunes
    hyperparameters -- fully unbiased, appropriate for n=189."""
    all_aucs = []
    for rep in range(n_repeats):
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed0 + rep)
        for tr_idx, te_idx in outer.split(X, y):
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed0 + rep)
            gs = GridSearchCV(model_pipeline, param_grid, scoring="roc_auc", cv=inner, n_jobs=-1)
            gs.fit(X.iloc[tr_idx], y[tr_idx])
            prob = gs.predict_proba(X.iloc[te_idx])[:, 1]
            if len(np.unique(y[te_idx])) < 2:
                continue
            all_aucs.append(roc_auc_score(y[te_idx], prob))
    return np.array(all_aucs)


def main():
    df = pd.read_csv(DATA)
    y = df["recurrence"].to_numpy()
    X = df[ALL_FEATURES]
    n = len(y)
    print(f"n={n}, events={y.sum()} ({y.mean()*100:.1f}%)")

    results = {}

    # ---- Ridge logistic ----
    ridge_pipe = make_pipeline(LogisticRegression(penalty="l2", solver="liblinear",
                                                    class_weight="balanced", max_iter=3000,
                                                    random_state=RANDOM_STATE))
    ridge_grid = {"select__k": [5, 8, 10, 15, 20], "model__C": [0.03, 0.1, 0.3, 1.0, 3.0]}
    aucs = nested_cv_auc(X, y, ridge_pipe, ridge_grid)
    results["ridge_logistic"] = {"mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std()), 4),
                                  "n_folds": len(aucs)}
    print(f"Ridge logistic:      nested-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f} (n={len(aucs)} folds)")

    # ---- Random forest ----
    rf_pipe = make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
    rf_grid = {"select__k": [8, 10, 15, "all" if False else 20],
               "model__n_estimators": [100, 200], "model__max_depth": [3, 5, None]}
    aucs = nested_cv_auc(X, y, rf_pipe, rf_grid)
    results["random_forest"] = {"mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std()), 4),
                                 "n_folds": len(aucs)}
    print(f"Random forest:       nested-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f} (n={len(aucs)} folds)")

    # ---- XGBoost ----
    if XGBClassifier is not None:
        xgb_pipe = make_pipeline(XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                                random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"))
        xgb_grid = {"select__k": [8, 10, 15, 20], "model__n_estimators": [80, 150],
                    "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.08]}
        aucs = nested_cv_auc(X, y, xgb_pipe, xgb_grid)
        results["xgboost"] = {"mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std()), 4),
                               "n_folds": len(aucs)}
        print(f"XGBoost:             nested-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f} (n={len(aucs)} folds)")

    # ---- Clinical-only baseline (no semantic features) ----
    clinical_only_pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cont", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), CONTINUOUS),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ])),
        ("model", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                                      max_iter=3000, random_state=RANDOM_STATE)),
    ])
    aucs = nested_cv_auc(X[CONTINUOUS + BINARY], y, clinical_only_pipe, {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]})
    results["clinical_only_ridge"] = {"mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std()), 4),
                                       "n_folds": len(aucs)}
    print(f"Clinical-only ridge: nested-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f} (n={len(aucs)} folds)")

    # ---- single hold-out for reference (secondary, not primary due to small N) ----
    tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    ridge_final = make_pipeline(LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                                                     max_iter=3000, random_state=RANDOM_STATE))
    gs = GridSearchCV(ridge_final, ridge_grid, scoring="roc_auc",
                       cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
    gs.fit(X.iloc[tr_idx], y[tr_idx])
    prob = gs.predict_proba(X.iloc[te_idx])[:, 1]
    holdout_auc = roc_auc_score(y[te_idx], prob)
    results["ridge_single_holdout_secondary"] = {"auc": round(float(holdout_auc), 4), "n_test": len(te_idx),
                                                   "n_test_events": int(y[te_idx].sum()),
                                                   "note": "SECONDARY only -- too few test events for a trustworthy headline number at this N"}
    print(f"\n[secondary] single 70/30 hold-out ridge AUC = {holdout_auc:.4f} "
          f"(test n={len(te_idx)}, events={int(y[te_idx].sum())} -- too few to be the headline)")

    pd.Series(y[te_idx]).to_csv(OUT_DIR / "tcia_recurrence_holdout_y.csv", index=False)
    pd.Series(prob).to_csv(OUT_DIR / "tcia_recurrence_holdout_ridge_prob.csv", index=False)

    import json
    (OUT_DIR / "tcia_recurrence_classical_benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR / "tcia_recurrence_classical_benchmark.json")


if __name__ == "__main__":
    main()

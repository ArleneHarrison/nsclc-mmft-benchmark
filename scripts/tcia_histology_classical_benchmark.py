"""Honest classical-ML benchmark for histology prediction (adenocarcinoma vs
squamous) from TCIA NSCLC-Radiogenomics CT semantic features -- a structurally
independent modality (radiologist-coded categorical descriptors, no PET, no
blood) from the primary PLOS-2024 model, in a fully independent cohort.

Nested CV is PRIMARY (small N); a single hold-out is reported as secondary.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 20260702

SEMANTIC = [
    "aim_Anatomic_Location", "aim_Axial_Location", "aim_Nodule_Attenuation",
    "aim_Nodule_Margins-Primary_Pattern", "aim_Nodule_Margins-Secondary_Pattern",
    "aim_Nodule_Shape", "aim_Nodule_Calcification", "aim_Nodule_Periphery",
    "aim_Satellite_Nodules_in_Primary_Lesion_Lobe_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Non-Lesion_Lobe_Same_Lung_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Contralateral_Lung_gretater_than_4mm_noncalcified",
    "aim_Centrilobular_Nodules_-_Diffuse_RB_type_nodules", "aim_Emphysema", "aim_Fibrosis",
]
CLINICAL_CONT = ["age", "pack_years"]
CLINICAL_BIN = ["gender_male", "ever_smoker"]
MOLECULAR = ["egfr_mutant", "kras_mutant", "alk_positive"]


def load_data():
    aim = pd.read_csv(ROOT / "public_data" / "tcia_semantic_features" / "aim_semantic_features_raw.csv")
    clin = pd.read_csv(ROOT / "public_data" / "nsclc_radiogenomics_clinical.csv")
    clin.columns = [c.strip() for c in clin.columns]
    df = aim.merge(clin, on="Case ID", how="inner")
    df = df[df["Histology"].isin(["Adenocarcinoma", "Squamous cell carcinoma"])].copy()
    df["histology"] = (df["Histology"] == "Squamous cell carcinoma").astype(int)
    df["age"] = pd.to_numeric(df["Age at Histological Diagnosis"], errors="coerce")
    df["pack_years"] = pd.to_numeric(df["Pack Years"], errors="coerce")
    df["gender_male"] = (df["Gender"] == "Male").astype(int)
    df["ever_smoker"] = df["Smoking status"].isin(["Current", "Former"]).astype(int)
    df["egfr_mutant"] = df["EGFR mutation status"].map({"Mutant": 1, "Wildtype": 0}).astype(float)
    df["kras_mutant"] = df["KRAS mutation status"].map({"Mutant": 1, "Wildtype": 0}).astype(float)
    df["alk_positive"] = df["ALK translocation status"].map({"Mutant": 1, "Wildtype": 0}).astype(float)
    return df


def make_pipeline(feature_block):
    cat_cols = [c for c in feature_block if c in SEMANTIC]
    cont_cols = [c for c in feature_block if c in CLINICAL_CONT]
    bin_cols = [c for c in feature_block if c in (CLINICAL_BIN + MOLECULAR)]
    transformers = []
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    if cont_cols:
        transformers.append(("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler())]), cont_cols))
    if bin_cols:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bin_cols))
    pre = ColumnTransformer(transformers)
    return Pipeline([
        ("pre", pre), ("var", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif)),
        ("model", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                                      max_iter=3000, random_state=RANDOM_STATE)),
    ])


def nested_cv_auc(X, y, pipe, grid, outer_splits=5, inner_splits=5, n_repeats=4, seed0=RANDOM_STATE):
    all_aucs = []
    for rep in range(n_repeats):
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed0 + rep)
        for tr_idx, te_idx in outer.split(X, y):
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed0 + rep)
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
            gs.fit(X.iloc[tr_idx], y[tr_idx])
            prob = gs.predict_proba(X.iloc[te_idx])[:, 1]
            if len(np.unique(y[te_idx])) >= 2:
                all_aucs.append(roc_auc_score(y[te_idx], prob))
    return np.array(all_aucs)


def main():
    df = load_data()
    y = df["histology"].to_numpy()
    n = len(y)
    print(f"n={n}, squamous={y.sum()} ({y.mean()*100:.1f}%)")

    blocks = {
        "clinical_only": CLINICAL_CONT + CLINICAL_BIN,
        "semantic_only": SEMANTIC,
        "semantic_plus_clinical": SEMANTIC + CLINICAL_CONT + CLINICAL_BIN,
        "semantic_plus_clinical_plus_molecular": SEMANTIC + CLINICAL_CONT + CLINICAL_BIN + MOLECULAR,
    }
    results = {}
    for name, feats in blocks.items():
        X = df[feats]
        max_k = len(feats) if len(feats) < 8 else None
        k_grid = [k for k in [3, 5, 8, 10, 15, 20] if k <= (X.shape[1] * 3)]  # rough cap; SelectKBest clips internally anyway
        pipe = make_pipeline(feats)
        grid = {"select__k": k_grid, "model__C": [0.03, 0.1, 0.3, 1.0, 3.0]}
        aucs = nested_cv_auc(X, y, pipe, grid)
        results[name] = {"mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std()), 4), "n_folds": len(aucs)}
        print(f"{name:45s} nested-CV AUC = {aucs.mean():.4f} +/- {aucs.std():.4f} (n={len(aucs)} folds)")

    # single hold-out (secondary) for the full block, and to produce predictions for later DeLong comparisons
    full_feats = blocks["semantic_plus_clinical_plus_molecular"]
    X_full = df[full_feats]
    tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    pipe = make_pipeline(full_feats)
    k_grid = [3, 5, 8, 10, 15, 20]
    grid = {"select__k": k_grid, "model__C": [0.03, 0.1, 0.3, 1.0, 3.0]}
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
    gs.fit(X_full.iloc[tr_idx], y[tr_idx])
    prob = gs.predict_proba(X_full.iloc[te_idx])[:, 1]
    holdout_auc = roc_auc_score(y[te_idx], prob)
    results["ridge_full_single_holdout_secondary"] = {"auc": round(float(holdout_auc), 4),
                                                        "n_test": len(te_idx), "best_params": gs.best_params_}
    print(f"\n[secondary] single hold-out (full block) ridge AUC = {holdout_auc:.4f} (n_test={len(te_idx)}, "
          f"best_k={gs.best_params_['select__k']})")

    pd.DataFrame({"Case ID": df.iloc[te_idx]["Case ID"].to_numpy(), "y": y[te_idx], "p_ridge": prob}).to_csv(
        OUT_DIR / "tcia_histology_holdout_ridge_predictions.csv", index=False, encoding="utf-8-sig")
    df.to_csv(ROOT / "public_data" / "tcia_semantic_features" / "tcia_histology_analysis_ready.csv",
              index=False, encoding="utf-8-sig")

    (OUT_DIR / "tcia_histology_classical_benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR / "tcia_histology_classical_benchmark.json")


if __name__ == "__main__":
    main()

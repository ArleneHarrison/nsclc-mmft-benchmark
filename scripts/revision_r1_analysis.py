"""Targeted statistical audit for PLOS ONE revision PONE-D-26-33174.

This script addresses the editor/reviewer requests that can change numerical
reporting: fold-level nesting, repeated-split PET increments, separate
bootstrap intervals, clinically interpretable thresholds, and correlated PET
predictors. All preprocessing and feature selection live inside a scikit-learn
Pipeline and are fitted only on the relevant training partition.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public_data" / "candidate_blood_datasets"
OUT_DIR = ROOT / "outputs" / "PONE-D-26-33174_revision_20260919" / "analysis"
RANDOM_STATE = 20260701

CLINICAL = ["gender", "age", "BMI", "smoking", "T stage", "N stage", "stage"]
BLOOD = ["WBC", "NEU", "LYM", "EOS", "BAS", "PLT", "CEA", "LDH", "ALB", "Ca2+", "NLR", "dNLR"]
METABOLIC = ["TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]
PRIMARY = CLINICAL + BLOOD + METABOLIC


def read_data() -> pd.DataFrame:
    baseline = pd.read_csv(
        DATA_DIR / "plos_2024_petct_baseline_s2_dataset_Baseline characteristics of patients.csv"
    )
    ct = pd.read_csv(
        DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_CT  radiomics features.csv"
    ).add_prefix("ct__").rename(columns={"ct__ID": "ID"})
    pet = pd.read_csv(
        DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_PET  radiomics features.csv"
    ).add_prefix("pet__").rename(columns={"pet__ID": "ID"})
    return baseline.merge(ct, on="ID", how="inner").merge(pet, on="ID", how="inner").replace(
        [np.inf, -np.inf], np.nan
    )


def make_pipeline(k: int = 5, c_value: float = 3.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_search(n_features: int, seed: int) -> GridSearchCV:
    k_grid = sorted({k for k in [3, 5, 8, 12, 20, min(24, n_features)] if k <= n_features})
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(
        make_pipeline(),
        {"select__k": k_grid, "model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        scoring="roc_auc",
        cv=inner,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )


def selected_names(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    after_variance = np.asarray(feature_names)[pipe.named_steps["variance"].get_support()]
    return after_variance[pipe.named_steps["select"].get_support()].tolist()


def stratified_bootstrap_auc(
    y: np.ndarray, probability: np.ndarray, n_boot: int = 10_000, seed: int = RANDOM_STATE
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    aucs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        aucs[i] = roc_auc_score(y[idx], probability[idx])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), aucs


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold_p_squamous": threshold,
        "sensitivity_squamous": tp / (tp + fn),
        "specificity_squamous": tn / (tn + fp),
        "ppv_squamous": tp / (tp + fp) if tp + fp else np.nan,
        "npv_squamous": tn / (tn + fn) if tn + fn else np.nan,
        "classified_squamous_n": int(tp + fp),
        "classified_adenocarcinoma_n": int(tn + fn),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    y = df["histology"].astype(int).to_numpy()

    # 1. Auditable nested CV: the full Pipeline is handed to each inner search.
    x_primary = df[PRIMARY].apply(pd.to_numeric, errors="coerce")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x_primary, y), start=1):
        # Match the prespecified analysis: the same reproducible inner-fold
        # generator is used in every outer fold; fitting still sees only that
        # outer fold's training partition.
        search = make_search(len(PRIMARY), RANDOM_STATE)
        search.fit(x_primary.iloc[train_idx], y[train_idx])
        probability = search.predict_proba(x_primary.iloc[test_idx])[:, 1]
        best = search.best_estimator_
        fold_rows.append(
            {
                "outer_fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "outer_fold_auc": roc_auc_score(y[test_idx], probability),
                "inner_best_auc": search.best_score_,
                "selected_k": search.best_params_["select__k"],
                "ridge_C": search.best_params_["model__C"],
                "selected_features": "; ".join(selected_names(best, PRIMARY)),
                "pipeline_audit": "imputation>variance>standardization>SelectKBest>ridge all refit in outer training fold",
            }
        )
    nested_df = pd.DataFrame(fold_rows)
    nested_df.to_csv(OUT_DIR / "R1_nested_cv_fold_audit.csv", index=False, encoding="utf-8-sig")

    # 2. Fifty paired repeated hold-outs; each feature block is re-tuned on its
    # own training partition, then evaluated on the same test patients.
    repeated_rows: list[dict[str, float]] = []
    x_blood = df[CLINICAL + BLOOD].apply(pd.to_numeric, errors="coerce")
    for split_no in range(50):
        seed = 1000 + split_no
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.30, random_state=seed, stratify=y
        )
        block_probabilities: dict[str, np.ndarray] = {}
        block_aucs: dict[str, float] = {}
        block_params: dict[str, str] = {}
        for label, x_data in [("clinical_blood", x_blood), ("plus_pet", x_primary)]:
            search = make_search(x_data.shape[1], seed)
            search.fit(x_data.iloc[train_idx], y[train_idx])
            probability = search.predict_proba(x_data.iloc[test_idx])[:, 1]
            block_probabilities[label] = probability
            block_aucs[label] = roc_auc_score(y[test_idx], probability)
            block_params[label] = json.dumps(search.best_params_, sort_keys=True)
        repeated_rows.append(
            {
                "split": split_no + 1,
                "seed": seed,
                "auc_clinical_blood": block_aucs["clinical_blood"],
                "auc_plus_pet": block_aucs["plus_pet"],
                "delta_auc_plus_pet": block_aucs["plus_pet"] - block_aucs["clinical_blood"],
                "clinical_blood_best_params": block_params["clinical_blood"],
                "plus_pet_best_params": block_params["plus_pet"],
            }
        )
    repeated_df = pd.DataFrame(repeated_rows)
    repeated_df.to_csv(OUT_DIR / "R1_pet_increment_50_paired_splits.csv", index=False, encoding="utf-8-sig")
    delta = repeated_df["delta_auc_plus_pet"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    mean_boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(20_000)])
    pet_summary = {
        "n_splits": len(delta),
        "mean_delta_auc": float(delta.mean()),
        "median_delta_auc": float(np.median(delta)),
        "empirical_p2_5": float(np.percentile(delta, 2.5)),
        "empirical_p97_5": float(np.percentile(delta, 97.5)),
        "bootstrap_95ci_mean_low": float(np.percentile(mean_boot, 2.5)),
        "bootstrap_95ci_mean_high": float(np.percentile(mean_boot, 97.5)),
        "proportion_delta_above_zero": float(np.mean(delta > 0)),
    }

    # 3. Locked test: separate CIs for the fixed k=5 primary model and the
    # k=12 incremental-analysis model, plus clinically interpretable thresholds.
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    locked_models = {"primary_k5": make_pipeline(5, 3.0), "incremental_k12": make_pipeline(12, 3.0)}
    locked_rows: list[dict[str, float]] = []
    locked_predictions: dict[str, np.ndarray] = {}
    for i, (label, model) in enumerate(locked_models.items()):
        model.fit(x_primary.iloc[train_idx], y[train_idx])
        probability = model.predict_proba(x_primary.iloc[test_idx])[:, 1]
        locked_predictions[label] = probability
        lo, hi, _ = stratified_bootstrap_auc(y[test_idx], probability, seed=RANDOM_STATE + i)
        locked_rows.append(
            {
                "model": label,
                "auc": roc_auc_score(y[test_idx], probability),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_bootstrap": 10_000,
            }
        )
    locked_df = pd.DataFrame(locked_rows)
    locked_df["prediction_correlation_with_other_model"] = np.corrcoef(
        locked_predictions["primary_k5"], locked_predictions["incremental_k12"]
    )[0, 1]
    locked_df.to_csv(OUT_DIR / "R1_separate_bootstrap_intervals.csv", index=False, encoding="utf-8-sig")

    threshold_df = pd.DataFrame(
        [
            threshold_metrics(y[test_idx], locked_predictions["primary_k5"], threshold)
            for threshold in [0.30, 0.50, 0.70]
        ]
    )
    threshold_df.to_csv(OUT_DIR / "R1_primary_threshold_metrics.csv", index=False, encoding="utf-8-sig")

    # 4. Correlation/suppression audit for the hand-score predictors.
    selected = ["gender", "CEA", "SUVmean", "SUVmax", "SUVmin"]
    corr = df[selected].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr.to_csv(OUT_DIR / "R1_selected_predictor_spearman_correlations.csv", encoding="utf-8-sig")
    pet_pairs = []
    for left, right in [("SUVmean", "SUVmax"), ("SUVmean", "SUVmin"), ("SUVmax", "SUVmin")]:
        rho, p_value = spearmanr(df[left], df[right], nan_policy="omit")
        pet_pairs.append({"predictor_1": left, "predictor_2": right, "spearman_rho": rho, "p_value": p_value})
    pd.DataFrame(pet_pairs).to_csv(OUT_DIR / "R1_pet_pairwise_correlations.csv", index=False, encoding="utf-8-sig")

    # 5. Multiplicity audit for the three sequential DeLong tests in Table 3.
    raw_p = np.array([0.5467, 0.0493, 0.0453])
    _, holm_p, _, _ = multipletests(raw_p, method="holm")
    multiplicity_df = pd.DataFrame(
        {
            "comparison": ["blood vs clinical", "PET vs clinical+blood", "radiomics vs PET block"],
            "unadjusted_delong_p": raw_p,
            "holm_adjusted_p": holm_p,
        }
    )
    multiplicity_df.to_csv(OUT_DIR / "R1_table3_multiplicity.csv", index=False, encoding="utf-8-sig")

    summary = {
        "nested_cv": {
            "mean_auc": float(nested_df["outer_fold_auc"].mean()),
            "sd_auc": float(nested_df["outer_fold_auc"].std(ddof=0)),
            "folds": nested_df["outer_fold_auc"].tolist(),
            "audit": "All transformations, SelectKBest, and k/C tuning were repeated inside each outer fold.",
        },
        "pet_increment_50_splits": pet_summary,
        "locked_model_bootstrap": locked_rows,
        "thresholds": threshold_df.to_dict(orient="records"),
        "pet_pairwise_correlations": pet_pairs,
        "multiplicity": multiplicity_df.to_dict(orient="records"),
    }
    (OUT_DIR / "R1_revision_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

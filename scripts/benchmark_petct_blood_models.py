from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public_data" / "candidate_blood_datasets"
OUT_DIR = ROOT / "server_results" / "petct_blood_benchmark"
RANDOM_STATE = 20260701


@dataclass(frozen=True)
class DatasetBlock:
    name: str
    features: list[str]


def read_data() -> tuple[pd.DataFrame, list[DatasetBlock]]:
    baseline_path = DATA_DIR / "plos_2024_petct_baseline_s2_dataset_Baseline characteristics of patients.csv"
    ct_path = DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_CT  radiomics features.csv"
    pet_path = DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_PET  radiomics features.csv"

    baseline = pd.read_csv(baseline_path)
    ct = pd.read_csv(ct_path).add_prefix("ct__")
    pet = pd.read_csv(pet_path).add_prefix("pet__")
    ct = ct.rename(columns={"ct__ID": "ID"})
    pet = pet.rename(columns={"pet__ID": "ID"})

    df = baseline.merge(ct, on="ID", how="inner").merge(pet, on="ID", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan)

    clinical_blood = [
        "gender",
        "age",
        "BMI",
        "smoking",
        "T stage",
        "N stage",
        "stage",
        "WBC",
        "NEU",
        "LYM",
        "EOS",
        "BAS",
        "PLT",
        "CEA",
        "LDH",
        "ALB",
        "Ca2+",
        "NLR",
        "dNLR",
    ]
    metabolic = ["TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]
    ct_features = [c for c in df.columns if c.startswith("ct__")]
    pet_features = [c for c in df.columns if c.startswith("pet__")]

    blocks = [
        DatasetBlock("clinical_blood", clinical_blood),
        DatasetBlock("clinical_blood_metabolic", clinical_blood + metabolic),
        DatasetBlock("ct_radiomics", ct_features),
        DatasetBlock("pet_radiomics", pet_features),
        DatasetBlock("ct_pet_radiomics", ct_features + pet_features),
        DatasetBlock("combined_all", clinical_blood + metabolic + ct_features + pet_features),
    ]
    return df, blocks


def make_pipeline(estimator, use_scaler: bool = True) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
    ]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.extend(
        [
            ("select", SelectKBest(score_func=f_classif, k=20)),
            ("model", estimator),
        ]
    )
    return Pipeline(steps)


def model_grid(n_features: int) -> list[tuple[str, Pipeline, dict]]:
    small_k = sorted({k for k in [5, 10, 15, 20] if k < n_features} | {min(n_features, 30)})
    large_k = sorted({k for k in [20, 50, 100, 200] if k < n_features} | {min(n_features, 300)})

    grids: list[tuple[str, Pipeline, dict]] = []
    grids.append(
        (
            "lasso_logistic",
            make_pipeline(
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=8000,
                    random_state=RANDOM_STATE,
                )
            ),
            {"select__k": small_k if n_features <= 40 else large_k, "model__C": [0.03, 0.1, 0.3, 1.0]},
        )
    )
    grids.append(
        (
            "ridge_logistic",
            make_pipeline(
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                )
            ),
            {"select__k": small_k if n_features <= 40 else large_k, "model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        )
    )
    grids.append(
        (
            "svm_rbf",
            make_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            {
                "select__k": small_k if n_features <= 40 else [k for k in large_k if k <= 100],
                "model__C": [0.3, 1.0, 3.0],
                "model__gamma": ["scale", 0.01],
            },
        )
    )
    grids.append(
        (
            "random_forest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_scaler=False,
            ),
            {
                "select__k": small_k if n_features <= 40 else [k for k in large_k if k <= 200],
                "model__max_depth": [2, 3, None],
                "model__min_samples_leaf": [2, 5],
            },
        )
    )
    grids.append(
        (
            "extra_trees",
            make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=800,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                use_scaler=False,
            ),
            {
                "select__k": small_k if n_features <= 40 else [k for k in large_k if k <= 200],
                "model__max_depth": [2, 3, None],
                "model__min_samples_leaf": [2, 5],
            },
        )
    )

    if XGBClassifier is not None:
        grids.append(
            (
                "xgboost",
                make_pipeline(
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    use_scaler=False,
                ),
                {
                    "select__k": small_k if n_features <= 40 else [k for k in large_k if k <= 200],
                    "model__n_estimators": [80, 180],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.03, 0.08],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                },
            )
        )

    if LGBMClassifier is not None:
        grids.append(
            (
                "lightgbm",
                make_pipeline(
                    LGBMClassifier(
                        objective="binary",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                        class_weight="balanced",
                    ),
                    use_scaler=False,
                ),
                {
                    "select__k": small_k if n_features <= 40 else [k for k in large_k if k <= 200],
                    "model__n_estimators": [80, 180],
                    "model__num_leaves": [7, 15],
                    "model__learning_rate": [0.03, 0.08],
                    "model__min_child_samples": [5, 10],
                },
            )
        )
    return grids


def score_binary(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1 / (1 + np.exp(-scores))


def selected_feature_names(fitted: Pipeline, input_features: list[str]) -> list[str]:
    support = fitted.named_steps["variance"].get_support()
    after_variance = np.array(input_features)[support]
    selector = fitted.named_steps["select"]
    return list(after_variance[selector.get_support()])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, blocks = read_data()
    y = df["histology"].astype(int)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "random_state": RANDOM_STATE,
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    results = []
    best = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for block in blocks:
        x = df[block.features].apply(pd.to_numeric, errors="coerce")
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        for model_name, pipe, params in model_grid(len(block.features)):
            search = GridSearchCV(
                estimator=pipe,
                param_grid=params,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score=np.nan,
            )
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            prob_train = predict_probability(fitted, x_train)
            prob_test = predict_probability(fitted, x_test)
            train_scores = score_binary(y_train.to_numpy(), prob_train)
            test_scores = score_binary(y_test.to_numpy(), prob_test)
            row = {
                "feature_block": block.name,
                "model": model_name,
                "n_features_input": len(block.features),
                "n_features_selected": int(fitted.named_steps["select"].get_support().sum()),
                "cv_auc_mean": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                **{f"train_{k}": v for k, v in train_scores.items()},
                **{f"test_{k}": v for k, v in test_scores.items()},
            }
            results.append(row)
            if best is None or row["test_auc"] > best["row"]["test_auc"]:
                best = {
                    "row": row,
                    "model": clone(fitted).fit(x_train, y_train),
                    "x_test": x_test,
                    "y_test": y_test,
                    "prob_test": prob_test,
                    "features": block.features,
                    "ids_test": df.loc[x_test.index, "ID"].tolist(),
                }
            print(f"{block.name:26s} {model_name:16s} cv_auc={search.best_score_:.3f} test_auc={test_scores['auc']:.3f}")

    results_df = pd.DataFrame(results).sort_values(["test_auc", "cv_auc_mean"], ascending=False)
    results_df.to_csv(OUT_DIR / "model_benchmark_results.csv", index=False, encoding="utf-8-sig")

    assert best is not None
    best_model = best["model"]
    best_features = selected_feature_names(best_model, best["features"])
    pd.DataFrame({"selected_feature": best_features}).to_csv(
        OUT_DIR / "best_model_selected_features.csv", index=False, encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame(
        {
            "ID": best["ids_test"],
            "histology": best["y_test"].to_numpy(),
            "predicted_probability": best["prob_test"],
        }
    )
    pred_df.to_csv(OUT_DIR / "best_model_test_predictions.csv", index=False, encoding="utf-8-sig")

    fpr, tpr, _ = roc_curve(best["y_test"], best["prob_test"])
    plt.figure(figsize=(5, 5), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC={best['row']['test_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Best model ROC: {best['row']['feature_block']} + {best['row']['model']}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_model_roc.png")
    plt.close()

    report_lines = [
        "# PET/CT Blood Model Benchmark",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Target counts: {summary['target_counts']}",
        f"- Best feature block: {best['row']['feature_block']}",
        f"- Best model: {best['row']['model']}",
        f"- Test AUC: {best['row']['test_auc']:.3f}",
        f"- Test accuracy: {best['row']['test_accuracy']:.3f}",
        f"- Test sensitivity: {best['row']['test_sensitivity']:.3f}",
        f"- Test specificity: {best['row']['test_specificity']:.3f}",
        "",
        "## Top 10 Runs",
        "",
        results_df[
            [
                "feature_block",
                "model",
                "cv_auc_mean",
                "test_auc",
                "test_accuracy",
                "test_sensitivity",
                "test_specificity",
                "n_features_selected",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False),
    ]
    (OUT_DIR / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

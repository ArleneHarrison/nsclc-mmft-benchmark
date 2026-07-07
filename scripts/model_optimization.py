"""Genuine model-optimization pass for the NSCLC subtype paper.

All experiments reuse the existing PLOS-2024 cohort (n=255, no new data) and the
same preprocessing conventions as scripts/benchmark_petct_blood_models.py /
scripts/supplementary_experiments.py. Everything here is cheap (pure local
compute, minutes not hours) and is designed to either genuinely improve the
model or produce an honest negative result that strengthens the parsimony
narrative -- never to p-hack a bigger headline number.

Six additions:
  1. Repeated randomized hold-out (50 splits) for the primary ridge model:
     replaces the single-seed point estimate with a distribution.
  2. Elastic-net vs pure ridge on the primary block.
  3. A proper "radiomics signature" via bootstrap-stability-selected LASSO on
     the radiomics block (literature-standard radiomics-signature approach,
     not naive SelectKBest on 3874 raw features) combined with the primary
     block. DeLong vs primary on the shared test set.
  4. A simple ridge+XGBoost stacking ensemble on the primary block.
  5. Cross-fitted Platt recalibration of the primary model (fixes slope 1.51).
  6. Learning curve (test AUC vs training-set fraction) -- motivates whether
     more data (hospital cohort) is likely to help.
  7. Subgroup AUC by overall stage with bootstrap CI.

Outputs -> outputs/model_optimization/
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public_data" / "candidate_blood_datasets"
OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 20260701

CLINICAL = ["gender", "age", "BMI", "smoking", "T stage", "N stage", "stage"]
BLOOD = ["WBC", "NEU", "LYM", "EOS", "BAS", "PLT", "CEA", "LDH", "ALB", "Ca2+", "NLR", "dNLR"]
METABOLIC = ["TLG", "SUVmean", "MTV", "SUVmax", "SUVmin"]
PRIMARY = CLINICAL + BLOOD + METABOLIC


def read_data():
    baseline = pd.read_csv(
        DATA_DIR / "plos_2024_petct_baseline_s2_dataset_Baseline characteristics of patients.csv"
    )
    ct = pd.read_csv(DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_CT  radiomics features.csv").add_prefix("ct__")
    pet = pd.read_csv(DATA_DIR / "plos_2024_petct_radiomics_s1_dataset_PET  radiomics features.csv").add_prefix("pet__")
    ct = ct.rename(columns={"ct__ID": "ID"})
    pet = pet.rename(columns={"pet__ID": "ID"})
    df = baseline.merge(ct, on="ID", how="inner").merge(pet, on="ID", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_pipe(k, C, penalty="l2", l1_ratio=None, use_scaler=True):
    steps = [("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    if penalty == "elasticnet":
        model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio,
                                    class_weight="balanced", max_iter=8000, C=C, random_state=RANDOM_STATE)
    else:
        model = LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                                    max_iter=3000, C=C, random_state=RANDOM_STATE)
    steps.append(("model", model))
    return Pipeline(steps)


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


def hosmer_lemeshow(y_true, prob, g=5):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(prob)})
    d["decile"] = pd.qcut(d["p"], q=g, duplicates="drop")
    obs = d.groupby("decile", observed=True)["y"].sum()
    exp = d.groupby("decile", observed=True)["p"].sum()
    n = d.groupby("decile", observed=True)["y"].count()
    hl = (((obs - exp) ** 2) / (exp * (1 - exp / n))).sum()
    dof = len(obs) - 2
    p = 1 - stats.chi2.cdf(hl, dof) if dof > 0 else np.nan
    return float(hl), int(dof), float(p)


def main():
    df = read_data()
    y = df["histology"].astype(int).to_numpy()
    n = len(y)
    idx = np.arange(n)
    radiomics = [c for c in df.columns if c.startswith(("ct__", "pet__"))]

    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    y_tr, y_te = y[tr_idx], y[te_idx]

    results = {}

    # ------------------------------------------------------------------ #
    # 0. Primary model -- FIXED to the established benchmark config
    #    (k=5, C=3.0; matches server_results/petct_blood_benchmark
    #    /model_benchmark_results.csv best row, test AUC 0.854, exactly).
    #    Not re-derived via a fresh grid search, so all "vs primary"
    #    comparisons below stay consistent with the paper's headline number.
    # ------------------------------------------------------------------ #
    Xp = df[PRIMARY].apply(pd.to_numeric, errors="coerce")
    kg_primary = sorted({k for k in [5, 10, 15, 20] if k < len(PRIMARY)} | {len(PRIMARY)})
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_c = 5, 3.0
    primary_model = make_pipe(best_k, best_c)
    primary_model.fit(Xp.iloc[tr_idx], y_tr)
    p_primary_test = primary_model.predict_proba(Xp.iloc[te_idx])[:, 1]
    auc_primary = roc_auc_score(y_te, p_primary_test)
    print(f"[0] primary reference (fixed, matches headline): k={best_k} C={best_c} test AUC={auc_primary:.3f}")
    results["primary_reference"] = {"k": best_k, "C": best_c, "test_auc": round(float(auc_primary), 3)}

    # ------------------------------------------------------------------ #
    # 1. Repeated randomized hold-out (50 splits) for the primary pipeline
    # ------------------------------------------------------------------ #
    print("[1] repeated randomized hold-out (50 splits)...")
    rep_aucs = []
    for seed in range(50):
        tri, tei = train_test_split(idx, test_size=0.30, random_state=1000 + seed, stratify=y)
        m = make_pipe(best_k, best_c)
        m.fit(Xp.iloc[tri], y[tri])
        p = m.predict_proba(Xp.iloc[tei])[:, 1]
        rep_aucs.append(roc_auc_score(y[tei], p))
    rep_aucs = np.array(rep_aucs)
    results["repeated_holdout_50splits"] = {
        "mean": round(float(rep_aucs.mean()), 3), "median": round(float(np.median(rep_aucs)), 3),
        "std": round(float(rep_aucs.std()), 3),
        "p2_5": round(float(np.percentile(rep_aucs, 2.5)), 3),
        "p97_5": round(float(np.percentile(rep_aucs, 97.5)), 3),
        "min": round(float(rep_aucs.min()), 3), "max": round(float(rep_aucs.max()), 3),
    }
    np.savetxt(OUT_DIR / "repeated_holdout_aucs.csv", rep_aucs, delimiter=",", header="auc", comments="")
    print("   ", results["repeated_holdout_50splits"])

    # ------------------------------------------------------------------ #
    # 2. Elastic-net vs ridge on the primary block
    # ------------------------------------------------------------------ #
    print("[2] elastic-net vs ridge...")
    enet_grid = {"select__k": kg_primary, "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                 "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
    enet_pipe = make_pipe(5, 1.0, penalty="elasticnet", l1_ratio=0.5)
    enet_search = GridSearchCV(enet_pipe, enet_grid, scoring="roc_auc", cv=cv5, n_jobs=-1)
    enet_search.fit(Xp.iloc[tr_idx], y_tr)
    enet_best = enet_search.best_estimator_
    p_enet_test = enet_best.predict_proba(Xp.iloc[te_idx])[:, 1]
    auc_enet = roc_auc_score(y_te, p_enet_test)
    _, _, p_enet_vs_ridge = delong_roc_test(y_te, p_primary_test, p_enet_test)
    results["elastic_net"] = {
        "best_params": enet_search.best_params_, "test_auc": round(float(auc_enet), 3),
        "delong_p_vs_ridge": round(p_enet_vs_ridge, 4),
    }
    print(f"    elastic-net test AUC={auc_enet:.3f} (params={enet_search.best_params_}) "
          f"DeLong vs ridge p={p_enet_vs_ridge:.4f}")

    # ------------------------------------------------------------------ #
    # 3. Radiomics signature via bootstrap-stability-selected LASSO
    # ------------------------------------------------------------------ #
    print("[3] radiomics signature (stability selection)...")
    Xr_tr = df.loc[tr_idx, radiomics].apply(pd.to_numeric, errors="coerce")
    Xr_te = df.loc[te_idx, radiomics].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median").fit(Xr_tr)
    Xr_tr_i = imputer.transform(Xr_tr)
    Xr_te_i = imputer.transform(Xr_te)
    vt = VarianceThreshold().fit(Xr_tr_i)
    Xr_tr_v = vt.transform(Xr_tr_i)
    Xr_te_v = vt.transform(Xr_te_i)
    scaler_r = StandardScaler().fit(Xr_tr_v)
    Xr_tr_s = scaler_r.transform(Xr_tr_v)
    Xr_te_s = scaler_r.transform(Xr_te_v)
    kept_names = np.array(radiomics)[vt.get_support()]

    rng = np.random.default_rng(RANDOM_STATE)
    n_boot = 100
    select_count = np.zeros(Xr_tr_s.shape[1])
    ntr = Xr_tr_s.shape[0]
    for b in range(n_boot):
        bidx = rng.integers(0, ntr, ntr)
        xb, yb = Xr_tr_s[bidx], y_tr[bidx]
        if len(np.unique(yb)) < 2:
            continue
        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                                    class_weight="balanced", max_iter=2000, random_state=b)
        lasso.fit(xb, yb)
        select_count += (np.abs(lasso.coef_[0]) > 1e-8).astype(int)
    freq = select_count / n_boot
    stable_mask = freq >= 0.4
    n_stable = int(stable_mask.sum())
    print(f"    stability selection: {n_stable} radiomics features selected in >=40% of {n_boot} bootstraps")
    if n_stable == 0:
        top_idx = np.argsort(-freq)[:5]
        stable_mask = np.zeros_like(freq, dtype=bool)
        stable_mask[top_idx] = True
        n_stable = int(stable_mask.sum())
        print(f"    (fallback) using top {n_stable} by selection frequency")

    stable_names = kept_names[stable_mask]
    pd.DataFrame({"radiomics_feature": stable_names, "selection_frequency": freq[stable_mask]}).sort_values(
        "selection_frequency", ascending=False
    ).to_csv(OUT_DIR / "radiomics_signature_stable_features.csv", index=False, encoding="utf-8-sig")

    # fit a compact logistic on the stable radiomics features -> "radiomics score"
    sig_model = LogisticRegression(penalty="l2", solver="liblinear", C=0.3,
                                    class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    sig_model.fit(Xr_tr_s[:, stable_mask], y_tr)
    radiomics_score_tr = sig_model.decision_function(Xr_tr_s[:, stable_mask])
    radiomics_score_te = sig_model.decision_function(Xr_te_s[:, stable_mask])
    radiomics_score_auc_alone = roc_auc_score(y_te, sig_model.predict_proba(Xr_te_s[:, stable_mask])[:, 1])
    print(f"    radiomics signature alone: test AUC={radiomics_score_auc_alone:.3f}")

    # Like-for-like combination: append radiomics_score to the EXACT feature
    # set already selected by the primary model (not a fresh SelectKBest over
    # 25 candidates, which reintroduces the k-instability problem fixed in
    # step 0). This isolates the marginal effect of the compact radiomics
    # signature on top of the known-good primary configuration.
    primary_support = primary_model.named_steps["variance"].get_support()
    primary_selector = primary_model.named_steps["select"].get_support()
    primary_feat_names = list(np.array(PRIMARY)[primary_support][primary_selector])
    print(f"    primary selected features: {primary_feat_names}")

    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    Xp_tr_sel = pre.fit_transform(Xp.iloc[tr_idx][primary_feat_names])
    Xp_te_sel = pre.transform(Xp.iloc[te_idx][primary_feat_names])
    Xp_tr_combo = np.column_stack([Xp_tr_sel, radiomics_score_tr])
    Xp_te_combo = np.column_stack([Xp_te_sel, radiomics_score_te])

    combo_search = GridSearchCV(
        LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                            max_iter=3000, random_state=RANDOM_STATE),
        {"C": [0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc", cv=cv5, n_jobs=-1,
    )
    combo_search.fit(Xp_tr_combo, y_tr)
    combo_best = combo_search.best_estimator_
    p_combo_test = combo_best.predict_proba(Xp_te_combo)[:, 1]
    auc_combo = roc_auc_score(y_te, p_combo_test)
    _, _, p_combo_vs_primary = delong_roc_test(y_te, p_primary_test, p_combo_test)
    results["radiomics_signature"] = {
        "n_stable_features": n_stable, "bootstrap_reps": n_boot, "selection_threshold": 0.4,
        "signature_alone_test_auc": round(float(radiomics_score_auc_alone), 3),
        "primary_plus_signature_test_auc_like_for_like": round(float(auc_combo), 3),
        "delong_p_vs_primary": round(p_combo_vs_primary, 4),
        "combo_best_C": combo_search.best_params_["C"],
        "note": "radiomics_score appended to the exact 5 primary features (no fresh SelectKBest), isolating its marginal effect",
    }
    print(f"    primary(5 feats)+radiomics_score test AUC={auc_combo:.3f} (C={combo_search.best_params_['C']}) "
          f"DeLong vs primary p={p_combo_vs_primary:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Stacking ensemble: ridge + XGBoost (simple probability average)
    # ------------------------------------------------------------------ #
    print("[4] stacking ensemble (ridge + XGBoost)...")
    if XGBClassifier is not None:
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif)),
            ("model", XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                     random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")),
        ])
        xgb_grid = {"select__k": kg_primary, "model__n_estimators": [80, 180],
                    "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.08]}
        xgb_search = GridSearchCV(xgb_pipe, xgb_grid, scoring="roc_auc", cv=cv5, n_jobs=-1)
        xgb_search.fit(Xp.iloc[tr_idx], y_tr)
        p_xgb_test = xgb_search.predict_proba(Xp.iloc[te_idx])[:, 1]
        auc_xgb = roc_auc_score(y_te, p_xgb_test)

        p_stack_test = 0.5 * p_primary_test + 0.5 * p_xgb_test
        auc_stack = roc_auc_score(y_te, p_stack_test)
        _, _, p_stack_vs_primary = delong_roc_test(y_te, p_primary_test, p_stack_test)
        results["stacking_ensemble"] = {
            "xgb_test_auc": round(float(auc_xgb), 3),
            "stack_test_auc": round(float(auc_stack), 3),
            "delong_p_vs_primary": round(p_stack_vs_primary, 4),
        }
        print(f"    XGBoost alone AUC={auc_xgb:.3f}; ridge+XGB average AUC={auc_stack:.3f} "
              f"DeLong vs primary p={p_stack_vs_primary:.4f}")
    else:
        results["stacking_ensemble"] = {"note": "xgboost not available"}
        print("    xgboost not available, skipped")

    # ------------------------------------------------------------------ #
    # 5. Cross-fitted Platt recalibration of the primary model
    # ------------------------------------------------------------------ #
    print("[5] cross-fitted recalibration...")
    oof_logit = np.zeros(len(tr_idx))
    for fold_tr, fold_va in cv5.split(Xp.iloc[tr_idx], y_tr):
        fm = make_pipe(best_k, best_c)
        fm.fit(Xp.iloc[tr_idx].iloc[fold_tr], y_tr[fold_tr])
        p_va = fm.predict_proba(Xp.iloc[tr_idx].iloc[fold_va])[:, 1]
        p_va = np.clip(p_va, 1e-6, 1 - 1e-6)
        oof_logit[fold_va] = np.log(p_va / (1 - p_va))
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt.fit(oof_logit.reshape(-1, 1), y_tr)

    p_test_raw = np.clip(p_primary_test, 1e-6, 1 - 1e-6)
    logit_test = np.log(p_test_raw / (1 - p_test_raw))
    p_test_recal = platt.predict_proba(logit_test.reshape(-1, 1))[:, 1]

    brier_before = brier_score_loss(y_te, p_primary_test)
    brier_after = brier_score_loss(y_te, p_test_recal)
    hl_before = hosmer_lemeshow(y_te, p_primary_test)
    hl_after = hosmer_lemeshow(y_te, p_test_recal)
    auc_recal = roc_auc_score(y_te, p_test_recal)  # AUC invariant to monotonic recalibration, sanity check
    results["recalibration"] = {
        "platt_slope": round(float(platt.coef_[0][0]), 3), "platt_intercept": round(float(platt.intercept_[0]), 3),
        "brier_before": round(float(brier_before), 3), "brier_after": round(float(brier_after), 3),
        "hl_p_before": round(hl_before[2], 3), "hl_p_after": round(hl_after[2], 3),
        "auc_unchanged_check": round(float(auc_recal), 3),
    }
    print("   ", results["recalibration"])
    pd.DataFrame({"ID": df.loc[te_idx, "ID"].to_numpy(), "y": y_te,
                  "p_raw": p_primary_test, "p_recalibrated": p_test_recal}).to_csv(
        OUT_DIR / "recalibrated_test_predictions.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    # 6. Learning curve: test AUC vs training-set fraction
    # ------------------------------------------------------------------ #
    print("[6] learning curve...")
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_rows = []
    for frac in fractions:
        aucs = []
        m_sub = max(20, int(round(frac * len(tr_idx))))
        for rep in range(20):
            rng2 = np.random.default_rng(2000 + rep)
            if frac >= 0.999:
                sub_idx = tr_idx
            else:
                sub_idx, _ = train_test_split(tr_idx, train_size=min(m_sub, len(tr_idx) - 1),
                                               random_state=2000 + rep, stratify=y_tr)
            if len(np.unique(y[sub_idx])) < 2:
                continue
            m = make_pipe(min(best_k, len(sub_idx) - 2), best_c)
            try:
                m.fit(Xp.loc[sub_idx], y[sub_idx])
                p = m.predict_proba(Xp.iloc[te_idx])[:, 1]
                aucs.append(roc_auc_score(y_te, p))
            except Exception:
                continue
        aucs = np.array(aucs)
        lc_rows.append({"train_fraction": frac, "n_train": m_sub,
                         "mean_auc": round(float(aucs.mean()), 3) if len(aucs) else np.nan,
                         "std_auc": round(float(aucs.std()), 3) if len(aucs) else np.nan})
        print(f"    frac={frac} n_train~{m_sub} mean AUC={lc_rows[-1]['mean_auc']}")
    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(OUT_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig")
    results["learning_curve"] = lc_rows

    # ------------------------------------------------------------------ #
    # 7. Subgroup AUC by dataset-coded stage substratum (bootstrap CI)
    #    NOTE: the source paper (Zhang et al., PLOS ONE 2024) states ALL 255
    #    patients are AJCC Stage III (single-lesion >1cm, stage III inclusion
    #    criterion). The "stage" column in the released CSV takes only values
    #    1/2/3 -- given the stage-III-only inclusion criterion, this almost
    #    certainly encodes IIIA/IIIB/IIIC sub-stage, NOT overall stage I/II/III.
    #    This mapping is not confirmed by a data dictionary, so we label the
    #    subgroup by its raw source code and flag the likely correspondence
    #    rather than asserting it as fact.
    # ------------------------------------------------------------------ #
    print("[7] subgroup by dataset-coded stage substratum (likely IIIA/IIIB/IIIC)...")
    stage_te = pd.to_numeric(df.loc[te_idx, "stage"], errors="coerce").to_numpy()
    subgroup_rows = []
    for grp_name, mask in [("stage-code 1 (likely IIIA)", stage_te == 1),
                            ("stage-code 2 (likely IIIB)", stage_te == 2),
                            ("stage-code 3 (likely IIIC)", stage_te == 3)]:
        yt, pt = y_te[mask], p_primary_test[mask]
        if len(np.unique(yt)) < 2 or mask.sum() < 8:
            subgroup_rows.append({"subgroup": grp_name, "n": int(mask.sum()), "auc": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        a = roc_auc_score(yt, pt)
        rng3 = np.random.default_rng(RANDOM_STATE)
        boots = []
        for _ in range(2000):
            bi = rng3.integers(0, len(yt), len(yt))
            if len(np.unique(yt[bi])) == 2:
                boots.append(roc_auc_score(yt[bi], pt[bi]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        subgroup_rows.append({"subgroup": grp_name, "n": int(mask.sum()), "auc": round(float(a), 3),
                               "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3)})
    pd.DataFrame(subgroup_rows).to_csv(OUT_DIR / "subgroup_by_stage.csv", index=False, encoding="utf-8-sig")
    results["subgroup_by_stage"] = subgroup_rows
    for r in subgroup_rows:
        print("   ", r)

    (OUT_DIR / "model_optimization_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

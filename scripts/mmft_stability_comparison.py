"""Definitive comparison: does biology-guided token injection stabilize MMFT
across random initializations, versus the original single-risk-token design?

Runs both variants over the SAME 10 seeds, on the SAME train/test split
(RANDOM_STATE from benchmark_petct_blood_models), and formally tests for
differences in mean (Mann-Whitney U) and variance (Levene's test, robust to
non-normality) of the resulting test-AUC distributions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_mmft_rank_refit import train_rank_refit as train_original
from scripts.mmft_biology_guided import train_biology_guided, delong_roc_test

OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(20261101, 20261111))  # 10 seeds


def main() -> None:
    orig_runs, bio_runs = [], []
    for s in SEEDS:
        ro = train_original(seed=s)
        rb = train_biology_guided(seed=s)
        orig_runs.append(ro); bio_runs.append(rb)
        print(f"seed={s}  original={ro['metrics']['auc']:.3f}  biology-guided={rb['metrics']['auc']:.3f}")

    orig_aucs = np.array([r["metrics"]["auc"] for r in orig_runs])
    bio_aucs = np.array([r["metrics"]["auc"] for r in bio_runs])

    mw_stat, mw_p = stats.mannwhitneyu(orig_aucs, bio_aucs, alternative="two-sided")
    lev_stat, lev_p = stats.levene(orig_aucs, bio_aucs, center="median")

    def summ(a):
        return {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
                "std": round(float(a.std(ddof=1)), 4), "min": round(float(a.min()), 3),
                "max": round(float(a.max()), 3), "range": round(float(a.max() - a.min()), 3)}

    result = {
        "n_seeds": len(SEEDS), "seeds": SEEDS,
        "original_mmft_rank_refit": summ(orig_aucs),
        "biology_guided_mmft": summ(bio_aucs),
        "mann_whitney_u_p_means_equal": round(float(mw_p), 4),
        "levene_p_variances_equal": round(float(lev_p), 4),
        "variance_ratio_original_over_biology": round(float(orig_aucs.var(ddof=1) / bio_aucs.var(ddof=1)), 2),
    }
    print("\n" + json.dumps(result, indent=2))

    pd.DataFrame({"seed": SEEDS, "original_mmft_auc": orig_aucs, "biology_guided_mmft_auc": bio_aucs}).to_csv(
        OUT_DIR / "mmft_stability_comparison_per_seed.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "mmft_stability_comparison_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # DeLong on median-performing seed of each variant, against primary ridge
    ridge_pred = pd.read_csv(ROOT / "server_results" / "petct_blood_benchmark" / "best_model_test_predictions.csv")

    def median_run(runs, aucs):
        idx = int(np.argsort(aucs)[len(aucs) // 2])
        return runs[idx]

    orig_med = median_run(orig_runs, orig_aucs)
    bio_med = median_run(bio_runs, bio_aucs)

    orig_df = pd.DataFrame({"ID": orig_med["ids"], "y": orig_med["y_test"], "p": orig_med["prob"]})
    bio_df = pd.DataFrame({"ID": bio_med["ids"], "y": bio_med["y_test"], "p": bio_med["prob"]})

    # NOTE: delong_roc_test(y, prob_a, prob_b) returns (auc_a, auc_b, p) --
    # auc_a/auc_b correspond to the ARGUMENT ORDER (prob_a, prob_b), not to
    # any label. Keep variable names matched to argument order to avoid a
    # mislabeling bug.
    m1 = ridge_pred.merge(orig_df, on="ID")
    ridge_auc_1, orig_auc_1, p1 = delong_roc_test(
        m1["histology"].to_numpy(), m1["predicted_probability"].to_numpy(), m1["p"].to_numpy())
    m2 = ridge_pred.merge(bio_df, on="ID")
    ridge_auc_2, bio_auc_2, p2 = delong_roc_test(
        m2["histology"].to_numpy(), m2["predicted_probability"].to_numpy(), m2["p"].to_numpy())
    m3 = orig_df.merge(bio_df, on="ID", suffixes=("_orig", "_bio"))
    orig_auc_3, bio_auc_3, p3 = delong_roc_test(
        m3["y_orig"].to_numpy(), m3["p_orig"].to_numpy(), m3["p_bio"].to_numpy())

    delong = {
        "median_seed_orig_vs_ridge": {"orig_auc": round(orig_auc_1, 3), "ridge_auc": round(ridge_auc_1, 3), "p": round(p1, 4)},
        "median_seed_bio_vs_ridge": {"bio_auc": round(bio_auc_2, 3), "ridge_auc": round(ridge_auc_2, 3), "p": round(p2, 4)},
        "median_seed_orig_vs_bio": {"orig_auc": round(orig_auc_3, 3), "bio_auc": round(bio_auc_3, 3), "p": round(p3, 4)},
    }
    print("\n" + json.dumps(delong, indent=2))
    (OUT_DIR / "mmft_stability_comparison_delong.json").write_text(
        json.dumps(delong, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

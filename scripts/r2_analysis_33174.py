from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from statsmodels.stats.multitest import multipletests

ROOT = Path(os.environ.get('NSCLC_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
OUT = Path(os.environ.get('NSCLC_REVISION_OUT', str(ROOT / 'outputs/PONE-D-26-33174_R2_20261108')))
sys.path.insert(0, str(ROOT / 'scripts'))
import revision_r1_analysis as legacy_r1
from revision_r1_analysis import read_data, make_pipeline, selected_names
from supplementary_experiments import delong_roc_test, CLINICAL, BLOOD, METABOLIC
legacy_r1.DATA_DIR = Path(os.environ.get('NSCLC_PETCT_DATA_DIR', str(ROOT / 'public_data/candidate_blood_datasets')))

SEED = 20260701


def ci(y, score, seed=SEED, n=10000):
    positive = np.asarray(score)[y == 1]
    negative = np.asarray(score)[y == 0]
    pairwise = (positive[:, None] > negative).astype(float) + .5 * (positive[:, None] == negative)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n)
    for i in range(n):
        pos = np.bincount(rng.choice(len(positive), len(positive), replace=True), minlength=len(positive))
        neg = np.bincount(rng.choice(len(negative), len(negative), replace=True), minlength=len(negative))
        bootstrap[i] = pos @ pairwise @ neg / (len(positive) * len(negative))
    return [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])]


def search(n_features):
    if n_features <= 40:
        ks = sorted({k for k in [3, 5, 8, 12, 20] if k < n_features} | {min(n_features, 24)})
    else:
        ks = [20, 50, 100, 200, 300]
    return GridSearchCV(
        make_pipeline(), {'select__k': ks, 'model__C': [.03, .1, .3, 1., 3.]},
        scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=4, refit=True, error_score='raise',
    )


def main():
    analysis = OUT / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    df = read_data()
    y = df.histology.astype(int).to_numpy()
    assert len(df) == 255 and df.ID.is_unique
    tr, te = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=SEED)
    assert not set(tr) & set(te)
    blocks = {'clinical': CLINICAL, 'clinical_blood': CLINICAL + BLOOD,
              'plus_pet': CLINICAL + BLOOD + METABOLIC}
    radiomics = [c for c in df if c.startswith(('ct__', 'pet__'))]
    assert len(radiomics) == 3850
    blocks['plus_radiomics'] = blocks['plus_pet'] + radiomics
    results, all_folds, probs = [], [], {}
    for block, features in blocks.items():
        x = df[features].apply(pd.to_numeric, errors='coerce')
        tuned = search(len(features))
        tuned.fit(x.iloc[tr], y[tr])
        score = tuned.predict_proba(x.iloc[te])[:, 1]
        probs[block] = score
        bounds = ci(y[te], score)
        folds = []
        for fold, (outer_train, outer_test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y), 1):
            inner = search(len(features))
            inner.fit(x.iloc[outer_train], y[outer_train])
            auc = roc_auc_score(y[outer_test], inner.predict_proba(x.iloc[outer_test])[:, 1])
            folds.append(auc)
            all_folds.append({'block': block, 'fold': fold, 'auc': float(auc),
                              'k': int(inner.best_params_['select__k']), 'C': float(inner.best_params_['model__C']),
                              'n_train': len(outer_train), 'n_test': len(outer_test),
                              'selected_features': '; '.join(selected_names(inner.best_estimator_, features))})
            print(f'{block} nested fold {fold}: {auc:.6f}', flush=True)
        row = {'block': block, 'candidate_features': len(features),
               'k': int(tuned.best_params_['select__k']), 'C': float(tuned.best_params_['model__C']),
               'heldout_auc': float(roc_auc_score(y[te], score)), 'ci_low': bounds[0], 'ci_high': bounds[1],
               'n_bootstrap': 10000, 'nested_mean': float(np.mean(folds)), 'nested_sd': float(np.std(folds)),
               'nested_folds': [float(z) for z in folds],
               'selected_features': selected_names(tuned.best_estimator_, features)}
        results.append(row)
        print(json.dumps(row), flush=True)

    tests = []
    for previous, current in zip(list(blocks)[:-1], list(blocks)[1:]):
        a, b, p = delong_roc_test(y[te], probs[previous], probs[current])
        tests.append({'from': previous, 'to': current, 'auc_from': a, 'auc_to': b, 'delta': b-a, 'p': p})
    adjusted = multipletests([r['p'] for r in tests], method='holm')[1]
    for row, adjusted_p in zip(tests, adjusted):
        row['p_holm'] = float(adjusted_p)

    smoke = pd.to_numeric(df.smoking, errors='coerce')
    def summary(s):
        return {'n': int(s.notna().sum()), 'median': float(s.median()), 'q1': float(s.quantile(.25)), 'q3': float(s.quantile(.75))}
    smoke_result = {'missing': int(smoke.isna().sum()), 'unique': int(smoke.nunique()), 'min': float(smoke.min()),
                    'max': float(smoke.max()), 'overall': summary(smoke), 'adeno': summary(smoke[y == 0]),
                    'squamous': summary(smoke[y == 1]), 'p': float(mannwhitneyu(smoke[y == 0], smoke[y == 1], alternative='two-sided').pvalue),
                    'definition': 'Numeric source smoking variable; not binary status; units not documented in the supplied CSV.'}
    legacy_k12 = make_pipeline(12, 3.)
    legacy_x = df[blocks['plus_pet']].apply(pd.to_numeric, errors='coerce')
    legacy_k12.fit(legacy_x.iloc[tr], y[tr])
    k12_scores = legacy_k12.predict_proba(legacy_x.iloc[te])[:, 1]
    legacy_ci = ci(y[te], k12_scores, seed=SEED+1)
    legacy = {'k': 12, 'C': 3., 'auc': float(roc_auc_score(y[te], k12_scores)),
              'ci_low': legacy_ci[0], 'ci_high': legacy_ci[1], 'n_bootstrap': 10000,
              'status': 'Historical configuration selected with full-cohort information; superseded in revised Table 3'}
    payload = {'seed': SEED, 'n': len(y), 'development_n': len(tr), 'heldout_n': len(te),
               'positive_class': 'squamous carcinoma (1)', 'blocks': results, 'comparisons': tests,
               'smoking': smoke_result, 'historical_k12': legacy,
               'interpretation': 'All new k/C searches restricted to development data or each outer-training fold; the original holdout has been examined previously and is not a new external or untouched confirmatory test.'}
    (analysis / 'R2_analysis_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(all_folds).to_csv(analysis / 'R2_nested_fold_audit.csv', index=False)
    pd.DataFrame(results).drop(columns=['nested_folds','selected_features']).to_csv(analysis / 'R2_table3_corrected.csv', index=False)
    pd.DataFrame(tests).to_csv(analysis / 'R2_table3_paired_delong.csv', index=False)
    provenance = {'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy_r1.DATA_DIR.glob('plos_2024_petct_*.csv')}, 'source_patient_ids_exported': False}
    (analysis / 'R2_analysis_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()

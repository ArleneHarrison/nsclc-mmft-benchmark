"""Which mechanistic token drives the stability effect found in
scripts/mmft_stability_comparison.py? 2x2 ablation: glycolysis token
on/off x inflammation token on/off, 10 seeds each, same train/test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mmft_biology_guided import train_biology_guided

OUT_DIR = ROOT / "outputs" / "model_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(20261101, 20261111))  # 10 seeds
CONFIGS = [
    ("neither (=original single risk token)", False, False),
    ("glycolysis-only", True, False),
    ("inflammation-only", False, True),
    ("both (=biology-guided)", True, True),
]


def main() -> None:
    all_rows = []
    per_config_aucs = {}
    for name, use_g, use_i in CONFIGS:
        aucs = []
        for s in SEEDS:
            r = train_biology_guided(seed=s, use_glycolysis=use_g, use_inflammation=use_i)
            aucs.append(r["metrics"]["auc"])
            all_rows.append({"config": name, "seed": s, "auc": r["metrics"]["auc"]})
        aucs = np.array(aucs)
        per_config_aucs[name] = aucs
        print(f"{name:40s} mean={aucs.mean():.4f} std={aucs.std(ddof=1):.4f} "
              f"range=[{aucs.min():.3f},{aucs.max():.3f}]")

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "mmft_token_ablation_per_seed.csv", index=False, encoding="utf-8-sig")

    baseline = per_config_aucs["neither (=original single risk token)"]
    summary = {}
    for name, aucs in per_config_aucs.items():
        lev_p = stats.levene(baseline, aucs, center="median")[1] if name != "neither (=original single risk token)" else None
        summary[name] = {
            "mean": round(float(aucs.mean()), 4), "std": round(float(aucs.std(ddof=1)), 4),
            "min": round(float(aucs.min()), 3), "max": round(float(aucs.max()), 3),
            "levene_p_vs_neither": round(float(lev_p), 4) if lev_p is not None else None,
        }
    print("\n" + json.dumps(summary, indent=2))
    (OUT_DIR / "mmft_token_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

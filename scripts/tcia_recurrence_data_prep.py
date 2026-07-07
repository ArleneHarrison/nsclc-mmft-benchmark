"""Build the clean analysis-ready feature matrix for the new TCIA
NSCLC-Radiogenomics recurrence-prediction study.

Modalities:
  - CT semantic phenotype (radiologist-coded, from AIM annotations, 11 core
    features with 0% missingness): anatomic location, axial location,
    attenuation, margins (primary+secondary), shape, calcification,
    periphery, satellite nodules (same lobe / same lung / contralateral),
    centrilobular nodules.
  - Clinical: age, gender, smoking status, pack-years.
  - Molecular: EGFR / KRAS / ALK status.
  - Histology (adenocarcinoma / squamous), for stratified sensitivity checks.

Outcome: post-surgical Recurrence (yes/no). n=189 analysable, 47 events (24.9%).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public_data" / "tcia_semantic_features"

CORE_SEMANTIC = [
    "aim_Anatomic_Location", "aim_Axial_Location", "aim_Nodule_Attenuation",
    "aim_Nodule_Margins-Primary_Pattern", "aim_Nodule_Margins-Secondary_Pattern",
    "aim_Nodule_Shape", "aim_Nodule_Calcification", "aim_Nodule_Periphery",
    "aim_Satellite_Nodules_in_Primary_Lesion_Lobe_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Non-Lesion_Lobe_Same_Lung_greater_than_4mm_noncalcified",
    "aim_Nodules_in_Contralateral_Lung_gretater_than_4mm_noncalcified",
    "aim_Centrilobular_Nodules_-_Diffuse_RB_type_nodules", "aim_Emphysema", "aim_Fibrosis",
]


def main() -> None:
    aim = pd.read_csv(OUT_DIR / "aim_semantic_features_raw.csv")
    clin = pd.read_csv(ROOT / "public_data" / "nsclc_radiogenomics_clinical.csv")
    clin.columns = [c.strip() for c in clin.columns]

    df = aim.merge(clin, on="Case ID", how="inner")
    df = df[df["Recurrence"].isin(["yes", "no"])].copy()
    df["recurrence"] = (df["Recurrence"] == "yes").astype(int)

    # Clinical
    df["age"] = pd.to_numeric(df["Age at Histological Diagnosis"], errors="coerce")
    df["gender_male"] = (df["Gender"] == "Male").astype(int)
    df["ever_smoker"] = df["Smoking status"].isin(["Current", "Former"]).astype(int)
    df["pack_years"] = pd.to_numeric(df["Pack Years"], errors="coerce")

    # Molecular (unknown/not collected -> NaN, handled by imputer downstream)
    def mut_flag(col):
        return df[col].map({"Mutant": 1, "Wildtype": 0}).astype(float)

    df["egfr_mutant"] = mut_flag("EGFR mutation status")
    df["kras_mutant"] = mut_flag("KRAS mutation status")
    df["alk_positive"] = df["ALK translocation status"].map({"Mutant": 1, "Wildtype": 0}).astype(float)

    # Histology (for stratified sensitivity check only, not a primary predictor)
    df["histology_squamous"] = (df["Histology"] == "Squamous cell carcinoma").astype(int)

    keep_cols = (["Case ID", "recurrence", "age", "gender_male", "ever_smoker", "pack_years",
                   "egfr_mutant", "kras_mutant", "alk_positive", "histology_squamous"]
                 + CORE_SEMANTIC)
    final = df[keep_cols].copy()

    print(f"Final analysable n={len(final)}, events={final['recurrence'].sum()} "
          f"({final['recurrence'].mean()*100:.1f}%)")
    print("\nMissingness:")
    print((final.isna().mean() * 100).round(1).sort_values(ascending=False))

    out_path = OUT_DIR / "tcia_recurrence_analysis_ready.csv"
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""Real Table 1 baseline characteristics by NSCLC subtype (PLOS 2024, n=255)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public_data" / "candidate_blood_datasets" / "plos_2024_petct_baseline_s2_dataset_Baseline characteristics of patients.csv"
OUT = ROOT / "outputs" / "supplementary_experiments"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["histology"] = df["histology"].astype(int)
ade = df[df.histology == 0]
squ = df[df.histology == 1]

def cont(name, col, digits=1):
    a, b = pd.to_numeric(ade[col], errors="coerce"), pd.to_numeric(squ[col], errors="coerce")
    try:
        p = stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue
    except Exception:
        p = np.nan
    fmt = lambda s: f"{s.median():.{digits}f} ({s.quantile(.25):.{digits}f}–{s.quantile(.75):.{digits}f})"
    return [name, fmt(pd.to_numeric(df[col], errors='coerce')), fmt(a), fmt(b),
            ("<0.001" if p < 0.001 else f"{p:.3f}")]

def binary(name, col, positive=1):
    def pct(sub):
        v = pd.to_numeric(sub[col], errors="coerce")
        n = (v == positive).sum(); tot = v.notna().sum()
        return f"{int(n)} ({100*n/tot:.1f}%)"
    tab = pd.crosstab(df[col] == positive, df.histology)
    try:
        p = stats.chi2_contingency(tab)[1]
    except Exception:
        p = np.nan
    return [name, pct(df), pct(ade), pct(squ), ("<0.001" if p < 0.001 else f"{p:.3f}")]

rows = [["Characteristic", f"Overall (n={len(df)})", f"Adenocarcinoma (n={len(ade)})",
         f"Squamous (n={len(squ)})", "p"]]
rows.append(cont("Age, years", "age"))
rows.append(binary("Male sex", "gender", 1))
rows.append(cont("BMI, kg/m^2", "BMI"))
rows.append(cont("WBC, x10^9/L", "WBC", 2))
rows.append(cont("Neutrophils, x10^9/L", "NEU", 2))
rows.append(cont("Lymphocytes, x10^9/L", "LYM", 2))
rows.append(cont("Platelets, x10^9/L", "PLT", 0))
rows.append(cont("NLR", "NLR", 2))
rows.append(cont("dNLR", "dNLR", 2))
rows.append(cont("CEA, ng/mL", "CEA", 2))
rows.append(cont("LDH, U/L", "LDH", 0))
rows.append(cont("Albumin, g/L", "ALB", 1))
rows.append(cont("SUVmax", "SUVmax", 2))
rows.append(cont("SUVmean", "SUVmean", 2))
rows.append(cont("SUVmin", "SUVmin", 2))
rows.append(cont("MTV, cm^3", "MTV", 1))
rows.append(cont("TLG", "TLG", 1))

out = pd.DataFrame(rows[1:], columns=rows[0])
out.to_csv(OUT / "Table1_baseline_real.csv", index=False, encoding="utf-8-sig")
# markdown
md = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
for r in rows[1:]:
    md.append("| " + " | ".join(str(x) for x in r) + " |")
(OUT / "Table1_baseline_real.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("\nNote: continuous = median (IQR), Mann-Whitney; categorical = n (%), chi-square.")

"""Unbiased Hallmark-pathway GSEA on TCGA LUAD vs LUSC, testing all 50
MSigDB Hallmark gene sets on equal footing (not hand-picking which pathways
to report) -- upgrades the project's earlier ~15-20-curated-gene TCGA
analysis to a much less cherry-pickable form of evidence for the glycolysis
and neutrophil-inflammatory axes already used to motivate the primary
model's features and (in an earlier, retracted attempt) an MMFT architecture.
"""
from __future__ import annotations

import json
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "public_data" / "tcga_xena_lung_external"
OUT_DIR = ROOT / "outputs" / "PONE-D-26-33174_revision_20260919" / "analysis" / "tcga_primary_tumor_gsea"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# a priori axes already used in this project (§8), for cross-referencing
# against the unbiased ranking below -- NOT used to select which pathways
# are reported; all 50 are reported.
A_PRIORI_GLYCOLYSIS_HYPOXIA = ["Glycolysis", "Hypoxia"]
A_PRIORI_INFLAMMATORY = ["Inflammatory Response", "TNF-alpha Signaling via NF-kB",
                          "IL-6/JAK/STAT3 Signaling", "Complement", "Interferon Gamma Response",
                          "IL6 JAK STAT3 Signaling"]


def main():
    luad = pd.read_csv(DATA_DIR / "TCGA_LUAD_hallmark_genes_tpm.csv", index_col=0)
    lusc = pd.read_csv(DATA_DIR / "TCGA_LUSC_hallmark_genes_tpm.csv", index_col=0)
    # GDC/Xena cohort matrices also contain adjacent-normal (code 11) and,
    # occasionally, recurrent-tumor (code 02) aliquots. Restrict both cohorts
    # to primary tumors (TCGA sample-type code 01) so the pathway analysis uses
    # the same 528 LUAD and 501 LUSC tumors as the curated-gene analysis.
    luad = luad.loc[[idx for idx in luad.index if len(str(idx)) >= 15 and str(idx)[13:15] == "01"]]
    lusc = lusc.loc[[idx for idx in lusc.index if len(str(idx)) >= 15 and str(idx)[13:15] == "01"]]
    print(f"Primary tumors only -- LUAD: {luad.shape}, LUSC: {lusc.shape}")

    common_genes = [g for g in luad.columns if g in lusc.columns]
    luad = luad[common_genes]; lusc = lusc[common_genes]
    print(f"Common genes: {len(common_genes)}")

    # ---- differential expression (LUSC - LUAD), Mann-Whitney + Welch t-stat for ranking ----
    rows = []
    for gene in common_genes:
        a = luad[gene].dropna().to_numpy()
        s = lusc[gene].dropna().to_numpy()
        if len(a) < 10 or len(s) < 10:
            continue
        log2fc = float(np.mean(s) - np.mean(a))  # Xena star_tpm values are already log2(TPM+0.001)
        try:
            u_stat, p_mw = stats.mannwhitneyu(s, a, alternative="two-sided")
        except ValueError:
            p_mw = np.nan
        t_stat, p_t = stats.ttest_ind(s, a, equal_var=False)
        rows.append({"gene": gene, "log2FC_LUSC_minus_LUAD": log2fc, "t_stat": t_stat,
                     "p_ttest": p_t, "p_mannwhitney": p_mw, "n_LUAD": len(a), "n_LUSC": len(s)})
    de = pd.DataFrame(rows).dropna(subset=["p_ttest"])
    de["fdr"] = multipletests(de["p_ttest"], method="fdr_bh")[1]
    de = de.sort_values("t_stat", ascending=False)
    de.to_csv(OUT_DIR / "tcga_hallmark_differential_expression.csv", index=False, encoding="utf-8-sig")
    print(f"\nDifferential expression computed for {len(de)} genes. "
          f"Significant (FDR<0.05): {(de['fdr']<0.05).sum()}")
    print("\nTop 10 LUSC-high (unbiased ranking, top of full Hallmark-gene list):")
    print(de.head(10)[["gene", "log2FC_LUSC_minus_LUAD", "t_stat", "fdr"]].to_string())
    print("\nTop 10 LUAD-high:")
    print(de.tail(10)[["gene", "log2FC_LUSC_minus_LUAD", "t_stat", "fdr"]].to_string())

    # ---- GSEA prerank against ALL 50 Hallmark gene sets ----
    with open(ROOT / "tmp" / "hallmark_genesets.json") as f:
        gene_sets = json.load(f)
    print(f"\nRunning GSEA prerank against all {len(gene_sets)} Hallmark pathways...")

    rnk = de.set_index("gene")["t_stat"].sort_values(ascending=False)
    pre_res = gp.prerank(rnk=rnk, gene_sets=gene_sets, min_size=5, max_size=1000,
                          permutation_num=1000, outdir=None, seed=20260704, threads=4)
    res_df = pre_res.res2d.copy()
    res_df = res_df.sort_values("NES", ascending=False, key=lambda s: s.astype(float))
    res_df.to_csv(OUT_DIR / "tcga_hallmark_gsea_all50_results.csv", index=False, encoding="utf-8-sig")

    print("\n=== ALL 50 Hallmark pathways, ranked by NES (unbiased -- nothing cherry-picked) ===")
    print(res_df[["Term", "NES", "FDR q-val", "Lead_genes"]].to_string())

    # cross-reference a priori axes against the unbiased full ranking
    print("\n=== Where do our a priori glycolysis/hypoxia and inflammatory axes rank? ===")
    res_df["rank"] = range(1, len(res_df) + 1)
    for term in A_PRIORI_GLYCOLYSIS_HYPOXIA + A_PRIORI_INFLAMMATORY:
        match = res_df[res_df["Term"].str.contains(term, case=False, na=False)]
        if len(match):
            r = match.iloc[0]
            print(f"  {term}: rank {r['rank']}/{len(res_df)}, NES={r['NES']}, FDR q={r['FDR q-val']}")
        else:
            print(f"  {term}: not found in library term names (name mismatch)")

    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()

"""Fetch TCGA LUAD/LUSC expression for the union of all 50 MSigDB Hallmark
pathway member genes (unbiased genome-wide-adjacent GSEA prep) -- extends the
project's existing ~70-curated-gene TCGA analysis (scripts/
create_tcga_external_bioinfo_figures.py) to a much larger, hypothesis-neutral
gene set so that pathway enrichment is not restricted to genes we ourselves
selected to fit the narrative.

Honesty note: this fetches the ~4,383 genes belonging to the 50 Hallmark
gene sets, not the full ~20,000-gene coding transcriptome. GSEA run against
this list tests whether a pathway's genes rank higher/lower relative to OTHER
HALLMARK GENES, not relative to the entire genome. This is still dramatically
less biased than hand-picking ~15-20 genes ourselves (all 50 pathways compete
on equal footing, we do not choose which ones "win"), and is reported as such
-- not mis-described as literal genome-wide GSEA.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "public_data" / "tcga_xena_lung_external"
DATA_OUT.mkdir(parents=True, exist_ok=True)

HOST = xena.PUBLIC_HUBS["gdcHub"]
COHORTS = {"LUAD": "TCGA-LUAD.star_tpm.tsv", "LUSC": "TCGA-LUSC.star_tpm.tsv"}

with open(ROOT / "tmp" / "hallmark_union_genes.txt") as f:
    GENES = [g.strip() for g in f if g.strip()]
print(f"Fetching {len(GENES)} Hallmark-pathway genes for LUAD and LUSC...")


def call_with_retry(fn, *args, max_retries=8, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = min(30, 3 * (attempt + 1))
            print(f"      retry {attempt+1}/{max_retries} after error ({e.__class__.__name__}: {e}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def fetch_batch_with_retry(dataset, samples, batch, max_retries=8):
    return call_with_retry(xena.dataset_gene_probe_avg, HOST, dataset, samples, batch, max_retries=max_retries)


def fetch_cohort(cohort_name: str, dataset: str, genes: list[str], out_path: Path, batch_size: int = 200) -> pd.DataFrame:
    all_samples = call_with_retry(xena.dataset_samples, HOST, dataset, None)
    samples = [s for s in all_samples if len(s) >= 15 and s[13:15] == "01"]
    print(f"  {cohort_name}: {len(samples)} primary-tumor samples (from {len(all_samples)} total aliquots)")

    partial_path = out_path.with_suffix(".partial.csv")
    done_frames = []
    done_genes = set()
    if partial_path.exists():
        existing = pd.read_csv(partial_path, index_col=0)
        done_frames.append(existing)
        done_genes = set(existing.columns)
        print(f"    resuming: {len(done_genes)} genes already fetched from a previous partial run")

    remaining = [g for g in genes if g not in done_genes]
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        t0 = time.time()
        result = fetch_batch_with_retry(dataset, samples, batch)
        col_data = {}
        for r in result:
            scores = r["scores"][0] if r.get("scores") else None
            if scores is None or len(scores) != len(samples):
                print(f"      WARNING: gene {r['gene']} returned {len(scores) if scores else 0} scores, "
                      f"expected {len(samples)} -- filling with NaN")
                col_data[r["gene"]] = [np.nan] * len(samples)
            else:
                col_data[r["gene"]] = scores
        df_batch = pd.DataFrame(col_data, index=samples)
        done_frames.append(df_batch)
        print(f"    batch {i}-{i+len(batch)}/{len(remaining)} remaining fetched in {time.time()-t0:.1f}s")
        pd.concat(done_frames, axis=1).to_csv(partial_path, encoding="utf-8-sig")  # incremental save after every batch

    final = pd.concat(done_frames, axis=1)
    final.to_csv(out_path, encoding="utf-8-sig")
    partial_path.unlink(missing_ok=True)
    return final


def main():
    for cohort_name, dataset in COHORTS.items():
        out_path = DATA_OUT / f"TCGA_{cohort_name}_hallmark_genes_tpm.csv"
        if out_path.exists():
            print(f"{out_path} already exists, skipping")
            continue
        df = fetch_cohort(cohort_name, dataset, GENES, out_path)
        print(f"Saved {out_path}, shape={df.shape}")


if __name__ == "__main__":
    main()

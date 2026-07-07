# NSCLC MMFT Benchmark

Analysis code for:

> Tao Z, Zhang Y, Fan G. **Regularized Linear Models Match Multimodal Deep Learning for
> NSCLC Histological-Subtype Prediction: A Leakage-Free, Two-Cohort, Three-Modality
> Benchmark With Transcriptomic Validation.** (submitted to PLOS ONE).

The study asks whether high-dimensional radiomics or multimodal deep learning
outperform a well-regularized linear model when predicting lung adenocarcinoma vs.
squamous-cell carcinoma from pretreatment clinical, blood-inflammatory, and
¹⁸F-FDG PET-metabolic data at realistic sample size (n≈255). Under strict leakage-free
model selection, across two independent public cohorts and three structurally
distinct feature modalities, ridge-logistic regression was never outperformed by
deep learning. The repository also includes the TCGA transcriptomic validation
(including an unbiased 50-pathway GSEA), the TCIA independent-cohort replication,
and the clinical nomogram / integer-score translation reported in the paper.

## What this repository contains

This repository holds **analysis code only**. No patient data, derived result
tables, or figures are included — all data used in the study are already public
(see *Data availability* below), and the derived result tables underlying every
figure/table are provided as Supporting Information (S1 Data) alongside the
manuscript itself.

```
scripts/    All Python and R analysis scripts (see "Script guide" below)
tests/      Unit tests for the MMFT transformer components
```

## Data availability

None of the raw data is redistributed here. To reproduce the analyses, download:

1. **Primary cohort** (clinical, blood, PET-metabolic, CT/PET radiomics; n=255) —
   from the Supporting Information of Zhang Y, Liu H, Chang C, Yin Y, Wang R.
   *PLoS ONE.* 2024;19(4):e0300170. https://doi.org/10.1371/journal.pone.0300170
2. **TCGA LUAD/LUSC transcriptomics** — via the UCSC Xena GDC hub
   (https://xenabrowser.net).
3. **TCIA NSCLC-Radiogenomics** — via The Cancer Imaging Archive
   (https://www.cancerimagingarchive.net), including the AIM v4 radiologist
   semantic annotations and EGFR/KRAS mutation status.

The scripts expect a project layout with sibling data directories at the repo
root — `public_data/`, `server_results/`, and `outputs/` — mirroring the paths
referenced inside each script (most scripts resolve paths relative to
`Path(__file__).resolve().parents[1]`). Recreate this layout locally after
downloading the public data above; scripts will not run standalone without it.

## Script guide (by paper section)

| Purpose | Scripts |
|---|---|
| Primary benchmark (ridge, LASSO, SVM, RF, XGBoost, LightGBM, FT-Transformer, MMFT) | `benchmark_petct_blood_models.py`, `train_ft_transformer_petct.py`, `train_mmft_transformer_petct.py`, `final_mmft_rank_refit.py` |
| Model-optimization / estimate-stability analyses (repeated hold-out, elastic-net, radiomics signature, stacking, recalibration, learning curve, subgroups) | `model_optimization.py`, `supplementary_experiments.py` |
| Leakage-free MMFT architecture/hyperparameter search | `mmft_architecture_search.py`, `mmft_stability_comparison.py` |
| TCGA-guided mechanistic-token idea (incl. bug-postmortem redo) | `mmft_biology_guided.py`, `mmft_named_tokens.py`, `mmft_search_with_biology_tokens.py`, `mmft_token_ablation.py` |
| TabPFN comparator | `tabpfn_comparator.py` |
| TCGA transcriptomic biology + unbiased 50-pathway GSEA | `fetch_tcga_hallmark_expression.py`, `tcga_hallmark_gsea.py` |
| TCIA independent-cohort replication (baseline associations, EGFR/KRAS) | `independent_cohort_plausibility.py` |
| TCIA CT semantic-phenotype benchmark (histology + recurrence) | `extract_aim_features.py`, `parse_aim_semantic_features.py`, `tcia_recurrence_data_prep.py`, `tcia_histology_classical_benchmark.py`, `tcia_histology_mmft.py`, `tcia_recurrence_classical_benchmark.py` |
| Clinical nomogram / simplified integer score | `create_nomogram.R` |
| Baseline table, public-model screening | `make_table1_baseline.py`, `analyze_public_model.R` |
| Figure generation | `create_supplementary_figures.py`, `create_mmft_shap_figures.py`, `create_tcga_external_bioinfo_figures.py`, `reproduce_reference_style_figures.py`, `make_*_figure.py`, `label_gptimage2_*.py` |
| Manuscript/submission-package assembly | `md_docx_render.py`, `build_final_docs.py`, `create_plos_one_submission_package.py`, `create_tripod_ai_checklist.py`, `create_*_docx.py` |

## Dependencies

Python (see `requirements.txt`): numpy, pandas, scipy, scikit-learn, statsmodels,
torch, xgboost, lightgbm, shap, gseapy, lifelines, matplotlib, seaborn,
python-docx, Pillow, xenaPython.

R: `rms`, `pROC` (used only by `create_nomogram.R` / `analyze_public_model.R`).

## License

MIT — see `LICENSE`.

## Citing

If you use this code, please cite the paper above. Please also cite the primary
data source (Zhang et al., PLoS ONE 2024) and TCGA/TCIA as appropriate for the
data you use.

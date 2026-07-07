"""Assemble the 4 final deliverables into polished .docx with embedded figures.

Reads content markdown from outputs/_content/{manuscript,roadmap,wetlab,budget}.md
(written from the drafting workflow results) and renders each to outputs/final/.
Also embeds the real figures as a gallery so every document is 图文并茂.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from md_docx_render import new_document, render_markdown, add_figure  # noqa: E402
from docx.shared import Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs" / "reference_style_reproduction"
GPT = ROOT / "outputs" / "gptimage2_diagrams"
SUPP = ROOT / "outputs" / "supplementary_figures"
SERVER = ROOT / "server_results" / "petct_blood_benchmark"
CONTENT = ROOT / "outputs" / "_content"
FINAL = ROOT / "outputs" / "final"
FINAL.mkdir(parents=True, exist_ok=True)

# ---- figure galleries (label, path, caption, one-line takeaway) ----
MANUSCRIPT_FIGS = [
    ("Figure 1", GPT / "Figure7_GPTimage2_research_workflow_final.png",
     "Figure 1. Study overview and analytical workflow.",
     "Public PET/CT+blood cohort for model development, TCGA (incl. unbiased GSEA) for biological validation, TCIA for independent-cohort and third-modality replication."),
    ("Figure 2", REF / "Figure1_reference_style_performance.png",
     "Figure 2. Model performance on the public hold-out test set (ROC, cross-validation stability, confusion matrices, risk distribution, threshold metrics).",
     "Primary ridge model AUC 0.854."),
    ("Figure 3", SUPP / "FigS7_incremental_value_and_honest_benchmark.png",
     "Figure 3. Incremental value of modalities, honest model benchmarking, unbiased nested CV, and calibration/decision-curve analysis.",
     "PET metabolic parameters drive the gain (DeLong p=0.049); radiomics hurts (p=0.045); deep learning is not superior to ridge (p=0.37); nested-CV AUC 0.778."),
    ("Figure 4", REF / "Figure3_reference_style_case_interpretability.png",
     "Figure 4. Case-level interpretability (SHAP) of the multimodal model.",
     "PET metabolic burden, CEA and the fused risk token dominate individual predictions."),
    ("Figure 5", REF / "Figure8_TCGA_external_bioinfo_validation.png",
     "Figure 5. TCGA-LUAD/LUSC transcriptomic biological validation of model-linked markers.",
     "Squamous tumors are more glycolytic (SLC2A1/HK2/LDHA ↔ SUV) and show a squamous-high neutrophil-chemotaxis signature (S100A8/9, CXCL8 ↔ NLR); adenocarcinoma retains CEACAM5/TTF-1."),
    ("Figure 6", SUPP / "Figure6_model_optimization.png",
     "Figure 6. Model-optimization attempts and estimate stability.",
     "Repeated hold-out converges with nested-CV at AUC~0.78; elastic-net, radiomics signature and stacking all fail to beat ridge; learning curve shows no plateau."),
    ("Figure 7", SUPP / "Figure7_independent_cohort_replication.png",
     "Figure 7. Independent-cohort (TCIA NSCLC-Radiogenomics) replication of subtype-associated biology.",
     "Male-predominance in squamous replicates across two independent cohorts (p=0.033); EGFR/KRAS mutations are essentially exclusive to adenocarcinoma (0% in squamous, p=0.007/0.026)."),
    ("Figure 8", SERVER / "primary_model_nomogram.png",
     "Figure 8. Clinical nomogram for the primary model.",
     "Points-based bedside tool from the 5 primary variables; a simplified integer score built the same way reproduces the model's AUC (0.858 vs 0.855) without software."),
    ("Figure 9", SUPP / "Figure9_cross_cohort_dl_vs_ridge.png",
     "Figure 9. Deep learning vs ridge: consistent across cohorts, modalities, and methods.",
     "2 independent cohorts x 3 feature modalities x 4 DL/optimization strategies, always the same answer: deep learning does not exceed the simple model once evaluated without leakage."),
    ("Figure 10", SUPP / "Figure10_gsea_all50_hallmark.png",
     "Figure 10. Unbiased GSEA across all 50 MSigDB Hallmark pathways (LUSC vs LUAD, TCGA).",
     "Glycolysis/Hypoxia trend squamous-high but miss pathway-level significance; Interferon Gamma Response, Complement and Allograft Rejection are significantly adenocarcinoma-high, refining the curated neutrophil-chemotaxis claim; proliferation pathways are an unhypothesised squamous-high finding."),
]

SUPP_FIGS_MS = [
    ("Figure S1", SUPP / "FigS1_data_preprocessing_missingness.png", "Figure S1. Data preprocessing, missingness and split.", None),
    ("Figure S2", SUPP / "FigS2_model_ablation.png", "Figure S2. Model/feature-block ablation.", None),
    ("Figure S3", SUPP / "FigS3_TCGA_GEO_key_gene_direction_heatmap.png", "Figure S3. TCGA key-gene direction heatmap.", None),
    ("Figure S4", SUPP / "FigS4_calibration_DCA.png", "Figure S4. Calibration and decision-curve analysis (detailed).", None),
    ("Figure S5", SUPP / "FigS5_SHAP_supplement_composite.png", "Figure S5. SHAP interpretability supplement.", None),
]

ROADMAP_FIGS = [
    ("图1 研究总流程", GPT / "Figure7_GPTimage2_research_workflow_final.png",
     "图1. 研究总体思路与技术路线", "公共队列建模、TCGA 做生物学解释（含无偏倚GSEA自查）、TCIA 独立队列+第三模态复现；全篇基于公开数据。"),
    ("图2 队列纳排", REF / "Figure6_reference_style_participant_flow.png",
     "图2. 队列与纳排流程", "255 例公共队列建模+内部测试；TCGA 干实验；TCIA 独立队列复现，均为公开数据。"),
    ("图3 模型性能", REF / "Figure1_reference_style_performance.png",
     "图3. 模型总体性能", "主模型 Ridge AUC 0.854。"),
    ("图4 增量价值与诚实基准", SUPP / "FigS7_incremental_value_and_honest_benchmark.png",
     "图4. 增量价值 / 诚实基准 / 无偏交叉验证 / 校准", "PET 代谢驱动增量(p=0.049)；radiomics 反而变差(p=0.045)；深度学习不优于Ridge(p=0.37)；nested-CV 0.778。"),
    ("图5 模型优化与稳健性", SUPP / "Figure6_model_optimization.png",
     "图5. 模型优化尝试与结果稳健性", "50次重复留出验证均值0.778，与nested-CV完全吻合；弹性网/radiomics signature/ridge+XGB集成三种优化均未超过简单模型；学习曲线未见平台，说明更大样本量原则上能带来真实提升（本研究未采集额外队列）。"),
    ("图6 可解释性", REF / "Figure3_reference_style_case_interpretability.png",
     "图6. 个体层面可解释性 (SHAP)", "PET 代谢、CEA 与融合风险 token 主导预测，模型不是黑箱。"),
    ("图7 TCGA 生物学验证", REF / "Figure8_TCGA_external_bioinfo_validation.png",
     "图7. TCGA 转录组生物学验证", "鳞癌更糖酵解(GLUT1/HK2/LDHA)、中性粒趋化基因更高(S100A8/9,CXCL8)，解释了为什么 SUV+血指标能反映亚型；图11无偏倚通路检验对该炎症轴做了细化。"),
    ("图8 独立队列复现（新增深化内容）", SUPP / "Figure7_independent_cohort_replication.png",
     "图8. 独立队列（TCIA NSCLC-Radiogenomics）复现",
     "与主队列完全独立的美国队列：性别关联方向一致(p=0.033)；EGFR/KRAS 突变几乎只见于腺癌(0% vs 0%，p=0.007/0.026)——多队列证据链，不是单一数据集的偶然发现。"),
    ("图9 诺莫图/床边评分（新增深化内容）", SERVER / "primary_model_nomogram.png",
     "图9. 临床诺莫图与床边整数评分表",
     "把5变量模型转成医生能直接用的打分工具；简化整数评分AUC 0.858，跟精确模型0.855几乎一致（用于验证换算正确）。"),
    ("图10 跨队列跨模态终极验证（核心新图）", SUPP / "Figure9_cross_cohort_dl_vs_ridge.png",
     "图10. 深度学习 vs 岭回归：跨队列、跨模态、跨方法始终一致（本次核心成果）",
     "2个独立队列 x 3种结构完全不同的特征模态(PET代谢+血液/CT影像组学/CT语义读片) x 4种深度学习或前沿方法，结论完全一致：深度学习从未打败简单模型。这是全篇最有分量的方法学证据。"),
    ("图11 无偏倚GSEA全通路富集（新增深化内容）", SUPP / "Figure10_gsea_all50_hallmark.png",
     "图11. 无偏倚GSEA：50条Hallmark通路全部纳入检验",
     "糖酵解/缺氧通路方向正确但通路层面未达显著(FDR q=0.29/0.16)；干扰素γ应答、补体、同种异体排斥这3条适应性免疫通路反而在腺癌显著更高，修正（而非推翻）了中性粒炎症轴的说法；增殖/细胞周期通路(E2F、G2-M、Myc、mTORC1)在鳞癌显著更高，是一个未预设、无偏倚筛选出的新发现。"),
    ("图12 临床路径", REF / "Figure5_reference_style_clinical_pathway.png",
     "图12. 临床落地路径", "术前风险分层的决策支持工具，不替代病理；需通过本院验证门槛。"),
]

WETLAB_FIGS = [
    ("图 TCGA 关键基因", SUPP / "FigS3_TCGA_GEO_key_gene_direction_heatmap.png",
     "附图. TCGA 关键基因方向热图（湿实验验证靶点来源）",
     "湿实验 IHC/qPCR 的靶点即来自此处：糖酵解、亚型标志、中性粒炎症三条轴。"),
    ("图 医院验证模板", SUPP / "FigS6_hospital_validation_template.png",
     "附图. 医院外部验证结果模板（待数据填充）",
     "医院队列数据到位后，直接替换为真实外部验证 ROC/校准/DCA。"),
]


def build(md_path, out_docx, title, subtitle, figs=None, supp_figs=None,
          gallery_heading="配图 / Figures", landscape=False):
    doc = new_document(landscape=landscape)
    # cover title
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from md_docx_render import _add_inline, BLUE, MUTED
    tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _add_inline(tp, title, size=19, base_bold=True, color=BLUE)
    if subtitle:
        sp = doc.add_paragraph(); _add_inline(sp, subtitle, size=11, color=MUTED)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    text = Path(md_path).read_text(encoding="utf-8")
    render_markdown(doc, text, base_dir=ROOT)

    if figs:
        doc.add_page_break()
        hp = doc.add_paragraph(); _add_inline(hp, gallery_heading, size=15, base_bold=True, color=BLUE)
        doc.add_paragraph()
        for _, path, cap, take in figs:
            add_figure(doc, path, caption=cap, width_in=6.5, takeaway=take)
    if supp_figs:
        doc.add_paragraph()
        hp = doc.add_paragraph(); _add_inline(hp, "补充图 / Supplementary figures", size=13, base_bold=True, color=BLUE)
        for _, path, cap, take in supp_figs:
            add_figure(doc, path, caption=cap, width_in=6.3, takeaway=take)

    doc.save(out_docx)
    print("saved", out_docx)


def main():
    # wetlab.md / budget.md described a planned retrospective hospital data-collection
    # phase. That phase was called off (no institutional cohort will be collected), so
    # those two documents are no longer part of the deliverable set. Their content and
    # figure lists (WETLAB_FIGS) are left on disk for reference but are not built here;
    # pass build_wetlab=True to reinstate them if plans change.
    jobs = [
        ("manuscript.md", FINAL / "Manuscript_NSCLC_multimodal_subtype.docx",
         "Regularized linear models match multimodal deep learning for NSCLC histological-subtype prediction",
         "A leakage-free, two-cohort, three-modality benchmark with transcriptomic validation (public data, n=255)",
         MANUSCRIPT_FIGS, SUPP_FIGS_MS, "Figures", False),
        ("roadmap.md", FINAL / "老板汇报_图文思路表_NSCLC.docx",
         "NSCLC 病理亚型多模态预测 · 图文思路总览（导师汇报版）",
         "术前 临床+血液炎症+PET代谢 预测腺癌/鳞癌 · 全篇基于公开数据，已定稿",
         ROADMAP_FIGS, None, "配图 / Figures", False),
    ]
    build_wetlab = False
    if build_wetlab:
        jobs += [
            ("wetlab.md", FINAL / "湿实验详细方案_NSCLC.docx",
             "NSCLC 项目 湿实验（临床队列 + 组织生物学验证）详细方案 / SOP",
             "医院前瞻队列采集 SOP + IHC / 多重免疫荧光 / RT-qPCR 组织验证",
             WETLAB_FIGS, None, "配图 / Figures", False),
            ("budget.md", FINAL / "预算表_NSCLC.docx",
             "NSCLC 项目 预算表（湿实验 + 分析 + 发表）",
             "精简版（必须）与完整版（含可选测序/多重免疫荧光）· 金额为市场估算",
             None, None, "", False),
        ]
    for md_name, out, title, sub, figs, supp, gh, land in jobs:
        md_path = CONTENT / md_name
        if not md_path.exists():
            print("SKIP (missing content):", md_name)
            continue
        build(md_path, out, title, sub, figs=figs, supp_figs=supp, gallery_heading=gh, landscape=land)


if __name__ == "__main__":
    main()

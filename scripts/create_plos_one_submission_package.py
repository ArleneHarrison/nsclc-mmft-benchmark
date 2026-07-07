from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PKG = OUT / "plos_one_submission_package"

TITLE = (
    "Regularized Linear Models Match Multimodal Deep Learning for NSCLC "
    "Histological-Subtype Prediction: A Leakage-Free, Two-Cohort, "
    "Three-Modality Benchmark With Transcriptomic Validation"
)
SHORT_TITLE = "Two-cohort, three-modality benchmark for NSCLC subtype"
AUTHORS = "Zhiyong Tao1, Yiwei Zhang1, Guohua Fan1*"
AFFILIATION = "1 Renmin Hospital of Wuhan University, Wuhan, Hubei, China"
CORRESPONDING = (
    "* Corresponding author: Guohua Fan, Renmin Hospital of Wuhan University, "
    "Zhangzhidong Road 99 / Jiefang Road 238, Wuchang District, Wuhan, Hubei "
    "Province, China. Email: fgh202604@163.com"
)
ORCIDS = (
    "ORCID iDs: Zhiyong Tao, 0009-0001-2454-2432; "
    "Yiwei Zhang, 0009-0008-7513-7956; "
    "Guohua Fan, 0009-0005-7470-7863"
)
KEYWORDS = (
    "non-small cell lung cancer; histological subtype; 18F-FDG PET/CT; "
    "leakage-free benchmarking; deep learning; clinical prediction model; "
    "multi-cohort replication"
)

DATA_AVAILABILITY = (
    "All data underlying the findings are publicly available without "
    "restriction. The primary clinical/PET-CT dataset is available from the "
    "supporting information of Zhang et al. (PLOS ONE 2024, "
    "doi:10.1371/journal.pone.0300170). TCGA LUAD/LUSC transcriptomic data "
    "are available from the UCSC Xena GDC hub. TCIA NSCLC-Radiogenomics data "
    "are available from The Cancer Imaging Archive. Derived source tables used "
    "to generate the figures, tables, and supplementary analyses are included "
    "in S1 Data in this submission package."
)
FUNDING = "The authors received no specific funding for this work."
COMPETING = "The authors have declared that no competing interests exist."
ETHICS = (
    "This study used only publicly available, de-identified data. No "
    "institutional patient data were collected or analysed; no additional "
    "ethics approval or data-use agreement was required beyond the public "
    "repositories' terms of use."
)
AUTHOR_CONTRIBUTIONS = (
    "Zhiyong Tao: Conceptualization, Data curation, Formal analysis, "
    "Investigation, Methodology, Software, Visualization, Writing - original "
    "draft. Yiwei Zhang: Data curation, Validation, Methodology, Writing - "
    "review & editing. Guohua Fan: Conceptualization, Supervision, Project "
    "administration, Writing - review & editing."
)
AI_STATEMENT = (
    "AI-assisted tools (OpenAI ChatGPT/Codex and Anthropic Claude) were used "
    "for code-drafting/debugging support, language polishing, and document "
    "preparation. These tools were not used to generate or alter the source "
    "data. The authors reviewed and verified the analyses, figures, "
    "interpretations, citations, and final manuscript, and take full "
    "responsibility for the content."
)

MAIN_FIGURES = [
    ("Fig1", OUT / "reference_style_reproduction" / "Figure7_reference_style_workflow_programmatic.png"),
    ("Fig2", OUT / "reference_style_reproduction" / "Figure1_reference_style_performance.png"),
    ("Fig3", OUT / "supplementary_figures" / "FigS7_incremental_value_and_honest_benchmark.png"),
    ("Fig4", OUT / "reference_style_reproduction" / "Figure3_reference_style_case_interpretability.png"),
    ("Fig5", OUT / "reference_style_reproduction" / "Figure8_TCGA_external_bioinfo_validation.png"),
    ("Fig6", OUT / "supplementary_figures" / "Figure6_model_optimization.png"),
    ("Fig7", OUT / "supplementary_figures" / "Figure7_independent_cohort_replication.png"),
    ("Fig8", ROOT / "server_results" / "petct_blood_benchmark" / "primary_model_nomogram.png"),
    ("Fig9", OUT / "supplementary_figures" / "Figure9_cross_cohort_dl_vs_ridge.png"),
    ("Fig10", OUT / "supplementary_figures" / "Figure10_gsea_all50_hallmark.png"),
]

SUPPLEMENTARY_FIGURES = [
    ("S1_Fig", OUT / "supplementary_figures" / "FigS1_data_preprocessing_missingness.png"),
    ("S2_Fig", OUT / "supplementary_figures" / "FigS2_model_ablation.png"),
    ("S3_Fig", OUT / "supplementary_figures" / "FigS3_TCGA_GEO_key_gene_direction_heatmap.png"),
    ("S4_Fig", OUT / "supplementary_figures" / "FigS4_calibration_DCA.png"),
    ("S5_Fig", OUT / "supplementary_figures" / "FigS5_SHAP_supplement_composite.png"),
]

SUPPLEMENTARY_LEGENDS = [
    (
        "S1 Fig. Data preprocessing, missingness, and train/test split.",
        "Variable-level missingness, leakage-controlled preprocessing order, and the stratified 70/30 split used for the primary public cohort analysis.",
    ),
    (
        "S2 Fig. Model and feature-block ablation.",
        "Comparison of clinical-blood, PET metabolic, radiomics, risk-token, and ranking-loss components to show which parts add signal and which parts do not.",
    ),
    (
        "S3 Fig. TCGA key-gene direction heatmap.",
        "External transcriptomic check of the direction of subtype-associated genes, including glycolysis/hypoxia and neutrophil-chemotaxis markers.",
    ),
    (
        "S4 Fig. Calibration and decision-curve analysis.",
        "Detailed calibration and net-benefit curves for the ridge model and MMFT-related comparisons.",
    ),
    (
        "S5 Fig. SHAP interpretability supplement.",
        "Composite SHAP visualization showing global feature importance and representative case-level explanations for the multimodal model.",
    ),
]

SOURCE_DATA_PATTERNS = [
    OUT / "reference_style_reproduction",
    OUT / "supplementary_figures",
    OUT / "supplementary_experiments",
    OUT / "model_optimization",
]
SOURCE_DATA_EXCLUDE_TOKENS = {
    "template",
    "simulated_not_for_inference",
    "not_for_inference",
    "reproduction_manifest.json",
    "supplementary_manifest.json",
}


def clear_output_dir() -> None:
    pkg_resolved = PKG.resolve()
    root_resolved = ROOT.resolve()
    if PKG.exists():
        if root_resolved not in pkg_resolved.parents:
            raise RuntimeError(f"Refusing to delete outside project root: {pkg_resolved}")
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True, exist_ok=True)


def set_para_text(para: Paragraph, text: str, *, bold: bool = False, size: int | None = None) -> None:
    para.clear()
    run = para.add_run(text)
    run.bold = bold
    if size is not None:
        run.font.size = Pt(size)


def insert_paragraph_before(paragraph: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addprevious(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def delete_paragraph(paragraph: Paragraph) -> None:
    p = paragraph._element
    p.getparent().remove(p)
    paragraph._p = paragraph._element = None


def strip_unreferenced_media(docx_path: Path) -> None:
    """Remove media files left over after deleting figure paragraphs."""
    tmp_path = docx_path.with_suffix(".cleaning.tmp.docx")
    rels_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    ct_ns = {"ct": "http://schemas.openxmlformats.org/package/2006/content-types"}
    ET.register_namespace("", rels_ns["rel"])
    ET.register_namespace("", ct_ns["ct"])

    with zipfile.ZipFile(docx_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith("word/media/"):
                continue
            data = zin.read(name)
            if name == "word/_rels/document.xml.rels":
                root = ET.fromstring(data)
                for rel in list(root):
                    target = rel.attrib.get("Target", "")
                    if target.startswith("media/"):
                        root.remove(rel)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            elif name == "[Content_Types].xml":
                root = ET.fromstring(data)
                for child in list(root):
                    part_name = child.attrib.get("PartName", "")
                    if part_name.startswith("/word/media/"):
                        root.remove(child)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            zout.writestr(item, data)

    tmp_path.replace(docx_path)


def format_section_heading(para: Paragraph) -> None:
    para.runs[0].bold = True
    para.runs[0].font.size = Pt(12)
    para.paragraph_format.space_before = Pt(10)
    para.paragraph_format.space_after = Pt(4)


def fix_refs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return "[" + match.group(1).replace(" ", "") + "]"

    return re.sub(r"\[refs?\s+([0-9,\-–\sand]+)\]", repl, text)


def make_plos_manuscript() -> Path:
    src = OUT / "final" / "Manuscript_NSCLC_multimodal_subtype.docx"
    dst_dir = PKG / "manuscript"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "PLOS_ONE_Manuscript_NSCLC_MMFT.docx"

    shutil.copy2(src, dst)
    doc = Document(dst)

    first_page = [
        TITLE,
        AUTHORS,
        AFFILIATION,
        CORRESPONDING,
        ORCIDS,
        f"Short title: {SHORT_TITLE}",
        f"Keywords: {KEYWORDS}",
        "Article type: Research Article",
        "",
    ]
    for idx, value in enumerate(first_page):
        set_para_text(doc.paragraphs[idx], value, bold=(idx == 0), size=(14 if idx == 0 else None))
        if idx == 0:
            doc.paragraphs[idx].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for para in doc.paragraphs:
        if "[ref" in para.text or "[refs" in para.text:
            set_para_text(para, fix_refs(para.text))
        if "补充图 / Supplementary figures" in para.text:
            set_para_text(para, "Supplementary figures")
        if "Analysis code and derived result tables are available in the study repository." in para.text:
            set_para_text(
                para,
                para.text.replace(
                    "Analysis code and derived result tables are available in the study repository.",
                    "Derived source tables used for figures and supplementary analyses are provided as Supporting Information (S1 Data).",
                ),
            )

    for para in list(doc.paragraphs):
        if "All citations should be checked against the original source" in para.text:
            delete_paragraph(para)

    ref_para = next(p for p in doc.paragraphs if p.text.strip() == "References")
    declarations = [
        ("Data Availability Statement", True),
        (DATA_AVAILABILITY, False),
        ("Funding", True),
        (FUNDING, False),
        ("Competing Interests", True),
        (COMPETING, False),
        ("Ethics Statement", True),
        (ETHICS, False),
        ("Author Contributions", True),
        (AUTHOR_CONTRIBUTIONS, False),
        ("Use of AI-assisted tools", True),
        (AI_STATEMENT, False),
        ("Acknowledgments", True),
        ("None.", False),
        ("", False),
    ]
    for text, is_heading in declarations:
        para = insert_paragraph_before(ref_para, text)
        if text and is_heading:
            format_section_heading(para)

    for para in reversed(doc.paragraphs):
        if para._element is not None and para._element.xpath(".//w:drawing") and not para.text.strip():
            delete_paragraph(para)

    doc.save(dst)
    return dst


def style_doc(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for para in doc.paragraphs:
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.line_spacing = 1.1


def add_heading(doc: Document, text: str, level: int = 1) -> Paragraph:
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = True
    run.font.size = Pt(16 if level == 1 else 13)
    run.font.color.rgb = None
    para.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    para.paragraph_format.space_after = Pt(4)
    return para


def write_cover_letter() -> Path:
    dst = PKG / "cover_letter" / "Cover_Letter_PLOS_ONE.docx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_doc(doc)
    doc.add_paragraph("Dear PLOS ONE Editors,")
    doc.add_paragraph()
    doc.add_paragraph(
        "We are pleased to submit our manuscript entitled "
        f'"{TITLE}" for consideration as a Research Article in PLOS ONE.'
    )
    doc.add_paragraph(
        "This study uses only public, de-identified datasets to evaluate whether "
        "multimodal deep learning meaningfully outperforms regularized linear "
        "models for NSCLC adenocarcinoma-versus-squamous subtype prediction at "
        "a realistic oncology sample size. The main contribution is a transparent, "
        "leakage-controlled benchmark with TCGA transcriptomic validation and "
        "TCIA independent-cohort plausibility checks, including negative results "
        "for radiomics and deep-learning optimization attempts."
    )
    doc.add_paragraph(
        "The work is aligned with PLOS ONE's emphasis on methodological soundness "
        "and transparency. We report both optimistic and conservative performance "
        "estimates, provide calibration and decision-curve analyses, disclose "
        "limitations plainly, and avoid claiming clinical deployment readiness "
        "because external institutional validation has not yet been performed."
    )
    doc.add_paragraph(
        "All data used in the study are publicly available, and derived source "
        "tables are provided as supporting information. No institutional patient "
        "data were collected or analysed. The authors declare no competing "
        "interests and no specific funding for this work. The manuscript is not "
        "under consideration elsewhere and has not been published previously."
    )
    doc.add_paragraph("Correspondence should be addressed to:")
    doc.add_paragraph("Guohua Fan, Renmin Hospital of Wuhan University")
    doc.add_paragraph("Email: fgh202604@163.com")
    doc.add_paragraph()
    doc.add_paragraph("Sincerely,")
    doc.add_paragraph("Zhiyong Tao, Yiwei Zhang, and Guohua Fan")
    doc.save(dst)
    return dst


def write_submission_metadata() -> Path:
    dst = PKG / "submission_fields" / "Submission_Metadata_and_Required_Statements.docx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_doc(doc)
    title = doc.add_paragraph()
    run = title.add_run("PLOS ONE Submission Metadata and Required Statements")
    run.bold = True
    run.font.size = Pt(16)

    for heading, body in [
        ("Manuscript title", TITLE),
        ("Short title", SHORT_TITLE),
        ("Authors", AUTHORS),
        ("Affiliation", AFFILIATION),
        ("Corresponding author", CORRESPONDING),
        ("ORCID iDs", ORCIDS),
        ("Keywords", KEYWORDS),
        ("Data Availability Statement", DATA_AVAILABILITY),
        ("Funding Statement", FUNDING),
        ("Competing Interests Statement", COMPETING),
        ("Ethics Statement", ETHICS),
        ("Author Contributions (CRediT)", AUTHOR_CONTRIBUTIONS),
        ("Use of AI-assisted tools", AI_STATEMENT),
    ]:
        add_heading(doc, heading, 2)
        doc.add_paragraph(body)

    add_heading(doc, "Upload file list", 2)
    for text in [
        "Main manuscript: manuscript/PLOS_ONE_Manuscript_NSCLC_MMFT.docx",
        "Cover letter: cover_letter/Cover_Letter_PLOS_ONE.docx",
        "Main figures: figures/main_figures_tiff/Fig1.tif through Fig10.tif",
        "Supporting figures: figures/supplementary_figures_tiff/S1_Fig.tif through S5_Fig.tif",
        "Supporting information legends: supporting_information/Supporting_Information_Legends.docx",
        "Derived source data: data_availability/S1_Data_derived_results.zip",
    ]:
        doc.add_paragraph(text, style=None)

    doc.save(dst)
    return dst


def write_supporting_legends() -> Path:
    dst = PKG / "supporting_information" / "Supporting_Information_Legends.docx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_doc(doc)
    title = doc.add_paragraph()
    run = title.add_run("Supporting Information Legends")
    run.bold = True
    run.font.size = Pt(16)
    for legend, detail in SUPPLEMENTARY_LEGENDS:
        add_heading(doc, legend, 2)
        doc.add_paragraph(detail)
    doc.save(dst)
    return dst


def write_readme() -> Path:
    dst = PKG / "00_README_投稿说明.docx"
    doc = Document()
    style_doc(doc)
    title = doc.add_paragraph()
    run = title.add_run("PLOS ONE 投稿包说明")
    run.bold = True
    run.font.size = Pt(16)
    doc.add_paragraph(
        "这个文件夹按 PLOS ONE 投稿时的上传逻辑整理：主文稿、cover letter、投稿系统字段、主图、补充图和源数据分开存放。"
    )
    add_heading(doc, "推荐上传顺序", 2)
    for item in [
        "1. 在投稿系统填写题目、短题目、作者、单位、通讯作者和 ORCID。",
        "2. 上传 manuscript/PLOS_ONE_Manuscript_NSCLC_MMFT.docx 作为主文稿。",
        "3. 上传 cover_letter/Cover_Letter_PLOS_ONE.docx 作为 cover letter。",
        "4. 主图上传 figures/main_figures_tiff/Fig1.tif 至 Fig10.tif。",
        "5. 补充图上传 figures/supplementary_figures_tiff/S1_Fig.tif 至 S5_Fig.tif。",
        "6. 上传 supporting_information/Supporting_Information_Legends.docx 和 data_availability/S1_Data_derived_results.zip。",
    ]:
        doc.add_paragraph(item)
    add_heading(doc, "提交前需要本人确认", 2)
    for item in [
        "作者英文名拼写，尤其是第二作者 Yiwei Zhang 是否为正确拼音。",
        "基金信息是否确实为无专项基金；如有课题号，需要替换 Funding Statement。",
        "利益冲突是否确实为无。",
        "是否愿意在文稿和系统中披露 OpenAI/Codex 与 Claude 的辅助使用。",
        "通讯作者邮箱、ORCID、单位地址是否完全准确。",
    ]:
        doc.add_paragraph(item)
    add_heading(doc, "已排除内容", 2)
    doc.add_paragraph(
        "FigS6_hospital_validation_template.png 是待本院外部验证数据补齐后的模板图，不应作为当前投稿材料上传。"
    )
    doc.save(dst)
    return dst


def write_text_statements() -> None:
    folder = PKG / "data_availability"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Data_Availability_Statement.txt").write_text(DATA_AVAILABILITY, encoding="utf-8")
    (folder / "Funding_Statement.txt").write_text(FUNDING, encoding="utf-8")
    (folder / "Competing_Interests_Statement.txt").write_text(COMPETING, encoding="utf-8")
    (folder / "Ethics_Statement.txt").write_text(ETHICS, encoding="utf-8")


def copy_and_convert_figures(items: list[tuple[str, Path]], png_dir: Path, tiff_dir: Path) -> list[dict[str, str]]:
    png_dir.mkdir(parents=True, exist_ok=True)
    tiff_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, src in items:
        if not src.exists():
            raise FileNotFoundError(src)
        png_dst = png_dir / f"{label}.png"
        tif_dst = tiff_dir / f"{label}.tif"
        shutil.copy2(src, png_dst)
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(tif_dst, dpi=(300, 300), compression="tiff_lzw")
            rows.append(
                {
                    "label": label,
                    "source": str(src.relative_to(ROOT)),
                    "png": str(png_dst.relative_to(PKG)),
                    "tiff": str(tif_dst.relative_to(PKG)),
                    "width_px": str(img.width),
                    "height_px": str(img.height),
                }
            )
    return rows


def package_source_data() -> Path:
    src_root = PKG / "source_data_for_figures"
    src_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for folder in SOURCE_DATA_PATTERNS:
        if not folder.exists():
            continue
        dest = src_root / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for file in folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".csv", ".json", ".txt", ".md"}:
                name_lower = file.name.lower()
                if any(token in name_lower for token in SOURCE_DATA_EXCLUDE_TOKENS):
                    continue
                rel = file.relative_to(folder)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
                copied.append(str(target.relative_to(PKG)))

    readme = src_root / "README_source_data.txt"
    readme.write_text(
        "This folder contains derived source tables and analysis-result files used "
        "to draw the manuscript figures, tables, and supplementary analyses. Raw "
        "patient-level data remain available from the public repositories cited "
        "in the manuscript and Data Availability Statement.\n",
        encoding="utf-8",
    )
    copied.append(str(readme.relative_to(PKG)))

    zip_path = PKG / "data_availability" / "S1_Data_derived_results.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in src_root.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(src_root))
    return zip_path


def write_manifest(rows: list[dict[str, str]], docs: list[Path], source_zip: Path) -> None:
    qc = PKG / "quality_control"
    qc.mkdir(parents=True, exist_ok=True)
    with (qc / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "source", "png", "tiff", "width_px", "height_px"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "PLOS ONE submission package manifest",
        "",
        "Documents:",
        *[f"- {p.relative_to(PKG)}" for p in docs],
        "",
        f"Derived source data zip: {source_zip.relative_to(PKG)}",
        "",
        "Figure files are provided as both PNG originals and 300-dpi TIFF exports.",
        "FigS6 hospital-validation template was intentionally excluded from submission files.",
    ]
    (qc / "PACKAGE_QA_NOTES.txt").write_text("\n".join(lines), encoding="utf-8")


def make_package_zip() -> Path:
    zip_path = OUT / "NSCLC_MMFT_PLOS_ONE_submission_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in PKG.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(PKG.parent))
    return zip_path


def main() -> None:
    clear_output_dir()
    docs = [
        make_plos_manuscript(),
        write_cover_letter(),
        write_submission_metadata(),
        write_supporting_legends(),
        write_readme(),
    ]
    write_text_statements()
    rows = []
    rows.extend(copy_and_convert_figures(MAIN_FIGURES, PKG / "figures" / "main_figures_png", PKG / "figures" / "main_figures_tiff"))
    rows.extend(
        copy_and_convert_figures(
            SUPPLEMENTARY_FIGURES,
            PKG / "figures" / "supplementary_figures_png",
            PKG / "figures" / "supplementary_figures_tiff",
        )
    )
    source_zip = package_source_data()
    write_manifest(rows, docs, source_zip)
    package_zip = make_package_zip()
    print(f"Created package directory: {PKG}")
    print(f"Created zip: {package_zip}")


if __name__ == "__main__":
    main()

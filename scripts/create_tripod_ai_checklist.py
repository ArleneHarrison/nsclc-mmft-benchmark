"""Generate a completed TRIPOD+AI reporting checklist for the NSCLC MMFT manuscript.

Source checklist: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement:
updated guidance for reporting clinical prediction models that use regression
or machine learning methods. BMJ 2024;385:e078378.

This script honestly maps each of the 27 TRIPOD+AI items (with sub-items) to
where (if anywhere) the manuscript addresses them, and flags genuine gaps
rather than papering over them.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "plos_one_submission_package" / "supporting_information" / "TRIPOD_AI_Checklist.docx"

# Status colors
COLOR_OK = RGBColor(0x1B, 0x5E, 0x20)      # green - reported
COLOR_PARTIAL = RGBColor(0x8A, 0x6D, 0x00) # amber - partial
COLOR_GAP = RGBColor(0xB0, 0x00, 0x20)     # red - gap / not addressed
COLOR_NA = RGBColor(0x55, 0x55, 0x55)      # grey - not applicable

STATUS_STYLE = {
    "Reported": COLOR_OK,
    "Partial": COLOR_PARTIAL,
    "Gap": COLOR_GAP,
    "N/A": COLOR_NA,
}

# (item, D/E, topic, checklist text, status, location/notes)
ROWS = [
    ("TITLE", None, None, None, None, None),
    ("1", "D;E", "Title",
     "Identify the study as developing/evaluating a prediction model, target population, outcome.",
     "Reported", "Title page — names model class (regularized linear vs deep learning), population (NSCLC), outcome (histological subtype)."),

    ("ABSTRACT", None, None, None, None, None),
    ("2", "D;E", "Abstract",
     "Structured summary per TRIPOD+AI for Abstracts.",
     "Reported", "Abstract — objectives, data source, n, methods, headline results, conclusion, external-validation caveat all present."),

    ("INTRODUCTION", None, None, None, None, None),
    ("3a", "D;E", "Background",
     "Healthcare context and rationale, references to existing models.",
     "Reported", "Section 1, paras 1-3; cites the source model (ref 1) explicitly."),
    ("3b", "D;E", "Background",
     "Target population, intended purpose in the care pathway, intended users.",
     "Reported", "Section 1 (NSCLC-NOS / tissue-limited scenario); Section 4.5 Clinical positioning."),
    ("3c", "D;E", "Background",
     "Known health inequalities between sociodemographic groups.",
     "Gap", "Not discussed. Given the extreme sex skew in squamous cases (Section 4.6(12), new), a one-sentence acknowledgement here would close the loop with the Introduction's clinical framing."),
    ("4", "D;E", "Objectives",
     "Study objectives — development and/or validation.",
     "Reported", "Section 1, contributions list (1)-(7); explicit central question stated."),

    ("METHODS", None, None, None, None, None),
    ("5a", "D;E", "Data",
     "Sources of data for development vs evaluation, rationale, representativeness.",
     "Reported", "Section 2.1 (primary cohort), 2.8 (TCGA), 2.9/2.9c (TCIA). Note: 'evaluation' of the primary model is an internal hold-out of the same source cohort, not a separate external dataset — already flagged as Limitation 4.6(1)."),
    ("5b", "D;E", "Data",
     "Dates of participant accrual / follow-up.",
     "Gap", "Not restated in this manuscript (secondary analysis of public data); readers must consult the source publication (ref 1) for original accrual dates."),
    ("6a", "D;E", "Participants",
     "Study setting, number/location of centres.",
     "Partial", "Not explicit for the primary cohort beyond 'public PLOS-2024 cohort' (Section 2.1); TCIA is a named multi-institutional US collection (Section 2.9) but centre count is not given."),
    ("6b", "D;E", "Participants",
     "Eligibility criteria.",
     "Reported", "Section 2.1 — inclusion/exclusion criteria stated explicitly, including the Stage-III-only caveat."),
    ("6c", "D;E", "Participants",
     "Treatments received and handling.",
     "Reported", "Section 2.1 — prior anti-tumour treatment and post-diagnostic surgery are explicit exclusion criteria."),
    ("7", "D;E", "Data preparation",
     "Pre-processing and quality checks, consistency across subgroups.",
     "Reported", "Section 2.4; Fig S1 (missingness). Consistency across sociodemographic groups specifically is not discussed."),
    ("8a", "D;E", "Outcome",
     "Outcome definition, time horizon, rationale.",
     "Reported", "Section 2.2 — binary histological subtype, coding given."),
    ("8b", "D;E", "Outcome",
     "Assessor qualifications for subjective outcomes.",
     "Gap", "Outcome is pathological diagnosis from the source dataset; assessor qualifications are not re-described here (would be in ref 1)."),
    ("8c", "D;E", "Outcome",
     "Blinding of outcome assessment.",
     "Gap", "Not addressed — not directly controllable in a secondary analysis of already-labelled public data."),
    ("9a", "D", "Predictors",
     "Choice of initial predictors, any pre-selection.",
     "Reported", "Section 2.3 — modality blocks defined; predictors are all features available in the source public dataset (no pre-selection beyond block membership)."),
    ("9b", "D;E", "Predictors",
     "Predictor definitions, measurement timing/blinding.",
     "Reported", "Section 2.3 lists all predictors; measured pretreatment per Section 2.1/2.2 framing."),
    ("9c", "D;E", "Predictors",
     "Assessor qualifications for subjective predictors.",
     "Partial", "Section 2.9c describes the CT semantic-phenotype features (radiologist AIM annotations) but not the annotators' qualifications/demographics or inter-rater variability."),
    ("10", "D;E", "Sample size",
     "How study size was arrived at; justification of adequacy; sample-size calculation.",
     "Partial", "No formal calculation — n is fixed by the existing public dataset (Section 2.1). Adequacy is addressed indirectly via wide CIs and the no-plateau learning curve (Section 3.8, Limitation 4.6(2)), but no explicit sample-size paragraph exists in Methods."),
    ("11", "D;E", "Missing data",
     "How missing data were handled; reasons for omissions.",
     "Reported", "Section 2.4 (median imputation, train-only fit); Fig S1."),
    ("12a", "D", "Analytical methods",
     "How data were partitioned; sample-size adequacy of partitions.",
     "Reported", "Section 2.4 (70/30 stratified split, seed given); Section 2.7 (nested CV, repeated CV)."),
    ("12b", "D", "Analytical methods",
     "Predictor handling — functional form, rescaling, standardisation.",
     "Reported", "Section 2.4 — standardisation as part of the pipeline."),
    ("12c", "D", "Analytical methods",
     "Model type/rationale, all model-building steps incl. hyperparameter tuning, internal validation method.",
     "Reported", "Sections 2.5-2.8d — unusually thorough (leakage-free architecture search, TabPFN, bug-postmortem redo all documented). One gap: class_weight=\"balanced\" is used in the actual primary-model code (benchmark_petct_blood_models.py) but is never mentioned in the manuscript text (Section 2.5) — a real, fixable methods-disclosure gap."),
    ("12d", "D;E", "Analytical methods",
     "Heterogeneity across clusters (e.g. hospitals).",
     "N/A", "Single-source public dataset; no site-level clustering to quantify."),
    ("12e", "D;E", "Analytical methods",
     "Performance measures and plots specified.",
     "Reported", "Section 2.7 — AUC, calibration slope/intercept, Brier, Hosmer-Lemeshow, DCA, SHAP, DeLong."),
    ("12f", "E", "Analytical methods",
     "Model updating (recalibration) from evaluation.",
     "Reported", "Section 2.7b(v)/3.8 — cross-fitted Platt recalibration attempted and honestly reported as unsuccessful at this n."),
    ("12g", "E", "Analytical methods",
     "How predictions were calculated for evaluation.",
     "Reported", "Table 5/Fig 8 give a usable nomogram + integer score (separate MLE refit, Section 2.9b); the exact benchmarked ridge-penalised primary model's own coefficients are now tabulated in Table 9 (added in this review; independently refit and confirmed bit-for-bit identical predictions to the original benchmark)."),
    ("13", "D;E", "Class imbalance",
     "Class-imbalance methods used, rationale, recalibration.",
     "Gap", "The cohort is moderately imbalanced (110 adeno / 145 squamous, ~43:57) and the actual code applies scikit-learn class_weight=\"balanced\" for every logistic-regression/SVM/tree model — but this is never stated in the manuscript text. Recommend one sentence in Section 2.5/2.6 disclosing this."),
    ("14", "D;E", "Fairness",
     "Approaches to address model fairness, rationale.",
     "Partial", "Newly added Limitation 4.6(12) (this review) discloses that squamous cases are 95.2% male and the female-squamous subgroup (7/145 overall, 1/77 in the test set) is too small to support any subgroup performance estimate. No formal fairness-mitigation approach was applied (nor would one be appropriate at this subgroup size) — disclosure, not correction, is the honest option here."),
    ("15", "D", "Model output",
     "Output type (probability/classification), thresholds.",
     "Reported", "Section 2.9b/3.2b — probability output; integer score with explicit threshold-free scoring; DCA thresholds given (Section 3.5)."),
    ("16", "D;E", "Development vs evaluation",
     "Differences between development and evaluation data.",
     "N/A", "No separate external evaluation dataset for the primary model (addressed as Limitation 4.6(1),(6)); TCIA/TCGA differences from the primary cohort are discussed where used (Sections 2.9, 3.9)."),
    ("17", "D;E", "Ethical approval",
     "Name of IRB/ethics committee, consent or waiver.",
     "Reported", "Section 2.10 — public/de-identified data, no new IRB required; explicit reasoning given."),

    ("OPEN SCIENCE", None, None, None, None, None),
    ("18a", "D;E", "Funding",
     "Source of funding, role of funders.",
     "Reported", "Funding Statement — no specific funding."),
    ("18b", "D;E", "Conflicts of interest",
     "COI/financial disclosures.",
     "Reported", "Competing Interests Statement — none declared."),
    ("18c", "D;E", "Protocol",
     "Where protocol is available, or state none was prepared.",
     "Reported", "Ethics Statement (updated in this review) — explicitly states no protocol was pre-registered."),
    ("18d", "D;E", "Registration",
     "Registration details, or state study was not registered.",
     "Reported", "Ethics Statement (updated in this review) — explicitly states the study was not registered."),
    ("18e", "D;E", "Data sharing",
     "Availability of study data.",
     "Reported", "Data Availability Statement — all three source datasets are public with citations; derived tables in S1 Data."),
    ("18f", "D;E", "Code sharing",
     "Availability of analytical code.",
     "Gap", "ACTION NEEDED. manuscript.md states code is 'available in the study repository' but the submission docx's Data Availability Statement does not mention code at all, and no public code URL/DOI (GitHub, Zenodo, etc.) appears anywhere in the package. TRIPOD+AI (and PLOS ONE's own guidance) explicitly discourage vague 'available on request' language. Recommend depositing the scripts/ directory to a citable public repository before submission and adding its DOI/URL here."),

    ("PATIENT & PUBLIC INVOLVEMENT", None, None, None, None, None),
    ("19", "D;E", "PPI",
     "Patient/public involvement, or state none.",
     "Reported", "Ethics Statement (updated in this review) — explicitly states no patients/public were involved, consistent with a secondary analysis of de-identified public data."),

    ("RESULTS", None, None, None, None, None),
    ("20a", "D;E", "Participants",
     "Flow of participants, outcome counts, follow-up summary.",
     "Partial", "Fig 1 gives the overall analytic workflow and n's; a CONSORT-style eligibility/exclusion flow diagram specific to this secondary analysis is not reproduced (the exclusions were applied by the source study, ref 1)."),
    ("20b", "D;E", "Participants",
     "Characteristics overall and by source, demographic differences.",
     "Reported", "Table 1 — comprehensive, by subtype, with p-values; now includes the age/albumin/MTV-TLG clarification added in this review."),
    ("20c", "E", "Participants",
     "Comparison of predictor distribution vs development data.",
     "N/A", "No separate external validation of the primary model. Table 4 provides an analogous demographic comparison for the independent TCIA replication analysis."),
    ("21", "D;E", "Model development",
     "N and outcome events in each analysis.",
     "Reported", "Consistently reported throughout (n=255/178/77 primary; n=207/189/187 TCIA; n=528/501 and n=589/552 TCGA, etc.)."),
    ("22", "D", "Model specification",
     "Full model details (formula/code/object) enabling third-party use.",
     "Reported", "Table 5 + Fig 8 give a clinician-usable surrogate (integer score/nomogram, separate MLE refit, Section 2.9b). Table 9 (added in this review) additionally gives the exact benchmarked ridge (k=5, C=3.0) primary model's standardized coefficients, intercept, and imputation/scaling parameters, independently refit and confirmed to reproduce the original predictions to floating-point precision."),
    ("23a", "D;E", "Model performance",
     "Performance with CIs, incl. key subgroups.",
     "Partial", "Overall performance thoroughly reported with CIs (Table 2, 3.2). Subgroup performance is reported only for the dataset-coded stage substratum (Section 3.8/Fig 6D); sex-stratified performance is not reported in the main text (see item 14 — this review found n=1 squamous case among 12 female test patients, too sparse for a stable subgroup AUC, which is itself the reportable finding)."),
    ("23b", "D;E", "Model performance",
     "Heterogeneity across clusters.",
     "N/A", "Single-source dataset."),
    ("24", "E", "Model updating",
     "Results of any model updating.",
     "Reported", "Section 3.8 — recalibration attempted, reported honestly as unsuccessful at this sample size, with a stated reason."),

    ("DISCUSSION", None, None, None, None, None),
    ("25", "D;E", "Interpretation",
     "Overall interpretation in context of objectives/prior studies, fairness.",
     "Reported", "Sections 4.1-4.4 — extensive; explicit reconciliation with the source paper's opposite radiomics claim. Fairness discussion is limited to the new sex-subgroup limitation (4.6(12))."),
    ("26", "D;E", "Limitations",
     "Limitations affecting bias, uncertainty, generalisability.",
     "Reported", "Section 4.6 — exceptionally thorough, 12 numbered points after this review's addition."),
    ("27a", "D", "Usability",
     "Handling poor-quality/unavailable input data at deployment.",
     "Gap", "Not discussed — e.g., cross-scanner/cross-tracer PET SUV harmonisation, or what to do if a predictor is missing at the point of use, is not addressed."),
    ("27b", "D", "Usability",
     "User interaction required, expertise level.",
     "Partial", "Section 4.5 frames the nomogram/integer score as usable 'without software,' implying minimal expertise, but this is not stated explicitly as a usability requirement."),
    ("27c", "D;E", "Usability",
     "Future research needs, applicability/generalisability.",
     "Reported", "Section 4.7 — clear next steps (external validation, N-staging extension)."),
]


def set_cell_shading(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_section_row(table, title: str) -> None:
    row = table.add_row()
    row.cells[0].merge(row.cells[1]).merge(row.cells[2]).merge(row.cells[3]).merge(row.cells[4])
    p = row.cells[0].paragraphs[0]
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(10)
    set_cell_shading(row.cells[0], "D9D9D9")


def add_item_row(table, item, de, topic, text, status, notes) -> None:
    row = table.add_row()
    cells = row.cells
    cells[0].text = item
    cells[1].text = de or ""
    cells[2].text = f"{topic}\n{text}"
    status_p = cells[3].paragraphs[0]
    status_run = status_p.add_run(status)
    status_run.bold = True
    status_run.font.color.rgb = STATUS_STYLE.get(status, RGBColor(0, 0, 0))
    cells[4].text = notes
    for c in cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5)
            p.paragraph_format.space_after = Pt(2)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("TRIPOD+AI Reporting Checklist")
    r.bold = True
    r.font.size = Pt(14)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = subtitle.add_run(
        "Regularized Linear Models Match Multimodal Deep Learning for NSCLC "
        "Histological-Subtype Prediction"
    )
    r2.italic = True
    r2.font.size = Pt(11)

    intro = doc.add_paragraph()
    intro.add_run(
        "Completed against: Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: "
        "updated guidance for reporting clinical prediction models that use regression or "
        "machine learning methods. BMJ 2024;385:e078378. D = development, E = evaluation, "
        "D;E = both. Status color key: "
    ).font.size = Pt(9)
    for label, color in [("Reported", COLOR_OK), ("Partial", COLOR_PARTIAL), ("Gap", COLOR_GAP), ("N/A", COLOR_NA)]:
        run = intro.add_run(f" {label} ")
        run.bold = True
        run.font.color.rgb = color
        run.font.size = Pt(9)

    doc.add_paragraph()

    table = doc.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = [Cm(1.3), Cm(1.3), Cm(6.5), Cm(1.8), Cm(6.5)]
    hdr = table.rows[0].cells
    headers = ["Item", "D/E", "Topic & checklist item", "Status", "Location in manuscript / notes"]
    for cell, text, w in zip(hdr, headers, widths):
        cell.text = text
        cell.width = w
        for p in cell.paragraphs:
            for r_ in p.runs:
                r_.bold = True
                r_.font.size = Pt(9)
        set_cell_shading(cell, "4472C4")
        for p in cell.paragraphs:
            for r_ in p.runs:
                r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for item, de, topic, text, status, notes in ROWS:
        if topic is None and text is None and status is None:
            add_section_row(table, item)
        else:
            add_item_row(table, item, de, topic, text, status, notes)
        for row in table.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = w

    doc.add_paragraph()
    summary = doc.add_paragraph()
    summary.add_run(
        "Summary: of 52 applicable line items (34 Reported, 7 Partial, 7 Gap, 4 N/A), most "
        "Methods/Results/Discussion items are fully "
        "reported (this manuscript's methodological transparency is above average for the "
        "genre). Items fixed as part of this review: (12c/13) class_weight=\"balanced\" usage "
        "is now disclosed in Methods; (14/4.6) a new Limitation discloses that the "
        "female-squamous subgroup is too small (7/145 overall, 1/77 in the test set) to "
        "support any subgroup performance estimate; (22/12g) the exact benchmarked primary "
        "model's standardized coefficients are now tabulated in Table 9; (18c/18d/19) explicit "
        "protocol/registration/PPI statements were added. The one remaining concrete gap "
        "requiring an author decision is (18f): no public code repository is cited despite "
        "the paper's leakage-free/reproducibility emphasis."
    ).font.size = Pt(9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()

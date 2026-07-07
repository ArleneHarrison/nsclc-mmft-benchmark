"""Parse all TCIA NSCLC-Radiogenomics AIM XML radiologist annotation files into
a clean, structured semantic-feature table (one row per patient).

These are NOT raw pixel/segmentation data -- they are coded, structured
radiological descriptors (axial location, margin pattern, attenuation,
calcification, associated findings, etc.), read directly by radiologists per
the AIM v4 standard. This is a legitimate, information-rich, and previously
UNUSED (in this project) feature source, distinct from both the PLOS-2024
tabular data and the TCGA transcriptomics already analysed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AIM_ZIP = ROOT / "public_data" / "nsclc_radiogenomics_aim.zip"
EXTRACT_DIR = ROOT / "tmp" / "aim_all"
OUT_DIR = ROOT / "public_data" / "tcia_semantic_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NS = {"a": "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM"}


CODE_LABEL_MAP: dict[str, str] = {}  # RadLex code -> human-readable label, built across all files first


def parse_one(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    person = root.find(".//a:person/a:id", NS)
    case_id = person.get("value") if person is not None else xml_path.stem

    row = {"Case ID": case_id}
    # Physical entity (anatomic location, e.g. "lower lobe of left lung")
    for ent in root.findall(".//a:ImagingPhysicalEntity", NS):
        label = ent.find("a:label", NS)
        tc = ent.find("a:typeCode", NS)
        if label is not None and tc is not None and label.get("value"):
            key = f"aim_{label.get('value').replace(' ', '_')}"
            row[key] = tc.get("codeSystem") or tc.get("code") or None

    # Observation-level type (e.g. "Lung Nodule")
    obs = root.find(".//a:ImagingObservationEntity/a:typeCode", NS)
    if obs is not None:
        row["aim_Observation_Type"] = obs.get("codeSystem") or obs.get("code") or None

    # All semantic characteristics (axial location, margins, attenuation, ...)
    # Some labels repeat (e.g. multiple "Nodule Associated Findings"); collect
    # repeats into a semicolon-joined string so no information is dropped.
    collected: dict[str, list[str]] = {}
    for char in root.findall(".//a:ImagingObservationCharacteristic", NS):
        label = char.find("a:label", NS)
        tc = char.find("a:typeCode", NS)
        if label is None or tc is None:
            continue
        key = f"aim_{label.get('value').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
        code, sys_ = tc.get("code"), tc.get("codeSystem")
        if code and sys_:
            CODE_LABEL_MAP[code] = sys_
        val = sys_ or (CODE_LABEL_MAP.get(code) if code else None) or code or ""
        if val:
            collected.setdefault(key, []).append(val)
    for key, vals in collected.items():
        row[key] = "; ".join(sorted(set(vals)))

    return row


def main() -> None:
    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.rglob("*.xml")):
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(AIM_ZIP) as z:
            z.extractall(EXTRACT_DIR)

    xml_files = sorted(EXTRACT_DIR.rglob("*.xml"))
    print(f"Found {len(xml_files)} AIM XML files")

    # Pass 1: build a code->label map from every file (some files omit
    # codeSystem for a code that IS labelled in other files' occurrences).
    for f in xml_files:
        parse_one(f)
    # Pass 2: parse for real, now every code can resolve to a label.
    rows = [parse_one(f) for f in xml_files]
    df = pd.DataFrame(rows)
    print(f"Parsed table shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("\nMissingness per column (%):")
    print((df.isna().mean() * 100).round(1).sort_values())

    out_path = OUT_DIR / "aim_semantic_features_raw.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved raw parsed table to {out_path}")


if __name__ == "__main__":
    main()

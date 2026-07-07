from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


NS = {
    "aim": "gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM",
    "iso": "uri:iso.org:21090",
}


def clean_value(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value.strip())
    value = value.lower()
    return value


def type_code_value(node: ET.Element) -> str:
    display = node.find("iso:displayName", NS)
    if display is not None:
        val = clean_value(display.attrib.get("value"))
        if val and val != "na":
            return val

    # Older AIM files sometimes put the human-readable value in codeSystem.
    for attr in ("codeSystem", "codeSystemName", "code"):
        val = clean_value(node.attrib.get(attr))
        if val and val not in {"radlex", "radlex3.8", "radlex.3.10", "radlex3.10_ns", "lungtemplate", "private"}:
            return val
    return ""


def is_positive_feature(values: list[str], keywords: list[str]) -> int:
    joined = " | ".join(values)
    return int(any(keyword in joined for keyword in keywords))


def first_matching(values: list[str], allowed: list[str]) -> str:
    joined = " | ".join(values)
    for item in allowed:
        if item in joined:
            return item
    return ""


def parse_aim_zip(zip_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".xml"):
                continue
            root = ET.fromstring(zf.read(member))
            patient_id = root.find(".//aim:person/aim:id", NS)
            case_id = patient_id.attrib.get("value") if patient_id is not None else Path(member).stem

            by_label: dict[str, list[str]] = {}
            for char in root.findall(".//aim:ImagingObservationCharacteristic", NS):
                label_node = char.find("aim:label", NS)
                label = label_node.attrib.get("value") if label_node is not None else ""
                label_key = clean_value(label)
                values = [type_code_value(tc) for tc in char.findall("aim:typeCode", NS)]
                values = [v for v in values if v]
                if label_key and values:
                    by_label.setdefault(label_key, []).extend(values)

            associated = by_label.get("nodule associated findings", [])
            internal = by_label.get("nodule internal features", [])
            margin_primary = by_label.get("nodule margins-primary pattern", [])
            margin_secondary = by_label.get("nodule margins-secondary pattern", [])

            margin_all = margin_primary + margin_secondary
            attenuation = by_label.get("nodule attenuation", [])
            row = {
                "Case ID": case_id,
                "aim_file": member,
                "axial_location": first_matching(by_label.get("axial location", []), ["central", "peripheral"]),
                "attenuation": first_matching(attenuation, ["solid", "partially solid", "ground glass", "non-solid"]),
                "is_solid_or_partsolid": is_positive_feature(attenuation, ["solid", "partially solid"]),
                "margin_primary": first_matching(margin_primary, ["spiculated", "irregular", "lobulated", "poorly defined", "smooth"]),
                "margin_secondary": first_matching(margin_secondary, ["spiculated", "irregular", "lobulated", "poorly defined", "smooth"]),
                "margin_spiculated_or_irregular": is_positive_feature(margin_all, ["spiculated", "irregular"]),
                "margin_lobulated": is_positive_feature(margin_all, ["lobulated"]),
                "shape": first_matching(by_label.get("nodule shape", []), ["complex", "round", "oval"]),
                "calcification_present": int(
                    first_matching(by_label.get("nodule calcification", []), ["peripheral", "central", "diffuse", "popcorn"]) != ""
                ),
                "pleural_retraction": is_positive_feature(associated, ["pleural retraction"]),
                "attachment_to_pleura": is_positive_feature(associated, ["attachment to pleura"]),
                "vascular_convergence": is_positive_feature(associated, ["vascular convergence"]),
                "entering_airway": is_positive_feature(associated, ["entering airway"]),
                "air_bronchogram": is_positive_feature(internal, ["air bronchogram"]),
                "cavitation_or_necrosis": is_positive_feature(internal, ["cavitated", "necrosis"]),
                "emphysema_present": is_positive_feature(by_label.get("emphysema", []), ["present"]),
                "fibrosis_present": is_positive_feature(by_label.get("fibrosis", []), ["present"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    zip_path = data_dir / "nsclc_radiogenomics_aim.zip"
    out_path = data_dir / "nsclc_radiogenomics_aim_features.csv"
    features = parse_aim_zip(zip_path)
    features.to_csv(out_path, index=False)
    print(f"wrote {out_path} with shape {features.shape}")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs" / "reference_style_reproduction" / "Figure7_GPTimage2_workflow_background.png"
OUT = ROOT / "outputs" / "reference_style_reproduction" / "Figure7_GPTimage2_workflow_labeled.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf") if bold else Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    wh: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int] = (31, 36, 48),
    size: int = 22,
    bold: bool = False,
    bg: tuple[int, int, int, int] = (255, 255, 255, 210),
) -> None:
    x, y = xy
    w, h = wh
    box = [x, y, x + w, y + h]
    draw.rounded_rectangle(box, radius=12, fill=bg, outline=(90, 106, 122, 190), width=1)
    f = font(size, bold=bold)
    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=f) for line in lines]
    heights = [b[3] - b[1] for b in bboxes]
    total_h = sum(heights) + (len(lines) - 1) * int(size * 0.25)
    cy = y + (h - total_h) / 2
    for line, bbox, lh in zip(lines, bboxes, heights):
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) / 2, cy), line, font=f, fill=fill)
        cy += lh + int(size * 0.25)


def main() -> None:
    img = Image.open(IN).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Section rail labels: keep large A/B/C from GPTimage2, add exact section subtitles.
    draw_label(draw, (24, 272), (76, 34), "Data", size=18, bold=True, bg=(255, 255, 255, 180))
    draw_label(draw, (24, 594), (76, 34), "Model", size=18, bold=True, bg=(255, 255, 255, 180))
    draw_label(draw, (24, 912), (76, 34), "Analysis", size=17, bold=True, bg=(255, 255, 255, 180))

    labels = [
        ((150, 236), (260, 54), "Public NSCLC cohort\nn = 255"),
        ((505, 236), (260, 54), "Preoperative CT/PET-CT\nSUVmean, SUVmax, SUVmin"),
        ((852, 236), (280, 54), "Blood markers\nCEA, WBC, NLR/dNLR"),
        ((1217, 236), (242, 54), "Feature matrix\nsimple clinical variables"),
        ((148, 600), (220, 54), "Multimodal input\nPET/CT + blood + CEA"),
        ((418, 600), (198, 54), "Tokenization\nvalue + modality ID"),
        ((688, 600), (250, 54), "MMFT-Transformer\ngated attention"),
        ((1015, 600), (150, 54), "Risk token\nridge prior"),
        ((1234, 600), (220, 54), "Histology output\nclass 0 vs class 1"),
        ((140, 900), (185, 54), "ROC / AUC\nmodel comparison"),
        ((360, 900), (160, 54), "Confusion\nmatrix"),
        ((560, 900), (200, 54), "Risk distribution\nsubgroups"),
        ((804, 900), (190, 54), "TCGA KM\nbio-risk proxy"),
        ((1050, 900), (170, 54), "SHAP\nbeeswarm"),
        ((1262, 900), (220, 54), "Hospital external\nvalidation"),
    ]
    for xy, wh, text in labels:
        draw_label(draw, xy, wh, text, size=18, bold=False)

    # Short title badge without covering the generated figure.
    draw_label(
        draw,
        (396, 16),
        (740, 48),
        "NSCLC-MMFT workflow: public modeling, biological support, and validation",
        size=20,
        bold=True,
        bg=(255, 255, 255, 225),
    )

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()

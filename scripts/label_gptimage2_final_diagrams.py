from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs" / "gptimage2_diagrams"
OUT_DIR = IN_DIR

WORKFLOW_IN = IN_DIR / "GPTimage2_research_workflow_background.png"
ARCH_IN = IN_DIR / "GPTimage2_model_architecture_background.png"
WORKFLOW_OUT = OUT_DIR / "Figure7_GPTimage2_research_workflow_final.png"
ARCH_OUT = OUT_DIR / "Figure9_GPTimage2_model_architecture_final.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(32, True)
FONT_SECTION = font(25, True)
FONT_LABEL = font(22, True)
FONT_SMALL = font(17)
FONT_TINY = font(15)


BLUE = (31, 78, 121)
BLUE_DARK = (22, 54, 92)
RED = (150, 40, 54)
GREEN = (65, 117, 70)
PURPLE = (92, 70, 135)
INK = (30, 41, 59)
GRAY = (75, 85, 99)
WHITE = (255, 255, 255)


def composite_box(
    base: Image.Image,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int, int] = (255, 255, 255, 232),
    outline: tuple[int, int, int] = BLUE,
    radius: int = 12,
    width: int = 2,
) -> ImageDraw.ImageDraw:
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline + (255,), width=width)
    base.alpha_composite(overlay)
    return ImageDraw.Draw(base)


def multiline_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = INK,
    spacing: int = 4,
) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_obj)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = xy[1] + (xy[3] - xy[1] - total_h) / 2 - 1
    for line, w, h in zip(lines, widths, heights):
        x = xy[0] + (xy[2] - xy[0] - w) / 2
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += h + spacing


def label_box(
    base: Image.Image,
    xy: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int] = BLUE,
    size: str = "normal",
) -> None:
    font_obj = FONT_LABEL if size == "normal" else FONT_SMALL if size == "small" else FONT_TINY
    draw = composite_box(base, xy, outline=color)
    multiline_center(draw, xy, text, font_obj, fill=INK if color != RED else (95, 20, 30))


def ribbon(base: Image.Image, xy: tuple[int, int, int, int], text: str, color: tuple[int, int, int]) -> None:
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(xy, radius=16, fill=color + (235,))
    base.alpha_composite(overlay)
    draw = ImageDraw.Draw(base)
    multiline_center(draw, xy, text, FONT_SECTION, fill=WHITE)


def title(base: Image.Image, text: str) -> None:
    xy = (420, 18, 1252, 68)
    draw = composite_box(base, xy, fill=(255, 255, 255, 245), outline=BLUE_DARK, radius=16, width=2)
    multiline_center(draw, xy, text, FONT_TITLE, fill=BLUE_DARK)


def make_workflow() -> None:
    base = Image.open(WORKFLOW_IN).convert("RGBA")
    title(base, "NSCLC-MMFT Research Workflow")

    ribbon(base, (18, 84, 118, 137), "A  Data", BLUE)
    ribbon(base, (18, 342, 130, 395), "B  Model", RED)
    ribbon(base, (18, 666, 170, 719), "C  Analysis", GREEN)

    label_box(base, (60, 229, 328, 267), "Chest CT\nsemantic signs", BLUE, "small")
    label_box(base, (430, 229, 676, 267), "PET/CT metabolism\nSUVmax / SUVmean", BLUE, "small")
    label_box(base, (762, 229, 954, 267), "Blood + CEA\nNLR / PLR / SII", BLUE, "small")
    # Replaces the original photorealistic "hospital validation" icon slot: that external
    # cohort was never collected. This slot now represents the independent public TCIA
    # cohort actually used for replication and the third feature modality (CT semantic
    # phenotype), drawn as a plain text card rather than a re-used photorealistic icon.
    draw_a4 = composite_box(base, (1010, 76, 1290, 267), fill=(247, 249, 252, 245), outline=BLUE, radius=16, width=2)
    multiline_center(draw_a4, (1010, 76, 1290, 267),
                      "TCIA independent\ncohort (n=207-211)\nreplication +\n3rd feature modality",
                      FONT_LABEL, fill=INK)
    label_box(base, (1366, 229, 1585, 267), "TCGA-LUAD/LUSC\nn = 1029", BLUE, "small")

    label_box(base, (60, 568, 343, 609), "Feature table\nclinical + blood + imaging", RED, "small")
    label_box(base, (432, 568, 664, 609), "Imputation\nstandardization", RED, "small")
    label_box(base, (718, 568, 955, 609), "MMFT rank-refit\nmultimodal model", RED, "small")
    label_box(base, (1025, 568, 1245, 609), "Risk probability\nhistology / pN+", RED, "small")
    label_box(base, (1336, 568, 1575, 609), "SHAP explanation\nfeature attribution", RED, "small")

    label_box(base, (82, 845, 345, 889), "Discrimination\nROC / AUC", GREEN, "small")
    label_box(base, (454, 845, 735, 889), "Robustness\nsubgroup AUC", GREEN, "small")
    label_box(base, (831, 845, 1118, 889), "External biology\nKM / pathway proxy", GREEN, "small")
    label_box(base, (1201, 845, 1582, 889), "Clinical pathway\nlocked model -> nomogram / bedside score", GREEN, "small")

    draw = ImageDraw.Draw(base)
    draw.text((35, 900), "AI background generated by GPTimage2; exact labels added programmatically.", font=FONT_TINY, fill=GRAY)
    base.convert("RGB").save(WORKFLOW_OUT, quality=96)


def make_architecture() -> None:
    base = Image.open(ARCH_IN).convert("RGBA")
    title(base, "MMFT Model Architecture")

    label_box(base, (46, 183, 252, 234), "Clinical\nage / sex / smoking", BLUE, "small")
    label_box(base, (46, 352, 252, 402), "Blood + CEA\nNLR / dNLR / PLT", RED, "small")
    label_box(base, (46, 540, 252, 588), "PET/CT metabolism\nSUV features", GREEN, "small")
    label_box(base, (46, 714, 252, 760), "Optional CT branch\nsemantic / radiomics", PURPLE, "small")

    label_box(base, (304, 151, 424, 210), "impute\nscale", BLUE, "tiny")
    label_box(base, (304, 326, 424, 386), "impute\nscale", RED, "tiny")
    label_box(base, (304, 511, 424, 571), "impute\nscale", GREEN, "tiny")
    label_box(base, (304, 688, 424, 748), "impute\nscale", PURPLE, "tiny")

    label_box(base, (468, 166, 600, 205), "tokens", BLUE, "tiny")
    label_box(base, (468, 342, 600, 381), "tokens", RED, "tiny")
    label_box(base, (468, 528, 600, 567), "tokens", GREEN, "tiny")
    label_box(base, (468, 704, 600, 743), "tokens", PURPLE, "tiny")

    label_box(base, (640, 151, 727, 210), "embed", BLUE, "tiny")
    label_box(base, (640, 326, 727, 386), "embed", RED, "tiny")
    label_box(base, (640, 511, 727, 571), "embed", GREEN, "tiny")
    label_box(base, (640, 688, 727, 748), "embed", PURPLE, "tiny")

    draw = composite_box(base, (830, 100, 1150, 144), fill=(255, 255, 255, 238), outline=BLUE_DARK, radius=12, width=2)
    multiline_center(draw, (830, 100, 1150, 144), "Transformer encoder", FONT_LABEL, fill=BLUE_DARK)
    label_box(base, (827, 246, 1125, 296), "cross-modal attention", BLUE_DARK, "small")
    label_box(base, (827, 514, 1125, 564), "feed-forward + residual", BLUE_DARK, "small")
    label_box(base, (890, 744, 1063, 791), "risk token", BLUE_DARK, "small")

    label_box(base, (1292, 121, 1412, 163), "rank-refit\ncalibration", (185, 110, 35), "tiny")
    label_box(base, (1292, 267, 1412, 309), "patient-level\nrisk score", RED, "tiny")
    label_box(base, (1292, 471, 1412, 513), "ROC / AUC", BLUE, "tiny")
    label_box(base, (1292, 626, 1412, 668), "calibration\nDCA", GREEN, "tiny")
    label_box(base, (1292, 778, 1412, 820), "SHAP\nbeeswarm", PURPLE, "tiny")

    draw = composite_box(base, (1360, 845, 1635, 895), fill=(255, 255, 255, 238), outline=GRAY, radius=12, width=2)
    multiline_center(draw, (1360, 845, 1635, 895), "locked model\nbedside deployment", FONT_TINY, fill=GRAY)

    draw = ImageDraw.Draw(base)
    draw.text((40, 900), "AI background generated by GPTimage2; exact model modules added programmatically.", font=FONT_TINY, fill=GRAY)
    base.convert("RGB").save(ARCH_OUT, quality=96)


def main() -> None:
    make_workflow()
    make_architecture()
    print(WORKFLOW_OUT)
    print(ARCH_OUT)


if __name__ == "__main__":
    main()

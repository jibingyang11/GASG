from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RIGHTVIEW_SAMPLE = ROOT / "results" / "rightview_generation" / "qualitative" / "sample_00000.png"
DEPTH_SAMPLE = ROOT / "results" / "pseudostereo_depth" / "visuals" / "gasg_defom" / "00000_pred.png"
OUT_PATHS = [
    ROOT / "paper_assets" / "final_paper" / "figures" / "Fig1_overall_pipeline.png",
    ROOT / "paper_mdpi_applied_sciences" / "figures" / "Fig1_overall_pipeline.png",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = load_font(58, bold=True)
FONT_HEAD = load_font(42, bold=True)
FONT_BODY = load_font(32, bold=False)
FONT_SMALL = load_font(26, bold=False)
FONT_SMALL_BOLD = load_font(27, bold=True)


def find_image_tiles(grid: Image.Image) -> list[Image.Image]:
    arr = np.asarray(grid.convert("RGB"))
    # Ignore the title row and detect non-white image panels.
    sub = arr[50:, :, :]
    mask = np.any(sub < 242, axis=2)
    proj_x = mask.sum(axis=0)
    active = proj_x > 80
    regions: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        if (not flag or i == len(active) - 1) and start is not None:
            end = i if not flag else i + 1
            if end - start > 180:
                regions.append((start, end))
            start = None

    tiles: list[Image.Image] = []
    for x0, x1 in regions:
        crop_mask = mask[:, x0:x1]
        proj_y = crop_mask.sum(axis=1)
        ys = np.where(proj_y > 80)[0]
        y0 = int(ys[0]) + 50
        y1 = int(ys[-1]) + 51
        tiles.append(grid.crop((x0, y0, x1, y1)))
    return tiles


def cover_resize(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = im.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = im.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def paste_rounded(canvas: Image.Image, im: Image.Image, box: tuple[int, int, int, int], radius: int = 26) -> None:
    x0, y0, x1, y1 = box
    im = cover_resize(im.convert("RGB"), (x1 - x0, y1 - y0))
    mask = Image.new("L", im.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, im.width, im.height), radius=radius, fill=255)
    canvas.paste(im, (x0, y0), mask)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=radius, outline=(82, 92, 105), width=5)


def draw_center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], font, fill) -> None:
    x0, y0, x1, y1 = box
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + (len(lines) - 1) * 12
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=font, fill=fill)
        y += h + 12


def rounded_module(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title: str,
    body: list[str],
) -> None:
    draw.rounded_rectangle(box, radius=32, fill=fill, outline=outline, width=5)
    x0, y0, x1, y1 = box
    draw.text((x0 + 34, y0 + 28), title, font=FONT_HEAD, fill=(20, 28, 38))
    y = y0 + 95
    for line in body:
        draw.text((x0 + 38, y), line, font=FONT_BODY, fill=(45, 55, 72))
        y += 45


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color, width: int = 10) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 32
    spread = math.radians(24)
    p1 = (x1 - head * math.cos(angle - spread), y1 - head * math.sin(angle - spread))
    p2 = (x1 - head * math.cos(angle + spread), y1 - head * math.sin(angle + spread))
    draw.polygon([(x1, y1), p1, p2], fill=color)


def draw_elbow_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color, width: int = 10) -> None:
    for a, b in zip(points[:-2], points[1:-1]):
        draw.line((*a, *b), fill=color, width=width)
    draw_arrow(draw, points[-2], points[-1], color, width)


def draw_camera(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rounded_rectangle((x, y + 30, x + 155, y + 118), radius=20, fill=(255, 255, 255), outline=(65, 75, 92), width=5)
    draw.rectangle((x + 36, y + 12, x + 94, y + 34), fill=(65, 75, 92))
    draw.ellipse((x + 55, y + 47, x + 124, y + 116), fill=(47, 124, 246), outline=(27, 68, 135), width=5)
    draw.ellipse((x + 77, y + 69, x + 102, y + 94), fill=(255, 255, 255))


def main() -> None:
    rightview_grid = Image.open(RIGHTVIEW_SAMPLE)
    tiles = find_image_tiles(rightview_grid)
    if len(tiles) < 3:
        raise RuntimeError(f"Expected at least three image tiles in {RIGHTVIEW_SAMPLE}, found {len(tiles)}")
    left_img = tiles[0]
    gasg_right = tiles[2]
    depth_img = Image.open(DEPTH_SAMPLE)

    W, H = 3600, 1500
    canvas = Image.new("RGB", (W, H), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)

    # Soft background bands.
    draw.rounded_rectangle((80, 90, 3520, 1345), radius=54, fill=(255, 255, 255), outline=(226, 232, 240), width=4)
    draw.rectangle((80, 1040, 3520, 1345), fill=(246, 248, 251))

    title = "GASG Pseudo-Stereo Pipeline for Single-Camera Absolute Depth"
    tw = draw.textbbox((0, 0), title, font=FONT_TITLE)[2]
    draw.text(((W - tw) / 2, 30), title, font=FONT_TITLE, fill=(20, 28, 38))

    # Camera and input image.
    draw_camera(draw, 185, 170)
    draw.text((365, 190), "Single physical camera", font=FONT_HEAD, fill=(20, 28, 38))
    draw.text((365, 247), "only left view is captured", font=FONT_BODY, fill=(75, 85, 99))
    left_box = (160, 470, 800, 755)
    paste_rounded(canvas, left_img, left_box)
    draw.text((295, 410), "Left image  I_L", font=FONT_HEAD, fill=(20, 28, 38))
    draw_arrow(draw, (360, 305), (360, 455), (82, 92, 105), 9)

    # Missing right camera note.
    draw.rounded_rectangle((100, 820, 860, 940), radius=28, fill=(255, 247, 237), outline=(251, 146, 60), width=4)
    draw.text((135, 842), "No second camera required", font=FONT_SMALL_BOLD, fill=(154, 52, 18))
    draw.text((135, 884), "GASG learns the missing right view.", font=FONT_SMALL, fill=(124, 45, 18))

    # GASG module and generated pseudo right image.
    gasg_box = (1000, 270, 1740, 525)
    rounded_module(
        draw,
        gasg_box,
        fill=(236, 253, 245),
        outline=(16, 185, 129),
        title="GASG  G(.)",
        body=[
            "task-specific left-to-right",
            "pseudo-stereo generator",
            "lightweight and plug-in ready",
        ],
    )
    draw.text((1120, 610), "Pseudo right image  I_R_hat", font=FONT_HEAD, fill=(20, 28, 38))
    pseudo_box = (1045, 675, 1685, 960)
    paste_rounded(canvas, gasg_right, pseudo_box)
    draw_arrow(draw, (800, 580), (990, 395), (16, 185, 129), 10)
    draw_arrow(draw, (1370, 525), (1370, 660), (16, 185, 129), 10)

    # Stereo estimator module.
    stereo_box = (1985, 420, 2775, 875)
    draw.rounded_rectangle(stereo_box, radius=34, fill=(239, 246, 255), outline=(37, 99, 235), width=5)
    draw.text((2045, 455), "Any stereo depth estimator  S(.,.)", font=FONT_HEAD, fill=(20, 28, 38))
    draw.text((2047, 515), "input:  (I_L, I_R_hat)", font=FONT_BODY, fill=(45, 55, 72))
    methods = [
        "DEFOM-Stereo",
        "RAFT-Stereo",
        "FoundationStereo",
        "Stereo Anything",
        "other left-right stereo models",
    ]
    y = 585
    for i, method in enumerate(methods):
        x = 2048 + (i % 2) * 340
        yy = y + (i // 2) * 68
        w = 305 if i < 4 else 650
        draw.rounded_rectangle((x, yy, x + w, yy + 48), radius=14, fill=(255, 255, 255), outline=(147, 197, 253), width=3)
        draw_center_text(draw, (x, yy, x + w, yy + 48), [method], FONT_SMALL, (30, 64, 175))

    # Two input arrows to the stereo module.
    draw_elbow_arrow(draw, [(800, 610), (900, 610), (900, 565), (1910, 565), (1980, 565)], (37, 99, 235), 10)
    draw_elbow_arrow(draw, [(1685, 820), (1910, 820), (1980, 710)], (37, 99, 235), 10)
    draw.text((1760, 520), "left input", font=FONT_SMALL, fill=(30, 64, 175))
    draw.text((1740, 835), "generated right input", font=FONT_SMALL, fill=(30, 64, 175))

    # Depth output.
    draw_arrow(draw, (2775, 650), (2935, 650), (217, 119, 6), 11)
    draw.text((3010, 390), "Absolute depth map  Z_hat", font=FONT_HEAD, fill=(20, 28, 38))
    depth_box = (2925, 455, 3485, 705)
    paste_rounded(canvas, depth_img, depth_box)
    draw.rounded_rectangle((2940, 760, 3470, 900), radius=28, fill=(255, 251, 235), outline=(245, 158, 11), width=4)
    draw_center_text(
        draw,
        (2960, 770, 3450, 892),
        ["metric stereo-style depth", "from a one-camera input"],
        FONT_BODY,
        (120, 53, 15),
    )

    # Bottom summary strip.
    steps = [
        ("1", "Capture only I_L", (160, 1120, 760, 1270), (241, 245, 249), (100, 116, 139)),
        ("2", "GASG generates I_R_hat", (900, 1120, 1500, 1270), (236, 253, 245), (16, 185, 129)),
        ("3", "Use any stereo estimator", (1640, 1120, 2240, 1270), (239, 246, 255), (37, 99, 235)),
        ("4", "Predict absolute depth", (2380, 1120, 2980, 1270), (255, 251, 235), (245, 158, 11)),
    ]
    for idx, text, box, fill, outline in steps:
        draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline, width=4)
        x0, y0, x1, y1 = box
        draw.ellipse((x0 + 28, y0 + 38, x0 + 88, y0 + 98), fill=outline)
        draw_center_text(draw, (x0 + 28, y0 + 38, x0 + 88, y0 + 98), [idx], FONT_SMALL_BOLD, (255, 255, 255))
        draw.text((x0 + 115, y0 + 53), text, font=FONT_BODY, fill=(30, 41, 59))
    for a, b in zip(steps[:-1], steps[1:]):
        draw_arrow(draw, (a[2][2] + 20, 1195), (b[2][0] - 20, 1195), (100, 116, 139), 7)

    formula = "I_R_hat = GASG(I_L)       Z_hat = StereoDepth(I_L, I_R_hat)"
    fw = draw.textbbox((0, 0), formula, font=FONT_BODY)[2]
    draw.text(((W - fw) / 2, 1310), formula, font=FONT_BODY, fill=(51, 65, 85))

    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, "PNG", dpi=(300, 300))
        print(out)


if __name__ == "__main__":
    main()

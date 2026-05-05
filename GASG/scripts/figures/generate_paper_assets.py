"""Build the final paper figures and LaTeX tables for the GASG paper.

The assets generated here are intentionally focused on the paper's central
claim: GASG is a left-to-right pseudo-stereo generator that can be plugged
into stereo depth estimators.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.person_dataset import PersonDataset
from scripts.eval_rightview_compare import (
    ConstantShiftGenerator,
    DAV2WarpGenerator,
    GASGGenerator,
    MoGeWarpGenerator,
    left_tensor_to_01,
    to_numpy_image,
)
from utils.visualization import colorize_depth


OUT_ROOT = ROOT / "paper_assets" / "final_paper"
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
DATA_DIR = OUT_ROOT / "data"
PAPER_DIR = ROOT / "paper_mdpi_applied_sciences"
PAPER_FIGURES = [
    "Fig1_overall_pipeline.png",
    "Fig2_gasg_architecture.png",
    "Fig3_table2_depth_metrics.png",
    "Fig4_rightview_quality.png",
    "Fig5_rightview_metrics.png",
    "Fig6_gasg_ablation.png",
    "Fig7_depth_visual.png",
    "Fig8_depth_error_visual.png",
]
PAPER_TABLES = [
    "Table1_rightview_generation.tex",
    "Table2_pseudostereo_depth.tex",
    "Table3_backend_generalization.tex",
]
STANDARD_FIG_SIZE = (3600, 1500)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = load_font(42, True)
FONT_HEAD = load_font(30, True)
FONT_BODY = load_font(24, False)
FONT_SMALL = load_font(20, False)
FONT_TINY = load_font(17, False)


def reset_dirs() -> None:
    for d in [FIG_DIR, TAB_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_file():
                p.unlink()


def pil_from_np(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def load_rgb(path: Path, size: Tuple[int, int] | None = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return im


def cover(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    tw, th = size
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    r = im.resize((int(sw * scale), int(sh * scale)), Image.Resampling.LANCZOS)
    x = (r.width - tw) // 2
    y = (r.height - th) // 2
    return r.crop((x, y, x + tw, y + th))


def fit(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    tw, th = size
    out = Image.new("RGB", size, (255, 255, 255))
    r = im.copy()
    r.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(r, ((tw - r.width) // 2, (th - r.height) // 2))
    return out


def standardize_canvas(path: Path, size: Tuple[int, int] = STANDARD_FIG_SIZE) -> None:
    target_w, target_h = size
    im = Image.open(path).convert("RGB")
    scale = min(target_w / im.width, target_h / im.height)
    new_w = max(1, int(round(im.width * scale)))
    new_h = max(1, int(round(im.height * scale)))
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    canvas.save(path)


def draw_label(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str,
               font=FONT_SMALL, fill=(20, 28, 38)) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2),
              text, font=font, fill=fill)


def make_grid(
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    out_path: Path,
    title: str,
    tile_size: Tuple[int, int] = (320, 142),
    title_h: int = 78,
    label_h: int = 44,
    gap: int = 14,
) -> None:
    cols = max(len(r) for r in rows)
    tw, th = tile_size
    w = cols * tw + (cols + 1) * gap
    h = title_h + len(rows) * (label_h + th + gap) + gap
    canvas = Image.new("RGB", (w, h), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (0, 10, w, title_h - 8), title, FONT_TITLE)
    y = title_h
    for row in rows:
        x = gap
        for label, im in row:
            draw.rounded_rectangle((x, y, x + tw, y + label_h),
                                   radius=10, fill=(241, 245, 249),
                                   outline=(226, 232, 240), width=2)
            draw_label(draw, (x + 4, y, x + tw - 4, y + label_h), label, FONT_SMALL)
            tile = fit(im.convert("RGB"), (tw, th))
            canvas.paste(tile, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + tw, y + label_h + th),
                           outline=(203, 213, 225), width=2)
            x += tw + gap
        y += label_h + th + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def make_fig2_architecture(out_path: Path) -> None:
    W, H = 3000, 1740
    canvas = Image.new("RGB", (W, H), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((60, 90, W - 60, H - 80), radius=46,
                           fill=(255, 255, 255), outline=(226, 232, 240), width=4)
    draw_label(draw, (0, 20, W, 88), "GASG Architecture: Geometry-Aware Left-to-Right Generation", FONT_TITLE)

    def box(x, y, w, h, title, body, fill, outline):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=22,
                               fill=fill, outline=outline, width=4)
        draw.text((x + 22, y + 18), title, font=FONT_HEAD, fill=(15, 23, 42))
        yy = y + 62
        for line in body:
            draw.text((x + 24, yy), line, font=FONT_SMALL, fill=(51, 65, 85))
            yy += 31

    def arrow(start, end, color=(71, 85, 105), width=8):
        x0, y0 = start
        x1, y1 = end
        draw.line((x0, y0, x1, y1), fill=color, width=width)
        ang = math.atan2(y1 - y0, x1 - x0)
        head = 26
        spread = math.radians(25)
        p1 = (x1 - head * math.cos(ang - spread), y1 - head * math.sin(ang - spread))
        p2 = (x1 - head * math.cos(ang + spread), y1 - head * math.sin(ang + spread))
        draw.polygon([(x1, y1), p1, p2], fill=color)

    # Input and backbone.
    box(120, 245, 330, 170, "Input", ["left image I_L", "single camera"], (248, 250, 252), (100, 116, 139))
    box(560, 190, 680, 280, "Encoder-Decoder Backbone",
        ["stem + texture blocks", "downsampling H/2, H/4, H/8, H/16",
         "bottleneck residual dense/detail blocks", "skip-fused decoder to full resolution"],
        (239, 246, 255), (37, 99, 235))
    box(1340, 245, 340, 170, "Feature", ["shared full-resolution", "representation F"], (238, 242, 255), (99, 102, 241))
    arrow((450, 330), (555, 330), (37, 99, 235))
    arrow((1240, 330), (1335, 330), (37, 99, 235))

    # Branches.
    box(1810, 140, 440, 190, "Disparity Branch", ["disp head -> d", "positive horizontal disparity", "warp: W(I_L, d)"], (236, 253, 245), (16, 185, 129))
    box(1810, 405, 440, 215, "Direct Detail Branch", ["inputs: F, I_L, warped", "and |I_L - warped|", "predicts local color/detail residual"], (255, 247, 237), (249, 115, 22))
    box(1810, 715, 440, 200, "Fusion Mask Branch", ["inputs: F, I_L, warped, direct", "predicts alpha mask", "blend warp and direct"], (254, 249, 195), (202, 138, 4))
    box(1810, 1010, 440, 200, "Final Refinement", ["small residual after fusion", "polishes human boundaries", "and dis-occluded regions"], (253, 242, 248), (219, 39, 119))
    for y in [235, 510, 815, 1110]:
        arrow((1680, 330), (1805, y), (99, 102, 241), 7)

    # Image-state modules.
    box(2380, 170, 430, 155, "Warped Right", ["I_w = W(I_L, d)"], (236, 253, 245), (16, 185, 129))
    box(2380, 440, 430, 155, "Direct Right", ["I_d = I_w + residual"], (255, 247, 237), (249, 115, 22))
    box(2380, 725, 430, 170, "Learned Blend", ["I_f = (1-alpha) I_w + alpha I_d"], (254, 249, 195), (202, 138, 4))
    box(2380, 1030, 430, 180, "Pseudo Right", ["I_R_hat = I_f + small refinement", "output to any stereo estimator"], (240, 253, 244), (22, 163, 74))
    arrow((2250, 235), (2375, 245), (16, 185, 129))
    arrow((2250, 510), (2375, 520), (249, 115, 22))
    arrow((2250, 815), (2375, 810), (202, 138, 4))
    arrow((2250, 1110), (2375, 1120), (219, 39, 119))
    arrow((2595, 325), (2595, 435), (71, 85, 105), 6)
    arrow((2595, 595), (2595, 720), (71, 85, 105), 6)
    arrow((2595, 895), (2595, 1025), (71, 85, 105), 6)

    # Bottom notes.
    notes = [
        ("Geometry", "explicit disparity-like warp preserves stereo correspondence"),
        ("Texture", "direct/detail branch repairs occlusion and pedestrian boundaries"),
        ("Fusion", "learned alpha chooses between warp and synthesis per pixel"),
        ("Plug-in", "the final right view keeps the standard left-right stereo interface"),
    ]
    x = 170
    for head, text in notes:
        draw.rounded_rectangle((x, 1390, x + 640, 1528), radius=22,
                               fill=(248, 250, 252), outline=(203, 213, 225), width=3)
        draw.text((x + 26, 1412), head, font=FONT_HEAD, fill=(15, 23, 42))
        draw.text((x + 26, 1462), text, font=FONT_SMALL, fill=(51, 65, 85))
        x += 690

    formula = "I_R_hat = GASG(I_L);    depth = StereoEstimator(I_L, I_R_hat)"
    draw_label(draw, (0, 1580, W, 1645), formula, FONT_BODY, (30, 41, 59))
    canvas.save(out_path, dpi=(300, 300))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rightview_rows() -> List[Dict]:
    rows = []
    with open(ROOT / "results" / "rightview_generation" / "summary.csv",
              encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for k in ["psnr", "ssim", "lpips", "inference_time_ms_mean",
                      "params_M", "n_eval"]:
                out[k] = float(out[k])
            rows.append(out)
    return rows


def depth_rows() -> List[Dict]:
    order = [
        "defom_identity",
        "defom_shift",
        "defom_dav2_warp",
        "defom_deep3d_svsm",
        "defom_pseudostereo_3dod",
        "defom_mono2stereo_dibr",
        "defom_moge_warp",
        "defom_zerostereo_tuned",
        "gasg_defom",
        "defom_real",
    ]
    rows = []
    root = ROOT / "results" / "pseudostereo_depth"
    for method in order:
        p = root / f"{method}.json"
        if p.exists():
            rows.append(load_json(p))
    return rows


def plot_rightview_bars(rows: List[Dict], out_path: Path) -> None:
    names = {
        "identity": "Copy-left",
        "shift_const": "Const shift",
        "dav2_warp": "DA-V2 warp",
        "moge_warp": "MoGe warp",
        "zerostereo_person_tuned": "ZeroStereo tuned",
        "gasg_full": "GASG",
    }
    keep = ["identity", "shift_const", "dav2_warp", "moge_warp",
            "zerostereo_person_tuned", "gasg_full"]
    by = {r["method"]: r for r in rows}
    rows = [by[m] for m in keep if m in by]
    labels = [names[r["method"]] for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.2), dpi=260, sharey=True)
    specs = [
        ("psnr", "PSNR (dB) up", "#2f6fbb", "{:.2f}"),
        ("ssim", "SSIM up", "#2f9e78", "{:.3f}"),
        ("lpips", "LPIPS down", "#d97706", "{:.3f}"),
    ]
    for ax, (key, title, color, fmt) in zip(axes, specs):
        vals = [float(r[key]) for r in rows]
        ax.barh(y, vals, color=color, height=0.62)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        xmax = max(vals) * 1.20 if max(vals) > 0 else 1
        ax.set_xlim(0, xmax)
        pad = xmax * 0.018
        for i, v in enumerate(vals):
            ax.text(v + pad, i, fmt.format(v), va="center", fontsize=6.4)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out_path)
    plt.close(fig)


def load_literature_rightview_rows() -> List[Dict]:
    path = ROOT / "results" / "literature_rightview_generation" / "summary.csv"
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in [
                "n_eval",
                "psnr",
                "ssim",
                "lpips",
                "inference_time_ms_mean",
                "inference_time_ms_median",
                "params_M",
            ]:
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def plot_literature_rightview_bars(out_path: Path) -> None:
    rows = load_literature_rightview_rows()
    if not rows:
        return
    names = {
        "identity": "Copy-left",
        "deep3d_svsm": "Deep3D/SVSM",
        "pseudostereo_3dod": "Pseudo-3DOD",
        "mono2stereo_dibr": "M2S-DIBR",
        "dav2_warp": "DA-V2",
        "moge_warp": "MoGe",
        "zerostereo_person_tuned": "ZeroStereo",
        "gasg_full": "GASG",
    }
    labels = [names.get(str(r["method"]), str(r["method"])) for r in rows]
    y = np.arange(len(rows))
    colors = ["#2f6fbb", "#2f9e78", "#d97706", "#7c3aed"]
    specs = [
        ("psnr", "PSNR (dB) up", "{:.2f}", False),
        ("ssim", "SSIM up", "{:.3f}", False),
        ("lpips", "LPIPS down", "{:.3f}", False),
        ("inference_time_ms_mean", "Time (ms) down", "{:.1f}", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 4.8), dpi=260, sharey=True)
    for ax, (key, title, fmt, log_axis), color in zip(axes, specs, colors):
        vals = [float(r[key]) for r in rows]
        ax.barh(y, vals, color=color, height=0.60)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if log_axis:
            ax.set_xscale("log")
            xmax = max(vals) * 1.9
            xmin = max(min(v for v in vals if v > 0) * 0.45, 0.05)
            ax.set_xlim(xmin, xmax)
            for i, v in enumerate(vals):
                ax.text(v * 1.08, i, fmt.format(v), va="center", fontsize=6.2)
        else:
            xmax = max(vals) * 1.22 if max(vals) > 0 else 1
            ax.set_xlim(0, xmax)
            pad = xmax * 0.018
            for i, v in enumerate(vals):
                ax.text(v + pad, i, fmt.format(v), va="center", fontsize=6.2)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.7)
    fig.savefig(out_path)
    plt.close(fig)


def plot_depth_bars(rows: List[Dict], out_path: Path) -> None:
    names = {
        "defom_identity": "Copy-left",
        "defom_shift": "Const shift",
        "defom_dav2_warp": "DA-V2 warp",
        "defom_deep3d_svsm": "Deep3D/SVSM",
        "defom_pseudostereo_3dod": "PseudoStereo",
        "defom_mono2stereo_dibr": "M2S-DIBR",
        "defom_moge_warp": "MoGe warp",
        "defom_zerostereo_tuned": "ZeroStereo tuned",
        "gasg_defom": "GASG",
        "defom_real": "True-right",
    }
    labels = [names.get(r["method"], r["method"]) for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.4), dpi=260, sharey=True)
    specs = [
        ("abs_rel", "AbsRel down", "#2f6fbb", "{:.3f}"),
        ("rmse", "RMSE down", "#d97706", "{:.2f}"),
        ("d1", "delta1 up", "#2f9e78", "{:.3f}"),
    ]
    for ax, (key, title, color, fmt) in zip(axes, specs):
        vals = [float(r[key]) for r in rows]
        ax.barh(y, vals, color=color, height=0.62)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        xmax = max(vals) * 1.20 if max(vals) > 0 else 1
        ax.set_xlim(0, xmax)
        pad = xmax * 0.018
        for i, v in enumerate(vals):
            ax.text(v + pad, i, fmt.format(v), va="center", fontsize=6.4)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out_path)
    plt.close(fig)


def create_visual_figures() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PersonDataset(str(ROOT / "data" / "PersonDataset"), split="test",
                       height=256, width=576)
    sample = ds[0]
    left = sample["left"].unsqueeze(0)
    gt_right = (sample["right"] + 1.0) * 0.5
    zmap = load_json(ROOT / "data" / "zerostereo_person_tuned_test" / "metadata.json")
    zpath = None
    for rec in zmap["records"]:
        if int(rec["person_index"]) == 0:
            zpath = ROOT / "data" / "zerostereo_person_tuned_test" / rec["right"]
            break

    calib_path = ROOT / "results" / "rightview_generation" / "warp_calibration.json"
    calib = load_json(calib_path) if calib_path.exists() else {}
    dav2_disp = float(calib.get("dav2_warp", {}).get("target_disp", 18.0))
    moge_disp = float(calib.get("moge_warp", {}).get("target_disp", 24.0))

    shift = ConstantShiftGenerator(device, disp_pixels=9.0)
    dav2 = DAV2WarpGenerator(device, target_max_disp=dav2_disp)
    moge = MoGeWarpGenerator(device, target_max_disp=moge_disp)
    gasg_full = GASGGenerator(str(ROOT / "checkpoints" / "gasg_best.pth"),
                              device, "full", gamma=1.0)
    gasg_warp = GASGGenerator(str(ROOT / "checkpoints" / "gasg_best.pth"),
                              device, "warp_only", gamma=1.0)
    gasg_direct = GASGGenerator(str(ROOT / "checkpoints" / "gasg_best.pth"),
                                device, "direct_only", gamma=1.0)
    gasg_norefine = GASGGenerator(str(ROOT / "checkpoints" / "gasg_best.pth"),
                                  device, "no_refine", gamma=1.0)

    with torch.no_grad():
        left_img = pil_from_np(to_numpy_image(left_tensor_to_01(left)))
        gt_img = pil_from_np(to_numpy_image(gt_right))
        shift_img = pil_from_np(to_numpy_image(shift(left)))
        dav2_img = pil_from_np(to_numpy_image(dav2(left)))
        moge_img = pil_from_np(to_numpy_image(moge(left)))
        gasg_img = pil_from_np(to_numpy_image(gasg_full(left)))
        warp_img = pil_from_np(to_numpy_image(gasg_warp(left)))
        direct_img = pil_from_np(to_numpy_image(gasg_direct(left)))
        norefine_img = pil_from_np(to_numpy_image(gasg_norefine(left)))

    right_rows = [[
        ("Left", left_img),
        ("GT right", gt_img),
        ("Const shift", shift_img),
        ("DA-V2 cal.", dav2_img),
        ("MoGe cal.", moge_img),
        ("ZeroStereo FT", load_rgb(zpath, (576, 256))),
        ("GASG", gasg_img),
    ]]
    make_grid(right_rows, FIG_DIR / "Fig3_rightview_quality.png",
              "Right-View Generation Quality on PersonDataset")
    qualitative_dir = ROOT / "results" / "rightview_generation" / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    make_grid(
        [[("Left", left_img), ("GT right", gt_img), ("GASG", gasg_img)]],
        qualitative_dir / "sample_00000.png",
        "GASG Pseudo-Right View Example",
    )

    ablation_rows = [
        [
            ("Left", left_img),
            ("GT right", gt_img),
            ("Warp-only", warp_img),
        ],
        [
            ("Direct-only", direct_img),
            ("No refine", norefine_img),
            ("Full GASG", gasg_img),
        ],
    ]
    make_grid(ablation_rows, FIG_DIR / "Fig4_gasg_ablation.png",
              "GASG Component Ablation")

    # Depth predictions and errors.
    vis_root = ROOT / "results" / "pseudostereo_depth" / "visuals"
    def depth_item(label: str, method: str, suffix: str = "pred") -> Tuple[str, Image.Image] | None:
        p = vis_root / method / f"00000_{suffix}.png"
        if not p.exists():
            return None
        return label, load_rgb(p, (576, 256))

    def error_item(label: str, method: str) -> Tuple[str, Image.Image] | None:
        p = vis_root / method / "00000_error.png"
        if not p.exists():
            return None
        return label, load_rgb(p, (576, 256))

    left_depth = ("Left", load_rgb(vis_root / "gasg_defom" / "00000_left.png", (576, 256)))
    depth_items = [
        left_depth,
        depth_item("GT depth", "gasg_defom", "gt"),
        depth_item("Copy-left", "defom_identity"),
        depth_item("DA-V2", "defom_dav2_warp"),
        depth_item("MoGe", "defom_moge_warp"),
        depth_item("Deep3D/SVSM", "defom_deep3d_svsm"),
        depth_item("PseudoStereo", "defom_pseudostereo_3dod"),
        depth_item("M2S-DIBR", "defom_mono2stereo_dibr"),
        depth_item("ZeroStereo FT", "defom_zerostereo_tuned"),
        depth_item("GASG", "gasg_defom"),
        depth_item("True-right", "defom_real"),
    ]
    err_items = [
        left_depth,
        error_item("Copy-left", "defom_identity"),
        error_item("DA-V2", "defom_dav2_warp"),
        error_item("MoGe", "defom_moge_warp"),
        error_item("Deep3D/SVSM", "defom_deep3d_svsm"),
        error_item("PseudoStereo", "defom_pseudostereo_3dod"),
        error_item("M2S-DIBR", "defom_mono2stereo_dibr"),
        error_item("ZeroStereo FT", "defom_zerostereo_tuned"),
        error_item("GASG", "gasg_defom"),
        error_item("True-right", "defom_real"),
    ]
    depth_items = [item for item in depth_items if item is not None]
    err_items = [item for item in err_items if item is not None]
    depth_rows = [depth_items[i:i + 4] for i in range(0, len(depth_items), 4)]
    err_rows = [err_items[i:i + 4] for i in range(0, len(err_items), 4)]
    make_grid(depth_rows, FIG_DIR / "Fig5_depth_visual.png",
              "Pseudo-Stereo Depth Prediction Comparison", tile_size=(280, 124))
    make_grid(err_rows, FIG_DIR / "Fig6_depth_error_visual.png",
              "Absolute Depth Error Comparison", tile_size=(280, 124))


def tex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_")


def write_table(path: Path, caption: str, label: str, header: Sequence[str],
                rows: Sequence[Sequence[str]]) -> None:
    cols = "l" + "r" * (len(header) - 1)
    tabcolsep = "3.2pt" if len(header) >= 8 else "4.5pt"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\makebox[\\linewidth][c]{%",
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_tables(right_rows: List[Dict], drows: List[Dict]) -> None:
    rv_names = {
        "identity": "Copy-left",
        "shift_const": "Constant shift",
        "dav2_warp": "DA-V2 warp",
        "moge_warp": "MoGe warp",
        "zerostereo_person_tuned": "ZeroStereo/StereoGen FT",
        "gasg_full": "GASG",
    }
    keep = ["identity", "shift_const", "dav2_warp", "moge_warp",
            "zerostereo_person_tuned", "gasg_full"]
    by = {r["method"]: r for r in right_rows}
    table = []
    for m in keep:
        if m not in by:
            continue
        r = by[m]
        table.append([
            tex_escape(rv_names[m]),
            str(int(float(r.get("n_eval", 0)))),
            f"{float(r['psnr']):.2f}",
            f"{float(r['ssim']):.4f}",
            f"{float(r['lpips']):.4f}",
            f"{float(r.get('inference_time_ms_mean', 0.0)):.1f}",
            f"{float(r.get('params_M', 0.0)):.2f}",
        ])
    write_table(
        TAB_DIR / "Table1_rightview_generation.tex",
        "Right-view generation quality on PersonDataset under the target-domain comparison protocol. ZeroStereo/StereoGen is fine-tuned on the PersonDataset training split; DA-V2 and MoGe use PersonDataset-calibrated warp scales; GASG uses the trained task-specialized checkpoint.",
        "tab:rightview_generation",
        ["Method", "N", "PSNR$\\uparrow$", "SSIM$\\uparrow$", "LPIPS$\\downarrow$", "Time$\\downarrow$", "Params$\\downarrow$"],
        table,
    )

    dnames = {
        "defom_identity": "Copy-left",
        "defom_shift": "Shift",
        "defom_dav2_warp": "DA-V2 cal.",
        "defom_deep3d_svsm": "Deep3D/SVSM",
        "defom_pseudostereo_3dod": "PseudoStereo",
        "defom_mono2stereo_dibr": "M2S-DIBR",
        "defom_moge_warp": "MoGe cal.",
        "defom_zerostereo_tuned": "ZeroStereo FT",
        "gasg_defom": "GASG",
        "defom_real": "True right",
    }
    table = []
    for r in drows:
        table.append([
            dnames.get(r["method"], tex_escape(r["pretty"])),
            str(int(r.get("n_eval", 0))),
            f"{float(r['abs_rel']):.4f}",
            f"{float(r['sq_rel']):.4f}",
            f"{float(r['rmse']):.4f}",
            f"{float(r['rmse_log']):.4f}",
            f"{float(r['d1']):.4f}",
            f"{float(r['d2']):.4f}",
            f"{float(r['d3']):.4f}",
        ])
    write_table(
        TAB_DIR / "Table2_pseudostereo_depth.tex",
        "Pseudo-stereo depth comparison on PersonDataset. All rows use DEFOM-Stereo as the same downstream stereo estimator; only the right-view source changes. Rlog denotes RMSElog.",
        "tab:pseudostereo_depth",
        ["Source", "N", "AbsRel$\\downarrow$", "SqRel$\\downarrow$", "RMSE$\\downarrow$", "Rlog$\\downarrow$", "$\\delta_1\\uparrow$", "$\\delta_2\\uparrow$", "$\\delta_3\\uparrow$"],
        table,
    )

    ab_keep = ["gasg_warp_only", "gasg_direct_only", "gasg_no_refine", "gasg_full"]
    ab_names = {
        "gasg_warp_only": "Warp-only",
        "gasg_direct_only": "Direct-only",
        "gasg_no_refine": "No final refinement",
        "gasg_full": "Full GASG",
    }
    table = []
    for m in ab_keep:
        if m not in by:
            continue
        r = by[m]
        table.append([
            ab_names[m],
            f"{float(r['psnr']):.2f}",
            f"{float(r['ssim']):.4f}",
            f"{float(r['lpips']):.4f}",
            f"{float(r.get('inference_time_ms_mean', 0.0)):.1f}",
        ])
    write_table(
        TAB_DIR / "Table3_gasg_ablation.tex",
        "GASG right-view synthesis ablation on PersonDataset. All variants use the same trained checkpoint and differ only in the inference assembly of the final right image.",
        "tab:gasg_ablation",
        ["Variant", "PSNR$\\uparrow$", "SSIM$\\uparrow$", "LPIPS$\\downarrow$", "Time (ms)$\\downarrow$"],
        table,
    )


def copy_backend_generalization_assets() -> None:
    result_root = ROOT / "results" / "stereo_backend_generalization"
    extra_fig = result_root / "figures" / "Fig9_multibackend_depth.png"
    extra_tab = result_root / "Table4_backend_generalization.tex"
    extra_csv = result_root / "summary.csv"
    if extra_fig.exists():
        shutil.copy2(extra_fig, FIG_DIR / extra_fig.name)
    if extra_tab.exists():
        shutil.copy2(extra_tab, TAB_DIR / extra_tab.name)
    if extra_csv.exists():
        shutil.copy2(extra_csv, DATA_DIR / "stereo_backend_generalization_summary.csv")


def copy_literature_rightview_assets() -> None:
    result_root = ROOT / "results" / "literature_rightview_generation"
    extra_fig = result_root / "figures" / "Fig10_literature_rightview.png"
    extra_tab = result_root / "Table5_literature_rightview.tex"
    extra_csv = result_root / "summary.csv"
    if extra_fig.exists():
        shutil.copy2(extra_fig, FIG_DIR / extra_fig.name)
    if extra_tab.exists():
        shutil.copy2(extra_tab, TAB_DIR / extra_tab.name)
    if extra_csv.exists():
        shutil.copy2(extra_csv, DATA_DIR / "literature_rightview_summary.csv")


def prepare_ordered_paper_assets() -> None:
    """Rename selected generated assets into the final paper numbering."""

    figure_renames = {
        "Fig10_literature_rightview.png": "Fig4_rightview_quality.png",
        "Fig4_gasg_ablation.png": "Fig6_gasg_ablation.png",
        "Fig5_depth_visual.png": "Fig7_depth_visual.png",
        "Fig6_depth_error_visual.png": "Fig8_depth_error_visual.png",
    }
    for src_name, dst_name in figure_renames.items():
        src = FIG_DIR / src_name
        if src.exists():
            shutil.copy2(src, FIG_DIR / dst_name)

    lit_table = TAB_DIR / "Table5_literature_rightview.tex"
    if lit_table.exists():
        tex = lit_table.read_text(encoding="utf-8")
        tex = tex.replace(
            "\\caption{Comparison with representative left-to-right generation methods on PersonDataset. Deep3D/SVSM-style denotes a PersonDataset-trained reproduction of the probabilistic disparity-selection branch used by Deep3D and Single View Stereo Matching. Pseudo-Stereo-3DOD and Mono2Stereo-DIBR denote reproducible image-level virtual-view routes based on published pseudo-stereo/warp-and-inpaint principles. ZeroStereo/StereoGen FT denotes the official StereoGen pipeline fine-tuned on PersonDataset.}",
            "\\caption{Right-view generation quality and efficiency on PersonDataset. Deep3D/SVSM-style, Pseudo-Stereo-3DOD and Mono2Stereo-DIBR are reproducible left-to-right generation routes; ZeroStereo/StereoGen FT is fine-tuned on PersonDataset.}",
        )
        tex = tex.replace("\\label{tab:literature_rightview}", "\\label{tab:rightview_generation}")
        tex = tex.replace("PseudoStereo-3DOD", "Pseudo-Stereo-3DOD")
        (TAB_DIR / "Table1_rightview_generation.tex").write_text(tex, encoding="utf-8")

    backend_table = TAB_DIR / "Table4_backend_generalization.tex"
    if backend_table.exists():
        tex = backend_table.read_text(encoding="utf-8")
        tex = tex.replace(
            "\\caption{Stereo-backend generalization on PersonDataset. Each backend is evaluated with the GASG-generated right view and with the true right view.}",
            "\\caption{Stereo-backend generalization on PersonDataset. Each backend is evaluated with the GASG-generated right view and with the true right view.}",
        )
        (TAB_DIR / "Table3_backend_generalization.tex").write_text(tex, encoding="utf-8")


def copy_to_paper() -> None:
    paper_figs = PAPER_DIR / "figures"
    paper_tabs = PAPER_DIR / "tables"
    paper_figs.mkdir(parents=True, exist_ok=True)
    paper_tabs.mkdir(parents=True, exist_ok=True)
    for p in list(paper_figs.glob("*")):
        if p.is_file():
            p.unlink()
    for p in list(paper_tabs.glob("*")):
        if p.is_file():
            p.unlink()
    for name in PAPER_FIGURES:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, paper_figs / name)
    for name in PAPER_TABLES:
        src = TAB_DIR / name
        if src.exists():
            shutil.copy2(src, paper_tabs / name)


def prune_final_assets() -> None:
    for p in FIG_DIR.glob("*"):
        if p.is_file() and p.name not in PAPER_FIGURES:
            p.unlink()
    for p in TAB_DIR.glob("*"):
        if p.is_file() and p.name not in PAPER_TABLES:
            p.unlink()


def standardize_final_figures() -> None:
    for name in PAPER_FIGURES:
        path = FIG_DIR / name
        if path.exists():
            standardize_canvas(path)


def main() -> None:
    reset_dirs()
    make_fig2_architecture(FIG_DIR / "Fig2_gasg_architecture.png")
    create_visual_figures()
    # Fig. 1 is generated by the existing dedicated script and copied here.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "figures" / "create_overall_pipeline_fig.py")],
        check=True,
    )
    src_fig1 = ROOT / "paper_mdpi_applied_sciences" / "figures" / "Fig1_overall_pipeline.png"
    shutil.copy2(src_fig1, FIG_DIR / "Fig1_overall_pipeline.png")

    right_rows = load_rightview_rows()
    drows = depth_rows()
    plot_rightview_bars(right_rows, FIG_DIR / "Fig7_rightview_metrics.png")
    plot_depth_bars(drows, FIG_DIR / "Fig8_depth_metrics.png")
    shutil.copy2(FIG_DIR / "Fig8_depth_metrics.png",
                 FIG_DIR / "Fig3_table2_depth_metrics.png")
    create_tables(right_rows, drows)
    copy_literature_rightview_assets()
    plot_literature_rightview_bars(FIG_DIR / "Fig5_rightview_metrics.png")
    copy_backend_generalization_assets()
    prepare_ordered_paper_assets()
    prune_final_assets()
    standardize_final_figures()

    shutil.copy2(ROOT / "results" / "rightview_generation" / "summary.csv",
                 DATA_DIR / "rightview_summary.csv")
    shutil.copy2(ROOT / "results" / "pseudostereo_depth" / "summary_depth_models.csv",
                 DATA_DIR / "pseudostereo_depth_summary.csv")
    copy_to_paper()
    print(f"Paper assets written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

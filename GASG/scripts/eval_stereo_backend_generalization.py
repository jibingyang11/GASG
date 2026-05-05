"""Evaluate GASG pseudo-right views with multiple stereo backends.

This script compares two right-view sources for each stereo backend:
  1. GASG pseudo right image generated from the left image.
  2. Ground-truth right image from PersonDataset.

The goal is to test whether GASG preserves a standard left-right interface
beyond the DEFOM-Stereo backend used in the main table.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.gasg_net import GASG
from scripts.eval_rightview_compare import select_indices
from utils.defom_runner import DEFOMRunner
from utils.depth_metrics import aggregate_metrics, compute_depth_errors
from utils.raft_stereo_runner import RAFTStereoRunner
from utils.visualization import colorize_depth


BACKEND_NAMES = {
    "defom": "DEFOM-Stereo",
    "raft": "RAFT-Stereo",
    "sgbm": "OpenCV SGBM",
}

SOURCE_NAMES = {
    "gasg": "GASG right",
    "real": "True right",
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor01_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x.squeeze(0)
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_gasg(checkpoint_path: str, device: torch.device) -> GASG:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    accepted = (
        "base_ch",
        "max_disp",
        "max_vshift",
        "use_vertical_flow",
        "positive_disp",
        "num_lowres_blocks",
        "num_bottleneck_blocks",
        "flow_downsample",
        "refine_ch",
        "coarse_residual_scale",
        "detail_residual_scale",
        "residual_scale",
        "fusion_bias",
        "final_refine_scale",
        "gamma_correction",
    )
    kwargs = {k: model_config[k] for k in accepted if k in model_config}
    model = GASG(**kwargs).to(device)
    model.load_state_dict(ckpt.get("generator", ckpt))
    return model.eval()


class SGBMRunner:
    def __init__(self, num_disparities: int = 160, block_size: int = 5):
        num_disparities = int(np.ceil(num_disparities / 16) * 16)
        block_size = max(3, int(block_size) | 1)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def disparity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_np = tensor01_to_uint8_hwc(left)
        right_np = tensor01_to_uint8_hwc(right)
        left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_np, cv2.COLOR_RGB2GRAY)
        disp = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp = np.maximum(disp, 1e-4)
        return torch.from_numpy(disp).unsqueeze(0)


def make_backend(name: str, args: argparse.Namespace, device: torch.device):
    if name == "defom":
        return DEFOMRunner(
            args.defom_ckpt,
            device,
            valid_iters=args.defom_valid_iters,
            scale_iters=args.defom_scale_iters,
        )
    if name == "raft":
        return RAFTStereoRunner(
            args.raft_ckpt,
            device,
            repo_dir=args.raft_repo,
            valid_iters=args.raft_valid_iters,
            corr_implementation=args.raft_corr,
        )
    if name == "sgbm":
        return SGBMRunner(args.sgbm_num_disparities, args.sgbm_block_size)
    raise ValueError(f"Unsupported backend: {name}")


def save_visual_bundle(
    out_dir: Path,
    tag: str,
    index: int,
    left: torch.Tensor,
    pred: np.ndarray,
    gt: np.ndarray,
    max_depth: float,
) -> None:
    pred_dir = out_dir / "predictions" / tag
    vis_dir = out_dir / "visuals" / tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{index:05d}.npy", pred.astype(np.float32))

    Image.fromarray(tensor01_to_uint8_hwc(left)).save(
        vis_dir / f"{index:05d}_left.png")
    Image.fromarray((colorize_depth(gt, min_d=0.0, max_d=max_depth) * 255)
                    .astype(np.uint8)).save(vis_dir / f"{index:05d}_gt.png")
    Image.fromarray((colorize_depth(pred, min_d=0.0, max_d=max_depth) * 255)
                    .astype(np.uint8)).save(vis_dir / f"{index:05d}_pred.png")
    err = np.abs(pred - gt)
    Image.fromarray((colorize_depth(err, min_d=0.0, max_d=5.0, cmap="inferno")
                    * 255).astype(np.uint8)).save(
        vis_dir / f"{index:05d}_error.png")


def load_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font,
    fill=(15, 23, 42),
    spacing: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - w) / 2, y),
                  line, font=font, fill=fill)
        y += h + spacing


def fit(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (255, 255, 255))
    r = im.convert("RGB")
    r.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(r, ((size[0] - r.width) // 2, (size[1] - r.height) // 2))
    return out


def make_grid(
    rows: List[List[Tuple[str, Image.Image]]],
    out_path: Path,
    title: str,
    tile_size: Tuple[int, int] = (270, 120),
) -> None:
    title_font = load_font(36, True)
    label_font = load_font(18, False)
    gap, title_h, label_h = 12, 70, 34
    cols = max(len(r) for r in rows)
    width = cols * tile_size[0] + (cols + 1) * gap
    height = title_h + len(rows) * (label_h + tile_size[1] + gap) + gap
    canvas = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((width - tw) / 2, 18), title, font=title_font, fill=(15, 23, 42))

    y = title_h
    for row in rows:
        x = gap
        for label, im in row:
            draw.rounded_rectangle((x, y, x + tile_size[0], y + label_h),
                                   radius=8, fill=(241, 245, 249),
                                   outline=(226, 232, 240), width=2)
            lb = draw.textbbox((0, 0), label, font=label_font)
            draw.text((x + (tile_size[0] - (lb[2] - lb[0])) / 2,
                       y + (label_h - (lb[3] - lb[1])) / 2),
                      label, font=label_font, fill=(15, 23, 42))
            canvas.paste(fit(im, tile_size), (x, y + label_h))
            draw.rectangle((x, y + label_h,
                            x + tile_size[0], y + label_h + tile_size[1]),
                           outline=(203, 213, 225), width=2)
            x += tile_size[0] + gap
        y += label_h + tile_size[1] + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def make_backend_comparison_grid(
    rows: List[Tuple[str, List[Tuple[str, Image.Image]]]],
    out_path: Path,
    title: str,
    tile_size: Tuple[int, int] = (300, 133),
) -> None:
    title_font = load_font(34, True)
    header_font = load_font(19, True)
    row_font = load_font(18, True)
    gap = 12
    title_h = 64
    header_h = 36
    row_w = 150
    cols = max(len(items) for _, items in rows)
    width = row_w + cols * tile_size[0] + (cols + 2) * gap
    height = title_h + header_h + len(rows) * (tile_size[1] + gap) + gap
    canvas = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    draw_centered_multiline(draw, (0, 12, width, title_h - 8),
                            title, title_font)

    # Column headers come from the first row and are shared by all rows.
    y = title_h
    draw.rounded_rectangle((gap, y, gap + row_w, y + header_h),
                           radius=8, fill=(226, 232, 240),
                           outline=(203, 213, 225), width=2)
    draw_centered_multiline(draw, (gap, y, gap + row_w, y + header_h),
                            "Input", header_font)
    x = row_w + 2 * gap
    for header, _ in rows[0][1]:
        draw.rounded_rectangle((x, y, x + tile_size[0], y + header_h),
                               radius=8, fill=(226, 232, 240),
                               outline=(203, 213, 225), width=2)
        draw_centered_multiline(draw, (x, y, x + tile_size[0], y + header_h),
                                header, header_font)
        x += tile_size[0] + gap

    y += header_h + gap
    for row_label, items in rows:
        draw.rounded_rectangle((gap, y, gap + row_w, y + tile_size[1]),
                               radius=8, fill=(241, 245, 249),
                               outline=(203, 213, 225), width=2)
        draw_centered_multiline(draw, (gap, y, gap + row_w, y + tile_size[1]),
                                row_label, row_font)
        x = row_w + 2 * gap
        for _, im in items:
            canvas.paste(fit(im, tile_size), (x, y))
            draw.rectangle((x, y, x + tile_size[0], y + tile_size[1]),
                           outline=(203, 213, 225), width=2)
            x += tile_size[0] + gap
        y += tile_size[1] + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def create_visual_figure(args: argparse.Namespace, sample_index: int) -> None:
    out_dir = Path(args.save_dir)
    def img(tag: str, kind: str) -> Image.Image:
        p = out_dir / "visuals" / tag / f"{sample_index:05d}_{kind}.png"
        return Image.open(p).convert("RGB")

    left = img("defom_gasg", "left")
    gt = img("defom_gasg", "gt")
    columns = [
        ("Left image", left),
        ("GT depth", gt),
        ("DEFOM-Stereo", img("defom_gasg", "pred")),
        ("RAFT-Stereo", img("raft_gasg", "pred")),
        ("OpenCV SGBM", img("sgbm_gasg", "pred")),
    ]
    rows = [
        ("Depth\nGASG right", columns),
        ("Depth\ntrue right", [
            ("Left image", left),
            ("GT depth", gt),
            ("DEFOM-Stereo", img("defom_real", "pred")),
            ("RAFT-Stereo", img("raft_real", "pred")),
            ("OpenCV SGBM", img("sgbm_real", "pred")),
        ]),
        ("Error\nGASG right", [
            ("Left image", left),
            ("GT depth", gt),
            ("DEFOM-Stereo", img("defom_gasg", "error")),
            ("RAFT-Stereo", img("raft_gasg", "error")),
            ("OpenCV SGBM", img("sgbm_gasg", "error")),
        ]),
        ("Error\ntrue right", [
            ("Left image", left),
            ("GT depth", gt),
            ("DEFOM-Stereo", img("defom_real", "error")),
            ("RAFT-Stereo", img("raft_real", "error")),
            ("OpenCV SGBM", img("sgbm_real", "error")),
        ]),
    ]
    fig_path = out_dir / "figures" / "Fig9_multibackend_depth.png"
    make_backend_comparison_grid(
        rows, fig_path,
        "Stereo-Backend Generalization: GASG Right vs. True Right",
    )

    root = Path(__file__).resolve().parents[1]
    for dst in [
        root / "paper_assets" / "final_paper" / "figures" / fig_path.name,
        root / "paper_mdpi_applied_sciences" / "figures" / fig_path.name,
    ]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(fig_path).save(dst, dpi=(300, 300))


def write_summaries(out_dir: Path, rows: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = [
        "backend", "right_source", "n_eval", "abs_rel", "sq_rel", "rmse",
        "rmse_log", "d1", "d2", "d3", "inference_time_ms_mean",
    ]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "| Backend | Right source | N | AbsRel | RMSE | delta1 | Time (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {BACKEND_NAMES[r['backend']]} | {SOURCE_NAMES[r['right_source']]} | "
            f"{r['n_eval']} | {r['abs_rel']:.4f} | {r['rmse']:.4f} | "
            f"{r['d1']:.4f} | {r['inference_time_ms_mean']:.1f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(root: Path, rows: List[Dict], out_dir: Path | None = None) -> None:
    by = {(r["backend"], r["right_source"]): r for r in rows}
    order = ["defom", "raft", "sgbm"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Stereo-backend generalization on PersonDataset. Each backend is evaluated with the GASG-generated right view and with the true right view.}",
        "\\label{tab:backend_generalization}",
        "\\small",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\makebox[\\linewidth][c]{%",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "Backend & Source & AbsRel$\\downarrow$ & RMSE$\\downarrow$ & $\\delta_1\\uparrow$ & $\\delta_2\\uparrow$ & Time (ms)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for backend in order:
        for source in ["gasg", "real"]:
            r = by.get((backend, source))
            if r is None:
                continue
            lines.append(
                f"{BACKEND_NAMES[backend]} & {SOURCE_NAMES[source]} & "
                f"{r['abs_rel']:.4f} & {r['rmse']:.4f} & {r['d1']:.4f} & "
                f"{r['d2']:.4f} & {r['inference_time_ms_mean']:.1f} \\\\"
            )
        if backend != order[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    text = "\n".join(lines) + "\n"
    paths = [
        root / "paper_assets" / "final_paper" / "tables" / "Table4_backend_generalization.tex",
        root / "paper_mdpi_applied_sciences" / "tables" / "Table4_backend_generalization.tex",
    ]
    if out_dir is not None:
        paths.append(out_dir / "Table4_backend_generalization.tex")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def evaluate(args: argparse.Namespace) -> List[Dict]:
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.save_dir)
    device = torch.device(args.device)
    ds = PersonDataset(args.person_root, split="test",
                       height=args.height, width=args.width)
    indices = select_indices(len(ds), args.max_samples)
    visual_indices = set()
    if args.save_visuals and indices:
        positions = select_indices(len(indices), min(args.num_visuals, len(indices)))
        visual_indices = {indices[pos] for pos in positions}

    gasg = load_gasg(args.gasg_ckpt, device)
    focal = args.focal_length or 600.0 * float(args.width) / 906.0
    rows: List[Dict] = []

    for backend_name in args.backends.split(","):
        backend_name = backend_name.strip()
        if not backend_name:
            continue
        backend = make_backend(backend_name, args, device)
        for source in ["gasg", "real"]:
            tag = f"{backend_name}_{source}"
            metrics_list = []
            times = []
            for idx in tqdm(indices, desc=f"{BACKEND_NAMES[backend_name]} + {SOURCE_NAMES[source]}"):
                sample = ds[idx]
                left = ((sample["left"].unsqueeze(0) + 1.0) * 0.5).to(device)
                if source == "gasg":
                    with torch.no_grad():
                        right = gasg.inference(
                            sample["left"].unsqueeze(0).to(device),
                            gamma=args.gasg_gamma,
                        )
                else:
                    right = ((sample["right"].unsqueeze(0) + 1.0) * 0.5).to(device)
                gt = sample["depth"].squeeze().numpy().astype(np.float32)

                sync(device)
                t0 = time.perf_counter()
                disp = backend.disparity(left, right)
                sync(device)
                times.append((time.perf_counter() - t0) * 1000.0)
                pred = focal * args.baseline / (
                    disp.squeeze().detach().cpu().numpy().astype(np.float32) + 1e-8)
                pred = np.clip(pred, 0.001, args.max_depth)
                metrics_list.append(compute_depth_errors(
                    gt, pred, max_depth=args.max_depth))
                if idx in visual_indices:
                    save_visual_bundle(out_dir, tag, idx, left, pred, gt,
                                       args.max_depth)

            avg = aggregate_metrics(metrics_list)
            avg.update({
                "backend": backend_name,
                "right_source": source,
                "n_eval": len(metrics_list),
                "height": args.height,
                "width": args.width,
                "focal_length": float(focal),
                "baseline": float(args.baseline),
                "inference_time_ms_mean": float(np.mean(times)),
                "inference_time_ms_median": float(np.median(times)),
            })
            rows.append(avg)
            with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
                json.dump(avg, f, indent=2)
            print(
                f"{BACKEND_NAMES[backend_name]} + {SOURCE_NAMES[source]}: "
                f"AbsRel={avg['abs_rel']:.4f}, RMSE={avg['rmse']:.4f}, "
                f"d1={avg['d1']:.4f}"
            )
        del backend
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summaries(out_dir, rows)
    write_latex_table(root, rows, out_dir)
    if args.save_visuals and visual_indices:
        create_visual_figure(args, sorted(visual_indices)[0])
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_dir", default="results/stereo_backend_generalization")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_depth", type=float, default=25.4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backends", default="defom,raft,sgbm")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--num_visuals", type=int, default=6)
    p.add_argument("--gasg_ckpt", default="checkpoints/gasg_best.pth")
    p.add_argument("--gasg_gamma", type=float, default=1.0)
    p.add_argument("--baseline", type=float, default=1.0)
    p.add_argument("--focal_length", type=float, default=None)
    p.add_argument("--defom_ckpt",
                   default="third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth")
    p.add_argument("--defom_valid_iters", type=int, default=12)
    p.add_argument("--defom_scale_iters", type=int, default=3)
    p.add_argument("--raft_repo", default="third_party/RAFT-Stereo")
    p.add_argument("--raft_ckpt",
                   default="third_party/RAFT-Stereo/models/raftstereo-middlebury.pth")
    p.add_argument("--raft_valid_iters", type=int, default=16)
    p.add_argument("--raft_corr", default="alt", choices=["reg", "alt"])
    p.add_argument("--sgbm_num_disparities", type=int, default=160)
    p.add_argument("--sgbm_block_size", type=int, default=5)
    args = p.parse_args()
    if args.max_samples == 0:
        args.max_samples = None
    return args


if __name__ == "__main__":
    evaluate(parse_args())

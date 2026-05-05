"""Evaluate literature-style left-to-right generators against GASG.

The table produced here is focused on methods that explicitly synthesize a
right view from a left image, or use the common monocular-geometry warp route
used by pseudo-stereo literature.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.deep3d_style import Deep3DStyleGenerator, count_parameters
from scripts.eval_rightview_compare import (
    DAV2WarpGenerator,
    GASGGenerator,
    MoGeWarpGenerator,
    left_tensor_to_01,
    select_indices,
    to_numpy_image,
)
from utils.image_metrics import ImageMetrics
from models.gasg_net import warp_left_to_right


METHOD_LABELS = {
    "identity": "Copy-left",
    "copy_left": "Copy-left",
    "deep3d_svsm": "Deep3D/SVSM-style",
    "dav2_warp": "DA-V2 warp",
    "moge_warp": "MoGe warp",
    "pseudostereo_3dod": "PseudoStereo-3DOD",
    "mono2stereo_dibr": "Mono2Stereo-DIBR",
    "zerostereo_person_tuned": "ZeroStereo FT",
    "gasg_full": "GASG",
}


class DAV2DisparityEstimator:
    def __init__(self, device: torch.device, target_max_disp: float = 96.0):
        self.device = device
        self.target_max_disp = float(target_max_disp)
        sys.path.insert(0, "third_party/Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2

        cfg = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(
            torch.load("checkpoints/depth_anything_v2_large.pth", map_location="cpu", weights_only=True)
        )
        self.model = self.model.to(device).eval()
        self._params = sum(p.numel() for p in self.model.parameters())

    def num_params(self) -> int:
        return self._params

    @torch.no_grad()
    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left01 = left_tensor_to_01(left_tensor).to(self.device)
        _, _, h, w = left01.shape
        img_np = (left01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        disp_raw = self.model.infer_image(img_np)
        disp = torch.from_numpy(disp_raw.astype(np.float32)).view(1, 1, *disp_raw.shape).to(self.device)
        if disp.shape[-2:] != (h, w):
            disp = torch.nn.functional.interpolate(
                disp, size=(h, w), mode="bilinear", align_corners=False
            )
        d_min, d_max = disp.amin(), disp.amax()
        if (d_max - d_min) > 1e-6:
            disp = (disp - d_min) / (d_max - d_min) * self.target_max_disp
        else:
            disp = torch.zeros_like(disp)
        return disp


def forward_splat_right(
    left01: torch.Tensor,
    disp: torch.Tensor,
    fill: str = "left",
) -> torch.Tensor:
    """Left image -> right image by forward splatting x_r = x_l - d."""

    left_np = to_numpy_image(left01)
    disp_np = disp.squeeze().detach().cpu().numpy().astype(np.float32)
    h, w, _ = left_np.shape
    xs = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    ys = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
    xr = np.rint(xs - disp_np).astype(np.int32)
    valid = (xr >= 0) & (xr < w)

    out = np.zeros_like(left_np)
    zbuf = np.full((h, w), -np.inf, dtype=np.float32)
    flat_idx = (ys[valid] * w + xr[valid]).reshape(-1)
    vals = disp_np[valid].reshape(-1)
    np.maximum.at(zbuf.reshape(-1), flat_idx, vals)
    keep = valid & (np.abs(disp_np - zbuf[ys, np.clip(xr, 0, w - 1)]) < 1e-6)
    out[ys[keep], xr[keep]] = left_np[ys[keep], np.arange(w)[None, :].repeat(h, axis=0)[keep]]
    holes = zbuf == -np.inf

    if fill == "left":
        out[holes] = left_np[holes]
    elif fill == "telea":
        seed = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
        mask = holes.astype(np.uint8) * 255
        if mask.any():
            seed = cv2.inpaint(seed, mask, 3, cv2.INPAINT_TELEA)
        out = seed.astype(np.float32) / 255.0
    else:
        raise ValueError(fill)

    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(left01.device, left01.dtype)


class PseudoStereo3DODGenerator:
    name = "pseudostereo_3dod"

    def __init__(self, disparity_estimator: DAV2DisparityEstimator):
        self.disp_est = disparity_estimator

    def num_params(self) -> int:
        return self.disp_est.num_params()

    @torch.no_grad()
    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left01 = left_tensor_to_01(left_tensor).to(self.disp_est.device)
        disp = self.disp_est(left_tensor)
        return forward_splat_right(left01, disp, fill="left").clamp(0.0, 1.0)


class Mono2StereoDIBRGenerator:
    name = "mono2stereo_dibr"

    def __init__(self, disparity_estimator: DAV2DisparityEstimator):
        self.disp_est = disparity_estimator

    def num_params(self) -> int:
        return self.disp_est.num_params()

    @torch.no_grad()
    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left01 = left_tensor_to_01(left_tensor).to(self.disp_est.device)
        disp = self.disp_est(left_tensor)
        return forward_splat_right(left01, disp, fill="telea").clamp(0.0, 1.0)


def load_deep3d_generator(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = Deep3DStyleGenerator(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def tensor01_to_uint8(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x.squeeze(0)
    return (np.clip(x.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0) * 255).astype(np.uint8)


def read_summary_rows(path: Path) -> Dict[str, Dict]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for key in ["psnr", "ssim", "lpips", "inference_time_ms_mean",
                        "inference_time_ms_median", "params_M", "n_eval"]:
                if key in out and out[key] != "":
                    out[key] = float(out[key])
            rows[out["method"]] = out
    return rows


@torch.no_grad()
def evaluate_deep3d(args, dataset, indices, metrics, device) -> Dict:
    model = load_deep3d_generator(args.deep3d_ckpt, device)
    psnr, ssim, lpips_vals, times = [], [], [], []
    warm = dataset[indices[0]]["left"].unsqueeze(0).to(device)
    for _ in range(2):
        _ = model(warm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for idx in tqdm(indices, desc="Deep3D/SVSM-style"):
        sample = dataset[idx]
        left = sample["left"].unsqueeze(0).to(device)
        target = ((sample["right"].unsqueeze(0).to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred = model(left)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        pred_np = to_numpy_image(pred)
        target_np = to_numpy_image(target)
        psnr.append(metrics.compute_psnr(pred_np, target_np))
        ssim.append(metrics.compute_ssim(pred_np, target_np))
        lpips_vals.append(metrics.compute_lpips(pred_np, target_np))

    row = {
        "method": "deep3d_svsm",
        "paper_role": "probabilistic disparity selection",
        "n_eval": len(indices),
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(lpips_vals)),
        "inference_time_ms_mean": float(np.mean(times)),
        "inference_time_ms_median": float(np.median(times)),
        "params_M": count_parameters(model) / 1e6,
    }
    return row


@torch.no_grad()
def evaluate_generator(generator, name: str, role: str, dataset, indices, metrics, device) -> Dict:
    psnr, ssim, lpips_vals, times = [], [], [], []
    warm = dataset[indices[0]]["left"].unsqueeze(0)
    for _ in range(2):
        _ = generator(warm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for idx in tqdm(indices, desc=METHOD_LABELS.get(name, name)):
        sample = dataset[idx]
        left = sample["left"].unsqueeze(0)
        target = ((sample["right"].unsqueeze(0).to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred = generator(left)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        pred_np = to_numpy_image(pred)
        target_np = to_numpy_image(target)
        psnr.append(metrics.compute_psnr(pred_np, target_np))
        ssim.append(metrics.compute_ssim(pred_np, target_np))
        lpips_vals.append(metrics.compute_lpips(pred_np, target_np))

    return {
        "method": name,
        "paper_role": role,
        "n_eval": len(indices),
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(lpips_vals)),
        "inference_time_ms_mean": float(np.mean(times)),
        "inference_time_ms_median": float(np.median(times)),
        "params_M": float(generator.num_params()) / 1e6,
    }


def write_outputs(root: Path, out_dir: Path, rows: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "paper_role", "n_eval", "psnr", "ssim", "lpips",
        "inference_time_ms_mean", "inference_time_ms_median", "params_M",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md = [
        "| Method | N | PSNR | SSIM | LPIPS | Time (ms) | Params (M) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {METHOD_LABELS.get(r['method'], r['method'])} | {int(r['n_eval'])} | "
            f"{r['psnr']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f} | "
            f"{r['inference_time_ms_mean']:.1f} | {r['params_M']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison with representative left-to-right generation methods on PersonDataset. Deep3D/SVSM-style denotes a PersonDataset-trained reproduction of the probabilistic disparity-selection branch used by Deep3D and Single View Stereo Matching. Pseudo-Stereo-3DOD and Mono2Stereo-DIBR denote reproducible image-level virtual-view routes based on published pseudo-stereo/warp-and-inpaint principles. ZeroStereo/StereoGen FT denotes the official StereoGen pipeline fine-tuned on PersonDataset.}",
        "\\label{tab:literature_rightview}",
        "\\small",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\makebox[\\linewidth][c]{%",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Method & N & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & Time (ms)$\\downarrow$ & Params (M)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        label = METHOD_LABELS.get(r["method"], r["method"])
        lines.append(
            f"{label} & {int(r['n_eval'])} & {r['psnr']:.2f} & {r['ssim']:.4f} & "
            f"{r['lpips']:.4f} & {r['inference_time_ms_mean']:.1f} & {r['params_M']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    tex = "\n".join(lines) + "\n"
    for p in [
        out_dir / "Table5_literature_rightview.tex",
        root / "paper_assets" / "final_paper" / "tables" / "Table5_literature_rightview.tex",
        root / "paper_mdpi_applied_sciences" / "tables" / "Table5_literature_rightview.tex",
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex, encoding="utf-8")


def load_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def fit(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (255, 255, 255))
    r = im.convert("RGB")
    r.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(r, ((size[0] - r.width) // 2, (size[1] - r.height) // 2))
    return out


def draw_label(draw, box, text, font):
    x0, y0, x1, y1 = box
    bb = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2,
               y0 + (y1 - y0 - (bb[3] - bb[1])) / 2),
              text, font=font, fill=(15, 23, 42))


@torch.no_grad()
def create_visual(args, root: Path, out_dir: Path, device: torch.device) -> None:
    ds = PersonDataset(args.person_root, split="test", height=args.height, width=args.width)
    sample = ds[0]
    left = sample["left"].unsqueeze(0)
    gt = ((sample["right"].unsqueeze(0) + 1.0) * 0.5).clamp(0.0, 1.0)

    def clear_cuda():
        if device.type == "cuda":
            torch.cuda.empty_cache()

    deep3d = load_deep3d_generator(args.deep3d_ckpt, device)
    deep3d_img = Image.fromarray(tensor01_to_uint8(deep3d(left.to(device))))
    del deep3d
    clear_cuda()

    dav2_disp = DAV2DisparityEstimator(device, target_max_disp=args.dav2_target_disp)
    pseudo = PseudoStereo3DODGenerator(dav2_disp)
    mono_dibr = Mono2StereoDIBRGenerator(dav2_disp)
    left01_dev = left_tensor_to_01(left).to(device)
    disp = dav2_disp(left)
    dav2_img = Image.fromarray(tensor01_to_uint8(warp_left_to_right(left01_dev, disp).clamp(0.0, 1.0)))
    pseudo_img = Image.fromarray(tensor01_to_uint8(pseudo(left)))
    mono_img = Image.fromarray(tensor01_to_uint8(mono_dibr(left)))
    del dav2_disp, pseudo, mono_dibr, disp
    clear_cuda()

    moge = MoGeWarpGenerator(device, target_max_disp=args.moge_target_disp)
    moge_img = Image.fromarray(tensor01_to_uint8(moge(left)))
    del moge
    clear_cuda()

    gasg = GASGGenerator(args.gasg_ckpt, device, variant="full", gamma=1.0)
    gasg_img = Image.fromarray(tensor01_to_uint8(gasg(left)))
    del gasg
    clear_cuda()

    zmeta = json.loads((root / "data" / "zerostereo_person_tuned_test" / "metadata.json").read_text(encoding="utf-8"))
    zpath = None
    for rec in zmeta["records"]:
        if int(rec["person_index"]) == 0:
            zpath = root / "data" / "zerostereo_person_tuned_test" / rec["right"]
            break

    rows = [
        [
            ("Left", Image.fromarray(tensor01_to_uint8(left_tensor_to_01(left)))),
            ("GT right", Image.fromarray(tensor01_to_uint8(gt))),
            ("Deep3D/SVSM", deep3d_img),
            ("PseudoStereo-3DOD", pseudo_img),
            ("Mono2Stereo-DIBR", mono_img),
        ],
        [
            ("DA-V2 warp", dav2_img),
            ("MoGe warp", moge_img),
            ("ZeroStereo FT", Image.open(zpath).convert("RGB").resize((args.width, args.height))),
            ("GASG", gasg_img),
        ],
    ]

    tile = (265, 118)
    gap, title_h, label_h = 14, 70, 38
    n_cols = max(len(r) for r in rows)
    width = n_cols * tile[0] + (n_cols + 1) * gap
    height = title_h + len(rows) * (label_h + tile[1]) + (len(rows) + 1) * gap
    canvas = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, True)
    label_font = load_font(15, True)
    draw_label(draw, (0, 10, width, title_h - 8),
               "Prior Left-to-Right Generation vs. GASG", title_font)
    y = title_h
    for row in rows:
        row_w = len(row) * tile[0] + (len(row) - 1) * gap
        x = (width - row_w) // 2
        for label, im in row:
            draw.rounded_rectangle((x, y, x + tile[0], y + label_h), radius=8,
                                   fill=(241, 245, 249), outline=(203, 213, 225), width=2)
            draw_label(draw, (x, y, x + tile[0], y + label_h), label, label_font)
            canvas.paste(fit(im, tile), (x, y + label_h))
            draw.rectangle((x, y + label_h, x + tile[0], y + label_h + tile[1]),
                           outline=(203, 213, 225), width=2)
            x += tile[0] + gap
        y += label_h + tile[1] + gap

    fig_path = out_dir / "figures" / "Fig10_literature_rightview.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fig_path, dpi=(300, 300))
    for p in [
        root / "paper_assets" / "final_paper" / "figures" / fig_path.name,
        root / "paper_mdpi_applied_sciences" / "figures" / fig_path.name,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_dir", default="results/literature_rightview_generation")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--deep3d_ckpt", default="checkpoints/deep3d_style_person.pth")
    p.add_argument("--gasg_ckpt", default="checkpoints/gasg_best.pth")
    p.add_argument("--dav2_target_disp", type=float, default=96.0)
    p.add_argument("--moge_target_disp", type=float, default=78.0)
    p.add_argument("--save_visual", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.save_dir)
    device = torch.device(args.device)
    ds = PersonDataset(args.person_root, split="test", height=args.height, width=args.width)
    indices = select_indices(len(ds), args.max_samples)
    metrics = ImageMetrics(device=args.device)

    deep3d_row = evaluate_deep3d(args, ds, indices, metrics, device)
    dav2_disp = DAV2DisparityEstimator(device, target_max_disp=args.dav2_target_disp)
    pseudostereo_row = evaluate_generator(
        PseudoStereo3DODGenerator(dav2_disp),
        "pseudostereo_3dod",
        "image-level virtual-view reproduction",
        ds,
        indices,
        metrics,
        device,
    )
    mono2stereo_row = evaluate_generator(
        Mono2StereoDIBRGenerator(dav2_disp),
        "mono2stereo_dibr",
        "warp-and-inpaint stereo conversion",
        ds,
        indices,
        metrics,
        device,
    )
    del dav2_disp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    existing = read_summary_rows(root / "results" / "rightview_generation" / "summary.csv")
    rows = []
    for key, role in [
        ("identity", "identity lower bound"),
        ("deep3d_svsm", "probabilistic disparity selection"),
        ("pseudostereo_3dod", "image-level virtual-view reproduction"),
        ("mono2stereo_dibr", "warp-and-inpaint stereo conversion"),
        ("dav2_warp", "monocular-depth warp"),
        ("moge_warp", "monocular-geometry warp"),
        ("zerostereo_person_tuned", "diffusion warp-and-inpaint"),
        ("gasg_full", "task-specialized generator"),
    ]:
        if key == "deep3d_svsm":
            row = deep3d_row
        elif key == "pseudostereo_3dod":
            row = pseudostereo_row
        elif key == "mono2stereo_dibr":
            row = mono2stereo_row
        else:
            row = dict(existing[key])
            row["paper_role"] = role
        rows.append(row)

    write_outputs(root, out_dir, rows)
    if args.save_visual:
        create_visual(args, root, out_dir, device)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

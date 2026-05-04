"""Calibrate frozen monocular-warp baselines on PersonDataset train split.

DA-V2 and MoGe are not right-view generators. In this project they are used as
published monocular-geometry sources followed by a differentiable horizontal
warp. To make that comparison less arbitrary, this script fits the only
trainable part of the warp pipeline: the mapping from relative monocular
geometry to pixel disparity range.

The calibrated value is selected on PersonDataset/train and then reused for
PersonDataset/test evaluation. The backbone weights remain official pretrained
weights; only the scalar disparity range is tuned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.gasg_net import warp_left_to_right
from scripts.eval_rightview_compare import left_tensor_to_01, select_indices
from utils.image_metrics import ImageMetrics


def normalize_disp(disp: torch.Tensor) -> torch.Tensor:
    d_min = disp.amin(dim=(-2, -1), keepdim=True)
    d_max = disp.amax(dim=(-2, -1), keepdim=True)
    return torch.where(
        (d_max - d_min) > 1e-6,
        (disp - d_min) / (d_max - d_min + 1e-6),
        torch.zeros_like(disp),
    )


def candidate_values(spec: str) -> List[float]:
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, step = [float(x) for x in item.split(":")]
            cur = start
            while cur <= stop + 1e-9:
                values.append(round(cur, 6))
                cur += step
        else:
            values.append(float(item))
    return values


@torch.no_grad()
def cache_dav2(dataset: PersonDataset, indices: Iterable[int],
               device: torch.device, args: argparse.Namespace) -> List[Dict]:
    sys.path.insert(0, args.dav2_third_party_dir)
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(
        args.dav2_ckpt, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    cache = []
    for idx in tqdm(list(indices), desc="Cache DA-V2 geometry"):
        sample = dataset[idx]
        left = left_tensor_to_01(sample["left"].unsqueeze(0)).to(device)
        right = left_tensor_to_01(sample["right"].unsqueeze(0)).to(device)
        img_np = (left.squeeze(0).permute(1, 2, 0).cpu().numpy()
                  * 255.0).astype(np.uint8)
        raw = model.infer_image(img_np).astype(np.float32)
        disp = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)
        if disp.shape[-2:] != left.shape[-2:]:
            disp = F.interpolate(
                disp, size=left.shape[-2:], mode="bilinear",
                align_corners=False)
        cache.append({"left": left, "right": right, "disp": normalize_disp(disp)})
    return cache


@torch.no_grad()
def cache_moge(dataset: PersonDataset, indices: Iterable[int],
               device: torch.device, args: argparse.Namespace) -> List[Dict]:
    import importlib
    import typing

    if "IO" not in typing.__all__:
        typing.__all__.append("IO")
    os.environ.setdefault("HF_HOME", "data/hf_cache")
    sys.path.insert(0, args.moge_third_party_dir)
    v1 = importlib.import_module("moge.model.v1")
    model = v1.MoGeModel.from_pretrained(args.moge_repo).to(device).eval()

    cache = []
    for idx in tqdm(list(indices), desc="Cache MoGe geometry"):
        sample = dataset[idx]
        left = left_tensor_to_01(sample["left"].unsqueeze(0)).to(device)
        right = left_tensor_to_01(sample["right"].unsqueeze(0)).to(device)
        out = model.forward(left, num_tokens=args.moge_tokens)
        depth = torch.clamp(out["points"][..., 2], min=1e-3)
        disp = (1.0 / depth).unsqueeze(1)
        cache.append({"left": left, "right": right, "disp": normalize_disp(disp)})
    return cache


@torch.no_grad()
def score_candidates(cache: List[Dict], values: List[float],
                     metrics: ImageMetrics, device: torch.device) -> Dict:
    rows = []
    for value in tqdm(values, desc="Score disparity ranges"):
        psnr, ssim = [], []
        for item in cache:
            pred = warp_left_to_right(item["left"], item["disp"] * value)
            pred_np = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            gt_np = item["right"].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            psnr.append(metrics.compute_psnr(pred_np, gt_np))
            ssim.append(metrics.compute_ssim(pred_np, gt_np))
        rows.append({
            "target_disp": float(value),
            "psnr": float(np.mean(psnr)),
            "ssim": float(np.mean(ssim)),
        })
    best = max(rows, key=lambda r: (r["psnr"], r["ssim"]))
    return {"best": best, "grid": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_root", default="data/PersonDataset")
    parser.add_argument("--save_dir", default="results/rightview_generation")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", default="6:42:3")
    parser.add_argument("--methods", default="dav2_warp,moge_warp")
    parser.add_argument("--dav2_ckpt",
                        default="checkpoints/depth_anything_v2_large.pth")
    parser.add_argument("--dav2_third_party_dir",
                        default="third_party/Depth-Anything-V2")
    parser.add_argument("--moge_repo", default="Ruicheng/moge-vitl")
    parser.add_argument("--moge_third_party_dir", default="third_party/MoGe")
    parser.add_argument("--moge_tokens", type=int, default=2400)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PersonDataset(args.person_root, split="train",
                            height=args.height, width=args.width)
    indices = select_indices(len(dataset), args.max_samples)
    values = candidate_values(args.candidates)
    metrics = ImageMetrics(device=args.device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "warp_calibration.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = {}
    out.update({
        "protocol": "PersonDataset train split scalar disparity calibration",
        "n_calibration": len(indices),
        "height": args.height,
        "width": args.width,
        "candidates": values,
    })

    if "dav2_warp" in methods:
        cache = cache_dav2(dataset, indices, device, args)
        result = score_candidates(cache, values, metrics, device)
        out["dav2_warp"] = {
            "target_disp": result["best"]["target_disp"],
            "psnr_train": result["best"]["psnr"],
            "ssim_train": result["best"]["ssim"],
            "grid": result["grid"],
        }
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "moge_warp" in methods:
        cache = cache_moge(dataset, indices, device, args)
        result = score_candidates(cache, values, metrics, device)
        out["moge_warp"] = {
            "target_disp": result["best"]["target_disp"],
            "psnr_train": result["best"]["psnr"],
            "ssim_train": result["best"]["ssim"],
            "grid": result["grid"],
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

"""Evaluate official ZeroStereo/StereoGen pretrained right-view generation.

This script intentionally writes into a separate working directory such as
data/zerostereo_person_test, so the true PersonDataset right views
under data/PersonDataset/test/right are never overwritten by the official
ZeroStereo generation scripts.

Pipeline:
  1. Prepare resized PersonDataset left images and a ZeroStereo filelist.
  2. Optionally run official generate_mono.py and generate_stereo.py.
  3. Evaluate generated right images against PersonDataset GT right images.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from datasets.person_dataset import PersonDataset
from scripts.eval_rightview_compare import select_indices
from utils.image_metrics import ImageMetrics


def as_posix(path: str) -> str:
    return path.replace("\\", "/")


def prepare_workspace(args, indices: List[int]) -> Dict:
    src = PersonDataset(args.person_root, split="test",
                        height=args.height, width=args.width)
    root = os.path.abspath(args.work_root)
    left_dir = os.path.join(root, "test", "left")
    os.makedirs(left_dir, exist_ok=True)

    records = []
    for out_i, idx in enumerate(tqdm(indices, desc="Preparing ZeroStereo data")):
        sample = src[idx]
        left_path = sample["left_path"]
        out_name = f"{out_i:05d}.png"
        out_left = os.path.join(left_dir, out_name)
        img = Image.open(left_path).convert("RGB").resize(
            (args.width, args.height), Image.BILINEAR)
        img.save(out_left)

        records.append({
            "out_index": out_i,
            "person_index": idx,
            "left": f"test/left/{out_name}",
            "right": f"test/right/{out_name}",
            "disp": f"test/disparity/{out_i:05d}.npy",
            "confidence": f"test/confidence/{out_i:05d}.npy",
            "mask_nocc": f"test/mask_nocc/{out_i:05d}.png",
            "mask_inpaint": f"test/mask_inpaint/{out_i:05d}.png",
        })

    os.makedirs(args.filelist_dir, exist_ok=True)
    filelist_path = os.path.abspath(os.path.join(
        args.filelist_dir, args.filelist_name))
    with open(filelist_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(" ".join([
                r["left"], r["right"], r["disp"], r["confidence"],
                r["mask_nocc"], r["mask_inpaint"],
            ]) + "\n")

    metadata = {
        "person_root": os.path.abspath(args.person_root),
        "work_root": root,
        "height": args.height,
        "width": args.width,
        "records": records,
    }
    with open(os.path.join(root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Prepared ZeroStereo workspace: {root}")
    print(f"Filelist: {filelist_path}")
    return metadata


def run_official_generation(args) -> float:
    repo = os.path.abspath(args.zerostereo_repo)
    root_rel = as_posix(os.path.relpath(os.path.abspath(args.work_root), repo))
    filelist_rel = as_posix(os.path.relpath(
        os.path.abspath(os.path.join(args.filelist_dir, args.filelist_name)),
        repo))
    depth_ckpt = as_posix(os.path.abspath(args.depth_anything_ckpt))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    py = sys.executable
    mono_cmd = [
        py, "generate_mono.py",
        f"dataset.MfS35K.instance.root={root_rel}",
        f"dataset.MfS35K.instance.filelist={filelist_rel}",
        f"checkpoint={depth_ckpt}",
        "dataloader.param.num_workers=0",
        "dataloader.param.pin_memory=False",
    ]
    stereo_cmd = [
        py, "generate_stereo.py",
        f"dataset.MfS35K.instance.root={root_rel}",
        f"dataset.MfS35K.instance.filelist={filelist_rel}",
        f"num_inference_step={args.num_inference_step}",
        "dataloader.param.num_workers=0",
        "dataloader.param.pin_memory=False",
    ]

    t0 = time.time()
    print("\n$ " + " ".join(mono_cmd), flush=True)
    subprocess.run(mono_cmd, cwd=repo, env=env, check=True)
    print("\n$ " + " ".join(stereo_cmd), flush=True)
    subprocess.run(stereo_cmd, cwd=repo, env=env, check=True)
    elapsed = time.time() - t0
    print(f"Official ZeroStereo generation wall time: {elapsed:.1f}s")
    return elapsed


def evaluate_generated(args, metadata: Dict,
                       generation_time_s: Optional[float] = None) -> Dict:
    ds = PersonDataset(args.person_root, split="test",
                       height=args.height, width=args.width)
    metrics = ImageMetrics(device=args.metric_device)

    psnr, ssim, lpips_scores = [], [], []
    missing = 0
    for r in tqdm(metadata["records"], desc="Evaluating ZeroStereo official"):
        gt_sample = ds[int(r["person_index"])]
        gt = ((gt_sample["right"] + 1.0) * 0.5).permute(1, 2, 0).numpy()
        gt = np.clip(gt, 0.0, 1.0)

        gen_path = os.path.join(args.work_root, r["right"])
        if not os.path.exists(gen_path):
            missing += 1
            continue
        pred = np.array(Image.open(gen_path).convert("RGB").resize(
            (args.width, args.height), Image.BILINEAR), dtype=np.float32) / 255.0

        psnr.append(metrics.compute_psnr(pred, gt))
        ssim.append(metrics.compute_ssim(pred, gt))
        lpips_scores.append(metrics.compute_lpips(pred, gt))

    if not psnr:
        raise FileNotFoundError(
            f"No generated right views found in {args.work_root}/test/right")

    n = len(psnr)
    row = {
        "method": "zerostereo_official",
        "n_eval": n,
        "missing": missing,
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(lpips_scores)),
        "psnr_std": float(np.std(psnr)),
        "ssim_std": float(np.std(ssim)),
        "lpips_std": float(np.std(lpips_scores)),
        "height": int(args.height),
        "width": int(args.width),
        "device": "cuda",
        "params_M": count_stereogen_params(args.zerostereo_repo),
        "num_inference_step": int(args.num_inference_step),
    }
    if generation_time_s is not None:
        row["inference_time_ms_mean"] = generation_time_s * 1000.0 / max(n, 1)
        row["inference_time_ms_median"] = row["inference_time_ms_mean"]

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "zerostereo_official.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(
        f"ZeroStereo official: PSNR={row['psnr']:.2f} | "
        f"SSIM={row['ssim']:.4f} | LPIPS={row['lpips']:.4f}"
    )
    print(f"Wrote: {out_path}")
    return row


def count_stereogen_params(zerostereo_repo: str) -> float:
    root = os.path.join(
        zerostereo_repo, "checkpoint", "hf_zerostereo", "StereoGen")
    files = [
        os.path.join(root, "unet", "diffusion_pytorch_model.safetensors"),
        os.path.join(root, "vae", "diffusion_pytorch_model.safetensors"),
        os.path.join(root, "text_encoder", "model.safetensors"),
    ]
    total = 0
    try:
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    shape = f.get_slice(key).get_shape()
                    total += int(np.prod(shape))
    except Exception:
        return 0.0
    return float(total) / 1e6


def load_or_prepare_metadata(args) -> Dict:
    metadata_path = os.path.join(args.work_root, "metadata.json")
    if os.path.exists(metadata_path) and not args.force_prepare:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ds = PersonDataset(args.person_root, split="test",
                       height=args.height, width=args.width)
    n_eval = None if args.max_samples is None or args.max_samples <= 0 \
        else args.max_samples
    indices = select_indices(len(ds), n_eval)
    if args.force_prepare and os.path.isdir(args.work_root):
        shutil.rmtree(args.work_root)
    return prepare_workspace(args, indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_root", default="data/PersonDataset")
    parser.add_argument("--zerostereo_repo", default="third_party/ZeroStereo")
    parser.add_argument("--work_root",
                        default="data/zerostereo_person_test")
    parser.add_argument("--filelist_dir",
                        default="third_party/ZeroStereo/filelist")
    parser.add_argument("--filelist_name", default="person_eval.txt")
    parser.add_argument("--save_dir", default="results/rightview_generation")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--num_inference_step", type=int, default=20)
    parser.add_argument("--depth_anything_ckpt",
                        default="checkpoints/depth_anything_v2_large.pth")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--metric_device", default="cuda")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--force_prepare", action="store_true")
    args = parser.parse_args()

    metadata = load_or_prepare_metadata(args)
    if args.prepare_only:
        return

    generation_time_s = None
    if args.run_generation:
        generation_time_s = run_official_generation(args)

    if args.evaluate_only or args.run_generation:
        evaluate_generated(args, metadata, generation_time_s)
    else:
        print("Prepared metadata only. Use --run_generation or --evaluate_only.")


if __name__ == "__main__":
    main()

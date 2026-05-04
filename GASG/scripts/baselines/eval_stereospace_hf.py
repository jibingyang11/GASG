"""Evaluate official StereoSpace HF Space on PersonDataset.

StereoSpace is a recent monocular-to-stereo diffusion method.  The official
local model is large, so this script uses the public Hugging Face Space API as
an official hosted reproduction path.  The generated right views are saved with
metadata compatible with scripts/eval_person_depth_model.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.person_dataset import PersonDataset
from scripts.eval_rightview_compare import select_indices
from utils.image_metrics import ImageMetrics


def sample_to_image(tensor, size_hw: tuple[int, int]) -> Image.Image:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = ((np.clip(arr, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return Image.fromarray(arr).resize((size_hw[1], size_hw[0]), Image.BILINEAR)


def image_to_np(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def pick_generated_path(result: Any) -> str:
    """Return the generated-view file path from a gradio_client result."""
    gallery = result[0] if isinstance(result, (list, tuple)) and result else result
    if isinstance(gallery, list):
        for item in gallery:
            caption = (item.get("caption") or "").lower() if isinstance(item, dict) else ""
            image = item.get("image") if isinstance(item, dict) else None
            if "generated" in caption:
                if isinstance(image, dict) and image.get("path"):
                    return image["path"]
                if isinstance(image, str):
                    return image
        for item in reversed(gallery):
            image = item.get("image") if isinstance(item, dict) else item
            if isinstance(image, dict) and image.get("path"):
                return image["path"]
            if isinstance(image, str):
                return image
    if isinstance(result, (list, tuple)) and len(result) > 1:
        image = result[1]
        if isinstance(image, dict) and image.get("path"):
            return image["path"]
        if isinstance(image, str):
            return image
    raise RuntimeError(f"Could not locate generated-view path in StereoSpace result: {type(result)}")


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    from gradio_client import Client, handle_file

    height, width = args.height, args.width
    work_root = Path(args.work_root)
    input_dir = work_root / "input"
    right_dir = work_root / "right"
    input_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    ds = PersonDataset(args.person_root, split="test", height=height, width=width)
    indices = select_indices(len(ds), args.max_samples)
    metrics = ImageMetrics(device=args.metric_device)
    client = Client(args.space)

    records: List[Dict[str, Any]] = []
    psnr, ssim, lpips_scores, runtimes = [], [], [], []

    for out_i, idx in enumerate(tqdm(indices, desc="StereoSpace HF")):
        sample = ds[idx]
        input_path = input_dir / f"{idx:05d}_left.png"
        right_path = right_dir / f"{idx:05d}_right.png"
        sample_to_image(sample["left"], (height, width)).save(input_path)

        if args.skip_existing and right_path.exists():
            elapsed_ms = None
        else:
            t0 = time.perf_counter()
            try:
                result = client.predict(
                    img=handle_file(str(input_path)),
                    current_mode="Generated view",
                    api_name="/process_upload_wrapper",
                )
            except Exception as exc:
                if args.continue_on_error and records:
                    print(f"WARNING: StereoSpace request failed at index {idx}: {exc}")
                    print("Writing partial results from completed samples.")
                    break
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            generated_path = pick_generated_path(result)
            shutil.copyfile(generated_path, right_path)

        pred = image_to_np(right_path, (height, width))
        gt = ((sample["right"].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5)
        gt = np.clip(gt, 0.0, 1.0)
        psnr.append(metrics.compute_psnr(pred, gt))
        ssim.append(metrics.compute_ssim(pred, gt))
        lpips_scores.append(metrics.compute_lpips(pred, gt))
        if elapsed_ms is not None:
            runtimes.append(elapsed_ms)

        records.append({
            "person_index": int(idx),
            "input": str(input_path.relative_to(work_root)).replace("\\", "/"),
            "right": str(right_path.relative_to(work_root)).replace("\\", "/"),
            "runtime_ms": elapsed_ms,
        })

    meta = {
        "method": "stereospace_hf",
        "space": args.space,
        "height": height,
        "width": width,
        "n_eval": len(records),
        "records": records,
    }
    (work_root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    avg = {
        "method": "stereospace_hf",
        "pretty": "StereoSpace official HF Space",
        "type": "published monocular-to-stereo generator",
        "n_eval": len(records),
        "height": height,
        "width": width,
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(lpips_scores)),
        "psnr_std": float(np.std(psnr)),
        "ssim_std": float(np.std(ssim)),
        "lpips_std": float(np.std(lpips_scores)),
        "inference_time_ms_mean": float(np.mean(runtimes)) if runtimes else 0.0,
        "inference_time_ms_median": float(np.median(runtimes)) if runtimes else 0.0,
        "params_M": 0.0,
        "runtime_note": "HF Space wall time; includes queue/network overhead.",
        "work_root": str(work_root),
    }
    out_json = Path(args.save_dir) / "stereospace_hf.json"
    out_json.write_text(json.dumps(avg, indent=2), encoding="utf-8")
    print(json.dumps(avg, indent=2))
    print(f"Wrote: {out_json}")
    print(f"Saved right views: {work_root}")
    return avg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--work_root", default="data/stereospace_person_test")
    p.add_argument("--save_dir", default="results/rightview_generation")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--space", default="toshas/stereospace")
    p.add_argument("--metric_device", default="cpu")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

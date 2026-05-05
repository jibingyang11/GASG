"""Train the Deep3D/SVSM-style right-view synthesis baseline on PersonDataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.deep3d_style import Deep3DStyleGenerator, count_parameters
from models.losses import LowResSSIMLoss, gradient_loss
from utils.image_metrics import ImageMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t.squeeze(0)
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def cosine_lr(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    if total_steps <= 1:
        return lr_end
    p = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def validate(model, dataset, indices, metrics, device) -> dict:
    model.eval()
    vals = {"psnr": [], "ssim": [], "lpips": []}
    for idx in indices:
        sample = dataset[idx]
        left = sample["left"].unsqueeze(0).to(device)
        target = ((sample["right"].unsqueeze(0).to(device) + 1.0) * 0.5).clamp(0.0, 1.0)
        pred = model(left)
        pred_np = to_numpy_image(pred)
        target_np = to_numpy_image(target)
        vals["psnr"].append(metrics.compute_psnr(pred_np, target_np))
        vals["ssim"].append(metrics.compute_ssim(pred_np, target_np))
        vals["lpips"].append(metrics.compute_lpips(pred_np, target_np))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_path", default="checkpoints/deep3d_style_person.pth")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr_start", type=float, default=2e-4)
    p.add_argument("--lr_end", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_disp", type=float, default=96.0)
    p.add_argument("--num_bins", type=int, default=33)
    p.add_argument("--base_ch", type=int, default=24)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--val_samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds = PersonDataset(
        args.person_root, split="train",
        height=args.height, width=args.width, augment=True,
    )
    test_ds = PersonDataset(
        args.person_root, split="test",
        height=args.height, width=args.width, augment=False,
    )
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.max_train_samples)))

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_indices = [int(i * len(test_ds) / args.val_samples) for i in range(args.val_samples)]

    model = Deep3DStyleGenerator(
        max_disp=args.max_disp,
        num_bins=args.num_bins,
        base_ch=args.base_ch,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr_start,
        weight_decay=args.weight_decay,
    )
    ssim_loss = LowResSSIMLoss(window_size=7, downsample=2).to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    metrics = ImageMetrics(device=args.device)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = save_path.with_suffix(".log.json")

    best_psnr = -1.0
    history = []
    total_steps = max(1, args.epochs * len(loader))
    step = 0
    print(f"Deep3D-style params: {count_parameters(model) / 1e6:.2f} M")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = []
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            left = batch["left"].to(device, non_blocking=True)
            target = ((batch["right"].to(device, non_blocking=True) + 1.0) * 0.5).clamp(0.0, 1.0)
            lr = cosine_lr(step, total_steps, args.lr_start, args.lr_end)
            for group in opt.param_groups:
                group["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                pred = model(left)
                loss = (
                    F.l1_loss(pred, target)
                    + 0.12 * ssim_loss(pred, target)
                    + 0.04 * gradient_loss(pred, target)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            loss_meter.append(float(loss.detach().cpu()))
            step += 1
            pbar.set_postfix(loss=f"{np.mean(loss_meter[-20:]):.4f}", lr=f"{lr:.2e}")

        val = validate(model, test_ds, val_indices, metrics, device)
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(loss_meter)),
            "seconds": float(time.time() - t0),
            **val,
        }
        history.append(rec)
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch}: loss={rec['loss']:.4f}, "
            f"PSNR={val['psnr']:.2f}, SSIM={val['ssim']:.4f}, LPIPS={val['lpips']:.4f}"
        )
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "max_disp": args.max_disp,
                        "num_bins": args.num_bins,
                        "base_ch": args.base_ch,
                    },
                    "params": count_parameters(model),
                    "best_psnr": best_psnr,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()

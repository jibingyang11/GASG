"""
Train GASG without discriminator / adversarial training.

Features:
  - generator-only training
  - warmup-cosine learning rate
  - optional two-stage training:
      stage 1: main structure / geometry training
      stage 2: PSNR-oriented fine-tuning
  - real-time visual monitoring:
      left / real right / generated right / absolute error
  - PSNR / SSIM / MAE monitoring
  - Windows-safe DataLoader defaults

Usage:
  python scripts/train_gasg.py --config configs/gasg_config.yaml

With visualization:
  python scripts/train_gasg.py --config configs/gasg_config.yaml --vis_every 500 --vis_num 4 --num_workers 0

Smoke test:
  python scripts/train_gasg.py --config configs/gasg_config.yaml --max_epochs 1 --max_batches 5 --num_workers 0
"""

import os

# Safer for Windows conda environments
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import csv
import math
import time
import yaml
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gasg_net import GASG
from models.losses import GASGLoss
from datasets.person_dataset import PersonDataset


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generator(model_config):
    """
    Build GASG from config.

    Only known keys are passed.
    Extra yaml keys are ignored safely.
    """
    kwargs = {}

    for key in (
        "base_ch",
        "width",
        "depths",
        "patch_size",
        "residual_scale",
        "gamma_correction",
        "num_bottleneck_blocks",
        "fusion_bias",
        "max_disp",
        "max_vshift",
        "use_vertical_flow",
        "positive_disp",
        "num_lowres_blocks",
        "flow_downsample",
        "refine_ch",
        "coarse_residual_scale",
        "detail_residual_scale",
    ):
        if key in model_config:
            kwargs[key] = model_config[key]

    return GASG(**kwargs)


def compute_lr(step, total_steps, warmup_steps, lr_start, lr_end):
    if warmup_steps > 0 and step < warmup_steps:
        return lr_start * float(step + 1) / float(warmup_steps)

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_end + (lr_start - lr_end) * cosine


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def make_grad_scaler(device, amp_enabled):
    try:
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


# ============================================================
# Config defaults
# ============================================================

def apply_config_defaults(config, args):
    if config is None:
        config = {}

    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("logging", {})
    config.setdefault("data", {})
    config.setdefault("loss_weights", {})

    # Dataset
    if args.person_root is not None:
        config["data"]["person_root"] = args.person_root
    else:
        config["data"].setdefault("person_root", "data/PersonDataset")

    # Model defaults
    config["model"].setdefault("base_ch", 24)
    config["model"].setdefault("max_disp", 24.0)
    config["model"].setdefault("positive_disp", True)
    config["model"].setdefault("residual_scale", 0.35)
    config["model"].setdefault("gamma_correction", 1.0)
    config["model"].setdefault("num_bottleneck_blocks", 3)

    # Training defaults
    config["training"].setdefault("seed", 42)
    config["training"].setdefault("total_epochs", 80)
    config["training"].setdefault("batch_size", 2)
    config["training"].setdefault("num_workers", 0)
    config["training"].setdefault("input_height", 256)
    config["training"].setdefault("input_width", 512)
    config["training"].setdefault("lr_start", 1.5e-4)
    config["training"].setdefault("lr_end", 1.0e-5)
    config["training"].setdefault("warmup_epochs", 3)
    config["training"].setdefault("optimizer", "adamw")
    config["training"].setdefault("betas", [0.9, 0.999])
    config["training"].setdefault("weight_decay", 1.0e-5)
    config["training"].setdefault("grad_clip_norm", 0.5)
    config["training"].setdefault("ema_decay", 0.995)
    config["training"].setdefault("amp", True)

    # Main-stage loss defaults
    main_default = {
        "photo": 1.0,
        "warp": 0.25,
        "ms": 0.20,
        "ssim": 0.04,
        "census": 0.08,
        "edge": 0.015,
        "lowfreq": 0.08,
        "smooth": 0.006,
        "flow": 0.00005,
        "mask": 0.005,
        "mask_tv": 0.005,
        "residual": 0.03,
    }

    for k, v in main_default.items():
        config["loss_weights"].setdefault(k, v)

    # PSNR fine-tuning defaults
    config.setdefault("psnr_finetune", {})
    config["psnr_finetune"].setdefault("enabled", False)
    config["psnr_finetune"].setdefault("start_epoch", 20)
    config["psnr_finetune"].setdefault("lr_start", 8.0e-5)
    config["psnr_finetune"].setdefault("lr_end", 1.0e-6)
    config["psnr_finetune"].setdefault("warmup_epochs", 0)
    config["psnr_finetune"].setdefault("loss_weights", {
        "photo": 1.0,
        "mse": 1.0,
        "warp": 0.15,
        "ms": 0.10,
        "ssim": 0.01,
        "census": 0.02,
        "edge": 0.005,
        "lowfreq": 0.03,
        "smooth": 0.003,
        "flow": 0.00002,
        "mask": 0.002,
        "mask_tv": 0.002,
        "residual": 0.01,
    })

    # Logging defaults
    config["logging"].setdefault("save_dir", "experiments")
    config["logging"].setdefault("exp_name", "gasg_train")
    config["logging"].setdefault("save_every", 5)
    config["logging"].setdefault("log_every", 50)

    return config


# ============================================================
# Loss stage control
# ============================================================

def build_loss_weights(config, ablation=None):
    weights = dict(config.get("loss_weights", {}))

    if ablation == "no_ssim" and "ssim" in weights:
        weights["ssim"] = 0.0
    elif ablation in ("no_edge", "no_grad") and "edge" in weights:
        weights["edge"] = 0.0
    elif ablation == "no_smooth" and "smooth" in weights:
        weights["smooth"] = 0.0
    elif ablation == "no_flow" and "flow" in weights:
        weights["flow"] = 0.0

    return weights


def build_psnr_finetune_weights(config):
    default = {
        "photo": 1.0,
        "mse": 1.0,
        "warp": 0.15,
        "ms": 0.10,
        "ssim": 0.01,
        "census": 0.02,
        "edge": 0.005,
        "lowfreq": 0.03,
        "smooth": 0.003,
        "flow": 0.00002,
        "mask": 0.002,
        "mask_tv": 0.002,
        "residual": 0.01,
    }

    psnr_cfg = config.get("psnr_finetune", {})
    user_weights = psnr_cfg.get("loss_weights", {})

    out = default.copy()
    for k, v in user_weights.items():
        out[str(k)] = float(v)

    return out


def get_stage_name(config, epoch_idx):
    """
    epoch_idx is zero-based.
    """
    psnr_cfg = config.get("psnr_finetune", {})
    enabled = bool(psnr_cfg.get("enabled", False))
    start_epoch = int(psnr_cfg.get("start_epoch", 20))

    if enabled and (epoch_idx + 1) >= start_epoch:
        return "psnr_finetune"

    return "main"


def get_stage_loss_weights(config, epoch_idx, ablation=None):
    stage = get_stage_name(config, epoch_idx)

    if stage == "psnr_finetune":
        weights = build_psnr_finetune_weights(config)
    else:
        weights = build_loss_weights(config, ablation)

    if ablation == "no_ssim" and "ssim" in weights:
        weights["ssim"] = 0.0
    elif ablation in ("no_edge", "no_grad") and "edge" in weights:
        weights["edge"] = 0.0
    elif ablation == "no_smooth" and "smooth" in weights:
        weights["smooth"] = 0.0
    elif ablation == "no_flow" and "flow" in weights:
        weights["flow"] = 0.0

    return weights


def compute_stage_lr(
    config,
    epoch_idx,
    batch_idx,
    steps_per_epoch,
    total_epochs,
):
    stage = get_stage_name(config, epoch_idx)

    if stage == "psnr_finetune":
        psnr_cfg = config.get("psnr_finetune", {})
        start_epoch = int(psnr_cfg.get("start_epoch", 20))

        lr_start = float(psnr_cfg.get("lr_start", 8.0e-5))
        lr_end = float(psnr_cfg.get("lr_end", 1.0e-6))
        warmup_epochs = int(psnr_cfg.get("warmup_epochs", 0))

        local_epoch = max((epoch_idx + 1) - start_epoch, 0)
        local_step = local_epoch * steps_per_epoch + batch_idx
        local_total_epochs = max(total_epochs - start_epoch + 1, 1)
        local_total_steps = max(local_total_epochs * steps_per_epoch, 1)
        local_warmup_steps = warmup_epochs * steps_per_epoch

        lr = compute_lr(
            local_step,
            local_total_steps,
            local_warmup_steps,
            lr_start,
            lr_end,
        )
        return lr, stage

    lr_start = float(config["training"]["lr_start"])
    lr_end = float(config["training"]["lr_end"])
    warmup_epochs = int(config["training"].get("warmup_epochs", 2))

    global_step = epoch_idx * steps_per_epoch + batch_idx
    total_steps = max(total_epochs * steps_per_epoch, 1)
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps // 2)

    lr = compute_lr(
        global_step,
        total_steps,
        warmup_steps,
        lr_start,
        lr_end,
    )
    return lr, stage


# ============================================================
# Monitor metrics and visualization
# ============================================================

def tensor_to_01(x):
    if x.min() < -0.05:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def compute_psnr(pred, target):
    pred = tensor_to_01(pred)
    target = tensor_to_01(target)

    mse = F.mse_loss(pred, target).detach().item()
    if mse <= 1e-10:
        return 99.0

    return -10.0 * math.log10(mse)


def compute_mae(pred, target):
    pred = tensor_to_01(pred)
    target = tensor_to_01(target)
    return torch.mean(torch.abs(pred - target)).detach().item()


def compute_simple_ssim(pred, target, window_size=7):
    pred = tensor_to_01(pred)
    target = tensor_to_01(target)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(
        pred * pred,
        window_size,
        stride=1,
        padding=pad,
    ) - mu_x ** 2

    sigma_y = F.avg_pool2d(
        target * target,
        window_size,
        stride=1,
        padding=pad,
    ) - mu_y ** 2

    sigma_xy = F.avg_pool2d(
        pred * target,
        window_size,
        stride=1,
        padding=pad,
    ) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1)
        * (sigma_x + sigma_y + c2)
        + 1e-8
    )

    return ssim.mean().detach().item()


def save_visual_comparison(left, real_right, gen_right, save_path, max_images=4):
    left = tensor_to_01(left).detach().cpu()
    real_right = tensor_to_01(real_right).detach().cpu()
    gen_right = tensor_to_01(gen_right).detach().cpu()

    n = min(left.shape[0], max_images)

    error = torch.abs(gen_right - real_right)
    error = error / (error.max() + 1e-8)

    rows = [
        ("Left", left),
        ("Right GT", real_right),
        ("Right GASG", gen_right),
        ("Abs Error", error),
    ]

    fig, axes = plt.subplots(len(rows), n, figsize=(4 * n, 10))

    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, (title, imgs) in enumerate(rows):
        for c in range(n):
            img = imgs[c].permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(img, 0.0, 1.0))
            axes[r, c].axis("off")

            if c == 0:
                axes[r, c].set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def append_monitor_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "batch",
                "global_step",
                "stage",
                "loss",
                "ema_loss",
                "psnr",
                "ssim",
                "mae",
                "lr",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


@torch.no_grad()
def run_visual_monitor(
    generator,
    fixed_batch,
    device,
    save_path,
    max_images=4,
):
    generator.eval()

    left = fixed_batch["left"].to(device, non_blocking=True)
    real_right = fixed_batch["right"].to(device, non_blocking=True)

    model_out = generator(left, return_dict=True)

    if isinstance(model_out, dict):
        gen_right = model_out["right"]
    else:
        gen_right = model_out

    psnr = compute_psnr(gen_right, real_right)
    ssim = compute_simple_ssim(gen_right, real_right)
    mae = compute_mae(gen_right, real_right)

    save_visual_comparison(
        left=left,
        real_right=real_right,
        gen_right=gen_right,
        save_path=save_path,
        max_images=max_images,
    )

    generator.train()

    return {
        "psnr": psnr,
        "ssim": ssim,
        "mae": mae,
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    epoch,
    generator,
    g_optimizer,
    config,
    loss_weights,
    ema_loss=None,
):
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "model_config": config["model"],
            "training_config": config["training"],
            "loss_weights": loss_weights,
            "ema_loss": ema_loss,
        },
        path,
    )


# ============================================================
# Training
# ============================================================

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/gasg_config.yaml")

    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=[None, "no_ssim", "no_edge", "no_grad", "no_smooth", "no_flow"],
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--exp_suffix", type=str, default=None)

    parser.add_argument(
        "--vis_every",
        type=int,
        default=500,
        help="Save visual comparison every N batches.",
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=4,
        help="Number of samples shown in visualization.",
    )
    parser.add_argument(
        "--disable_vis",
        action="store_true",
        help="Disable visual monitoring.",
    )
    parser.add_argument(
        "--person_root",
        type=str,
        default=None,
        help="Override PersonDataset root path.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = apply_config_defaults(config, args)

    set_seed(config["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Device: {device}", flush=True)

    # ----------------------------
    # Dataset
    # ----------------------------
    train_dataset = PersonDataset(
        root=config["data"]["person_root"],
        split="train",
        height=config["training"]["input_height"],
        width=config["training"]["input_width"],
        augment=True,
    )

    batch_size = args.batch_size or int(config["training"]["batch_size"])

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["training"].get("num_workers", 0))
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }

    # Windows 下 persistent_workers 容易卡住，所以默认不打开
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # ----------------------------
    # Model / Loss / Optimizer
    # ----------------------------
    generator = build_generator(config["model"]).to(device)
    criterion = GASGLoss().to(device)

    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    optimizer_cls = optim.AdamW if optimizer_name == "adamw" else optim.Adam

    g_optimizer = optimizer_cls(
        generator.parameters(),
        lr=float(config["training"]["lr_start"]),
        betas=tuple(config["training"].get("betas", [0.9, 0.999])),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)

    # ----------------------------
    # Resume
    # ----------------------------
    global_epoch = 0
    ema_loss = None

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)

        if "generator" in ckpt:
            generator.load_state_dict(ckpt["generator"])
        else:
            generator.load_state_dict(ckpt)

        if "g_optimizer" in ckpt:
            g_optimizer.load_state_dict(ckpt["g_optimizer"])

        global_epoch = int(ckpt.get("epoch", 0))
        ema_loss = ckpt.get("ema_loss", None)

        print(f"Resumed from {args.resume} at epoch {global_epoch}", flush=True)

    # ----------------------------
    # Experiment directories
    # ----------------------------
    exp_name = config["logging"]["exp_name"]

    if args.ablation:
        exp_name += f"_ablation_{args.ablation}"

    if args.exp_suffix:
        exp_name += f"_{args.exp_suffix}"
    elif args.max_epochs is not None or args.max_batches is not None:
        exp_name += "_smoke"

    save_dir = os.path.join(config["logging"]["save_dir"], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    vis_dir = os.path.join(save_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    monitor_csv = os.path.join(save_dir, "monitor_metrics.csv")

    # ----------------------------
    # Fixed visual batch
    # ----------------------------
    fixed_vis_batch = None

    if not args.disable_vis:
        vis_dataset = PersonDataset(
            root=config["data"]["person_root"],
            split="test",
            height=config["training"]["input_height"],
            width=config["training"]["input_width"],
            augment=False,
        )

        vis_loader = DataLoader(
            vis_dataset,
            batch_size=max(args.vis_num, 1),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        fixed_vis_batch = next(iter(vis_loader))

        print(
            f"Visual monitor enabled. Images will be saved to: {vis_dir}",
            flush=True,
        )

    # ----------------------------
    # Schedule
    # ----------------------------
    total_epochs = args.max_epochs or int(config["training"]["total_epochs"])

    steps_per_epoch = len(train_loader)
    if args.max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_batches)

    grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
    log_every = int(config["logging"].get("log_every", 50))
    save_every = int(config["logging"].get("save_every", 5))
    ema_decay = float(config["training"].get("ema_decay", 0.995))

    main_weights = build_loss_weights(config, args.ablation)
    psnr_weights = build_psnr_finetune_weights(config)

    print("\n" + "=" * 60, flush=True)
    print("Training schedule: generator-only stage-aware warmup-cosine", flush=True)
    print("Adversarial training: disabled", flush=True)
    print(f"Main loss weights: {main_weights}", flush=True)
    print(
        f"PSNR finetune: {config['psnr_finetune'].get('enabled', False)} "
        f"| start_epoch: {config['psnr_finetune'].get('start_epoch', 20)}",
        flush=True,
    )
    print(f"PSNR loss weights: {psnr_weights}", flush=True)
    print(f"AMP: {amp_enabled} | Optimizer: {optimizer_name}", flush=True)
    print(f"Epochs: {total_epochs}", flush=True)
    print(f"Batch size: {batch_size} | Num workers: {num_workers}", flush=True)
    print(f"Save dir: {save_dir}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()

    # ========================================================
    # Main training loop
    # ========================================================
    while global_epoch < total_epochs:
        generator.train()

        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            global_step = global_epoch * steps_per_epoch + batch_idx

            lr, stage_name = compute_stage_lr(
                config=config,
                epoch_idx=global_epoch,
                batch_idx=batch_idx,
                steps_per_epoch=steps_per_epoch,
                total_epochs=total_epochs,
            )

            set_optimizer_lr(g_optimizer, lr)

            current_loss_weights = get_stage_loss_weights(
                config=config,
                epoch_idx=global_epoch,
                ablation=args.ablation,
            )

            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)

            g_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                model_out = generator(left, return_dict=True)

                if isinstance(model_out, torch.Tensor):
                    model_out = {"right": model_out}

                total_loss, loss_dict = criterion(
                    model_out,
                    right,
                    left,
                    weights=current_loss_weights,
                )

            if not torch.isfinite(total_loss):
                print(
                    f"Warning: non-finite loss at epoch {global_epoch + 1}, "
                    f"batch {batch_idx}. Skip this batch.",
                    flush=True,
                )
                continue

            scaler.scale(total_loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(),
                    grad_clip_norm,
                )

            scaler.step(g_optimizer)
            scaler.update()

            loss_value = float(total_loss.detach().item())

            epoch_loss += loss_value
            n_batches += 1

            if ema_loss is None:
                ema_loss = loss_value
            else:
                ema_loss = ema_decay * ema_loss + (1.0 - ema_decay) * loss_value

            # ----------------------------
            # Text log
            # ----------------------------
            if batch_idx % log_every == 0:
                loss_msg = " ".join(
                    f"{k}:{v.detach().item():.4f}"
                    for k, v in loss_dict.items()
                    if k != "total"
                )

                print(
                    f"  Epoch {global_epoch + 1} [{batch_idx}/{len(train_loader)}] "
                    f"stage:{stage_name} "
                    f"lr:{lr:.2e} "
                    f"Loss:{loss_value:.4f} "
                    f"EMA:{ema_loss:.4f} "
                    f"{loss_msg}",
                    flush=True,
                )

            # ----------------------------
            # Visual monitor
            # ----------------------------
            if (
                not args.disable_vis
                and fixed_vis_batch is not None
                and args.vis_every > 0
                and batch_idx % args.vis_every == 0
            ):
                vis_path = os.path.join(
                    vis_dir,
                    f"epoch_{global_epoch + 1:03d}_batch_{batch_idx:05d}.png",
                )

                metrics = run_visual_monitor(
                    generator=generator,
                    fixed_batch=fixed_vis_batch,
                    device=device,
                    save_path=vis_path,
                    max_images=args.vis_num,
                )

                append_monitor_csv(
                    monitor_csv,
                    {
                        "epoch": global_epoch + 1,
                        "batch": batch_idx,
                        "global_step": global_step,
                        "stage": stage_name,
                        "loss": loss_value,
                        "ema_loss": ema_loss,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "mae": metrics["mae"],
                        "lr": lr,
                    },
                )

                print(
                    f"    Monitor | "
                    f"stage:{stage_name} "
                    f"PSNR:{metrics['psnr']:.2f} "
                    f"SSIM:{metrics['ssim']:.4f} "
                    f"MAE:{metrics['mae']:.4f} "
                    f"| Saved: {vis_path}",
                    flush=True,
                )

        # ====================================================
        # End of epoch
        # ====================================================
        if n_batches == 0:
            print("No valid batches in this epoch. Stop training.", flush=True)
            break

        epoch_loss /= n_batches
        sec_per_batch = (time.time() - epoch_start) / max(n_batches, 1)
        elapsed = (time.time() - start_time) / 3600.0
        epoch_stage = get_stage_name(config, global_epoch)

        print(
            f"Epoch {global_epoch + 1}/{total_epochs} | "
            f"Stage: {epoch_stage} | "
            f"Loss: {epoch_loss:.4f} | "
            f"EMA: {ema_loss:.4f} | "
            f"{sec_per_batch:.3f}s/batch | "
            f"Time: {elapsed:.1f}h",
            flush=True,
        )

        if (global_epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(
                save_dir,
                f"gasg_epoch_{global_epoch + 1}.pth",
            )

            save_checkpoint(
                path=ckpt_path,
                epoch=global_epoch + 1,
                generator=generator,
                g_optimizer=g_optimizer,
                config=config,
                loss_weights=get_stage_loss_weights(
                    config,
                    global_epoch,
                    args.ablation,
                ),
                ema_loss=ema_loss,
            )

            print(f"  Saved: {ckpt_path}", flush=True)

        global_epoch += 1

    # ========================================================
    # Final checkpoint
    # ========================================================
    final_path = os.path.join(save_dir, "gasg_best.pth")

    save_checkpoint(
        path=final_path,
        epoch=global_epoch,
        generator=generator,
        g_optimizer=g_optimizer,
        config=config,
        loss_weights=get_stage_loss_weights(
            config,
            max(global_epoch - 1, 0),
            args.ablation,
        ),
        ema_loss=ema_loss,
    )

    total_time = (time.time() - start_time) / 3600.0

    print("\nTraining complete!", flush=True)
    print(f"Total time: {total_time:.1f}h", flush=True)
    print(f"Final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    train()
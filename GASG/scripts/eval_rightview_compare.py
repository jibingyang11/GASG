"""
Unified right-view synthesis comparison + ablation on PersonDataset.

This script evaluates a set of right-view *generation* methods against the
ground-truth right view from PersonDataset and reports PSNR / SSIM / LPIPS
together with per-image inference time. All methods are run on the SAME
subset of test images for an apples-to-apples comparison.

Methods evaluated
-----------------
Comparison baselines (right-view generators):
  * identity            -- copy the left image as the predicted right view.
  * shift_const         -- horizontally shift the left image by a constant
                            disparity (mean disparity learned by GASG, in px).
  * oracle_warp         -- compute disparity from the GT (left, right) pair
                            via OpenCV StereoSGBM, then warp the left image.
                            This is the *upper bound* of any pure warp-based
                            method (no inpainting of occlusions).
  * dav2_warp           -- Depth-Anything-V2-Large (CVPR 2024) monocular
                            disparity + bilinear warp. The classical
                            "monocular + warp" pseudo-stereo baseline.
  * moge_warp           -- MoGe-1 ViT-L (NeurIPS 2024) monocular geometry
                            (point map -> Z-depth -> 1/depth disparity)
                            + bilinear warp.
  * gasg_untrained      -- the same GASG architecture & param count, with
                            random weights (no training). Isolates the value
                            of training vs. architectural priors.

GASG variants (ablations of the trained checkpoint):
  * gasg_warp_only      -- output only the geometry warp (forward dict 'warped')
  * gasg_direct_only    -- bypass the geometric warp; trust direct branch only
                            (alpha forced to 1.0)
  * gasg_no_refine      -- skip the final-refinement branch (output = fused)
  * gasg_full           -- the full GASG (paper hero method)

All metrics are computed in [0, 1] data range with LPIPS-AlexNet. The default
resolution is 256x576 because checkpoints/gasg_best.pth was trained at that
size; using 256x512 underestimates the current checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.gasg_net import GASG, count_parameters, warp_left_to_right
from utils.image_metrics import ImageMetrics, aggregate_image_metrics


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) tensor in [0,1] -> (H,W,3) numpy in [0,1]."""
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def left_tensor_to_01(left_tensor: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]."""
    return torch.clamp((left_tensor + 1.0) * 0.5, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Generators (each returns a (1,3,H,W) tensor in [0,1])
# -----------------------------------------------------------------------------


class IdentityGenerator:
    """Copy left image as right view."""

    name = "identity"

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        return left_tensor_to_01(left_tensor)

    def num_params(self) -> int:
        return 0


class ConstantShiftGenerator:
    """Horizontally shift left image by a constant disparity in pixels.

    disparity > 0 means the right image content is shifted to the LEFT relative
    to the left image. Stereo convention: x_left = x_right + disp ->
    right(x) = left(x + disp). We implement that with grid_sample, padding by
    border replication.
    """

    name = "shift_const"

    def __init__(self, device: torch.device, disp_pixels: float = 6.0) -> None:
        self.device = device
        self.disp = float(disp_pixels)

    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left01 = left_tensor_to_01(left_tensor).to(self.device)
        b, c, h, w = left01.shape
        disp = torch.full((b, 1, h, w), self.disp,
                          device=left01.device, dtype=left01.dtype)
        return torch.clamp(warp_left_to_right(left01, disp), 0.0, 1.0)

    def num_params(self) -> int:
        return 0


class DAV2WarpGenerator:
    """Monocular Depth-Anything-V2 disparity + bilinear warp.

    DA-V2 outputs a dense relative-depth-like (actually disparity-like) map.
    Larger value => closer object. We linearly rescale the raw output into a
    plausible pixel-disparity range (calibrated against the mean GT disparity
    of GASG so that the comparison is not unfairly penalised by scale alone)
    and warp the left image with grid_sample.
    """

    name = "dav2_warp"

    def __init__(
        self,
        device: torch.device,
        ckpt_dir: str = "checkpoints",
        third_party_dir: str = "third_party/Depth-Anything-V2",
        model_size: str = "Large",
        target_max_disp: float = 18.0,
    ) -> None:
        self.device = device
        self.target_max_disp = float(target_max_disp)

        sys.path.insert(0, third_party_dir)
        from depth_anything_v2.dpt import DepthAnythingV2

        configs = {
            "Small": {"encoder": "vits", "features": 64,
                       "out_channels": [48, 96, 192, 384]},
            "Base": {"encoder": "vitb", "features": 128,
                       "out_channels": [96, 192, 384, 768]},
            "Large": {"encoder": "vitl", "features": 256,
                       "out_channels": [256, 512, 1024, 1024]},
        }

        ckpt_path = os.path.join(
            ckpt_dir, f"depth_anything_v2_{model_size.lower()}.pth")

        self.model = DepthAnythingV2(**configs[model_size])
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True))
        self.model = self.model.to(device).eval()
        self._params = sum(p.numel() for p in self.model.parameters())

    def num_params(self) -> int:
        return self._params

    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        # left_tensor is (1,3,H,W) in [-1, 1]
        left01 = left_tensor_to_01(left_tensor).to(self.device)
        _, _, h, w = left01.shape

        # DA-V2 expects (H, W, 3) uint8 RGB
        img_np = (left01.squeeze(0).permute(1, 2, 0).cpu().numpy()
                  * 255.0).astype(np.uint8)

        with torch.no_grad():
            disp_raw = self.model.infer_image(img_np)  # (H', W'), float

        disp = torch.from_numpy(disp_raw.astype(np.float32))
        disp = disp.unsqueeze(0).unsqueeze(0).to(self.device)

        if disp.shape[-2:] != (h, w):
            disp = F.interpolate(
                disp, size=(h, w), mode="bilinear", align_corners=False)

        # Rescale the relative disparity to a pixel range.
        d_min, d_max = disp.amin(), disp.amax()
        if (d_max - d_min) > 1e-6:
            disp = (disp - d_min) / (d_max - d_min) * self.target_max_disp
        else:
            disp = torch.zeros_like(disp)

        warped = warp_left_to_right(left01, disp)
        return torch.clamp(warped, 0.0, 1.0)


class OracleSGBMWarpGenerator:
    """Oracle warp baseline.

    Computes a dense disparity from the *ground-truth* (left, right) pair
    using OpenCV StereoSGBM, then bilinearly warps the left image into the
    right-view frame. This is the upper bound of what any warp-only pseudo-
    stereo method could ever achieve on PersonDataset, because it has access
    to the true right-view image when computing disparity.

    Note: this generator REQUIRES the GT right view at call time. The eval
    loop passes the full sample dict so we can read it.
    """

    name = "oracle_warp"

    def __init__(self, device: torch.device,
                 num_disparities: int = 64, block_size: int = 5) -> None:
        import cv2  # noqa: F401  - imported here so failure surfaces in build()
        self.device = device
        self.num_disp = int(num_disparities)
        self.block = int(block_size)
        self._cv2 = __import__("cv2")
        self._matcher = self._cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block,
            P1=8 * 3 * self.block * self.block,
            P2=32 * 3 * self.block * self.block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
        )

    def num_params(self) -> int:
        return 0

    # The eval loop passes (left, right_gt) for this generator only.
    def __call__(self, left_tensor: torch.Tensor,
                 right_gt_tensor: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        assert right_gt_tensor is not None, \
            "oracle_warp needs the GT right view"

        left01 = left_tensor_to_01(left_tensor).to(self.device)
        right01 = left_tensor_to_01(right_gt_tensor).to(self.device)

        left_np = (left01.squeeze(0).permute(1, 2, 0).cpu().numpy()
                   * 255.0).astype(np.uint8)
        right_np = (right01.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    * 255.0).astype(np.uint8)

        gleft = self._cv2.cvtColor(left_np, self._cv2.COLOR_RGB2GRAY)
        gright = self._cv2.cvtColor(right_np, self._cv2.COLOR_RGB2GRAY)

        # SGBM returns disparity * 16 in fixed-point.
        disp_raw = self._matcher.compute(
            gleft, gright).astype(np.float32) / 16.0
        # Invalid pixels become -1 — replace with median of valid pixels.
        valid = disp_raw > 0
        if valid.any():
            fill = float(np.median(disp_raw[valid]))
            disp_raw[~valid] = fill
        else:
            disp_raw[:] = 0.0

        disp = torch.from_numpy(disp_raw).unsqueeze(0).unsqueeze(0).to(
            self.device).to(left01.dtype)
        warped = warp_left_to_right(left01, disp)
        return torch.clamp(warped, 0.0, 1.0)


class MoGeWarpGenerator:
    """MoGe-1 (ViT-L) monocular geometry + warp.

    MoGe predicts a per-pixel 3D point map. We take the Z (depth) channel,
    convert to inverse depth (disparity-like), normalise to [0, max_disp],
    and warp the left image. The forward() path is used directly to bypass
    the broken `infer()` post-processing in this MoGe vendor copy.
    """

    name = "moge_warp"

    def __init__(self, device: torch.device,
                 hf_repo: str = "Ruicheng/moge-vitl",
                 third_party_dir: str = "third_party/MoGe",
                 target_max_disp: float = 24.0,
                 num_tokens: int = 2400) -> None:
        # Workaround: MoGe v1.py uses `IO[bytes]` after `from typing import *`
        # but `IO` is not in `typing.__all__` on Python 3.9. Inject it.
        import typing
        if "IO" not in typing.__all__:
            typing.__all__.append("IO")

        os.environ.setdefault("HF_HOME", "data/hf_cache")
        sys.path.insert(0, third_party_dir)
        import importlib
        v1 = importlib.import_module("moge.model.v1")
        self.MoGeModel = v1.MoGeModel

        self.device = device
        self.target_max_disp = float(target_max_disp)
        self.num_tokens = int(num_tokens)

        self.model = self.MoGeModel.from_pretrained(hf_repo).to(device).eval()
        self._params = sum(p.numel() for p in self.model.parameters())

    def num_params(self) -> int:
        return self._params

    @torch.no_grad()
    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left01 = left_tensor_to_01(left_tensor).to(self.device)
        # MoGe wants (B, 3, H, W) in [0, 1].
        out = self.model.forward(left01, num_tokens=self.num_tokens)
        # points: (B, H, W, 3) — last channel is Z (depth).
        depth = out["points"][..., 2]  # (B, H, W)
        depth = torch.clamp(depth, min=1e-3)

        disp = 1.0 / depth  # disparity-like, larger = closer
        d_min, d_max = disp.amin(), disp.amax()
        if (d_max - d_min) > 1e-6:
            disp = (disp - d_min) / (d_max - d_min) * self.target_max_disp
        else:
            disp = torch.zeros_like(disp)

        disp = disp.unsqueeze(1)  # (B, 1, H, W)
        warped = warp_left_to_right(left01, disp)
        return torch.clamp(warped, 0.0, 1.0)


class GASGGenerator:
    """Wraps GASG with a configurable variant for ablation.

    variant in {full, warp_only, direct_only, no_refine}. All variants share
    the SAME trained weights -- only the assembling of the final output
    differs at inference time.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        variant: str = "full",
        gamma: float = 0.8,
        load_weights: bool = True,
    ) -> None:
        assert variant in {"full", "warp_only", "direct_only",
                           "no_refine", "untrained"}
        self.variant = variant
        self.device = device
        self.gamma = float(gamma)
        self.name = f"gasg_{variant}"

        # Even for the "untrained" variant we read the checkpoint to recover
        # the same architecture configuration as the trained model. This
        # keeps the params budget identical -- only the weights differ.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

        accepted = {
            "base_ch", "max_disp", "max_vshift", "use_vertical_flow",
            "positive_disp", "num_lowres_blocks", "num_bottleneck_blocks",
            "flow_downsample", "refine_ch", "coarse_residual_scale",
            "detail_residual_scale", "residual_scale", "fusion_bias",
            "final_refine_scale", "gamma_correction",
        }
        kwargs = {k: v for k, v in cfg.items() if k in accepted}

        self.model = GASG(**kwargs).to(device)
        if variant != "untrained" and load_weights:
            self.model.load_state_dict(ckpt.get("generator", ckpt))
        self.model.eval()
        self._params = count_parameters(self.model)

        self._gamma = self.gamma

    def num_params(self) -> int:
        return self._params

    @torch.no_grad()
    def __call__(self, left_tensor: torch.Tensor) -> torch.Tensor:
        left = left_tensor.to(self.device)
        out = self.model(left, return_dict=True)

        warped = out["warped"]
        direct = out["direct"]
        alpha = out["mask"]

        if self.variant == "warp_only":
            right = warped
        elif self.variant == "direct_only":
            right = direct
        elif self.variant == "no_refine":
            right = (1.0 - alpha) * warped + alpha * direct
        else:  # full or untrained -> use the assembled output of the model
            right = out["right"]

        right = torch.clamp(right, -1.0, 1.0)
        # Match model.inference: [-1,1] -> [0,1] then optional gamma.
        right01 = torch.clamp((right + 1.0) * 0.5, 0.0, 1.0)
        if abs(self._gamma - 1.0) > 1e-6:
            right01 = right01.pow(self._gamma)
        return torch.clamp(right01, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------


def select_indices(n_total: int, n_eval: Optional[int],
                   stride_seed: int = 0) -> List[int]:
    """Deterministically pick a subset of n_eval indices spread across the
    test split. Always identical given (n_total, n_eval, stride_seed) so all
    methods see the same images."""

    if n_eval is None or n_eval >= n_total:
        return list(range(n_total))

    # Evenly spaced picks for diversity; offset by seed for reproducibility.
    step = n_total / n_eval
    return [int(i * step) % n_total for i in range(n_eval)]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_method(
    generator: Callable[[torch.Tensor], torch.Tensor],
    name: str,
    dataset: PersonDataset,
    indices: List[int],
    img_metrics: ImageMetrics,
    device: torch.device,
    warmup: int = 2,
) -> Dict[str, float]:
    """Run `generator` on the chosen test indices, return aggregate metrics."""

    # The oracle SGBM generator needs the GT right view at call time; all
    # other generators only need the left image.
    needs_gt_right = isinstance(generator, OracleSGBMWarpGenerator)

    # Warm up the generator (first call is dominated by allocation/JIT).
    if warmup > 0:
        warm_sample = dataset[indices[0]]
        warm_left = warm_sample["left"].unsqueeze(0)
        warm_right = warm_sample["right"].unsqueeze(0)
        for _ in range(warmup):
            if needs_gt_right:
                _ = generator(warm_left, warm_right)
            else:
                _ = generator(warm_left)
        _sync_if_cuda(device)

    psnr, ssim, lpips_scores, runtimes = [], [], [], []

    for idx in tqdm(indices, desc=f"  {name}", leave=False):
        sample = dataset[idx]
        left = sample["left"].unsqueeze(0)
        right_gt = sample["right"]

        _sync_if_cuda(device)
        t0 = time.time()
        if needs_gt_right:
            right_pred = generator(left, right_gt.unsqueeze(0))
        else:
            right_pred = generator(left)
        _sync_if_cuda(device)
        elapsed_ms = (time.time() - t0) * 1000.0
        runtimes.append(elapsed_ms)

        pred_np = to_numpy_image(right_pred)
        gt_np = to_numpy_image((right_gt + 1.0) * 0.5)

        psnr.append(img_metrics.compute_psnr(pred_np, gt_np))
        ssim.append(img_metrics.compute_ssim(pred_np, gt_np))
        lpips_scores.append(img_metrics.compute_lpips(pred_np, gt_np))

    return {
        "method": name,
        "n_eval": len(indices),
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "lpips": float(np.mean(lpips_scores)),
        "psnr_std": float(np.std(psnr)),
        "ssim_std": float(np.std(ssim)),
        "lpips_std": float(np.std(lpips_scores)),
        "inference_time_ms_mean": float(np.mean(runtimes)),
        "inference_time_ms_median": float(np.median(runtimes)),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_summary(rows: List[Dict[str, float]], out_dir: str,
                  device: str, n_eval: int, max_disp: float,
                  height: int, width: int) -> None:
    """Write a markdown + CSV summary across all methods."""

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary.csv")
    md_path = os.path.join(out_dir, "summary.md")
    json_path = os.path.join(out_dir, "summary.json")

    cols = ["method", "psnr", "ssim", "lpips",
            "inference_time_ms_mean", "inference_time_ms_median",
            "params_M", "n_eval"]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float)
                else str(r.get(c, "")) for c in cols) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Right-view synthesis comparison on PersonDataset (test)\n\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Evaluated samples: **{n_eval}**\n")
        f.write(f"- Image resolution: {height} x {width}\n")
        f.write(f"- Constant-shift baseline disparity: {max_disp:.1f} px\n\n")
        f.write("| Method | PSNR (dB) | SSIM | LPIPS | Time (ms) | Params (M) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| `{r['method']}` "
                f"| {r['psnr']:.2f} "
                f"| {r['ssim']:.4f} "
                f"| {r['lpips']:.4f} "
                f"| {r['inference_time_ms_mean']:.1f} "
                f"| {r.get('params_M', 0.0):.2f} |\n"
            )
        f.write("\n_PSNR / SSIM: higher is better. LPIPS / Time: lower is better._\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSummary written to: {md_path}")
    print(f"                      {csv_path}")
    print(f"                      {json_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Right-view synthesis: comparison + ablation")
    p.add_argument("--gasg_ckpt", default="checkpoints/gasg_best.pth")
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_dir", default="results/rightview_generation")
    p.add_argument("--device", default="cuda",
                   help="Device for inference and LPIPS.")
    p.add_argument("--height", type=int, default=256,
                   help="Evaluation image height. Matches GASG training by default.")
    p.add_argument("--width", type=int, default=576,
                   help="Evaluation image width. Current gasg_best.pth was trained at 256x576.")
    p.add_argument("--max_samples", type=int, default=200,
                   help="How many test images to evaluate per method "
                        "(0 / negative = all).")
    p.add_argument("--methods", default=None,
                   help="Comma-separated subset of method names to run. "
                        "Default: run them all.")
    p.add_argument("--gasg_gamma", type=float, default=1.0,
                   help="Inference-time gamma applied to GASG variants "
                        "(1.0 matches the current checkpoint config).")
    p.add_argument("--shift_disp", type=float, default=9.0,
                   help="Constant disparity (px) for the shift baseline. "
                        "9 px is roughly the 256x576 equivalent of the old "
                        "8 px baseline at 256x512.")
    p.add_argument("--dav2_size", default="Large",
                   choices=["Small", "Base", "Large"])
    p.add_argument("--dav2_target_disp", type=float, default=18.0,
                   help="Max pixel disparity used to normalise DA-V2's raw "
                        "relative output before warping.")
    p.add_argument("--moge_target_disp", type=float, default=24.0,
                   help="Max pixel disparity used to normalise MoGe's "
                        "1/depth disparity-like output before warping.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = PersonDataset(
        root=args.person_root, split="test",
        height=args.height, width=args.width)

    n_total = len(dataset)
    n_eval = None if args.max_samples is None or args.max_samples <= 0 \
        else args.max_samples
    indices = select_indices(n_total, n_eval)
    print(f"Evaluating {len(indices)} / {n_total} test samples.")

    img_metrics = ImageMetrics(device=args.device)

    # Lazily build only the requested methods to avoid loading DA-V2 when
    # it is not needed.
    requested = (
        [m.strip() for m in args.methods.split(",")]
        if args.methods else None
    )

    def want(name: str) -> bool:
        return requested is None or name in requested

    builders: Dict[str, Callable[[], Callable]] = {
        # ----- Comparison baselines -----
        "identity": lambda: IdentityGenerator(device),
        "shift_const": lambda: ConstantShiftGenerator(
            device, disp_pixels=args.shift_disp),
        "oracle_warp": lambda: OracleSGBMWarpGenerator(device),
        "dav2_warp": lambda: DAV2WarpGenerator(
            device, model_size=args.dav2_size,
            target_max_disp=args.dav2_target_disp),
        "moge_warp": lambda: MoGeWarpGenerator(
            device, target_max_disp=args.moge_target_disp),
        # ----- GASG ablations -----
        "gasg_untrained": lambda: GASGGenerator(
            args.gasg_ckpt, device, variant="untrained",
            gamma=args.gasg_gamma),
        "gasg_warp_only": lambda: GASGGenerator(
            args.gasg_ckpt, device, variant="warp_only",
            gamma=args.gasg_gamma),
        "gasg_direct_only": lambda: GASGGenerator(
            args.gasg_ckpt, device, variant="direct_only",
            gamma=args.gasg_gamma),
        "gasg_no_refine": lambda: GASGGenerator(
            args.gasg_ckpt, device, variant="no_refine",
            gamma=args.gasg_gamma),
        "gasg_full": lambda: GASGGenerator(
            args.gasg_ckpt, device, variant="full",
            gamma=args.gasg_gamma),
    }

    rows: List[Dict[str, float]] = []
    for name, build in builders.items():
        if not want(name):
            continue
        print(f"\n[Method: {name}]")
        try:
            gen = build()
        except Exception as e:
            print(f"  Skipped {name}: {type(e).__name__}: {e}")
            continue

        params_m = float(getattr(gen, "num_params", lambda: 0)()) / 1e6

        row = evaluate_method(
            gen, name, dataset, indices, img_metrics, device)
        row["params_M"] = params_m
        row["height"] = int(args.height)
        row["width"] = int(args.width)
        row["device"] = str(args.device)
        rows.append(row)

        # Per-method JSON for quick re-reading.
        with open(os.path.join(args.save_dir, f"{name}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        print(
            f"  PSNR={row['psnr']:.2f} dB | "
            f"SSIM={row['ssim']:.4f} | "
            f"LPIPS={row['lpips']:.4f} | "
            f"time={row['inference_time_ms_mean']:.1f} ms | "
            f"params={params_m:.2f} M"
        )

        # Free memory between heavy methods.
        del gen
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary(rows, args.save_dir, args.device, len(indices),
                  args.shift_disp, args.height, args.width)


if __name__ == "__main__":
    main()

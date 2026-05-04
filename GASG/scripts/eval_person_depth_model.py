"""Evaluate one depth-estimation method on PersonDataset.

The script is intentionally single-method.  Third-party depth models often
import generic module names such as ``depth_anything_v2`` or ``core``; running
each method in a fresh process keeps the comparison reproducible.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.person_dataset import PersonDataset
from models.gasg_net import GASG
from scripts.eval_rightview_compare import (
    ConstantShiftGenerator,
    DAV2WarpGenerator,
    MoGeWarpGenerator,
    select_indices,
)
from utils.defom_runner import DEFOMRunner
from utils.depth_metrics import (
    compute_depth_errors,
    median_scaling,
    aggregate_metrics,
    format_metrics,
)
from utils.visualization import colorize_depth


METHOD_INFO: Dict[str, Dict[str, str]] = {
    "gasg_defom": {
        "pretty": "GASG right view -> DEFOM-Stereo",
        "type": "GASG pseudo-stereo",
        "scale": "metric",
    },
    "defom_real": {
        "pretty": "DEFOM-Stereo (real right upper bound)",
        "type": "stereo upper bound",
        "scale": "metric",
    },
    "defom_identity": {
        "pretty": "DEFOM-Stereo (copy-left right)",
        "type": "pseudo-stereo floor",
        "scale": "metric",
    },
    "defom_shift": {
        "pretty": "DEFOM-Stereo (constant-shift right)",
        "type": "pseudo-stereo floor",
        "scale": "metric",
    },
    "defom_dav2_warp": {
        "pretty": "DA-V2 warp right view -> DEFOM-Stereo",
        "type": "pseudo-stereo generator",
        "scale": "metric",
    },
    "defom_moge_warp": {
        "pretty": "MoGe warp right view -> DEFOM-Stereo",
        "type": "pseudo-stereo generator",
        "scale": "metric",
    },
    "defom_zerostereo": {
        "pretty": "ZeroStereo/StereoGen right view -> DEFOM-Stereo",
        "type": "published pseudo-stereo generator",
        "scale": "metric",
    },
    "defom_stereospace": {
        "pretty": "StereoSpace right view -> DEFOM-Stereo",
        "type": "published pseudo-stereo generator",
        "scale": "metric",
    },
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def left_01(sample: Dict, device: torch.device) -> torch.Tensor:
    return ((sample["left"].unsqueeze(0) + 1.0) * 0.5).to(device)


def tensor01_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x.squeeze(0)
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_depth(pred: np.ndarray, shape_hw) -> np.ndarray:
    if pred.shape == shape_hw:
        return pred.astype(np.float32)
    return np.array(
        Image.fromarray(pred.astype(np.float32)).resize(
            (shape_hw[1], shape_hw[0]), Image.BILINEAR),
        dtype=np.float32,
    )


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


def load_zerostereo_map(work_root: str) -> Dict[int, str]:
    meta_path = os.path.join(work_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"ZeroStereo metadata not found: {meta_path}. "
            "Run scripts/baselines/eval_zerostereo_official.py first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping = {}
    for rec in meta["records"]:
        mapping[int(rec["person_index"])] = os.path.join(work_root, rec["right"])
    return mapping


def load_saved_right_map(work_root: str, method_name: str) -> Dict[int, str]:
    meta_path = os.path.join(work_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{method_name} metadata not found: {meta_path}. "
            f"Generate saved right views before running {method_name} depth eval."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping = {}
    for rec in meta["records"]:
        rel = rec.get("right") or rec.get("generated") or rec.get("output")
        if rel is None:
            raise KeyError(
                f"{method_name} metadata record lacks a right/generated path: {rec}"
            )
        mapping[int(rec["person_index"])] = os.path.join(work_root, rel)
    return mapping


class Predictor:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device

    def predict(self, sample: Dict, index: int) -> np.ndarray:
        raise NotImplementedError


class DefomPredictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device,
                 mode: str):
        super().__init__(args, device)
        self.mode = mode
        self.focal = args.focal_length or 600.0 * float(args.width) / 906.0
        self.gasg: Optional[GASG] = None
        self.shift = None
        self.zerostereo = None
        self.saved_right = None
        self.right_generator = None
        if mode == "gasg":
            self.gasg = load_gasg(args.gasg_ckpt, device)
        elif mode == "shift":
            self.shift = ConstantShiftGenerator(device, disp_pixels=args.shift_disp)
        elif mode == "zerostereo":
            self.zerostereo = load_zerostereo_map(args.zerostereo_work_root)
        elif mode == "stereospace":
            self.saved_right = load_saved_right_map(
                args.stereospace_work_root, "StereoSpace")
        elif mode == "dav2_warp":
            self.right_generator = DAV2WarpGenerator(
                device,
                ckpt_dir=args.dav2_ckpt_dir,
                third_party_dir=args.dav2_third_party_dir,
                model_size=args.dav2_size,
                target_max_disp=args.dav2_target_disp,
            )
        elif mode == "moge_warp":
            self.right_generator = MoGeWarpGenerator(
                device,
                hf_repo=args.moge_repo,
                third_party_dir=args.moge_third_party_dir,
                target_max_disp=args.moge_target_disp,
                num_tokens=args.moge_tokens,
            )
        self.defom = DEFOMRunner(
            args.defom_ckpt, device,
            valid_iters=args.valid_iters,
            scale_iters=args.scale_iters,
        )

    def _right(self, sample: Dict, index: int, left: torch.Tensor) -> torch.Tensor:
        if self.mode == "gasg":
            assert self.gasg is not None
            return self.gasg.inference(
                sample["left"].unsqueeze(0).to(self.device),
                gamma=self.args.gasg_gamma,
            )
        if self.mode == "real":
            return ((sample["right"].unsqueeze(0) + 1.0) * 0.5).to(self.device)
        if self.mode == "identity":
            return left
        if self.mode == "shift":
            assert self.shift is not None
            return self.shift(sample["left"].unsqueeze(0))
        if self.mode == "zerostereo":
            assert self.zerostereo is not None
            path = self.zerostereo.get(int(index))
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(
                    f"No ZeroStereo right view for PersonDataset index {index}")
            img = Image.open(path).convert("RGB").resize(
                (self.args.width, self.args.height), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
                self.device)
        if self.mode == "stereospace":
            assert self.saved_right is not None
            path = self.saved_right.get(int(index))
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(
                    f"No StereoSpace right view for PersonDataset index {index}")
            img = Image.open(path).convert("RGB").resize(
                (self.args.width, self.args.height), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
                self.device)
        if self.mode in {"dav2_warp", "moge_warp"}:
            assert self.right_generator is not None
            return self.right_generator(sample["left"].unsqueeze(0))
        raise ValueError(self.mode)

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        left = left_01(sample, self.device)
        right = self._right(sample, index, left)
        disp = self.defom.disparity(left, right)
        return self.focal * self.args.baseline / (
            disp.squeeze().detach().cpu().numpy() + 1e-8)


class DAV2Predictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(args, device)
        sys.path.insert(0, "third_party/Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2

        cfg = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        self.model = DepthAnythingV2(**cfg)
        ckpt = torch.load(
            args.dav2_ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        img = tensor01_to_uint8_hwc((sample["left"] + 1.0) * 0.5)
        pred = self.model.infer_image(img).astype(np.float32)
        return 1.0 / (pred + 1e-6)


class MoGePredictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(args, device)
        import importlib
        import typing

        if "IO" not in typing.__all__:
            typing.__all__.append("IO")
        sys.path.insert(0, "third_party/MoGe")
        v1 = importlib.import_module("moge.model.v1")
        self.model = v1.MoGeModel.from_pretrained(
            args.moge_repo).to(device).eval()

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        img = ((sample["left"].unsqueeze(0) + 1.0) * 0.5).to(self.device)
        out = self.model.forward(img, num_tokens=self.args.moge_tokens)
        depth = out["points"][..., 2].squeeze(0)
        depth = torch.clamp(depth, min=1e-4)
        return depth.detach().cpu().numpy().astype(np.float32)


class DepthProPredictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(args, device)
        sys.path.insert(0, "third_party/depth_pro/src")
        import depth_pro
        from depth_pro.depth_pro import (
            DepthProConfig,
            DEFAULT_MONODEPTH_CONFIG_DICT,
        )

        self.depth_pro = depth_pro
        cfg = DepthProConfig(
            **{
                **DEFAULT_MONODEPTH_CONFIG_DICT.__dict__,
                "checkpoint_uri": os.path.abspath(args.depth_pro_ckpt),
            }
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=cfg,
            device=device,
            precision=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        image, _, f_px = self.depth_pro.load_rgb(sample["left_path"])
        image = self.transform(image).to(self.device)
        pred = self.model.infer(image, f_px=f_px)["depth"]
        return pred.squeeze().detach().cpu().numpy().astype(np.float32)


class Metric3DPredictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(args, device)
        self.model = torch.hub.load(
            "yvanyin/metric3d", "metric3d_vit_large",
            pretrain=False, trust_repo=True)
        ckpt = torch.load(args.metric3d_ckpt, map_location="cpu",
                          weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt),
                                   strict=False)
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        img = tensor01_to_uint8_hwc((sample["left"] + 1.0) * 0.5)
        rgb = img.astype(np.float32)
        in_h, in_w = 616, 1064
        h, w = rgb.shape[:2]
        scale = min(in_h / h, in_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = np.asarray(Image.fromarray(img).resize(
            (new_w, new_h), Image.BILINEAR), dtype=np.float32)
        pad = [123.675, 116.28, 103.53]
        pad_h = in_h - new_h
        pad_w = in_w - new_w
        top = pad_h // 2
        left = pad_w // 2
        canvas = np.zeros((in_h, in_w, 3), dtype=np.float32)
        canvas[:, :, :] = np.asarray(pad, dtype=np.float32)
        canvas[top:top + new_h, left:left + new_w] = rgb
        mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
        x = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
        x = ((x - mean) / std).unsqueeze(0).to(self.device)
        pred, _, _ = self.model.inference({"input": x})
        pred = pred.squeeze()[top:top + new_h, left:left + new_w]
        pred = F.interpolate(
            pred[None, None], size=(h, w), mode="bilinear",
            align_corners=False).squeeze()
        return pred.detach().cpu().numpy().astype(np.float32)


class DA3Predictor(Predictor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(args, device)
        sys.path.insert(0, "third_party/Depth-Anything-3/src")
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(args.da3_repo).to(device)

    @torch.inference_mode()
    def predict(self, sample: Dict, index: int) -> np.ndarray:
        img = Image.fromarray(tensor01_to_uint8_hwc(
            (sample["left"] + 1.0) * 0.5))
        out = self.model.inference([img], process_res=self.args.da3_process_res)
        if isinstance(out, dict):
            pred = out.get("depth")
            if pred is None:
                pred = out.get("depths")
        else:
            pred = getattr(out, "depth", None)
        if isinstance(pred, list):
            pred = pred[0]
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred).squeeze().astype(np.float32)
        return pred


def build_predictor(method: str, args: argparse.Namespace,
                    device: torch.device) -> Predictor:
    if method == "gasg_defom":
        return DefomPredictor(args, device, "gasg")
    if method == "defom_real":
        return DefomPredictor(args, device, "real")
    if method == "defom_identity":
        return DefomPredictor(args, device, "identity")
    if method == "defom_shift":
        return DefomPredictor(args, device, "shift")
    if method == "defom_dav2_warp":
        return DefomPredictor(args, device, "dav2_warp")
    if method == "defom_moge_warp":
        return DefomPredictor(args, device, "moge_warp")
    if method == "defom_zerostereo":
        return DefomPredictor(args, device, "zerostereo")
    if method == "defom_stereospace":
        return DefomPredictor(args, device, "stereospace")
    raise ValueError(f"Unknown method: {method}")


def save_visual_bundle(args: argparse.Namespace, method: str, index: int,
                       sample: Dict, pred: np.ndarray, gt: np.ndarray) -> None:
    pred_dir = os.path.join(args.save_dir, "predictions", method)
    vis_dir = os.path.join(args.save_dir, "visuals", method)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f"{index:05d}.npy"), pred.astype(np.float32))

    left_img = tensor01_to_uint8_hwc((sample["left"] + 1.0) * 0.5)
    Image.fromarray(left_img).save(os.path.join(vis_dir, f"{index:05d}_left.png"))

    max_d = args.max_depth
    Image.fromarray((colorize_depth(gt, min_d=0.0, max_d=max_d) * 255).astype(
        np.uint8)).save(os.path.join(vis_dir, f"{index:05d}_gt.png"))
    Image.fromarray((colorize_depth(pred, min_d=0.0, max_d=max_d) * 255).astype(
        np.uint8)).save(os.path.join(vis_dir, f"{index:05d}_pred.png"))
    err = np.abs(pred - gt)
    Image.fromarray((colorize_depth(err, min_d=0.0, max_d=5.0, cmap="inferno")
                     * 255).astype(np.uint8)).save(
        os.path.join(vis_dir, f"{index:05d}_error.png"))


def evaluate(args: argparse.Namespace) -> Dict:
    if args.method not in METHOD_INFO:
        raise ValueError(f"Unknown method {args.method}. Choices: "
                         f"{', '.join(METHOD_INFO)}")
    device = torch.device(args.device)
    ds = PersonDataset(
        args.person_root, split="test", height=args.height, width=args.width)
    if args.use_saved_indices and args.method == "defom_stereospace":
        indices = sorted(load_saved_right_map(
            args.stereospace_work_root, "StereoSpace"))
        if args.max_samples is not None:
            indices = indices[:args.max_samples]
    elif args.use_saved_indices and args.method == "defom_zerostereo":
        indices = sorted(load_zerostereo_map(args.zerostereo_work_root))
        if args.max_samples is not None:
            indices = indices[:args.max_samples]
    else:
        indices = select_indices(len(ds), args.max_samples)
    vis_indices = set()
    if args.save_visuals and args.num_visuals > 0 and indices:
        vis_positions = select_indices(
            len(indices), min(args.num_visuals, len(indices)))
        vis_indices = set(indices[pos] for pos in vis_positions)

    predictor = build_predictor(args.method, args, device)
    info = METHOD_INFO[args.method]
    metrics_list: List[Dict[str, float]] = []
    times: List[float] = []

    for idx in tqdm(indices, desc=info["pretty"]):
        sample = ds[idx]
        gt = sample["depth"].squeeze().numpy().astype(np.float32)
        sync(device)
        t0 = time.perf_counter()
        pred = predictor.predict(sample, idx)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        pred = resize_depth(pred, gt.shape)
        if info["scale"] == "median":
            pred = median_scaling(pred, gt, max_depth=args.max_depth)
        metrics_list.append(compute_depth_errors(
            gt, pred, max_depth=args.max_depth))
        if idx in vis_indices:
            save_visual_bundle(args, args.method, idx, sample, pred, gt)

    avg = aggregate_metrics(metrics_list)
    for key in list(avg.keys()):
        vals = [m[key] for m in metrics_list]
        avg[f"{key}_std"] = float(np.std(vals))
    avg.update({
        "method": args.method,
        "pretty": info["pretty"],
        "type": info["type"],
        "scale_alignment": info["scale"],
        "n_eval": len(metrics_list),
        "height": int(args.height),
        "width": int(args.width),
        "device": str(device),
        "max_depth": float(args.max_depth),
        "inference_time_ms_mean": float(np.mean(times)),
        "inference_time_ms_median": float(np.median(times)),
    })
    if args.method.startswith("defom") or args.method == "gasg_defom":
        avg["focal_length"] = float(
            args.focal_length or 600.0 * float(args.width) / 906.0)
        avg["baseline"] = float(args.baseline)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{args.method}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)

    print("\n" + format_metrics(avg, info["pretty"]))
    print(f"Wrote: {out_path}")
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return avg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=sorted(METHOD_INFO))
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_dir", default="results/pseudostereo_depth")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--max_samples", type=int, default=200,
                   help="Use 0 or a value >=1600 for all test images.")
    p.add_argument("--max_depth", type=float, default=25.4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--num_visuals", type=int, default=6)
    p.add_argument("--use_saved_indices", action="store_true",
                   help="Evaluate only indices present in saved right-view metadata.")
    p.add_argument("--gasg_ckpt", default="checkpoints/gasg_best.pth")
    p.add_argument("--gasg_gamma", type=float, default=1.0)
    p.add_argument("--defom_ckpt",
                   default="third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth")
    p.add_argument("--valid_iters", type=int, default=12)
    p.add_argument("--scale_iters", type=int, default=3)
    p.add_argument("--baseline", type=float, default=1.0)
    p.add_argument("--focal_length", type=float, default=None)
    p.add_argument("--shift_disp", type=float, default=9.0)
    p.add_argument("--zerostereo_work_root",
                   default="data/zerostereo_person_test")
    p.add_argument("--stereospace_work_root",
                   default="data/stereospace_person_test")
    p.add_argument("--dav2_ckpt",
                   default="checkpoints/depth_anything_v2_large.pth")
    p.add_argument("--dav2_ckpt_dir", default="checkpoints")
    p.add_argument("--dav2_third_party_dir",
                   default="third_party/Depth-Anything-V2")
    p.add_argument("--dav2_size", default="Large",
                   choices=["Small", "Base", "Large"])
    p.add_argument("--dav2_target_disp", type=float, default=18.0)
    p.add_argument("--moge_repo", default="Ruicheng/moge-vitl")
    p.add_argument("--moge_third_party_dir", default="third_party/MoGe")
    p.add_argument("--moge_target_disp", type=float, default=24.0)
    p.add_argument("--moge_tokens", type=int, default=2400)
    args = p.parse_args()
    if args.max_samples == 0:
        args.max_samples = None
    return args


if __name__ == "__main__":
    evaluate(parse_args())

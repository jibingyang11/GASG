"""Thin inference wrapper around the official DEFOM-Stereo code."""

import os
import sys
from argparse import Namespace

import torch


def _drop_cached_modules(prefixes):
    """Remove modules that DEFOM imports by generic top-level names."""
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in prefixes):
            del sys.modules[name]


def _default_args():
    return Namespace(
        mixed_precision=False,
        valid_iters=12,
        scale_iters=3,
        dinov2_encoder="vitl",
        idepth_scale=0.5,
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=2,
        corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
        n_downsample=2,
        context_norm="batch",
        n_gru_layers=3,
    )


class DEFOMRunner:
    """Load DEFOM-Stereo once and run disparity inference on tensors."""

    def __init__(self, checkpoint, device, repo_dir="third_party/DEFOM-Stereo",
                 valid_iters=12, scale_iters=3):
        self.device = torch.device(device)
        self.repo_dir = os.path.abspath(repo_dir)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"DEFOM checkpoint not found: {checkpoint}. "
                "Run the official DEFOM download or point --defom_ckpt to "
                "third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth"
            )

        # DEFOM-Stereo vendors a modified depth_anything_v2 package.  If a
        # qualitative script imported the public Depth-Anything-V2 package
        # first, Python would otherwise reuse the cached public module and the
        # DEFOM checkpoint would no longer match the instantiated model.
        _drop_cached_modules(["core", "depth_anything_v2"])
        sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir)
                    != self.repo_dir]
        sys.path.insert(0, self.repo_dir)
        from core.defom_stereo import DEFOMStereo
        from core.utils.utils import InputPadder

        args = _default_args()
        args.valid_iters = valid_iters
        args.scale_iters = scale_iters
        self.args = args
        self.InputPadder = InputPadder

        model = DEFOMStereo(args)
        ckpt = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(ckpt.get("model", ckpt))
        self.model = model.to(self.device).eval()

    def disparity(self, left, right):
        """
        Args:
            left/right: (B, 3, H, W), either [0, 1] or [0, 255].
        Returns:
            (B, H, W) disparity in input-image pixels.
        """
        left = left.to(self.device).float()
        right = right.to(self.device).float()
        if left.max() <= 2.0:
            left = left * 255.0
        if right.max() <= 2.0:
            right = right * 255.0

        padder = self.InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with torch.no_grad():
            disp = self.model(
                left, right,
                iters=self.args.valid_iters,
                scale_iters=self.args.scale_iters,
                test_mode=True,
            )
        disp = padder.unpad(disp)
        return disp.squeeze(1)

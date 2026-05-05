"""Thin inference wrapper around the official RAFT-Stereo code."""

from __future__ import annotations

import os
import sys
from argparse import Namespace

import torch


def _drop_cached_modules(prefixes):
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in prefixes):
            del sys.modules[name]


def _default_args():
    return Namespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="alt",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=False,
    )


class RAFTStereoRunner:
    """Load RAFT-Stereo once and run disparity inference on tensors."""

    def __init__(
        self,
        checkpoint,
        device,
        repo_dir="third_party/RAFT-Stereo",
        valid_iters=16,
        corr_implementation="alt",
    ):
        self.device = torch.device(device)
        self.repo_dir = os.path.abspath(repo_dir)
        self.valid_iters = valid_iters
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"RAFT-Stereo checkpoint not found: {checkpoint}. "
                "Run third_party/RAFT-Stereo/download_models.sh or point "
                "--raft_ckpt to an official RAFT-Stereo checkpoint."
            )

        _drop_cached_modules(["core", "raft_stereo"])
        sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir)
                    != self.repo_dir]
        sys.path.insert(0, self.repo_dir)

        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder

        args = _default_args()
        args.corr_implementation = corr_implementation
        self.args = args
        self.InputPadder = InputPadder

        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        state = torch.load(checkpoint, map_location=self.device,
                           weights_only=False)
        model.load_state_dict(state, strict=True)
        self.model = model.module.to(self.device).eval()

    def disparity(self, left, right):
        """
        Args:
            left/right: (B, 3, H, W), either [0, 1] or [0, 255].
        Returns:
            (B, H, W) positive disparity in input-image pixels.
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
            _, flow_up = self.model(
                left, right,
                iters=self.valid_iters,
                test_mode=True,
            )
        flow_up = padder.unpad(flow_up)
        return (-flow_up.squeeze(1)).clamp_min(1e-4)

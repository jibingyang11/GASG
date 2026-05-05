"""Deep3D / Single-View-Stereo-Matching style right-view synthesis.

This module implements the common view-synthesis idea used by Deep3D and the
upper synthesis branch of Single View Stereo Matching: a CNN predicts a
probability volume over a fixed set of horizontal disparities, and a selection
layer renders the right view by softly choosing shifted pixels from the left
view.

It is intentionally a faithful baseline, not a GASG variant: it cannot invent
new content outside the left-view pixels, so dis-occluded regions remain the
main failure mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gasg_net import warp_left_to_right


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Deep3DStyleGenerator(nn.Module):
    """Probabilistic disparity-selection generator.

    Args:
        max_disp: largest candidate disparity in resized-image pixels.
        num_bins: number of discrete disparity candidates.
        base_ch: width of the encoder-decoder CNN.
    """

    def __init__(
        self,
        max_disp: float = 96.0,
        num_bins: int = 33,
        base_ch: int = 24,
    ) -> None:
        super().__init__()
        self.max_disp = float(max_disp)
        self.num_bins = int(num_bins)
        self.register_buffer(
            "disp_values",
            torch.linspace(0.0, self.max_disp, self.num_bins).view(1, self.num_bins, 1, 1),
        )

        c = int(base_ch)
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2, stride=2)
        self.enc3 = ConvBlock(c * 2, c * 4, stride=2)
        self.enc4 = ConvBlock(c * 4, c * 8, stride=2)
        self.mid = ConvBlock(c * 8, c * 8)

        self.up3 = nn.Conv2d(c * 8 + c * 4, c * 4, 3, padding=1)
        self.dec3 = ConvBlock(c * 4, c * 4)
        self.up2 = nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1)
        self.dec2 = ConvBlock(c * 2, c * 2)
        self.up1 = nn.Conv2d(c * 2 + c, c, 3, padding=1)
        self.dec1 = ConvBlock(c, c)
        self.head = nn.Conv2d(c, self.num_bins, 3, padding=1)

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        if x.min() < -0.05:
            x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)

    def _features(self, left01: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(left01)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        x = self.mid(e4)

        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(F.relu(self.up3(torch.cat([x, e3], dim=1)), inplace=True))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(F.relu(self.up2(torch.cat([x, e2], dim=1)), inplace=True))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(F.relu(self.up1(torch.cat([x, e1], dim=1)), inplace=True))
        return x

    def render_from_probs(self, left01: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = left01.shape
        k = self.num_bins
        disp = self.disp_values.to(left01.device, left01.dtype)
        disp = disp.expand(b, k, h, w).reshape(b * k, 1, h, w)

        # Candidate shifted images are constants with respect to the CNN
        # weights; keeping them outside the autograd graph saves memory.
        with torch.no_grad():
            candidates = warp_left_to_right(
                left01[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w),
                disp,
            ).reshape(b, k, c, h, w)

        return (probs[:, :, None] * candidates).sum(dim=1).clamp(0.0, 1.0)

    def forward(self, left: torch.Tensor, return_dict: bool = False):
        left01 = self._to_01(left)
        logits = self.head(self._features(left01))
        probs = torch.softmax(logits, dim=1)
        right = self.render_from_probs(left01, probs)
        if return_dict:
            disp = (probs * self.disp_values.to(probs.device, probs.dtype)).sum(dim=1, keepdim=True)
            return {"right": right, "probs": probs, "disp": disp}
        return right

    @torch.no_grad()
    def inference(self, left: torch.Tensor) -> torch.Tensor:
        return self.forward(left)

"""
GASG: stronger warp + direct-detail fusion generator.

Designed for left-to-right view synthesis.

Main architecture features:
  1. Deeper encoder-decoder backbone.
  2. Residual Dense Blocks for stronger local texture learning.
  3. Stronger direct detail/color branch.
  4. Learned fusion between warped image and direct image.
  5. Final refinement branch after fusion.

Output dict remains compatible with the existing training script and losses.py:
  right
  warped
  direct
  disp
  residual
  mask
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================

def _groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            ConvGNAct(ch, ch, 3),
            ConvGNAct(ch, ch, 3, act=False),
        )

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta.tanh() * self.body(x)


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()

        hidden = max(ch // reduction, 8)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.cat([avg, mx], dim=1)
        return x * self.net(att)


class ResidualDenseBlock(nn.Module):
    """
    Residual dense block.

    This block increases local texture learning capacity.
    It is useful for recovering clothing color and human details.
    """

    def __init__(
        self,
        ch: int,
        growth_ch: Optional[int] = None,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        if growth_ch is None:
            growth_ch = max(ch // 2, 16)

        self.layers = nn.ModuleList()
        in_ch = ch

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(_groups(growth_ch), growth_ch),
                    nn.SiLU(inplace=True),
                )
            )
            in_ch += growth_ch

        self.lff = nn.Sequential(
            nn.Conv2d(in_ch, ch, 1, bias=False),
            nn.GroupNorm(_groups(ch), ch),
        )

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]

        for layer in self.layers:
            y = layer(torch.cat(feats, dim=1))
            feats.append(y)

        y = self.lff(torch.cat(feats, dim=1))
        return x + self.beta.tanh() * y


class DetailBlock(nn.Module):
    """
    Low/high-frequency separated residual block.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()

        self.low = nn.Sequential(
            ConvGNAct(ch, ch, 3),
            ChannelAttention(ch),
        )

        self.high = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(_groups(ch), ch),
            nn.SiLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            ConvGNAct(ch * 2, ch, 1),
            ConvGNAct(ch, ch, 3, act=False),
        )

        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low

        y = torch.cat([self.low(low), self.high(high)], dim=1)
        y = self.fuse(y)
        y = self.ca(y)
        y = self.sa(y)

        return x + self.beta.tanh() * y


class StrongTextureBlock(nn.Module):
    """
    Stronger texture block:
      ResidualDenseBlock + DetailBlock + ResBlock
    """

    def __init__(self, ch: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ResidualDenseBlock(ch, num_layers=4),
            DetailBlock(ch),
            ResBlock(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Encoder / Decoder
# ============================================================

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2) -> None:
        super().__init__()

        layers = [ConvGNAct(in_ch, out_ch, 3, stride=2)]

        for _ in range(blocks):
            layers.append(StrongTextureBlock(out_ch))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpFuseBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        blocks: int = 2,
    ) -> None:
        super().__init__()

        self.pre = ConvGNAct(in_ch, out_ch, 3)

        layers = [ConvGNAct(out_ch + skip_ch, out_ch, 3)]

        for _ in range(blocks):
            layers.append(StrongTextureBlock(out_ch))

        self.fuse = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# ============================================================
# Geometry warp
# ============================================================

def warp_left_to_right(left: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """
    Warp left image to right-view coordinates.

    Stereo convention:
      x_left = x_right + disparity

    Therefore, target right pixel samples from left at x + disp.
    """
    b, c, h, w = left.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=left.device, dtype=left.dtype),
        torch.linspace(-1.0, 1.0, w, device=left.device, dtype=left.dtype),
        indexing="ij",
    )

    base_grid = torch.stack([xx, yy], dim=-1)
    base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)

    disp_norm = 2.0 * disp.squeeze(1) / max(w - 1, 1)

    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] + disp_norm

    warped = F.grid_sample(
        left,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped


# ============================================================
# Direct / fusion / final-refine branches
# ============================================================

class StrongDirectDetailBranch(nn.Module):
    """
    Strong direct color/detail branch.

    Input:
      feature + left + warped + |left - warped|

    Output:
      direct residual in [-1, 1]
    """

    def __init__(self, feat_ch: int) -> None:
        super().__init__()

        in_ch = feat_ch + 9

        self.head = ConvGNAct(in_ch, feat_ch, 3)

        self.body = nn.Sequential(
            StrongTextureBlock(feat_ch),
            StrongTextureBlock(feat_ch),
            StrongTextureBlock(feat_ch),
            ResidualDenseBlock(feat_ch, num_layers=5),
            DetailBlock(feat_ch),
            ConvGNAct(feat_ch, feat_ch, 3),
        )

        self.out = nn.Conv2d(feat_ch, 3, 3, padding=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        feat: torch.Tensor,
        left: torch.Tensor,
        warped: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(left - warped)
        x = torch.cat([feat, left, warped, diff], dim=1)

        x = self.head(x)
        x = self.body(x)

        return torch.tanh(self.out(x))


class StrongFusionMaskBranch(nn.Module):
    """
    Learned fusion mask alpha.

    alpha close to 0:
      trust warped image

    alpha close to 1:
      trust direct image

    The default bias lets the direct branch participate while preserving the
    geometric warp as the initial explanation.
    """

    def __init__(self, feat_ch: int, init_bias: float = -1.2) -> None:
        super().__init__()

        in_ch = feat_ch + 12

        self.net = nn.Sequential(
            ConvGNAct(in_ch, feat_ch, 3),
            StrongTextureBlock(feat_ch),
            ResBlock(feat_ch),
            ConvGNAct(feat_ch, feat_ch, 3),
            nn.Conv2d(feat_ch, 1, 3, padding=1),
        )

        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(
        self,
        feat: torch.Tensor,
        left: torch.Tensor,
        warped: torch.Tensor,
        direct: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(direct - warped)
        x = torch.cat([feat, left, warped, direct, diff], dim=1)
        return torch.sigmoid(self.net(x))


class FinalRefineBranch(nn.Module):
    """
    Final refinement after warp/direct fusion.

    It only predicts a small residual to polish colors and local details.
    """

    def __init__(self, feat_ch: int) -> None:
        super().__init__()

        in_ch = feat_ch + 13

        self.net = nn.Sequential(
            ConvGNAct(in_ch, feat_ch, 3),
            StrongTextureBlock(feat_ch),
            StrongTextureBlock(feat_ch),
            ConvGNAct(feat_ch, feat_ch, 3),
            nn.Conv2d(feat_ch, 3, 3, padding=1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        feat: torch.Tensor,
        left: torch.Tensor,
        warped: torch.Tensor,
        direct: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(direct - warped)
        x = torch.cat([feat, left, warped, direct, diff, alpha], dim=1)
        return torch.tanh(self.net(x))


# ============================================================
# GASG
# ============================================================

class GASG(nn.Module):
    """
    GASG.

    Output dict:
      right       - final synthesized right image in [-1, 1]
      warped      - left image warped by predicted disparity
      direct      - direct-detail corrected image
      disp        - predicted horizontal disparity in pixels
      residual    - direct residual + final residual
      mask        - fusion alpha
    """

    def __init__(
        self,
        base_ch: int = 32,
        max_disp: float = 24.0,
        residual_scale: float = 1.0,
        gamma_correction: float = 1.0,
        num_bottleneck_blocks: int = 4,
        positive_disp: bool = True,
        fusion_bias: float = -1.2,
        final_refine_scale: float = 0.25,
        **_: object,
    ) -> None:
        super().__init__()

        self.base_ch = int(base_ch)
        self.max_disp = float(max_disp)
        self.direct_scale = float(residual_scale)
        self.gamma_correction = float(gamma_correction)
        self.positive_disp = bool(positive_disp)
        self.final_refine_scale = float(final_refine_scale)

        c1 = self.base_ch
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 6
        c5 = c1 * 8

        # Full-resolution shallow feature
        self.stem = nn.Sequential(
            ConvGNAct(3, c1, 3),
            StrongTextureBlock(c1),
            StrongTextureBlock(c1),
        )

        # Deeper encoder
        self.down1 = DownBlock(c1, c2, blocks=2)  # H/2
        self.down2 = DownBlock(c2, c3, blocks=2)  # H/4
        self.down3 = DownBlock(c3, c4, blocks=2)  # H/8
        self.down4 = DownBlock(c4, c5, blocks=2)  # H/16

        # Strong bottleneck
        bottleneck = []
        for _ in range(max(int(num_bottleneck_blocks), 1)):
            bottleneck.append(StrongTextureBlock(c5))
        self.bottleneck = nn.Sequential(*bottleneck)

        # Decoder
        self.up3 = UpFuseBlock(c5, c4, c4, blocks=2)
        self.up2 = UpFuseBlock(c4, c3, c3, blocks=2)
        self.up1 = UpFuseBlock(c3, c2, c2, blocks=2)
        self.up0 = UpFuseBlock(c2, c1, c1, blocks=2)

        self.refine = nn.Sequential(
            StrongTextureBlock(c1),
            StrongTextureBlock(c1),
            ConvGNAct(c1, c1, 3),
        )

        # Disparity branch
        self.disp_head = nn.Sequential(
            StrongTextureBlock(c1),
            ConvGNAct(c1, c1, 3),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

        # Direct / fusion / final refine
        self.direct_branch = StrongDirectDetailBranch(c1)
        self.fusion_branch = StrongFusionMaskBranch(c1, init_bias=fusion_bias)
        self.final_refine_branch = FinalRefineBranch(c1)

        # Stable disparity initialization.
        nn.init.zeros_(self.disp_head[-1].weight)
        nn.init.constant_(self.disp_head[-1].bias, -2.2)

    def encode_decode(self, left: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(left)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)

        b = self.bottleneck(s4)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return self.refine(d0)

    def forward(self, left: torch.Tensor, return_dict: bool = False):
        feat = self.encode_decode(left)

        raw_disp = self.disp_head(feat)

        if self.positive_disp:
            disp = self.max_disp * torch.sigmoid(raw_disp)
        else:
            disp = self.max_disp * torch.tanh(raw_disp)

        warped = warp_left_to_right(left, disp)

        # Strong direct color/detail prediction
        direct_residual = self.direct_branch(feat, left, warped)
        direct = torch.clamp(
            warped + self.direct_scale * direct_residual,
            -1.0,
            1.0,
        )

        # Learned fusion
        alpha = self.fusion_branch(feat, left, warped, direct)
        fused = (1.0 - alpha) * warped + alpha * direct

        # Final small refinement
        final_delta = self.final_refine_branch(feat, left, warped, direct, alpha)
        right = torch.clamp(
            fused + self.final_refine_scale * self.direct_scale * final_delta,
            -1.0,
            1.0,
        )

        residual = direct_residual + self.final_refine_scale * final_delta

        if not return_dict:
            return right

        return {
            "right": right,
            "warped": warped,
            "direct": direct,
            "disp": disp,
            "residual": residual,
            "mask": alpha,
        }

    @torch.no_grad()
    def inference(
        self,
        left: torch.Tensor,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        out = self.forward(left, return_dict=False)
        img = torch.clamp((out + 1.0) * 0.5, 0.0, 1.0)

        if gamma is None:
            gamma = self.gamma_correction

        if gamma is not None and float(gamma) != 1.0:
            img = img.pow(float(gamma))

        return torch.clamp(img, 0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GASG(
        base_ch=32,
        max_disp=24.0,
        residual_scale=1.0,
        fusion_bias=-1.2,
    ).to(device)

    x = torch.randn(1, 3, 256, 512, device=device)

    with torch.no_grad():
        out = model(x, return_dict=True)

    print(f"right: {tuple(out['right'].shape)}")
    print(f"warped: {tuple(out['warped'].shape)}")
    print(f"direct: {tuple(out['direct'].shape)}")
    print(f"disp: {tuple(out['disp'].shape)}")
    print(f"mask: {tuple(out['mask'].shape)}")
    print(f"residual: {tuple(out['residual'].shape)}")
    print(f"params: {count_parameters(model) / 1e6:.3f}M")

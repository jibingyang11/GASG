"""
Geometry-constrained GASG loss with:
  - final right reconstruction
  - warped image reconstruction
  - direct-branch supervision
  - disparity smoothness
  - fusion-mask / residual regularization
  - wavelet high-frequency alignment
  - near-region high-disparity weighted losses

Designed for GASG v5:
  warped branch + direct detail/color branch + learned fusion mask.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


LossWeights = Union[Mapping[str, float], Sequence[float], None]


# ============================================================
# Basic utils
# ============================================================

def _as_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor to [0, 1] if it appears to be in [-1, 1].
    """
    if x.min() < -0.05:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _grad_x(x: torch.Tensor) -> torch.Tensor:
    return x[..., :, 1:] - x[..., :, :-1]


def _grad_y(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:, :] - x[..., :-1, :]


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_01(pred)
    target = _as_01(target)
    return F.mse_loss(pred, target)


def weighted_charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    pred, target: [B, C, H, W]
    weight:       [B, 1, H, W]
    """
    diff = torch.sqrt((pred - target) ** 2 + eps ** 2)
    return (diff * weight).mean()


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target: [B, C, H, W]
    weight:       [B, 1, H, W]
    """
    return (((pred - target) ** 2) * weight).mean()


# ============================================================
# Reconstruction / structure losses
# ============================================================

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_01(pred)
    target = _as_01(target)

    return charbonnier_loss(_grad_x(pred), _grad_x(target)) + charbonnier_loss(
        _grad_y(pred),
        _grad_y(target),
    )


def multi_scale_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_01(pred)
    target = _as_01(target)

    loss = charbonnier_loss(pred, target)

    for scale, weight in ((2, 0.5), (4, 0.25), (8, 0.125)):
        pred_s = F.avg_pool2d(pred, scale, stride=scale)
        target_s = F.avg_pool2d(target, scale, stride=scale)
        loss = loss + weight * charbonnier_loss(pred_s, target_s)

    return loss


class LowResSSIMLoss(nn.Module):
    def __init__(self, window_size: int = 7, downsample: int = 4) -> None:
        super().__init__()

        self.window_size = int(window_size)
        self.downsample = int(downsample)
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = _as_01(pred)
        target = _as_01(target)

        if self.downsample > 1:
            pred = F.avg_pool2d(
                pred,
                kernel_size=self.downsample,
                stride=self.downsample,
            )
            target = F.avg_pool2d(
                target,
                kernel_size=self.downsample,
                stride=self.downsample,
            )

        pad = self.window_size // 2

        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)

        sigma_x = F.avg_pool2d(
            pred * pred,
            self.window_size,
            stride=1,
            padding=pad,
        ) - mu_x ** 2

        sigma_y = F.avg_pool2d(
            target * target,
            self.window_size,
            stride=1,
            padding=pad,
        ) - mu_y ** 2

        sigma_xy = F.avg_pool2d(
            pred * target,
            self.window_size,
            stride=1,
            padding=pad,
        ) - mu_x * mu_y

        numerator = (2.0 * mu_x * mu_y + self.c1) * (
            2.0 * sigma_xy + self.c2
        )
        denominator = (mu_x ** 2 + mu_y ** 2 + self.c1) * (
            sigma_x + sigma_y + self.c2
        )

        ssim = numerator / (denominator + 1e-8)
        return torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0).mean()


def census_like_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Lightweight local structure loss.
    """
    pred = _as_01(pred)
    target = _as_01(target)

    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)

    pred_local = pred_gray - F.avg_pool2d(
        pred_gray,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    target_local = target_gray - F.avg_pool2d(
        target_gray,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    return charbonnier_loss(pred_local, target_local)


def low_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_01(pred)
    target = _as_01(target)

    loss = pred.new_tensor(0.0)

    for k, weight in ((8, 1.0), (16, 0.5)):
        pred_low = F.avg_pool2d(pred, k, stride=k)
        target_low = F.avg_pool2d(target, k, stride=k)
        loss = loss + weight * charbonnier_loss(pred_low, target_low)

    return loss


# ============================================================
# Haar wavelet loss
# ============================================================

class HaarDWT(nn.Module):
    """
    Differentiable Haar wavelet decomposition.

    Input:
        x: [B, C, H, W]

    Output:
        ll, lh, hl, hh: each [B, C, H/2, W/2]
    """

    def __init__(self) -> None:
        super().__init__()

        ll = torch.tensor(
            [[0.5, 0.5],
             [0.5, 0.5]],
            dtype=torch.float32,
        )
        lh = torch.tensor(
            [[-0.5, -0.5],
             [ 0.5,  0.5]],
            dtype=torch.float32,
        )
        hl = torch.tensor(
            [[-0.5,  0.5],
             [-0.5,  0.5]],
            dtype=torch.float32,
        )
        hh = torch.tensor(
            [[ 0.5, -0.5],
             [-0.5,  0.5]],
            dtype=torch.float32,
        )

        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        b, c, h, w = x.shape

        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            b, c, h, w = x.shape

        filt = self.filt.to(device=x.device, dtype=x.dtype)
        filt = filt.repeat(c, 1, 1, 1)

        y = F.conv2d(
            x,
            filt,
            stride=2,
            padding=0,
            groups=c,
        )

        y = y.view(b, c, 4, h // 2, w // 2)

        ll = y[:, :, 0]
        lh = y[:, :, 1]
        hl = y[:, :, 2]
        hh = y[:, :, 3]

        return ll, lh, hl, hh


class WaveletLoss(nn.Module):
    """
    Multi-level Haar wavelet alignment.
    """

    def __init__(self, levels: int = 2) -> None:
        super().__init__()

        self.levels = int(levels)
        self.dwt = HaarDWT()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pred = _as_01(pred)
        target = _as_01(target)

        loss_high = pred.new_tensor(0.0)
        loss_low = pred.new_tensor(0.0)

        cur_pred = pred
        cur_target = target

        for level in range(self.levels):
            ll_p, lh_p, hl_p, hh_p = self.dwt(cur_pred)
            ll_t, lh_t, hl_t, hh_t = self.dwt(cur_target)

            level_weight = 1.0 / float(level + 1)

            loss_high = loss_high + level_weight * (
                charbonnier_loss(lh_p, lh_t)
                + charbonnier_loss(hl_p, hl_t)
                + charbonnier_loss(hh_p, hh_t)
            )

            loss_low = loss_low + 0.5 * level_weight * charbonnier_loss(
                ll_p,
                ll_t,
            )

            cur_pred = ll_p
            cur_target = ll_t

        return loss_high, loss_low


# ============================================================
# Geometry / near-region / mask losses
# ============================================================

def disparity_smoothness_loss(
    disp: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    image = _as_01(image)

    if image.shape[-2:] != disp.shape[-2:]:
        image = F.interpolate(
            image,
            size=disp.shape[-2:],
            mode="area",
        )

    gray = image.mean(dim=1, keepdim=True)

    dx_disp = _grad_x(disp).abs()
    dy_disp = _grad_y(disp).abs()

    dx_img = _grad_x(gray).abs()
    dy_img = _grad_y(gray).abs()

    loss_x = dx_disp * torch.exp(-10.0 * dx_img)
    loss_y = dy_disp * torch.exp(-10.0 * dy_img)

    return loss_x.mean() + loss_y.mean()


def mask_tv_loss(mask: torch.Tensor) -> torch.Tensor:
    return _grad_x(mask).abs().mean() + _grad_y(mask).abs().mean()


def build_near_weight(
    disp: Optional[torch.Tensor],
    image_like: torch.Tensor,
    strength: float = 2.0,
) -> torch.Tensor:
    """
    Larger disparity -> larger weight.
    This focuses supervision on near / foreground objects.
    """
    if disp is None:
        return torch.ones_like(image_like[:, :1])

    if disp.shape[-2:] != image_like.shape[-2:]:
        disp = F.interpolate(
            disp,
            size=image_like.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    disp_detached = disp.detach()
    max_disp = disp_detached.amax(dim=(1, 2, 3), keepdim=True)
    disp_norm = disp_detached / (max_disp + 1e-6)

    near_weight = 1.0 + strength * disp_norm
    return near_weight


# ============================================================
# Weight normalization
# ============================================================

def normalize_weights(weights: LossWeights) -> Dict[str, float]:
    default = {
        # final right losses
        "photo": 1.0,
        "mse": 0.8,
        "warp": 0.18,
        "ms": 0.15,
        "ssim": 0.03,
        "census": 0.05,
        "edge": 0.01,
        "lowfreq": 0.04,
        "wavelet_high": 0.04,
        "wavelet_low": 0.01,
        "near_photo": 0.40,
        "near_mse": 0.40,

        # direct branch losses
        "direct_photo": 0.35,
        "direct_mse": 0.30,
        "direct_wavelet_high": 0.03,
        "direct_wavelet_low": 0.005,
        "direct_near_photo": 0.25,
        "direct_near_mse": 0.25,

        # regularization
        "smooth": 0.004,
        "flow": 0.00003,
        "mask": 0.0005,
        "mask_tv": 0.0005,
        "residual": 0.002,
    }

    if weights is None:
        return default

    if isinstance(weights, Mapping):
        out = default.copy()
        for k, v in weights.items():
            k = str(k)
            if k in out:
                out[k] = float(v)
        return out

    return default


# ============================================================
# Main GASG loss
# ============================================================

class GASGLoss(nn.Module):
    def __init__(
        self,
        use_discriminator: bool = False,
        ssim_downsample: int = 4,
        detail_weight_strength: float = 0.0,
        wavelet_levels: int = 2,
        near_weight_strength: float = 2.0,
    ) -> None:
        super().__init__()

        self.use_discriminator = False
        self.detail_weight_strength = float(detail_weight_strength)
        self.near_weight_strength = float(near_weight_strength)

        self.ssim = LowResSSIMLoss(
            window_size=7,
            downsample=ssim_downsample,
        )

        self.wavelet = WaveletLoss(
            levels=wavelet_levels,
        )

    def forward(
        self,
        model_out: Union[Dict[str, torch.Tensor], torch.Tensor],
        target: torch.Tensor,
        left_img: Optional[torch.Tensor] = None,
        weights: LossWeights = None,
        disp_preds: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        **_: object,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if isinstance(model_out, dict):
            pred = model_out["right"]
            warped = model_out.get("warped", None)
            direct = model_out.get("direct", None)
            disp = model_out.get("disp", model_out.get("disp_low", None))
            residual = model_out.get("residual", None)
            mask = model_out.get("mask", None)
        else:
            pred = model_out
            warped = None
            direct = None
            disp = None
            residual = None
            mask = None

        if disp is None and disp_preds is not None:
            disp = disp_preds[-1] if isinstance(disp_preds, (list, tuple)) else disp_preds

        w = normalize_weights(weights)

        pred_01 = _as_01(pred)
        target_01 = _as_01(target)

        losses: Dict[str, torch.Tensor] = {}

        # ====================================================
        # Final right supervision
        # ====================================================
        losses["photo"] = charbonnier_loss(pred_01, target_01)
        losses["mse"] = mse_loss(pred_01, target_01)
        losses["ms"] = multi_scale_l1(pred_01, target_01)
        losses["ssim"] = self.ssim(pred_01, target_01)
        losses["census"] = census_like_loss(pred_01, target_01)
        losses["edge"] = gradient_loss(pred_01, target_01)
        losses["lowfreq"] = low_frequency_loss(pred_01, target_01)

        wavelet_high, wavelet_low = self.wavelet(pred_01, target_01)
        losses["wavelet_high"] = wavelet_high
        losses["wavelet_low"] = wavelet_low

        near_weight = build_near_weight(
            disp=disp,
            image_like=pred_01,
            strength=self.near_weight_strength,
        )

        losses["near_photo"] = weighted_charbonnier_loss(
            pred_01,
            target_01,
            near_weight,
        )
        losses["near_mse"] = weighted_mse_loss(
            pred_01,
            target_01,
            near_weight,
        )

        # ====================================================
        # Direct branch supervision
        # ====================================================
        if direct is not None:
            direct_01 = _as_01(direct)

            losses["direct_photo"] = charbonnier_loss(direct_01, target_01)
            losses["direct_mse"] = mse_loss(direct_01, target_01)

            direct_wavelet_high, direct_wavelet_low = self.wavelet(
                direct_01,
                target_01,
            )
            losses["direct_wavelet_high"] = direct_wavelet_high
            losses["direct_wavelet_low"] = direct_wavelet_low

            losses["direct_near_photo"] = weighted_charbonnier_loss(
                direct_01,
                target_01,
                near_weight,
            )
            losses["direct_near_mse"] = weighted_mse_loss(
                direct_01,
                target_01,
                near_weight,
            )
        else:
            losses["direct_photo"] = pred.new_tensor(0.0)
            losses["direct_mse"] = pred.new_tensor(0.0)
            losses["direct_wavelet_high"] = pred.new_tensor(0.0)
            losses["direct_wavelet_low"] = pred.new_tensor(0.0)
            losses["direct_near_photo"] = pred.new_tensor(0.0)
            losses["direct_near_mse"] = pred.new_tensor(0.0)

        # ====================================================
        # Warp reconstruction
        # ====================================================
        if warped is not None:
            losses["warp"] = charbonnier_loss(_as_01(warped), target_01)
        else:
            losses["warp"] = pred.new_tensor(0.0)

        # ====================================================
        # Disparity regularization
        # ====================================================
        if disp is not None and left_img is not None:
            losses["smooth"] = disparity_smoothness_loss(disp, left_img)
            losses["flow"] = disp.abs().mean()
        else:
            losses["smooth"] = pred.new_tensor(0.0)
            losses["flow"] = pred.new_tensor(0.0)

        # ====================================================
        # Fusion mask regularization
        # In v5, mask is alpha for direct branch.
        # ====================================================
        if mask is not None:
            losses["mask"] = mask.mean()
            losses["mask_tv"] = mask_tv_loss(mask)
        else:
            losses["mask"] = pred.new_tensor(0.0)
            losses["mask_tv"] = pred.new_tensor(0.0)

        # ====================================================
        # Residual regularization
        # ====================================================
        if residual is not None:
            losses["residual"] = residual.abs().mean()
        else:
            losses["residual"] = pred.new_tensor(0.0)

        total = (
            # final right
            w["photo"] * losses["photo"]
            + w["mse"] * losses["mse"]
            + w["warp"] * losses["warp"]
            + w["ms"] * losses["ms"]
            + w["ssim"] * losses["ssim"]
            + w["census"] * losses["census"]
            + w["edge"] * losses["edge"]
            + w["lowfreq"] * losses["lowfreq"]
            + w["wavelet_high"] * losses["wavelet_high"]
            + w["wavelet_low"] * losses["wavelet_low"]
            + w["near_photo"] * losses["near_photo"]
            + w["near_mse"] * losses["near_mse"]

            # direct branch
            + w["direct_photo"] * losses["direct_photo"]
            + w["direct_mse"] * losses["direct_mse"]
            + w["direct_wavelet_high"] * losses["direct_wavelet_high"]
            + w["direct_wavelet_low"] * losses["direct_wavelet_low"]
            + w["direct_near_photo"] * losses["direct_near_photo"]
            + w["direct_near_mse"] * losses["direct_near_mse"]

            # regularization
            + w["smooth"] * losses["smooth"]
            + w["flow"] * losses["flow"]
            + w["mask"] * losses["mask"]
            + w["mask_tv"] * losses["mask_tv"]
            + w["residual"] * losses["residual"]
        )

        losses["total"] = total

        return total, losses
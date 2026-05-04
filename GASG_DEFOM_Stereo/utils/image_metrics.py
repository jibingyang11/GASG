"""
Image quality metrics for right-view synthesis evaluation.
  - PSNR
  - SSIM
  - LPIPS (using lpips library v0.1.4, AlexNet backbone)
"""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips==0.1.4")

import torch


class ImageMetrics:
    """Compute PSNR, SSIM, LPIPS for image pairs."""

    def __init__(self, device='cpu'):
        self.device = device
        if _LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

    def compute_psnr(self, pred, target):
        """
        Args:
            pred, target: (H, W, 3) numpy arrays in [0, 1]
        """
        return peak_signal_noise_ratio(target, pred, data_range=1.0)

    def compute_ssim(self, pred, target):
        """
        Args:
            pred, target: (H, W, 3) numpy arrays in [0, 1]
        """
        return structural_similarity(target, pred, multichannel=True,
                                     channel_axis=2, data_range=1.0)

    def compute_lpips(self, pred, target):
        """
        Args:
            pred, target: (H, W, 3) numpy arrays in [0, 1]
        Returns:
            LPIPS distance (lower is better)
        """
        if self.lpips_fn is None:
            return 0.0

        # Convert to (1, 3, H, W) tensor in [-1, 1]
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
        pred_t = pred_t * 2.0 - 1.0
        target_t = target_t * 2.0 - 1.0

        with torch.no_grad():
            dist = self.lpips_fn(pred_t.to(self.device),
                                 target_t.to(self.device))
        return dist.item()

    def compute_all(self, pred, target):
        """Compute all three metrics."""
        return {
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target),
            'lpips': self.compute_lpips(pred, target),
        }


def aggregate_image_metrics(metrics_list):
    """Average a list of per-image metric dicts."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

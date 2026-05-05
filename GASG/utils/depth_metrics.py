"""
Standard depth evaluation metrics following Eigen et al. (NeurIPS 2014).

Metrics:
  - Abs Rel:  mean(|pred - gt| / gt)
  - Sq Rel:   mean((pred - gt)^2 / gt)
  - RMSE:     sqrt(mean((pred - gt)^2))
  - RMSE_log: sqrt(mean((log(pred) - log(gt))^2))
  - delta_k:  % of pixels where max(pred/gt, gt/pred) < 1.25^k
"""

import numpy as np


def compute_depth_errors(gt, pred, min_depth=0.001, max_depth=80.0):
    """
    Compute standard depth metrics.

    Args:
        gt:   (H, W) ground-truth depth in meters
        pred: (H, W) predicted depth in meters
        min_depth: minimum valid depth
        max_depth: maximum valid depth

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
    """
    mask = (gt > min_depth) & (gt < max_depth)
    gt = gt[mask]
    pred = pred[mask]

    if len(gt) == 0:
        return {k: 0.0 for k in
                ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']}

    # Clamp predictions
    pred = np.clip(pred, min_depth, max_depth)

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean((gt - pred) ** 2 / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'd1': d1,
        'd2': d2,
        'd3': d3,
    }


def median_scaling(pred, gt, min_depth=0.001, max_depth=80.0):
    """
    Apply median scaling to align predicted (relative) depth to GT.
    Used for affine-invariant monocular models.

    Args:
        pred: (H, W) predicted depth (relative)
        gt:   (H, W) ground-truth depth (metric)

    Returns:
        Scaled prediction
    """
    mask = (gt > min_depth) & (gt < max_depth)
    if mask.sum() == 0:
        return pred
    scale = np.median(gt[mask]) / np.median(pred[mask])
    return pred * scale


def aggregate_metrics(metrics_list):
    """Average a list of per-image metric dicts."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def format_metrics(metrics, method_name=""):
    """Pretty-print metrics."""
    header = f"{'Method':<30} {'AbsRel':>8} {'SqRel':>8} " \
             f"{'RMSE':>8} {'RMSElog':>8} " \
             f"{'d1':>6} {'d2':>6} {'d3':>6}"
    row = f"{method_name:<30} " \
          f"{metrics['abs_rel']:8.4f} {metrics['sq_rel']:8.4f} " \
          f"{metrics['rmse']:8.4f} {metrics['rmse_log']:8.4f} " \
          f"{metrics['d1']:6.4f} {metrics['d2']:6.4f} {metrics['d3']:6.4f}"
    return header + "\n" + row

#!/usr/bin/env python3
"""Pure numerical helpers for Reviewer 1 Comment 3 condition ablation.

These functions deliberately avoid importing PyTorch/NPU modules so that the
statistics and mask logic can be unit-tested independently of the accelerator
runtime used by the paper model.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

CONDITION_NAMES = ("Bx", "By", "Bz", "Vsw", "Pdyn")
HIGHER_IS_BETTER = {"ssim", "psnr", "r2"}
LOWER_IS_BETTER = {"rmse", "mae"}


def create_paper_mask(image_shape, mlat_range=(60.0, 80.0)) -> np.ndarray:
    """Return the controlled mask used in the manuscript OVATION evaluation.

    The 80x96 grid spans 50--90 deg MLAT and 0--24 h MLT. Pixels in the
    requested MLAT band are hidden over 18--24 and 0--6 MLT. The DDPM code
    uses 1 for observed/preserved pixels and 0 for the artificially missing
    region.
    """
    h, w = map(int, image_shape)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    mlat_min, mlat_max = map(float, mlat_range)
    if not (50.0 <= mlat_min < mlat_max <= 90.0):
        raise ValueError("mlat_range must lie inside [50, 90] with min < max")

    row_min = int((90.0 - mlat_max) * h / 40.0)
    row_max = int((90.0 - mlat_min) * h / 40.0)
    col_18 = int(18.0 * w / 24.0)
    col_24 = w
    col_00 = 0
    col_06 = int(6.0 * w / 24.0)

    mask = np.ones((h, w), dtype=np.float32)
    mask[row_min:row_max, col_18:col_24] = 0.0
    mask[row_min:row_max, col_00:col_06] = 0.0
    return mask


def make_derangement(n: int, seed: int = 2026) -> np.ndarray:
    """Return a deterministic permutation with no fixed points.

    A Sattolo shuffle creates one cycle, so every selected test case receives
    a condition from a different timestamp. This avoids diluting permutation
    importance with accidental unchanged samples.
    """
    n = int(n)
    if n < 2:
        raise ValueError("A derangement requires at least two samples")
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=int)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    if np.any(perm == np.arange(n)):
        raise RuntimeError("Internal error: Sattolo shuffle produced a fixed point")
    return perm


def apply_condition_variant(
    conditions: np.ndarray,
    variant: str,
    permutation: np.ndarray,
) -> np.ndarray:
    """Apply a paired permutation ablation without changing marginal values."""
    x = np.asarray(conditions)
    perm = np.asarray(permutation, dtype=int)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError(f"conditions must have shape (N, 5); got {x.shape}")
    if perm.shape != (x.shape[0],):
        raise ValueError(f"permutation must have shape ({x.shape[0]},); got {perm.shape}")
    if set(perm.tolist()) != set(range(x.shape[0])):
        raise ValueError("permutation must contain every sample index exactly once")

    out = x.copy()
    if variant == "full":
        return out
    if variant == "permute_all":
        return x[perm].copy()
    if variant.startswith("permute_"):
        name = variant[len("permute_") :]
        if name not in CONDITION_NAMES:
            raise ValueError(f"Unknown condition variable in variant {variant!r}")
        col = CONDITION_NAMES.index(name)
        out[:, col] = x[perm, col]
        return out
    raise ValueError(f"Unknown condition variant: {variant}")


def compute_masked_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compute manuscript metrics only on artificially hidden pixels (mask=0)."""
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.asarray(mask)
    if truth.shape != pred.shape or truth.shape != mask.shape:
        raise ValueError("truth, pred, and mask must have the same shape")

    region = (mask == 0) & np.isfinite(truth) & np.isfinite(pred)
    y = truth[region]
    p = pred[region]
    if y.size == 0:
        raise ValueError("No finite pixels in the masked evaluation region")

    err = p - y
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    if mse == 0.0:
        psnr = float("inf")
    else:
        max_pixel = float(np.max(truth))
        psnr = float(20.0 * np.log10(max(max_pixel, 1e-12) / np.sqrt(mse)))

    mu_y = float(np.mean(y))
    mu_p = float(np.mean(p))
    var_y = float(np.var(y))
    var_p = float(np.var(p))
    if y.size > 1:
        cov = float(np.cov(y, p)[0, 1])
    else:
        cov = 0.0
    dynamic_max = max(float(np.max(truth)), float(np.max(pred)), 1e-12)
    c1 = (0.01 * dynamic_max) ** 2
    c2 = (0.03 * dynamic_max) ** 2
    numerator = (2.0 * mu_y * mu_p + c1) * (2.0 * cov + c2)
    denominator = (mu_y**2 + mu_p**2 + c1) * (var_y + var_p + c2)
    ssim = float(numerator / denominator) if denominator != 0 else float("nan")

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - mu_y) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if np.isfinite(r2):
        r2 = max(r2, -1.0)

    return {
        "ssim": ssim,
        "psnr": psnr,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "n_pixels": int(y.size),
    }


def metric_degradation(
    full_metrics: Mapping[str, float],
    ablated_metrics: Mapping[str, float],
) -> Dict[str, float]:
    """Return degradation with a common sign convention: positive means worse."""
    out: Dict[str, float] = {}
    for metric in ("ssim", "psnr", "rmse", "r2", "mae"):
        f = float(full_metrics[metric])
        a = float(ablated_metrics[metric])
        if metric in HIGHER_IS_BETTER:
            d = f - a
        else:
            d = a - f
        out[metric] = float(np.round(d, 12))
    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval for the mean degradation."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    draw_idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    boot_means = np.mean(x[draw_idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(x.size),
    }


def paired_wilcoxon_pvalue(values: np.ndarray) -> float:
    """Two-sided paired Wilcoxon test of degradation against zero."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(result.pvalue)
    except Exception:
        return float("nan")

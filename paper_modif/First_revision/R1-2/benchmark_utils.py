from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

METRICS = ("ssim", "psnr", "rmse", "r2", "mae")
HIGHER_IS_BETTER = {"ssim", "psnr", "r2"}
LOWER_IS_BETTER = {"rmse", "mae"}


def create_paper_mask(image_shape, mlat_range=(60.0, 80.0)) -> np.ndarray:
    """Controlled manuscript mask: hide 18--06 MLT inside 60--80 MLAT."""
    h, w = map(int, image_shape)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")
    mlat_min, mlat_max = map(float, mlat_range)
    if not (50.0 <= mlat_min < mlat_max <= 90.0):
        raise ValueError("mlat_range must lie inside [50, 90] with min < max")
    row_min = int((90.0 - mlat_max) * h / 40.0)
    row_max = int((90.0 - mlat_min) * h / 40.0)
    col_18 = int(18.0 * w / 24.0)
    col_06 = int(6.0 * w / 24.0)
    mask = np.ones((h, w), dtype=np.float32)
    mask[row_min:row_max, col_18:w] = 0.0
    mask[row_min:row_max, 0:col_06] = 0.0
    return mask


def compute_masked_metrics(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute manuscript metrics only over the artificially hidden region."""
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
    cov = float(np.cov(y, p)[0, 1]) if y.size > 1 else 0.0
    dynamic_max = max(float(np.max(truth)), float(np.max(pred)), 1e-12)
    c1 = (0.01 * dynamic_max) ** 2
    c2 = (0.03 * dynamic_max) ** 2
    numerator = (2.0 * mu_y * mu_p + c1) * (2.0 * cov + c2)
    denominator = (mu_y**2 + mu_p**2 + c1) * (var_y + var_p + c2)
    ssim = float(numerator / denominator) if denominator != 0.0 else float("nan")

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - mu_y) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
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


def interpolate_inpainting(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation baseline matching the manuscript implementation."""
    from scipy.interpolate import griddata

    corrupted = np.asarray(corrupted, dtype=np.float64)
    mask = np.asarray(mask)
    if corrupted.shape != mask.shape:
        raise ValueError("corrupted and mask must have the same shape")
    known = np.column_stack(np.where(mask == 1))
    unknown = np.column_stack(np.where(mask == 0))
    if len(known) == 0 or len(unknown) == 0:
        return corrupted.copy().astype(np.float32)

    values = corrupted[mask == 1]
    interp = griddata(known, values, unknown, method="linear", fill_value=0.0)
    out = corrupted.copy()
    out[mask == 0] = interp
    return out.astype(np.float32)


def metric_improvement(conditional_value: float, baseline_value: float, metric: str) -> float:
    """Return paired improvement with a common sign: positive means conditional is better."""
    metric = str(metric)
    if metric in HIGHER_IS_BETTER:
        return float(conditional_value - baseline_value)
    if metric in LOWER_IS_BETTER:
        return float(baseline_value - conditional_value)
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> Dict[str, float]:
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


def summarize_methods(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_value,
    bootstrap_n: int = 10000,
    seed: int = 2026,
) -> List[Dict[str, float]]:
    sub = df.loc[df[group_col] == group_value].copy()
    rows: List[Dict[str, float]] = []
    for mi, method in enumerate(sorted(sub["method"].unique())):
        part = sub.loc[sub["method"] == method]
        record: Dict[str, float] = {
            "group": group_value,
            "method": method,
            "n": int(part["sample_id"].nunique()),
        }
        for qi, metric in enumerate(METRICS):
            vals = pd.to_numeric(part[metric], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            ci = bootstrap_mean_ci(vals, n_boot=bootstrap_n, seed=seed + 100 * mi + qi)
            record[f"mean_{metric}"] = ci["mean"]
            record[f"std_{metric}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            record[f"ci_low_{metric}"] = ci["ci_low"]
            record[f"ci_high_{metric}"] = ci["ci_high"]
        rows.append(record)
    return rows


def paired_comparison(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_value,
    baseline: str,
    bootstrap_n: int = 10000,
    seed: int = 2026,
) -> Dict[str, float]:
    sub = df.loc[df[group_col] == group_value].copy()
    cond = sub.loc[sub["method"] == "conditional"].set_index("sample_id")
    base = sub.loc[sub["method"] == baseline].set_index("sample_id")
    shared = sorted(set(cond.index) & set(base.index))
    record: Dict[str, float] = {
        "group": group_value,
        "baseline": baseline,
        "n": len(shared),
    }
    for qi, metric in enumerate(METRICS):
        improvements = np.asarray(
            [
                metric_improvement(
                    float(cond.loc[sid, metric]),
                    float(base.loc[sid, metric]),
                    metric,
                )
                for sid in shared
            ],
            dtype=float,
        )
        ci = bootstrap_mean_ci(improvements, n_boot=bootstrap_n, seed=seed + qi)
        record[f"mean_improvement_{metric}"] = ci["mean"]
        record[f"ci_low_improvement_{metric}"] = ci["ci_low"]
        record[f"ci_high_improvement_{metric}"] = ci["ci_high"]
        record[f"p_{metric}"] = paired_wilcoxon_pvalue(improvements)
    return record

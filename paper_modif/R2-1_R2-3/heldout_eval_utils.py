from __future__ import annotations
import numpy as np

def periodic_mlt_window_mask(mlt_1d, center_h: float, width_h: float):
    mlt = np.asarray(mlt_1d, dtype=float)
    delta = ((mlt - float(center_h) + 12.0) % 24.0) - 12.0
    half = float(width_h) / 2.0
    return (delta >= -half) & (delta < half)

def _pearson(y_true, y_pred):
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def _gradient_values(arr, mask):
    arr = np.asarray(arr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    vals = []
    pair_h = mask[:, :-1] & mask[:, 1:]
    if np.any(pair_h):
        vals.append(np.abs(arr[:, 1:] - arr[:, :-1])[pair_h])
    pair_v = mask[:-1, :] & mask[1:, :]
    if np.any(pair_v):
        vals.append(np.abs(arr[1:, :] - arr[:-1, :])[pair_v])
    if not vals:
        return np.empty(0, dtype=float)
    return np.concatenate(vals)

def compute_metrics(truth, pred, holdout_mask):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.asarray(holdout_mask, dtype=bool) & np.isfinite(truth) & np.isfinite(pred)
    y = truth[mask]
    p = pred[mask]
    if y.size == 0:
        raise ValueError("holdout_mask contains no finite truth/prediction pixels")
    err = p - y
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    peak_true = float(np.max(y))
    peak_pred = float(np.max(p))
    peak_ratio = float(peak_pred / peak_true) if peak_true > 0 else float("nan")
    g_true = _gradient_values(truth, mask)
    g_pred = _gradient_values(pred, mask)
    if g_true.size and np.mean(g_true) > 0:
        gradient_ratio = float(np.mean(g_pred) / np.mean(g_true))
        grad_true_mean = float(np.mean(g_true))
        grad_pred_mean = float(np.mean(g_pred))
    else:
        gradient_ratio = float("nan")
        grad_true_mean = float(np.mean(g_true)) if g_true.size else float("nan")
        grad_pred_mean = float(np.mean(g_pred)) if g_pred.size else float("nan")
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - p) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n_pixels": int(y.size),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": _pearson(y, p),
        "r2": r2,
        "truth_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "truth_peak": peak_true,
        "pred_peak": peak_pred,
        "peak_ratio": peak_ratio,
        "truth_gradient_mean": grad_true_mean,
        "pred_gradient_mean": grad_pred_mean,
        "gradient_ratio": gradient_ratio,
    }

def select_holdout_windows(
    flux, obs_mask, mlat_1d, mlt_1d, *,
    width_h: float,
    n_windows: int = 2,
    band=(60.0, 80.0),
    min_coverage: float = 0.5,
    min_signal_fraction: float = 0.1,
    signal_threshold: float = 0.1,
    candidate_step_h: float = 0.5,
):
    flux = np.asarray(flux, dtype=float)
    obs = np.asarray(obs_mask, dtype=bool)
    mlat = np.asarray(mlat_1d, dtype=float)
    mlt = np.asarray(mlt_1d, dtype=float)
    band_rows = (mlat >= float(band[0])) & (mlat <= float(band[1]))
    centers = np.arange(0.0, 24.0, float(candidate_step_h))
    candidates = []
    for center in centers:
        cols = periodic_mlt_window_mask(mlt, center, width_h)
        region = np.outer(band_rows, cols)
        total = int(region.sum())
        if total == 0:
            continue
        holdout = region & obs & np.isfinite(flux)
        n = int(holdout.sum())
        coverage = n / total
        if n == 0 or coverage < min_coverage:
            continue
        vals = flux[holdout]
        signal_fraction = float(np.mean(vals >= signal_threshold))
        if signal_fraction < min_signal_fraction:
            continue
        score = float(np.sum(np.log1p(np.clip(vals, 0.0, None))))
        candidates.append({
            "center_h": float(center),
            "width_h": float(width_h),
            "coverage": float(coverage),
            "signal_fraction": signal_fraction,
            "n_pixels": n,
            "score": score,
            "mask": holdout,
        })
    candidates.sort(key=lambda x: (-x["score"], x["center_h"]))
    selected = []
    def circdist(a,b):
        return abs(((a - b + 12.0) % 24.0) - 12.0)
    for cand in candidates:
        if all(circdist(cand["center_h"], s["center_h"]) >= width_h for s in selected):
            selected.append(cand)
            if len(selected) >= n_windows:
                break
    return selected

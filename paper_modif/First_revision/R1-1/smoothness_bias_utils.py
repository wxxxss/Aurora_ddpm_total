"""Pure numerical utilities for Reviewer-1 Comment-1 smoothness diagnostics."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import periodogram


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
    }


def supported_edge_gradients(
    flux: np.ndarray,
    support: np.ndarray,
    mlat_1d: np.ndarray,
    band_min: float = 60.0,
    band_max: float = 80.0,
) -> np.ndarray:
    """Return absolute nearest-neighbor flux differences on supported edges.

    Only edges whose two endpoints are finite, swath-supported, and inside the
    requested MLAT band are used. MLT is treated conservatively without adding
    an explicit 23.75--0 h wrap edge; circularity is handled separately for the
    PSD segment extraction.
    """
    f = np.asarray(flux, dtype=float)
    s = np.asarray(support, dtype=bool)
    lat = np.asarray(mlat_1d, dtype=float)
    if f.ndim != 2 or s.shape != f.shape:
        raise ValueError("flux and support must be 2-D arrays with identical shape")
    if lat.ndim != 1 or lat.size != f.shape[0]:
        raise ValueError("mlat_1d length must match the first flux dimension")

    in_band = (lat >= band_min) & (lat <= band_max)
    finite = np.isfinite(f)
    valid = s & finite & in_band[:, None]

    h_ok = valid[:, :-1] & valid[:, 1:]
    h_grad = np.abs(f[:, 1:] - f[:, :-1])[h_ok]

    v_ok = valid[:-1, :] & valid[1:, :]
    v_grad = np.abs(f[1:, :] - f[:-1, :])[v_ok]

    if h_grad.size == 0 and v_grad.size == 0:
        return np.empty(0, dtype=float)
    return np.concatenate([h_grad, v_grad]).astype(float)


def contiguous_true_segments_circular(mask: np.ndarray) -> List[np.ndarray]:
    """Return circular contiguous True runs as ordered index arrays."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = m.size
    if n == 0 or not np.any(m):
        return []
    if np.all(m):
        return [np.arange(n, dtype=int)]

    false_idx = np.flatnonzero(~m)
    start = int((false_idx[0] + 1) % n)
    order = (start + np.arange(n)) % n
    rotated = m[order]

    segments: List[np.ndarray] = []
    run_start = None
    for j, flag in enumerate(rotated):
        if flag and run_start is None:
            run_start = j
        if run_start is not None and (not flag or j == n - 1):
            end = j if not flag else j + 1
            segments.append(order[run_start:end].astype(int))
            run_start = None
    return segments


def segment_periodogram_metrics(
    values: np.ndarray,
    sample_spacing_hours: float = 0.25,
    high_freq_cutoff: float = 0.5,
    slope_min_freq: float = 0.25,
    slope_max_freq: float = 1.5,
) -> Dict[str, float | np.ndarray]:
    """Compute shape-focused 1-D PSD metrics for one continuous MLT segment.

    The profile is detrended by scipy.signal.periodogram and standardized to
    unit variance before PSD calculation. Consequently, the high-frequency
    fraction compares morphology rather than absolute auroral intensity.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("PSD segment must contain at least 8 finite samples")
    if sample_spacing_hours <= 0:
        raise ValueError("sample_spacing_hours must be positive")

    std = float(np.std(x))
    if std <= 0 or not np.isfinite(std):
        return {
            "high_frequency_fraction": 0.0,
            "spectral_slope": float("nan"),
            "frequency": np.fft.rfftfreq(x.size, d=sample_spacing_hours)[1:],
            "psd": np.zeros(x.size // 2, dtype=float),
        }

    z = (x - np.mean(x)) / std
    fs = 1.0 / sample_spacing_hours
    freq, psd = periodogram(z, fs=fs, window="hann", detrend="linear", scaling="density")
    keep = freq > 0
    freq = freq[keep]
    psd = psd[keep]

    total = float(np.sum(psd))
    hf = float(np.sum(psd[freq >= high_freq_cutoff]))
    high_fraction = hf / total if total > 0 else 0.0

    slope_mask = (
        (freq >= slope_min_freq)
        & (freq <= slope_max_freq)
        & np.isfinite(psd)
        & (psd > 0)
    )
    if np.count_nonzero(slope_mask) >= 3:
        slope = float(
            np.polyfit(np.log10(freq[slope_mask]), np.log10(psd[slope_mask]), 1)[0]
        )
    else:
        slope = float("nan")

    return {
        "high_frequency_fraction": float(high_fraction),
        "spectral_slope": slope,
        "frequency": freq.astype(float),
        "psd": psd.astype(float),
    }


def collect_matched_psd_segments(
    ssusi_flux: np.ndarray,
    ovation_flux: np.ndarray,
    support: np.ndarray,
    mlat_1d: np.ndarray,
    mlt_1d: np.ndarray,
    band_min: float = 60.0,
    band_max: float = 80.0,
    min_segment_bins: int = 16,
    signal_threshold: float = 0.1,
    min_signal_fraction: float = 0.25,
    high_freq_cutoff: float = 0.5,
) -> List[Dict[str, object]]:
    """Collect paired PSD diagnostics from identical SSUSI-supported MLT runs.

    Each returned segment uses exactly the same grid cells in SSUSI and
    OVATION. The segment must be fully swath-supported, lie within the MLAT
    band, meet a minimum length, and contain enough SSUSI auroral signal.
    """
    a = np.asarray(ssusi_flux, dtype=float)
    b = np.asarray(ovation_flux, dtype=float)
    s = np.asarray(support, dtype=bool)
    lat = np.asarray(mlat_1d, dtype=float)
    mlt = np.asarray(mlt_1d, dtype=float)
    if a.shape != b.shape or a.shape != s.shape or a.ndim != 2:
        raise ValueError("ssusi_flux, ovation_flux, and support must share one 2-D shape")
    if lat.size != a.shape[0] or mlt.size != a.shape[1]:
        raise ValueError("coordinate lengths must match flux dimensions")
    if mlt.size < 2:
        raise ValueError("mlt_1d must contain at least two points")

    spacing = float(np.median(np.diff(mlt)))
    rows: List[Dict[str, object]] = []
    for r, lat_value in enumerate(lat):
        if not (band_min <= lat_value <= band_max):
            continue
        valid = s[r] & np.isfinite(a[r]) & np.isfinite(b[r])
        for indices in contiguous_true_segments_circular(valid):
            if indices.size < min_segment_bins:
                continue
            aa = a[r, indices]
            bb = b[r, indices]
            signal_fraction = float(np.mean(aa >= signal_threshold))
            if signal_fraction < min_signal_fraction:
                continue
            ma = segment_periodogram_metrics(
                aa,
                sample_spacing_hours=spacing,
                high_freq_cutoff=high_freq_cutoff,
            )
            mb = segment_periodogram_metrics(
                bb,
                sample_spacing_hours=spacing,
                high_freq_cutoff=high_freq_cutoff,
            )
            rows.append({
                "row_index": int(r),
                "mlat": float(lat_value),
                "n_bins": int(indices.size),
                "start_mlt": float(mlt[int(indices[0])]),
                "end_mlt": float(mlt[int(indices[-1])]),
                "signal_fraction": signal_fraction,
                "indices": indices,
                "ssusi_high_frequency_fraction": float(ma["high_frequency_fraction"]),
                "ovation_high_frequency_fraction": float(mb["high_frequency_fraction"]),
                "ssusi_spectral_slope": float(ma["spectral_slope"]),
                "ovation_spectral_slope": float(mb["spectral_slope"]),
                "ssusi_frequency": ma["frequency"],
                "ssusi_psd": ma["psd"],
                "ovation_frequency": mb["frequency"],
                "ovation_psd": mb["psd"],
            })
    return rows


def paired_bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Bootstrap the paired mean difference a-b."""
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[valid]
    bb = bb[valid]
    if aa.size == 0:
        return {
            "n": 0,
            "mean_difference": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    d = aa - bb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    boot = np.mean(d[idx], axis=1)
    return {
        "n": int(d.size),
        "mean_difference": float(np.mean(d)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }

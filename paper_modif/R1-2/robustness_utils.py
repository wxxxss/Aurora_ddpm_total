from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

SOLAR_FIELDS = ("Bx", "By", "Bz", "V", "P")

PHYSICAL_RANGES = {
    "Bx": (-100.0, 100.0),
    "By": (-100.0, 100.0),
    "Bz": (-100.0, 100.0),
    "V": (150.0, 2500.0),
    "P": (0.0, 100.0),
    "Kp": (0.0, 9.0),
    "AE": (-2000.0, 5000.0),
    "SYM_H": (-1000.0, 500.0),
}


def normalize_kp(values: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, 1.0
    q99 = float(np.nanquantile(np.abs(finite), 0.99))
    divisor = 10.0 if 9.0 < q99 <= 90.0 else 1.0
    arr /= divisor
    arr[(arr < 0.0) | (arr > 9.0)] = np.nan
    return arr, divisor


def apply_physical_range(series: pd.Series, field: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    if field not in PHYSICAL_RANGES:
        return out
    lo, hi = PHYSICAL_RANGES[field]
    return out.where((out >= lo) & (out <= hi), np.nan)


def regularize_hourly(
    df: pd.DataFrame,
    year: int,
    continuous_fields: Sequence[str] = SOLAR_FIELDS + ("AE", "SYM_H"),
    max_interp_hours: int = 3,
) -> pd.DataFrame:
    if "utc" not in df.columns:
        raise KeyError("DataFrame must contain a 'utc' column")
    work = df.copy()
    work["utc"] = pd.to_datetime(work["utc"], utc=False)
    work = work.sort_values("utc").drop_duplicates("utc", keep="first").set_index("utc")
    start = pd.Timestamp(f"{year:04d}-01-01 00:00:00")
    end = pd.Timestamp(f"{year + 1:04d}-01-01 00:00:00") - pd.Timedelta(hours=1)
    hourly = pd.date_range(start, end, freq="1h")
    work.index = work.index.round("1h")
    work = work[~work.index.duplicated(keep="first")].reindex(hourly)
    for field in continuous_fields:
        if field in work.columns:
            work[field] = work[field].interpolate(
                method="time", limit=max_interp_hours, limit_area="inside"
            )
    if "Kp" in work.columns:
        work["Kp"] = work["Kp"].ffill(limit=2).bfill(limit=2)
    work.index.name = "utc"
    return work.reset_index()


def valid_condition_mask(df: pd.DataFrame, require_kp: bool = True) -> np.ndarray:
    fields = list(SOLAR_FIELDS) + (["Kp"] if require_kp else [])
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    return np.all(np.isfinite(df[fields].to_numpy(dtype=float)), axis=1)


def select_evenly_spaced_valid(df: pd.DataFrame, n_samples: int, valid_mask: np.ndarray) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(valid_mask) != len(df):
        raise ValueError("valid_mask length mismatch")
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < n_samples:
        raise ValueError(f"Only {valid_idx.size} valid rows for {n_samples} requested samples")
    targets = (np.arange(n_samples, dtype=float) + 0.5) * len(df) / n_samples - 0.5
    chosen, used = [], set()
    for target in targets:
        order = valid_idx[np.argsort(np.abs(valid_idx - target))]
        pick = next((int(i) for i in order if int(i) not in used), None)
        if pick is None:
            raise RuntimeError("Could not select unique valid sample")
        chosen.append(pick)
        used.add(pick)
    return np.asarray(sorted(chosen), dtype=int)


def select_activity_samples(frames: Sequence[pd.DataFrame], n_per_group: int = 48, seed: int = 2026) -> pd.DataFrame:
    pool = []
    for df in frames:
        if "year" not in df.columns:
            raise KeyError("Each frame must include a 'year' column")
        mask = valid_condition_mask(df, require_kp=True)
        cols = ["utc", "year", "Bx", "By", "Bz", "V", "P", "Kp"]
        pool.append(df.loc[mask, cols].copy())
    all_df = pd.concat(pool, ignore_index=True)
    rng = np.random.default_rng(seed)
    outputs = []
    for label, cond in (("Kp<=3", all_df["Kp"] <= 3.0), ("Kp>=4", all_df["Kp"] >= 4.0)):
        cand = all_df.loc[cond].copy()
        if cand.empty:
            raise ValueError(f"No samples available for {label}")
        years = sorted(cand["year"].unique())
        quota = {int(y): n_per_group // len(years) for y in years}
        for y in years[: n_per_group % len(years)]:
            quota[int(y)] += 1
        picks = []
        for y in years:
            yr = cand[cand["year"] == y]
            k = min(quota[int(y)], len(yr))
            if k:
                picks.extend(rng.choice(yr.index.to_numpy(), size=k, replace=False).tolist())
        if len(picks) < n_per_group:
            remaining = cand.drop(index=picks, errors="ignore")
            k = min(n_per_group - len(picks), len(remaining))
            if k:
                picks.extend(rng.choice(remaining.index.to_numpy(), size=k, replace=False).tolist())
        selected = cand.loc[picks].copy()
        selected["activity_group"] = label
        outputs.append(selected)
    return pd.concat(outputs, ignore_index=True).sort_values(["activity_group", "utc"])


def newell_coupling(by: np.ndarray, bz: np.ndarray, v: np.ndarray) -> np.ndarray:
    by = np.asarray(by, dtype=float)
    bz = np.asarray(bz, dtype=float)
    v = np.asarray(v, dtype=float)
    bt = np.sqrt(by**2 + bz**2)
    theta = np.arctan2(by, np.where(bz == 0.0, 1e-3, bz))
    neg = bt * np.cos(theta) * bz < 0
    theta = np.where(neg, theta + np.pi, theta)
    sintc = np.abs(np.sin(theta / 2.0))
    return (v**1.33333) * (sintc**2.66667) * (bt**0.66667)


def weighted_newell_coupling(df: pd.DataFrame, idx: int, hours: int = 4, decay: float = 0.65) -> float:
    lo = max(0, idx - hours + 1)
    sub = df.iloc[lo: idx + 1]
    if len(sub) == 0:
        return float("nan")
    ec = newell_coupling(sub["By"].to_numpy(float), sub["Bz"].to_numpy(float), sub["V"].to_numpy(float))
    age = np.arange(len(sub) - 1, -1, -1, dtype=float)
    w = decay ** age
    ok = np.isfinite(ec)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(ec[ok] * w[ok]) / np.sum(w[ok]))

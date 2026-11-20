#!/usr/bin/env python3
"""Generate OVATION Prime test maps for R1.2 multi-year robustness analysis.

Uses the repository's auroramaps implementation directly. The default sample
pool contains 48 evenly distributed samples from each of 2001 (solar maximum),
2005 (declining phase), and 2009 (solar minimum), plus balanced pooled samples
for Kp<=3 and Kp>=4. Duplicate timestamps are generated only once.

With --coupling-mode auto, the script first compares instant and 4-h weighted
Newell coupling against the existing January-2005 OVATION array, when present,
and uses the lower-MAE mode for all years. This keeps the new test maps aligned
with the historical generation pipeline rather than assuming a coupling mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from auroramaps import ovation as ao
from robustness_utils import (
    newell_coupling,
    select_activity_samples,
    select_evenly_spaced_valid,
    valid_condition_mask,
    weighted_newell_coupling,
)

DEFAULT_OMNI_ROOT = Path("/home/docker/data/private/AuroraData/omni_real_data/hourly_r1_2")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "r1_2_prepared"
DEFAULT_REFERENCE_2005 = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/aurora_img_20050101.npy"
)
YEARS = (2001, 2005, 2009)
PHASE_LABEL = {2001: "solar_maximum", 2005: "declining_phase", 2009: "solar_minimum"}
SOLAR_FIELDS = ("Bx", "By", "Bz", "V", "P")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omni-root", type=Path, default=DEFAULT_OMNI_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    p.add_argument("--phase-samples", type=int, default=48)
    p.add_argument("--kp-samples-per-group", type=int, default=48)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--coupling-mode", choices=("auto", "instant", "weighted4h"), default="auto")
    p.add_argument("--reference-2005", type=Path, default=DEFAULT_REFERENCE_2005)
    p.add_argument("--calibration-samples", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="Generate only first N unique samples for smoke testing; 0=all")
    return p.parse_args()


def load_year(root: Path, year: int) -> pd.DataFrame:
    path = root / str(year) / f"omni_{year}_hourly_r1_2.npy"
    if not path.exists():
        raise FileNotFoundError(f"Prepared hourly OMNI file not found: {path}")
    arr = np.load(path, allow_pickle=True)
    df = pd.DataFrame({name: arr[name] for name in arr.dtype.names})
    df["utc"] = pd.to_datetime(df["utc"])
    df["year"] = year
    return df


def coupling_for_row(df: pd.DataFrame, idx: int, mode: str) -> float:
    row = df.iloc[idx]
    if mode == "instant":
        return float(newell_coupling(np.array([row.By]), np.array([row.Bz]), np.array([row.V]))[0])
    if mode == "weighted4h":
        return weighted_newell_coupling(df, idx, hours=4, decay=0.65)
    raise ValueError(mode)


def make_map(diff, mono, timestamp: pd.Timestamp, ec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = timestamp.to_pydatetime()
    mlat, mlt, fd = diff.get_flux_for_time(dt, ec)
    _, _, fm = mono.get_flux_for_time(dt, ec)
    flux = np.asarray(fd, dtype=np.float32) + np.asarray(fm, dtype=np.float32)
    if flux.shape != (80, 96):
        raise ValueError(f"Unexpected OVATION map shape {flux.shape}; expected (80, 96)")
    return np.asarray(mlat, dtype=np.float32), np.asarray(mlt, dtype=np.float32), np.clip(flux, 0.0, None)


def calibrate_coupling_mode(df2005: pd.DataFrame, diff, mono, reference_path: Path, n: int) -> tuple[str, Dict]:
    if not reference_path.exists():
        return "weighted4h", {"status": "reference_missing", "fallback": "weighted4h", "reference": str(reference_path)}
    ref = np.load(reference_path, allow_pickle=False)
    if ref.ndim != 3 or ref.shape[1:] != (80, 96):
        return "weighted4h", {"status": "reference_shape_unexpected", "shape": list(ref.shape), "fallback": "weighted4h"}
    jan = df2005[(df2005.utc >= "2005-01-01") & (df2005.utc < "2005-02-01")].copy().reset_index(drop=True)
    valid = valid_condition_mask(jan, require_kp=False)
    idxs = select_evenly_spaced_valid(jan, min(n, int(valid.sum())), valid)
    records = []
    for local_idx in idxs:
        row = jan.iloc[int(local_idx)]
        hits = np.flatnonzero(df2005.utc.to_numpy(dtype="datetime64[ns]") == np.datetime64(row.utc, "ns"))
        if len(hits) != 1:
            continue
        global_idx = int(hits[0])
        minute_idx = int((row.utc - pd.Timestamp("2005-01-01")).total_seconds() // 60)
        if not (0 <= minute_idx < len(ref)):
            continue
        target = np.asarray(ref[minute_idx], dtype=np.float32)
        for mode in ("instant", "weighted4h"):
            ec = coupling_for_row(df2005, global_idx, mode)
            _, _, pred = make_map(diff, mono, row.utc, ec)
            records.append({
                "time": row.utc.isoformat(), "mode": mode, "ec": ec,
                "mae": float(np.mean(np.abs(pred - target))),
                "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
            })
    if not records:
        return "weighted4h", {"status": "no_calibration_rows", "fallback": "weighted4h"}
    tab = pd.DataFrame(records)
    summary = tab.groupby("mode")[["mae", "rmse"]].mean().to_dict(orient="index")
    selected = min(summary, key=lambda m: summary[m]["mae"])
    return selected, {"status": "calibrated", "selected": selected, "summary": summary, "records": records, "reference": str(reference_path)}


def build_manifest(frames: Dict[int, pd.DataFrame], phase_n: int, kp_n: int, seed: int) -> pd.DataFrame:
    rows = []
    for year, df in frames.items():
        valid = valid_condition_mask(df, require_kp=True)
        selected = select_evenly_spaced_valid(df, phase_n, valid)
        for idx in selected:
            row = df.iloc[int(idx)]
            rows.append({
                "utc": row.utc, "year": year, "phase": PHASE_LABEL.get(year, str(year)),
                "phase_sample": True, "activity_group": "",
            })
    activity = select_activity_samples(list(frames.values()), n_per_group=kp_n, seed=seed)
    for _, row in activity.iterrows():
        rows.append({
            "utc": row.utc, "year": int(row.year), "phase": PHASE_LABEL.get(int(row.year), str(row.year)),
            "phase_sample": False, "activity_group": row.activity_group,
        })
    raw = pd.DataFrame(rows)
    grouped = []
    for (year, utc), g in raw.groupby(["year", "utc"], sort=True):
        groups = sorted({x for x in g.activity_group.astype(str) if x})
        grouped.append({
            "year": int(year), "utc": pd.Timestamp(utc), "phase": g.phase.iloc[0],
            "phase_sample": bool(g.phase_sample.any()),
            "activity_low": "Kp<=3" in groups,
            "activity_high": "Kp>=4" in groups,
        })
    return pd.DataFrame(grouped).sort_values(["year", "utc"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames = {int(y): load_year(args.omni_root, int(y)) for y in args.years}

    print("=" * 88)
    print("R1.2 MULTI-YEAR OVATION TEST-SET GENERATION")
    print("=" * 88)
    for y, df in frames.items():
        valid = valid_condition_mask(df, require_kp=True)
        print(f"{y}: rows={len(df)}, valid+Kp={int(valid.sum())}, Kp<=3={int(np.sum(valid & (df.Kp<=3)))}, Kp>=4={int(np.sum(valid & (df.Kp>=4)))}")

    print("Initializing auroramaps FluxEstimator objects...")
    diff = ao.FluxEstimator("diff", "electron energy flux")
    mono = ao.FluxEstimator("mono", "electron energy flux")

    calibration = {"status": "not_requested"}
    mode = args.coupling_mode
    if mode == "auto":
        if 2005 not in frames:
            mode = "weighted4h"
            calibration = {"status": "2005_not_loaded", "fallback": mode}
        else:
            mode, calibration = calibrate_coupling_mode(frames[2005], diff, mono, args.reference_2005, args.calibration_samples)
    print(f"Coupling mode used: {mode}")
    if calibration.get("status") == "calibrated":
        print(f"Calibration mean errors: {calibration['summary']}")

    manifest = build_manifest(frames, args.phase_samples, args.kp_samples_per_group, args.seed)
    if args.limit > 0:
        manifest = manifest.iloc[:args.limit].copy()
        print(f"SMOKE LIMIT active: generating first {len(manifest)} unique samples")
    print(f"Unique maps to generate: {len(manifest)}")

    maps, solar, kps, ecs, times, years, phases, phase_flag, low_flag, high_flag = [], [], [], [], [], [], [], [], [], []
    mlat_ref = mlt_ref = None
    for i, rec in manifest.iterrows():
        year = int(rec.year)
        df = frames[year]
        hits = np.flatnonzero(df.utc.to_numpy(dtype="datetime64[ns]") == np.datetime64(rec.utc, "ns"))
        if len(hits) != 1:
            raise RuntimeError(f"Could not uniquely locate {rec.utc} in year {year}")
        idx = int(hits[0])
        row = df.iloc[idx]
        ec = coupling_for_row(df, idx, mode)
        mlat, mlt, flux = make_map(diff, mono, row.utc, ec)
        if mlat_ref is None:
            mlat_ref, mlt_ref = mlat, mlt
        maps.append(flux)
        solar.append([float(row[f]) for f in SOLAR_FIELDS])
        kps.append(float(row.Kp)); ecs.append(float(ec)); times.append(np.datetime64(row.utc, "ns")); years.append(year)
        phases.append(str(rec.phase)); phase_flag.append(bool(rec.phase_sample)); low_flag.append(bool(rec.activity_low)); high_flag.append(bool(rec.activity_high))
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(manifest):
            print(f"[{i+1:03d}/{len(manifest):03d}] {row.utc} year={year} Kp={row.Kp:.1f} Ec={ec:.1f} mean={flux.mean():.3f} max={flux.max():.3f}")

    stack = np.stack(maps).astype(np.float32)
    out_npz = args.output_root / ("multiyear_testset_smoke.npz" if args.limit > 0 else "multiyear_testset.npz")
    np.savez_compressed(
        out_npz,
        aurora=stack,
        solar=np.asarray(solar, dtype=np.float32),
        kp=np.asarray(kps, dtype=np.float32),
        ec=np.asarray(ecs, dtype=np.float32),
        utc=np.asarray(times, dtype="datetime64[ns]"),
        year=np.asarray(years, dtype=np.int16),
        phase=np.asarray(phases, dtype="U32"),
        phase_sample=np.asarray(phase_flag, dtype=bool),
        activity_low=np.asarray(low_flag, dtype=bool),
        activity_high=np.asarray(high_flag, dtype=bool),
        mlat=np.asarray(mlat_ref, dtype=np.float32),
        mlt=np.asarray(mlt_ref, dtype=np.float32),
    )
    manifest_out = manifest.copy()
    manifest_out["Kp"] = kps
    manifest_out["Ec"] = ecs
    manifest_path = args.output_root / ("manifest_smoke.csv" if args.limit > 0 else "manifest.csv")
    manifest_out.to_csv(manifest_path, index=False)

    audit = {
        "years": [int(y) for y in args.years],
        "phase_labels": PHASE_LABEL,
        "phase_samples_per_year_requested": int(args.phase_samples),
        "kp_samples_per_group_requested": int(args.kp_samples_per_group),
        "unique_generated_maps": int(len(manifest)),
        "coupling_mode": mode,
        "coupling_calibration": calibration,
        "output_npz": str(out_npz),
        "manifest": str(manifest_path),
        "aurora_shape": list(stack.shape),
        "solar_fields": list(SOLAR_FIELDS),
    }
    audit_path = args.output_root / ("generation_audit_smoke.json" if args.limit > 0 else "generation_audit.json")
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("\nDONE")
    print(f"NPZ:      {out_npz}")
    print(f"Manifest: {manifest_path}")
    print(f"Audit:    {audit_path}")
    print("Please send the console output and generation audit after the smoke run before generating the full set.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Second-stage SSUSI preprocessing audit for Reviewer 2, Comments 1 and 3.

Key correction relative to the first audit:
- actual SSUSI observational support is identified from UT_N, whose values are
  documented as non-zero over the SSUSI swath;
- the EDR-provided MLT_GRID_MAP is used directly rather than reconstructing MLT
  from geomagnetic longitude;
- target MLT grid is periodic [0, 24) with 96 bins (0, 0.25, ..., 23.75 h);
- interpolation is explicitly masked to local observational support so linear
  interpolation cannot turn unobserved gaps into apparent measurements.

The script audits the two manuscript SSUSI examples by default:
- Figure 6 nominal time: 2005-01-01 15:21 UT
- Figure 7 nominal time: 2005-01-04 03:03 UT

Run from repository root:
    python paper_modif/R2-1_R2-3/audit_ssusi_swath_v2.py

Outputs are written to:
    paper_modif/R2-1_R2-3/ssusi_swath_audit_v2/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
DEFAULT_RAW_ROOT = Path("/home/docker/data/private/AuroraData/real_aurora_data_ssusi/2005")
DEFAULT_PROCESSED = Path("/home/docker/data/private/AuroraData/process_ssusi/aurora_2005_ssusi.npy")
DEFAULT_OUTDIR = SCRIPT.with_name("ssusi_swath_audit_v2")

TARGETS = [
    ("Figure6", pd.Timestamp("2005-01-01T15:21:00")),
    ("Figure7", pd.Timestamp("2005-01-04T03:03:00")),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swath-aware SSUSI preprocessing audit")
    p.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--processed", type=str, default=str(DEFAULT_PROCESSED))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    p.add_argument(
        "--support-dilation",
        type=int,
        default=1,
        help="Number of target-grid cells used to expand direct swath support. Default: 1.",
    )
    p.add_argument(
        "--gap-widths",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 6.0],
        help="Synthetic MLT gap widths in hours.",
    )
    return p.parse_args()


def find_nc_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {root}")
    return files


def scalar(ds: xr.Dataset, name: str):
    arr = np.asarray(ds[name].values)
    return arr.reshape(-1)[0]


def edr_mid_time(ds: xr.Dataset) -> pd.Timestamp:
    year = int(scalar(ds, "YEAR"))
    doy = int(scalar(ds, "DOY"))
    seconds = float(scalar(ds, "TIME"))
    base = datetime(year - 1, 12, 31) + timedelta(days=doy)
    return pd.Timestamp(base + timedelta(seconds=seconds))


def nearest_files(files: List[Path], targets) -> List[Tuple[str, pd.Timestamp, Path, pd.Timestamp]]:
    stamped = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                stamped.append((f, edr_mid_time(ds)))
        except Exception:
            continue
    out = []
    for label, target in targets:
        f, ts = min(stamped, key=lambda x: abs((x[1] - target).total_seconds()))
        out.append((label, target, f, ts))
    return out


def periodic_mlt_distance(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, 24.0 - d)


def legacy_mlt_from_mlon(mlon):
    mlon_shifted = (mlon + 180.0) % 360.0 - 180.0
    return (mlon_shifted / 15.0 + 12.0) % 24.0


def circular_abs_error(a, b):
    return periodic_mlt_distance(a, b)


def target_grid():
    mlat = np.linspace(50.0, 90.0, 80)
    mlt = np.arange(96, dtype=np.float64) * (24.0 / 96.0)
    mlat2d, mlt2d = np.meshgrid(mlat, mlt, indexing="ij")
    return mlat, mlt, mlat2d, mlt2d


def periodic_linear_griddata(mlat_src, mlt_src, values, mlat2d, mlt2d):
    mlat_src = np.asarray(mlat_src, dtype=np.float64).ravel()
    mlt_src = np.asarray(mlt_src, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    good = np.isfinite(mlat_src) & np.isfinite(mlt_src) & np.isfinite(values)
    mlat_src = mlat_src[good]
    mlt_src = mlt_src[good]
    values = values[good]

    # Duplicate points by +/-24 h so interpolation is periodic across the 0/24 MLT seam.
    mlat_aug = np.concatenate([mlat_src, mlat_src, mlat_src])
    mlt_aug = np.concatenate([mlt_src - 24.0, mlt_src, mlt_src + 24.0])
    val_aug = np.concatenate([values, values, values])

    points = np.column_stack([mlat_aug, mlt_aug])
    target = np.column_stack([mlat2d.ravel(), mlt2d.ravel()])
    out = griddata(points, val_aug, target, method="linear", fill_value=np.nan)
    return out.reshape(mlat2d.shape)


def legacy_griddata(mlat_src, mlon_src, values):
    # Exact geometry used in data/extract_aurora.py, including the duplicated 24 h endpoint.
    mlt_legacy = legacy_mlt_from_mlon(mlon_src)
    tlat = np.linspace(50.0, 90.0, 80)
    tmlt = np.linspace(0.0, 24.0, 96)
    lat2d, mlt2d = np.meshgrid(tlat, tmlt, indexing="ij")
    good = np.isfinite(mlat_src) & np.isfinite(mlt_legacy) & np.isfinite(values)
    points = np.column_stack([mlat_src[good].ravel(), mlt_legacy[good].ravel()])
    vals = values[good].ravel()
    target = np.column_stack([lat2d.ravel(), mlt2d.ravel()])
    out = griddata(points, vals, target, method="linear", fill_value=np.nan)
    return np.nan_to_num(out.reshape(lat2d.shape), nan=0.0)


def bin_swath_support(mlat_src, mlt_src, valid_src, mlat_target, mlt_target):
    support = np.zeros((len(mlat_target), len(mlt_target)), dtype=bool)
    mlat_step = mlat_target[1] - mlat_target[0]
    mlt_step = 24.0 / len(mlt_target)

    lat = np.asarray(mlat_src)[valid_src]
    mlt = np.asarray(mlt_src)[valid_src] % 24.0
    good = np.isfinite(lat) & np.isfinite(mlt) & (lat >= 50.0) & (lat <= 90.0)
    lat = lat[good]
    mlt = mlt[good]

    ilat = np.rint((lat - mlat_target[0]) / mlat_step).astype(int)
    imlt = np.rint(mlt / mlt_step).astype(int) % len(mlt_target)
    ok = (ilat >= 0) & (ilat < len(mlat_target))
    support[ilat[ok], imlt[ok]] = True
    return support


def dilate_support(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, n)):
        prev = out.copy()
        new = prev.copy()
        # periodic in MLT
        new |= np.roll(prev, 1, axis=1)
        new |= np.roll(prev, -1, axis=1)
        # non-periodic in MLAT
        new[1:, :] |= prev[:-1, :]
        new[:-1, :] |= prev[1:, :]
        # diagonals
        left = np.roll(prev, 1, axis=1)
        right = np.roll(prev, -1, axis=1)
        new[1:, :] |= left[:-1, :] | right[:-1, :]
        new[:-1, :] |= left[1:, :] | right[1:, :]
        out = new
    return out


def mlt_coverage(mask: np.ndarray, mlat_target: np.ndarray):
    band = (mlat_target >= 60.0) & (mlat_target <= 80.0)
    by_mlt = np.any(mask[band, :], axis=0)
    hours = by_mlt.sum() * 24.0 / by_mlt.size

    # Largest periodic gap in bins.
    if by_mlt.all():
        largest = 0
    elif not by_mlt.any():
        largest = len(by_mlt)
    else:
        doubled = np.concatenate([~by_mlt, ~by_mlt])
        best = cur = 0
        for v in doubled:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        largest = min(best, len(by_mlt))
    gap_h = largest * 24.0 / by_mlt.size
    return float(hours), float(gap_h)


def parse_filename_interval(path: Path):
    # Expected form: ..._YYYYDOYT.hhmmss-YYYYDOYT.hhmmss-REVxxxxx_...
    import re
    m = re.search(r"_(\d{7})T(\d{6})-(\d{7})T(\d{6})-REV(\d+)", path.name)
    if not m:
        return None, None, None
    def parse(yday, hms):
        year = int(yday[:4]); doy = int(yday[4:])
        dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return pd.Timestamp(dt.replace(hour=int(hms[:2]), minute=int(hms[2:4]), second=int(hms[4:])))
    return parse(m.group(1), m.group(2)), parse(m.group(3), m.group(4)), m.group(5)


def load_processed(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def nearest_processed(processed, ts: pd.Timestamp):
    if processed is None:
        return None, None
    pts = pd.to_datetime(processed["utc"])
    diffs = np.abs((pts - ts).total_seconds())
    idx = int(np.argmin(diffs))
    return np.asarray(processed["aurora_flux"][idx], dtype=np.float64), pd.Timestamp(pts[idx])


def circular_gap_mask(mlt2d, center, width):
    return periodic_mlt_distance(mlt2d, center) <= width / 2.0


def analyze_case(label, nominal_target, file_path, ts, processed, outdir: Path, dilation: int, gap_widths):
    with xr.open_dataset(file_path) as ds:
        required = [
            "ENERGY_FLUX_NORTH_MAP",
            "LATITUDE_GEOMAGNETIC_GRID_MAP",
            "LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP",
            "MLT_GRID_MAP",
            "UT_N",
        ]
        missing = [x for x in required if x not in ds]
        if missing:
            raise KeyError(f"{file_path.name} missing required variables: {missing}")

        flux = np.asarray(ds["ENERGY_FLUX_NORTH_MAP"].values, dtype=np.float64).squeeze()
        mlat = np.asarray(ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values, dtype=np.float64).squeeze()
        mlon = np.asarray(ds["LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP"].values, dtype=np.float64).squeeze()
        mlt_product = np.asarray(ds["MLT_GRID_MAP"].values, dtype=np.float64).squeeze()
        ut_n = np.asarray(ds["UT_N"].values, dtype=np.float64).squeeze()

        if not (flux.shape == mlat.shape == mlon.shape == mlt_product.shape == ut_n.shape):
            raise ValueError(
                f"Shape mismatch: flux={flux.shape}, mlat={mlat.shape}, mlon={mlon.shape}, "
                f"mlt={mlt_product.shape}, UT_N={ut_n.shape}"
            )

        north_data = None
        if "NORTH_DATA" in ds:
            try:
                north_data = int(np.asarray(ds["NORTH_DATA"].values).reshape(-1)[0])
            except Exception:
                north_data = None

        dq_global = None
        if "DATA_QUALITY_GLOBAL" in ds:
            try:
                dq_global = int(np.asarray(ds["DATA_QUALITY_GLOBAL"].values).reshape(-1)[0])
            except Exception:
                dq_global = None

    mlat_t, mlt_t, mlat2d, mlt2d = target_grid()

    finite_coords = np.isfinite(mlat) & np.isfinite(mlt_product)
    swath_native = finite_coords & np.isfinite(ut_n) & (ut_n > 0)
    swath_flux_native = swath_native & np.isfinite(flux)

    # Product-vs-legacy MLT coordinate error, assessed on actual SSUSI swath only.
    legacy_mlt = legacy_mlt_from_mlon(mlon)
    coord_good = swath_native & np.isfinite(legacy_mlt)
    mlt_err = circular_abs_error(legacy_mlt[coord_good], mlt_product[coord_good])

    # Reproduce the exact legacy preprocessing.
    legacy_flux = legacy_griddata(mlat, mlon, flux)

    # Regrid only actual swath measurements using the product-provided MLT coordinate.
    linear_swath_flux = periodic_linear_griddata(
        mlat[swath_flux_native], mlt_product[swath_flux_native], flux[swath_flux_native], mlat2d, mlt2d
    )

    direct_support = bin_swath_support(mlat, mlt_product, swath_native, mlat_t, mlt_t)
    local_support = dilate_support(direct_support, dilation)
    safe_flux = np.where(local_support, linear_swath_flux, np.nan)

    direct_cov_h, direct_gap_h = mlt_coverage(direct_support, mlat_t)
    local_cov_h, local_gap_h = mlt_coverage(local_support, mlat_t)

    # Fractions of apparent legacy auroral signal falling outside actual local swath support.
    band = (mlat2d >= 60.0) & (mlat2d <= 80.0)
    signal_metrics = {}
    for thr in [0.01, 0.2, 1.0]:
        sig = (legacy_flux > thr) & band
        outside = sig & (~local_support)
        signal_metrics[f"legacy_signal_gt_{thr:g}_pixels"] = int(sig.sum())
        signal_metrics[f"legacy_signal_gt_{thr:g}_outside_swath_pixels"] = int(outside.sum())
        signal_metrics[f"legacy_signal_gt_{thr:g}_outside_swath_fraction"] = (
            float(outside.sum() / sig.sum()) if sig.sum() else float("nan")
        )

    # Compare exact legacy recreation against stored processed array.
    processed_flux, processed_ts = nearest_processed(processed, ts)
    if processed_flux is not None and processed_flux.shape == legacy_flux.shape:
        diff = np.abs(processed_flux - legacy_flux)
        processed_mae = float(np.mean(diff))
        processed_max = float(np.max(diff))
        processed_dt = float(abs((processed_ts - ts).total_seconds()))
    else:
        processed_mae = processed_max = processed_dt = float("nan")

    start, stop, orbit = parse_filename_interval(file_path)

    # Synthetic-gap test performed only on true native swath points.
    swath_mlt_values = mlt_product[swath_flux_native] % 24.0
    # Circular mean gives a stable center even if swath approaches the 0/24 seam.
    ang = 2.0 * np.pi * swath_mlt_values / 24.0
    center = (np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * np.pi)) * 24.0 / (2 * np.pi)

    gap_rows = []
    for width in gap_widths:
        remove_src = swath_flux_native & (periodic_mlt_distance(mlt_product % 24.0, center) <= width / 2.0)
        retained = swath_flux_native & (~remove_src)

        unsafe = periodic_linear_griddata(
            mlat[retained], mlt_product[retained], flux[retained], mlat2d, mlt2d
        )
        support_after = bin_swath_support(mlat, mlt_product, retained, mlat_t, mlt_t)
        local_after = dilate_support(support_after, dilation)
        safe_after = np.where(local_after, unsafe, np.nan)

        target_gap = circular_gap_mask(mlt2d, center, width) & band
        n_target = int(target_gap.sum())
        unsafe_filled = int((target_gap & np.isfinite(unsafe)).sum())
        safe_filled = int((target_gap & np.isfinite(safe_after)).sum())
        gap_rows.append({
            "case": label,
            "timestamp": ts.isoformat(),
            "gap_center_mlt_h": float(center),
            "gap_width_h": float(width),
            "target_gap_cells": n_target,
            "unsafe_linear_filled_cells": unsafe_filled,
            "unsafe_bridge_fraction": float(unsafe_filled / n_target) if n_target else np.nan,
            "coverage_masked_filled_cells": safe_filled,
            "coverage_masked_bridge_fraction": float(safe_filled / n_target) if n_target else np.nan,
        })

    row: Dict[str, object] = {
        "case": label,
        "nominal_manuscript_time": nominal_target.isoformat(),
        "edr_mid_time": ts.isoformat(),
        "source_file": str(file_path),
        "orbit": orbit,
        "file_start_time": None if start is None else start.isoformat(),
        "file_stop_time": None if stop is None else stop.isoformat(),
        "file_duration_minutes": None if start is None else float((stop - start).total_seconds() / 60.0),
        "north_data": north_data,
        "data_quality_global": dq_global,
        "native_grid_shape": list(flux.shape),
        "native_total_pixels": int(flux.size),
        "native_swath_pixels_from_UT_N": int(swath_native.sum()),
        "native_swath_fraction_from_UT_N": float(swath_native.mean()),
        "native_swath_flux_pixels": int(swath_flux_native.sum()),
        "legacy_vs_product_mlt_median_abs_error_h_on_swath": float(np.nanmedian(mlt_err)),
        "legacy_vs_product_mlt_p95_abs_error_h_on_swath": float(np.nanpercentile(mlt_err, 95)),
        "legacy_vs_product_mlt_max_abs_error_h_on_swath": float(np.nanmax(mlt_err)),
        "direct_target_support_fraction": float(direct_support.mean()),
        "local_target_support_fraction": float(local_support.mean()),
        "direct_mlt_coverage_h_60_80": direct_cov_h,
        "direct_largest_mlt_gap_h_60_80": direct_gap_h,
        "local_mlt_coverage_h_60_80": local_cov_h,
        "local_largest_mlt_gap_h_60_80": local_gap_h,
        "processed_match_time": None if processed_ts is None else processed_ts.isoformat(),
        "processed_time_difference_seconds": processed_dt,
        "processed_mean_abs_difference_vs_legacy": processed_mae,
        "processed_max_abs_difference_vs_legacy": processed_max,
    }
    row.update(signal_metrics)

    # Diagnostics figure.
    legacy_outside = (legacy_flux > 0.2) & band & (~local_support)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    extent = [0, 24, 50, 90]

    im0 = axes[0, 0].imshow(legacy_flux, origin="lower", aspect="auto", extent=extent)
    axes[0, 0].set_title("Legacy preprocessing flux\n(legacy MLON→MLT, all finite map pixels)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(np.nan_to_num(linear_swath_flux, nan=0.0), origin="lower", aspect="auto", extent=extent)
    axes[0, 1].set_title("Linear interpolation from UT_N swath only\n(using EDR MLT_GRID_MAP)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(direct_support.astype(float), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[0, 2].set_title("Direct target-grid support\nfrom UT_N > 0")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    im3 = axes[1, 0].imshow(local_support.astype(float), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1, 0].set_title(f"Local support envelope\n({dilation}-cell dilation)")
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    im4 = axes[1, 1].imshow(legacy_outside.astype(float), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1, 1].set_title("Legacy signal > 0.2 outside\nUT_N-based local support")
    fig.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    im5 = axes[1, 2].imshow(np.ma.masked_invalid(safe_flux), origin="lower", aspect="auto", extent=extent)
    axes[1, 2].set_title("Coverage-preserving regridded flux\n(outside support = missing)")
    fig.colorbar(im5, ax=axes[1, 2], shrink=0.8)

    for ax in axes.ravel():
        ax.set_xlabel("MLT [h]")
        ax.set_ylabel("MLAT [deg]")
        ax.set_xlim(0, 24)
        ax.set_ylim(50, 90)

    fig.suptitle(
        f"SSUSI swath-aware audit — {label} — EDR midpoint {ts}\n"
        f"{file_path.name}\n"
        f"Orbit {orbit}, interval {start} to {stop}",
        fontsize=12,
    )
    fig_path = outdir / f"swath_audit_{label}_{ts.strftime('%Y%m%dT%H%M%S')}.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    np.savez_compressed(
        outdir / f"swath_audit_{label}_{ts.strftime('%Y%m%dT%H%M%S')}.npz",
        legacy_flux=legacy_flux,
        linear_swath_flux=linear_swath_flux,
        direct_support=direct_support,
        local_support=local_support,
        safe_flux=safe_flux,
        mlat=mlat_t,
        mlt=mlt_t,
    )

    return row, gap_rows, fig_path


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    processed_path = Path(args.processed).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SSUSI SWATH-AWARE PREPROCESSING AUDIT V2")
    print("=" * 80)
    print(f"Raw EDR root:    {raw_root}")
    print(f"Processed array: {processed_path}")
    print(f"Output dir:      {outdir}")
    print("Support definition: UT_N > 0 (actual SSUSI swath support)")
    print("Coordinate definition: product MLT_GRID_MAP")

    files = find_nc_files(raw_root)
    selected = nearest_files(files, TARGETS)
    processed = load_processed(processed_path)

    rows = []
    gap_rows_all = []
    metadata = []

    for label, target, f, ts in selected:
        print("\n" + "-" * 80)
        print(f"{label}: nominal manuscript time {target}, nearest EDR midpoint {ts}")
        print(f"File: {f}")
        row, gap_rows, fig_path = analyze_case(
            label, target, f, ts, processed, outdir, args.support_dilation, args.gap_widths
        )
        rows.append(row)
        gap_rows_all.extend(gap_rows)
        metadata.append(row)

        print(f"Orbit: {row['orbit']}")
        print(f"Acquisition interval: {row['file_start_time']} to {row['file_stop_time']}")
        print(f"Duration: {row['file_duration_minutes']:.2f} min")
        print(f"Native UT_N swath fraction: {row['native_swath_fraction_from_UT_N']:.4f}")
        print(
            "Legacy-vs-product MLT error on swath: "
            f"median={row['legacy_vs_product_mlt_median_abs_error_h_on_swath']:.3f} h, "
            f"p95={row['legacy_vs_product_mlt_p95_abs_error_h_on_swath']:.3f} h, "
            f"max={row['legacy_vs_product_mlt_max_abs_error_h_on_swath']:.3f} h"
        )
        print(
            "60–80 MLAT coverage: "
            f"direct={row['direct_mlt_coverage_h_60_80']:.2f} h, "
            f"local envelope={row['local_mlt_coverage_h_60_80']:.2f} h"
        )
        print(
            "Largest 60–80 MLAT MLT gap: "
            f"direct={row['direct_largest_mlt_gap_h_60_80']:.2f} h, "
            f"local envelope={row['local_largest_mlt_gap_h_60_80']:.2f} h"
        )
        for thr in [0.01, 0.2, 1.0]:
            key = f"legacy_signal_gt_{thr:g}_outside_swath_fraction"
            print(f"Legacy signal > {thr:g} outside local swath support: {row[key]:.4f}")
        print(f"Legacy processed-array reproduction MAE: {row['processed_mean_abs_difference_vs_legacy']:.6g}")
        print(f"Diagnostic: {fig_path}")

    df = pd.DataFrame(rows)
    gaps = pd.DataFrame(gap_rows_all)
    df.to_csv(outdir / "ssusi_swath_summary_v2.csv", index=False)
    gaps.to_csv(outdir / "ssusi_swath_synthetic_gap_v2.csv", index=False)
    with (outdir / "ssusi_swath_metadata_v2.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SYNTHETIC GAP RESULTS")
    print("=" * 80)
    if not gaps.empty:
        print(
            gaps.groupby("gap_width_h")[["unsafe_bridge_fraction", "coverage_masked_bridge_fraction"]]
            .agg(["mean", "min", "max"])
            .to_string()
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(outdir / "ssusi_swath_summary_v2.csv")
    print(outdir / "ssusi_swath_synthetic_gap_v2.csv")
    print(outdir / "ssusi_swath_metadata_v2.json")
    print("Please send the complete console output and both swath_audit_*.png files.")


if __name__ == "__main__":
    main()

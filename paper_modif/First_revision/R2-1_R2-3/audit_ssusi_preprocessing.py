#!/usr/bin/env python3
"""Audit SSUSI EDR preprocessing for Reviewer 2, Comments 1 and 3.

Purpose
-------
This script does NOT retrain or run the diffusion model. It audits the
DMSP/SSUSI preprocessing chain that converts a native Auroral-EDR map to the
80 x 96 MLAT-MLT grid used by the paper.

It tests two specific concerns:
1. Coordinate handling: the historical preprocessing reconstructed MLT from
   LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP using a fixed longitude-to-MLT formula.
   SSUSI Auroral-EDR products normally provide MLT_GRID_MAP directly. When that
   variable is available, this script quantifies the disagreement.
2. Coverage inflation: scipy.griddata(method='linear') fills the convex hull of
   the available samples and can therefore bridge genuinely unobserved spatial
   gaps. The script compares linear-interpolation support against a conservative
   support mask obtained directly from native valid SSUSI pixels, and performs a
   synthetic MLT-gap stress test.

The historical data are never overwritten.

Typical run from repository root:
    python paper_modif/R2-1_R2-3/audit_ssusi_preprocessing.py

Useful options:
    --raw-dir /path/to/raw/2005/ssusi
    --processed-npy /path/to/aurora_2005_ssusi.npy
    --sample-count 20
    --target-times 2005-01-01T15:21:00 2005-01-04T03:03:00

The two default target times are the Figure 6 and Figure 7 cases in the current
manuscript, respectively.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata


DEFAULT_RAW_DIR = Path(
    "/home/docker/data/private/AuroraData/real_aurora_data_ssusi/2005"
)
DEFAULT_PROCESSED_NPY = Path(
    "/home/docker/data/private/AuroraData/process_ssusi/aurora_2005_ssusi.npy"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ssusi_coverage_audit"
DEFAULT_TARGET_TIMES = [
    "2005-01-01T15:21:00",  # Figure 6 in the current manuscript
    "2005-01-04T03:03:00",  # Figure 7 in the current manuscript
]

# Paper grid. The corrected diagnostic grid uses endpoint=False for MLT so that
# 0 and 24 MLT are not duplicated. The historical pipeline used linspace(0,24,96)
# with the endpoint included; that exact grid is reproduced separately below.
TARGET_MLAT = np.linspace(50.0, 90.0, 80)
TARGET_MLT = np.linspace(0.0, 24.0, 96, endpoint=False)
TARGET_MLAT_2D, TARGET_MLT_2D = np.meshgrid(
    TARGET_MLAT, TARGET_MLT, indexing="ij"
)
TARGET_POINTS = np.column_stack((TARGET_MLAT_2D.ravel(), TARGET_MLT_2D.ravel()))

LEGACY_TARGET_MLT = np.linspace(0.0, 24.0, 96)  # exact historical code
LEGACY_MLAT_2D, LEGACY_MLT_2D = np.meshgrid(
    TARGET_MLAT, LEGACY_TARGET_MLT, indexing="ij"
)
LEGACY_TARGET_POINTS = np.column_stack(
    (LEGACY_MLAT_2D.ravel(), LEGACY_MLT_2D.ravel())
)

AURORAL_BAND = (60.0, 80.0)


@dataclass
class EDRRecord:
    path: Path
    timestamp: datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit SSUSI EDR coordinate conversion and coverage inflation."
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--processed-npy", type=Path, default=DEFAULT_PROCESSED_NPY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--target-times",
        nargs="*",
        default=DEFAULT_TARGET_TIMES,
        help="UTC times to audit in detail; nearest EDR file is selected.",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="Additional files, evenly sampled across the directory, for baseline statistics.",
    )
    p.add_argument(
        "--dilation-cells",
        type=int,
        default=1,
        help=(
            "Number of target-grid cells used to enlarge direct native support. "
            "One cell is deliberately lenient and avoids labeling small regridding offsets as artifacts."
        ),
    )
    p.add_argument(
        "--gap-widths",
        nargs="*",
        type=float,
        default=[1.0, 2.0, 4.0, 6.0],
        help="Synthetic MLT gap widths in hours.",
    )
    return p.parse_args()


def scalar(ds: xr.Dataset, name: str) -> float:
    value = ds[name].values
    return float(np.asarray(value).squeeze().item())


def edr_timestamp(ds: xr.Dataset) -> datetime:
    year = int(scalar(ds, "YEAR"))
    doy = int(scalar(ds, "DOY"))
    seconds = scalar(ds, "TIME")
    return datetime(year - 1, 12, 31) + timedelta(days=doy, seconds=seconds)


def scan_records(raw_dir: Path) -> List[EDRRecord]:
    files = sorted(raw_dir.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {raw_dir}")

    records: List[EDRRecord] = []
    print(f"Scanning timestamps for {len(files)} raw EDR files ...")
    for i, path in enumerate(files, 1):
        try:
            with xr.open_dataset(path) as ds:
                ts = edr_timestamp(ds)
            records.append(EDRRecord(path=path, timestamp=ts))
        except Exception as exc:
            print(f"  [skip metadata] {path}: {exc}")
        if i % 100 == 0:
            print(f"  scanned {i}/{len(files)}")

    records.sort(key=lambda r: r.timestamp)
    if not records:
        raise RuntimeError("No readable EDR files with YEAR/DOY/TIME metadata were found.")
    return records


def nearest_record(records: Sequence[EDRRecord], target: datetime) -> EDRRecord:
    return min(records, key=lambda r: abs((r.timestamp - target).total_seconds()))


def as_grid_like(array: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(array).squeeze()
    if arr.shape == shape:
        return arr
    if arr.shape == shape[::-1]:
        return arr.T
    if arr.ndim == 1 and arr.size == shape[0]:
        return np.repeat(arr[:, None], shape[1], axis=1)
    if arr.ndim == 1 and arr.size == shape[1]:
        return np.repeat(arr[None, :], shape[0], axis=0)
    raise ValueError(f"Cannot broadcast {name} with shape {arr.shape} to {shape}")


def load_edr(path: Path) -> Dict[str, object]:
    with xr.open_dataset(path) as ds:
        flux = np.asarray(ds["ENERGY_FLUX_NORTH_MAP"].values).squeeze().astype(float)
        if flux.ndim != 2:
            raise ValueError(f"ENERGY_FLUX_NORTH_MAP must be 2-D after squeeze, got {flux.shape}")

        mlat = as_grid_like(
            ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values,
            flux.shape,
            "LATITUDE_GEOMAGNETIC_GRID_MAP",
        ).astype(float)
        mlon = as_grid_like(
            ds["LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP"].values,
            flux.shape,
            "LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP",
        ).astype(float)

        product_mlt = None
        if "MLT_GRID_MAP" in ds.variables:
            product_mlt = as_grid_like(
                ds["MLT_GRID_MAP"].values,
                flux.shape,
                "MLT_GRID_MAP",
            ).astype(float)

        ts = edr_timestamp(ds)
        candidate_metadata = {}
        keywords = ("ORBIT", "START", "STOP", "TIME")
        for key in list(ds.variables) + list(ds.attrs):
            upper = str(key).upper()
            if any(k in upper for k in keywords):
                try:
                    val = ds[key].values if key in ds.variables else ds.attrs[key]
                    arr = np.asarray(val)
                    if arr.size <= 20:
                        candidate_metadata[str(key)] = arr.tolist() if arr.ndim else arr.item()
                except Exception:
                    pass

        variable_names = sorted(str(v) for v in ds.variables)

    valid = np.isfinite(flux) & np.isfinite(mlat) & np.isfinite(mlon)
    if product_mlt is not None:
        valid &= np.isfinite(product_mlt)

    # Historical conversion in data/extract_aurora.py. It is intentionally
    # reproduced exactly here for comparison; it is not assumed to be correct.
    mlon_shifted = (mlon + 180.0) % 360.0 - 180.0
    legacy_mlt = (mlon_shifted / 15.0 + 12.0) % 24.0

    return {
        "flux": flux,
        "mlat": mlat,
        "mlon": mlon,
        "product_mlt": product_mlt,
        "legacy_mlt": legacy_mlt,
        "valid": valid,
        "timestamp": ts,
        "metadata": candidate_metadata,
        "variables": variable_names,
    }


def circular_distance_hours(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 12.0) % 24.0 - 12.0)


def linear_regrid(
    mlat: np.ndarray,
    mlt: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    target_points: np.ndarray = TARGET_POINTS,
    target_shape: Tuple[int, int] = (80, 96),
) -> np.ndarray:
    points = np.column_stack((mlat[valid], mlt[valid]))
    values = flux[valid]
    if len(values) < 3:
        return np.full(target_shape, np.nan, dtype=float)
    out = griddata(points, values, target_points, method="linear", fill_value=np.nan)
    return out.reshape(target_shape)


def direct_support(mlat: np.ndarray, mlt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rasterize actual native valid pixels onto the standard 80x96 target grid."""
    lat = mlat[valid]
    lt = mlt[valid] % 24.0

    inside = (lat >= TARGET_MLAT.min()) & (lat <= TARGET_MLAT.max())
    lat = lat[inside]
    lt = lt[inside]

    mask = np.zeros((len(TARGET_MLAT), len(TARGET_MLT)), dtype=bool)
    if lat.size == 0:
        return mask

    dlat = TARGET_MLAT[1] - TARGET_MLAT[0]
    dmlt = 24.0 / len(TARGET_MLT)
    i = np.rint((lat - TARGET_MLAT[0]) / dlat).astype(int)
    j = np.rint(lt / dmlt).astype(int) % len(TARGET_MLT)
    i = np.clip(i, 0, len(TARGET_MLAT) - 1)
    mask[i, j] = True
    return mask


def dilate_periodic(mask: np.ndarray, cells: int) -> np.ndarray:
    out = mask.copy()
    base = mask.copy()
    for _ in range(max(0, cells)):
        expanded = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(out, shift=dc, axis=1)  # MLT is periodic
                if dr == 1:
                    shifted = np.vstack([np.zeros((1, shifted.shape[1]), bool), shifted[:-1]])
                elif dr == -1:
                    shifted = np.vstack([shifted[1:], np.zeros((1, shifted.shape[1]), bool)])
                expanded |= shifted
        out = expanded
    return out


def band_rows() -> np.ndarray:
    return (TARGET_MLAT >= AURORAL_BAND[0]) & (TARGET_MLAT <= AURORAL_BAND[1])


def fraction(mask: np.ndarray, region: Optional[np.ndarray] = None) -> float:
    if region is None:
        return float(mask.mean())
    denom = int(region.sum())
    return float((mask & region).sum() / denom) if denom else float("nan")


def artifact_fraction(linear_support: np.ndarray, local_support: np.ndarray, region: np.ndarray) -> float:
    denom = int((linear_support & region).sum())
    if denom == 0:
        return float("nan")
    return float((linear_support & ~local_support & region).sum() / denom)


def mlt_coverage(mask: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.any(mask[rows, :], axis=0)


def largest_circular_false_run(covered: np.ndarray) -> int:
    covered = np.asarray(covered, dtype=bool)
    n = covered.size
    if n == 0 or covered.all():
        return 0
    if (~covered).all():
        return n
    doubled = np.concatenate([~covered, ~covered])
    best = cur = 0
    for val in doubled:
        cur = cur + 1 if val else 0
        best = max(best, cur)
    return min(best, n)


def mlt_metrics(mask: np.ndarray) -> Tuple[float, float]:
    covered = mlt_coverage(mask, band_rows())
    hours_per_bin = 24.0 / len(TARGET_MLT)
    coverage_h = float(covered.sum() * hours_per_bin)
    largest_gap_h = float(largest_circular_false_run(covered) * hours_per_bin)
    return coverage_h, largest_gap_h


def circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 12.0) % 24.0 - 12.0)


def compare_processed_npy(
    processed_path: Path,
    timestamp: datetime,
    legacy_exact_grid: np.ndarray,
) -> Dict[str, float]:
    if not processed_path.exists():
        return {"processed_npy_found": False}
    arr = np.load(processed_path, allow_pickle=True)
    if "utc" not in arr.dtype.names or "aurora_flux" not in arr.dtype.names:
        return {"processed_npy_found": True, "processed_npy_compatible": False}

    times = pd.to_datetime(arr["utc"])
    target = pd.Timestamp(timestamp)
    idx = int(np.argmin(np.abs(times - target)))
    matched_time = times[idx]
    observed = np.asarray(arr["aurora_flux"][idx], dtype=float)
    if observed.shape != legacy_exact_grid.shape:
        return {
            "processed_npy_found": True,
            "processed_npy_compatible": False,
            "processed_shape": list(observed.shape),
        }

    reproduced = np.nan_to_num(legacy_exact_grid, nan=0.0)
    diff = np.abs(reproduced - observed)
    return {
        "processed_npy_found": True,
        "processed_npy_compatible": True,
        "processed_match_time": str(matched_time),
        "processed_time_difference_seconds": float(abs((matched_time - target).total_seconds())),
        "processed_mean_abs_difference": float(np.mean(diff)),
        "processed_max_abs_difference": float(np.max(diff)),
    }


def choose_stress_center(mlt: np.ndarray, mlat: np.ndarray, valid: np.ndarray) -> float:
    in_band = valid & (mlat >= AURORAL_BAND[0]) & (mlat <= AURORAL_BAND[1])
    vals = (mlt[in_band] % 24.0)
    if vals.size == 0:
        return 12.0
    hist, edges = np.histogram(vals, bins=np.linspace(0.0, 24.0, 97))
    j = int(np.argmax(hist))
    return float((edges[j] + edges[j + 1]) / 2.0)


def synthetic_gap_test(
    flux: np.ndarray,
    mlat: np.ndarray,
    mlt: np.ndarray,
    valid: np.ndarray,
    widths: Sequence[float],
) -> List[Dict[str, float]]:
    center = choose_stress_center(mlt, mlat, valid)
    rows = band_rows()
    band_region = np.repeat(rows[:, None], len(TARGET_MLT), axis=1)
    results = []

    for width in widths:
        remove = circular_distance_hours(mlt, center) <= width / 2.0
        retained = valid & ~remove
        interp = linear_regrid(mlat, mlt, flux, retained)
        finite = np.isfinite(interp)

        target_gap_cols = circular_distance_hours(TARGET_MLT, center) <= width / 2.0
        gap_region = band_region & np.repeat(target_gap_cols[None, :], len(TARGET_MLAT), axis=0)
        denom = int(gap_region.sum())
        bridged = int((finite & gap_region).sum())
        results.append(
            {
                "gap_center_mlt_h": center,
                "gap_width_h": float(width),
                "target_gap_cells": denom,
                "linearly_filled_gap_cells": bridged,
                "bridge_fraction": float(bridged / denom) if denom else float("nan"),
            }
        )
    return results


def audit_file(path: Path, dilation_cells: int, processed_npy: Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray], List[Dict[str, float]]]:
    edr = load_edr(path)
    flux = edr["flux"]
    mlat = edr["mlat"]
    legacy_mlt = edr["legacy_mlt"]
    product_mlt = edr["product_mlt"]
    valid = edr["valid"]
    timestamp = edr["timestamp"]

    if product_mlt is None:
        raise KeyError(
            f"{path.name} has no MLT_GRID_MAP. The audit needs either that EDR variable "
            "or a documented time-dependent MLON-to-MLT conversion."
        )

    # Coordinate discrepancy between the historical formula and the product's own MLT grid.
    coord_valid = valid & np.isfinite(product_mlt) & np.isfinite(legacy_mlt)
    err = circular_error(legacy_mlt[coord_valid], product_mlt[coord_valid] % 24.0)

    # Linear interpolation on a common, nonduplicated target grid.
    interp_legacy_coord = linear_regrid(mlat, legacy_mlt, flux, valid)
    interp_product_coord = linear_regrid(mlat, product_mlt % 24.0, flux, valid)

    # Exact historical grid, solely to verify reproducibility against aurora_2005_ssusi.npy.
    legacy_exact = linear_regrid(
        mlat,
        legacy_mlt,
        flux,
        valid,
        target_points=LEGACY_TARGET_POINTS,
        target_shape=(80, 96),
    )

    direct = direct_support(mlat, product_mlt % 24.0, valid)
    local = dilate_periodic(direct, dilation_cells)
    linear_support = np.isfinite(interp_product_coord)
    legacy_support = np.isfinite(interp_legacy_coord)
    artifact = linear_support & ~local

    rows = band_rows()
    band_region = np.repeat(rows[:, None], len(TARGET_MLT), axis=1)

    direct_hours, direct_gap = mlt_metrics(direct)
    local_hours, local_gap = mlt_metrics(local)
    linear_hours, linear_gap = mlt_metrics(linear_support)

    corrected_flux = np.where(local, interp_product_coord, np.nan)

    metrics: Dict[str, object] = {
        "file": str(path),
        "timestamp": timestamp.isoformat(),
        "native_shape": list(flux.shape),
        "native_valid_pixels": int(valid.sum()),
        "native_valid_fraction": float(valid.mean()),
        "negative_finite_flux_pixels": int((valid & (flux < 0)).sum()),
        "product_mlt_available": True,
        "legacy_vs_product_mlt_median_abs_error_h": float(np.median(err)) if err.size else float("nan"),
        "legacy_vs_product_mlt_p95_abs_error_h": float(np.percentile(err, 95)) if err.size else float("nan"),
        "legacy_vs_product_mlt_max_abs_error_h": float(np.max(err)) if err.size else float("nan"),
        "direct_support_fraction_all": float(direct.mean()),
        "dilated_support_fraction_all": float(local.mean()),
        "linear_support_fraction_all": float(linear_support.mean()),
        "linear_outside_dilated_support_fraction_all": artifact_fraction(
            linear_support, local, np.ones_like(local, dtype=bool)
        ),
        "direct_support_fraction_60_80_mlat": fraction(direct, band_region),
        "dilated_support_fraction_60_80_mlat": fraction(local, band_region),
        "linear_support_fraction_60_80_mlat": fraction(linear_support, band_region),
        "linear_outside_dilated_support_fraction_60_80_mlat": artifact_fraction(
            linear_support, local, band_region
        ),
        "direct_mlt_coverage_h_60_80": direct_hours,
        "direct_largest_mlt_gap_h_60_80": direct_gap,
        "dilated_mlt_coverage_h_60_80": local_hours,
        "dilated_largest_mlt_gap_h_60_80": local_gap,
        "linear_mlt_coverage_h_60_80": linear_hours,
        "linear_largest_mlt_gap_h_60_80": linear_gap,
        "legacy_linear_support_fraction_all": float(legacy_support.mean()),
    }
    metrics.update(compare_processed_npy(processed_npy, timestamp, legacy_exact))

    stress = synthetic_gap_test(
        flux,
        mlat,
        product_mlt % 24.0,
        valid,
        widths=[],  # filled by caller for detailed target files
    )

    arrays = {
        "interp_legacy_coord": interp_legacy_coord,
        "interp_product_coord": interp_product_coord,
        "direct_support": direct,
        "dilated_support": local,
        "linear_support": linear_support,
        "artifact": artifact,
        "corrected_flux": corrected_flux,
    }
    return metrics, arrays, stress


def plot_diagnostic(
    metrics: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    extent = [0, 24, TARGET_MLAT.min(), TARGET_MLAT.max()]

    items = [
        (arrays["interp_legacy_coord"], "Legacy linear flux\n(legacy MLON→MLT)"),
        (arrays["interp_product_coord"], "Linear flux using EDR MLT_GRID_MAP"),
        (arrays["direct_support"].astype(float), "Direct native-pixel support"),
        (arrays["dilated_support"].astype(float), "1-cell local support envelope"),
        (arrays["artifact"].astype(float), "Linear-interpolation support\noutside local envelope"),
        (arrays["corrected_flux"], "Coverage-masked diagnostic flux"),
    ]

    for ax, (data, title) in zip(axes.ravel(), items):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("MLT [h]")
        ax.set_ylabel("MLAT [deg]")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"SSUSI preprocessing audit — {metrics['timestamp']}\n{Path(str(metrics['file'])).name}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sanitize_time(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%S")


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    processed_npy = args.processed_npy.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SSUSI PREPROCESSING AUDIT FOR REVIEWER 2, COMMENT 3")
    print("=" * 80)
    print(f"Raw EDR directory: {raw_dir}")
    print(f"Historical processed NPY: {processed_npy}")
    print(f"Output directory: {output_dir}")

    records = scan_records(raw_dir)
    print(f"Readable EDR records: {len(records)}")
    print(f"Time span: {records[0].timestamp} -- {records[-1].timestamp}")

    target_records: List[EDRRecord] = []
    print("\nTarget manuscript cases:")
    for text in args.target_times:
        target = datetime.fromisoformat(text.replace("Z", ""))
        rec = nearest_record(records, target)
        dt = abs((rec.timestamp - target).total_seconds())
        print(f"  requested {target} -> {rec.timestamp} ({dt:.1f} s), {rec.path.name}")
        if rec.path not in [r.path for r in target_records]:
            target_records.append(rec)

    # Add evenly distributed sample records for a broader baseline check.
    sample_records: List[EDRRecord] = []
    if args.sample_count > 0:
        idx = np.linspace(0, len(records) - 1, min(args.sample_count, len(records)), dtype=int)
        for i in idx:
            rec = records[int(i)]
            if rec.path not in [r.path for r in target_records]:
                sample_records.append(rec)

    all_records = target_records + sample_records
    print(f"\nAuditing {len(all_records)} files ({len(target_records)} manuscript targets + {len(sample_records)} sample files).")

    summary_rows: List[Dict[str, object]] = []
    stress_rows: List[Dict[str, object]] = []
    metadata_output: Dict[str, object] = {}

    for n, rec in enumerate(all_records, 1):
        print(f"\n[{n}/{len(all_records)}] {rec.timestamp}  {rec.path.name}")
        try:
            metrics, arrays, _ = audit_file(rec.path, args.dilation_cells, processed_npy)
            metrics["is_manuscript_target"] = rec.path in [r.path for r in target_records]
            summary_rows.append(metrics)

            print(
                "  MLT coordinate error (legacy vs product): "
                f"median={metrics['legacy_vs_product_mlt_median_abs_error_h']:.3f} h, "
                f"p95={metrics['legacy_vs_product_mlt_p95_abs_error_h']:.3f} h, "
                f"max={metrics['legacy_vs_product_mlt_max_abs_error_h']:.3f} h"
            )
            print(
                "  60–80 MLAT support: "
                f"direct={metrics['direct_support_fraction_60_80_mlat']:.3f}, "
                f"1-cell envelope={metrics['dilated_support_fraction_60_80_mlat']:.3f}, "
                f"linear={metrics['linear_support_fraction_60_80_mlat']:.3f}"
            )
            print(
                "  Linear support outside local envelope (60–80 MLAT): "
                f"{metrics['linear_outside_dilated_support_fraction_60_80_mlat']:.3f}"
            )
            print(
                "  MLT coverage (60–80 MLAT): "
                f"direct={metrics['direct_mlt_coverage_h_60_80']:.2f} h, "
                f"envelope={metrics['dilated_mlt_coverage_h_60_80']:.2f} h, "
                f"linear={metrics['linear_mlt_coverage_h_60_80']:.2f} h"
            )

            if metrics["is_manuscript_target"]:
                edr = load_edr(rec.path)
                metadata_output[rec.timestamp.isoformat()] = {
                    "file": str(rec.path),
                    "metadata_candidates": edr["metadata"],
                    "variables": edr["variables"],
                }

                stress = synthetic_gap_test(
                    edr["flux"],
                    edr["mlat"],
                    edr["product_mlt"] % 24.0,
                    edr["valid"],
                    args.gap_widths,
                )
                for row in stress:
                    stress_rows.append(
                        {
                            "file": str(rec.path),
                            "timestamp": rec.timestamp.isoformat(),
                            **row,
                        }
                    )
                    print(
                        f"  synthetic gap {row['gap_width_h']:.1f} h: "
                        f"bridge_fraction={row['bridge_fraction']:.3f}"
                    )

                diag_path = output_dir / f"diagnostic_{sanitize_time(rec.timestamp)}.png"
                plot_diagnostic(metrics, arrays, diag_path)
                np.savez_compressed(
                    output_dir / f"diagnostic_{sanitize_time(rec.timestamp)}.npz",
                    target_mlat=TARGET_MLAT,
                    target_mlt=TARGET_MLT,
                    **arrays,
                )
                print(f"  saved diagnostic: {diag_path}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary_rows.append(
                {
                    "file": str(rec.path),
                    "timestamp": rec.timestamp.isoformat(),
                    "is_manuscript_target": rec.path in [r.path for r in target_records],
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    stress_df = pd.DataFrame(stress_rows)
    summary_csv = output_dir / "ssusi_preprocessing_summary.csv"
    stress_csv = output_dir / "ssusi_synthetic_gap_stress.csv"
    summary_df.to_csv(summary_csv, index=False)
    stress_df.to_csv(stress_csv, index=False)

    with (output_dir / "target_edr_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_output, f, indent=2, ensure_ascii=False, default=json_safe)

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    numeric_col = "linear_outside_dilated_support_fraction_60_80_mlat"
    if numeric_col in summary_df.columns:
        vals = pd.to_numeric(summary_df[numeric_col], errors="coerce").dropna()
        if len(vals):
            print(f"Files with valid coverage metrics: {len(vals)}")
            print(f"Median interpolation-expansion fraction (60–80 MLAT): {vals.median():.4f}")
            print(f"Mean interpolation-expansion fraction (60–80 MLAT):   {vals.mean():.4f}")
            print(f"Maximum interpolation-expansion fraction:              {vals.max():.4f}")

    if not stress_df.empty:
        grouped = stress_df.groupby("gap_width_h")["bridge_fraction"].agg(["mean", "min", "max"])
        print("\nSynthetic-gap bridge fractions:")
        print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nOutputs:")
    print(f"  {summary_csv}")
    print(f"  {stress_csv}")
    print(f"  {output_dir / 'target_edr_metadata.json'}")
    print("  diagnostic_*.png and diagnostic_*.npz for manuscript target cases")
    print("\nPlease send me the complete console output plus the two diagnostic PNGs for Figure 6/7.")


if __name__ == "__main__":
    main()

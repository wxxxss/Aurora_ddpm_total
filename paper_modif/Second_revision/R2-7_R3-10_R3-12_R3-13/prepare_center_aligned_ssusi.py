#!/usr/bin/env python3
"""Rebuild Figure 6/7 SSUSI EDR inputs using *center-aligned* MLAT/MLT binning.

Corrects the edge-zero MLT bins in the first-revision preparation. Coordinates
are assigned to nearest grid centers: 80 northern-hemisphere MLAT centers
linspace(50,90,80); 96 cyclic MLT centers 0,0.25,...23.75 hours.

The native UT_N>0 support and finite native energy flux/coordinates determine
the actual observed mask. Averages all valid native measurements within each
target cell. Unsupported cells remain NaN, never interpolated. Saves the
same NPZ and metadata keys expected by first-revision recon and benchmark
scripts; does NOT edit earlier data.

Run (from repo root):
 python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/prepare_center_aligned_ssusi.py

Outputs into Second_revision/.../center_aligned_inputs/
  Figure6/Figure6_swath_case.npz, Figure6/Figure6_metadata.json
  Figure7/Figure7_swath_case.npz, Figure7/Figure7_metadata.json
  center_alignment_manifest.json

Requires: numpy, netCDF4, and original local SSUSI EDR files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_AUDIT_CSV = (
    ROOT / "paper_modif/First_revision/R2-1_R2-3/"
    "ssusi_swath_audit_v2/ssusi_swath_summary_v2.csv"
)
DEFAULT_OUTPUT = HERE / "center_aligned_inputs"
LEGACY_PREPARED = (
    ROOT / "paper_modif/R2-1_R2-3/swath_prepared_cases"
)
ARCHIVED_PREPARED = (
    ROOT / "paper_modif/First_revision/R2-1_R2-3/swath_prepared_cases"
)
LAT_CENTERS = np.linspace(50.0, 90.0, 80, dtype=np.float64)
MLT_CENTERS = np.arange(96, dtype=np.float64) * 0.25
LAT_STEP = float(LAT_CENTERS[1] - LAT_CENTERS[0])
MLT_STEP = 0.25

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def native_to_centered(flux, lat, mlt, ut):
    """Return flux_grid, observed mask, count grid, and audit counters.

    Tie behavior follows np.rint (ties to even) in the archived swath-v2 audit.
    This is nearest-center binning, unlike the original edge-zero MLT bins.
    """
    if not (flux.shape == lat.shape == mlt.shape == ut.shape):
        raise ValueError("Raw SSUSI variables do not have matching native shapes.")
    raw_valid = (
        np.isfinite(ut) & (ut > 0)
        & np.isfinite(flux) & np.isfinite(lat) & np.isfinite(mlt)
        & (lat >= 50.0) & (lat <= 90.0)
    )
    fv = flux[raw_valid].astype(np.float64)
    lv = lat[raw_valid].astype(np.float64)
    tv = np.mod(mlt[raw_valid].astype(np.float64), 24.0)

    lat_index = np.rint((lv - LAT_CENTERS[0]) / LAT_STEP).astype(np.int64)
    mlt_index = np.rint(tv / MLT_STEP).astype(np.int64) % 96
    inside = (lat_index >= 0) & (lat_index < 80)
    sums = np.zeros((80, 96), dtype=np.float64)
    counts = np.zeros((80, 96), dtype=np.int32)
    np.add.at(sums, (lat_index[inside], mlt_index[inside]), fv[inside])
    np.add.at(counts, (lat_index[inside], mlt_index[inside]), 1)
    mask = counts > 0
    out = np.full((80, 96), np.nan, dtype=np.float32)
    out[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    assert int(counts.sum()) == int(inside.sum())
    assert np.all(np.isnan(out[~mask]))
    return out, mask.astype(np.uint8), counts, {
        "raw_valid_samples": int(raw_valid.sum()),
        "assigned_native_samples": int(inside.sum()),
        "supported_cells": int(mask.sum()),
        "supported_fraction": float(mask.mean()),
        "cells_with_multiple_samples": int((counts > 1).sum()),
        "max_native_samples_per_cell": int(counts.max()),
        "mean_samples_per_supported_cell": float(counts[mask].mean()) if np.any(mask) else None,
        "nearly_at_mlt_half_center_count": int(np.count_nonzero(
            np.isclose(np.mod(tv / MLT_STEP, 1.0), 0.5, rtol=0, atol=1e-8)
        )),
    }

def read_raw(path: Path):
    with Dataset(str(path), "r") as ds:
        def array(key):
            # Convert to floating-point *before* filling masked integer EDR
            # arrays with NaN; otherwise .filled(np.nan) can raise TypeError.
            raw = np.ma.asarray(ds.variables[key][:], dtype=np.float64)
            return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64).squeeze()
        flux = array("ENERGY_FLUX_NORTH_MAP")
        lat = array("LATITUDE_GEOMAGNETIC_GRID_MAP")
        mlt = array("MLT_GRID_MAP")
        ut = array("UT_N")
        attrs = {}
        for key in ("STARTING_ORBIT_NUMBER", "STOPPING_ORBIT_NUMBER",
                    "DATA_DESCRIPTION", "DATA_VERSION"):
            if key in ds.ncattrs():
                value = ds.getncattr(key)
                attrs[key] = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    return flux, lat, mlt, ut, attrs

def archived_comparison(case: str, obs_mask: np.ndarray):
    for base in (ARCHIVED_PREPARED, LEGACY_PREPARED):
        p = base / case / f"{case}_swath_case.npz"
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            old = np.asarray(z["obs_mask"], dtype=bool)
        if old.shape != obs_mask.shape:
            raise ValueError(f"Old support mask shape not (80,96): {p}: {old.shape}")
        return {
            "old_source": str(p),
            "old_support_cells": int(old.sum()),
            "new_support_cells": int(obs_mask.sum()),
            "support_symmetric_difference_cells": int(np.sum(old ^ obs_mask.astype(bool))),
        }
    return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swath-summary", type=Path, default=DEFAULT_AUDIT_CSV)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true",
                   help="Explicitly replace an existing center-aligned output dataset")
    args = p.parse_args()
    if not args.swath_summary.is_file():
        raise FileNotFoundError(args.swath_summary)
    with args.swath_summary.open(encoding="utf-8-sig", newline="") as fh:
        archived = {r["case"]: r for r in csv.DictReader(fh)}

    report = {
        "grid_latitude_centers_deg": [float(LAT_CENTERS[0]), float(LAT_CENTERS[-1])],
        "latitude_center_step_deg": LAT_STEP,
        "grid_mlt_centers_h": [float(MLT_CENTERS[0]), float(MLT_CENTERS[-1])],
        "mlt_step_h": MLT_STEP,
        "mlt_periodicity_h": 24.0,
        "assignment_rule": "nearest grid center, np.rint tie-to-even, cyclic 96-bin MLT",
        "first_revision_inputs_untouched": True,
        "cases": {}
    }
    for case in ("Figure6", "Figure7"):
        if case not in archived:
            raise KeyError(f"Missing {case} from archived swath CSV")
        row = archived[case]
        raw_path = Path(row["source_file"])
        if not raw_path.is_file():
            raise FileNotFoundError(f"{case} original raw EDR absent: {raw_path}")
        dst = args.output_root / case
        npz_path = dst / f"{case}_swath_case.npz"
        metadata_path = dst / f"{case}_metadata.json"
        if (npz_path.exists() or metadata_path.exists()) and not args.overwrite:
            raise FileExistsError(f"{case} output exists at {dst}; add --overwrite only for an intentional repeat run")

        flux, lat, mlt, ut, attrs = read_raw(raw_path)
        grid, support, cnt, audit = native_to_centered(flux, lat, mlt, ut)
        expected = round(float(row["direct_target_support_fraction"]) * 7680)
        if audit["supported_cells"] != expected:
            raise AssertionError(
                f"{case}: expected {expected} center-support cells from prior swath-v2 audit, "
                f"but now found {audit['supported_cells']}; investigate before saving"
            )
        dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, flux_grid=grid, obs_mask=support,
                            count_grid=cnt, mlat_1d=LAT_CENTERS, mlt_1d=MLT_CENTERS)
        t0 = row["file_start_time"]
        t1 = row["file_stop_time"]
        meta = {
            "case_name": case,
            "nominal_manuscript_time": row["nominal_manuscript_time"],
            "matched_edr_file": str(raw_path),
            "source_edr_sha256": sha256(raw_path),
            "edr_midpoint": row["edr_mid_time"],
            "acquisition_start": t0,
            "acquisition_end": t1,
            "obs_fraction": audit["supported_fraction"],
            "mean_flux": float(np.nanmean(grid)),
            "max_flux": float(np.nanmax(grid)),
            "netcdf_metadata": attrs,
            "grid_assignment_method": report["assignment_rule"],
            "mlt_center_offset_corrected_from_first_revision": True,
        }
        metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        report["cases"][case] = {
            "source": str(raw_path),
            "npz_path": str(npz_path),
            "metadata_path": str(metadata_path),
            "source_sha256": meta["source_edr_sha256"],
            "audit": audit,
            "comparison_with_old": archived_comparison(case, support),
        }
        print(f"{case}: center-aligned native support {audit['supported_cells']}/7680; "
              f"outputs: {dst}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "center_alignment_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

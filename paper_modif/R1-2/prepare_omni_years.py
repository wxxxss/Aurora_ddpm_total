#!/usr/bin/env python3
"""Prepare hourly OMNI data for R1.2 multi-year robustness tests.

Reads local OMNI CDF files for 2001, 2005, and 2009, resolves common variable
aliases, converts fill/out-of-range values to NaN, regularizes to hourly cadence,
fills only short continuous gaps, and writes one structured NPY per year.

Output fields:
    utc, Bx, By, Bz, V, P, Kp, AE, SYM_H

The five model-conditioning fields are always [Bx, By, Bz, V, P]. Kp is kept
for the reviewer-requested Kp<=3 versus Kp>=4 robustness split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robustness_utils import apply_physical_range, normalize_kp, regularize_hourly

DEFAULT_CDF_ROOT = Path("/home/docker/data/ro-share/omni/omni_cdaweb/hourly")
DEFAULT_OUTPUT_ROOT = Path("/home/docker/data/private/AuroraData/omni_real_data/hourly_r1_2")
DEFAULT_YEARS = (2001, 2005, 2009)

ALIASES = {
    "Epoch": ("Epoch", "EPOCH", "epoch"),
    "Bx": ("BX_GSE", "BX_GSM", "Bx", "BX"),
    "By": ("BY_GSM", "BY_GSE", "By", "BY"),
    "Bz": ("BZ_GSM", "BZ_GSE", "Bz", "BZ"),
    "V": ("flow_speed", "V", "Speed", "speed", "SW_speed"),
    "P": ("Pressure", "pressure", "P", "P_dyn", "Pdyn"),
    "Kp": ("KP", "Kp", "kp", "KP_INDEX", "Kp_index", "KP1800"),
    "AE": ("AE_INDEX", "AE", "AE_index"),
    "SYM_H": ("SYM_H", "SYM-H", "SYMH", "SYM_H_INDEX"),
    "Vx": ("Vx", "VX_GSE", "VX_GSM"),
    "Vy": ("Vy", "VY_GSE", "VY_GSM"),
    "Vz": ("Vz", "VZ_GSE", "VZ_GSM"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdf-root", type=Path, default=DEFAULT_CDF_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    p.add_argument("--max-interp-hours", type=int, default=3)
    p.add_argument("--audit", type=Path, default=SCRIPT_DIR / "omni_preparation_audit.json")
    return p.parse_args()


def cdf_variable_names(cdf: Any) -> set[str]:
    info = cdf.cdf_info()
    return set(getattr(info, "rVariables", []) or []) | set(getattr(info, "zVariables", []) or [])


def resolve_alias(available: set[str], key: str) -> Optional[str]:
    for name in ALIASES[key]:
        if name in available:
            return name
    return None


def read_1d(cdf: Any, name: str) -> np.ndarray:
    x = np.asarray(cdf.varget(name)).squeeze()
    return x.reshape(-1)


def extract_cdf(path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        import cdflib
    except ImportError as exc:
        raise RuntimeError("cdflib is required. Install with: pip install cdflib") from exc

    cdf = cdflib.CDF(str(path))
    try:
        available = cdf_variable_names(cdf)
        epoch_name = resolve_alias(available, "Epoch")
        if epoch_name is None:
            raise KeyError(f"No Epoch variable in {path.name}; available={sorted(available)[:100]}")
        utc = pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget(epoch_name)))

        mapping: Dict[str, Any] = {"Epoch": epoch_name}
        data: Dict[str, Any] = {"utc": utc}
        for field in ("Bx", "By", "Bz", "P", "Kp", "AE", "SYM_H"):
            src = resolve_alias(available, field)
            mapping[field] = src
            data[field] = read_1d(cdf, src).astype(float) if src is not None else np.full(len(utc), np.nan)

        vsrc = resolve_alias(available, "V")
        if vsrc is not None:
            mapping["V"] = vsrc
            data["V"] = read_1d(cdf, vsrc).astype(float)
        else:
            vx, vy, vz = (resolve_alias(available, k) for k in ("Vx", "Vy", "Vz"))
            if all(x is not None for x in (vx, vy, vz)):
                mapping["V"] = [vx, vy, vz]
                xv, yv, zv = (read_1d(cdf, x).astype(float) for x in (vx, vy, vz))
                data["V"] = np.sqrt(xv*xv + yv*yv + zv*zv)
            else:
                mapping["V"] = None
                data["V"] = np.full(len(utc), np.nan)

        lengths = {k: len(np.asarray(v)) for k, v in data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Variable length mismatch in {path.name}: {lengths}")
        return pd.DataFrame(data), {"file": str(path), "mapping": mapping, "available_count": len(available)}
    finally:
        try:
            cdf.close()
        except Exception:
            pass


def discover_year_files(root: Path, year: int) -> list[Path]:
    year_dir = root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")
    files = sorted(year_dir.rglob("*.cdf"))
    if not files:
        raise FileNotFoundError(f"No CDF files found under {year_dir}")
    return files


def to_structured(df: pd.DataFrame) -> np.ndarray:
    dtype = [
        ("utc", "datetime64[ns]"),
        ("Bx", "<f4"), ("By", "<f4"), ("Bz", "<f4"),
        ("V", "<f4"), ("P", "<f4"), ("Kp", "<f4"),
        ("AE", "<f4"), ("SYM_H", "<f4"),
    ]
    arr = np.empty(len(df), dtype=dtype)
    arr["utc"] = pd.to_datetime(df["utc"]).to_numpy(dtype="datetime64[ns]")
    for name, _ in dtype[1:]:
        arr[name] = df[name].to_numpy(dtype=np.float32)
    return arr


def process_year(year: int, args: argparse.Namespace) -> tuple[pd.DataFrame, Dict[str, Any]]:
    files = discover_year_files(args.cdf_root, year)
    frames, file_audit = [], []
    print(f"\n{'='*88}\nYEAR {year}: {len(files)} CDF file(s)\n{'='*88}")
    for path in files:
        print(f"Reading {path}")
        frame, audit = extract_cdf(path)
        frames.append(frame)
        file_audit.append(audit)

    df = pd.concat(frames, ignore_index=True)
    df["utc"] = pd.to_datetime(df["utc"])
    df = df[df["utc"].dt.year == year].sort_values("utc").drop_duplicates("utc", keep="first")

    kp_scaled, kp_divisor = normalize_kp(df["Kp"].to_numpy(float))
    df["Kp"] = kp_scaled
    for field in ("Bx", "By", "Bz", "V", "P", "Kp", "AE", "SYM_H"):
        df[field] = apply_physical_range(df[field], field)

    raw_nan = {f: int(df[f].isna().sum()) for f in ("Bx", "By", "Bz", "V", "P", "Kp", "AE", "SYM_H")}
    hourly = regularize_hourly(df, year, max_interp_hours=args.max_interp_hours)
    hourly["year"] = year
    final_nan = {f: int(hourly[f].isna().sum()) for f in ("Bx", "By", "Bz", "V", "P", "Kp", "AE", "SYM_H")}

    out = args.output_root / str(year) / f"omni_{year}_hourly_r1_2.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, to_structured(hourly))

    valid5 = np.all(np.isfinite(hourly[["Bx", "By", "Bz", "V", "P"]].to_numpy(float)), axis=1)
    valid6 = valid5 & np.isfinite(hourly["Kp"].to_numpy(float))
    audit = {
        "year": year,
        "cdf_files": file_audit,
        "kp_divisor_applied": kp_divisor,
        "rows_hourly": int(len(hourly)),
        "raw_nan_counts": raw_nan,
        "final_nan_counts": final_nan,
        "valid_model_condition_rows": int(valid5.sum()),
        "valid_model_plus_kp_rows": int(valid6.sum()),
        "kp_le_3_rows": int(np.sum(valid6 & (hourly["Kp"].to_numpy(float) <= 3.0))),
        "kp_ge_4_rows": int(np.sum(valid6 & (hourly["Kp"].to_numpy(float) >= 4.0))),
        "output": str(out),
    }
    print(f"Saved {out}")
    print(f"Hourly rows={len(hourly)}, valid condition={valid5.sum()}, valid+Kp={valid6.sum()}")
    print(f"Kp<=3: {audit['kp_le_3_rows']} | Kp>=4: {audit['kp_ge_4_rows']} | Kp divisor={kp_divisor:g}")
    return hourly, audit


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_audit = {"cdf_root": str(args.cdf_root), "output_root": str(args.output_root), "years": {}}
    for year in args.years:
        _, audit = process_year(int(year), args)
        all_audit["years"][str(year)] = audit
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as f:
        json.dump(all_audit, f, indent=2, default=str)
    print(f"\nAudit JSON: {args.audit}")
    print("Please send the complete console output and omni_preparation_audit.json before the OVATION generation step.")


if __name__ == "__main__":
    main()

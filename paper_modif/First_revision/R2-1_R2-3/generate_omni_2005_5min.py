#!/usr/bin/env python3
"""Generate the 2005 5-min OMNI conditioning file used by the final DDPM.

Final model conditioning schema:
    [Bx, By, Bz, V, P]

Primary HRO2 CDF source variables:
    Epoch, BX_GSE, BY_GSM, BZ_GSM, flow_speed, Pressure

For compatibility with other OMNI CDF variants, velocity is resolved in this
order:
    flow_speed -> V -> sqrt(Vx**2 + Vy**2 + Vz**2)

The output preserves the native time axis and uses the historical processing
choice of linear interpolation for invalid/fill values so its row indexing
remains compatible with the existing 2005 OVATION array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/docker/data/ro-share/omni/omni_cdaweb/hro2_5min/2005/"
    "omni_hro2_5min_20050101_v01.cdf"
)
DEFAULT_OUTPUT = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2005/"
    "omni_20050101_5min.npy"
)
DEFAULT_AUDIT = Path(__file__).resolve().with_name("omni_2005_5min_audit.json")

MODEL_FIELDS = ("Bx", "By", "Bz", "V", "P")
STATIC_SOURCE_FIELDS = {
    "Bx": "BX_GSE",
    "By": "BY_GSM",
    "Bz": "BZ_GSM",
    "P": "Pressure",
}
VALID_RANGES = {
    "Bx": (-50.0, 50.0),
    "By": (-50.0, 50.0),
    "Bz": (-50.0, 50.0),
    "V": (200.0, 2000.0),
    "P": (0.05, 80.0),
}
UNITS = {
    "Bx": "nT (GSE)",
    "By": "nT (GSM)",
    "Bz": "nT (GSM)",
    "V": "km/s",
    "P": "nPa",
}
EVENT_TIMES = {
    "Figure6_EDR_midpoint": "2005-01-01T15:35:29.500000",
    "Figure7_EDR_midpoint": "2005-01-04T03:03:11",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--year", type=int, default=2005)
    return parser.parse_args()


def _max_consecutive_true(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def resolve_source_mapping(available: set[str]) -> Tuple[Dict[str, Any], str]:
    """Resolve CDF variables to the five conditioning variables used by the model.

    The OMNI HRO2 5-min product uses ``flow_speed`` for scalar solar-wind speed.
    Some older OMNI files expose ``V`` instead. If neither scalar field exists,
    the speed is computed from Vx, Vy, and Vz.
    """
    missing_static = [name for name in ("Epoch", *STATIC_SOURCE_FIELDS.values()) if name not in available]
    if missing_static:
        raise KeyError(
            f"CDF is missing required variables {sorted(missing_static)}. "
            f"Available variables include: {sorted(available)[:80]}"
        )

    mapping: Dict[str, Any] = dict(STATIC_SOURCE_FIELDS)
    if "flow_speed" in available:
        mapping["V"] = "flow_speed"
        velocity_mode = "direct_flow_speed"
    elif "V" in available:
        mapping["V"] = "V"
        velocity_mode = "direct_V"
    elif all(name in available for name in ("Vx", "Vy", "Vz")):
        mapping["V"] = ("Vx", "Vy", "Vz")
        velocity_mode = "vector_magnitude"
    else:
        raise KeyError(
            "CDF has no usable solar-wind speed variable. Expected 'flow_speed', 'V', "
            "or all of ['Vx', 'Vy', 'Vz']. "
            f"Available variables include: {sorted(available)[:80]}"
        )

    # Keep the final model order explicit for auditability.
    ordered = {field: mapping[field] for field in MODEL_FIELDS}
    return ordered, velocity_mode


def _read_cdf_1d(cdf: Any, name: str) -> np.ndarray:
    values = np.asarray(cdf.varget(name)).squeeze()
    if values.ndim != 1:
        values = values.reshape(-1)
    return values.astype(np.float64)


def extract_raw_dataframe(cdf_path: Path, year: int = 2005) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        import cdflib
    except ImportError as exc:
        raise RuntimeError("cdflib is required to read the OMNI CDF file.") from exc

    if not cdf_path.exists():
        raise FileNotFoundError(f"OMNI CDF not found: {cdf_path}")

    cdf = cdflib.CDF(str(cdf_path))
    try:
        info = cdf.cdf_info()
        available = set(info.rVariables) | set(info.zVariables)
        mapping, velocity_mode = resolve_source_mapping(available)

        epoch_raw = cdf.varget("Epoch")
        utc = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch_raw))
        data: Dict[str, Any] = {"utc": utc}

        for out_name in ("Bx", "By", "Bz", "P"):
            data[out_name] = _read_cdf_1d(cdf, mapping[out_name])

        velocity_source = mapping["V"]
        if isinstance(velocity_source, tuple):
            vx = _read_cdf_1d(cdf, velocity_source[0])
            vy = _read_cdf_1d(cdf, velocity_source[1])
            vz = _read_cdf_1d(cdf, velocity_source[2])
            data["V"] = np.sqrt(vx * vx + vy * vy + vz * vz)
        else:
            data["V"] = _read_cdf_1d(cdf, velocity_source)
    finally:
        try:
            cdf.close()
        except Exception:
            pass

    # Reorder columns to the final model schema immediately.
    df = pd.DataFrame(data)[["utc", *MODEL_FIELDS]]
    lengths = {name: len(df[name]) for name in df.columns}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"CDF variable lengths are inconsistent: {lengths}")

    df = df.loc[df["utc"].dt.year == year].copy()
    df = df.sort_values("utc").reset_index(drop=True)
    duplicate_count = int(df["utc"].duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates(subset="utc", keep="first").reset_index(drop=True)

    serializable_mapping = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in mapping.items()
    }
    source_audit = {
        "cdf_path": str(cdf_path),
        "model_to_source_mapping": serializable_mapping,
        "velocity_source_mode": velocity_mode,
        "rows_after_year_filter": int(len(df)),
        "duplicate_timestamps_removed": duplicate_count,
    }
    return df, source_audit


def clean_model_input_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = ["utc", *MODEL_FIELDS]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")

    clean = df[required].copy()
    clean["utc"] = pd.to_datetime(clean["utc"])

    invalid_counts: Dict[str, int] = {}
    max_gap_samples: Dict[str, int] = {}
    minmax_before: Dict[str, Dict[str, float | None]] = {}

    for field in MODEL_FIELDS:
        values = pd.to_numeric(clean[field], errors="coerce").astype(float)
        finite = np.isfinite(values.to_numpy())
        lo, hi = VALID_RANGES[field]
        valid = finite & (values.to_numpy() >= lo) & (values.to_numpy() <= hi)
        invalid = ~valid
        invalid_counts[field] = int(invalid.sum())
        max_gap_samples[field] = _max_consecutive_true(invalid)

        finite_values = values.to_numpy()[finite]
        minmax_before[field] = {
            "min": float(np.min(finite_values)) if finite_values.size else None,
            "max": float(np.max(finite_values)) if finite_values.size else None,
        }
        values.loc[invalid] = np.nan
        clean[field] = values

    # Preserve the original 5-min row indexing expected by the existing OVATION array.
    # This mirrors the legacy OMNI preparation choice: linearly fill invalid/fill
    # values and use both directions for boundary gaps.
    clean[list(MODEL_FIELDS)] = clean[list(MODEL_FIELDS)].interpolate(
        method="linear", axis=0, limit_direction="both"
    )

    remaining_nan = clean[list(MODEL_FIELDS)].isna().sum().astype(int).to_dict()
    if any(remaining_nan.values()):
        raise ValueError(f"NaNs remain after interpolation: {remaining_nan}")

    for field in MODEL_FIELDS:
        clean[field] = clean[field].astype(np.float32)

    audit = {
        "invalid_before_interpolation": invalid_counts,
        "max_consecutive_invalid_samples": max_gap_samples,
        "max_consecutive_invalid_minutes": {
            key: int(value * 5) for key, value in max_gap_samples.items()
        },
        "remaining_nan_after_interpolation": remaining_nan,
        "raw_finite_minmax_before_range_filter": minmax_before,
    }
    return clean, audit


def to_structured_array(df: pd.DataFrame) -> np.ndarray:
    dtype = [
        ("utc", "datetime64[ns]"),
        ("Bx", "<f4"),
        ("By", "<f4"),
        ("Bz", "<f4"),
        ("V", "<f4"),
        ("P", "<f4"),
    ]
    arr = np.empty(len(df), dtype=dtype)
    arr["utc"] = pd.to_datetime(df["utc"]).to_numpy(dtype="datetime64[ns]")
    for field in MODEL_FIELDS:
        arr[field] = df[field].to_numpy(dtype=np.float32)
    return arr


def cadence_audit(times: pd.Series) -> Dict[str, Any]:
    t = pd.to_datetime(times)
    if len(t) < 2:
        return {"samples": int(len(t))}
    dt_min = np.diff(t.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[s]").astype(float) / 60.0
    return {
        "samples": int(len(t)),
        "start": t.iloc[0].isoformat(),
        "end": t.iloc[-1].isoformat(),
        "median_cadence_min": float(np.median(dt_min)),
        "min_cadence_min": float(np.min(dt_min)),
        "max_cadence_min": float(np.max(dt_min)),
        "non_5min_intervals": int(np.sum(~np.isclose(dt_min, 5.0))),
        "expected_2005_samples_at_5min": 365 * 24 * 12,
    }


def event_audit(df: pd.DataFrame) -> Dict[str, Any]:
    times_ns = pd.to_datetime(df["utc"]).to_numpy(dtype="datetime64[ns]")
    times_i64 = times_ns.astype("int64")
    out: Dict[str, Any] = {}
    for label, timestamp in EVENT_TIMES.items():
        target = np.datetime64(pd.Timestamp(timestamp), "ns")
        idx = int(np.argmin(np.abs(times_i64 - target.astype("int64"))))
        matched = pd.Timestamp(times_ns[idx])
        target_pd = pd.Timestamp(timestamp)
        row = df.iloc[idx]
        out[label] = {
            "target_time": target_pd.isoformat(),
            "nearest_omni_time": matched.isoformat(),
            "absolute_time_offset_s": float(abs((matched - target_pd).total_seconds())),
            "index": idx,
            "Bx": float(row["Bx"]),
            "By": float(row["By"]),
            "Bz": float(row["Bz"]),
            "V": float(row["V"]),
            "P": float(row["P"]),
        }
    return out


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("GENERATE 2005 5-MIN OMNI CONDITIONING FILE")
    print("=" * 80)
    print(f"Input CDF:  {args.input}")
    print(f"Output NPY: {args.output}")
    print(f"Audit JSON: {args.audit}")
    print("Final-model fields: [Bx, By, Bz, V, P]")

    raw_df, source_info = extract_raw_dataframe(args.input, year=args.year)
    clean_df, cleaning_info = clean_model_input_dataframe(raw_df)
    array = to_structured_array(clean_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)

    audit: Dict[str, Any] = {
        "year": args.year,
        "input_cdf": str(args.input),
        "output_npy": str(args.output),
        "output_fields": list(array.dtype.names or ()),
        "units": UNITS,
        "valid_ranges": {k: list(v) for k, v in VALID_RANGES.items()},
        "source": source_info,
        "cleaning": cleaning_info,
        "cadence": cadence_audit(clean_df["utc"]),
        "events": event_audit(clean_df),
        "model_compatibility": {
            "final_condition_order": list(MODEL_FIELDS),
            "number_of_condition_variables": len(MODEL_FIELDS),
        },
    }

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\nCDF -> final model mapping")
    print(f"  Epoch        -> utc")
    for out_name in MODEL_FIELDS:
        source = source_info["model_to_source_mapping"][out_name]
        if isinstance(source, list):
            source_text = f"sqrt({source[0]}^2 + {source[1]}^2 + {source[2]}^2)"
        else:
            source_text = str(source)
        print(f"  {source_text:28s} -> {out_name}")
    print(f"  velocity mode: {source_info['velocity_source_mode']}")

    print("\nCleaning summary")
    for field in MODEL_FIELDS:
        nbad = cleaning_info["invalid_before_interpolation"][field]
        longest = cleaning_info["max_consecutive_invalid_minutes"][field]
        print(f"  {field:2s}: invalid/fill={nbad:6d}, longest gap={longest:5d} min")

    cad = audit["cadence"]
    print("\nCadence")
    print(f"  samples:             {cad['samples']}")
    print(f"  time range:          {cad['start']} -> {cad['end']}")
    print(f"  median cadence:      {cad['median_cadence_min']:.3f} min")
    print(f"  non-5min intervals:  {cad['non_5min_intervals']}")
    print(f"  expected full year:  {cad['expected_2005_samples_at_5min']}")

    print("\nReviewer-event OMNI conditions")
    for label, info in audit["events"].items():
        print(f"  {label}")
        print(
            f"    nearest={info['nearest_omni_time']}, offset={info['absolute_time_offset_s']:.1f} s, "
            f"Bx={info['Bx']:.3f}, By={info['By']:.3f}, Bz={info['Bz']:.3f} nT, "
            f"V={info['V']:.3f} km/s, P={info['P']:.3f} nPa"
        )

    print("\nSaved:")
    print(f"  {args.output}")
    print(f"  {args.audit}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit true 2006 OMNI HRO2 CDF epochs/schema, then optionally convert.

R2-8 / R3-1 independent-year real SSUSI validation. Reuses the established
first-revision OMNI cleaning method and [Bx,By,Bz,V,P] conditioning order,
but does NOT reuse 2005 Figure6/7 event audit or copy 2005 conditions.

Audit (no new NPY):
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_and_prepare_2006_omni.py --month 1

Convert ONLY if the above report says ready_for_conversion=true:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_and_prepare_2006_omni.py --month 1 --convert

Strict safeguards:
- raw CDF Epoch must cover exactly the chosen 2006 calendar month, on a
  unique, regular 5-minute grid, without silently dropping other-year rows;
- source CDF must contain real model input fields, with expected mapping;
- out-of-range/fill values are measured BEFORE interpolation; default
  conversion forbids any channel with >5% invalid data or >60 minutes
  consecutive missing values. Do not lift these thresholds silently.
- conversion refuses to overwrite an existing NPY, verifies round-trip
  UTC and field schema, and writes a small JSON audit.
- no model inference or satellite target flux is read.
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
LEGACY=REPO/"paper_modif/First_revision/R2-1_R2-3/generate_omni_2005_5min.py"
CDF_ROOT=Path("/home/docker/data/ro-share/omni/omni_cdaweb/hro2_5min")
OUT_ROOT=Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min")
MODEL_FIELDS=("Bx","By","Bz","V","P")

def load_legacy():
    if not LEGACY.is_file():
        raise FileNotFoundError(f"Previous generation module missing: {LEGACY}")
    spec=importlib.util.spec_from_file_location("existing_omni_hro2_preparation",LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import OMNI preparation utilities from {LEGACY}")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def month_grid(year,month):
    start=np.datetime64(datetime(year,month,1),"ns")
    return start+np.arange(calendar.monthrange(year,month)[1]*24*12)*np.timedelta64(5,"m")

def validate_calendar_grid(raw_times,year,month):
    actual=np.asarray(raw_times,dtype="datetime64[ns]").reshape(-1)
    expected=month_grid(year,month)
    result={"expected_utc_count":int(len(expected)),
            "raw_utc_count":int(len(actual)),
            "utc_first":str(actual[0]) if len(actual) else None,
            "utc_last":str(actual[-1]) if len(actual) else None,
            "same_length":bool(len(actual)==len(expected)),
            "exact_regular_complete_5min_utc":bool(
                len(actual)==len(expected) and np.array_equal(actual,expected))}
    if len(actual)>1:
        step=np.diff(actual).astype("timedelta64[s]").astype(np.int64)
        result["n_irregular_or_duplicate_5min_steps"]=int((step!=300).sum())
        result["n_non_increasing_timestamps"]=int((step<=0).sum())
    else:
        result["n_irregular_or_duplicate_5min_steps"]=None
        result["n_non_increasing_timestamps"]=None
    return result

def get_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()

def audit_cdf(path,year,month,max_missing_fraction,max_gap_minutes,legacy):
    df,source=legacy.extract_raw_dataframe(path,year=year)
    if set(MODEL_FIELDS)-set(df.columns):
        raise ValueError("The raw OMNI source does not expose all five model-conditioning fields")
    grid=validate_calendar_grid(df["utc"].to_numpy(),year,month)
    # Examine raw, noninterpolated CDF values using the same strict physical
    # ranges as in the established 2005 input preprocessing script.
    quality={}
    all_ok=True
    for field in MODEL_FIELDS:
        values=pd.to_numeric(df[field],errors="coerce").to_numpy(dtype=float)
        lo,hi=legacy.VALID_RANGES[field]
        good=np.isfinite(values)&(values>=lo)&(values<=hi)
        missing=~good
        max_samples=legacy._max_consecutive_true(missing)
        fraction=float(missing.mean()) if len(missing) else 1.0
        quality[field]={
          "raw_invalid_samples":int(missing.sum()),
          "raw_invalid_fraction":fraction,
          "longest_consecutive_invalid_minutes":int(max_samples*5),
          "safe_for_linear_interpolation":bool(
            fraction<=max_missing_fraction and max_samples*5<=max_gap_minutes),
          "finite_valid_min":float(np.min(values[good])) if np.any(good) else None,
          "finite_valid_max":float(np.max(values[good])) if np.any(good) else None,
        }
        all_ok &= quality[field]["safe_for_linear_interpolation"]
    valid=bool(grid["exact_regular_complete_5min_utc"] and all_ok)
    report={
      "source_cdf":str(path),"source_cdf_bytes":path.stat().st_size,
      "source_cdf_sha256":get_sha256(path),
      "year":year,"month":month,
      "model_input_order":list(MODEL_FIELDS),
      "cdf_to_model_field_mapping":source["model_to_source_mapping"],
      "velocity_source_mode":source["velocity_source_mode"],
      "calendar_time_grid":grid,
      "max_missing_fraction_allowed":max_missing_fraction,
      "max_consecutive_bad_minutes_allowed":max_gap_minutes,
      "raw_input_quality":quality,
      "ready_for_conversion":valid,
      "source_covariance_limit":"This checks 2006 OMNI observations and interpolation limits, not the true provenance or calibration of the existing OVATION monthly maps.",
    }
    return df,report

def write_npy_safely(path,array):
    """Do not clobber historical NPYs; use same-directory temporary file."""
    path=Path(path)
    if path.exists():raise FileExistsError(f"Refusing to overwrite original file: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp_name=None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.stem+"_tmp_",suffix=".npy",
                                         dir=path.parent,delete=False) as fp:
            tmp_name=fp.name
            np.save(fp,array,allow_pickle=False)
        if path.exists():raise FileExistsError(f"Refusing to replace existing file: {path}")
        os.rename(tmp_name,path)
        tmp_name=None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year",type=int,default=2006)
    p.add_argument("--month",type=int,default=1)
    p.add_argument("--cdf",type=Path,default=None)
    p.add_argument("--output-npy",type=Path,default=None)
    p.add_argument("--report",type=Path,default=None)
    p.add_argument("--max-invalid-fraction",type=float,default=.05)
    p.add_argument("--max-invalid-gap-minutes",type=int,default=60)
    p.add_argument("--convert",action="store_true",
                   help="Generate year-specific verified 5-min OMNI .npy after audit")
    args=p.parse_args()
    if not 1<=args.month<=12:raise ValueError("--month must be 1..12")
    if not 0<=args.max_invalid_fraction<=1 or args.max_invalid_gap_minutes<0:
        raise ValueError("Invalid interpolation safety limits")
    cdf=args.cdf or CDF_ROOT/str(args.year)/f"omni_hro2_5min_{args.year}{args.month:02d}01_v01.cdf"
    output=args.output_npy or OUT_ROOT/str(args.year)/f"omni_{args.year}{args.month:02d}01_5min.npy"
    report_path=args.report or HERE/f"omni_{args.year}{args.month:02d}_cdf_audit.json"
    report={"source_cdf":str(cdf),"output_npy":str(output),
            "requested_convert":bool(args.convert),"error":None,
            "conversion_successful":False,"ready_for_conversion":False}
    try:
        if not cdf.is_file() or cdf.stat().st_size==0:
            raise FileNotFoundError(f"Missing or empty source CDF: {cdf}")
        legacy=load_legacy()
        df,detail=audit_cdf(cdf,args.year,args.month,args.max_invalid_fraction,
                            args.max_invalid_gap_minutes,legacy)
        report.update(detail)
        if args.convert:
            if not report["ready_for_conversion"]:
                raise ValueError("Unsafe or incomplete 5-min 2006 CDF; conversion refused. See raw_input_quality and calendar_time_grid.")
            clean,cleaning=legacy.clean_model_input_dataframe(df)
            arr=legacy.to_structured_array(clean)
            if not np.array_equal(arr["utc"],month_grid(args.year,args.month)):
                raise AssertionError("Cleaned 5-minute OMNI timestamps mismatch month UTC grid")
            if tuple(arr.dtype.names)!=("utc",*MODEL_FIELDS):
                raise AssertionError(f"Wrong OMNI final dtype: {arr.dtype.names}")
            if not np.all(np.isfinite(np.column_stack([arr[k] for k in MODEL_FIELDS]))):
                raise AssertionError("Nonfinite model-conditioning values after cleaning")
            report["cleaning"]=cleaning
            write_npy_safely(output,arr)
            check=np.load(output,mmap_mode="r",allow_pickle=False)
            if not np.array_equal(check["utc"],arr["utc"]):
                raise AssertionError("Round-trip written timestamps differ")
            report["conversion_successful"]=True
            report["output_npy_bytes"]=output.stat().st_size
            report["output_npy_sha256"]=get_sha256(output)
    except (EOFError,ValueError,KeyError,OSError,ImportError,TypeError,
            AssertionError,FileNotFoundError,MemoryError) as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"source_cdf":report["source_cdf"],
      "source_map":report.get("cdf_to_model_field_mapping"),
      "calendar_grid":report.get("calendar_time_grid"),
      "ready_for_conversion":report.get("ready_for_conversion"),
      "quality":report.get("raw_input_quality"),
      "conversion_successful":report["conversion_successful"],
      "error":report.get("error"),"output":str(output),
      "report":str(report_path)},indent=2,ensure_ascii=False))
    if report["error"] or (args.convert and not report["conversion_successful"]):
        raise SystemExit(2)

if __name__=="__main__":main()

#!/usr/bin/env python3
"""Prepare a 2006 5-minute OMNI array WITHOUT filling its 825-minute outage.

This is a deliberately separate, more restrictive *event-qualified* path
than the existing whole-month --convert, which must continue to refuse
any calendar-wide gap >60 minutes.

Default: audit only, NO output NPY. It requires all preselected F16/SSUSI
acquisitions to have 100% raw-valid [Bx,By,Bz,V,P] records inside their
acquisition intervals, independently of the image/auroral flux.

With --write, only missing runs of up to 12 samples (60 min), BRACKETED by
valid values on both ends, may be linearly interpolated. Long gaps are
explicitly kept NaN in the saved month. Matching EDR intervals are checked
again against the original source, so the downstream per-EDR solar averages
can never rely on long-gap infilling. The resulting product MUST be passed
with explicit --omni to both third-batch preflight and benchmark.

Commands:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_2006_gap_preserving_omni.py --write

NEVER treat this output as a fully observed month; do not train against it.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from audit_and_prepare_2006_omni import (
    HERE, CDF_ROOT, OUT_ROOT, MODEL_FIELDS, load_legacy,
    month_grid, validate_calendar_grid, write_npy_safely, get_sha256
)
from audit_2006_omni_event_coverage import run as audit_events

DEFAULT_CDF=CDF_ROOT/"2006/omni_hro2_5min_20060101_v01.cdf"
DEFAULT_CATALOG=HERE/"real_2006_01_inputs/catalog_preview.json"
DEFAULT_OUT=OUT_ROOT/"2006/omni_20060101_5min_gap_preserving.npy"

def clean_short_bracketed(values,lo,hi,max_run_samples=12):
    """Fill only interior invalid runs of <= max_run_samples; keep long ones NaN."""
    v=np.asarray(values,dtype=np.float64).reshape(-1).copy()
    valid=np.isfinite(v)&(v>=lo)&(v<=hi)
    v[~valid]=np.nan
    bad=np.flatnonzero(~valid)
    nshort=0
    if bad.size:
        boundaries=np.flatnonzero(np.diff(bad)>1)
        starts=np.r_[0,boundaries+1]
        ends=np.r_[boundaries,bad.size-1]
        for first,last in zip(starts,ends):
            a=int(bad[first]);b=int(bad[last])
            n=b-a+1
            if n<=max_run_samples and a>0 and b<len(v)-1 and valid[a-1] and valid[b+1]:
                v[a:b+1]=np.linspace(v[a-1],v[b+1],n+2)[1:-1]
                nshort+=n
    return v.astype(np.float32),{
       "raw_invalid":int((~valid).sum()),
       "short_gap_interpolated_samples":nshort,
       "samples_remain_nan":int(np.isnan(v).sum()),
       "long_or_unbracketed_gaps_preserved":bool(np.isnan(v).any())
    }

def process(df,catalog,legacy,max_short_gap_minutes=60):
    if max_short_gap_minutes<0 or max_short_gap_minutes%5:
        raise ValueError("Only nonnegative integer multiples of 5 minutes allowed")
    if max_short_gap_minutes>60:
        raise ValueError("This protocol NEVER interpolates more than 60 min gaps")
    qa=audit_events(catalog,df,legacy)
    if not qa["all_preselected_acquisitions_raw_valid"]:
        raise ValueError("Some preselected EDR intervals contain invalid raw OMNI. "
                         "Do not generate data or start inference until a prospectively "
                         "quality-filtered cohort is selected; see event coverage audit.")
    n=len(df)
    clean=pd.DataFrame({"utc":pd.to_datetime(month_grid(2006,1))})
    fields={}
    for f in MODEL_FIELDS:
        lo,hi=legacy.VALID_RANGES[f]
        clean[f],fields[f]=clean_short_bracketed(
            np.asarray(df[f]),lo,hi,max_short_gap_minutes//5)
    arr=legacy.to_structured_array(clean)
    if tuple(arr.dtype.names)!=("utc",*MODEL_FIELDS):
        raise AssertionError("Wrong output schema")
    if n!=len(arr) or not np.array_equal(arr["utc"],month_grid(2006,1)):
        raise AssertionError("Post-preparation UTC index differs from raw CDF")
    t=arr["utc"]
    for case in catalog["cases"]:
        a=np.datetime64(case["acquisition_start"],"ns")
        b=np.datetime64(case["acquisition_end"],"ns")
        inside=(t>=a)&(t<=b)
        if not np.any(inside):raise ValueError(f"No solar records inside {case['case_id']}")
        for field in MODEL_FIELDS:
            saved=arr[field][inside].astype(float)
            raw=np.asarray(df[field],dtype=float)[inside]
            if not np.all(np.isfinite(saved)):
                raise AssertionError(f"NaNs inside known-valid EDR {case['case_id']} / {field}")
            if not np.allclose(saved,raw,atol=2e-5,rtol=1e-6):
                raise AssertionError(f"Known EDR solar wind was changed {case['case_id']} / {field}")
    return arr,fields,qa

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdf",type=Path,default=DEFAULT_CDF)
    p.add_argument("--catalog",type=Path,default=DEFAULT_CATALOG)
    p.add_argument("--output-npy",type=Path,default=DEFAULT_OUT)
    p.add_argument("--output-report",type=Path,default=HERE/"omni_200601_gap_preserving_audit.json")
    p.add_argument("--write",action="store_true")
    args=p.parse_args()
    audit={"cdf":str(args.cdf),"catalog":str(args.catalog),
           "output_npy":str(args.output_npy),
           "output_report":str(args.output_report),
           "requested_write":args.write,
           "ready_for_event_qualified_write":False,"saved":False,"error":None}
    try:
        catalog=json.loads(args.catalog.read_text(encoding="utf-8"))
        legacy=load_legacy()
        df,meta=legacy.extract_raw_dataframe(args.cdf,year=2006)
        grid=validate_calendar_grid(df["utc"].to_numpy(),2006,1)
        if not grid["exact_regular_complete_5min_utc"]:
            raise ValueError("Source has incomplete/irregular calendar UTC")
        arr,fields,event_qa=process(df,catalog,legacy)
        audit.update(source_cdf_sha256=get_sha256(args.cdf),
                     calendar_time_grid=grid,
                     source_mapping=meta["model_to_source_mapping"],
                     fields=fields,
                     n_preselected_raw_valid_events=event_qa["n_fully_raw_valid_acquisitions"],
                     n_preselected_events=event_qa["n_catalog_acquisitions"],
                     n_source_calendar_samples=len(arr),
                     n_unresolved_any_field_samples=int(sum(
                       int(not all(np.isfinite(arr[f][i]) for f in MODEL_FIELDS))
                       for i in range(len(arr)))),
                     event_input_coverage=event_qa["cases"],
                     ready_for_event_qualified_write=True)
        if args.write:
            write_npy_safely(args.output_npy,arr)
            # The on-disk source is not represented as complete monthly
            # conditioning; NaN gaps intentionally remain NaN.
            check=np.load(args.output_npy,mmap_mode="r",allow_pickle=False)
            if not np.array_equal(check["utc"],arr["utc"]):
                raise AssertionError("Roundtrip UTC verification failed")
            audit.update(saved=True,output_npy_sha256=get_sha256(args.output_npy),
                         output_npy_bytes=args.output_npy.stat().st_size)
    except (EOFError,ValueError,OSError,ImportError,AssertionError,
            KeyError,FileNotFoundError,TypeError) as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
    args.output_report.parent.mkdir(parents=True,exist_ok=True)
    args.output_report.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:audit.get(k) for k in (
        "n_preselected_events","n_preselected_raw_valid_events",
        "n_unresolved_any_field_samples","ready_for_event_qualified_write",
        "saved","error","output_npy","output_npy_sha256","output_report")},indent=2))
    if audit["error"]:
        raise SystemExit(2)

if __name__=="__main__":
    main()

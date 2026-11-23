#!/usr/bin/env python3
"""No-inference SSUSI EDR-by-EDR audit of 2006 January *raw* OMNI validity.

The OMNI CDF has a complete 5-min UTC axis but 825-minute continuous
invalid runs. The monthly NPY conversion was RIGHTLY refused; NEVER increase
the maximum interpolation gap to 825 minutes.

This script compares previously time-selected, flux-blind SSUSI acquisition
windows to the original, unfilled CDF. It does not read any SSUSI flux,
alter a prior catalog, generate NPYs, or run a DDPM.

From repository root:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_omni_event_coverage.py

Default requires 100% RAW VALID values in [Bx,By,Bz,V,P] within each EDR
interval. A separate summary shows missing CDF intervals and valid acquisition
counts, so any changes to sampling design must occur BEFORE model inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from audit_and_prepare_2006_omni import (
    CDF_ROOT, HERE, MODEL_FIELDS, audit_cdf, load_legacy, month_grid
)

DEFAULT_CATALOG=HERE/"real_2006_01_inputs/catalog_preview.json"
DEFAULT_CDF=CDF_ROOT/"2006/omni_hro2_5min_20060101_v01.cdf"

def calendar_validity(df,year,month,ranges):
    """Strict raw range and finiteness; zero filling or interpolation prohibited."""
    times=pd.to_datetime(df["utc"]).values.astype("datetime64[ns]")
    expected=month_grid(year,month)
    if not np.array_equal(times,expected):
        raise ValueError("Source CDF is NOT a complete exact 5-minute calendar grid")
    mask=np.ones((len(times),len(MODEL_FIELDS)),dtype=bool)
    for i,field in enumerate(MODEL_FIELDS):
        v=np.asarray(pd.to_numeric(df[field],errors="coerce"),dtype=float)
        lo,hi=ranges[field]
        mask[:,i]=np.isfinite(v)&(v>=lo)&(v<=hi)
    return times,mask

def audit_acquisition(record,times,raw_valid):
    a=np.datetime64(record["acquisition_start"],"ns")
    b=np.datetime64(record["acquisition_end"],"ns")
    if b<=a:raise ValueError(f"{record['case_id']}: acquisition has negative duration")
    include=(times>=a)&(times<=b)
    index=np.flatnonzero(include)
    if not len(index):raise ValueError(f"{record['case_id']}: no OMNI UTC samples inside acquisition")
    rows=raw_valid[index]
    fail_counts=(~rows).sum(axis=0)
    invalid_any=np.flatnonzero(~rows.all(axis=1))
    minutes=(b-a)/np.timedelta64(1,"m")
    # The monthly source uses regular 5-minute bins. The max interval
    # between an EDR endpoint and an interior OMNI sample should be <=5m.
    first_lag=float((times[index[0]]-a)/np.timedelta64(1,"m"))
    last_lag=float((b-times[index[-1]])/np.timedelta64(1,"m"))
    enough=bool(len(index)>=15 and first_lag<=5 and last_lag<=5 and minutes<240)
    return {
        "case_id":record["case_id"],
        "acquisition_start":record["acquisition_start"],
        "acquisition_end":record["acquisition_end"],
        "n_omni_5min_records_inside":int(len(index)),
        "n_any_field_invalid_inside":int(len(invalid_any)),
        "n_invalid_by_field":{field:int(fail_counts[j]) for j,field in enumerate(MODEL_FIELDS)},
        "first_invalid_utc":[str(times[index[j]].astype("datetime64[s]")) for j in invalid_any[:8]],
        "endpoints_bracketed_by_5min_grid":bool(enough),
        "raw_fully_valid_acquisition":bool(enough and rows.all()),
        "strict_qualified_for_source_clean_test":bool(enough and rows.all()),
    }

def run(catalog,df,legacy,source_audit=None):
    if catalog.get("month")!="2006-01":
        raise ValueError("Expected a prospectively selected January 2006 catalog")
    rows=catalog.get("cases",[])
    if not rows or len({r["case_id"] for r in rows})!=len(rows):
        raise ValueError("No cases or duplicated cases in EDR catalog")
    times,valid=calendar_validity(df,2006,1,legacy.VALID_RANGES)
    records=[audit_acquisition(r,times,valid) for r in rows]
    n_good=sum(int(v["raw_fully_valid_acquisition"]) for v in records)
    return {
       "protocol":"Prior time-selected, flux-blind F16 Jan 2006 orbit cohort; audit only the independent 2006 original raw OMNI CDF",
       "source_audit":source_audit,
       "n_catalog_acquisitions":len(records),
       "n_fully_raw_valid_acquisitions":n_good,
       "all_preselected_acquisitions_raw_valid":n_good==len(records),
       "raw_long_gap_exists_outside_or_within_orbits":bool((~valid.all(axis=1)).any()),
       "raw_invalid_total_5min_records_any_field":int((~valid.all(axis=1)).sum()),
       "n_raw_valid_5min_records_in_calendar":int(valid.all(axis=1).sum()),
       "cases":records,
       "next_action_if_any_fail":"Do not relax 60-minute limit or infer that finite nanmean is valid. Select a new cohort using ONLY acquisition date, native support availability, and prespecified OMNI source quality; do not select on SSUSI flux or reconstruction performance.",
       "limits":["A raw-valid EDR may be affected by solar wind transit-delay assumptions, not tested here.",
                 "A complete UTC calendar grid does not imply valid physical solar wind values.",
                 "The check contains no DDPM reconstruction and does not resolve training checkpoint lineage."]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",type=Path,default=DEFAULT_CATALOG)
    p.add_argument("--cdf",type=Path,default=DEFAULT_CDF)
    p.add_argument("--output",type=Path,default=HERE/"omni_200601_event_coverage_audit.json")
    args=p.parse_args()
    catalog=json.loads(args.catalog.read_text(encoding="utf-8"))
    legacy=load_legacy()
    df,src=legacy.extract_raw_dataframe(args.cdf,year=2006)
    out=run(catalog,df,legacy,src)
    out["source_cdf"]=str(args.cdf)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "n_preselected_EDRs":out["n_catalog_acquisitions"],
        "n_raw_clean_EDRs":out["n_fully_raw_valid_acquisitions"],
        "all_twelve_clean":out["all_preselected_acquisitions_raw_valid"],
        "raw_invalid_5min_ticks":out["raw_invalid_total_5min_records_any_field"],
        "unqualified_orbits":[{"case":r["case_id"],
                               "bad_records":r["n_any_field_invalid_inside"]}
                              for r in out["cases"] if not r["raw_fully_valid_acquisition"]],
        "output":str(args.output)},indent=2))

if __name__=="__main__":
    main()

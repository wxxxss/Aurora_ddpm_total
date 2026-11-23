#!/usr/bin/env python3
"""Audit independent-2006 OMNI solar normalization against stored 1996 bounds.

No model inference, no checkpoint loading, no EDR flux reading. This reports
whether each source-qualified January-2006 SSUSI EDR's five-field *raw
OMNI interval mean* would be clipped by the SAME [0,1] conditioning
normalization used by the conditional DDPM (legacy normalize_solar).

IMPORTANT: This is a domain-shift diagnostic, NOT a new post hoc event
selection rule. Run on ALL 12 preselected EDRs; do not omit extreme drivers,
renormalize on 2006 targets or retune the saved model to improve its scores.

 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_solar_normalization.py
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIELDS=("Bx","By","Bz","V","P")
DEFAULT_MANIFEST=HERE/"real_2006_01_omni_qualified_inputs/manifest.json"
DEFAULT_OMNI=Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy")
DEFAULT_NORM=ROOT/"ckpt/cond/norm_params.pkl"

def analyze(manifest,omni,norm):
    if manifest.get("month")!="2006-01":
        raise ValueError("Conditioning range audit requires January 2006 cohort")
    cases=manifest.get("cases",[])
    if len(cases)!=12 or len({x["case_id"] for x in cases})!=12:
        raise ValueError("Expected exactly the unchanged twelve source-qualified EDRs")
    if omni.dtype.names is None or not {"utc",*FIELDS}.issubset(omni.dtype.names):
        raise ValueError("OMNI must contain utc,Bx,By,Bz,V,P")
    if "omni" not in norm or any(k not in norm["omni"] for k in ("min","range")):
        raise KeyError("Conditional model normalization must contain omni.min and omni.range")
    lo=np.asarray(norm["omni"]["min"],dtype=np.float64).reshape(-1)
    span=np.asarray(norm["omni"]["range"],dtype=np.float64).reshape(-1)
    if lo.shape!=(5,) or span.shape!=(5,) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(span)) or np.any(span<=0):
        raise ValueError("Expected finite strictly positive 5-feature training normalization spans")
    hi=lo+span
    time=np.asarray(omni["utc"],dtype="datetime64[ns]")
    if len(time)!=31*24*12 or time[0]!=np.datetime64("2006-01-01T00:00:00","ns") or time[-1]!=np.datetime64("2006-01-31T23:55:00","ns") or not np.all(np.diff(time)==np.timedelta64(5,"m")):
        raise ValueError("Input OMNI must have complete January 2006 5-min UTC grid")
    records=[]
    for case in cases:
        a=np.datetime64(case["acquisition_start"],"ns")
        b=np.datetime64(case["acquisition_end"],"ns")
        selected=(time>=a)&(time<=b)
        values=np.column_stack([np.asarray(omni[field][selected],dtype=np.float64) for field in FIELDS])
        if len(values)<15:
            raise ValueError(f"{case['case_id']}: fewer than 15 conditioning records")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{case['case_id']}: nonfinite conditioning samples; abort, never nanmean")
        mean=values.mean(axis=0)
        scaled=(mean-lo)/span
        frac=np.mean((values<lo)|(values>hi),axis=0)
        record={
            "case_id":case["case_id"],
            "orbit":case.get("orbit"),
            "start":case["acquisition_start"],
            "end":case["acquisition_end"],
            "n_raw_valid_5min_records":int(len(values)),
            "interval_mean_raw":{k:float(mean[i]) for i,k in enumerate(FIELDS)},
            "interval_mean_before_clipping":{k:float(scaled[i]) for i,k in enumerate(FIELDS)},
            "interval_mean_would_clip_low":[FIELDS[i] for i in range(5) if scaled[i]<0],
            "interval_mean_would_clip_high":[FIELDS[i] for i in range(5) if scaled[i]>1],
            "fraction_of_individual_raw_records_outside_normalization_bounds":{
                k:float(frac[i]) for i,k in enumerate(FIELDS)
            },
        }
        record["any_interval_mean_clipping"]=bool(np.any((scaled<0)|(scaled>1)))
        records.append(record)
    return {
       "n_acquisitions":len(records),
       "year_month":"2006-01",
       "solar_field_order":list(FIELDS),
       "stored_training_normalization":{
           k:{"min":float(lo[i]),"max":float(hi[i]),"range":float(span[i])}
           for i,k in enumerate(FIELDS)},
       "n_acquisitions_with_any_interval_mean_clipping":sum(int(r["any_interval_mean_clipping"]) for r in records),
       "n_acquisitions_with_interval_mean_clipped_by_field":{
           k:sum(int(k in r["interval_mean_would_clip_low"] or
                     k in r["interval_mean_would_clip_high"]) for r in records)
           for k in FIELDS},
       "acquisitions":records,
       "policy":"Do not exclude or retrain based on these post-selection 2006 clipping results; this is a full-cohort external-validation domain-shift diagnostic.",
       "limitations":[
         "Stored normalization bounds are the inference scaling range; they do not prove the full 1996 input distribution or original checkpoint lineage.",
         "Only source-qualified January 2006 F16 acquisitions are assessed, not every 2006 day or a solar cycle.",
         "A mean clipped at 0/1 means some potentially informative forcing variation is lost; it is not by itself a model-error or causal Kp estimate.",
       ],
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    p.add_argument("--omni",type=Path,default=DEFAULT_OMNI)
    p.add_argument("--conditional-norm",type=Path,default=DEFAULT_NORM)
    p.add_argument("--output",type=Path,default=HERE/"real_2006_01_solar_conditioning_range_audit.json")
    args=p.parse_args()
    catalog=json.loads(args.manifest.read_text(encoding="utf-8"))
    omni=np.load(args.omni,allow_pickle=False)
    # Trusted, local author-produced normalizer only. Never fetch or
    # deserialize third-party pickle files.
    with args.conditional_norm.open("rb") as src:
        norm=pickle.load(src)
    result=analyze(catalog,omni,norm)
    result["source_manifest"]=str(args.manifest)
    result["source_omni"]=str(args.omni)
    result["source_normalizer"]=str(args.conditional_norm)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
       "n_acquisitions":result["n_acquisitions"],
       "n_acquisitions_with_any_interval_mean_clipping":result["n_acquisitions_with_any_interval_mean_clipping"],
       "n_clipped_by_field":result["n_acquisitions_with_interval_mean_clipped_by_field"],
       "output":str(args.output)},indent=2))
if __name__=="__main__":
    main()

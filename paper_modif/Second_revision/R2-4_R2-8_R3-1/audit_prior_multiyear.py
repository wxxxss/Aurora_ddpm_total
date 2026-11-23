#!/usr/bin/env python3
"""Reassess FIRST-round 2001/2005/2009 synthetic OVATION experiments by Kp.

No DDPM inference. Reads archived per-sample CSV and preserves exact target
provenance: synthetic OVATION maps, NOT independent real-satellite truth.

Primary objective: give an honest count of Kp>5 cases (the reviewer asked
Kp>5, not merely Kp>=4), check the number of distinct acquisition timestamps,
days and nominal 3-h Kp intervals, and distinguish synthetic robustness from
real-satellite generalization.

From repository root:
  python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_prior_multiyear.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEFAULT=ROOT/"paper_modif/First_revision/R1-2/r1_2_results/metrics_per_sample.csv"
METHODS=("conditional","unconditional","interpolation")
BINS=(("Kp<=3", lambda kp: kp<=3),
      ("3<Kp<=5", lambda kp:(kp>3)&(kp<=5)),
      ("Kp>5",lambda kp:kp>5))

def audit(df):
    required={"sample_id","utc","year","kp","method","rmse","mae"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(required-set(df.columns))}")
    if df[["sample_id","method"]].duplicated().any():
        raise ValueError("Repeated model evaluation on same sample")
    expected=set(METHODS)
    by_sample=df.groupby("sample_id")
    for sid,part in by_sample:
        if set(part.method)!=expected or len(part)!=len(expected):
            raise ValueError(f"sample {sid} lacks complete method triplet")
        if part[["utc","year","kp"]].nunique().max()!=1:
            raise ValueError(f"sample {sid} has inconsistent provenance")
    work=df.copy()
    work["utc"]=pd.to_datetime(work["utc"])
    work["year"]=work["year"].astype(int)
    work["kp"]=pd.to_numeric(work["kp"],errors="raise")
    work["kp_3h_interval"]=work["utc"].dt.floor("3h")
    work["day"]=work["utc"].dt.strftime("%Y-%m-%d")
    if not work.kp.between(0,9).all():raise ValueError("Unphysical Kp labels")
    out={"source_domain":"Synthetic OVATION Prime-generated targets; NOT independent measured satellite energy flux.",
         "n_unique_target_times":int(work["sample_id"].nunique()),
         "n_method_rows":int(len(work)),
         "years":{},
         "activity_groups":{},
         "interpretation":[
           "First-round Kp>=4 grouping is NOT the strict Kp>5 storm bin requested by the reviewer.",
           "This result measures the ability to reproduce the synthetic OVATION target generator; excellent scores do not prove independent satellite reconstruction.",
           "Kp is a 3-hour index repeated across hourly records, so hours within a Kp interval cannot be treated as independent Kp measurements.",
           "Different days may share a geomagnetic storm or correlated solar-wind drivers; day counts are descriptive.",
           "No independent high-Kp SSUSI inference is performed by this script."
         ]}
    for yr,sub in work.groupby("year"):
        samples=sub[sub.method=="conditional"]
        out["years"][str(int(yr))]={"n_test_timepoints":len(samples),
            "n_strict_kp_gt_5":int((samples.kp>5).sum()),
            "n_kp_ge_4":int((samples.kp>=4).sum()),
            "n_distinct_calendar_days":int(samples.day.nunique()),
            "n_distinct_3h_kp_intervals":int(samples.kp_3h_interval.nunique())}
    for group,predicate in BINS:
        block=work[predicate(work.kp)]
        samples=block[block.method=="conditional"]
        by_method={}
        for method,rows in block.groupby("method"):
            by_method[method]={
                "n_sample_times":int(len(rows)),
                "mean_rmse":float(rows.rmse.mean()),
                "median_rmse":float(rows.rmse.median()),
                "mean_mae":float(rows.mae.mean())
            }
        group_data={
           "n_distinct_sample_times":len(samples),
           "n_distinct_3h_kp_intervals":int(samples[["year","kp_3h_interval"]].drop_duplicates().shape[0]),
           "n_distinct_calendar_days":int(samples[["year","day"]].drop_duplicates().shape[0]),
           "n_years":int(samples.year.nunique()),
           "method_metrics":by_method,
           "per_year_timepoint_counts":{str(int(y)):int(v) for y,v in samples.year.value_counts().sort_index().items()}
        }
        # These are within-sample paired differences but NOT proof of real-world skill.
        mat=block.pivot(index="sample_id",columns="method",values="rmse")
        if not mat.empty:
            for baseline in ("unconditional","interpolation"):
                delta=mat["conditional"]-mat[baseline]
                group_data["conditional_minus_"+baseline+"_mean_rmse"]=float(delta.mean())
                group_data["conditional_wins_vs_"+baseline]=int((delta<0).sum())
        out["activity_groups"][group]=group_data
    if out["n_unique_target_times"]!=240:
        out["warning"]="Archive sample count differs from previously recorded 240; manually inspect run provenance."
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics",type=Path,default=DEFAULT)
    p.add_argument("--output",type=Path,default=HERE/"prior_multiyear_strict_kp_audit.json")
    args=p.parse_args()
    if not args.metrics.is_file():raise FileNotFoundError(args.metrics)
    data=pd.read_csv(args.metrics)
    result=audit(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"n_synthetic_times":result["n_unique_target_times"],
                      "years":result["years"],
                      "strict_kp_gt5":result["activity_groups"]["Kp>5"],
                      "output":str(args.output)},indent=2))

if __name__=="__main__":main()

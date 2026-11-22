#!/usr/bin/env python3
"""Event-clustered comparison of real-SSUSI holdout reconstructions.

Each EDR acquisition contributes ONE equally weighted event mean per method.
Windows and pixels within a given EDR are correlated, and are not resampled
as independent data points. Cluster-bootstrap intervals over acquisition
means are optional only when >=8 temporally distinct acquisitions exist;
even then, repeated space-weather drivers may reduce effective independence.

python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py \
 --metrics paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/benchmarks/random_primary/metrics_per_holdout.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/"benchmarks/random_primary/metrics_per_holdout.csv"
METHODS=("conditional_ddpm","unconditional_ddpm","ovation_prime","ovation_with_observations","interpolation")
BASELINES=("unconditional_ddpm","ovation_prime","interpolation")
METRICS=("rmse","mae","pearson_r","r2","peak_ratio","gradient_ratio","peak_abs_error","gradient_ratio_abs_error")

def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f"No metric results in {path}")
    return rows

def value(row,field):
    if field=="peak_abs_error":
        return abs(float(row["pred_peak"])-float(row["truth_peak"]))
    if field=="gradient_ratio_abs_error":
        v=float(row["gradient_ratio"])
        return abs(v-1.)
    return float(row[field])

def mean_finite(arr):
    xs=np.asarray(arr,dtype=float)
    a=xs[np.isfinite(xs)]
    return float(a.mean()) if len(a) else None

def validate_and_aggregate(rows):
    cases=sorted(set(r["case"] for r in rows))
    methods=sorted(set(r["method"] for r in rows))
    missing=set(METHODS)-set(methods)
    if missing:raise ValueError(f"Missing required methods: {sorted(missing)}")
    # Matching sector keys are essential for paired comparisons.
    groups={}
    for r in rows:
        key=(r["case"],float(r["width_h"]),int(r["rank"]),r["method"])
        if key in groups:raise ValueError(f"Duplicate method/holdout evaluation: {key}")
        groups[key]=r
    windows=sorted(set(k[:3] for k in groups))
    for key in windows:
        if any(key+(m,) not in groups for m in METHODS):
            raise ValueError(f"Method missing for common held-out sector {key}")
        direct=groups[key+("ovation_prime",)]
        fused=groups[key+("ovation_with_observations",)]
        for metric in METRICS:
            a=value(direct,metric);b=value(fused,metric)
            if math.isfinite(a) and math.isfinite(b) and abs(a-b)>1e-8:
                raise AssertionError(f"Direct/fused OVATION differ on hidden pixels: {key} {metric} {a} {b}")
    event_rows=[]
    for case in cases:
        for method in METHODS:
            rs=[r for r in rows if r["case"]==case and r["method"]==method]
            rec={"case":case,"method":method,
                 "n_sectors":len(rs),
                 "n_distinct_widths":len(set(float(r["width_h"]) for r in rs))}
            for metric in METRICS:
                rec[metric]=mean_finite([value(r,metric) for r in rs])
            event_rows.append(rec)
    event_lookup={(r["case"],r["method"]):r for r in event_rows}
    paired_rows=[]
    for case in cases:
        for baseline in BASELINES:
            x=event_lookup[(case,"conditional_ddpm")]
            y=event_lookup[(case,baseline)]
            z={"case":case,"baseline":baseline}
            for m in METRICS:
                a=x[m];b=y[m]
                z["conditional_minus_baseline_"+m]=(a-b) if a is not None and b is not None else None
            paired_rows.append(z)
    return cases,windows,event_rows,paired_rows

def interval_from_events(xs,seed,n_boot):
    x=np.asarray(xs,dtype=float);x=x[np.isfinite(x)]
    if len(x)<8 or n_boot<=0:
        return {"mean":mean_finite(x),"ci95":None,"n_finite_events":len(x),
                "ci_unavailable_reason":"Fewer than 8 finite acquisition clusters or bootstrap disabled"}
    rng=np.random.default_rng(seed)
    draws=np.mean(rng.choice(x,size=(n_boot,len(x)),replace=True),axis=1)
    a,b=np.quantile(draws,[0.025,0.975])
    return {"mean":float(np.mean(x)),"ci95":[float(a),float(b)],
            "n_finite_events":len(x),
            "ci_note":"Acquisition-cluster bootstrap; still conditional on time-sampling and underlying correlated geomagnetic activity"}

def write_csv(path,rows):
    if not rows:return
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def summarize(rows,seed=2026,n_boot=5000):
    cases,windows,event_rows,paired=validate_and_aggregate(rows)
    event_summary={}
    for method in METHODS:
        sub=[r for r in event_rows if r["method"]==method]
        event_summary[method]={metric:interval_from_events(
            [np.nan if r[metric] is None else r[metric] for r in sub],
            seed+len(method)*131+len(metric),n_boot)
            for metric in METRICS}
    diff_summary={}
    for baseline in BASELINES:
        sub=[r for r in paired if r["baseline"]==baseline]
        diff_summary[baseline]={metric:interval_from_events(
            [np.nan if r["conditional_minus_baseline_"+metric] is None
             else r["conditional_minus_baseline_"+metric] for r in sub],
            seed+len(baseline)*83+len(metric),n_boot)
            for metric in METRICS}
    return event_rows,paired,{
      "n_distinct_EDR_acquisitions":len(cases),
      "n_correlated_heldout_sectors":len(windows),
      "n_method_evaluations":len(rows),
      "n_unique_spaceweather_storms":"not established",
      "primary_selection_protocol":sorted(set(r.get("selection_mode","unspecified") for r in rows)),
      "equal_acquisition_weight_method_metrics":event_summary,
      "paired_event_conditional_minus_baseline":diff_summary,
      "interpretation":"Descriptive event-level comparisons; 95% cluster-bootstrap intervals are only emitted for >=8 finite acquisitions. Nonoverlapping EDR acquisitions are not guaranteed independent magnetic storms. Strong, systematic model superiority cannot be inferred from two original examples.",
      "peak_and_gradient_metrics":"Peak ratio should be close to 1, not maximized without bound. Also evaluate absolute peak error and |gradient_ratio-1|."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics",type=Path,default=DEFAULT)
    p.add_argument("--output-root",type=Path,default=None)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=2026)
    args=p.parse_args()
    rows=read_rows(args.metrics)
    events,paired,summary=summarize(rows,args.seed,args.bootstrap)
    out=args.output_root or args.metrics.parent
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"metrics_by_acquisition.csv",events)
    write_csv(out/"paired_acquisition_differences.csv",paired)
    (out/"event_level_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    view={"n_distinct_EDR_acquisitions":summary["n_distinct_EDR_acquisitions"],
          "n_sectors":summary["n_correlated_heldout_sectors"],
          "conditional_rmse":summary["equal_acquisition_weight_method_metrics"]["conditional_ddpm"]["rmse"],
          "rmse_delta_vs_ovation":summary["paired_event_conditional_minus_baseline"]["ovation_prime"]["rmse"],
          "rmse_delta_vs_interpolation":summary["paired_event_conditional_minus_baseline"]["interpolation"]["rmse"]}
    print(json.dumps(view,indent=2))
    print("Saved:",out/"event_level_summary.json")

if __name__=="__main__":
    main()

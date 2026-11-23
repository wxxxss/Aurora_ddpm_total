#!/usr/bin/env python3
"""Label the THIRD-batch real-SSUSI holdouts by strict Kp>5, without rerun.

The January 2005 12-EDR random and active benchmarks already evaluate real
withheld measurements. Link each EDR to contemporaneous hourly OMNI Kp,
then report a paired, per-acquisition comparison for Kp<=3, 3<Kp<=5, and
Kp>5. Do not substitute the synthetic R1.2 benchmark for real truth.

Run:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_existing_real_kp.py

No model weights or original raw EDRs needed, provided the third-batch
manifest and metric CSVs exist on the author workstation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREVIOUS=ROOT/"paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5"
sys.path.insert(0,str(HERE))
from kp_time_matching import DEFAULT_KP,load_hourly_kp,match_kp

METHODS=("conditional_ddpm","unconditional_ddpm","ovation_prime","interpolation")
STUDIES=("random_primary","active_stress")
STATS=("rmse","mae","pearson_r","peak_ratio","gradient_ratio")
LEVELS=("Kp<=3","3<Kp<=5","Kp>5")

def report_kp(manifest,metrics,times,kps,protocol):
    cases=manifest["cases"]
    ids=[r["case_id"] for r in cases]
    if len(ids)!=len(set(ids)):raise ValueError("Repeated case_id in manifest")
    kp_by_case={r["case_id"]:match_kp(times,kps,r["acquisition_start"],r["acquisition_end"])
                for r in cases}
    df=pd.read_csv(metrics)
    if set(df.case.unique())!=set(ids):raise ValueError(f"{protocol}: metric acquisitions differ from preregistered catalog")
    if set(df.selection_mode.unique())!={protocol.split("_")[0].replace("primary","random").replace("stress","active")}:
        # Exact expected values below, rather than implicitly trusting path.
        mode="random" if protocol=="random_primary" else "active"
        if set(df.selection_mode.unique())!={mode}:
            raise ValueError(f"Expected selection_mode={mode}, got {list(df.selection_mode.unique())}")
    m=df.groupby(["case","method"],sort=True)[list(STATS)].mean().reset_index()
    rows=[]
    for case in ids:
        subset=m[m.case==case]
        if not set(METHODS).issubset(set(subset.method)):
            raise ValueError(f"{case}: missing method in experiment")
        cond=float(subset[subset.method=="conditional_ddpm"].rmse.iloc[0])
        for _,row in subset.iterrows():
            rec={"case":case,"protocol":protocol,"method":str(row.method),**kp_by_case[case]}
            for stat in STATS:rec[stat]=float(row[stat])
            if row.method in METHODS:
                rec["conditional_minus_method_rmse"]=cond-float(row.rmse)
            rows.append(rec)
    return rows,kp_by_case

def mean_or_none(values):
    a=np.asarray(values,dtype=float)
    good=a[np.isfinite(a)]
    return float(np.mean(good)) if len(good) else None

def summarize(rows):
    d=pd.DataFrame(rows)
    output={}
    for protocol in STUDIES:
        sub=d[d.protocol==protocol]
        output[protocol]={}
        for group in LEVELS+("unavailable",):
            a=sub[sub.group==group]
            if a.empty:
                output[protocol][group]={"n_acquisitions":0,"status":"no eligible matched acquisitions"}
                continue
            method_means={}
            for method in METHODS:
                r=a[a.method==method]
                method_means[method]={stat:mean_or_none(r[stat]) for stat in STATS}
            paired={}
            for baseline in ("unconditional_ddpm","ovation_prime","interpolation"):
                r=a[a.method==baseline]
                diff=np.asarray(r["conditional_minus_method_rmse"],float)
                paired[baseline]={"mean_delta_rmse":float(np.mean(diff)),
                                  "conditional_wins":int(np.sum(diff<0)),
                                  "n_acquisitions":len(diff)}
            output[protocol][group]={
                "n_acquisitions":a.case.nunique(),
                "n_distinct_calendar_days":len(set(str(v)[:10] for v in a.drop_duplicates("case")["kp_midpoint_utc"] if pd.notna(v))),
                "method_event_equal_means":method_means,
                "paired_conditional_minus_baseline_rmse":paired,
                "n_acquisitions_with_kp_transition":int(a.drop_duplicates("case")["kp_values_change_within_acquisition"].fillna(False).sum()),
            }
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=PREVIOUS/"independent_cases/manifest.json")
    p.add_argument("--kp-hourly",type=Path,default=DEFAULT_KP)
    p.add_argument("--random-metrics",type=Path,default=PREVIOUS/"benchmarks/random_primary/metrics_per_holdout.csv")
    p.add_argument("--active-metrics",type=Path,default=PREVIOUS/"benchmarks/active_stress/metrics_per_holdout.csv")
    p.add_argument("--output-root",type=Path,default=HERE/"existing_real_kp")
    args=p.parse_args()
    data=json.loads(args.manifest.read_text(encoding="utf-8"))
    times,kp=load_hourly_kp(args.kp_hourly)
    allrows=[]
    events=None
    for name,path in (("random_primary",args.random_metrics),("active_stress",args.active_metrics)):
        rows,e=report_kp(data,path,times,kp,name)
        allrows.extend(rows)
        if events is not None and events!=e:
            raise AssertionError("Kp labels differ between random and active protocols")
        events=e
    args.output_root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(allrows).to_csv(args.output_root/"kp_joined_event_method_metrics.csv",index=False)
    result={"month":data.get("month"),"kp_source":str(args.kp_hourly),
            "n_temporally_selected_acquisitions":len(data["cases"]),
            "n_kp_gt5_acquisitions":sum(int(x["group"]=="Kp>5") for x in events.values()),
            "per_event_kp":events,
            "evaluation_by_activity":summarize(allrows),
            "limitations":[
              "Kp is nominally a three-hour index; each EDR is approximately 102 min and can straddle Kp intervals.",
              "At least one high-Kp *independent storm* cannot be established merely by several EDRs from one active day.",
              "The real SSUSI heldout targets were selected before Kp labels were attached. No model retraining or new inference occurs here.",
              "If high-Kp EDR count is small or zero, do not assert that reviewer R2-4 has been fully answered; select additional high-Kp original EDRs a priori."
            ]}
    (args.output_root/"existing_real_kp_summary.json").write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"n_EDRs":len(data["cases"]),
                      "strict_high_kp_EDRs":result["n_kp_gt5_acquisitions"],
                      "by_protocol":{s:{k:v["n_acquisitions"] for k,v in result["evaluation_by_activity"][s].items()}
                                     for s in STUDIES},
                      "output":str(args.output_root)},indent=2))

if __name__=="__main__":main()

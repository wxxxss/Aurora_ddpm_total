#!/usr/bin/env python3
"""Reproduce third-batch primary-vs-active SSUSI holdout comparisons.

Use the SAME 12 time-selected EDRs but keep flux-blind RANDOM primary
and flux-selected ACTIVE stress-test datasets separate. Scores are computed
only on valid withheld measured SSUSI cells. Do not treat 36 related
sectors as 36 independent geomagnetic events.

From repository root:
 python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/compare_random_active.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = {"conditional_ddpm","unconditional_ddpm","ovation_prime",
           "ovation_with_observations","interpolation"}
MATCH = ["case","width_h","rank"]
STAT = ["rmse","mae","pearson_r","r2","peak_ratio","gradient_ratio"]
TRUTH = ["holdout_pixels","n_pixels","coverage","signal_fraction",
         "truth_mean","truth_peak","truth_gradient_mean","center_mlt_h"]

def check(df,mode):
    required = {"case","width_h","rank","method","selection_mode","selection_score",
                "rmse","mae","pearson_r","peak_ratio","gradient_ratio",
                "pred_peak","truth_peak","pred_gradient_mean","truth_gradient_mean",
                *TRUTH}
    if not required.issubset(df):
        raise ValueError(f"{mode}: missing columns {sorted(required-set(df))}")
    if set(df.selection_mode)!={mode}:raise ValueError(f"Expected {mode} selection")
    if mode=="random" and not np.allclose(df.selection_score,0):
        raise ValueError("Random primary holdouts contain flux-dependent ranking scores")
    if df[MATCH+["method"]].duplicated().any():
        raise ValueError("Duplicate method evaluation on one sector")
    by_sector=df.groupby(MATCH)
    for key,g in by_sector:
        if len(g)!=5 or set(g.method)!=METHODS:
            raise ValueError(f"Unmatched method rows: {key}")
        for k in TRUTH:
            v=pd.to_numeric(g[k],errors="raise").to_numpy(dtype=float)
            if not np.allclose(v,v[0],equal_nan=True,atol=1e-9,rtol=0):
                raise AssertionError(f"Different withheld target/mask: {key} {k}")
        a=g[g.method=="ovation_prime"].iloc[0]
        b=g[g.method=="ovation_with_observations"].iloc[0]
        for k in STAT+["pred_peak","pred_gradient_mean"]:
            if not np.isclose(a[k],b[k],equal_nan=True,atol=1e-10,rtol=0):
                raise AssertionError(f"Direct/fused OVATION differ on hidden pixels: {key} {k}")
    n_cases=df.case.nunique()
    n_sectors=df[MATCH].drop_duplicates().shape[0]
    if n_cases!=12 or n_sectors!=36 or len(df)!=180:
        raise ValueError(f"Unexpected benchmark completeness: {n_cases} EDRs, {n_sectors} sectors, {len(df)} rows")
    if set(df.width_h)!={1.,2.,4.}:
        raise ValueError("MLT sector widths differ from 1,2,4 h")
    if any("REV06228" in v or "REV06263" in v for v in df.case.unique()):
        raise AssertionError("Figure 6/7 development orbit was included")
    return {"n_cases":n_cases,"n_sectors":n_sectors,"n_evaluations":len(df)}

def describe(df):
    events=df.groupby(["case","method"],sort=True)[
        ["rmse","mae","pearson_r","peak_ratio","gradient_ratio",
         "truth_peak","pred_peak","truth_gradient_mean","pred_gradient_mean"]].mean()
    methods={}
    for method in sorted(METHODS):
        a=events.xs(method,level="method")
        methods[method]={
          "rmse":float(a.rmse.mean()),
          "mae":float(a.mae.mean()),
          "pearson_r":float(a.pearson_r.mean()),
          "mean_sector_peak_ratio":float(a.peak_ratio.mean()),
          "mean_sector_gradient_ratio":float(a.gradient_ratio.mean()),
          "ratio_of_predicted_to_measured_peak_means":float(a.pred_peak.mean()/a.truth_peak.mean()),
          "ratio_of_predicted_vs_measured_gradient_means":float(a.pred_gradient_mean.mean()/a.truth_gradient_mean.mean()),
          "mean_peak_absolute_error":float(np.mean(np.abs(
              df.loc[df.method==method,"pred_peak"].to_numpy() -
              df.loc[df.method==method,"truth_peak"].to_numpy()))),
        }
    cond=df[df.method=="conditional_ddpm"].set_index(MATCH)
    paired={}
    for method in ("unconditional_ddpm","ovation_prime","interpolation"):
        base=df[df.method==method].set_index(MATCH)
        d=(cond.rmse-base.rmse)
        by_case=d.groupby(level="case").mean()
        paired[method]={
          "conditional_minus_baseline_RMSE":float(by_case.mean()),
          "conditional_wins_acquisitions":int((by_case<0).sum()),
          "conditional_wins_sectors":int((d<0).sum()),
          "paired_RMSE_difference_by_acquisition":{k:float(v) for k,v in by_case.items()},
        }
    return {"methods":methods,"paired":paired,
       "true_flux_mean_over_sectors":float(cond.truth_mean.mean()),
       "true_peak_mean_over_sectors":float(cond.truth_peak.mean()),
       "true_gradient_mean_over_sectors":float(cond.truth_gradient_mean.mean()),
       "n_true_peaks_above_5":int((cond.truth_peak>5).sum()),
       "n_sectors_true_gradient_below_0_01":int((cond.truth_gradient_mean<.01).sum())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--random",type=Path,default=HERE/"benchmarks/random_primary/metrics_per_holdout.csv")
    p.add_argument("--active",type=Path,default=HERE/"benchmarks/active_stress/metrics_per_holdout.csv")
    p.add_argument("--output",type=Path,default=HERE/"benchmarks/random_active_comparison.json")
    args=p.parse_args()
    results={}
    frames={}
    for mode,path in (("random",args.random),("active",args.active)):
        df=pd.read_csv(path)
        results[mode]={**check(df,mode),**describe(df)}
        frames[mode]=df
    if set(frames["random"].case)!=set(frames["active"].case):
        raise AssertionError("Two protocols evaluated different acquisitions")
    results["limitations"]=[
       "Both protocols use the same 12 January 2005 EDR acquisitions, not 12 distinct geomagnetic storms.",
       "The active sectors were selected using true SSUSI flux; they are exploratory stress tests, not a representative or independently randomized validation sample.",
       "Means of per-sector gradient/peak ratios differ from ratios of aggregate means; near-zero reference gradients can destabilize the former.",
       "Qualitative images clip intensities at 5 erg cm^-2 s^-1; numerical peak metrics use unclipped SSUSI flux.",
       "Observed preservation outside artificial gaps does not demonstrate accuracy of generated values inside the gaps.",
    ]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases":results["random"]["n_cases"],
      "random":{"conditional_rmse":results["random"]["methods"]["conditional_ddpm"]["rmse"],
                 "conditional_peak_ratio":results["random"]["methods"]["conditional_ddpm"]["mean_sector_peak_ratio"]},
      "active":{"conditional_rmse":results["active"]["methods"]["conditional_ddpm"]["rmse"],
                 "conditional_peak_ratio":results["active"]["methods"]["conditional_ddpm"]["mean_sector_peak_ratio"],
                 "conditional_gradient_ratio":results["active"]["methods"]["conditional_ddpm"]["mean_sector_gradient_ratio"]},
      "active_paired_RMSE":{m:v["conditional_minus_baseline_RMSE"]
                            for m,v in results["active"]["paired"].items()},
      "output":str(args.output)},indent=2))
if __name__=="__main__":
    main()

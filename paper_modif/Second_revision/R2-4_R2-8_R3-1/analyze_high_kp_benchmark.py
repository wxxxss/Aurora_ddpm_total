#!/usr/bin/env python3
"""Compare matched real-SSUSI reconstructions within Kp-stratified EDRs.

Expected inputs:
  --manifest    kp_stratified_cases/manifest.json
  --metrics     kp_benchmark/metrics_per_holdout.csv

Uses orbit-mean withheld-pixel RMSE/MAE and separate strict Kp>5 / Kp<=3
strata. Multiple MLT sectors within the same orbit are correlated; labels
belong to the acquisition, not each independently sampled pixel.
No CI/p value is reported without multiple DISTINCT storm events established.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_MANIFEST=HERE/"kp_stratified_cases/manifest.json"
DEFAULT_METRICS=HERE/"kp_benchmark/metrics_per_holdout.csv"
METHODS=("conditional_ddpm","unconditional_ddpm","ovation_prime",
         "ovation_with_observations","interpolation")

def analyze(manifest,frame):
    recs={r["case_id"]:r for r in manifest["cases"]}
    if len(recs)!=len(manifest["cases"]):raise ValueError("Duplicated case in storm manifest")
    if set(frame.case)!=set(recs):raise ValueError("Benchmark cases differ from Kp-stratified manifest")
    if frame[["case","width_h","rank","method"]].duplicated().any():
        raise ValueError("Duplicate evaluation row")
    if "selection_score" not in frame or not np.allclose(frame["selection_score"],0,rtol=0,atol=1e-12):
        raise ValueError("Strict Kp stratified evaluation must be flux-blind; expected zero selection scores")
    for case,info in recs.items():
        k=float(info["kp_midpoint"])
        expected="Kp>5" if k>5 else "Kp<=3" if k<=3 else "other"
        if expected not in ("Kp>5","Kp<=3") or info["kp_group"]!=expected:
            raise ValueError(f"Unexpected group for {case}: Kp={k}, label={info['kp_group']}")
    for key,g in frame.groupby(["case","width_h","rank"]):
        if set(g.method)!=set(METHODS) or len(g)!=len(METHODS):
            raise ValueError(f"Missing methods for sector {key}")
        # All four predictors plus the observed-fused OVATION display must
        # be scored at the EXACT same measured withheld SSUSI pixels.
        for field in ("n_pixels","holdout_pixels","center_mlt_h","coverage",
                      "truth_mean","truth_peak","truth_gradient_mean"):
            values=g[field].astype(float).to_numpy()
            if not np.allclose(values,values[0],rtol=0,atol=1e-9,equal_nan=True):
                raise AssertionError(f"Different withheld targets at {key}: {field}")
        a=g[g.method=="ovation_prime"].iloc[0]
        b=g[g.method=="ovation_with_observations"].iloc[0]
        for stat in ("rmse","mae","pearson_r","peak_ratio","gradient_ratio"):
            if not np.isclose(float(a[stat]),float(b[stat]),rtol=0,atol=1e-9,equal_nan=True):
                raise AssertionError(f"OVATION fused/direct mismatch at {key} / {stat}")
    if "selection_mode" not in frame or set(frame.selection_mode)!={"random"}:
        raise ValueError("Storm-stratified primary uses only flux-blind random-sector holdouts")
    frame=frame.copy()
    frame["kp_group"]=frame.case.map(lambda x:recs[x]["kp_group"])
    frame["kp_midpoint"]=frame.case.map(lambda x:recs[x]["kp_midpoint"])
    if not set(frame.kp_group).issubset({"Kp>5","Kp<=3"}):
        raise ValueError("Unexpected mid-range Kp in high/low stratified benchmark")
    events=frame.groupby(["case","method","kp_group"],sort=True)[
        ["rmse","mae","pearson_r","peak_ratio","gradient_ratio",
         "truth_mean","truth_peak","pred_peak",
         "truth_gradient_mean","pred_gradient_mean"]].mean().reset_index()
    event_wise=[]
    for case in sorted(recs):
        sub=events[events.case==case]
        cond=float(sub[sub.method=="conditional_ddpm"].rmse.iloc[0])
        for baseline in ("unconditional_ddpm","ovation_prime","interpolation"):
            other=float(sub[sub.method==baseline].rmse.iloc[0])
            event_wise.append({
               "case":case,"kp_group":recs[case]["kp_group"],
               "kp_midpoint":recs[case]["kp_midpoint"],
               "baseline":baseline,"delta_conditional_minus_baseline_rmse":cond-other,
               "conditional_rmse_lower":bool(cond<other)
            })
    out={"n_acquisition_cases":len(recs),"by_group":{},
         "independence_limitation":"Different EDRs or UTC days can remain embedded in the same geomagnetic disturbance. Case-stratified means are descriptive, not a 12-storm statistical test.",
         "reference_truth":"Artificially withheld, native-supported SSUSI energy flux, NOT generated OVATION maps.",
         "no_inference_claim":"No bootstrap CI or p value is reported from a handful of Kp>5 dates.",
         "metric_warning":"Per-sector peak and gradient ratios can be inflated by nearly zero quiet-sector reference values; also report ratios of aggregate predicted and measured magnitudes.",
         "flux_domain_warning":"Larger absolute RMSE under Kp>5 is confounded by the substantially larger true withheld flux distribution; the group difference is not a causal Kp effect."}
    for label in ("Kp>5","Kp<=3"):
        sub=events[events.kp_group==label]
        cmp=[r for r in event_wise if r["kp_group"]==label]
        if sub.empty:
            out["by_group"][label]={"n_acquisitions":0,"status":"no available events"}
            continue
        # Aggregate ratios use PHYSICAL magnitudes, not arithmetic means
        # of potentially unstable individual withheld-sector ratios.
        scale_stats={}
        for method in METHODS:
            rows=sub[sub.method==method]
            true_peaks=float(rows["truth_peak"].mean())
            true_grads=float(rows["truth_gradient_mean"].mean())
            scale_stats[method]={
                "mean_peak_abs_error":float((rows["pred_peak"]-rows["truth_peak"]).abs().mean()),
                "ratio_of_mean_predicted_to_mean_true_peaks":(
                    float(rows["pred_peak"].mean()/true_peaks) if true_peaks>0 else None),
                "ratio_of_mean_predicted_to_mean_true_gradients":(
                    float(rows["pred_gradient_mean"].mean()/true_grads) if true_grads>0 else None),
                "mean_reference_flux":float(rows["truth_mean"].mean()),
                "mean_reference_peak":true_peaks,
                "mean_reference_gradient":true_grads,
            }
        cond_sectors=frame[(frame.kp_group==label)&(frame.method=="conditional_ddpm")]
        out["by_group"][label]={
          "n_heldout_sectors":int(len(cond_sectors)),
          "n_heldout_sectors_true_gradient_lt_0_01":int((cond_sectors.truth_gradient_mean<.01).sum()),
          "scale_robust_diagnostics":scale_stats,
          "n_acquisitions":sub.case.nunique(),
          "n_distinct_calendar_days":len(set(
              str(recs[c]["acquisition_start"])[:10] for c in sub.case.unique())),
          "method_event_equal_means":{
             method:{stat:float(row[stat].mean()) for stat in
                     ("rmse","mae","pearson_r","peak_ratio","gradient_ratio")}
             for method,row in ((m,sub[sub.method==m]) for m in METHODS)
          },
          "paired_conditional_minus_baseline_rmse":{
             base:{
                "mean":float(np.mean([r["delta_conditional_minus_baseline_rmse"]
                                     for r in cmp if r["baseline"]==base])),
                "n_acquisitions_conditional_lower":sum(
                     r["conditional_rmse_lower"] for r in cmp if r["baseline"]==base)
             } for base in ("unconditional_ddpm","ovation_prime","interpolation")
          }
        }
    return out,events,event_wise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    p.add_argument("--metrics",type=Path,default=DEFAULT_METRICS)
    p.add_argument("--output-root",type=Path,default=HERE/"kp_benchmark")
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    frame=pd.read_csv(args.metrics)
    output,by_case,paired=analyze(manifest,frame)
    args.output_root.mkdir(parents=True,exist_ok=True)
    by_case.to_csv(args.output_root/"kp_group_event_method_metrics.csv",index=False)
    pd.DataFrame(paired).to_csv(args.output_root/"kp_group_paired_event_differences.csv",index=False)
    (args.output_root/"kp_group_real_holdout_summary.json").write_text(
        json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2))
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Recompute per-event descriptive SSUSI-minus-OVATION PSD differences.

Uses center-aligned smoothness_bias_analysis.py outputs. Checks that the paired
segment-level CSV produces the same 44/35-event and pooled mean differences
stored in psd_comparison.csv. Segment p-values are intentionally NOT used as
independent-event inference.

python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/verify_psd_event_stats.py
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/"center_aligned_smoothness"
def read_csv(p):
    with p.open("r",newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def avg(xs):
    good=[x for x in xs if math.isfinite(x)]
    if not good:raise ValueError("No finite values")
    return sum(good)/len(good)
def verify(segments,comparison):
    comp={r["case"]:r for r in comparison}
    output={}
    for case,n in (("Figure6",44),("Figure7",35),("combined",79)):
        ss=segments if case=="combined" else [r for r in segments if r["case"]==case]
        if len(ss)!=n:raise AssertionError(f"{case}: got {len(ss)} segments; expected {n}")
        hf=avg([float(r["ssusi_high_frequency_fraction"])-float(r["ovation_high_frequency_fraction"]) for r in ss])
        slope=avg([float(r["ssusi_spectral_slope"])-float(r["ovation_spectral_slope"]) for r in ss])
        summary=comp[case]
        for name,x in (("hf_mean_difference_ssusi_minus_ovation",hf),
                       ("slope_mean_difference_ssusi_minus_ovation",slope)):
            if abs(x-float(summary[name]))>1e-8:
                raise AssertionError(f"{case}: {name} disagrees: {x} versus {summary[name]}")
        output[case]={"n_spatially_correlated_segments":n,
                     "mean_high_frequency_fraction_difference":hf,
                     "mean_spectral_slope_difference":slope}
    output["independent_EDR_cases"]=2
    output["event_equal_weight_high_frequency_mean"]=avg([output[c]["mean_high_frequency_fraction_difference"] for c in ("Figure6","Figure7")])
    output["event_equal_weight_slope_mean"]=avg([output[c]["mean_spectral_slope_difference"] for c in ("Figure6","Figure7")])
    output["interpretation"]="Descriptive only: do not treat 79 spatial segments as 79 independent events or use segment-bootstrap CIs to infer event-level generality."
    return output
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir",type=Path,default=DEFAULT)
    args=p.parse_args()
    s=read_csv(args.dir/"psd_segment_metrics.csv")
    c=read_csv(args.dir/"psd_comparison.csv")
    print(json.dumps(verify(s,c),indent=2))
if __name__=="__main__":main()

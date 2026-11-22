#!/usr/bin/env python3
"""Audit the measured-pixel 12-acquisition real-SSUSI benchmark, without model rerun.

Reads the *actual* metrics_per_holdout.csv and event_level_summary.json.
Checks common masked SSUSI truth, five-method pairing, exact identity of
direct vs known-observation-fused OVATION in hidden cells, all expected MLT
widths, and the event-aggregated summary values. Produces a transparent,
acquisition-level result, not pixel-level pseudoreplication.

Run from the repository root:
python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/audit_multievent_results.py

The high- vs low-flux groupings are EXPLORATORY POST-HOC descriptors, and
the 12 acquisition clusters do not establish 12 distinct magnetic storms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "benchmarks/random_primary"
METHODS = (
    "conditional_ddpm", "unconditional_ddpm", "ovation_prime",
    "ovation_with_observations", "interpolation",
)
BASELINES = ("unconditional_ddpm", "ovation_prime", "interpolation")
MATCH = ["case", "width_h", "rank"]
OBS = ["n_pixels", "holdout_pixels", "center_mlt_h", "coverage",
       "signal_fraction", "truth_mean", "truth_peak", "truth_gradient_mean"]
EVALUATE = ["rmse", "mae", "pearson_r", "r2",
            "peak_ratio", "gradient_ratio", "truth_peak", "pred_peak",
            "truth_gradient_mean", "pred_gradient_mean"]


def grouped_truth_check(frame):
    if frame[MATCH + ["method"]].duplicated().any():
        raise ValueError("Duplicate case/width/rank/method evaluations")
    groups = list(frame.groupby(MATCH, dropna=False))
    if len(groups) == 0:
        raise ValueError("No evaluated holdout sectors")
    for key, g in groups:
        if set(g["method"]) != set(METHODS) or len(g) != len(METHODS):
            raise ValueError(f"{key}: incomplete or duplicate method evaluations")
        for field in OBS:
            vals = pd.to_numeric(g[field], errors="raise").to_numpy(dtype=float)
            if not np.allclose(vals, vals[0], equal_nan=True, atol=1e-10, rtol=0):
                raise AssertionError(f"{key}: {field} truth/mask differs across methods")
        a = g[g.method == "ovation_prime"].iloc[0]
        b = g[g.method == "ovation_with_observations"].iloc[0]
        for field in EVALUATE:
            if not np.isclose(float(a[field]), float(b[field]),
                              rtol=0, atol=1e-10, equal_nan=True):
                raise AssertionError(f"{key}: fused and direct OVATION "
                                     f"have differing hidden-pixel {field}")
    if set(frame["selection_mode"]) != {"random"}:
        raise ValueError("Expected only flux-blind random primary protocol")
    if not np.allclose(frame["selection_score"], 0, atol=1e-12):
        raise ValueError("Nonzero selection scores suggest an active-flux selection")
    return groups


def check_external_summary(frame, summary):
    count_cases = int(frame["case"].nunique())
    if count_cases != int(summary["n_distinct_EDR_acquisitions"]):
        raise AssertionError("Mismatch with event-level summary: acquisitions")
    n_sectors = len(frame[MATCH].drop_duplicates())
    if n_sectors != int(summary["n_correlated_heldout_sectors"]):
        raise AssertionError("Mismatch with event-level summary: sectors")
    if len(frame) != int(summary["n_method_evaluations"]):
        raise AssertionError("Mismatch with event-level summary: evaluations")
    for method in METHODS:
        sub = frame[frame.method == method]
        event_means = sub.groupby("case", sort=True)[
            ["rmse", "mae", "pearson_r", "r2", "peak_ratio", "gradient_ratio"]
        ].mean()
        for metric in event_means:
            expected = float(summary["equal_acquisition_weight_method_metrics"]
                             [method][metric]["mean"])
            got = float(event_means[metric].mean())
            if not np.isclose(expected, got, rtol=1e-8, atol=1e-9):
                raise AssertionError(f"Event summary disagrees: {method} "
                                     f"{metric}: {got} != {expected}")


def summarize(frame):
    groups = grouped_truth_check(frame)
    independent_cases = sorted(frame["case"].unique())
    w_counts = frame[frame.method == "conditional_ddpm"].groupby("case")[
        "width_h"].agg(lambda v: sorted(set(map(float, v)))).to_dict()
    if any(v != [1., 2., 4.] for v in w_counts.values()):
        raise ValueError("Missing or unexpected widths in at least one EDR")

    means = frame.groupby(["case", "method"], sort=True)[
        ["rmse", "mae", "pearson_r", "r2",
         "peak_ratio", "gradient_ratio", "truth_peak", "pred_peak",
         "truth_gradient_mean", "pred_gradient_mean"]
    ].mean()
    method_means = {}
    for method in METHODS:
        m = means.xs(method, level="method")
        method_means[method] = {
            col: float(m[col].mean()) for col in m.columns
        }
        method_means[method]["ratio_of_mean_pred_to_truth_gradient"] = (
            float(m["pred_gradient_mean"].mean() / m["truth_gradient_mean"].mean())
            if m["truth_gradient_mean"].mean() > 0 else None
        )
        method_means[method]["ratio_of_mean_pred_to_truth_peak"] = (
            float(m["pred_peak"].mean() / m["truth_peak"].mean())
            if m["truth_peak"].mean() > 0 else None
        )

    cond = frame[frame.method == "conditional_ddpm"].set_index(MATCH)
    paired = {}
    for method in BASELINES:
        baseline = frame[frame.method == method].set_index(MATCH)
        delta = cond["rmse"] - baseline["rmse"]
        per_event = delta.groupby(level="case").mean()
        paired[method] = {
            "rmse_delta_conditional_minus_baseline_event_equal_mean":
                float(per_event.mean()),
            "rmse_delta_conditional_minus_baseline_event_median":
                float(per_event.median()),
            "n_event_rmse_wins_for_conditional":int((per_event < 0).sum()),
            "n_sector_rmse_wins_for_conditional":int((delta < 0).sum()),
            "n_events":len(per_event), "n_sectors":len(delta),
            "by_event_rmse_delta":{str(k):float(v) for k,v in per_event.items()},
        }
    width_table = frame.groupby(["width_h", "method"])["rmse"].mean().unstack()
    sectors = frame[frame.method == "conditional_ddpm"].copy()
    gradient_threshold = 0.01
    near_flat = sectors["truth_gradient_mean"] < gradient_threshold
    out = {
       "n_acquisitions": len(independent_cases),
       "n_valid_holdout_sectors":len(groups),
       "n_method_evaluations":len(frame),
       "acquisition_ids":independent_cases,
       "method_event_equal_means":method_means,
       "paired_rmse":paired,
       "mean_rmse_by_width_h":{
           str(float(k)):{str(m):float(v) for m,v in vals.items()}
           for k,vals in width_table.iterrows()
       },
       "random_protocol_audit":{
           "selection_mode":"random",
           "all_five_methods_same_ground_truth_and_mask":True,
           "fused_ovation_same_hidden_predictions_as_direct_ovation":True,
           "development_orbits_absent":not any(
               "REV06228" in x or "REV06263" in x for x in independent_cases),
           "n_withheld_sectors_with_truth_peak_lt_1":
               int((sectors["truth_peak"] < 1).sum()),
           "n_withheld_sectors_with_true_gradient_lt_0_01":
               int(near_flat.sum()),
           "true_gradient_near_zero_threshold":gradient_threshold,
           "n_sectors_peak_ratio_gt_1_for_conditional":
               int((sectors["peak_ratio"] > 1).sum()),
       },
       "metric_interpretation":{
           "gradient_ratio_mean_caution":
               "Mean of per-sector gradient ratios may be very large when "
               "denominator is nearly zero. Compare ratio of aggregated gradient "
               "means and absolute errors as robustness descriptors.",
           "peak_ratio_mean_caution":
               "Peak ratio averages values with highly variable true peaks "
               "and can overstate structure preservation; also report "
               "absolute peak error and ratio of mean peaks.",
           "r2_caution":
               "R-squared is highly unstable for nearly constant, low-flux "
               "heldout sectors; case-level RMSE/MAE are primary metrics.",
           "significance":
               "Only 12 acquisition clusters, selected in January 2005. "
               "CIs from acquisition-cluster resampling are conditional "
               "on this month's sample; no proof of independence of storms.",
           "exploratory":
               "Any subsequent stratification by held-out true flux "
               "must be labeled post hoc, not a new primary experiment."
       }
    }
    if not out["random_protocol_audit"]["development_orbits_absent"]:
        raise AssertionError("Primary validation includes development orbit")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics",type=Path,default=DEFAULT/"metrics_per_holdout.csv")
    p.add_argument("--event-summary",type=Path,default=DEFAULT/"event_level_summary.json")
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    frame=pd.read_csv(args.metrics)
    summary=json.loads(args.event_summary.read_text(encoding="utf-8"))
    check_external_summary(frame,summary)
    report=summarize(frame)
    report["paired_rmse_event_cluster_ci95"]={
        b:summary["paired_event_conditional_minus_baseline"][b]["rmse"]["ci95"]
        for b in BASELINES
    }
    out=args.output or args.metrics.parent/"audit_multievent_report.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "n_acquisitions":report["n_acquisitions"],
        "n_holdouts":report["n_valid_holdout_sectors"],
        "conditional_rmse":report["method_event_equal_means"]["conditional_ddpm"]["rmse"],
        "paired_rmse":{k:{q:v for q,v in vv.items() if q not in
                         ("by_event_rmse_delta",)}
                       for k,vv in report["paired_rmse"].items()},
        "warnings":report["metric_interpretation"]
    },indent=2))
    print("Saved",out)

if __name__=="__main__":
    main()

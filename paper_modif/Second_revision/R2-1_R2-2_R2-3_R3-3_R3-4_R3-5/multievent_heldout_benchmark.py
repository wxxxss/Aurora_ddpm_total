#!/usr/bin/env python3
"""Multi-acquisition real SSUSI holdout benchmark: R2-1/2/3, R3-3/4/5.

Reuses the first-revision model loading, sampling and metric implementation,
but injects a preregistered *time-selected* case list and target-flux-BLIND
random sector selection. Old first-round scripts and outputs remain unchanged.

Run from repo root AFTER prepare_multievent_ssusi.py:
 python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/multievent_heldout_benchmark.py

Default: 1 random sector per 1, 2, 4 h width per acquisition.
--selection-mode active is EXPLORATORY and uses withheld truth to choose
brightest windows; must not be confused with the random primary protocol.
This script forces --bootstrap 0; event-level intervals are computed by
event_level_statistics.py rather than by resampling correlated sectors.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD_DIR=ROOT/"paper_modif/First_revision/R2-1_R2-3"
OLD_SCRIPT=OLD_DIR/"ssusi_heldout_benchmark.py"
DEFAULT_PREPARED=HERE/"independent_cases"
DEFAULT_OUTPUT=HERE/"benchmarks/random_primary"

sys.path.insert(0,str(OLD_DIR))
sys.path.insert(0,str(HERE))
from selection_protocol import random_observed_holdouts, assert_no_target_leakage

def load_legacy():
    spec=importlib.util.spec_from_file_location("original_ssusi_holdout_benchmark",OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load original benchmark from {OLD_SCRIPT}")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",type=Path,default=DEFAULT_PREPARED)
    p.add_argument("--manifest",type=Path,default=None)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--selection-mode",choices=("random","active"),default="random")
    p.add_argument("--min-events",type=int,default=6)
    p.add_argument("--windows-per-width",type=int,default=1)
    p.add_argument("--seed",type=int,default=2026)
    args,rest=p.parse_known_args()
    forbidden={"--prepared-root","--output-root","--manifest","--selection-mode",
               "--min-events","--windows-per-width","--seed","--bootstrap"}
    if any(x in forbidden for x in rest):
        raise ValueError("Pass selection, paths, or bootstrap arguments once using the explicit wrapper options")
    return args,rest

def check_manifest(manifest,prepared_root,min_events,ovation_path):
    rows=manifest["cases"]
    ids=[str(r["case_id"]) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("Repeated case names in manifest")
    if len(ids)<min_events:raise RuntimeError(f"Need >= {min_events} selected acquisitions; got {len(ids)}")
    if manifest.get("include_development_figures"):
        raise ValueError("Primary validation cannot include Figure6/7 development orbits")
    if any("REV06228" in x or "REV06263" in x for x in ids):
        raise ValueError("Figure6/7 development orbit in validation set")
    month=str(manifest["month"])
    if ovation_path.stem[-8:]!=month.replace("-","")+"01":
        raise ValueError(f"OVATION file {ovation_path} does not match validation month {month}")
    periods=[]
    for row in rows:
        case=str(row["case_id"])
        start=datetime.fromisoformat(row["acquisition_start"])
        end=datetime.fromisoformat(row["acquisition_end"])
        if not start<end:raise ValueError(f"{case}: invalid EDR interval")
        a=prepared_root/case/(case+"_swath_case.npz")
        b=prepared_root/case/(case+"_metadata.json")
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError(f"Missing prepared data for {case}: {a} or {b}")
        periods.append((start,end,case))
    for i,(start,end,case) in enumerate(sorted(periods)):
        if i>0 and start < sorted(periods)[i-1][1]:
            raise ValueError(f"Overlapping EDR acquisitions in manifest near {case}")
    return ids

def random_selector(flux,obs,mlat,mlt,**kwargs):
    # Derive deterministic RNG seed from the coverage-mask geometry, NOT from
    # withheld energy flux or model predictions.
    seed=int.from_bytes(hashlib.sha256(
        np.asarray(obs,dtype=np.uint8).tobytes() +
        str(kwargs.get("width_h")).encode()).digest()[:8],"little")%(2**32)
    kwargs.pop("min_signal_fraction",None)
    kwargs.pop("signal_threshold",None)
    windows=random_observed_holdouts(flux,obs,mlat,mlt,seed=seed,**kwargs)
    for w in windows:
        input_mask=np.asarray(obs,dtype=bool)&~w["mask"]
        assert_no_target_leakage(flux,input_mask,w["mask"],
                                 np.where(input_mask,flux,0.0))
    return windows

def strict_ovation_coverage(oldfunc):
    def func(ovation,ovation_start,start,end):
        i0=int(np.floor((start-ovation_start).total_seconds()/60))
        i1=int(np.ceil((end-ovation_start).total_seconds()/60))
        if i0<0 or i1>=len(ovation):
            raise ValueError(f"OVATION does not fully cover {start}..{end}: {i0}..{i1} of {len(ovation)}")
        return oldfunc(ovation,ovation_start,start,end)
    return func

def append_fused_ovation(df,legacy):
    # Fused OVATION pastes known satellite pixels into the map outside the
    # artificial holdout. Therefore its hidden-region prediction is EXACTLY
    # direct OVATION, and all holdout metrics must be identical.
    original=df[df["method"]=="ovation_prime"].copy()
    if original.empty:raise ValueError("Direct OVATION evaluation missing")
    original["method"]="ovation_with_observations"
    output=legacy.pd.concat([df,original],ignore_index=True)
    return output

def main():
    args,other=parse_args()
    manifest_path=args.manifest or args.prepared_root/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No time-selected catalog at {manifest_path}; prepare_multievent_ssusi.py must run first")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    # Determine which OVATION series the historical loader will use.
    old=load_legacy()
    ovation_path=old.DEFAULT_OVATION
    if "--ovation" in other:
        i=other.index("--ovation")
        if i+1>=len(other):raise ValueError("--ovation requires a file")
        ovation_path=Path(other[i+1])
    case_ids=check_manifest(manifest,args.prepared_root,args.min_events,ovation_path)
    old.CASES=tuple(case_ids)
    old.ovation_interval_mean=strict_ovation_coverage(old.ovation_interval_mean)
    if args.selection_mode=="random":
        old.select_holdout_windows=random_selector
    # Else the original flux-conditioned active selector remains in place.
    # Run the original tested model execution with EXACTLY the new case list.
    old_argv=sys.argv
    try:
        sys.argv=[str(OLD_SCRIPT)]+other+[
          "--prepared-root",str(args.prepared_root),
          "--output-root",str(args.output_root),
          "--windows-per-width",str(args.windows_per_width),
          "--bootstrap","0",
          "--seed",str(args.seed)
        ]
        old.main()
    finally:
        sys.argv=old_argv
    csv_path=args.output_root/"metrics_per_holdout.csv"
    frame=old.pd.read_csv(csv_path)
    frame["selection_mode"]=args.selection_mode
    frame["catalog_month"]=manifest["month"]
    frame=append_fused_ovation(frame,old)
    frame.to_csv(csv_path,index=False)
    old.summarize_metrics(frame,n_boot=0,seed=args.seed).to_csv(
        args.output_root/"metrics_summary.csv",index=False)
    bpath=args.output_root/"benchmark_summary.json"
    summary=json.loads(bpath.read_text(encoding="utf-8"))
    summary["n_method_evaluations"]=len(frame)
    summary["experiment"]["selection_mode"]=args.selection_mode
    summary["experiment"]["catalog_manifest"]=str(manifest_path)
    summary["experiment"]["n_acquisitions"]=len(case_ids)
    summary["experiment"]["inference_unit"]="EDR acquisition/temporal cluster, not sector or pixel"
    summary["experiment"]["fused_ovation_note"]="Fused OVATION and direct OVATION have identical withheld-pixel predictions; fusion only restores known satellite pixels outside the evaluation region."
    summary["experiment"]["no_sector_level_bootstrap"]=True
    bpath.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    protocol={
      "case_ids":case_ids,
      "number_of_preselected_acquisitions":len(case_ids),
      "selection_mode":args.selection_mode,
      "flux_blind_selection":args.selection_mode=="random",
      "number_of_evaluated_sectors":int(frame[["case","width_h","rank"]].drop_duplicates().shape[0]),
      "method_counts":{str(k):int(v) for k,v in frame.method.value_counts().items()},
      "note":"Disjoint acquisition dates are analysis clusters, not guaranteed independent magnetic storms. Compare methods using matched sector masks, aggregate at the acquisition level."
    }
    (args.output_root/"protocol_verification.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(protocol,indent=2))
    print("NEXT: python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/event_level_statistics.py --metrics",csv_path)

if __name__=="__main__":
    main()

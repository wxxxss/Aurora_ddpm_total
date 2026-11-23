#!/usr/bin/env python3
"""Reviewer R3-8: fixed-mask, multiple-seed stochastic DDPM reconstruction.

The Jan 2006 12-acquisition / 36-sector primary validation is COMPLETE.
This supplemental experiment does NOT retrain or select new satellite
observations, checkpoints, solar forcing, target regions, or masks.

For each of the exact existing 36 preselected 1/2/4-hour SSUSI holdouts,
run the SAME conditional DDPM with five independently seeded reverse
diffusion trajectories, with the original N_SAMPLE=1 and 300 denoising
steps. Changing sample(n_sample=...) would instead change the *internal
resampling schedule*, NOT create independent model draws.

- Reads the original 2006 metrics_per_holdout.csv as the mask anchor.
- Recreates each flux-blind original mask using the same initial
  selection seed, case order and observed support mask, and asserts
  original center, observed heldout cell count, coverage.
- Uses no withheld flux in masking or model input. Truth is used ONLY
  to calculate scoring metrics/empirical uncertainty on heldout cells.
- Stores per-window small JSON reports to permit restart with --resume.
- Computes empirical per-pixel 10/90%-quantile band inclusion ONLY as
  descriptive diagnostics, NOT a calibrated predictive interval.
- Never replaces the 12-event primary RMSE tables with best-of-5 or
  ensemble-selection results.

Quick check without any model weight loading or inference:
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_stochastic_reconstruction.py --dry-run

Then full measured 12-event supplementary experiment:
python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_2006_stochastic_reconstruction.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
THIRD=REPO/"paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5"
if str(THIRD) not in sys.path:
    sys.path.insert(0,str(THIRD))

from multievent_heldout_benchmark import random_selector

DEFAULT_PREP=HERE/"real_2006_01_omni_qualified_inputs"
DEFAULT_CSV=HERE/"real_2006_01_benchmark/metrics_per_holdout.csv"
DEFAULT_OUTPUT=HERE/"real_2006_01_stochastic_uncertainty"
DEFAULT_OMNI=Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2006/omni_20060101_5min_gap_preserving.npy")
WIDTHS=(1.,2.,4.)
MLAT=np.linspace(50.,90.,80)
MLT=np.linspace(0.,24.,96,endpoint=False)
N_REPEATS=5
SEED_DISTANCE=100000

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for chunk in iter(lambda:h.read(8<<20),b""):
            d.update(chunk)
    return d.hexdigest()

def read_reference(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as fp:
        rows=list(csv.DictReader(fp))
    if len(rows)!=180:
        raise ValueError(f"Expected original 180 metric rows, found {len(rows)}")
    original={}
    expected_methods={
        "conditional_ddpm","unconditional_ddpm","ovation_prime",
        "ovation_with_observations","interpolation"}
    method_keys={}
    for r in rows:
        key=(r["case"],float(r["width_h"]),int(r["rank"]))
        method_keys.setdefault(key,set()).add(r["method"])
        if r["method"]=="conditional_ddpm":
            if key in original:
                raise ValueError(f"Duplicate original conditional sample: {key}")
            original[key]=r
    if len(original)!=36 or len(method_keys)!=36 or any(
        m!=expected_methods for m in method_keys.values()):
        raise ValueError("Original 2006 comparison must have 36 identical five-method sectors")
    return original

def make_mask_check(case,width,rank,flux,obs,ref,selection_seed):
    wins=random_selector(
        flux,obs,MLAT,MLT,selection_seed=selection_seed,
        width_h=width,n_windows=1,band=(60.,80.),
        min_coverage=.50,min_pixels=20,candidate_step_h=.5)
    if len(wins)!=1 or rank!=1:
        raise ValueError(f"No preselected coverage-qualified random holdout: {case} width={width}")
    w=wins[0]
    if abs(float(ref["center_mlt_h"])-float(w["center_h"]))>1e-6:
        raise AssertionError(f"Changed original MLT mask center {case} width={width}")
    if int(ref["holdout_pixels"])!=int(w["n_pixels"]):
        raise AssertionError(f"Changed original masked observed pixel set size {case} width={width}")
    if abs(float(ref["coverage"])-float(w["coverage"]))>1e-5:
        raise AssertionError(f"Changed original masked observation coverage {case} width={width}")
    target=np.asarray(w["mask"],dtype=bool)
    known=np.asarray(obs,dtype=bool)&~target
    if np.any(target&known) or np.any(~np.isfinite(flux[target])) or (
        np.any(np.where(known,flux,0.)[target]!=0)
    ):
        raise AssertionError("Heldout SSUSI target contamination or nonfinite truth")
    return target,known,w

def describe_ensemble(samples,truth):
    """Samples shape [draw, heldout-observed-pixel]. No missing truth imputation."""
    draws=np.asarray(samples,dtype=np.float64)
    y=np.asarray(truth,dtype=np.float64)
    if draws.ndim!=2 or len(y)!=draws.shape[1] or draws.shape[0]<2 or not len(y):
        raise ValueError("Expected >=2 fixed-mask draws on >=1 heldout observed pixel")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite stochastic predictions/ground truth")
    residual=draws-y[np.newaxis,:]
    per_rmse=np.sqrt(np.mean(residual**2,axis=1))
    per_mae=np.mean(np.abs(residual),axis=1)
    ensemble_mean=draws.mean(axis=0)
    pixel_std=draws.std(axis=0,ddof=1)
    q10,q90=np.quantile(draws,[.1,.9],axis=0)
    return {
        "n_draws":int(draws.shape[0]),
        "n_heldout_real_observed_pixels":int(draws.shape[1]),
        "per_draw_rmse":[float(v) for v in per_rmse],
        "per_draw_mae":[float(v) for v in per_mae],
        "mean_per_draw_rmse":float(per_rmse.mean()),
        "std_per_draw_rmse":float(per_rmse.std(ddof=1)),
        "min_per_draw_rmse":float(per_rmse.min()),
        "max_per_draw_rmse":float(per_rmse.max()),
        "rmse_of_ensemble_mean":float(np.sqrt(np.mean((ensemble_mean-y)**2))),
        "mean_per_draw_mae":float(per_mae.mean()),
        "mean_pixelwise_stochastic_std":float(pixel_std.mean()),
        "median_pixelwise_stochastic_std":float(np.median(pixel_std)),
        "fraction_truth_inside_empirical_draw_q10_q90":float(np.mean(
            (y>=q10)&(y<=q90))),
        "mean_truth_flux":float(y.mean()),
        "fraction_truth_exactly_zero":float(np.mean(y==0)),
    }

def aggregate(reports):
    if len(reports)!=36 or len({(r["case"],r["width_h"],r["rank"]) for r in reports})!=36:
        raise ValueError("Need all 36 original fixed-mask sector reports")
    cases=sorted(set(r["case"] for r in reports))
    if len(cases)!=12 or any(sum(r["case"]==c for r in reports)!=3 for c in cases):
        raise ValueError("Need 3 sectors per each of the 12 original EDR acquisitions")
    keys=("mean_per_draw_rmse","std_per_draw_rmse",
          "mean_pixelwise_stochastic_std",
          "fraction_truth_inside_empirical_draw_q10_q90")
    per_case={}
    for c in cases:
        rs=[r for r in reports if r["case"]==c]
        per_case[c]={k:float(np.mean([r[k] for r in rs])) for k in keys}
    event_equal={k:float(np.mean([per_case[c][k] for c in cases])) for k in keys}
    return {
       "n_acquisition_clusters":12,"n_fixed_observed_sector_masks":36,
       "n_independent_diffusion_draws_per_mask":N_REPEATS,
       "n_total_conditional_model_draws":36*N_REPEATS,
       "acquisition_equal_summary":event_equal,
       "per_acquisition":per_case,
       "limitations":[
          "Fixed-mask, fixed-model, fixed-solar-wind stochastic sampling variability is not variation across acquisition times or independent storms.",
          "Five diffusion draws do not establish statistically calibrated predictive intervals; quantile inclusion is descriptive only.",
          "The original primary 2006 single-draw RMSE (12 acquisition means) and paired baseline comparisons remain unchanged.",
          "Do not select best draw among five using withheld SSUSI truth or use new seeds to tune training/inference parameters."
       ]
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",type=Path,default=DEFAULT_PREP)
    p.add_argument("--reference-csv",type=Path,default=DEFAULT_CSV)
    p.add_argument("--omni",type=Path,default=DEFAULT_OMNI)
    p.add_argument("--cond-checkpoint",type=Path,
        default=REPO/"ckpt/cond/aurora_diff_best.pth")
    p.add_argument("--cond-norm",type=Path,
        default=REPO/"ckpt/cond/norm_params.pkl")
    p.add_argument("--selection-seed",type=int,default=2026)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--device",default="auto")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.selection_seed!=2026:
        raise ValueError("Mask selection seed must remain 2026 to match the original published 2006 benchmark")
    refs=read_reference(args.reference_csv)
    manifest=json.loads((args.prepared_root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("month")!="2006-01" or len(manifest["cases"])!=12:
        raise ValueError("Expected the 12 original source-qualified January 2006 acquisitions")
    case_ids=[r["case_id"] for r in manifest["cases"]]
    if len(set(case_ids))!=12:
        raise ValueError("Duplicate original acquisition")
    if set(c for c,_,_ in refs)!=set(case_ids):
        raise ValueError("Reference metric cases differ from prepared manifest")
    if {(c,w,k) for c in case_ids for w in WIDTHS for k in (1,)}!=set(refs):
        raise ValueError("Reference MLT widths/ranks differ from original 1/2/4-hour protocol")

    # Lazy import: mask preflight also works on machines without NPU.
    from multievent_heldout_benchmark import load_legacy
    legacy=load_legacy()
    data=[]
    for case_idx,case in enumerate(case_ids):
        arrays,meta=legacy.load_case(args.prepared_root,case)
        flux=np.asarray(arrays["flux_grid"],dtype=np.float32)
        obs=np.asarray(arrays["obs_mask"],dtype=bool)
        if flux.shape!=(80,96) or obs.shape!=(80,96):
            raise ValueError(f"{case} unsupported SSUSI grid")
        if np.any(obs&~np.isfinite(flux)):
            raise ValueError(f"{case} source SSUSI support has nonfinite flux")
        for width_idx,w in enumerate(WIDTHS):
            target,known,choice=make_mask_check(
                 case,w,1,flux,obs,refs[(case,w,1)],args.selection_seed)
            data.append((case_idx,case,width_idx,w,flux,obs,meta,target,known,choice))
    if len(data)!=36:
        raise AssertionError("36 fixed real observed holdouts are required")

    manifest_info={
        "experiment":"R3-8 2006 fixed-mask conditional-DDPM stochastic reconstruction uncertainty",
        "year_month":"2006-01",
        "n_acquisitions":12,
        "n_preexisting_fixed_holdout_masks":36,
        "draws_per_mask":N_REPEATS,
        "total_conditional_draws":36*N_REPEATS,
        "selection_seed":args.selection_seed,
        "reference_primary_csv":str(args.reference_csv),
        "reference_primary_csv_sha256":sha256(args.reference_csv),
        "trained_conditional_checkpoint":str(args.cond_checkpoint),
        "trained_conditional_checkpoint_sha256":sha256(args.cond_checkpoint) if not args.dry_run else None,
        "trained_conditional_norm_sha256":sha256(args.cond_norm) if not args.dry_run else None,
        "source_omni_sha256":sha256(args.omni) if not args.dry_run else None,
        "source_manifest":str(args.prepared_root/"manifest.json"),
        "source_manifest_sha256":sha256(args.prepared_root/"manifest.json"),
        "mask_geometry":[
            {"case":case,"width_h":w,"rank":1,"center_h":float(choice["center_h"]),
             "heldout_pixels":int(choice["n_pixels"]),
             "coverage":float(choice["coverage"])}
            for _,case,_,w,_,_,_,_,_,choice in data
        ],
        "fixed_protocol":"Original models and nominal n_sample=1, 300 diffusion steps; fixed 2006 SSUSI heldout masks and 2006 OMNI values; independent sampling RNG streams only",
    }
    if args.dry_run:
        print(json.dumps({
            "ready_for_stochastic_inference":True,
            "n_original_fixed_masks":len(data),
            "n_repeated_draws_per_mask":N_REPEATS,
            "n_requested_conditional_model_runs":len(data)*N_REPEATS,
            "all_mask_centers_and_support_counts_match_primary_csv":True,
            "first_three_mask_checks":manifest_info["mask_geometry"][:3]
        },indent=2))
        return

    args.output_root.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output_root/"stochastic_protocol.json"
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        for k in ("reference_primary_csv_sha256",
                  "trained_conditional_checkpoint_sha256",
                  "trained_conditional_norm_sha256","source_omni_sha256",
                  "source_manifest_sha256","mask_geometry"):
            if prior.get(k)!=manifest_info[k]:
                raise RuntimeError(f"Previously started stochastic output mismatches changed source {k}")
    else:
        manifest_path.write_text(json.dumps(manifest_info,indent=2)+"\n",encoding="utf-8")
    result_dir=args.output_root/"sector_results"
    result_dir.mkdir(exist_ok=True)

    omni=np.load(args.omni,allow_pickle=False)
    device=legacy.choose_device(args.device)
    norm=legacy.load_pickle(args.cond_norm)
    model=legacy.build_conditional_model(args.cond_checkpoint,device)
    reports=[]
    for case_idx,case,width_idx,w,flux,obs,meta,target,known,choice in data:
        output_file=result_dir/f"{case}_width{w:g}h_rank1.json"
        if output_file.exists():
            if not args.resume:
                raise FileExistsError(f"{output_file} already exists; rerun with --resume if protocol unchanged")
            report=json.loads(output_file.read_text(encoding="utf-8"))
            expected_seeds=[
                args.selection_seed+case_idx*1000+width_idx*100+10+
                SEED_DISTANCE*repeat for repeat in range(N_REPEATS)]
            if (report.get("case")!=case or
                report.get("width_h")!=w or
                report.get("rank")!=1 or
                report.get("random_seeds")!=expected_seeds or
                report.get("n_draws")!=N_REPEATS or
                report.get("n_heldout_real_observed_pixels")!=int(choice["n_pixels"])):
                raise RuntimeError(f"Resumed per-sector output does not match immutable seed/holdout protocol: {output_file}")
            reports.append(report)
            print("EXISTING:",output_file.name,flush=True)
            continue
        start=legacy.as_datetime(meta["acquisition_start"])
        end=legacy.as_datetime(meta["acquisition_end"])
        arr=legacy.interval_mean_solar(omni,start,end)
        # All conditioning interval ticks were previously audited
        # raw-valid. Do not silently use NaN means in this secondary run.
        times=np.asarray(omni["utc"],dtype="datetime64[ns]")
        inside=(times>=np.datetime64(start,"ns"))&(times<=np.datetime64(end,"ns"))
        if any(not np.all(np.isfinite(np.asarray(omni[k][inside],dtype=float)))
               for k in ("Bx","By","Bz","V","P")):
            raise ValueError(f"{case}: invalid original OMNI inside acquisition")
        solar_norm=legacy.normalize_solar(arr,norm)
        draws=[]
        seeds=[]
        for repeat in range(N_REPEATS):
            # repeat zero recreates the original single-draw benchmark RNG
            # stream; other repeats alter sampling noise ONLY.
            s=args.selection_seed+case_idx*1000+width_idx*100+10+SEED_DISTANCE*repeat
            pred=legacy.run_conditional(model,device,flux,known,solar_norm,norm,seed=s)
            if not np.all(np.isfinite(pred[target])):
                raise ValueError(f"Nonfinite generated sample {case} width={w} draw={repeat}")
            if not np.array_equal(pred[known],flux[known]):
                raise AssertionError(f"Known SSUSI observations not preserved: {case} width={w}")
            draws.append(pred[target].copy())
            seeds.append(s)
            print(f"COMPLETE: {case}, {w:g}h, seed={s}",flush=True)
        metrics=describe_ensemble(np.stack(draws),flux[target])
        baseline_rmse=float(refs[(case,w,1)]["rmse"])
        # Quantifies replay reproducibility; NPU kernels may not be
        # deterministic across runs, so report discrepancies honestly.
        replay_error=metrics["per_draw_rmse"][0]-baseline_rmse
        report={
            "case":case,"width_h":w,"rank":1,
            "center_mlt_h":float(choice["center_h"]),
            "coverage":float(choice["coverage"]),
            "random_seeds":seeds,
            "original_primary_single_draw_rmse":baseline_rmse,
            "first_draw_minus_original_rmse":float(replay_error),
            **metrics
        }
        # Atomic completed-sector write so --resume can skip validated work.
        tmp=output_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        tmp.replace(output_file)
        reports.append(report)

    summary=aggregate(reports)
    summary["reference_primary_csv_sha256"]=manifest_info["reference_primary_csv_sha256"]
    summary["trained_conditional_checkpoint_sha256"]=manifest_info["trained_conditional_checkpoint_sha256"]
    summary["max_abs_replay_rmse_error"]=float(max(abs(r["first_draw_minus_original_rmse"]) for r in reports))
    summary_path=args.output_root/"stochastic_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "n_fixed_masks":summary["n_fixed_observed_sector_masks"],
        "n_total_draws":summary["n_total_conditional_model_draws"],
        "event_equal_uncertainty":summary["acquisition_equal_summary"],
        "max_abs_replay_rmse_error":summary["max_abs_replay_rmse_error"],
        "output":str(summary_path)
    },indent=2))

if __name__=="__main__":
    main()

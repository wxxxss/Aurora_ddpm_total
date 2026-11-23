#!/usr/bin/env python3
"""CRITICAL R3-1: audit the temporal pairing of 1996 training OVATION and OMNI.

The public data/dataset_diff2.py reads aurora_img_YYYYMM01.npy and
omni_YYYYMM01_5min.npy independently, concatenates *all* rows of each,
then truncates the two concatenations to the same length. It does NOT use
OMNI.utc to align them.

If OVATION is a 1-minute cube and OMNI is 5-minute data, truncation
silently pairs different times. This script checks exact file shapes and
UTC timestamps on the author's original server BEFORE any new training or
claims about solar-wind-conditioned physical generalization.

This does NOT prove which checkpoint was trained by this particular script.
Checkpoint training provenance must be verified separately.

Run:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/audit_training_alignment.py
"""
from __future__ import annotations
import argparse,calendar,json
from datetime import datetime
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_AURORA=Path("/home/docker/data/private/AuroraData/generated_aurora_data")
DEFAULT_OMNI=Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min")

def expected_counts(year,month):
    days=calendar.monthrange(year,month)[1]
    return days*24*60,days*24*12

def infer_image_cadence(n,year,month):
    n_min,n_5min=expected_counts(year,month)
    if n==n_min:return (1,"complete 1-minute calendar month")
    if n==n_5min:return (5,"complete 5-minute calendar month")
    return (None,f"cannot infer time axis from shape; expected {n_min} 1-min or {n_5min} 5-min frames")

def date_as_np(year,month):
    return np.datetime64(datetime(year,month,1),"s")

def month_inputs(aurora_root,omni_root,year,month):
    aurora=aurora_root/f"{year}_omni_aurora"/f"aurora_img_{year}{month:02d}01.npy"
    omni=omni_root/f"{year}"/f"omni_{year}{month:02d}01_5min.npy"
    if not aurora.is_file() or not omni.is_file():
        return {"year":year,"month":month,"available":False,
                "missing_files":[str(x) for x in (aurora,omni) if not x.is_file()]}
    imgs=np.load(aurora,mmap_mode="r",allow_pickle=False)
    sw=np.load(omni,allow_pickle=True)
    if imgs.ndim!=3 or imgs.shape[1:]!=(80,96):
        raise ValueError(f"{aurora}: expected [time,80,96], got {imgs.shape}")
    if sw.dtype.names is None or "utc" not in sw.dtype.names:
        raise ValueError(f"{omni}: OMNI array has no UTC field, cannot audit training alignment")
    times=np.asarray(sw["utc"],dtype="datetime64[s]").reshape(-1)
    if len(times)!=len(sw):raise ValueError("OMNI UTC count differs from NPY records (multidimensional UTC field?)")
    if len(times)>1 and np.any(np.diff(times)<np.timedelta64(0,"s")):
        raise ValueError("OMNI UTC records are not chronologically sorted; training loader does not sort them")
    if len(times)>1:
        minutes=np.diff(times).astype("timedelta64[s]").astype(np.int64)/60.
        cadence=float(np.median(minutes))
        jumps=int(np.count_nonzero(np.abs(minutes-cadence)>1e-6))
    else:
        cadence=None;jumps=None
    inferred,explanation=infer_image_cadence(len(imgs),year,month)
    expected_start=date_as_np(year,month)
    return {
        "year":year,"month":month,"available":True,
        "image_source":str(aurora),"omni_source":str(omni),
        "n_aurora_frames":int(len(imgs)),
        "n_omni_records":int(len(sw)),
        "inferred_aurora_cadence_minutes":inferred,
        "aurora_cadence_evidence":explanation,
        "omni_median_utc_step_minutes":cadence,
        "omni_nonregular_step_count":jumps,
        "omni_first_utc":str(times[0]) if len(times) else None,
        "omni_last_utc":str(times[-1]) if len(times) else None,
        "omni_start_matches_filename":bool(len(times)>0 and times[0]==expected_start),
        # No image timestamps in cube: inference relies on complete-month frames.
        "_omni_times":times
    }

def audit(aurora_root,omni_root,year=1996,months=(4,5)):
    records=[month_inputs(aurora_root,omni_root,year,m) for m in months]
    available=[r for r in records if r.get("available")]
    full_ok=len(available)==len(months)
    aligned_under_concat=None
    mismatch=None
    if full_ok and all(r["inferred_aurora_cadence_minutes"] is not None for r in records):
        at=[];ot=[]
        for r in records:
            start=date_as_np(r["year"],r["month"])
            dt=np.timedelta64(int(r["inferred_aurora_cadence_minutes"]),"m")
            at.append(start+np.arange(r["n_aurora_frames"])*dt)
            ot.append(r["_omni_times"])
        # Match actual historical concatenation and global min-length truncation.
        image_times=np.concatenate(at)
        omni_times=np.concatenate(ot)
        n=min(len(image_times),len(omni_times))
        delta_minutes=(omni_times[:n]-image_times[:n]).astype("timedelta64[s]").astype(np.int64)/60.
        errors=np.abs(delta_minutes)
        aligned_under_concat=bool(np.all(errors<=2))
        mismatch={
          "joined_aurora_length":len(image_times),
          "joined_omni_length":len(omni_times),
          "historical_truncated_length":n,
          "historical_discarded_aurora_frames":len(image_times)-n,
          "historical_discarded_omni_records":len(omni_times)-n,
          "pairs_within_2min":int(np.sum(errors<=2)),
          "pairs_with_time_difference_gt_60min":int(np.sum(errors>60)),
          "fraction_pairs_mismatched_gt_60min":float(np.mean(errors>60)) if n else None,
          "median_abs_time_difference_minutes":float(np.median(errors)) if n else None,
          "max_abs_time_difference_minutes":float(np.max(errors)) if n else None,
          "first_6_pair_differences_min":delta_minutes[:6].tolist(),
          "last_6_pair_differences_min":delta_minutes[-6:].tolist(),
        }
    report={
      "training_config_from_public_train_py":{"years":[1996],"months":[4,5],
         "split_ratio":0.9,"loader_method":"data.dataset_diff2.get_dataloaders",
         "pairing_rule":"Independent concatenation of all OVATION rows and all OMNI rows, then truncate to min length; no timestamp alignment."},
      "months":[{k:v for k,v in r.items() if not k.startswith("_")} for r in records],
      "all_original_training_files_available":full_ok,
      "can_infer_exact_array_time_alignment":mismatch is not None,
      "historical_array_pairs_time_aligned_under_inferred_cadences":aligned_under_concat,
      "historical_pairing_errors":mismatch,
      "critical_checkpoint_provenance_limit":"This verifies what the public data loader WOULD pair. It does NOT independently establish that the published inference checkpoint was trained by that exact code/data version. Recover checkpoint metadata and logs before concluding the saved model has incorrect conditioning.",
      "action_if_misaligned":"STOP strong-activity / cross-year inference claims for that checkpoint. Correct time pairing, retrain with chronologically independent train/validation separation, and rerun independent real-satellite holdout tests for any checkpoint trained on misaligned data.",
      "action_if_indeterminate":"Recover original OVATION frame timestamps and OMNI UTC index and rerun alignment validation. Never assert pairing was correct based solely on equal array lengths."
    }
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aurora-root",type=Path,default=DEFAULT_AURORA)
    p.add_argument("--omni-root",type=Path,default=DEFAULT_OMNI)
    p.add_argument("--year",type=int,default=1996)
    p.add_argument("--months",type=int,nargs="+",default=[4,5])
    p.add_argument("--output",type=Path,default=HERE/"training_alignment_audit.json")
    args=p.parse_args()
    result=audit(args.aurora_root,args.omni_root,args.year,args.months)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "all_files_available":result["all_original_training_files_available"],
        "time_alignment_inferable":result["can_infer_exact_array_time_alignment"],
        "historical_pairing_is_time_aligned":result["historical_array_pairs_time_aligned_under_inferred_cadences"],
        "pair_errors":result["historical_pairing_errors"],
        "output":str(args.output)
    },indent=2))
if __name__=="__main__":
    main()

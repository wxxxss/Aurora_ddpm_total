#!/usr/bin/env python3
"""Inspect the exact Polar/UVI examples and saved reconstructions (R3-11).

This is an input/output provenance diagnostic, not an inference-time experiment.
From repository root:
    python paper_modif/Second_revision/R2-5_R3-2_R3-11/check_polar_mask_cases.py

The legacy Polar script chooses S=1 where flux >= 1, independent of OVATION.
The manuscript instead describes an OVATION-dependent reconstruction mask.
This diagnostic reports both flux-threshold and OVATION-positive categories
for Figure 4 / Figure 5 and compares the saved reconstruction with measured
values. It cannot by itself establish which checkpoint produced a saved file.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVENTS = {
    "Figure4": datetime(1996, 4, 3, 2, 15),
    "Figure5": datetime(1996, 4, 1, 8, 40),
}
DEFAULT_OBS = Path(
    "/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/"
    "resampled_5min_1996_0405.npy"
)
DEFAULT_RECON = Path(
    "/home/docker/code/Aurora_DDPM/reasult/polar_res/new_res/"
    "repaired_polar_unetV3_ckptv2.npy"
)
DEFAULT_OVATION = Path(
    "/home/docker/data/private/AuroraData/generated_aurora_data/"
    "1996_omni_aurora/aurora_img_19960401.npy"
)
DEFAULT_OMNI = Path(
    "/home/docker/data/private/AuroraData/omni_real_data/"
    "omni_5min/1996/omni_19960401_5min.npy"
)

def nearest_record(records, target, limit_seconds, label):
    if records.dtype.names is None or "utc" not in records.dtype.names:
        raise ValueError(f"{label} must be a structured array with 'utc'")
    ts = np.asarray(records["utc"], dtype="datetime64[s]")
    t = np.datetime64(target, "s")
    seconds = np.abs((ts-t).astype("timedelta64[s]").astype(np.int64))
    i = int(seconds.argmin())
    if int(seconds[i]) > limit_seconds:
        raise ValueError(f"{label}: nearest record is {int(seconds[i])} s from {target} (allowed {limit_seconds} s)")
    return i, str(ts[i]), int(seconds[i])

def fraction_or_none(a):
    return round(float(np.mean(a)), 6) if a.size else None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path, default=DEFAULT_OBS)
    p.add_argument("--reconstruction", type=Path, default=DEFAULT_RECON)
    p.add_argument("--ovation", type=Path, default=DEFAULT_OVATION)
    p.add_argument("--omni", type=Path, default=DEFAULT_OMNI)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--ovation-positive-threshold", type=float, default=0.0)
    p.add_argument("--max-seconds", type=int, default=150)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent/"polar_mask_case_diagnostic.json")
    args = p.parse_args()
    for key in ("observation","reconstruction","ovation","omni"):
        fp=getattr(args,key)
        if not fp.is_file():
            raise FileNotFoundError(f"{key}: {fp}. Override using --{key} PATH.")
    obs_all=np.load(args.observation,allow_pickle=True)
    recon_all=np.load(args.reconstruction,allow_pickle=True)
    ov_all=np.load(args.ovation,allow_pickle=False)
    omni_all=np.load(args.omni,allow_pickle=True)
    if obs_all.dtype.names is None or "aurora_image" not in obs_all.dtype.names:
        raise KeyError("Polar observations must contain 'aurora_image'")
    if recon_all.dtype.names is None or "image" not in recon_all.dtype.names:
        raise KeyError("Saved reconstructions must contain 'image'")
    if ov_all.ndim!=3:
        raise ValueError(f"OVATION maps should be (N,H,W), got {ov_all.shape}")
    report={"inputs":{k:str(getattr(args,k)) for k in ("observation","reconstruction","ovation","omni")},
            "note":("The legacy Polar source threshold is not sufficient to "
                    "attribute the masks used for manuscript Figures 4-5; "
                    "normalization, quality control and saved-model provenance "
                    "must be checked against the actual production run."),
            "events":{}}
    for name,target in EVENTS.items():
        oidx,ots,odt=nearest_record(obs_all,target,args.max_seconds,"Polar")
        ridx,rts,rdt=nearest_record(recon_all,target,args.max_seconds,"Reconstruction")
        midx,mts,mdt=nearest_record(omni_all,target,args.max_seconds,"OMNI")
        if midx>=len(ov_all):
            raise IndexError(f"OVATION has {len(ov_all)} maps but matched OMNI index is {midx}; "
                             "provide a time-aligned OVATION file.")
        measured=np.asarray(obs_all["aurora_image"][oidx],dtype=float)
        reconstructed=np.asarray(recon_all["image"][ridx],dtype=float)
        simulated=np.asarray(ov_all[midx],dtype=float)
        if measured.shape!=reconstructed.shape or measured.shape!=simulated.shape:
            raise ValueError(f"Shape mismatch for {name}: {measured.shape}, {reconstructed.shape}, {simulated.shape}")
        finite=np.isfinite(measured)
        finite_recon=np.isfinite(reconstructed)
        # This exactly reproduces the legacy mask except for explicit reporting of non-finite pixels.
        low=finite & (measured < args.threshold)
        high=finite & (measured >= args.threshold)
        simulated_positive=np.isfinite(simulated)&(simulated>args.ovation_positive_threshold)
        reliable=high&finite_recon
        recon_abs=np.abs(measured[reliable]-reconstructed[reliable])
        report["events"][name]={
            "target_utc":target.isoformat(),
            "matched_times":{"observation":ots,"reconstruction":rts,"omni":mts},
            "offset_seconds":{"observation":odt,"reconstruction":rdt,"omni":mdt},
            "grid_shape":list(measured.shape),
            "total_cells":int(measured.size),
            "finite_observation_cells":int(finite.sum()),
            "nonfinite_observation_cells":int((~finite).sum()),
            "legacy_candidate_flux_threshold":float(args.threshold),
            "legacy_candidate_missing_cells":int(low.sum()),
            "legacy_candidate_observed_cells":int(high.sum()),
            "low_flux_and_OVATION_positive":int((low&simulated_positive).sum()),
            "low_flux_and_OVATION_nonpositive":int((low&~simulated_positive).sum()),
            "observed_flux_positive_where_OVATION_nonpositive":int((high&~simulated_positive).sum()),
            "candidate_mask_fraction_of_all_cells":fraction_or_none(low),
            "saved_reconstruction_minus_observation_on_high_flux_cells":{
                "count":int(recon_abs.size),
                "mean_absolute_error":float(recon_abs.mean()) if recon_abs.size else None,
                "maximum_absolute_error":float(recon_abs.max()) if recon_abs.size else None,
            }
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print("Saved:",args.output)

if __name__=="__main__":
    main()

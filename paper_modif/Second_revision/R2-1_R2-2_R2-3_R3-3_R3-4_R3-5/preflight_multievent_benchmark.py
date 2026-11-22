#!/usr/bin/env python3
"""Low-cost, no-DDPM preflight on the prepared January 2005 SSUSI catalog.

Checks every planned acquisition against the *same* flux-blind selection,
coverage, grid, input availability, and OMNI / OVATION UTC requirements as
the primary held-out benchmark. Writes a JSON report even if one case fails.
Do not start the model comparison until all selected cases have evaluable
holdouts for the intended MLT widths.

From repository root:
  python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/preflight_multievent_benchmark.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
DEFAULT_PREPARED=HERE/"independent_cases"
DEFAULT_OMNI=Path("/home/docker/data/private/AuroraData/omni_real_data/omni_5min/2005/omni_20050101_5min.npy")
DEFAULT_OVATION=Path("/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/aurora_img_20050101.npy")
LAT=np.linspace(50.,90.,80)
MLT=np.arange(96)*.25
SOLAR_FIELDS=("Bx","By","Bz","V","P")

from selection_protocol import random_observed_holdouts, assert_no_target_leakage

def case_seed(observed,width,seed):
    return int.from_bytes(hashlib.sha256(
        np.asarray(observed,dtype=np.uint8).tobytes()+
        str(float(width)).encode()+
        str(int(seed)).encode()
    ).digest()[:8],"little")%(2**32)

def scan_case(directory,record,widths,min_coverage,seed,windows_per_width):
    name=record["case_id"]
    npz=directory/name/(name+"_swath_case.npz")
    meta_path=directory/name/(name+"_metadata.json")
    with np.load(npz,allow_pickle=False) as arrays:
        f=np.asarray(arrays["flux_grid"],dtype=float)
        mask=np.asarray(arrays["obs_mask"],dtype=bool)
        counts=np.asarray(arrays["count_grid"],dtype=int)
        lat=np.asarray(arrays["mlat_1d"],dtype=float)
        mlt=np.asarray(arrays["mlt_1d"],dtype=float)
    if f.shape!=(80,96) or mask.shape!=(80,96) or counts.shape!=(80,96):
        raise ValueError(f"{name}: expected 80x96 grid")
    if not np.allclose(lat,LAT,atol=1e-10) or not np.allclose(mlt,MLT,atol=1e-10):
        raise ValueError(f"{name}: grid centers disagree with declared grid")
    if not np.array_equal(mask,counts>0) or np.any(counts<0):
        raise ValueError(f"{name}: mask and native count grid disagree")
    if not np.all(np.isfinite(f[mask])) or np.any(np.isfinite(f[~mask])):
        raise ValueError(f"{name}: unsupported or observed flux contains unexpected finite state")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    start=datetime.fromisoformat(meta["acquisition_start"])
    end=datetime.fromisoformat(meta["acquisition_end"])
    if start.isoformat()!=record["acquisition_start"] or end.isoformat()!=record["acquisition_end"]:
        raise ValueError(f"{name}: manifest and metadata time intervals disagree")
    result={"case":name,"orbit":record.get("orbit"),"start_utc":start.isoformat(),
            "end_utc":end.isoformat(),"native_supported_cells":int(mask.sum()),
            "support_fraction":float(mask.mean()),"selection_by_width":{}}
    for width in widths:
        wins=random_observed_holdouts(
            f,mask,lat,mlt,width_h=width,n_windows=windows_per_width,
            band=(60.,80.),min_coverage=min_coverage,min_pixels=20,
            candidate_step_h=.5,seed=case_seed(mask,width,seed))
        for w in wins:
            known=mask&~w["mask"]
            assert_no_target_leakage(f,known,w["mask"],np.where(known,f,0.))
        result["selection_by_width"][str(width)]={
            "n_selected":len(wins),
            "required":windows_per_width,
            "selected_mlt_centers_h":[float(w["center_h"]) for w in wins],
            "withheld_valid_pixels":[int(w["n_pixels"]) for w in wins],
            "min_coverage_met":all(w["coverage"]>=min_coverage for w in wins),
            "usable":len(wins)==windows_per_width}
    result["all_holdouts_evaluable"]=all(x["usable"] for x in result["selection_by_width"].values())
    return result

def check_inputs(month,cases,omni_path,ovation_path):
    report={"omni":str(omni_path),"ovation":str(ovation_path),"errors":[],"per_case":{}}
    if not omni_path.is_file() or not ovation_path.is_file():
        report["errors"].append("Missing OMNI or OVATION source array")
        return report
    ova_date=datetime.strptime(ovation_path.stem.split("_")[-1],"%Y%m%d")
    if ova_date.strftime("%Y-%m")!=month or ova_date.day!=1:
        report["errors"].append("OVATION array start-date does not match catalog month")
        return report
    ova=np.load(ovation_path,mmap_mode="r",allow_pickle=False)
    if ova.ndim!=3 or tuple(ova.shape[1:])!=(80,96):
        report["errors"].append(f"OVATION shape unexpected: {ova.shape}")
        return report
    omni=np.load(omni_path,allow_pickle=True)
    if omni.dtype.names is None or not {"utc",*SOLAR_FIELDS}.issubset(omni.dtype.names):
        report["errors"].append("OMNI structured array must contain utc,Bx,By,Bz,V,P")
        return report
    times=np.asarray(omni["utc"],dtype="datetime64[s]")
    for case in cases:
        start=np.datetime64(case["acquisition_start"],"s")
        end=np.datetime64(case["acquisition_end"],"s")
        indices=(times>=start)&(times<=end)
        elapsed_start=(datetime.fromisoformat(case["acquisition_start"])-ova_date).total_seconds()/60
        elapsed_end=(datetime.fromisoformat(case["acquisition_end"])-ova_date).total_seconds()/60
        i0=int(np.floor(elapsed_start))
        i1=int(np.ceil(elapsed_end))
        enough=0<=i0<=i1<len(ova)
        vals=[float(np.nanmean(np.asarray(omni[k][indices],dtype=float))) if indices.any() else np.nan for k in SOLAR_FIELDS]
        finite=bool(np.all(np.isfinite(vals)))
        report["per_case"][case["case_id"]]={
            "omni_interval_records":int(indices.sum()),
            "omni_all_conditioning_means_finite":finite,
            "ovation_frame_start":i0,
            "ovation_frame_end":i1,
            "ovation_full_interval_available":enough}
        if not indices.any() or not finite or not enough:
            report["errors"].append(case["case_id"]+": incomplete/nonfinite OMNI/OVATION UTC support")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",type=Path,default=DEFAULT_PREPARED)
    p.add_argument("--widths",type=float,nargs="+",default=[1.,2.,4.])
    p.add_argument("--min-coverage",type=float,default=.5)
    p.add_argument("--windows-per-width",type=int,default=1)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--omni",type=Path,default=DEFAULT_OMNI)
    p.add_argument("--ovation",type=Path,default=DEFAULT_OVATION)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    source=args.prepared_root/"manifest.json"
    if not source.is_file():
        raise FileNotFoundError(f"Prepare inputs first: {source}")
    manifest=json.loads(source.read_text(encoding="utf-8"))
    records=manifest["cases"]
    report={"source_manifest":str(source),"n_catalog_cases":len(records),
            "selection_parameters":{
                "widths_h":args.widths,"min_coverage":args.min_coverage,
                "windows_per_width":args.windows_per_width,"seed":args.seed,
                "band_mlat":[60.,80.],"min_pixels":20,"candidate_step_h":.5},
            "case_audits":[],"errors":[],"sources":{}}
    try:
        for row in records:
            result=scan_case(args.prepared_root,row,args.widths,args.min_coverage,args.seed,args.windows_per_width)
            report["case_audits"].append(result)
            if not result["all_holdouts_evaluable"]:
                report["errors"].append(row["case_id"]+": insufficient flux-blind observed sectors for one or more requested widths")
        report["sources"]=check_inputs(manifest["month"],records,args.omni,args.ovation)
        report["errors"]+=report["sources"].get("errors",[])
    except (ValueError,KeyError,FileNotFoundError,AssertionError,TypeError) as exc:
        report["errors"].append(str(exc))
    report["n_fully_evaluable_acquisitions"]=sum(int(r["all_holdouts_evaluable"]) for r in report["case_audits"])
    report["ready_for_primary_inference"]=len(records)>=6 and len(report["case_audits"])==len(records) and len(report["errors"])==0
    output=args.output or args.prepared_root.parent/"preflight_multievent_report.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"n_catalog_cases":len(records),
          "n_fully_evaluable_acquisitions":report["n_fully_evaluable_acquisitions"],
          "ready_for_primary_inference":report["ready_for_primary_inference"],
          "errors":report["errors"],"output":str(output)},indent=2))
    if not report["ready_for_primary_inference"]:
        raise SystemExit("Preflight failed: inspect report before running any DDPM evaluations.")

if __name__=="__main__":
    main()

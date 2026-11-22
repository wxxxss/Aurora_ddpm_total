#!/usr/bin/env python3
"""Time-selected independent-ACQUISITION SSUSI catalog (not storm-independent).

From repository root:
 python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py --dry-run
 python paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5/prepare_multievent_ssusi.py

Uses the same center-aligned native UT_N>0 binning as second revision.
Selection uses acquisition dates and EDR file availability ONLY: no energy
flux values, OVATION scores, or model predictions determine which orbit is
chosen. Original Figure6/7 orbits are excluded by default to avoid
re-evaluating the two development examples as new validation events.

Only 2005-01 has been preconfigured with OMNI/OVATION arrays. For another
month, supply appropriately time-aligned arrays to the benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/"paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13"
sys.path.insert(0,str(PREP))
from prepare_center_aligned_ssusi import read_raw, native_to_centered, LAT_CENTERS, MLT_CENTERS, sha256

DEFAULT_EDR=Path("/home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2005")
DEFAULT_OUTPUT=HERE/"independent_cases"
FILENAME_RE=re.compile(r"_(20\d{5})T(\d{6})-(20\d{5})T(\d{6})-REV(\d+)",re.I)
EXCLUDED_ORBITS={"06228","06263"}

def parse_edr_filename(path):
    match=FILENAME_RE.search(path.name)
    if not match: return None
    a=datetime.strptime(match.group(1)+match.group(2),"%Y%j%H%M%S")
    b=datetime.strptime(match.group(3)+match.group(4),"%Y%j%H%M%S")
    if b<=a:raise ValueError(f"EDR has non-positive interval: {path}")
    return {"path":path,"start":a,"end":b,"midpoint":a+(b-a)/2,"orbit":match.group(5)}

def month_bounds(month):
    start=datetime.strptime(month+"-01","%Y-%m-%d")
    end=datetime(start.year+(start.month==12),start.month%12+1,1)
    return start,end

def select_acquisitions(records, month, n, min_gap_h, include_figures=False):
    if n<1 or min_gap_h<0:raise ValueError("Invalid catalog limits")
    start,end=month_bounds(month)
    eligible=[r for r in records if start<=r["start"] and r["end"]<end and
              (include_figures or r["orbit"] not in EXCLUDED_ORBITS)]
    eligible.sort(key=lambda r:(r["midpoint"],str(r["path"])))
    chosen=[]
    # Space *target dates* over the entire month rather than choosing the
    # earliest records or the most aurorally active acquisitions.
    goals=[start+(end-start)*((k+0.5)/n) for k in range(n)]
    for goal in goals:
        for row in sorted(eligible,key=lambda r:(abs((r["midpoint"]-goal).total_seconds()),r["midpoint"])):
            if row in chosen:continue
            if any(not (row["start"]>=c["end"]+timedelta(hours=min_gap_h)
                   or c["start"]>=row["end"]+timedelta(hours=min_gap_h))
                   for c in chosen):
                continue
            chosen.append(row)
            break
    return sorted(chosen,key=lambda r:r["start"]),len(eligible)

def case_name(r):
    return f"F16_{r['start'].strftime('%Y%jT%H%M%S')}_REV{r['orbit']}"

def prepare_case(r,output_root):
    case=case_name(r)
    flux,lat,mlt,ut,attrs=read_raw(r["path"])
    grid,mask,counts,audit=native_to_centered(flux,lat,mlt,ut)
    case_dir=output_root/case
    case_dir.mkdir(parents=True,exist_ok=True)
    npz=case_dir/(case+"_swath_case.npz")
    meta_path=case_dir/(case+"_metadata.json")
    if npz.exists() or meta_path.exists():
        raise FileExistsError(f"Will not overwrite an existing output: {case_dir}")
    np.savez_compressed(npz,flux_grid=grid,obs_mask=mask,count_grid=counts,
                        mlat_1d=LAT_CENTERS,mlt_1d=MLT_CENTERS)
    meta={
        "case_name":case,
        "matched_edr_file":str(r["path"]),
        "source_edr_sha256":sha256(r["path"]),
        "source_orbit":r["orbit"],
        "edr_midpoint":r["midpoint"].isoformat(),
        "acquisition_start":r["start"].isoformat(),
        "acquisition_end":r["end"].isoformat(),
        "obs_fraction":audit["supported_fraction"],
        "mean_flux":float(np.nanmean(grid)) if mask.any() else None,
        "max_flux":float(np.nanmax(grid)) if mask.any() else None,
        "netcdf_metadata":attrs,
        "grid_assignment_method":"nearest 80x96 MLAT-MLT center, UT_N>0 native support, duplicate-bin average",
        "selection_protocol":"pre-selected by orbital acquisition time, excluding Figure6/7 development cases"
    }
    meta_path.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    return {"case_id":case,"orbit":r["orbit"],
            "acquisition_start":r["start"].isoformat(),
            "acquisition_end":r["end"].isoformat(),
            "midpoint":r["midpoint"].isoformat(),
            "source_file":str(r["path"]),
            "source_sha256":meta["source_edr_sha256"],
            "n_supported_cells":audit["supported_cells"],
            "fraction_supported":audit["supported_fraction"],
            "valid_native_samples":audit["raw_valid_samples"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edr-root",type=Path,default=DEFAULT_EDR)
    p.add_argument("--month",default="2005-01")
    p.add_argument("--max-events",type=int,default=12)
    p.add_argument("--min-events",type=int,default=6)
    p.add_argument("--min-gap-hours",type=float,default=24)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--include-figure-events",action="store_true",
                   help="Exploratory only: includes orbits used to develop earlier Figure6/7 results")
    args=p.parse_args()
    if not args.edr_root.exists():raise FileNotFoundError(args.edr_root)
    allpaths=sorted(args.edr_root.rglob("*.nc"))
    parsed=[]
    for path in allpaths:
        x=parse_edr_filename(path)
        if x is not None:parsed.append(x)
    selected,n_eligible=select_acquisitions(parsed,args.month,args.max_events,args.min_gap_hours,args.include_figure_events)
    metadata={
      "experiment":"Time-based, flux-blind SSUSI EDR acquisition selection",
      "month":args.month,
      "n_edr_paths_scanned":len(allpaths),
      "n_eligible_month_paths":n_eligible,
      "n_selected":len(selected),
      "min_temporal_gap_hours":args.min_gap_hours,
      "max_events_requested":args.max_events,
      "include_development_figures":bool(args.include_figure_events),
      "caveat":"Nonoverlapping orbital acquisitions spaced by 24h are separate event CLUSTERS for analysis; they are not automatically independent geomagnetic storms or years.",
      "cases":[{"case_id":case_name(r),"orbit":r["orbit"],
                "acquisition_start":r["start"].isoformat(),
                "acquisition_end":r["end"].isoformat(),"source_file":str(r["path"])}
               for r in selected]
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        path=args.output_root/"catalog_preview.json"
        path.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(metadata,indent=2))
        return
    if len(selected)<args.min_events:
        raise RuntimeError(f"Only {len(selected)} date-separated EDR acquisitions selected, "
                           f"but --min-events={args.min_events}. Add more original EDR files "
                           "or extend the date window; do not imply independent-event validation.")
    for r in selected:
        case=prepare_case(r,args.output_root)
        metadata["cases"][next(i for i,x in enumerate(metadata["cases"]) if x["case_id"]==case["case_id"])]=case
        print(case["case_id"],case["n_supported_cells"],"supported target cells")
    metadata["n_prepared"]=len(selected)
    (args.output_root/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(selected)} temporally spaced acquisitions. Manifest: {args.output_root/'manifest.json'}")

if __name__=="__main__":
    main()

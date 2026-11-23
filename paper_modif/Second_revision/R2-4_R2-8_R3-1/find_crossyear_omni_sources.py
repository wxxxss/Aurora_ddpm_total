#!/usr/bin/env python3
"""Locate genuine 2006 OMNI sources for an independent-year SSUSI benchmark.

The cross-year survey checked only:
  AuroraData/omni_real_data/omni_5min/2006/omni_2006MM01_5min.npy
Finding nothing THERE does not imply that all raw OMNI inputs are missing.

This script searches only the expected local OMNI roots for 2006-source
CDF files or existing NPY arrays and verifies NPY time axes and conditioning
field schema where feasible. NO bytes are copied, written to the model input
tree or automatically interpolated. Never reuse 2005 solar conditions or
OVATION targets for a 2006 EDR interval.

From repository root:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/find_crossyear_omni_sources.py

Output:
 paper_modif/Second_revision/R2-4_R2-8_R3-1/crossyear_omni_sources.json
"""
from __future__ import annotations

import argparse
import calendar
import json
from datetime import datetime
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=Path("/home/docker/data")
DEFAULT_SEARCH_ROOTS=[
 BASE/"ro-share/omni/omni_cdaweb/hro2_5min",
 BASE/"ro-share/omni/omni_cdaweb/hro_5min",
 BASE/"ro-share/omni/omni_cdaweb/hourly",
 BASE/"private/AuroraData/omni_real_data/omni_5min",
 BASE/"private/AuroraData/omni_real_data/hourly_r1_2",
]
COND=("Bx","By","Bz","V","P")

def inspect_npy(path,year):
    r={"source_type":"NPY","path":str(path),"size_bytes":path.stat().st_size,
       "read_status":"unread","fields":None,"shape":None,
       "utc_min":None,"utc_max":None,"utc_median_step_minutes":None,
       "contains_requested_year":False,"has_5_conditioning_fields":False}
    if r["size_bytes"]==0:
        r.update(read_status="empty",error="zero-byte NPY file")
        return r
    try:
        # OMNI timestamp fields can include datetime64 or Python objects.
        data=np.load(path,allow_pickle=True,mmap_mode=None)
        r["shape"]=list(data.shape)
        r["fields"]=list(data.dtype.names or [])
        names=set(data.dtype.names or ())
        r["has_5_conditioning_fields"]=all(field in names for field in COND)
        if "utc" in names and data.size:
            time=np.asarray(data["utc"],dtype="datetime64[s]").reshape(-1)
            good=np.isnat(time)==False
            if good.any():
                time=time[good]
                r["utc_min"]=str(time.min())
                r["utc_max"]=str(time.max())
                r["contains_requested_year"]=bool(
                    np.any((time>=np.datetime64(f"{year}-01-01"))&
                           (time<np.datetime64(f"{year+1}-01-01"))))
                if len(time)>1:
                    dif=np.diff(time).astype("timedelta64[s]").astype(np.int64)
                    r["utc_median_step_minutes"]=float(np.median(dif)/60.)
                    r["utc_has_duplicate_or_reverse"]=bool((dif<=0).any())
        r["read_status"]="valid"
    except (EOFError,ValueError,TypeError,OSError,MemoryError) as exc:
        r.update(read_status="unreadable",error=f"{type(exc).__name__}: {exc}")
    return r

def inspect_cdf(path):
    # Deliberately avoid loading multi-GB CDF variable arrays.
    return {"source_type":"CDF","path":str(path),
            "size_bytes":path.stat().st_size,
            "read_status":"empty" if path.stat().st_size==0 else "header_not_yet_checked",
            "utc_range_verified":False,
            "note":"Candidate ONLY: filename/year match is not proof of complete 2006 5-min UTC coverage or required CDF variables. Validate before conversion."}

def find(year,roots,max_files=5000):
    if max_files<1:raise ValueError("max_files must be positive")
    discovered=[]
    checked=[]
    dedup=set()
    for root in roots:
        root=Path(root)
        checked.append({"root":str(root),"exists":root.is_dir()})
        if not root.is_dir():continue
        # Search likely year subfolders first, then the OMNI product root.
        # Avoid scanning /home/docker/data or the entire author's filesystem.
        for candidate in [root/str(year),root]:
            if not candidate.is_dir():continue
            for pattern in (f"*{year}*.cdf",f"*{year}*.CDF",
                            f"*{year}*.npy",f"*{year}*.npz"):
                for path in candidate.rglob(pattern):
                    if not path.is_file():continue
                    k=str(path.resolve())
                    if k in dedup:continue
                    dedup.add(k)
                    if len(discovered)>=max_files:
                        return {"search_roots":checked,"files":discovered,
                                "truncated_at_max_files":True}
                    if path.suffix.lower()==".npy":
                        row=inspect_npy(path,year)
                    elif path.suffix.lower()==".cdf":
                        row=inspect_cdf(path)
                    else:
                        row={"source_type":"NPZ","path":str(path),
                             "size_bytes":path.stat().st_size,
                             "read_status":"manual_schema_review_required",
                             "note":"Archive member schema and UTC index have not yet been verified"}
                    discovered.append(row)
    return {"search_roots":checked,"files":discovered,
            "truncated_at_max_files":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year",type=int,default=2006)
    p.add_argument("--roots",type=Path,nargs="+",default=DEFAULT_SEARCH_ROOTS)
    p.add_argument("--max-files",type=int,default=5000)
    p.add_argument("--output",type=Path,default=HERE/"crossyear_omni_sources.json")
    args=p.parse_args()
    result=find(args.year,args.roots,args.max_files)
    result["target_year"]=args.year
    result["n_candidates"]=len(result["files"])
    result["n_cdf_candidates"]=sum(x["source_type"]=="CDF" for x in result["files"])
    result["n_npy_candidates_with_2006_utc_and_schema"]=sum(
        x["source_type"]=="NPY" and x["contains_requested_year"] and
        x["has_5_conditioning_fields"] for x in result["files"])
    result["cautions"]=[
        "Files matching the year number are candidates only; CDF variable schema and datetime coverage must be checked before use.",
        "Hourly OMNI is not equivalent to 5-minute OMNI for an exact comparison; interpolation may change the conditions and must be documented.",
        "Do not regenerate existing OVATION maps solely because matching OMNI conditioning arrays are absent at the expected path.",
        "The 2006 March OVATION monthly file is zero bytes and should not be used.",
        "Do not copy candidate files into model input folders or modify old baseline results without provenance checks."
    ]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "year":args.year,"n_candidates":result["n_candidates"],
        "n_cdf_candidates":result["n_cdf_candidates"],
        "n_npy_with_conditioning_and_2006_utc":
            result["n_npy_candidates_with_2006_utc_and_schema"],
        "first_candidates":[{
            "path":x["path"],"source_type":x["source_type"],
            "size_bytes":x["size_bytes"],
            "utc_min":x.get("utc_min"),"utc_max":x.get("utc_max"),
            "step_minutes":x.get("utc_median_step_minutes")}
            for x in result["files"][:20]],
        "output":str(args.output)
    },indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()

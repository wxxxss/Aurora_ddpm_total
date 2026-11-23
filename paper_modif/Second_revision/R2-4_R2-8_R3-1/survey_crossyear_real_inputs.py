#!/usr/bin/env python3
"""R2-8/R3-1: find truly observed DMSP F16/SSUSI data OUTSIDE January 2005.

Inventory-only; no weights, full arrays, model inference, or SSUSI flux read.
For each requested year/month: count original F16 .nc EDRs, check that the
matching five-minute OMNI and OVATION monthly arrays are present, and read
only the NPY headers (mmap) to classify the OVATION time resolution.

Importantly, a real SSUSI EDR file without matching UTC-conditioned
OMNI and OVATION inputs does NOT constitute a ready-to-run validation case.

From repo root:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/survey_crossyear_real_inputs.py

The F16 satellite may not exist for some early comparison years; do NOT
invent satellite cases to match the 2001 synthetic-only OVATION set.
"""
from __future__ import annotations
import argparse,calendar,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=Path("/home/docker/data/private/AuroraData")
DEFAULT_YEARS=(2004,2006,2007,2008,2009)
DEFAULT_OUT=HERE/"crossyear_real_input_survey.json"

def audit_month(year,month,root):
    # F16 originals use Julian day directories, not YYYY/MM.
    yearroot=root/"real_aurora_data_ssusi/f16"/str(year)
    days=calendar.monthrange(year,month)[1]
    start_yd=1+sum(calendar.monthrange(year,i)[1] for i in range(1,month))
    edrs=[]
    if yearroot.is_dir():
        for yday in range(start_yd,start_yd+days):
            folder=yearroot/f"{yday:03d}"
            if folder.is_dir():
                edrs.extend(folder.glob("*.nc"))
    omni=root/"omni_real_data/omni_5min"/str(year)/f"omni_{year}{month:02d}01_5min.npy"
    ovation=root/"generated_aurora_data"/f"{year}_omni_aurora"/f"aurora_img_{year}{month:02d}01.npy"
    report={
        "year":year,"month":month,"n_native_ssusi_edr":len(edrs),
        "omni_5min_exists":omni.is_file(),
        "ovation_npy_exists":ovation.is_file(),
        "omni_file_size_bytes":omni.stat().st_size if omni.is_file() else None,
        "ovation_file_size_bytes":ovation.stat().st_size if ovation.is_file() else None,
        "omni_status":"missing","ovation_status":"missing",
        "omni_error":None,"ovation_error":None,
        "omni_shape":None,"ovation_shape":None,
        "omni_utc_field_present":False,
        "ovation_expected_grid_80x96":False,
        "ovation_calendar_cadence":None,
        "native_observation_year_disjoint_from_1996":year!=1996,
        "not_january_2005":(year,month)!=(2005,1),
    }
    if omni.is_file():
        if report["omni_file_size_bytes"]==0:
            report["omni_status"]="empty"
            report["omni_error"]="Zero-byte file: array was not generated or writing failed"
        else:
            try:
                # OMNI may include object-valued timestamp fields.
                arr=np.load(omni,allow_pickle=True)
                report["omni_shape"]=list(arr.shape)
                report["omni_utc_field_present"]=bool(
                    arr.dtype.names and "utc" in arr.dtype.names)
                report["omni_status"]="valid" if (
                    report["omni_utc_field_present"] and len(arr)>0
                ) else "invalid_schema"
                if report["omni_status"]!="valid":
                    report["omni_error"]="Missing utc field or zero OMNI records"
            except (EOFError, ValueError, OSError, TypeError, MemoryError) as exc:
                report["omni_status"]="unreadable"
                report["omni_error"]=f"{type(exc).__name__}: {exc}"
    if ovation.is_file():
        if report["ovation_file_size_bytes"]==0:
            report["ovation_status"]="empty"
            report["ovation_error"]="Zero-byte file: array was not generated or writing failed"
        else:
            try:
                # Only map a valid NPY header/data range; never load full cubes.
                arr=np.load(ovation,mmap_mode="r",allow_pickle=False)
                report["ovation_shape"]=list(arr.shape)
                n=days*24*60
                n5=days*24*12
                report["ovation_calendar_cadence"] = (
                    "1-minute full month" if len(arr)==n
                    else "5-minute full month" if len(arr)==n5
                    else "nonstandard / incomplete")
                report["ovation_expected_grid_80x96"]=bool(
                    arr.ndim==3 and tuple(arr.shape[1:])==(80,96))
                report["ovation_status"]="valid" if (
                    report["ovation_expected_grid_80x96"] and
                    report["ovation_calendar_cadence"] in (
                        "1-minute full month","5-minute full month")
                ) else "invalid_schema"
                if report["ovation_status"]!="valid":
                    report["ovation_error"]="Unexpected grid shape or incomplete/nonstandard monthly sequence"
                del arr
            except (EOFError, ValueError, OSError, TypeError, MemoryError) as exc:
                report["ovation_status"]="unreadable"
                report["ovation_error"]=f"{type(exc).__name__}: {exc}"
    report["ready_for_preflight_candidate_scan"]=bool(
        len(edrs)>0 and report["omni_status"]=="valid" and
        report["ovation_status"]=="valid")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years",type=int,nargs="+",default=DEFAULT_YEARS)
    p.add_argument("--data-root",type=Path,default=BASE)
    p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    records=[audit_month(y,m,args.data_root) for y in args.years for m in range(1,13)]
    viable=[r for r in records if r["ready_for_preflight_candidate_scan"]]
    failures=[
        {"year":r["year"],"month":r["month"],
         "n_EDRs":r["n_native_ssusi_edr"],
         "omni_status":r["omni_status"],
         "ovation_status":r["ovation_status"],
         "omni_error":r["omni_error"],
         "ovation_error":r["ovation_error"],
         "omni_bytes":r["omni_file_size_bytes"],
         "ovation_bytes":r["ovation_file_size_bytes"]}
        for r in records if r["n_native_ssusi_edr"]>0 and
        (r["omni_status"]!="valid" or r["ovation_status"]!="valid")
    ]
    report={
        "scope":"DMSP F16 real observed SSUSI EDR, disjoint in calendar year from January 2005 and April-May 1996 training",
        "years_checked":args.years,
        "n_months_checked":len(records),
        "n_months_with_any_ssusi":sum(r["n_native_ssusi_edr"]>0 for r in records),
        "n_months_with_all_potential_inputs":len(viable),
        "n_ssusi_months_with_missing_or_unreadable_inputs":len(failures),
        "ssusi_month_input_issues":failures,
        "potentially_usable_months":[{"year":r["year"],"month":r["month"],
             "n_EDRs":r["n_native_ssusi_edr"],
             "ovation_calendar_cadence":r.get("ovation_calendar_cadence")}
             for r in viable],
        "months":records,
        "cautions":[
            "File existence and NPY shape alone do not establish valid satellite measurement coverage, UTC alignment, or suitable Kp/event diversity.",
            "The 1996 source has matching five-minute input arrays, but current checkpoint training lineage remains unverified.",
            "Months without complete OVATION/OMNI arrays require generation and timestamp verification before baseline/inference, not silent reuse of 2005 arrays.",
            "A multi-year real SSUSI test should select orbit dates before seeing target flux or model error and must not use 2005 development cases."
        ]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("years_checked","n_months_checked",
        "n_months_with_any_ssusi","n_months_with_all_potential_inputs",
        "potentially_usable_months",
        "n_ssusi_months_with_missing_or_unreadable_inputs",
        "ssusi_month_input_issues")},indent=2))

if __name__=="__main__":main()

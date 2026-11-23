#!/usr/bin/env python3
"""Reselect 2006 Jan F16/SSUSI EDRs BEFORE inference by ORIGINAL OMNI quality.

Scientific protocol
-------------------
The initial time-only, flux-blind 12-orbit January 2006 cohort failed raw
five-field OMNI completeness in 4 orbits. NEVER interpolate the 825-minute
outage. Retain the 8 old, fully raw-valid EDRs. For each of the 4 invalid
original EDRs, choose the nearest-in-time replacement among the other
January EDRs whose original five-field HRO2 5-min OMNI observations are
100% valid within that EDR interval, subject to >=24 h separation BETWEEN
whole EDR intervals. Tie break is timestamp then source filename.

No SSUSI flux, NetCDF pixel arrays, OVATION targets or DDPM predictions
are consulted in selection. The 2006 original EDR file names only encode
time/orbit; the 2006 original OMNI CDF provides source-quality flags.
Conserve the old catalog in real_2006_01_inputs. Write the selected
cohort ONLY to real_2006_01_omni_qualified_inputs.

From repository root, in order:
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/reselect_2006_omni_qualified_ssusi.py --dry-run
 python paper_modif/Second_revision/R2-4_R2-8_R3-1/reselect_2006_omni_qualified_ssusi.py --prepare

The --prepare stage is safe ONLY if preview.ready_for_preparation is true.
New prepared native SSUSI gridding is performed only at --prepare, and
the output is never a replacement for earlier 2005 experiments.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
THIRD=REPO/"paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5"
if str(THIRD) not in sys.path:
    sys.path.insert(0,str(THIRD))

from prepare_multievent_ssusi import (
    parse_edr_filename, case_name, prepare_case, month_bounds, EXCLUDED_ORBITS
)
from audit_and_prepare_2006_omni import CDF_ROOT, load_legacy, get_sha256
from audit_2006_omni_event_coverage import calendar_validity, audit_acquisition

DEFAULT_EDR=Path("/home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2006")
DEFAULT_CDF=CDF_ROOT/"2006/omni_hro2_5min_20060101_v01.cdf"
DEFAULT_OLD=HERE/"real_2006_01_inputs/catalog_preview.json"
DEFAULT_OLD_AUDIT=HERE/"omni_200601_event_coverage_audit.json"
DEFAULT_NEW=HERE/"real_2006_01_omni_qualified_inputs"

def temporal_separation(a,b,min_gap_h=24.):
    """Whole EDR intervals must be separated, not just orbit midpoints."""
    gap=timedelta(hours=min_gap_h)
    return a["start"]>=b["end"]+gap or b["start"]>=a["end"]+gap

EDR_VERSION_RE=re.compile(
    r"_v([A-Za-z])(\d+(?:\.\d+)*)r(\d+)(?=\.nc$)", re.IGNORECASE
)

def source_version_key(record):
    """Explicit processing version for deterministic identical-time duplicates."""
    m=EDR_VERSION_RE.search(record["path"].name)
    if m is None:
        return (-1,(-1,-1,-1,-1),-1)
    numeric=tuple(int(v) for v in m.group(2).split("."))[:4]
    return (ord(m.group(1).upper()), numeric+(-1,)*(4-len(numeric)),
            int(m.group(3)))

def deduplicate_edr_versions(parsed,old_cases=None):
    """Deduplicate only by filename time/orbit and record every decision.

    Preserve EXACT source file used in initial selection; exclude NEW
    ambiguous end times. For identical intervals, prefer highest
    processing version then lexical path. No SSUSI flux is read.
    """
    original={r["case_id"]:r for r in (old_cases or [])}
    groups={}
    for rec in parsed:
        groups.setdefault(case_name(rec),[]).append(rec)
    selected=[]
    collision_log=[]
    for name,variants in sorted(groups.items()):
        if len(variants)==1:
            selected.append(variants[0])
            continue
        intervals={(v["start"],v["end"]) for v in variants}
        chosen=None
        if name in original:
            source=original[name].get("source_file")
            if not source:
                raise ValueError(f"Preselected {name} is missing original source_file")
            matching=[v for v in variants if str(v["path"])==source]
            if len(matching)!=1:
                raise ValueError(f"Original EDR source unavailable or repeated: {name}, {source}")
            chosen=matching[0]
            reason="preserve_exact_original_catalog_file"
        elif len(intervals)>1:
            reason="conflicting_end_times_excluded"
        else:
            # Stable descending version / ascending path. Never use
            # NetCDF flux, observed coverage or reconstruction metrics.
            chosen=min(variants,key=lambda v:(
                tuple(-x for x in (source_version_key(v)[0],
                                    *source_version_key(v)[1],
                                    source_version_key(v)[2])),
                str(v["path"])))
            reason="highest_processing_version_then_lexical_path"
        collision_log.append({
            "case_id":name,"n_files":len(variants),
            "n_distinct_time_intervals":len(intervals),
            "resolution":reason,
            "selected_source_file":str(chosen["path"]) if chosen else None,
            "versions":[{
                "source_file":str(v["path"]),
                "start":v["start"].isoformat(),
                "end":v["end"].isoformat(),
                "filename_version_key":list(source_version_key(v))
            } for v in sorted(variants,key=lambda r:str(r["path"]))]
        })
        if chosen is not None:
            selected.append(chosen)
    selected.sort(key=lambda r:(r["midpoint"],str(r["path"])))
    return selected,collision_log

def scan_month(edr_root,month,old_cases=None):
    start,end=month_bounds(month)
    paths=sorted(edr_root.rglob("*.nc"))
    parsed=[]
    rejected=[]
    for path in paths:
        try:
            record=parse_edr_filename(path)
        except ValueError as exc:
            rejected.append({"file":str(path),"reason":str(exc)})
            continue
        if record is None:continue
        if start<=record["start"] and record["end"]<end and record["orbit"] not in EXCLUDED_ORBITS:
            parsed.append(record)
    parsed.sort(key=lambda r:(r["midpoint"],str(r["path"])))
    selected,collisions=deduplicate_edr_versions(parsed,old_cases)
    return selected,{
       "n_edr_files_scanned":len(paths),
       "n_calendar_eligible_files":len(parsed),
       "n_unique_usable_acquisitions":len(selected),
       "n_duplicate_identity_groups":len(collisions),
       "n_extra_duplicate_files":sum(x["n_files"]-1 for x in collisions),
       "n_conflicting_end_time_groups_excluded":sum(
           x["resolution"]=="conflicting_end_times_excluded" for x in collisions),
       "duplicate_identity_resolution":collisions,
       "n_filename_rejected_any_month":len(rejected),
       "filename_rejection_examples":rejected[:15]
    }

def source_quality_index(records,times,mask):
    quality={}
    for rec in records:
        name=case_name(rec)
        try:
            data=audit_acquisition({
                "case_id":name,
                "acquisition_start":rec["start"].isoformat(),
                "acquisition_end":rec["end"].isoformat()
            },times,mask)
        except ValueError as exc:
            # Some source EDRs may be too short to contain an OMNI tick.
            # They are unusable, but must not abort the full January scan.
            data={
                "case_id":name,"raw_fully_valid_acquisition":False,
                "n_omni_5min_records_inside":0,
                "n_any_field_invalid_inside":0,
                "n_invalid_by_field":{k:None for k in ("Bx","By","Bz","V","P")},
                "quality_error":str(exc)
            }
        quality[name]=data
    return quality

def choose_quality_cohort(old_cases,records,quality,min_gap_h=24.,expected=12):
    if expected<1 or min_gap_h<0:
        raise ValueError("Invalid selection limits")
    if len(old_cases)!=expected:
        raise ValueError(f"Old catalog expected {expected} cases, got {len(old_cases)}")
    by_id={case_name(x):x for x in records}
    if len(by_id)!=len(records):
        raise ValueError("Duplicate EDR identifiers")
    old_ids=[row["case_id"] for row in old_cases]
    if len(set(old_ids))!=len(old_ids):
        raise ValueError("Original EDR catalog contains duplicate IDs")
    if set(old_ids)-set(by_id):
        raise ValueError("Old catalog cases not found in original EDR directory")
    for row in old_cases:
        rec=by_id[row["case_id"]]
        if (rec["start"].isoformat()!=row["acquisition_start"] or
            rec["end"].isoformat()!=row["acquisition_end"]):
            raise ValueError("Original catalog acquisition times changed unexpectedly")
        if row.get("source_file") and str(rec["path"])!=row["source_file"]:
            raise ValueError("Original catalog EDR source path changed unexpectedly")
    retained=[by_id[name] for name in old_ids
              if quality[name]["raw_fully_valid_acquisition"]]
    invalid=[row for row in old_cases if
             not quality[row["case_id"]]["raw_fully_valid_acquisition"]]
    if any(not temporal_separation(a,b,min_gap_h) for i,a in enumerate(retained) for b in retained[i+1:]):
        raise ValueError("Old valid acquisitions do not satisfy claimed interval separation")
    chosen=retained[:]
    chosen_ids={case_name(x) for x in chosen}
    # Prefer the original selected orbit's time as a fixed, pre-model
    # calendar sampling target. Replacements need NOT have same UTC date;
    # all must be fully source-qualified and independently separated.
    acceptable=[r for r in records if quality[case_name(r)]["raw_fully_valid_acquisition"]]
    replacements=[]
    failures=[]
    for original in invalid:
        first=datetime.fromisoformat(original["acquisition_start"])
        last=datetime.fromisoformat(original["acquisition_end"])
        goal=first+(last-first)/2
        ranked=sorted(acceptable,key=lambda r:(
            abs((r["midpoint"]-goal).total_seconds()),
            r["midpoint"],str(r["path"])))
        candidate=next((r for r in ranked
                        if case_name(r) not in chosen_ids and
                        all(temporal_separation(r,c,min_gap_h) for c in chosen)),None)
        if candidate is None:
            failures.append(original["case_id"])
            continue
        chosen.append(candidate)
        chosen_ids.add(case_name(candidate))
        replacements.append({
            "original_invalid_case":original["case_id"],
            "original_bad_5min_records":quality[original["case_id"]]["n_any_field_invalid_inside"],
            "replacement_case":case_name(candidate),
            "midpoint_offset_from_replaced_case_hours":round(
                 (candidate["midpoint"]-goal).total_seconds()/3600.,3),
            "replacement_has_100_percent_raw_valid_omni":True
        })
    chosen.sort(key=lambda r:(r["start"],r["orbit"]))
    all_separated=all(temporal_separation(a,b,min_gap_h)
                      for i,a in enumerate(chosen) for b in chosen[i+1:])
    if not all_separated:
        raise AssertionError("Temporal separation was violated")
    report={
        "n_original_cases":len(old_cases),
        "n_original_valid_preserved":len(retained),
        "n_original_invalid":len(invalid),
        "invalid_originals":[{
            "case_id":x["case_id"],
            "invalid_5min_records":quality[x["case_id"]]["n_any_field_invalid_inside"],
            "invalid_by_field":quality[x["case_id"]]["n_invalid_by_field"]}
            for x in invalid],
        "replacements":replacements,
        "original_cases_with_no_eligible_replacement":failures,
        "n_output_cases":len(chosen),
        "n_available_raw_omni_qualified_candidates":len(acceptable),
        "all_selected_raw_omni_valid":all(
            quality[case_name(x)]["raw_fully_valid_acquisition"] for x in chosen),
        "min_gap_hours":min_gap_h,
        "all_selected_intervals_separated":all_separated,
        "ready_for_preparation":len(chosen)==expected and not failures
    }
    return chosen,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edr-root",type=Path,default=DEFAULT_EDR)
    p.add_argument("--old-catalog",type=Path,default=DEFAULT_OLD)
    p.add_argument("--old-source-audit",type=Path,default=DEFAULT_OLD_AUDIT)
    p.add_argument("--prior-cdf-audit",type=Path,
        default=HERE/"omni_200601_cdf_audit.json")
    p.add_argument("--cdf",type=Path,default=DEFAULT_CDF)
    p.add_argument("--output-root",type=Path,default=DEFAULT_NEW)
    p.add_argument("--min-gap-hours",type=float,default=24.)
    p.add_argument("--target-events",type=int,default=12)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run",action="store_true")
    group.add_argument("--prepare",action="store_true")
    args=p.parse_args()
    if args.target_events!=12:
        raise ValueError("This protocol is pre-specified for the 12 original 2006 January calendar targets")
    if args.min_gap_hours<24:
        raise ValueError("Do not relax the already-published >=24-hour separation constraint")
    old=json.loads(args.old_catalog.read_text(encoding="utf-8"))
    if old["month"]!="2006-01":
        raise ValueError("Wrong input year-month")
    # The original CDF audit records the byte-level source used when
    # declaring the 825-minute gap. Do not unknowingly switch revisions.
    current_sha=get_sha256(args.cdf)
    if args.prior_cdf_audit.is_file():
        prior_cdf=json.loads(args.prior_cdf_audit.read_text(encoding="utf-8"))
        original_sha=prior_cdf.get("source_cdf_sha256")
        if original_sha and original_sha!=current_sha:
            raise RuntimeError("2006 source CDF SHA differs from previous gap audit. Re-audit before cohort selection.")
    legacy=load_legacy()
    df,meta=legacy.extract_raw_dataframe(args.cdf,year=2006)
    times,valid=calendar_validity(df,2006,1,legacy.VALID_RANGES)
    records,scan=scan_month(args.edr_root,"2006-01",old_cases=old["cases"])
    quality=source_quality_index(records,times,valid)
    # Compare the author's prior original-quality audit. A changed raw
    # CDF / timestamp interpretation must not silently change eligibility.
    if args.old_source_audit.is_file():
        prior=json.loads(args.old_source_audit.read_text(encoding="utf-8"))
        for row in prior["cases"]:
            old_result=quality.get(row["case_id"])
            if old_result is None or (
                bool(old_result["raw_fully_valid_acquisition"])!=
                bool(row["raw_fully_valid_acquisition"])):
                raise RuntimeError("Original 12-case OMNI quality differs from previously audited source")
    selected,selection=choose_quality_cohort(
        old["cases"],records,quality,args.min_gap_hours,args.target_events)
    ordered=[{
        "case_id":case_name(rec),"orbit":rec["orbit"],
        "acquisition_start":rec["start"].isoformat(),
        "acquisition_end":rec["end"].isoformat(),
        "source_file":str(rec["path"]),
        "n_omni_records":quality[case_name(rec)]["n_omni_5min_records_inside"],
        "n_invalid_omni_records":quality[case_name(rec)]["n_any_field_invalid_inside"],
        "raw_omni_valid":quality[case_name(rec)]["raw_fully_valid_acquisition"]
    } for rec in selected]
    summary={
        "experiment":"Independent 2006 real SSUSI time-selected, quality-qualified validation",
        "month":"2006-01",
        "selection_protocol":"Preserve originally time-selected raw-OMNI-valid acquisitions, replace only failed cases by nearest-time 100%-source-valid acquisition, >=24h separation; never inspect SSUSI flux, OVATION, or model performance",
        "selection_uses_ssusi_flux":False,
        "selection_uses_model_performance":False,
        "selection_uses_ovation_brightness":False,
        "independent_calendar_year_from_training_1996_and_prior_satellite_2005":True,
        "n_selected":len(ordered),
        "max_events_requested":args.target_events,
        "min_temporal_gap_hours":args.min_gap_hours,
        "include_development_figures":False,
        "source_cdf":str(args.cdf),
        "source_cdf_sha256":current_sha,
        "source_mapping":meta["model_to_source_mapping"],
        "original_time_only_catalog":str(args.old_catalog),
        "source_scan":scan,
        "quality_selection":selection,
        "cases":ordered,
        "ready_for_preparation":bool(selection["ready_for_preparation"]),
        "caveats":[
            "Source OMNI quality selection is not random and can alter the solar-wind activity distribution; report original four excluded cases and replacement rule.",
            "Independent calendar-year EDRs are not independent geomagnetic storms.",
            "Original training checkpoint lineage remains incompletely documented.",
            "Later native SSUSI support and artificial holdout coverage must pass separate model-free preflight."
        ]
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    preview=args.output_root/"catalog_preview.json"
    if args.dry_run:
        preview.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"n_source_EDRs":scan["n_unique_usable_acquisitions"],
          "n_duplicate_identity_groups":scan["n_duplicate_identity_groups"],
          "n_conflicting_end_time_groups_excluded":scan["n_conflicting_end_time_groups_excluded"],
          "n_raw_valid_source_EDRs":selection["n_available_raw_omni_qualified_candidates"],
          "retained":selection["n_original_valid_preserved"],
          "replaced":len(selection["replacements"]),
          "n_selected":len(ordered),
          "ready_for_preparation":summary["ready_for_preparation"],
          "replacement_details":selection["replacements"],
          "output":str(preview)},indent=2))
        return
    if not summary["ready_for_preparation"]:
        raise RuntimeError("Not all original failed EDRs can be replaced under the fixed timing/quality protocol. Report preview; do not relax rules.")
    if preview.exists():
        previously=json.loads(preview.read_text(encoding="utf-8"))
        if [(x["case_id"],x["source_file"]) for x in previously["cases"]] != [
                (x["case_id"],x["source_file"]) for x in ordered]:
            raise RuntimeError("Selected case IDs or EDR source versions differ from the previous preview.")
        if previously.get("source_cdf_sha256")!=current_sha:
            raise RuntimeError("Source 2006 OMNI CDF differs from the previous preview.")
    for rec in selected:
        details=prepare_case(rec,args.output_root)
        case=case_name(rec)
        item=next(x for x in summary["cases"] if x["case_id"]==case)
        item.update({k:v for k,v in details.items() if k not in item})
        path=args.output_root/case/(case+"_metadata.json")
        meta_case=json.loads(path.read_text(encoding="utf-8"))
        meta_case["selection_protocol"]=summary["selection_protocol"]
        meta_case["raw_2006_omni_valid"]=True
        meta_case["n_raw_2006_omni_records"]=quality[case]["n_omni_5min_records_inside"]
        path.write_text(json.dumps(meta_case,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(case,details["n_supported_cells"],"native supported cells")
    summary["n_prepared"]=len(selected)
    manifest=args.output_root/"manifest.json"
    manifest.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"n_prepared":len(selected),"manifest":str(manifest),
                      "n_retained":len(selected)-len(selection["replacements"]),
                      "n_replaced":len(selection["replacements"])},indent=2))
if __name__=="__main__":
    main()

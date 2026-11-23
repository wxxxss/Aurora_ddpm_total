#!/usr/bin/env python3
"""Select and optionally prepare NEW 2005-01 F16/SSUSI orbits by Kp alone.

This is a SEPARATE, activity-stratified evaluation cohort for R2-4. It does
NOT use auroral energy flux, reconstruction performance, or OVATION outputs
for orbit selection. The January 2005 12-orbit random/active validation
cohort and development Figure6/7 orbits are excluded by default.

Kp>5 is STRICT, not Kp>=5 or Kp>=4. Kp is matched from an hourly OMNI
product at EDR acquisition midpoint with <=31 minute tolerance, and the
3-hour Kp nature / within-orbit Kp transitions are documented.

Run first without loading native SSUSI arrays:
  python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_high_kp_real_ssusi.py --dry-run

Then, if independent high-Kp days are available, prepare a time-selected
grouped catalog for the existing third-batch benchmark wrapper:
  python paper_modif/Second_revision/R2-4_R2-8_R3-1/prepare_high_kp_real_ssusi.py --prepare

If fewer than 3 high-Kp calendar days remain in January, the evidence is
too sparse to claim robust strong-storm validation. Do not pad with multiple
near-identical passes through the same Kp/storm interval.
"""
from __future__ import annotations
import argparse,csv,json,sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
THIRD=ROOT/"paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5"
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(THIRD))
from kp_time_matching import DEFAULT_KP,load_hourly_kp,match_kp
from prepare_multievent_ssusi import parse_edr_filename,prepare_case,case_name

DEFAULT_EDR=Path("/home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2005")
DEFAULT_PREVIOUS=THIRD/"independent_cases/manifest.json"
DEFAULT_OUTPUT=HERE/"kp_stratified_cases"
EXCLUDE_DEV={"06228","06263"}

def excluded_ids(path):
    ids=set(EXCLUDE_DEV)
    if path.is_file():
        manifest=json.loads(path.read_text(encoding="utf-8"))
        for rec in manifest["cases"]:
            orbit=rec.get("orbit") or rec.get("source_orbit")
            if orbit:ids.add(str(orbit))
            elif "_REV" in rec["case_id"]:ids.add(rec["case_id"].split("_REV")[-1])
    return ids

def inspect_candidates(root,month,times,kps,excluded):
    observed=0
    missing_time=0
    bad=[]
    eligible=[]
    for file in sorted(root.rglob("*.nc")):
        record=parse_edr_filename(file)
        if record is None:continue
        if record["start"].strftime("%Y-%m")!=month:continue
        observed+=1
        if record["orbit"] in excluded:continue
        kp=match_kp(times,kps,record["start"],record["end"])
        if kp["group"]=="unavailable":
            missing_time+=1
            continue
        eligible.append({**record,"kp":kp})
    return eligible,{"n_monthly_raw_files":observed,
                     "n_time_matched_candidates":len(eligible),
                     "n_excluded_orbits":len(excluded),
                     "n_unmatched_kp_files":missing_time,
                     "n_invalid_file_intervals":len(bad)}

def evenly_pick_days(rows,max_events,forbidden_days=frozenset()):
    # Choose <=1 EDR per UTC calendar day. Preferred pass: acquisition
    # nearest midday, not the brightest satellite flux nor largest Kp.
    per_day=defaultdict(list)
    for r in rows:
        day=r["midpoint"].date().isoformat()
        if day not in forbidden_days:per_day[day].append(r)
    pick=[]
    for day,group in sorted(per_day.items()):
        target=datetime.fromisoformat(day+"T12:00:00")
        chosen=min(group,key=lambda r:(abs((r["midpoint"]-target).total_seconds()),
                                       r["midpoint"],str(r["path"])))
        pick.append(chosen)
    if len(pick)<=max_events:return pick
    # Time-distributed deterministic subsample of available days.
    n=max_events
    positions=[int((i+.5)*len(pick)/n) for i in range(n)]
    return [pick[i] for i in positions]

def select_kp_cohorts(records,max_each):
    high=[r for r in records if r["kp"]["group"]=="Kp>5"]
    quiet=[r for r in records if r["kp"]["group"]=="Kp<=3"]
    storm=evenly_pick_days(high,max_each)
    highdays={r["midpoint"].date().isoformat() for r in storm}
    low=evenly_pick_days(quiet,min(len(storm),max_each),forbidden_days=highdays)
    return storm,low,{
       "n_high_kp_eligible_edorbits":len(high),
       "n_high_kp_distinct_days":len({r["midpoint"].date().isoformat() for r in high}),
       "n_quiet_eligible_edorbits":len(quiet),
       "n_quiet_distinct_days":len({r["midpoint"].date().isoformat() for r in quiet}),
    }

def export_record(rec):
    x=rec["kp"]
    return {"case_id":case_name(rec),"orbit":rec["orbit"],
      "acquisition_start":rec["start"].isoformat(),
      "acquisition_end":rec["end"].isoformat(),
      "kp_midpoint":x["kp_midpoint"],
      "kp_group":x["group"],
      "kp_values_change_within_acquisition":x["kp_values_change_within_acquisition"],
      "kp_inside_min":x["kp_inside_acquisition_min"],
      "kp_inside_max":x["kp_inside_acquisition_max"],
      "source_file":str(rec["path"])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edr-root",type=Path,default=DEFAULT_EDR)
    p.add_argument("--previous-manifest",type=Path,default=DEFAULT_PREVIOUS)
    p.add_argument("--kp-hourly",type=Path,default=DEFAULT_KP)
    p.add_argument("--month",default="2005-01")
    p.add_argument("--max-per-group",type=int,default=6)
    p.add_argument("--min-distinct-high-days",type=int,default=3)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run",action="store_true")
    group.add_argument("--prepare",action="store_true")
    args=p.parse_args()
    if args.month!="2005-01":
        raise ValueError("Current benchmark's time-aligned OVATION is January 2005 only; do not select another month without a corresponding OVATION series")
    if args.max_per_group<1:raise ValueError("max-per-group must be positive")
    times,kps=load_hourly_kp(args.kp_hourly)
    prior=excluded_ids(args.previous_manifest)
    if not args.previous_manifest.is_file():
        raise FileNotFoundError("Primary 12-acquisition validation manifest is required to exclude previous test cases")
    records,scan=inspect_candidates(args.edr_root,args.month,times,kps,prior)
    high,quiet,avail=select_kp_cohorts(records,args.max_per_group)
    chosen=sorted(high+quiet,key=lambda r:r["start"])
    metadata={
      "month":args.month,
      "protocol":"Kp-only, unobserved-flux-blind activity-stratified cohort; at most one EDR per UTC date; high strict Kp>5, quiet Kp<=3.",
      "selection_uses_heldout_ssusi_flux":False,
      "kp_hourly_source":str(args.kp_hourly),
      "excluded_development_and_primary_orbits":sorted(prior),
      "scan":scan,
      "availability":avail,
      "n_distinct_high_kp_dates":len(high),
      "min_distinct_high_kp_dates_required_for_storm_stratum":args.min_distinct_high_days,
      "high_kp_sufficient_for_descriptive_cohort":len(high)>=args.min_distinct_high_days,
      "n_selected":len(chosen),
      "n_selected_high_kp":len(high),
      "n_selected_quiet":len(quiet),
      "include_development_figures":False,
      "cases":[export_record(r) for r in chosen],
      "limits":[
         "One EDR per UTC day reduces close-orbit pseudo-replication but does not ensure independent storms.",
         "The hourly OMNI Kp product may contain forward/back-filled three-hour values; quantify source quality before treating as event-level truth.",
         "Kp>5 at orbit midpoint does not guarantee the same Kp throughout the EDR acquisition.",
         "Prepared inputs by themselves provide no model accuracy evidence."
      ]
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        out=args.output_root/"kp_catalog_preview.json"
        out.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"scan":scan,"availability":avail,
         "n_high_kp_distinct_dates":len(high),
         "n_quiet_selected":len(quiet),
         "n_total_selected":len(chosen),
         "high_kp_sufficient":metadata["high_kp_sufficient_for_descriptive_cohort"],
         "output":str(out)},indent=2))
        return
    if not metadata["high_kp_sufficient_for_descriptive_cohort"]:
        raise RuntimeError("Insufficient distinct Kp>5 days in 2005-01 after excluding previous orbits; expand scientifically justified time/data coverage rather than padding with correlated EDRs")
    for record in chosen:
        enriched=prepare_case(record,args.output_root)
        case=case_name(record)
        entry=next(x for x in metadata["cases"] if x["case_id"]==case)
        entry.update({k:v for k,v in enriched.items() if k not in entry})
        print(f"{case}: Kp={entry['kp_midpoint']:.1f}, {entry['kp_group']}, {enriched['n_supported_cells']}/7680 observed cells")
    metadata["n_prepared"]=len(chosen)
    out=args.output_root/"manifest.json"
    out.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"prepared","high_kp_dates":len(high),"quiet_dates":len(quiet),"manifest":str(out)},indent=2))
if __name__=="__main__":
    main()

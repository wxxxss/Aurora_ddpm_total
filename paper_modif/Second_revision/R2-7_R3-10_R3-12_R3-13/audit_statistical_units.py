#!/usr/bin/env python3
"""R3-13: separate spatial sampling units from independent event units.

Reads existing first-revision CSV files, without rerunning reconstruction or
inventing unavailable real-observation samples.

From the repository root:
    python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_statistical_units.py

Outputs:
    statistical_units_audit.json
    psd_event_level_descriptive.csv

The two EDR cases are the ONLY independent event units. No bootstrap CI
or p value over 79 correlated segments or 8202 correlated edges is reported.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper_modif/First_revision/R1-1/r1_1_results"
OUTPUT = Path(__file__).resolve().parent

def records(path):
    if not path.is_file():
        raise FileNotFoundError(f"First-revision result file missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))

def mean(items):
    xs = [float(v) for v in items if v is not None and math.isfinite(float(v))]
    return sum(xs)/len(xs) if xs else None

def event_psd(segments):
    by_case=defaultdict(list)
    for r in segments:
        by_case[r["case"]].append(r)
    if set(by_case) != {"Figure6","Figure7"}:
        raise ValueError(f"Expected both EDR events, found {list(by_case)}")
    output=[]
    for case in ("Figure6","Figure7"):
        rows=by_case[case]
        hf=[float(r["ssusi_high_frequency_fraction"])-float(r["ovation_high_frequency_fraction"]) for r in rows]
        slope=[float(r["ssusi_spectral_slope"])-float(r["ovation_spectral_slope"])
               for r in rows
               if math.isfinite(float(r["ssusi_spectral_slope"]))
               and math.isfinite(float(r["ovation_spectral_slope"]))]
        output.append({
            "case":case,
            "n_correlated_segments":len(rows),
            "n_distinct_mlat_rows":len(set(int(r["row_index"]) for r in rows)),
            "mean_hf_fraction_difference_ssusi_minus_ovation":mean(hf),
            "n_finite_slope_pairs":len(slope),
            "mean_slope_difference_ssusi_minus_ovation":mean(slope),
            "segment_size_min_bins":min(int(r["n_bins"]) for r in rows),
            "segment_size_max_bins":max(int(r["n_bins"]) for r in rows)
        })
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,default=SOURCE)
    p.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=p.parse_args()
    psd=records(args.source_dir/"psd_segment_metrics.csv")
    grad=records(args.source_dir/"gradient_summary.csv")
    expected={r["case"]:int(r["n"]) for r in grad
              if r["space"]=="log1p_flux" and r["source"]=="SSUSI"}
    events=event_psd(psd)
    actual={e["case"]:e["n_correlated_segments"] for e in events}
    if sum(actual.values())!=len(psd):
        raise AssertionError("PSD counts do not sum to total")
    if "combined" not in expected or expected["combined"] != expected.get("Figure6",0)+expected.get("Figure7",0):
        raise AssertionError("Gradient edge accounting disagrees across cases")
    total_grad=expected["combined"]
    if total_grad != 8202 or len(psd)!=79:
        raise ValueError(f"Audit source changed since manuscript values: {total_grad} edges / {len(psd)} PSD segments")
    mean_event_hf=mean([e["mean_hf_fraction_difference_ssusi_minus_ovation"] for e in events])
    mean_event_slope=mean([e["mean_slope_difference_ssusi_minus_ovation"] for e in events])
    output={
      "source_directory":str(args.source_dir),
      "n_independent_edr_cases":len(events),
      "n_correlated_gradient_edges":total_grad,
      "gradient_edges_by_case":{k:v for k,v in expected.items() if k!="combined"},
      "n_correlated_psd_segments":len(psd),
      "psd_segments_by_case":actual,
      "event_level_descriptive":events,
      "unweighted_mean_across_two_event_means":{
          "hf_fraction_difference":mean_event_hf,
          "spectral_slope_difference":mean_event_slope},
      "inference_warning":(
          "Edges and PSD segments share neighboring cells/MLAT rows and "
          "are nested within two EDR events. A pixel-level KS statistic is "
          "descriptive, not a p value from independent samples. "
          "Segment-resampling 95% CIs and segment-level Wilcoxon p values "
          "from first revision must not be interpreted as event-level "
          "generalization. Only two independent events are available; "
          "no robust across-event confidence interval or hypothesis test "
          "is reported in this audit."
      ),
      "outcome_warning":(
          "PSD fraction differences change sign between Figure6 and Figure7; "
          "the combined segment result alone masks this heterogeneity."
      )
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/"statistical_units_audit.json").open("w",encoding="utf-8") as h:
        json.dump(output,h,indent=2,ensure_ascii=False)
    with (args.output_dir/"psd_event_level_descriptive.csv").open("w",newline="",encoding="utf-8") as h:
        writer=csv.DictWriter(h,fieldnames=list(events[0].keys()))
        writer.writeheader();writer.writerows(events)
    print(json.dumps(output,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Fail fast if SSUSI figure products still use the old OMNI/OVATION index mismatch.

After running rerun_ssusi_reconstruction_timecorrected.py:
 python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/verify_timecorrected_reconstruction.py

Reads the time-corrected output JSON and verifies orbit-mean timing metadata,
center-aligned observation support, and exact retention of observed pixels.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPECTED={"Figure6":4345,"Figure7":4279}
MINUTE_START=datetime(2005,1,1,0,0)

def frame_indices(start,stop):
    s=datetime.fromisoformat(start)
    e=datetime.fromisoformat(stop)
    def minutes(d):return (d-MINUTE_START).total_seconds()/60
    from math import floor,ceil
    return floor(minutes(s)),ceil(minutes(e))

def verify(summaries):
    result={}
    for case,expected_n in EXPECTED.items():
        d=summaries[case]
        i0,i1=frame_indices(d["acquisition_start"],d["acquisition_end"])
        got_i0=int(d["ovation_frame_start"])
        got_i1=int(d["ovation_frame_end"])
        if (got_i0,got_i1)!=(i0,i1):
            raise AssertionError(f"{case}: OVATION interval indices {got_i0}..{got_i1} not {i0}..{i1}")
        if d["conditioning_time_basis"]!="OMNI interval mean over EDR acquisition":
            raise AssertionError(f"{case}: unexpected solar conditioning rule")
        if d["ovation_time_basis"]!="OVATION 1-min interval mean over EDR acquisition":
            raise AssertionError(f"{case}: unexpected OVATION baseline time rule")
        actual=round(float(d["observed_fraction"])*80*96)
        if actual!=expected_n:
            raise AssertionError(f"{case}: {actual} supported cells, expected {expected_n}")
        if float(d["observed_exact_max_abs_error"])>1e-6:
            raise AssertionError(f"{case}: observed values not preserved")
        result[case]={"expected_ovation_frames":[i0,i1],
                      "figure_ovation_frames":[got_i0,got_i1],
                      "observed_supported_cells":actual,
                      "observed_exact_max_abs_error":float(d["observed_exact_max_abs_error"])}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,
        default=HERE/"center_aligned_reconstruction_timecorrected/reconstruction_summary.json")
    args=p.parse_args()
    if not args.summary.exists():
        raise FileNotFoundError(f"Run time-corrected SSUSI reconstruction first: {args.summary}")
    with args.summary.open(encoding="utf-8") as f:
        summaries=json.load(f)
    output=verify(summaries)
    print(json.dumps({"status":"time alignment metadata verified","cases":output},indent=2))

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""R3-12 sensitivity of the Polar >40 flux screening rule, BEFORE censorship.

Requires a user-provided array of *calibrated but uncensored* physical flux;
do not feed a dataset that has already removed values above 40.
The script measures inclusion-rate sensitivity only, NOT flux-calibration
accuracy or an evidence-based optimal threshold.

Example:
 python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_uncensored_polar_threshold.py \
     --uncensored-flux /path/to/pre_QC_calibrated_LBHL_flux.npy \
     --field aurora_image --thresholds 20 30 40 50 60

Input can be a 2-D or 3-D numeric .npy array or structured .npy with
'field'. For structured arrays, --field selects the flux image per record.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

GROUP=Path(__file__).resolve().parent

def extract(data,field):
    if isinstance(data,np.lib.npyio.NpzFile):
        if field not in data.files:
            raise KeyError(f"NPZ key {field!r} unavailable: {data.files}")
        arr=np.asarray(data[field])
    elif data.dtype.names is not None:
        if field not in data.dtype.names:
            raise KeyError(f"Structured field {field!r} unavailable: {data.dtype.names}")
        arr=np.asarray(np.stack(data[field]),dtype=float)
    else:
        arr=np.asarray(data,dtype=float)
    if arr.ndim not in (2,3):raise ValueError(f"Expected 2-D/3-D uncensored flux array; got {arr.shape}")
    return np.asarray(arr,dtype=float)

def audit(arr, thresholds):
    finite=arr[np.isfinite(arr)]
    if finite.size==0:raise ValueError("No finite input flux")
    if np.any(finite<0):
        print("WARNING: Negative values present; verify calibrated background handling.")
    result={
       "shape":list(arr.shape),
       "n_finite":int(finite.size),
       "range":[float(finite.min()),float(finite.max())],
       "percentiles":[float(v) for v in np.percentile(finite,[50,90,95,99,99.9])],
       "threshold_results":[],
    }
    for threshold in thresholds:
        result["threshold_results"].append({
           "reject_if_flux_gt":float(threshold),
           "rejected_pixels":int(np.count_nonzero(finite>threshold)),
           "rejected_fraction":float(np.mean(finite>threshold)),
           "retained_finite_pixels":int(np.count_nonzero(finite<=threshold))
        })
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uncensored-flux",type=Path,required=True,
                   help="Uncensored calibrated PHYSICAL energy-flux array, BEFORE 40 cutoff")
    p.add_argument("--field",default="aurora_image")
    p.add_argument("--thresholds",type=float,nargs="+",default=[20,30,40,50,60])
    p.add_argument("--output",type=Path,default=GROUP/"uncensored_polar_threshold_audit.json")
    args=p.parse_args()
    if not args.uncensored_flux.is_file():
        raise FileNotFoundError(args.uncensored_flux)
    if any(t<=0 for t in args.thresholds):
        raise ValueError("All thresholds must be positive physical flux values")
    data=np.load(args.uncensored_flux,allow_pickle=True)
    try:arr=extract(data,args.field)
    finally:
        if isinstance(data,np.lib.npyio.NpzFile):data.close()
    res=audit(arr,args.thresholds)
    res["assumption"]="Input is pre-quality-cutoff calibrated physical flux, not normalized/log flux."
    res["limitation"]="This threshold-count sensitivity cannot demonstrate that the >40 readings are physically spurious or define the optimal cutoff. Comparison with instrument quality flags or concurrent independent measurements is needed."
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(res,indent=2))
    print("Saved:",args.output)

if __name__=="__main__":
    main()

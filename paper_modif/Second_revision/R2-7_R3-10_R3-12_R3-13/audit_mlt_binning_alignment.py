#!/usr/bin/env python3
"""R3-10: compare two archived SSUSI MLT binning conventions on identical native data.

'edge_zero' reproduces prepare_ssusi_swath_cases.py (MLT bin edges
0, 0.25, 0.50,...). 'nearest_center' reproduces the swath-v2 audit convention
(np.rint(MLT/0.25)), consistent with declared MLT centers
0, 0.25,...,23.75 h.

Read the original EDR filenames from the archived swath-audit CSV; nothing
is interpolated. Check whether the archived prepared NPZ support matches
the edge-zero convention.

From repository root:
 python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_mlt_binning_alignment.py

Outputs mlt_binning_alignment_audit.json.
Requires numpy and netCDF4, plus original EDR files on the server.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SWATH_CSV=ROOT/"paper_modif/First_revision/R2-1_R2-3/ssusi_swath_audit_v2/ssusi_swath_summary_v2.csv"
PREPARED=ROOT/"paper_modif/First_revision/R2-1_R2-3/swath_prepared_cases"
LEGACY_PREPARED=ROOT/"paper_modif/R2-1_R2-3/swath_prepared_cases"
N_LAT,N_MLT=80,96
LAT_CENTERS=np.linspace(50.0,90.0,N_LAT)
LAT_STEP=LAT_CENTERS[1]-LAT_CENTERS[0]
MLT_STEP=24.0/N_MLT
LAT_EDGES=np.r_[LAT_CENTERS[0]-LAT_STEP/2,
                 (LAT_CENTERS[1:]+LAT_CENTERS[:-1])/2,
                 LAT_CENTERS[-1]+LAT_STEP/2]
MLT_EDGES=np.arange(N_MLT+1,dtype=float)*MLT_STEP

def read_native(path):
    with Dataset(path,"r") as ds:
        def v(key):return np.array(ds.variables[key][:],dtype=np.float64).squeeze()
        flux=v("ENERGY_FLUX_NORTH_MAP")
        lat=v("LATITUDE_GEOMAGNETIC_GRID_MAP")
        mlt=v("MLT_GRID_MAP")
        ut=v("UT_N")
    if not (flux.shape==lat.shape==mlt.shape==ut.shape):
        raise ValueError(f"Unexpected native SSUSI variable dimensions in {path}")
    valid=np.isfinite(ut)&(ut>0)&np.isfinite(flux)&np.isfinite(lat)&np.isfinite(mlt)&(lat>=50)&(lat<=90)
    return flux[valid],lat[valid],np.mod(mlt[valid],24.0),int(valid.sum())

def grid_assignments(lat,mlt,rule):
    if rule=="edge_zero":
        ilat=np.digitize(lat,LAT_EDGES)-1
        imlt=np.digitize(mlt,MLT_EDGES)-1
        imlt[imlt==N_MLT]=0
    elif rule=="nearest_center":
        ilat=np.rint((lat-LAT_CENTERS[0])/LAT_STEP).astype(int)
        imlt=np.rint(mlt/MLT_STEP).astype(int)%N_MLT
    else: raise ValueError(rule)
    ok=(ilat>=0)&(ilat<N_LAT)&(imlt>=0)&(imlt<N_MLT)
    return ilat,imlt,ok

def map_to_grid(flux,lat,mlt,rule):
    ilat,imlt,ok=grid_assignments(lat,mlt,rule)
    counts=np.zeros((N_LAT,N_MLT),dtype=np.int32)
    sums=np.zeros((N_LAT,N_MLT),dtype=np.float64)
    np.add.at(counts,(ilat[ok],imlt[ok]),1)
    np.add.at(sums,(ilat[ok],imlt[ok]),flux[ok])
    means=np.full((N_LAT,N_MLT),np.nan)
    valid=counts>0
    means[valid]=sums[valid]/counts[valid]
    return valid,counts,means,ok,(ilat,imlt)

def compare_case(case,source,prepared):
    flux,lat,mlt,n=read_native(source)
    zero,cz,mz,okz,idxz=map_to_grid(flux,lat,mlt,"edge_zero")
    center,cc,mc,okc,idxc=map_to_grid(flux,lat,mlt,"nearest_center")
    both=zero&center
    differing_native=okz&okc&((idxz[0]!=idxc[0])|(idxz[1]!=idxc[1]))
    union=zero|center
    report={
        "source_edr":str(source),
        "n_valid_native_samples":n,
        "edge_zero_supported_cells":int(zero.sum()),
        "nearest_center_supported_cells":int(center.sum()),
        "edge_zero_fraction":float(zero.mean()),
        "nearest_center_fraction":float(center.mean()),
        "differing_native_target_cell_assignments":int(differing_native.sum()),
        "fraction_differing_native_assignments":float(differing_native.mean()) if n else None,
        "support_cell_symmetric_difference":int((zero^center).sum()),
        "support_cell_intersection":int(both.sum()),
        "support_cell_union":int(union.sum()),
        "support_jaccard":float(both.sum()/union.sum()) if union.any() else None,
        "same_support_cell_flux_mean_abs_difference":float(np.mean(np.abs(mz[both]-mc[both]))) if both.any() else None,
        "note":"The support-count difference alone may be small even when many native samples are assigned to different cells."
    }
    candidates=[prepared/case/(case+"_swath_case.npz"),
                LEGACY_PREPARED/case/(case+"_swath_case.npz")]
    for path in candidates:
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as f:
            m=np.asarray(f["obs_mask"],dtype=bool)
            if m.shape!=zero.shape:raise ValueError(f"NPZ observation mask has wrong shape: {m.shape}")
        report["prepared_input_path"]=str(path)
        report["prepared_supported_cells"]=int(m.sum())
        report["prepared_vs_edge_zero_mask_difference"]=int((m^zero).sum())
        report["prepared_vs_nearest_center_mask_difference"]=int((m^center).sum())
        break
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swath-summary",type=Path,default=SWATH_CSV)
    p.add_argument("--prepared-root",type=Path,default=PREPARED)
    p.add_argument("--output",type=Path,default=HERE/"mlt_binning_alignment_audit.json")
    args=p.parse_args()
    with args.swath_summary.open(encoding="utf-8-sig",newline="") as f:
        inputs={r["case"]:Path(r["source_file"]) for r in csv.DictReader(f)}
    cases={}
    for case in ("Figure6","Figure7"):
        fp=inputs[case]
        if not fp.is_file():
            raise FileNotFoundError(f"Original EDR not found: {fp}")
        cases[case]=compare_case(case,fp,args.prepared_root)
    report={
        "reference_mlt_centers_hours":"0, 0.25, ..., 23.75 (cyclic 24h)",
        "edge_zero_rule":"[0,0.25) -> target index 0, [0.25,0.5) -> index 1, etc.",
        "nearest_center_rule":"nearest 0.25h target center, periodic across midnight",
        "difference":"edge_zero boundaries are displaced by +0.125 h relative to the center-consistent half-step boundaries",
        "cases":cases,
        "conclusion_limit":"This comparison characterizes the discrepancy only. Before changing production preprocessing, re-run impacted SSUSI reconstructions/metrics with a single documented aligned binning rule."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()

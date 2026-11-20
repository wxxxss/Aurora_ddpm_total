#!/usr/bin/env python3
"""R3-10 / R3-12 / R2-7: grid, native-support and Polar threshold audit.

Run from repository root:
    python paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13/audit_grid_calibration.py

Uses first-revision prepared SSUSI NPZ files when present. Does not mistake
processed Polar zeros or weak flux values for confirmed missing observations.
Raw Polar/UVI photon radiances and independent calibration records are not in
this repo; this script cannot estimate instrument calibration uncertainty.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
GROUP=Path(__file__).resolve().parent
DEFAULT_PREPARED=ROOT/"paper_modif/First_revision/R2-1_R2-3/swath_prepared_cases"
DEFAULT_POLAR=Path("/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/resampled_5min_1996_0405.npy")
FIGURE_TIMES={"Figure4":"1996-04-03T02:15:00","Figure5":"1996-04-01T08:40:00"}
LAT_CENTERS=np.linspace(50.0,90.0,80)
MLT_CENTERS=np.linspace(0.0,24.0,96,endpoint=False)

def masked_average_audit(flux,mask,count):
    """Validate that no values are attributed to unsupported target cells."""
    flux=np.asarray(flux,dtype=float)
    mask=np.asarray(mask,dtype=bool)
    count=np.asarray(count,dtype=int)
    expected=(80,96)
    if flux.shape!=expected or mask.shape!=expected or count.shape!=expected:
        raise ValueError(f"Unexpected grid shape: flux={flux.shape}, mask={mask.shape}, count={count.shape}")
    if np.any(count<0):
        raise ValueError("Negative target count")
    if np.any(mask!=(count>0)):
        raise AssertionError("Observation mask differs from native sample count>0")
    if np.any(~np.isfinite(flux[mask])):
        raise AssertionError("Supported target cells contain nonfinite energy flux")
    if np.any(np.isfinite(flux[~mask])):
        raise AssertionError("Unsupported target cells contain finite energy flux")
    return {
       "total_grid_cells":int(mask.size),
       "cells_with_native_support":int(mask.sum()),
       "cells_without_native_support":int((~mask).sum()),
       "fraction_with_native_support":float(mask.mean()),
       "cells_receiving_two_or_more_native_samples":int((count>1).sum()),
       "fraction_supported_cells_with_multiple_native_samples":float(np.mean(count[mask]>1)) if mask.any() else None,
       "total_native_samples_in_target_cells":int(count.sum()),
       "max_native_samples_per_cell":int(count.max()),
       "mean_native_samples_per_supported_cell":float(count[mask].mean()) if mask.any() else None
    }

def polar_event_summary(data,event,threshold):
    times=np.asarray(data["utc"],dtype="datetime64[s]")
    target=np.datetime64(FIGURE_TIMES[event],"s")
    delta=np.abs((times-target).astype("timedelta64[s]").astype(np.int64))
    idx=int(delta.argmin())
    if int(delta[idx])>150:raise ValueError(f"{event}: nearest Polar record offset {int(delta[idx])}s exceeds 150s")
    a=np.asarray(data["aurora_image"][idx],dtype=float)
    if a.shape!=(80,96):raise ValueError(f"{event}: unexpected Polar shape: {a.shape}")
    finite=np.isfinite(a)
    return {
      "utc":str(times[idx]),"offset_seconds":int(delta[idx]),
      "finite_cells":int(finite.sum()),"nonfinite_cells":int((~finite).sum()),
      "cells_below_flux_1":int(np.count_nonzero(finite & (a<1.0))),
      "cells_at_or_above_flux_1":int(np.count_nonzero(finite & (a>=1.0))),
      "cells_above_flux_20":int(np.count_nonzero(finite & (a>20.0))),
      "cells_above_flux_40_after_gridding":int(np.count_nonzero(finite & (a>40.0))),
      "cells_above_flux_60_after_gridding":int(np.count_nonzero(finite & (a>60.0))),
      "max_finite_flux":float(np.max(a[finite])) if finite.any() else None,
      "interpretation":"Postprocessed flux-map diagnostics; not a validation of the original 40 erg cm^-2 s^-1 rejection threshold. Zero or weak flux can be valid signal or a fill value, depending on the upstream quality flags."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",type=Path,default=DEFAULT_PREPARED)
    p.add_argument("--polar-data",type=Path,default=DEFAULT_POLAR)
    p.add_argument("--output",type=Path,default=GROUP/"grid_calibration_audit.json")
    args=p.parse_args()
    assert LAT_CENTERS.size==80 and MLT_CENTERS.size==96
    assert abs(float(np.diff(LAT_CENTERS).mean())-40.0/79.0)<1e-12
    assert abs(float(np.diff(MLT_CENTERS).mean())-0.25)<1e-12
    assert MLT_CENTERS[-1]==23.75
    report={
        "grid":{
          "hemisphere":"Northern (positive magnetic latitude)",
          "shape":[80,96],
          "mlat_centers_range_deg":[50.0,90.0],
          "mlat_center_spacing_deg":40.0/79.0,
          "mlt_centers_range_hours":[0.0,23.75],
          "mlt_center_spacing_hours":0.25,
          "mlt_periodic_hours":24.0,
          "mlt_endpoint_24_is_not_a_second_center":True,
          "resampling":"SSUSI: direct binning of finite UT_N>0 native samples; arithmetic mean for duplicate target-cell hits; do not interpolate unsupported cells"
        },
        "lbhl_conversion":{
          "adopted_R_per_erg_cm_minus2_s_minus1":110.0,
          "conversion_formula":"estimated electron energy flux [erg cm^-2 s^-1] = corrected LBHL brightness [R] / 110",
          "flux_40_corresponding_brightness_R":4400.0,
          "threshold_40_interpretation":"user-defined artifact-screening cutoff, not a documented physical upper limit or independently validated quality-control threshold",
          "calibration_uncertainty_not_measured":True,
          "limitations":"The empirical factor depends on assumed nitrogen emission yield, electron precipitation spectrum, atmospheric composition, viewing geometry, background/dayglow subtraction and radiometric calibration. Without calibrated raw imagery and independent truth, no quantitative uncertainty interval or cutoff sensitivity can be claimed."
        },
        "ssusi_prepared_cases":{},
        "polar_postprocessed_cases":{},
        "missing_inputs":[],
    }
    for case in ("Figure6","Figure7"):
        path=args.prepared_root/case/(case+"_swath_case.npz")
        if not path.is_file():
            report["missing_inputs"].append(str(path))
            report["ssusi_prepared_cases"][case]={"available":False}
            continue
        with np.load(path,allow_pickle=False) as npz:
            for key in ("flux_grid","obs_mask","count_grid"):
                if key not in npz:
                    raise KeyError(f"{path}: missing {key}")
            report["ssusi_prepared_cases"][case]={"available":True,
                **masked_average_audit(npz["flux_grid"],npz["obs_mask"],npz["count_grid"])}
    if args.polar_data.is_file():
        data=np.load(args.polar_data,allow_pickle=True)
        if data.dtype.names is None or not {"utc","aurora_image"}.issubset(data.dtype.names):
            raise ValueError("Polar input needs structured 'utc' and 'aurora_image' fields")
        for case in FIGURE_TIMES:
            report["polar_postprocessed_cases"][case]=polar_event_summary(data,case,40.0)
    else:
        report["missing_inputs"].append(str(args.polar_data))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report["missing_inputs"]:
        print("WARNING: Some local private inputs are unavailable; corresponding results are INCOMPLETE.")

if __name__=="__main__":
    main()

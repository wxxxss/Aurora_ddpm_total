#!/usr/bin/env python3
"""Pre-registered holdout selection on REAL observed SSUSI pixels.

Random protocol ranks candidate MLT windows using a deterministic RNG, after
filtering on UT_N-derived spatial coverage and latitude only. Crucially, the
satellite energy-flux values are NOT used to choose windows. Ground truth is
read only to characterize the selected windows and score predictions.
"""
from __future__ import annotations

import numpy as np

def periodic_mlt_mask(centers, center_h, width_h):
    a=np.asarray(centers,dtype=float)
    d=((a-float(center_h)+12.0)%24.0)-12.0
    return (d >= -float(width_h)/2)&(d < float(width_h)/2)

def circular_separation_hours(a,b):
    return abs(((float(a)-float(b)+12)%24)-12)

def random_observed_holdouts(flux, support_mask, mlat_centers, mlt_centers,
                             *, width_h, n_windows=1, band=(60.,80.),
                             min_coverage=0.50, min_pixels=20,
                             candidate_step_h=0.50, seed=2026):
    truth=np.asarray(flux,dtype=float)
    support=np.asarray(support_mask,dtype=bool)
    lat=np.asarray(mlat_centers,dtype=float)
    mlt=np.asarray(mlt_centers,dtype=float)
    if truth.shape!=(len(lat),len(mlt)) or support.shape!=truth.shape:
        raise ValueError("Grid dimensions do not match")
    if not (0<float(width_h)<24):
        raise ValueError("width_h must be in (0,24)")
    if candidate_step_h<=0 or n_windows<1:raise ValueError("Invalid selection settings")
    if np.any(support & ~np.isfinite(truth)):
        raise ValueError("Supported cells contain nonfinite flux")
    band_rows=(lat>=band[0])&(lat<=band[1])
    candidates=[]
    for center in np.arange(0,24,candidate_step_h):
        cols=periodic_mlt_mask(mlt,center,width_h)
        region=np.outer(band_rows,cols)
        denominator=int(region.sum())
        if denominator==0:continue
        observed=region&support
        n=int(observed.sum())
        coverage=n/denominator
        if n<min_pixels or coverage<min_coverage:continue
        candidates.append(dict(center_h=float(center),width_h=float(width_h),
                               n_pixels=n,coverage=float(coverage),
                               mask=observed))
    # Rank using the seed and the OBSERVATION-COVERAGE mask only, never flux.
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(candidates))
    chosen=[]
    for i in order:
        cand=candidates[int(i)]
        if all(circular_separation_hours(cand["center_h"],s["center_h"])>=width_h for s in chosen):
            vals=truth[cand["mask"]]
            # Observed brightness is descriptive metadata, not selection input.
            cand["signal_fraction"]=float(np.mean(vals>=0.10))
            cand["mean_heldout_true_flux_diagnostic_only"]=float(np.mean(vals))
            cand["score"]=0.0
            chosen.append(cand)
            if len(chosen)==n_windows:break
    return chosen

def assert_no_target_leakage(truth, input_mask, target_mask, prefilled_input):
    t=np.asarray(truth)
    known=np.asarray(input_mask,dtype=bool)
    target=np.asarray(target_mask,dtype=bool)
    supplied=np.asarray(prefilled_input)
    if np.any(known&target):raise AssertionError("A target pixel remained in the observation mask")
    if np.any(np.asarray(supplied)[target]!=0):raise AssertionError("Held-out truth leaked through the initial input")
    if np.any(~np.isfinite(t[known])):raise AssertionError("Known observation has nonfinite values")
    return True

"""Strict Kp activity matching for the SSUSI multi-orbit validation.

Kp is a THREE-HOUR index, not a native 5-minute solar wind condition.
First revision has hourly OMNI arrays containing the Kp field (sometimes
repeated for each hour in a three-hour Kp interval). Match by the nearest
hourly UTC sample to acquisition midpoint with <=31-minute tolerance,
without using auroral truth or model output.

For each ~102-minute EDR, also report all Kp values available within the
EDR. An acquisition straddling a Kp transition is flagged explicitly.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np

DEFAULT_KP=Path("/home/docker/data/private/AuroraData/omni_real_data/hourly_r1_2/2005/omni_2005_hourly_r1_2.npy")

def parse_utc(value):
    return np.datetime64(value,"s")

def load_hourly_kp(path=DEFAULT_KP):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hourly OMNI Kp file missing: {path}. If located elsewhere, "
            "supply --kp-hourly /path/to/omni_2005_hourly_r1_2.npy"
        )
    data=np.load(path,allow_pickle=True)
    if data.dtype.names is None or not {"utc","Kp"}.issubset(data.dtype.names):
        raise ValueError("Hourly Kp data must be structured NPY with utc and Kp fields")
    times=np.asarray(data["utc"],dtype="datetime64[s]")
    values=np.asarray(data["Kp"],dtype=float)
    if len(times)!=len(values) or len(times)<3:raise ValueError("Invalid hourly time series")
    order=np.argsort(times)
    times=times[order]
    values=values[order]
    if len(np.unique(times))!=len(times):raise ValueError("Hourly Kp UTC stamps repeated")
    if np.any(np.isfinite(values)&((values<0)|(values>9))):
        raise ValueError("Kp outside physical 0..9 range; check scaling/fill flags")
    return times,values

def activity_group(kp):
    if not np.isfinite(kp):return "unavailable"
    if kp>5:return "Kp>5"
    if kp<=3:return "Kp<=3"
    return "3<Kp<=5"

def match_kp(times,values,start,end,max_offset_minutes=31):
    a,b=parse_utc(start),parse_utc(end)
    if not b>a:raise ValueError("Nonpositive EDR interval")
    mid=a+(b-a)//2
    offsets=np.abs((times-mid).astype("timedelta64[s]").astype("int64"))
    index=int(np.argmin(offsets))
    offset_min=float(offsets[index]/60.)
    kp=float(values[index])
    # A nonfinite value cannot be silently converted to quiet conditions.
    if offset_min>max_offset_minutes or not np.isfinite(kp):
        return {"group":"unavailable","kp_midpoint":None,"offset_min":offset_min,
                "reason":"No finite Kp hour within permitted midpoint tolerance"}
    within=(times>=a)&(times<=b)&np.isfinite(values)
    covered=values[within]
    # Check the Kp values observed throughout the orbit, if they exist.
    min_inside=float(np.min(covered)) if covered.size else None
    max_inside=float(np.max(covered)) if covered.size else None
    changing=bool(covered.size>1 and np.max(covered)-np.min(covered)>0.05)
    return {
       "group":activity_group(kp),"kp_midpoint":kp,
       "kp_midpoint_utc":str(times[index].astype("datetime64[s]")),
       "kp_midpoint_offset_minutes":offset_min,
       "kp_inside_acquisition_min":min_inside,
       "kp_inside_acquisition_max":max_inside,
       "kp_values_change_within_acquisition":changing,
       "strictly_high_throughout_available_hours":
           bool(covered.size>0 and np.all(covered>5)),
       "n_finite_hourly_kp_inside_acquisition":int(covered.size),
       "time_alignment_warning":"Hourly Kp values repeat a three-hour geomagnetic index. A selected acquisition is a temporal cluster, not a new independent Kp measurement or storm."
    }

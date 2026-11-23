#!/usr/bin/env python3
"""No checkpoint/EDR needed; synthetic full-month conditioning range tests."""
from __future__ import annotations
import unittest
from datetime import datetime,timedelta
import numpy as np
from audit_2006_solar_normalization import analyze
from audit_and_prepare_2006_omni import month_grid

FIELDS=("Bx","By","Bz","V","P")

def fixture():
    time=month_grid(2006,1)
    arr=np.zeros(len(time),dtype=[("utc","datetime64[ns]")]+[
        (name,"f4") for name in FIELDS])
    arr["utc"]=time
    for name,value in zip(FIELDS,[1,1,1,400,1]):
        arr[name]=value
    dates=(2,4,7,10,12,14,17,20,22,25,28,30)
    cases=[]
    for day in dates:
        a=datetime(2006,1,day,3)
        b=a+timedelta(minutes=100)
        cases.append({"case_id":f"test_day_{day}","orbit":str(day),
                      "acquisition_start":a.isoformat(),
                      "acquisition_end":b.isoformat()})
    norm={"omni":{"min":np.array([-10,-10,-10,100,.1]),
                  "range":np.array([20,20,20,600,10])}}
    return {"month":"2006-01","cases":cases},arr,norm

class TestRangeClipping(unittest.TestCase):
    def test_detects_outside_1996_range_without_modifying_cohort(self):
        catalog,arr,norm=fixture()
        selected=(arr["utc"]>=np.datetime64("2006-01-02T03:00")) & (
                  arr["utc"]<=np.datetime64("2006-01-02T04:40"))
        arr["V"][selected]=1000
        summary=analyze(catalog,arr,norm)
        self.assertEqual(summary["n_acquisitions"],12)
        self.assertEqual(summary["n_acquisitions_with_any_interval_mean_clipping"],1)
        self.assertEqual(summary["n_acquisitions_with_interval_mean_clipped_by_field"]["V"],1)
        self.assertEqual(summary["acquisitions"][0]["interval_mean_would_clip_high"],["V"])
        self.assertTrue(all(r["case_id"] in [c["case_id"] for c in catalog["cases"]]
                            for r in summary["acquisitions"]))

    def test_rejects_even_one_missing_sample_in_edr(self):
        catalog,arr,norm=fixture()
        at=np.flatnonzero(arr["utc"]==np.datetime64("2006-01-02T03:00"))[0]
        arr["Bx"][at]=np.nan
        with self.assertRaises(ValueError):
            analyze(catalog,arr,norm)

    def test_refuses_incomplete_month_time_grid(self):
        catalog,arr,norm=fixture()
        with self.assertRaises(ValueError):
            analyze(catalog,arr[:-1],norm)

    def test_rejects_wrong_solar_normalization_vector(self):
        catalog,arr,norm=fixture()
        norm["omni"]["range"]=np.array([20,20,20,600])
        with self.assertRaises(ValueError):
            analyze(catalog,arr,norm)

if __name__=="__main__":
    unittest.main()

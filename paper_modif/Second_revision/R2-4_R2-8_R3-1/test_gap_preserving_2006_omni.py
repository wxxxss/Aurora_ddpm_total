#!/usr/bin/env python3
"""Guarantee long raw CDF outages are NOT represented as imputed solar wind."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from audit_and_prepare_2006_omni import month_grid
from prepare_2006_gap_preserving_omni import clean_short_bracketed,process

FIELDS=("Bx","By","Bz","V","P")
RANGES={"Bx":(-50.,50.),"By":(-50.,50.),"Bz":(-50.,50.),"V":(200.,2000.),"P":(.05,80.)}

def fixture():
    times=month_grid(2006,1)
    d={"utc":pd.to_datetime(times)}
    for field in FIELDS:
        d[field]=np.full(len(times),400. if field=="V" else
                                    2. if field=="P" else 1.)
    catalogue={"month":"2006-01","cases":[{
        "case_id":"early_Jan_EDR",
        "acquisition_start":"2006-01-02T05:54:08",
        "acquisition_end":"2006-01-02T07:35:59"}]}
    # local stub is provided intentionally to exercise output dtype contract
    def to_structured_array(clean):
        data=np.zeros(len(clean),dtype=[("utc","datetime64[ns]")]+[
            (field,"f4") for field in FIELDS])
        data["utc"]=clean["utc"].to_numpy(dtype="datetime64[ns]")
        for field in FIELDS:data[field]=clean[field].to_numpy(dtype=np.float32)
        return data
    legacy=SimpleNamespace(VALID_RANGES=RANGES,to_structured_array=to_structured_array)
    return pd.DataFrame(d),catalogue,legacy

class TestGapPreservation(unittest.TestCase):
    def test_short_interior_gap_interpolated(self):
        v=np.array([1.,np.nan,np.nan,4.])
        a,qa=clean_short_bracketed(v,0.,10.,max_run_samples=2)
        np.testing.assert_allclose(a,[1.,2.,3.,4.])
        self.assertEqual(qa["samples_remain_nan"],0)

    def test_long_gap_and_edges_are_never_filled(self):
        v=np.array([np.nan,1.,np.nan,np.nan,np.nan,5.,np.nan])
        a,qa=clean_short_bracketed(v,0.,10.,max_run_samples=2)
        self.assertTrue(np.isnan(a[0]))
        self.assertTrue(np.isnan(a[2:5]).all())
        self.assertTrue(np.isnan(a[-1]))
        self.assertEqual(qa["short_gap_interpolated_samples"],0)

    def test_13_hour_bad_source_outside_event_kept_nan(self):
        df,catalog,legacy=fixture()
        wrong=(df.utc>=pd.Timestamp("2006-01-20T00:00")) & (
               df.utc<=pd.Timestamp("2006-01-20T13:40"))
        df.loc[wrong,FIELDS]=np.nan
        arr,qa,events=process(df,catalog,legacy)
        self.assertTrue(events["all_preselected_acquisitions_raw_valid"])
        self.assertTrue(np.isnan(arr["Bx"][wrong.to_numpy()]).all())
        self.assertEqual(qa["Bx"]["samples_remain_nan"],165)

    def test_bad_source_inside_edr_must_abort(self):
        df,catalog,legacy=fixture()
        df.loc[df.utc==pd.Timestamp("2006-01-02T06:00"),"P"]=np.nan
        with self.assertRaises(ValueError):
            process(df,catalog,legacy)

    def test_refuse_interpolation_longer_than_60min(self):
        df,catalog,legacy=fixture()
        with self.assertRaises(ValueError):
            process(df,catalog,legacy,max_short_gap_minutes=65)

if __name__=="__main__":unittest.main()

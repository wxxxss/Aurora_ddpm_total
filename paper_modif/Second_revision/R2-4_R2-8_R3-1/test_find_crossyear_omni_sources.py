#!/usr/bin/env python3
"""No-private-data test of 2006 OMNI alternative-source discovery."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import numpy as np
from find_crossyear_omni_sources import find,inspect_npy

class TestFindSources(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_valid_structured_2006_5min_npy(self):
        path=self.root/"omni_5min"/"2006"/"omni_20060101_5min.npy"
        path.parent.mkdir(parents=True)
        x=np.zeros(12,dtype=[("utc","datetime64[s]")]+[
            (k,"f4") for k in ("Bx","By","Bz","V","P")])
        x["utc"]=np.datetime64("2006-01-01T00:00:00") + (
            np.arange(12)*np.timedelta64(5,"m"))
        np.save(path,x)
        r=find(2006,[self.root/"omni_5min"])
        self.assertEqual(len(r["files"]),1)
        self.assertEqual(r["files"][0]["source_type"],"NPY")
        self.assertTrue(r["files"][0]["contains_requested_year"])
        self.assertTrue(r["files"][0]["has_5_conditioning_fields"])
        self.assertAlmostEqual(r["files"][0]["utc_median_step_minutes"],5.0)

    def test_hourly_source_not_labeled_five_minute(self):
        path=self.root/"hourly"/"2006"/"omni_2006_hourly_r1_2.npy"
        path.parent.mkdir(parents=True)
        x=np.zeros(4,dtype=[("utc","datetime64[s]")]+[
            (k,"f4") for k in ("Bx","By","Bz","V","P")])
        x["utc"]=np.datetime64("2006-01-01T00:00:00")+np.arange(4)*np.timedelta64(1,"h")
        np.save(path,x)
        result=inspect_npy(path,2006)
        self.assertEqual(result["utc_median_step_minutes"],60.0)

    def test_reports_cdf_candidate_without_reading_it(self):
        p=self.root/"hro2_5min"/"2006"/"omni_hro2_5min_20060101_v01.cdf"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"candidate only")
        a=find(2006,[self.root/"hro2_5min"])
        self.assertEqual(a["files"][0]["source_type"],"CDF")
        self.assertFalse(a["files"][0]["utc_range_verified"])

    def test_2020_june_not_confused_with_2006(self):
        wrong=self.root/"hro2_5min"/"2020"/"omni_hro2_5min_20200601_v01.cdf"
        right=self.root/"hro2_5min"/"2006"/"omni_hro2_5min_20060101_v01.cdf"
        wrong.parent.mkdir(parents=True)
        right.parent.mkdir(parents=True)
        wrong.write_bytes(b"2020 only")
        right.write_bytes(b"2006 only")
        survey=find(2006,[self.root/"hro2_5min"])
        self.assertEqual(len(survey["files"]),1)
        self.assertIn("/2006/",survey["files"][0]["path"])

    def test_empty_npy_does_not_abort_search(self):
        p=self.root/"omni_5min"/"2006"/"omni_20060101_5min.npy"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        x=find(2006,[self.root/"omni_5min"])
        self.assertEqual(x["files"][0]["read_status"],"empty")

if __name__=="__main__":
    unittest.main()

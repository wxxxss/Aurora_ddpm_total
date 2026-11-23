#!/usr/bin/env python3
"""Lightweight regression tests for Kp>5, UTC matching and 1996 cadence audit.

Run:
 python -m unittest discover -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'
"""
from __future__ import annotations
import unittest
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from kp_time_matching import activity_group,match_kp
from audit_training_alignment import infer_image_cadence
from audit_prior_multiyear import audit as audit_synthetic
from prepare_high_kp_real_ssusi import evenly_pick_days,select_kp_cohorts

class TestKp(unittest.TestCase):
    def setUp(self):
        # Simulate a genuine 3-h Kp index repeated at 1-h intervals.
        self.time=np.datetime64("2005-01-01T00:00:00")+np.arange(10)*np.timedelta64(1,"h")
        self.kp=np.array([5.0,5.0,5.0,5.3,5.3,5.3,2.0,2.0,2.0,2.0])

    def test_strict_storm_threshold(self):
        self.assertEqual(activity_group(5.0),"3<Kp<=5")
        self.assertEqual(activity_group(5.00001),"Kp>5")
        self.assertEqual(activity_group(3.0),"Kp<=3")
        self.assertEqual(activity_group(float("nan")),"unavailable")

    def test_midpoint_utc_match_and_transition(self):
        v=match_kp(self.time,self.kp,"2005-01-01T02:40:00","2005-01-01T04:10:00")
        self.assertEqual(v["group"],"Kp>5")
        self.assertAlmostEqual(v["kp_midpoint"],5.3)
        self.assertFalse(v["kp_values_change_within_acquisition"])
        u=match_kp(self.time,self.kp,"2005-01-01T04:30:00","2005-01-01T06:40:00")
        self.assertTrue(u["kp_values_change_within_acquisition"])

    def test_invalid_midpoint_or_missing_kp(self):
        u=match_kp(self.time,self.kp,"2005-01-03T12:00:00","2005-01-03T14:00:00")
        self.assertEqual(u["group"],"unavailable")
        with self.assertRaises(ValueError):
            match_kp(self.time,self.kp,"2005-01-03T13:00:00","2005-01-03T12:00:00")

class TestCatalog(unittest.TestCase):
    def rec(self,day,hour,kp,orbit):
        a=datetime(2005,1,day,hour,0)
        return dict(path=Path(f"dummy{orbit}.nc"),start=a,
                    end=a+timedelta(hours=1),
                    midpoint=a+timedelta(minutes=30),orbit=orbit,
                    kp={"group":activity_group(kp),"kp_midpoint":kp})

    def test_at_most_one_acquisition_per_day(self):
        a=[self.rec(3,3,6.0,"001"),self.rec(3,10,6.5,"002"),
           self.rec(5,14,7.0,"003"),self.rec(9,9,6.0,"004"),
           self.rec(11,7,2.0,"005"),self.rec(13,8,2.0,"006"),
           self.rec(15,9,2.0,"007")]
        strong,quiet,meta=select_kp_cohorts(a,max_each=3)
        self.assertEqual(len(strong),3)
        self.assertEqual(len(quiet),3)
        self.assertEqual(len({r["midpoint"].date() for r in strong}),3)
        self.assertTrue(all(r["kp"]["group"]=="Kp>5" for r in strong))
        self.assertTrue(all(r["kp"]["group"]=="Kp<=3" for r in quiet))

class TestTrainingCadence(unittest.TestCase):
    def test_monthly_frame_count_proves_possible_cadence_only(self):
        self.assertEqual(infer_image_cadence(30*24*60,1996,4)[0],1)
        self.assertEqual(infer_image_cadence(30*24*12,1996,4)[0],5)
        self.assertIsNone(infer_image_cadence(123,1996,4)[0])

class TestArchivedSynthetic(unittest.TestCase):
    def test_labels_strict_gt5_not_ge4(self):
        x=[]
        for sid,kp in [(1,5.0),(2,5.3),(3,2.0)]:
            for meth in ("conditional","unconditional","interpolation"):
                x.append(dict(sample_id=sid,utc=f"2001-01-{sid+1:02d}T12:00:00",
                              year=2001,kp=kp,method=meth,rmse=.1,mae=.05))
        result=audit_synthetic(pd.DataFrame(x))
        self.assertEqual(result["activity_groups"]["Kp>5"]["n_distinct_sample_times"],1)
        self.assertEqual(result["activity_groups"]["3<Kp<=5"]["n_distinct_sample_times"],1)
        self.assertEqual(result["activity_groups"]["Kp<=3"]["n_distinct_sample_times"],1)

if __name__=="__main__":
    unittest.main()

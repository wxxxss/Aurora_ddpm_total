#!/usr/bin/env python3
"""No-private-data regression tests for third-batch statistical validity.

python -m unittest discover \
  -s paper_modif/Second_revision/R2-1_R2-2_R2-3_R3-3_R3-4_R3-5 -p 'test_*.py'
"""
from __future__ import annotations
import unittest
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

from selection_protocol import random_observed_holdouts,assert_no_target_leakage
from event_level_statistics import validate_and_aggregate, summarize
from prepare_multievent_ssusi import parse_edr_filename,select_acquisitions

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.lat=np.linspace(50,90,80)
        self.mlt=np.arange(96)*0.25
        self.mask=np.zeros((80,96),dtype=bool)
        # Valid observed cells over many MLT sectors and 60-80 MLAT.
        rows=(self.lat>=60)&(self.lat<=80)
        self.mask[np.ix_(rows,np.arange(96))]=True
        self.a=np.full((80,96),np.nan)
        self.a[self.mask]=0.0
        self.b=np.full((80,96),np.nan)
        self.b[self.mask]=20.0

    def test_random_sector_selection_is_flux_blind(self):
        opts=dict(width_h=2,n_windows=3,min_coverage=.5,seed=123)
        x=random_observed_holdouts(self.a,self.mask,self.lat,self.mlt,**opts)
        y=random_observed_holdouts(self.b,self.mask,self.lat,self.mlt,**opts)
        self.assertEqual(len(x),3)
        self.assertEqual([q["center_h"] for q in x],[q["center_h"] for q in y])
        for left,right in zip(x,y):
            np.testing.assert_array_equal(left["mask"],right["mask"])

    def test_zeroed_holdout_and_no_truth_leak(self):
        w=random_observed_holdouts(self.b,self.mask,self.lat,self.mlt,width_h=2,seed=7)[0]
        known=self.mask&~w["mask"]
        clean=np.where(known,self.b,0.0)
        self.assertTrue(assert_no_target_leakage(self.b,known,w["mask"],clean))
        bad=clean.copy()
        bad[w["mask"]]=20.
        with self.assertRaises(AssertionError):
            assert_no_target_leakage(self.b,known,w["mask"],bad)

    def test_weak_but_valid_zeros_remain_observations(self):
        out=random_observed_holdouts(self.a,self.mask,self.lat,self.mlt,width_h=1,seed=4)
        self.assertGreater(len(out),0)
        self.assertTrue(any(w["n_pixels"]>=20 for w in out))

class AcquisitionCatalogTests(unittest.TestCase):
    def test_filename_date_parser(self):
        p=Path("dmspf16_ssusi_edr-aurora_2005001T144434-2005001T162625-REV06228_vA8.2.0r000.nc")
        r=parse_edr_filename(p)
        self.assertEqual(r["orbit"],"06228")
        self.assertEqual(r["start"],datetime(2005,1,1,14,44,34))
        self.assertEqual(r["end"],datetime(2005,1,1,16,26,25))

    def test_midnight_rollover_same_day_tokens(self):
        # The end YYYYDDD can repeat the start date even when the EDR
        # crosses midnight. This filename was reported by the author.
        p=Path("dmspf16_ssusi_edr-aurora_2005244T235954-2005244T014123-REV09652_vA8.2.0r000.nc")
        r=parse_edr_filename(p)
        self.assertEqual(r["start"],datetime(2005,9,1,23,59,54))
        self.assertEqual(r["end"],datetime(2005,9,2,1,41,23))
        self.assertEqual(r["end"]-r["start"],timedelta(hours=1,minutes=41,seconds=29))
        self.assertTrue(r["filename_midnight_rollover"])

    def test_cross_year_with_correct_end_day(self):
        p=Path("dmspf16_ssusi_edr-aurora_2005365T230000-2006001T010000-REV09999_vA8.2.0r000.nc")
        r=parse_edr_filename(p)
        self.assertEqual(r["end"],datetime(2006,1,1,1))
        self.assertFalse(r["filename_midnight_rollover"])

    def test_nonmidnight_backward_clock_rejected(self):
        p=Path("dmspf16_ssusi_edr-aurora_2005244T120000-2005244T100000-REV09652_vA8.2.0r000.nc")
        with self.assertRaises(ValueError):
            parse_edr_filename(p)

    def test_unreasonably_long_acquisition_rejected(self):
        p=Path("dmspf16_ssusi_edr-aurora_2005244T235954-2005245T120000-REV09652_vA8.2.0r000.nc")
        with self.assertRaises(ValueError):
            parse_edr_filename(p)

    def test_time_selection_excludes_development_orbits(self):
        records=[]
        for day,orbit in [(1,"06228"),(3,"06263"),(7,"06500"),(12,"06600"),(21,"06700")]:
            a=datetime(2005,1,day,14)
            records.append({"path":Path("dummy"+orbit),"start":a,
                            "end":a+timedelta(hours=2),
                            "midpoint":a+timedelta(hours=1),"orbit":orbit})
        chosen,n=select_acquisitions(records,"2005-01",3,24)
        self.assertEqual(n,3)
        self.assertEqual(len(chosen),3)
        self.assertEqual({r["orbit"] for r in chosen},{"06500","06600","06700"})

class EventStatisticsTests(unittest.TestCase):
    def make_rows(self,n=8):
        rows=[]
        vals={"conditional_ddpm":3.1,"unconditional_ddpm":3.3,"ovation_prime":3.2,
              "ovation_with_observations":3.2,"interpolation":3.0}
        for k in range(n):
            for width in (1.,2.):
                for method,rmse in vals.items():
                    # All methods evaluated against the SAME case/width/rank.
                    rows.append({
                       "case":f"F16_2005_{k:02d}",
                       "width_h":str(width),"rank":"1","method":method,
                       "rmse":rmse+.02*k,"mae":1.,"pearson_r":.25,"r2":.1,
                       "truth_peak":4.,"pred_peak":1.,
                       "peak_ratio":.25,"gradient_ratio":.2
                    })
        return rows

    def test_event_not_sector_is_statistical_unit(self):
        case,windows,events,pairs=validate_and_aggregate(self.make_rows())
        self.assertEqual(len(case),8)
        self.assertEqual(len(windows),16)
        self.assertEqual(len(events),8*5)
        self.assertEqual(len(pairs),8*3)
        e,p,s=summarize(self.make_rows(),seed=11,n_boot=100)
        self.assertEqual(s["n_distinct_EDR_acquisitions"],8)
        self.assertAlmostEqual(s["paired_event_conditional_minus_baseline"]
                               ["ovation_prime"]["rmse"]["mean"],-.1)
        self.assertIsNotNone(s["paired_event_conditional_minus_baseline"]
                             ["ovation_prime"]["rmse"]["ci95"])

    def test_two_events_have_no_inferential_interval(self):
        _,_,s=summarize(self.make_rows(n=2),n_boot=100)
        x=s["paired_event_conditional_minus_baseline"]["ovation_prime"]["rmse"]
        self.assertIsNone(x["ci95"])
        self.assertEqual(x["n_finite_events"],2)

    def test_missing_matched_method_rejected(self):
        rows=self.make_rows()
        rows.pop()
        with self.assertRaises(ValueError):
            validate_and_aggregate(rows)

    def test_fused_ovation_mismatch_rejected(self):
        rows=self.make_rows()
        for r in rows:
            if r["method"]=="ovation_with_observations":
                r["rmse"]=9.
                break
        with self.assertRaises(AssertionError):
            validate_and_aggregate(rows)

if __name__=="__main__":
    unittest.main()

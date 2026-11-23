#!/usr/bin/env python3
"""Tests for the pre-inference 2006 quality-based temporal replacement rule.

Neither satellite flux, model predictions, CDF readers, nor torch is loaded.
"""
from __future__ import annotations
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from reselect_2006_omni_qualified_ssusi import (
    choose_quality_cohort, temporal_separation
)
from prepare_multievent_ssusi import case_name

def edr(day,hour,orbit):
    a=datetime(2006,1,day,hour,0)
    b=a+timedelta(minutes=100)
    return {"path":Path(f"/unread_satellite/2006/01/EDR_orbit_{orbit}.nc"),
            "start":a,"end":b,"midpoint":a+(b-a)/2,
            "orbit":f"{orbit:05d}","filename_midnight_rollover":False}

def record(row):
    return {"case_id":case_name(row),
            "acquisition_start":row["start"].isoformat(),
            "acquisition_end":row["end"].isoformat()}

def good(is_good,invalid=0):
    return {"raw_fully_valid_acquisition":is_good,
            "n_any_field_invalid_inside":invalid,
            "n_invalid_by_field":{k:invalid for k in ("Bx","By","Bz","V","P")}}

class TestReselect(unittest.TestCase):
    def fixture(self):
        original=[edr(2,6,1),edr(5,6,2),edr(8,6,3),edr(11,6,4)]
        replacements=[edr(5,8,5),edr(11,8,6)]
        records=original+replacements
        quality={case_name(r):good(r not in (original[1],original[3]),
                                  int(r in (original[1],original[3])))
                 for r in records}
        return [record(r) for r in original],records,quality,original,replacements

    def test_retains_two_and_replaces_two_nearest_timestamps(self):
        old,records,quality,original,replacements=self.fixture()
        selected,meta=choose_quality_cohort(old,records,quality,
                                            min_gap_h=24.,expected=4)
        self.assertTrue(meta["ready_for_preparation"])
        self.assertEqual(meta["n_original_valid_preserved"],2)
        self.assertEqual(meta["n_original_invalid"],2)
        self.assertEqual(len(meta["replacements"]),2)
        self.assertEqual({case_name(v) for v in selected},
                         {case_name(x) for x in (
                             original[0],original[2],*replacements)})
        self.assertTrue(all(temporal_separation(a,b,24)
                    for i,a in enumerate(selected) for b in selected[i+1:]))

    def test_never_uses_bad_omni_even_if_nearest(self):
        old,records,quality,original,replacements=self.fixture()
        quality[case_name(replacements[0])]=good(False,2)
        chosen,meta=choose_quality_cohort(old,records,quality,expected=4)
        self.assertFalse(meta["ready_for_preparation"])
        self.assertEqual(len(meta["original_cases_with_no_eligible_replacement"]),1)
        self.assertEqual(len(chosen),3)

    def test_refuses_shortening_24_hour_spacing(self):
        a=edr(2,0,1);b=edr(3,0,2)
        self.assertFalse(temporal_separation(a,b,24))
        c=edr(3,2,3)
        self.assertTrue(temporal_separation(a,c,24))

    def test_requires_original_catalog_unchanged(self):
        old,records,quality,orig,_=self.fixture()
        old[0]["acquisition_start"]="2006-01-03T06:00:00"
        with self.assertRaises(ValueError):
            choose_quality_cohort(old,records,quality,expected=4)

if __name__=="__main__":
    unittest.main()

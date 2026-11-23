#!/usr/bin/env python3
"""Tests for the pre-inference 2006 quality-based temporal replacement rule.

Neither satellite flux, model predictions, CDF readers, nor torch is loaded.
"""
from __future__ import annotations
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from reselect_2006_omni_qualified_ssusi import (
    choose_quality_cohort, temporal_separation, scan_month, source_version_key
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

    def test_identical_orbit_and_interval_uses_latest_processing_revision(self):
        # Same physical EDR acquisition, two reprocessed files.
        # Must count as ONE orbit, not an independent second event.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stem=("dmspf16_ssusi_edr-aurora_2006012T222703-"
                  "2006013T000454-REV11544_")
            earlier=root/(stem+"vA8.2.0r000.nc")
            later=root/(stem+"vA8.2.0r001.nc")
            earlier.write_bytes(b"old placeholder; no SSUSI flux parsed")
            later.write_bytes(b"new placeholder; no SSUSI flux parsed")
            out,scan=scan_month(root,"2006-01")
            self.assertEqual(len(out),1)
            self.assertEqual(str(out[0]["path"]),str(later))
            self.assertEqual(scan["n_duplicate_identity_groups"],1)
            self.assertEqual(scan["n_extra_duplicate_files"],1)
            self.assertEqual(
                scan["duplicate_identity_resolution"][0]["resolution"],
                "highest_processing_version_then_lexical_path")

    def test_original_catalog_version_stays_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stem=("dmspf16_ssusi_edr-aurora_2006012T222703-"
                  "2006013T000454-REV11544_")
            earlier=root/(stem+"vA8.2.0r000.nc")
            later=root/(stem+"vA8.2.0r001.nc")
            earlier.write_bytes(b"original")
            later.write_bytes(b"later")
            without_pin,_=scan_month(root,"2006-01")
            prior=[{**record(without_pin[0]),"source_file":str(earlier)}]
            with_pin,scan=scan_month(root,"2006-01",old_cases=prior)
            self.assertEqual(str(with_pin[0]["path"]),str(earlier))
            self.assertEqual(
                scan["duplicate_identity_resolution"][0]["resolution"],
                "preserve_exact_original_catalog_file")

    def test_new_conflicting_end_times_excluded_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=root/("dmspf16_ssusi_edr-aurora_2006012T222703-"
                        "2006013T000454-REV11544_vA8.2.0r000.nc")
            other=root/("dmspf16_ssusi_edr-aurora_2006012T222703-"
                        "2006013T001454-REV11544_vA8.2.0r001.nc")
            first.write_bytes(b"a")
            other.write_bytes(b"b")
            selected,scan=scan_month(root,"2006-01")
            self.assertEqual(selected,[])
            self.assertEqual(scan["n_conflicting_end_time_groups_excluded"],1)
            self.assertIsNone(
                scan["duplicate_identity_resolution"][0]["selected_source_file"])

    def test_nonexistent_original_source_cannot_be_silently_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stem=("dmspf16_ssusi_edr-aurora_2006012T222703-"
                  "2006013T000454-REV11544_")
            for rev in ("r000","r001"):
                (root/(stem+"vA8.2.0"+rev+".nc")).write_bytes(b"a")
            records,_=scan_month(root,"2006-01")
            prior=[{**record(records[0]),
                    "source_file":str(root/"file_that_does_not_exist.nc")}]
            with self.assertRaises(ValueError):
                scan_month(root,"2006-01",old_cases=prior)

    def test_requires_original_catalog_unchanged(self):
        old,records,quality,orig,_=self.fixture()
        old[0]["acquisition_start"]="2006-01-03T06:00:00"
        with self.assertRaises(ValueError):
            choose_quality_cohort(old,records,quality,expected=4)

if __name__=="__main__":
    unittest.main()

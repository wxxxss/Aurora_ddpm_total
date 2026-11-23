#!/usr/bin/env python3
"""No private data required. Regression for empty/truncated OVATION .npy scans."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from survey_crossyear_real_inputs import audit_month, main

class TestCrossyearSurvey(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.edr=self.root/"real_aurora_data_ssusi/f16/2006/001/edr_fake.nc"
        self.edr.parent.mkdir(parents=True)
        self.edr.write_bytes(b"EDR placeholder; survey counts filenames only")
        self.omni=self.root/"omni_real_data/omni_5min/2006/omni_20060101_5min.npy"
        self.omni.parent.mkdir(parents=True)
        omni=np.zeros(2,dtype=[("utc","datetime64[s]"),("Bx","f4")])
        omni["utc"]=np.datetime64("2006-01-01")+np.arange(2)*np.timedelta64(5,"m")
        np.save(self.omni,omni)
        self.ovation=self.root/"generated_aurora_data/2006_omni_aurora/aurora_img_20060101.npy"
        self.ovation.parent.mkdir(parents=True)

    def test_zero_byte_ovation_does_not_abort(self):
        self.ovation.write_bytes(b"")
        r=audit_month(2006,1,self.root)
        self.assertEqual(r["n_native_ssusi_edr"],1)
        self.assertEqual(r["ovation_status"],"empty")
        self.assertEqual(r["ovation_file_size_bytes"],0)
        self.assertEqual(r["omni_status"],"valid")
        self.assertFalse(r["ready_for_preflight_candidate_scan"])

    def test_invalid_binary_ovation_does_not_abort(self):
        self.ovation.write_bytes(b"not a numpy file")
        r=audit_month(2006,1,self.root)
        self.assertEqual(r["ovation_status"],"unreadable")
        self.assertIn("ValueError",r["ovation_error"])
        self.assertFalse(r["ready_for_preflight_candidate_scan"])

    def test_truncated_npy_does_not_abort(self):
        np.save(self.ovation,np.zeros((3,80,96),dtype=np.float32))
        with self.ovation.open("r+b") as f:
            f.truncate(256)
        r=audit_month(2006,1,self.root)
        self.assertEqual(r["ovation_status"],"unreadable")
        self.assertIn("Error",r["ovation_error"])
        self.assertFalse(r["ready_for_preflight_candidate_scan"])

    def test_writes_report_despite_bad_month(self):
        self.ovation.write_bytes(b"")
        destination=self.root/"crossyear_real_input_survey.json"
        with patch.object(sys,"argv",["survey_crossyear_real_inputs.py",
                                      "--data-root",str(self.root),
                                      "--years","2006","--output",str(destination)]):
            main()
        data=json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(data["n_months_checked"],12)
        self.assertEqual(data["n_months_with_any_ssusi"],1)
        self.assertEqual(data["n_months_with_all_potential_inputs"],0)
        self.assertEqual(data["n_ssusi_months_with_missing_or_unreadable_inputs"],1)
        self.assertEqual(data["ssusi_month_input_issues"][0]["ovation_status"],"empty")

if __name__=="__main__":
    unittest.main()

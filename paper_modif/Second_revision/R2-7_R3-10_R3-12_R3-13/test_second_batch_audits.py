#!/usr/bin/env python3
"""Synthetic no-private-data checks for batch-2 audit invariants.
Run from repository root:
    python -m unittest discover -s paper_modif/Second_revision/R2-7_R3-10_R3-12_R3-13 -p "test_*.py"
"""
from __future__ import annotations
import unittest
import numpy as np

from audit_grid_calibration import masked_average_audit
from audit_statistical_units import event_psd

class TestGridSupport(unittest.TestCase):
    def test_native_support_and_duplicate_bin(self):
        f=np.full((80,96),np.nan)
        count=np.zeros((80,96),dtype=int)
        count[20,40]=2
        f[20,40]=1.5
        s=count>0
        r=masked_average_audit(f,s,count)
        self.assertEqual(r["cells_with_native_support"],1)
        self.assertEqual(r["cells_receiving_two_or_more_native_samples"],1)
        self.assertEqual(r["total_native_samples_in_target_cells"],2)
        self.assertEqual(r["cells_without_native_support"],7679)

    def test_no_synthetic_measurements_in_missing_cells(self):
        f=np.zeros((80,96),dtype=float)  # fake observed zeros everywhere
        count=np.zeros((80,96),dtype=int)
        with self.assertRaises(AssertionError):
            masked_average_audit(f,count>0,count)

    def test_inconsistent_mask_rejected(self):
        f=np.full((80,96),np.nan)
        count=np.zeros((80,96),dtype=int)
        mask=np.zeros((80,96),dtype=bool)
        mask[0,0]=True
        with self.assertRaises(AssertionError):
            masked_average_audit(f,mask,count)

class TestCorrelation(unittest.TestCase):
    def test_event_level_not_segment_level(self):
        def r(case,row,s,o):
            return {"case":case,"row_index":str(row),"n_bins":"19",
                    "ssusi_high_frequency_fraction":str(s),
                    "ovation_high_frequency_fraction":str(o),
                    "ssusi_spectral_slope":"-2","ovation_spectral_slope":"-1"}
        rows=[r("Figure6",4,.4,.2),r("Figure6",5,.6,.3),
              r("Figure7",6,.2,.4),r("Figure7",7,.1,.3)]
        groups=event_psd(rows)
        self.assertEqual(len(groups),2)
        self.assertEqual([g["n_correlated_segments"] for g in groups],[2,2])
        self.assertAlmostEqual(groups[0]["mean_hf_fraction_difference_ssusi_minus_ovation"],.25)
        self.assertAlmostEqual(groups[1]["mean_hf_fraction_difference_ssusi_minus_ovation"],-.2)

if __name__=="__main__":
    unittest.main()

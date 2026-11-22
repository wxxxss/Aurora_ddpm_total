#!/usr/bin/env python3
"""Regression tests for the summary-only real-SSUSI benchmark audit."""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from audit_multievent_results import grouped_truth_check,summarize,check_external_summary

METHODS=("conditional_ddpm","unconditional_ddpm","ovation_prime",
         "ovation_with_observations","interpolation")

def fake_frame():
    rows=[]
    for event in range(12):
        for width in (1.,2.,4.):
            for m in METHODS:
                base={"case":f"F16_{event:02d}","width_h":width,
                      "rank":1,"center_mlt_h":.5,"holdout_pixels":120,
                      "coverage":.75,"signal_fraction":.2,"selection_score":0.,
                      "n_pixels":120,"truth_mean":.4,"truth_peak":6.,
                      "truth_gradient_mean":.25,"selection_mode":"random"}
                baseline_rmse={"conditional_ddpm":1.2,"unconditional_ddpm":1.3,
                               "ovation_prime":1.6,"ovation_with_observations":1.6,
                               "interpolation":1.1}[m]
                row={**base,"method":m,"rmse":baseline_rmse+.03*event,
                     "mae":.5,"pearson_r":.4,"r2":.2,"peak_ratio":.25,
                     "gradient_ratio":.2,"pred_peak":1.5,
                     "pred_gradient_mean":.05}
                rows.append(row)
    return pd.DataFrame(rows)

class AuditChecks(unittest.TestCase):
    def test_complete_paired_random_protocol(self):
        frame=fake_frame()
        self.assertEqual(len(grouped_truth_check(frame)),36)
        x=summarize(frame)
        self.assertEqual(x["n_acquisitions"],12)
        self.assertEqual(x["n_valid_holdout_sectors"],36)
        self.assertEqual(x["n_method_evaluations"],180)
        self.assertAlmostEqual(x["paired_rmse"]["ovation_prime"]
                               ["rmse_delta_conditional_minus_baseline_event_equal_mean"],-.4)

    def test_fused_ovation_disagreement_rejected(self):
        frame=fake_frame()
        frame.loc[frame.method=="ovation_with_observations","rmse"]+=.2
        with self.assertRaises(AssertionError):
            grouped_truth_check(frame)

    def test_target_mask_mismatch_rejected(self):
        frame=fake_frame()
        frame.loc[0,"truth_peak"]=999
        with self.assertRaises(AssertionError):
            grouped_truth_check(frame)

    def test_active_protocol_not_mislabeled_random(self):
        frame=fake_frame()
        frame["selection_mode"]="active"
        with self.assertRaises(ValueError):
            grouped_truth_check(frame)

if __name__=="__main__":
    unittest.main()

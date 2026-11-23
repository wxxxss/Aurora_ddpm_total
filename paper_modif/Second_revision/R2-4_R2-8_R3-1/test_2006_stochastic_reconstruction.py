#!/usr/bin/env python3
"""No weights or SSUSI files needed: synthetic stochastic R3-8 protocol tests."""
from __future__ import annotations
import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_2006_stochastic_reconstruction import (
    WIDTHS, N_REPEATS, describe_ensemble, aggregate,
    read_reference, make_mask_check
)

class TestStochasticUncertainty(unittest.TestCase):
    def test_zero_spread_and_exact_prediction_coverage(self):
        y=np.array([0.,1.,2.,3.])
        x=np.repeat(y[None,:],5,axis=0)
        rec=describe_ensemble(x,y)
        self.assertEqual(rec["n_draws"],5)
        self.assertEqual(rec["mean_pixelwise_stochastic_std"],0.)
        self.assertEqual(rec["mean_per_draw_rmse"],0.)
        self.assertEqual(rec["rmse_of_ensemble_mean"],0.)
        self.assertEqual(rec["fraction_truth_inside_empirical_draw_q10_q90"],1.)
        self.assertEqual(rec["fraction_truth_exactly_zero"],.25)

    def test_multiple_seed_spread_is_not_confused_with_ensemble_mean_error(self):
        y=np.zeros(2)
        x=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.]])
        rec=describe_ensemble(x,y)
        self.assertEqual(rec["n_draws"],N_REPEATS)
        self.assertEqual(rec["mean_per_draw_rmse"],2.)
        self.assertEqual(rec["rmse_of_ensemble_mean"],2.)
        self.assertAlmostEqual(rec["std_per_draw_rmse"],math.sqrt(2.5))
        self.assertAlmostEqual(rec["mean_pixelwise_stochastic_std"],math.sqrt(2.5))
        self.assertEqual(rec["fraction_truth_inside_empirical_draw_q10_q90"],0.)

    def test_zero_truth_does_not_create_nan_ratios(self):
        x=np.array([[.1,.2],[0.,.0],[.2,.3],[.0,.0],[.3,.3]])
        rec=describe_ensemble(x,np.zeros(2))
        self.assertEqual(rec["fraction_truth_exactly_zero"],1.)
        self.assertTrue(math.isfinite(rec["mean_per_draw_rmse"]))
        self.assertTrue(math.isfinite(rec["mean_pixelwise_stochastic_std"]))

    def test_nonfinite_prediction_is_not_silent(self):
        with self.assertRaises(ValueError):
            describe_ensemble(np.array([[0.,1.],[0.,np.nan]]),np.array([0.,1.]))

    def test_exact_36_window_and_five_method_reference_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"metrics.csv"
            methods=["conditional_ddpm","unconditional_ddpm","ovation_prime",
                     "ovation_with_observations","interpolation"]
            with p.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=["case","width_h","rank","method"])
                writer.writeheader()
                for i in range(12):
                    for width in WIDTHS:
                        for m in methods:
                            writer.writerow({"case":str(i),"width_h":width,
                                             "rank":1,"method":m})
            r=read_reference(p)
            self.assertEqual(len(r),36)
            for k,v in r.items():
                self.assertEqual(v["method"],"conditional_ddpm")
            # One method accidentally missing in a sector is a hard failure.
            p.write_text(p.read_text().replace("ovation_prime","not_a_baseline",1))
            with self.assertRaises(ValueError):
                read_reference(p)

    def test_aggregate_requires_all_twelve_cases_and_fixed_sectors(self):
        one={
            "rank":1,"mean_per_draw_rmse":.6,"std_per_draw_rmse":.1,
            "mean_pixelwise_stochastic_std":.2,
            "fraction_truth_inside_empirical_draw_q10_q90":.3
        }
        rows=[{**one,"case":f"case{i}","width_h":float(w)}
              for i in range(12) for w in (1,2,4)]
        agg=aggregate(rows)
        self.assertEqual(agg["n_total_conditional_model_draws"],180)
        self.assertEqual(agg["acquisition_equal_summary"]["std_per_draw_rmse"],.1)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])

if __name__=="__main__":
    unittest.main()

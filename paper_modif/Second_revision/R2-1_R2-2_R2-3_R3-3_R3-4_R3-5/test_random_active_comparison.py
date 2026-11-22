#!/usr/bin/env python3
"""Self-contained synthetic checks for random-versus-active SSUSI audit."""
from __future__ import annotations
import unittest
import pandas as pd
from compare_random_active import check, describe

METHODS=("conditional_ddpm","unconditional_ddpm","ovation_prime",
         "ovation_with_observations","interpolation")

def synthetic(mode="random"):
    rows=[]
    for event in range(12):
        for width in (1.,2.,4.):
            for method in METHODS:
                score=0. if mode=="random" else 2. + width
                rmse={
                    "conditional_ddpm":1.0,
                    "unconditional_ddpm":1.1,
                    "ovation_prime":1.4,
                    "ovation_with_observations":1.4,
                    "interpolation":0.9,
                }[method]
                rows.append(dict(case=f"F16_{event:02d}",width_h=width,rank=1,
                    center_mlt_h=6.,holdout_pixels=40,n_pixels=40,
                    coverage=.9,signal_fraction=.5,
                    truth_mean=1.,truth_peak=6.,truth_gradient_mean=.4,
                    pred_mean=.5,pred_peak=2.,pred_gradient_mean=.2,
                    selection_mode=mode,selection_score=score,
                    method=method,rmse=rmse,mae=.5,pearson_r=.5,r2=.25,
                    peak_ratio=1/3,gradient_ratio=.5))
    return pd.DataFrame(rows)

class TestAudit(unittest.TestCase):
    def test_random_and_active_complete(self):
        for mode in ("random","active"):
            df=synthetic(mode)
            status=check(df,mode)
            self.assertEqual(status["n_cases"],12)
            self.assertEqual(status["n_sectors"],36)
            self.assertEqual(status["n_evaluations"],180)
            desc=describe(df)
            self.assertEqual(desc["paired"]["ovation_prime"]["conditional_wins_acquisitions"],12)
            self.assertAlmostEqual(desc["methods"]["conditional_ddpm"]
                                   ["ratio_of_predicted_to_measured_gradient_means"],.5)

    def test_fused_and_direct_ovation_must_match(self):
        df=synthetic()
        df.loc[df.method=="ovation_with_observations","rmse"]=2.0
        with self.assertRaises(AssertionError):
            check(df,"random")

    def test_flux_based_selection_not_allowed_in_random_primary(self):
        df=synthetic()
        df.loc[0,"selection_score"]=10.
        with self.assertRaises(ValueError):
            check(df,"random")

    def test_target_disagreement_rejected(self):
        df=synthetic()
        df.loc[0,"truth_peak"]=100.
        with self.assertRaises(AssertionError):
            check(df,"random")

if __name__=="__main__":
    unittest.main()

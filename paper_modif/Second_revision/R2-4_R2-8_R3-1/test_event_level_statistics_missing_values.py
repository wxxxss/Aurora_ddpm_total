#!/usr/bin/env python3
"""Regression: undefined real SSUSI metrics are blank, NOT zero.

Run: python -m unittest discover -s paper_modif/Second_revision/R2-4_R2-8_R3-1 -p 'test_*.py'
No model weights, private SSUSI, or OMNI input are needed.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
THIRD=HERE.parent/"R2-1_R2-2_R2-3_R3-3_R3-4_R3-5"
if str(THIRD) not in sys.path:
    sys.path.insert(0,str(THIRD))

from event_level_statistics import (METHODS, read_rows, summarize,
                                    validate_and_aggregate, value)

def make_synthetic_metrics():
    rows=[]
    for case in ("F16_2006010T004723_REV11503","F16_2006017T143452_REV11610"):
        for width in (1.,2.,4.):
            for method in METHODS:
                # The first holdout has identically zero observed flux.
                # Some Pearson/ratio metrics are mathematically undefined.
                undefined=width==1.
                rows.append({
                    "case":case,"width_h":str(width),"rank":"1",
                    "selection_mode":"random","method":method,
                    "rmse":"0.0" if method=="interpolation" and undefined else "0.5",
                    "mae":"0.0" if method=="interpolation" and undefined else "0.25",
                    "pearson_r":"" if undefined else "0.4",
                    "r2":"" if undefined else "-0.2",
                    "peak_ratio":"" if undefined else "0.3",
                    "gradient_ratio":"" if undefined else "0.2",
                    "pred_peak":"0.2",
                    "truth_peak":"0.0" if undefined else "0.8"
                })
    return rows

class TestUndefinedMetrics(unittest.TestCase):
    def test_blank_nan_optional_metrics_are_preserved_as_undefined(self):
        rows=make_synthetic_metrics()
        self.assertTrue(value(rows[0],"pearson_r")!=value(rows[0],"pearson_r"))
        self.assertTrue(value(rows[0],"peak_ratio")!=value(rows[0],"peak_ratio"))
        self.assertAlmostEqual(value(rows[0],"peak_abs_error"),.2)
        events,pairs,summary=summarize(rows,n_boot=0)
        # Each EDR has three masks, but only two have defined correlations.
        event=next(r for r in events if r["method"]=="conditional_ddpm")
        self.assertEqual(event["n_valid_pearson_r"],2)
        self.assertEqual(event["n_valid_rmse"],3)
        self.assertAlmostEqual(event["pearson_r"],.4)
        pair=next(r for r in pairs if r["baseline"]=="interpolation")
        self.assertEqual(pair["n_valid_paired_pearson_r"],2)
        self.assertEqual(pair["n_valid_paired_rmse"],3)
        self.assertEqual(summary["n_correlated_heldout_sectors"],6)
        # No NaN/Infinity is serialized into JSON output.
        json.dumps(summary,allow_nan=False)

    def test_empty_metrics_for_whole_event_produce_null_not_zero(self):
        rows=make_synthetic_metrics()
        for row in rows:
            if row["case"]=="F16_2006010T004723_REV11503":
                row["pearson_r"]=""
                row["r2"]=""
                row["peak_ratio"]=""
                row["gradient_ratio"]=""
        events,pairs,summary=summarize(rows,n_boot=0)
        first=next(r for r in events if r["case"]=="F16_2006010T004723_REV11503")
        self.assertIsNone(first["pearson_r"])
        self.assertEqual(first["n_valid_pearson_r"],0)
        self.assertIsNone(next(r for r in pairs if r["case"]==first["case"])[
            "conditional_minus_baseline_pearson_r"])
        json.dumps(summary,allow_nan=False)

    def test_direct_fused_both_undefined_is_ok_but_mismatch_is_error(self):
        rows=make_synthetic_metrics()
        fused=next(r for r in rows if r["method"]=="ovation_with_observations"
                   and r["width_h"]=="1.0")
        fused["pearson_r"]="0.4"
        with self.assertRaises(AssertionError):
            validate_and_aggregate(rows)

    def test_required_rmse_cannot_be_missing(self):
        rows=make_synthetic_metrics()
        rows[0]["rmse"]=""
        with self.assertRaisesRegex(ValueError,"Required metric"):
            validate_and_aggregate(rows)

    def test_malformed_optional_metric_is_not_silently_ignored(self):
        rows=make_synthetic_metrics()
        rows[0]["pearson_r"]="corrupt_not_number"
        with self.assertRaisesRegex(ValueError,"Malformed numeric"):
            validate_and_aggregate(rows)

if __name__=="__main__":
    unittest.main()

#!/usr/bin/env python3
"""Synthetic clock/atomic-write regressions; no CDF, satellite or model needed."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np

from audit_and_prepare_2006_omni import (
    month_grid,validate_calendar_grid,write_npy_safely
)

class TestAuditedOMNI(unittest.TestCase):
    def test_january_2006_5minute_axis(self):
        x=month_grid(2006,1)
        self.assertEqual(len(x),31*24*12)
        self.assertEqual(str(x[0]),"2006-01-01T00:00:00.000000000")
        self.assertEqual(str(x[-1]),"2006-01-31T23:55:00.000000000")
        y=validate_calendar_grid(x,2006,1)
        self.assertTrue(y["exact_regular_complete_5min_utc"])
        self.assertEqual(y["n_irregular_or_duplicate_5min_steps"],0)

    def test_rejects_missing_or_extra_ticks(self):
        x=month_grid(2006,1)
        self.assertFalse(validate_calendar_grid(x[:-1],2006,1)[
            "exact_regular_complete_5min_utc"])
        shifted=x.copy()
        shifted[60]+=np.timedelta64(1,"m")
        self.assertFalse(validate_calendar_grid(shifted,2006,1)[
            "exact_regular_complete_5min_utc"])

    def test_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"omni_20060101_5min.npy"
            data=np.zeros(2,dtype=[("utc","datetime64[ns]"),
                                   ("Bx","f4"),("By","f4"),("Bz","f4"),
                                   ("V","f4"),("P","f4")])
            write_npy_safely(out,data)
            self.assertTrue(out.is_file())
            self.assertEqual(np.load(out,mmap_mode="r").shape[0],2)
            with self.assertRaises(FileExistsError):
                write_npy_safely(out,data)

if __name__=="__main__":unittest.main()

#!/usr/bin/env python3
"""No raw CDF or DDPM needed: exact masked-orbit OMNI validity regression."""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from audit_2006_omni_event_coverage import run
from audit_and_prepare_2006_omni import month_grid

RANGES={"Bx":(-50.,50.),"By":(-50.,50.),"Bz":(-50.,50.),"V":(200.,2000.),"P":(.05,80.)}
FIELDS=("Bx","By","Bz","V","P")

def make_data():
    grid=month_grid(2006,1)
    data={"utc":pd.to_datetime(grid)}
    for f in FIELDS:
        data[f]=np.full(len(grid),400. if f=="V" else
                             2. if f=="P" else 1.,dtype=float)
    cases=[
        {"case_id":"first","acquisition_start":"2006-01-02T05:54:08",
         "acquisition_end":"2006-01-02T07:35:59"},
        {"case_id":"second","acquisition_start":"2006-01-04T20:45:51",
         "acquisition_end":"2006-01-04T22:27:42"}
    ]
    return pd.DataFrame(data),{"month":"2006-01","cases":cases}

class TestEventOMNI(unittest.TestCase):
    def test_intermittent_13_hour_gap_outside_orbit_is_not_imputed(self):
        df,catalog=make_data()
        # Both cases are 2 and 4 Jan, missing block is 20 Jan.
        missing=(df.utc>=pd.Timestamp("2006-01-20T00:00")) & (
                 df.utc<=pd.Timestamp("2006-01-20T13:40"))
        df.loc[missing,list(FIELDS)]=np.nan
        out=run(catalog,df,SimpleNamespace(VALID_RANGES=RANGES))
        self.assertEqual(out["n_fully_raw_valid_acquisitions"],2)
        self.assertEqual(out["raw_invalid_total_5min_records_any_field"],165)

    def test_invalid_omni_inside_one_orbit_is_flagged(self):
        df,catalog=make_data()
        df.loc[df.utc==pd.Timestamp("2006-01-02T06:00:00"),"Bx"]=np.nan
        out=run(catalog,df,SimpleNamespace(VALID_RANGES=RANGES))
        self.assertFalse(out["all_preselected_acquisitions_raw_valid"])
        self.assertEqual(out["n_fully_raw_valid_acquisitions"],1)
        self.assertEqual(out["cases"][0]["n_invalid_by_field"]["Bx"],1)
        self.assertEqual(out["cases"][1]["n_any_field_invalid_inside"],0)

    def test_invalid_month_selection_rejected(self):
        df,catalog=make_data()
        catalog["month"]="2005-01"
        with self.assertRaises(ValueError):
            run(catalog,df,SimpleNamespace(VALID_RANGES=RANGES))

if __name__=="__main__":
    unittest.main()

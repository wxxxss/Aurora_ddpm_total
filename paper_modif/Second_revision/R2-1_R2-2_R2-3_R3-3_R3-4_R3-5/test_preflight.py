#!/usr/bin/env python3
"""Synthetic preflight tests. Do not require original EDR or model weights."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from preflight_multievent_benchmark import scan_case, check_inputs

class PreflightTests(unittest.TestCase):
    def test_sufficient_native_support_produces_random_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            name="F16_synthetic_REV00001"
            folder=root/name
            folder.mkdir()
            lat=np.linspace(50,90,80)
            mlt=np.arange(96)*.25
            rows=(lat>=60)&(lat<=80)
            mask=np.zeros((80,96),dtype=bool)
            mask[rows]=True
            count=mask.astype(np.int32)
            flux=np.full((80,96),np.nan,dtype=np.float32)
            flux[mask]=0.0  # valid zero flux, must not be dropped
            np.savez_compressed(folder/(name+"_swath_case.npz"),
                                flux_grid=flux,obs_mask=mask.astype(np.uint8),
                                count_grid=count,mlat_1d=lat,mlt_1d=mlt)
            meta={"acquisition_start":"2005-01-02T06:01:58",
                  "acquisition_end":"2005-01-02T07:43:49"}
            (folder/(name+"_metadata.json")).write_text(json.dumps(meta))
            result=scan_case(root,{"case_id":name,"orbit":"00001",**meta},
                             widths=[1.,2.,4.],min_coverage=.5,seed=2026,windows_per_width=1)
            self.assertTrue(result["all_holdouts_evaluable"])
            self.assertTrue(all(v["n_selected"]==1 for v in result["selection_by_width"].values()))
            self.assertEqual(result["native_supported_cells"],int(mask.sum()))

    def test_omni_and_ovation_utc_coverage_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            omn=root/"omni_20050101_5min.npy"
            ova=root/"aurora_img_20050101.npy"
            fields=[("utc","datetime64[s]")]+[(k,"f4") for k in ("Bx","By","Bz","V","P")]
            values=np.zeros(12,dtype=fields)
            values["utc"]=np.datetime64("2005-01-01T00:00:00")+np.arange(12)*np.timedelta64(5,"m")
            for k in ("Bx","By","Bz","V","P"):
                values[k]=1
            np.save(omn,values)
            np.save(ova,np.zeros((61,80,96),dtype=np.float32))
            row={"case_id":"test","acquisition_start":"2005-01-01T00:10:00",
                 "acquisition_end":"2005-01-01T00:20:00"}
            result=check_inputs("2005-01",[row],omn,ova)
            self.assertFalse(result["errors"])
            self.assertTrue(result["per_case"]["test"]["ovation_full_interval_available"])
            np.save(ova,np.zeros((10,80,96),dtype=np.float32))
            bad=check_inputs("2005-01",[row],omn,ova)
            self.assertTrue(bad["errors"])

if __name__=="__main__":
    unittest.main()

#!/usr/bin/env python3
"""Synthetic, model-free tests: four-case time basis + lossless PNG annotation."""
from __future__ import annotations
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from annotate_figures_4_7_solar_wind import (
    EXPECTED_INTERVALS, source_context, label
)

class TestFigureSolarWindAnnotations(unittest.TestCase):
    def test_original_scientific_pixels_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            source=d/"original.png"
            result=d/"labeled.png"
            original=Image.new("RGB",(1600,700),(27,75,121))
            original.save(source)
            row={"interval":"2005-01-01 14:44:34--16:26:25",
                 "Bz":"1.693333","V":"459.076","P":"3.184761"}
            info=label("Figure6",source,result,row)
            self.assertTrue(info["original_pixels_preserved"])
            with Image.open(result) as img:
                self.assertEqual(img.size[0],1600)
                self.assertGreater(img.size[1],700)
                self.assertEqual(img.crop((0,0,1600,700)).tobytes(),original.tobytes())
            self.assertEqual(info["omni_time_basis"],"acquisition-interval mean of 5-minute OMNI")

    def test_polar_instantaneous_basis(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            im=d/"p.png"
            dst=d/"labeled.png"
            Image.new("RGB",(1600,700),"black").save(im)
            row={"interval":"1996-04-03 02:15",
                 "Bz":"-1.57","V":"325.4","P":"2.91"}
            info=label("Figure4",im,dst,row)
            self.assertEqual(info["omni_time_basis"],"exact-UTC 5-minute OMNI measurement")

    def test_rejects_polar_wrong_omni_time(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"source.csv"
            with p.open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=["event","interval",
                    "Bz","V","P","condition_matched_time",
                    "condition_offset_s","condition_source"])
                writer.writeheader()
                for fig,when in EXPECTED_INTERVALS.items():
                    writer.writerow({
                        "event":fig.replace("Figure","Figure "),
                        "interval":when,
                        "Bz":-1.0,"V":400.0,"P":2.0,
                        "condition_matched_time":("1996-04-03T02:10:00" if fig=="Figure4" else when.replace(" ","T")+":00"),
                        "condition_offset_s":0.,
                        "condition_source":("1996/omni.npy" if fig in ("Figure4","Figure5") else "2005/omni.npy")
                    })
            with self.assertRaisesRegex(ValueError,"expected exactly matched"):
                source_context(p)

if __name__=="__main__":
    unittest.main()

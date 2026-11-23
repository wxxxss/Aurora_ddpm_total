#!/usr/bin/env python3
"""AST only: no torch imports, no checkpoint reads, no code execution."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from inspect_historical_training_sources import inspect,class_structure

class TestHistoricalUNetStructure(unittest.TestCase):
    def test_two_different_class_bodies(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"unet_a.py"
            b=Path(d)/"unet_b.py"
            a.write_text("class UNet:\n    def __init__(self, a=1):\n        self.a=a\n")
            b.write_text("class UNet:\n    def __init__(self, a=1):\n        self.a=a\n        self.extra=2\n")
            result=inspect([a,b],[])
            self.assertEqual(result["n_distinct_available_unet_source_hashes"],2)
            self.assertFalse(result["same_unet_ast_if_multiple_versions"])
            self.assertEqual(result["unet_structural_comparison"][0][
                "unet_constructor_arguments"],["self","a"])
            self.assertFalse(result["checkpoint_training_lineage_verified"])

    def test_no_loading_untrusted_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"arch.py"
            p.write_text("raise RuntimeError('this file must NEVER run')\n"
                         "class UNet:\n    pass\n")
            r=class_structure(p)
            self.assertTrue(r["unet_class_found"])

if __name__=="__main__":
    unittest.main()

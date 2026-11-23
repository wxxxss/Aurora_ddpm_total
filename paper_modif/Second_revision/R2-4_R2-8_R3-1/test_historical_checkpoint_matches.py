#!/usr/bin/env python3
"""Synthetic bytes only: provenance audit does not load Torch or private weights."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from audit_historical_checkpoint_matches import audit,list_checkpoints

class TestCheckpointInventory(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.current=self.root/"current"
        self.archive=self.root/"old_archive"
        self.current.mkdir()
        self.archive.mkdir()

    def test_byte_identity_and_same_size_distinguished(self):
        cond=self.current/"conditional.pth"
        uncond=self.current/"unconditional.pth"
        cond.write_bytes(b"this is the conditional trained checkpoint")
        uncond.write_bytes(b"UNCONDITIONAL FILE")
        d=self.archive/"ckptv4_unetv1"
        d.mkdir()
        identical=d/"checkpoint_epoch_85.pth"
        identical.write_bytes(cond.read_bytes())
        same_size=d/"aurora_diff_best.pth"
        same_size.write_bytes(b"!"*cond.stat().st_size)
        unrelated=d/"other.pth"
        unrelated.write_bytes(b"a different length")
        result=audit(cond,uncond,[self.archive])
        self.assertEqual(result["n_archive_checkpoint_files"],3)
        self.assertIn(str(identical),result["byte_identical_archive_paths"]["conditional"])
        self.assertIn(str(same_size),result["matching_size_archive_paths"]["conditional"])
        self.assertNotIn(str(same_size),result["byte_identical_archive_paths"]["conditional"])
        self.assertEqual(result["byte_identical_archive_paths"]["unconditional"],[])
        self.assertFalse(result["training_lineage_verified"])
        self.assertEqual(result["n_archive_files_sha256_read"],2)

    def test_does_not_silently_truncate_inventory(self):
        for i in range(115):
            (self.archive/f"checkpoint_epoch_{i:03d}.pth").write_bytes(b"a")
        files,roots=list_checkpoints([self.archive])
        self.assertEqual(len(files),115)
        self.assertTrue(roots[0]["exists"])

    def test_missing_root_reported_and_not_failed(self):
        cond=self.current/"conditional.pth"
        uncond=self.current/"unconditional.pth"
        cond.write_bytes(b"abc")
        uncond.write_bytes(b"xyz")
        missing=self.root/"old_does_not_exist"
        result=audit(cond,uncond,[missing])
        self.assertEqual(result["n_archive_checkpoint_files"],0)
        self.assertFalse(result["archive_roots"][0]["exists"])
        self.assertFalse(result["binary_identity_confirmed_for_conditional"])

    def test_missing_actual_checkpoint_is_error(self):
        with self.assertRaises(FileNotFoundError):
            audit(self.root/"no_model.pth",self.root/"nothing.pth",[self.archive])

if __name__=="__main__":
    unittest.main()

#!/usr/bin/env python3
"""Synthetic unit tests for center-aligned SSUSI binning; no private EDR data."""
import unittest
import numpy as np
from prepare_center_aligned_ssusi import native_to_centered, LAT_CENTERS

class TestCenterAligned(unittest.TestCase):
    def test_mlt_nearest_center_and_midnight_wrap(self):
        # 0.20h belongs to center 0.25h and 23.90h wraps to center 0h.
        # Use an EXACT declared latitude-grid center rather than 70 deg,
        # which lies halfway between rows 39 and 40. np.linspace-derived
        # LAT_STEP differs slightly from the theoretical 40/79 floating
        # value, so midpoint rounding is implementation-sensitive.
        f=np.array([[2.0,4.0,6.0]])
        lat=np.full((1,3),float(LAT_CENTERS[40]))
        mlt=np.array([[0.20,23.90,0.01]])
        ut=np.ones_like(f)
        flux,mask,count,info=native_to_centered(f,lat,mlt,ut)
        j=40
        self.assertEqual(count[j,1],1)
        self.assertEqual(count[j,0],2)
        self.assertAlmostEqual(float(flux[j,1]),2.0)
        self.assertAlmostEqual(float(flux[j,0]),5.0)
        self.assertEqual(info["assigned_native_samples"],3)

    def test_no_support_when_ut_absent(self):
        f=np.array([[3.0,7.0]])
        lat=np.array([[69.0,69.0]])
        mlt=np.array([[2.0,2.25]])
        ut=np.array([[0.0,np.nan]])
        flux,mask,count,info=native_to_centered(f,lat,mlt,ut)
        self.assertEqual(count.sum(),0)
        self.assertEqual(mask.sum(),0)
        self.assertTrue(np.isnan(flux).all())
        self.assertEqual(info["supported_cells"],0)

    def test_low_but_valid_flux_preserved(self):
        f=np.array([[0.0,0.01]])
        lat=np.array([[72.0,72.0]])
        mlt=np.array([[11.0,11.0]])
        ut=np.array([[1.0,2.0]])
        flux,mask,count,info=native_to_centered(f,lat,mlt,ut)
        j=int(np.rint((72.0-50.0)/(40.0/79.0)))
        self.assertEqual(count[j,44],2)
        self.assertAlmostEqual(float(flux[j,44]),0.005,places=5)
        self.assertEqual(mask[j,44],1)

    def test_latitude_beyond_domain_ignored(self):
        f=np.array([[1.0,2.0,3.0]])
        lat=np.array([[45.0,75.0,91.0]])
        mlt=np.array([[10.0,10.0,10.0]])
        ut=np.ones_like(f)
        flux,mask,count,info=native_to_centered(f,lat,mlt,ut)
        self.assertEqual(count.sum(),1)
        self.assertEqual(info["assigned_native_samples"],1)

if __name__=="__main__":
    unittest.main()

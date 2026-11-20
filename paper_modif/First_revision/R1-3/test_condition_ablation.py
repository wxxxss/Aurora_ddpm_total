#!/usr/bin/env python3
"""Unit tests for Reviewer 1 Comment 3 condition-ablation helpers."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from condition_ablation_utils import (
    apply_condition_variant,
    bootstrap_mean_ci,
    compute_masked_metrics,
    create_paper_mask,
    make_derangement,
    metric_degradation,
)


def test_create_paper_mask_matches_manuscript_missing_sector():
    mask = create_paper_mask((80, 96), mlat_range=(60.0, 80.0))
    assert mask.shape == (80, 96)
    assert set(np.unique(mask)) == {0.0, 1.0}
    # Rows 20:60 correspond to 60--80 MLAT in the legacy paper grid;
    # columns 72:96 and 0:24 correspond to 18--24 and 0--6 MLT.
    assert int(np.sum(mask == 0)) == 40 * 48
    assert np.all(mask[20:60, 72:96] == 0)
    assert np.all(mask[20:60, 0:24] == 0)
    assert np.all(mask[:20] == 1)


def test_make_derangement_is_deterministic_and_has_no_fixed_points():
    a = make_derangement(12, seed=2026)
    b = make_derangement(12, seed=2026)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(a), np.arange(12))
    assert np.all(a != np.arange(12))


def test_apply_condition_variant_changes_only_requested_component():
    conditions = np.arange(20, dtype=float).reshape(4, 5)
    perm = np.array([2, 3, 0, 1])

    full = apply_condition_variant(conditions, "full", perm)
    bz = apply_condition_variant(conditions, "permute_Bz", perm)
    all_perm = apply_condition_variant(conditions, "permute_all", perm)

    np.testing.assert_array_equal(full, conditions)
    np.testing.assert_array_equal(bz[:, :2], conditions[:, :2])
    np.testing.assert_array_equal(bz[:, 3:], conditions[:, 3:])
    np.testing.assert_array_equal(bz[:, 2], conditions[perm, 2])
    np.testing.assert_array_equal(all_perm, conditions[perm])
    # The input array must never be modified in place.
    np.testing.assert_array_equal(conditions, np.arange(20, dtype=float).reshape(4, 5))


def test_metric_degradation_is_positive_when_ablation_is_worse():
    full = {"ssim": 0.90, "psnr": 25.0, "rmse": 0.20, "r2": 0.85, "mae": 0.10}
    worse = {"ssim": 0.80, "psnr": 22.0, "rmse": 0.30, "r2": 0.70, "mae": 0.15}
    delta = metric_degradation(full, worse)
    assert delta["ssim"] == 0.10
    assert delta["psnr"] == 3.0
    assert np.isclose(delta["rmse"], 0.10)
    assert np.isclose(delta["r2"], 0.15)
    assert np.isclose(delta["mae"], 0.05)


def test_masked_metrics_and_bootstrap_are_deterministic():
    truth = np.array([[0.0, 1.0], [2.0, 3.0]])
    pred = np.array([[0.0, 1.5], [1.5, 3.0]])
    # 1 means observed/preserved, 0 means held out; evaluate the two off-diagonal pixels.
    mask = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics = compute_masked_metrics(truth, pred, mask)
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["rmse"], 0.5)

    values = np.array([1.0, 2.0, 3.0, 4.0])
    a = bootstrap_mean_ci(values, n_boot=1000, seed=2026)
    b = bootstrap_mean_ci(values, n_boot=1000, seed=2026)
    assert a == b
    assert np.isclose(a["mean"], 2.5)
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]

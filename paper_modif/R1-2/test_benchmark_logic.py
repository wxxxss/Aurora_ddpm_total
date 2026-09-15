import numpy as np
import pandas as pd
import pytest

from benchmark_utils import (
    bootstrap_mean_ci,
    create_paper_mask,
    metric_improvement,
    paired_comparison,
    summarize_methods,
)


def test_create_paper_mask_matches_manuscript_sector():
    mask = create_paper_mask((80, 96), mlat_range=(60, 80))
    assert mask.shape == (80, 96)
    assert np.sum(mask == 0) == 40 * 48
    assert np.all(mask[20:60, :24] == 0)
    assert np.all(mask[20:60, 72:] == 0)
    assert np.all(mask[:20] == 1)
    assert np.all(mask[60:] == 1)


def test_metric_improvement_positive_means_conditional_is_better():
    assert metric_improvement(0.10, 0.20, "rmse") == pytest.approx(0.10)
    assert metric_improvement(0.90, 0.80, "ssim") == pytest.approx(0.10)
    assert metric_improvement(0.85, 0.70, "r2") == pytest.approx(0.15)


def test_bootstrap_ci_is_reproducible_and_contains_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    a = bootstrap_mean_ci(x, n_boot=2000, seed=3)
    b = bootstrap_mean_ci(x, n_boot=2000, seed=3)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["n"] == 4


def test_summarize_methods_filters_group_and_method():
    df = pd.DataFrame([
        {"sample_id": 0, "phase": "solar_maximum", "method": "conditional", "rmse": 1.0, "ssim": 0.9, "psnr": 20.0, "r2": 0.8, "mae": 0.5},
        {"sample_id": 1, "phase": "solar_maximum", "method": "conditional", "rmse": 3.0, "ssim": 0.7, "psnr": 18.0, "r2": 0.6, "mae": 1.5},
        {"sample_id": 0, "phase": "solar_maximum", "method": "interpolation", "rmse": 2.0, "ssim": 0.8, "psnr": 19.0, "r2": 0.7, "mae": 1.0},
    ])
    out = summarize_methods(df, group_col="phase", group_value="solar_maximum", bootstrap_n=500, seed=1)
    row = next(r for r in out if r["method"] == "conditional")
    assert row["n"] == 2
    assert row["mean_rmse"] == pytest.approx(2.0)
    assert row["std_rmse"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_paired_comparison_uses_only_shared_sample_ids():
    df = pd.DataFrame([
        {"sample_id": 0, "group": "g", "method": "conditional", "rmse": 1.0, "ssim": 0.9, "psnr": 20.0, "r2": 0.8, "mae": 0.5},
        {"sample_id": 1, "group": "g", "method": "conditional", "rmse": 2.0, "ssim": 0.8, "psnr": 19.0, "r2": 0.7, "mae": 1.0},
        {"sample_id": 2, "group": "g", "method": "conditional", "rmse": 3.0, "ssim": 0.7, "psnr": 18.0, "r2": 0.6, "mae": 1.5},
        {"sample_id": 0, "group": "g", "method": "interpolation", "rmse": 2.0, "ssim": 0.8, "psnr": 19.0, "r2": 0.7, "mae": 1.0},
        {"sample_id": 2, "group": "g", "method": "interpolation", "rmse": 5.0, "ssim": 0.6, "psnr": 17.0, "r2": 0.5, "mae": 2.0},
    ])
    out = paired_comparison(df, group_col="group", group_value="g", baseline="interpolation", bootstrap_n=500, seed=1)
    assert out["n"] == 2
    assert out["mean_improvement_rmse"] == pytest.approx((1.0 + 2.0) / 2.0)


def test_interpolation_matches_manuscript_zero_fill_outside_convex_hull():
    from benchmark_utils import interpolate_inpainting

    image = np.arange(16, dtype=float).reshape(4, 4)
    mask = np.ones((4, 4), dtype=float)
    mask[:, 0] = 0.0
    corrupted = image.copy()
    corrupted[mask == 0] = 0.0
    out = interpolate_inpainting(corrupted, mask)
    assert np.all(out[:, 0] == 0.0)

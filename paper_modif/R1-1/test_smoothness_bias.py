import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smoothness_bias_utils import (
    contiguous_true_segments_circular,
    supported_edge_gradients,
    segment_periodogram_metrics,
    summarize_distribution,
)


def test_supported_edge_gradients_uses_only_supported_edges_and_lat_band():
    flux = np.array([
        [0., 1., 3., 6.],
        [0., 2., 5., 9.],
        [0., 4., 9., 16.],
    ])
    support = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=bool)
    mlat = np.array([55., 65., 75.])

    out = supported_edge_gradients(flux, support, mlat, band_min=60., band_max=80.)

    expected = np.array([2., 3., 4., 5., 7., 0., 2., 4.])
    np.testing.assert_allclose(np.sort(out), np.sort(expected))


def test_contiguous_true_segments_circular_merges_wraparound_support():
    mask = np.array([1, 1, 0, 0, 1, 1, 1], dtype=bool)
    segments = contiguous_true_segments_circular(mask)
    assert len(segments) == 1
    assert segments[0].tolist() == [4, 5, 6, 0, 1]


def test_segment_periodogram_metrics_detects_more_high_frequency_power():
    x = np.arange(64) * 0.25
    low = np.sin(2 * np.pi * 0.125 * x)
    high = low + 0.8 * np.sin(2 * np.pi * 1.0 * x)

    m_low = segment_periodogram_metrics(low, sample_spacing_hours=0.25, high_freq_cutoff=0.5)
    m_high = segment_periodogram_metrics(high, sample_spacing_hours=0.25, high_freq_cutoff=0.5)

    assert m_high["high_frequency_fraction"] > m_low["high_frequency_fraction"] + 0.2
    assert np.isfinite(m_high["spectral_slope"])


def test_summarize_distribution_reports_expected_quantiles():
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    s = summarize_distribution(x)
    assert s["n"] == 10
    assert s["median"] == pytest.approx(4.5)
    assert s["p90"] == pytest.approx(8.1)
    assert s["p95"] == pytest.approx(8.55)
    assert s["p99"] == pytest.approx(8.91)


def test_collect_matched_psd_segments_uses_same_supported_pixels_and_filters_signal():
    from smoothness_bias_utils import collect_matched_psd_segments

    mlat = np.array([55., 65., 75.])
    mlt = np.arange(16) * 0.25
    a = np.zeros((3, 16), dtype=float)
    b = np.zeros((3, 16), dtype=float)
    a[1] = 1.0 + np.sin(np.linspace(0, 4 * np.pi, 16))
    b[1] = 1.0 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, 16))
    support = np.zeros_like(a, dtype=bool)
    support[1] = True
    support[2] = True

    rows = collect_matched_psd_segments(
        a, b, support, mlat, mlt,
        band_min=60., band_max=80., min_segment_bins=12,
        signal_threshold=0.1, min_signal_fraction=0.25,
    )

    assert len(rows) == 1
    assert rows[0]["mlat"] == pytest.approx(65.0)
    assert rows[0]["n_bins"] == 16
    assert rows[0]["ssusi_high_frequency_fraction"] >= 0.0
    assert rows[0]["ovation_high_frequency_fraction"] >= 0.0


def test_paired_bootstrap_mean_difference_is_reproducible_and_positive():
    from smoothness_bias_utils import paired_bootstrap_mean_difference

    a = np.array([2., 3., 4., 5.])
    b = np.array([1., 1., 2., 2.])
    out1 = paired_bootstrap_mean_difference(a, b, n_boot=500, seed=7)
    out2 = paired_bootstrap_mean_difference(a, b, n_boot=500, seed=7)
    assert out1 == out2
    assert out1["mean_difference"] > 0
    assert out1["ci95_low"] > 0

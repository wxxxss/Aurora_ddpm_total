import numpy as np
from heldout_eval_utils import periodic_mlt_window_mask, compute_metrics, select_holdout_windows


def test_periodic_window_wraps_midnight():
    mlt = np.arange(0, 24, 0.25)
    mask = periodic_mlt_window_mask(mlt, center_h=23.5, width_h=2.0)
    selected = mlt[mask]
    assert 23.0 in selected
    assert 0.0 in selected
    assert 1.0 not in selected
    assert len(selected) == 8


def test_perfect_prediction_metrics():
    truth = np.arange(16, dtype=float).reshape(4, 4)
    pred = truth.copy()
    holdout = np.ones_like(truth, dtype=bool)
    metrics = compute_metrics(truth, pred, holdout)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert abs(metrics["pearson_r"] - 1.0) < 1e-12
    assert abs(metrics["peak_ratio"] - 1.0) < 1e-12
    assert abs(metrics["gradient_ratio"] - 1.0) < 1e-12


def test_select_holdout_windows_targets_observed_signal():
    mlat = np.linspace(50, 90, 80)
    mlt = np.arange(0, 24, 0.25)
    obs = np.ones((80, 96), dtype=bool)
    flux = np.zeros((80, 96), dtype=float)
    band = (mlat >= 60) & (mlat <= 80)
    flux[np.ix_(band, (mlt >= 5) & (mlt < 7))] = 4.0
    flux[np.ix_(band, (mlt >= 17) & (mlt < 19))] = 2.0
    wins = select_holdout_windows(
        flux, obs, mlat, mlt,
        width_h=2.0,
        n_windows=2,
        band=(60, 80),
        min_coverage=0.8,
        min_signal_fraction=0.2,
        signal_threshold=0.1,
        candidate_step_h=0.5,
    )
    assert len(wins) == 2
    centers = [w["center_h"] for w in wins]
    def circdist(a, b):
        return abs(((a - b + 12) % 24) - 12)
    assert any(circdist(c, 6) <= 1.0 for c in centers)
    assert any(circdist(c, 18) <= 1.0 for c in centers)

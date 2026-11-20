import numpy as np
import pandas as pd

from robustness_utils import (
    newell_coupling,
    normalize_kp,
    regularize_hourly,
    select_activity_samples,
    select_evenly_spaced_valid,
    weighted_newell_coupling,
)


def test_normalize_kp_handles_kp_times_ten():
    x, d = normalize_kp(np.array([0.0, 13.0, 43.0, 90.0, np.nan]))
    assert d == 10.0
    np.testing.assert_allclose(x[:4], [0.0, 1.3, 4.3, 9.0])


def test_regularize_hourly_interpolates_only_short_continuous_gaps():
    df = pd.DataFrame({
        "utc": pd.to_datetime(["2001-01-01 00:00", "2001-01-01 02:00", "2001-01-01 03:00"]),
        "Bx": [0.0, 2.0, 3.0], "By": [0.0, 2.0, 3.0], "Bz": [0.0, 2.0, 3.0],
        "V": [400.0, 420.0, 430.0], "P": [2.0, 2.2, 2.3], "Kp": [1.0, 2.0, 2.0],
    })
    out = regularize_hourly(df, 2001, max_interp_hours=1)
    assert len(out) == 8760
    assert out.loc[1, "Bx"] == 1.0
    assert out.loc[1, "Kp"] == 1.0


def test_select_evenly_spaced_valid_is_unique_and_distributed():
    n = 120
    df = pd.DataFrame({"utc": pd.date_range("2001-01-01", periods=n, freq="h")})
    valid = np.ones(n, dtype=bool)
    valid[55:60] = False
    idx = select_evenly_spaced_valid(df, 12, valid)
    assert len(idx) == 12
    assert len(set(idx.tolist())) == 12
    assert valid[idx].all()
    assert idx[0] < 10 and idx[-1] > 110


def test_activity_selection_contains_requested_groups_and_unique_3h_kp_bins():
    rows = []
    for year in [2001, 2005, 2009]:
        # Four low-Kp 3-h bins followed by four high-Kp 3-h bins.
        # Three hourly rows share each nominal Kp interval.
        for i in range(24):
            rows.append({
                "utc": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(hours=i),
                "year": year, "Bx": 1, "By": 1, "Bz": -1, "V": 400, "P": 2,
                "Kp": 2.0 if i < 12 else 5.0,
            })
    frame = pd.DataFrame(rows)
    parts = [frame[frame.year == y].copy() for y in [2001, 2005, 2009]]
    out = select_activity_samples(parts, n_per_group=12, seed=1)

    assert (out.activity_group == "Kp<=3").sum() == 12
    assert (out.activity_group == "Kp>=4").sum() == 12
    assert set(out.year) == {2001, 2005, 2009}

    # With 12 samples per activity group and three years, selection should be
    # stratified 4/year/group and must not count multiple hours from the same
    # nominal 3-h Kp interval as independent activity samples.
    for (group, year), part in out.groupby(["activity_group", "year"]):
        assert len(part) == 4, (group, year, len(part))
        kp_bins = pd.to_datetime(part["utc"]).dt.floor("3h")
        assert kp_bins.nunique() == len(part), (group, year, part[["utc", "Kp"]])


def test_newell_and_weighted_coupling_are_finite():
    ec = newell_coupling(np.array([2.0]), np.array([-3.0]), np.array([450.0]))
    assert np.isfinite(ec[0]) and ec[0] > 0
    df = pd.DataFrame({
        "By": [1, 2, 3, 4], "Bz": [-1, -2, -3, -4], "V": [400, 410, 420, 430]
    })
    w = weighted_newell_coupling(df, 3)
    assert np.isfinite(w) and w > 0

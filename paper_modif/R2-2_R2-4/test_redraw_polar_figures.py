import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("redraw_polar_figures.py")
spec = importlib.util.spec_from_file_location("redraw_polar_figures", MODULE_PATH)
redraw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(redraw)


def make_structured(times, field_name, images):
    dtype = [("utc", "datetime64[s]"), (field_name, object)]
    arr = np.empty(len(times), dtype=dtype)
    arr["utc"] = np.asarray(times, dtype="datetime64[s]")
    arr[field_name] = list(images)
    return arr


def test_find_nearest_time_index_returns_expected_row_and_delta():
    times = np.asarray([
        "1996-04-01T08:35:00",
        "1996-04-01T08:40:00",
        "1996-04-01T08:45:00",
    ], dtype="datetime64[s]")
    idx, matched, delta_s = redraw.find_nearest_time_index(
        times, pd.Timestamp("1996-04-01T08:40:30").to_pydatetime()
    )
    assert idx == 1
    assert matched == pd.Timestamp("1996-04-01T08:40:00").to_pydatetime()
    assert delta_s == pytest.approx(30.0)


def test_extract_structured_image_uses_named_field():
    images = [np.full((80, 96), 1.0, dtype=np.float32), np.full((80, 96), 2.0, dtype=np.float32)]
    arr = make_structured(["1996-04-01T08:35:00", "1996-04-01T08:40:00"], "image", images)
    out = redraw.extract_structured_image(arr, 1, "image")
    assert out.shape == (80, 96)
    np.testing.assert_allclose(out, 2.0)


def test_observation_plot_array_masks_same_missing_pixels_as_legacy_repair():
    flux = np.asarray([[0.0, 0.5, 0.999, 1.0, 2.0, np.nan]], dtype=np.float32)
    plotted = redraw.prepare_observation_for_plot(flux, support_threshold=1.0)
    assert np.isnan(plotted[0, 0])
    assert np.isnan(plotted[0, 1])
    assert np.isnan(plotted[0, 2])
    assert plotted[0, 3] == pytest.approx(1.0)
    assert plotted[0, 4] == pytest.approx(2.0)
    assert np.isnan(plotted[0, 5])


def test_validate_panel_shapes_rejects_mismatch():
    obs = np.zeros((80, 96), dtype=np.float32)
    recon = np.zeros((80, 96), dtype=np.float32)
    ovation = np.zeros((79, 96), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        redraw.validate_panel_shapes(obs, recon, ovation)


def test_match_event_builds_three_panels_from_saved_products():
    target = pd.Timestamp("1996-04-01T08:40:00").to_pydatetime()
    polar_images = [np.full((80, 96), 3.0, dtype=np.float32)]
    repaired_images = [np.full((80, 96), 4.0, dtype=np.float32)]
    polar = make_structured([target], "aurora_image", polar_images)
    repaired = make_structured([target], "image", repaired_images)
    omni = np.empty(1, dtype=[("utc", "datetime64[s]")])
    omni["utc"] = np.asarray([target], dtype="datetime64[s]")
    ovation = np.full((1, 80, 96), 5.0, dtype=np.float32)
    case = redraw.match_event(
        target=target,
        polar_data=polar,
        repaired_data=repaired,
        omni_data=omni,
        ovation_data=ovation,
        max_delta_seconds=1.0,
    )
    np.testing.assert_allclose(case["observation"], 3.0)
    np.testing.assert_allclose(case["reconstruction"], 4.0)
    np.testing.assert_allclose(case["ovation"], 5.0)
    assert case["polar_time"] == target
    assert case["repaired_time"] == target
    assert case["ovation_time"] == target

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name('collect_event_context.py')
spec = importlib.util.spec_from_file_location('collect_event_context', MODULE)
mod = importlib.util.module_from_spec(spec)
sys.modules['collect_event_context'] = mod
spec.loader.exec_module(mod)


def test_summarize_window_filters_invalid_values_and_reports_extrema():
    df = pd.DataFrame({
        'utc': pd.to_datetime(['2005-01-01T00:00:00','2005-01-01T00:05:00','2005-01-01T00:10:00']),
        'Bx': [1.0, 9999.0, 3.0], 'By': [2.0, 4.0, 6.0], 'Bz': [-1.0, -2.0, -3.0],
        'V': [400.0, 450.0, 500.0], 'P': [2.0, 3.0, 4.0],
        'AE': [100.0, 600.0, 300.0], 'SYM_H': [-5.0, -40.0, -10.0],
    })
    out = mod.summarize_omni_window(df, pd.Timestamp('2005-01-01T00:00:00'), pd.Timestamp('2005-01-01T00:10:00'))
    assert out['n_samples'] == 3
    assert np.isclose(out['Bx_mean'], 2.0)
    assert np.isclose(out['Bz_mean'], -2.0)
    assert np.isclose(out['AE_max'], 600.0)
    assert np.isclose(out['SYM_H_min'], -40.0)


def test_find_polar_crossing_selects_segment_containing_event():
    times = pd.date_range('1996-04-03T00:00:00', periods=8, freq='10min')
    mlat = np.array([40, 55, 62, 70, 68, 58, 45, 30], dtype=float)
    out = mod.find_polar_crossing(times, mlat, pd.Timestamp('1996-04-03T00:35:00'), threshold=60.0)
    assert out['crossing_start'] == pd.Timestamp('1996-04-03T00:20:00')
    assert out['crossing_end'] == pd.Timestamp('1996-04-03T00:40:00')
    assert out['contains_event'] is True


def test_build_latex_table_uses_na_for_missing_kp():
    rows = [{
        'event':'Figure 4','instrument':'Polar/UVI','interval':'1996-04-03 02:15 UT','orbit':'123',
        'Bx':1.2,'By':-2.3,'Bz':-4.5,'V':420.0,'P':2.2,'Kp':None,'AE_max':350.0,'SYM_H_min':-20.0,
    }]
    tex = mod.build_latex_table(rows)
    assert 'Figure 4' in tex
    assert 'Polar/UVI' in tex
    assert '--' in tex
    assert r'$B_z$' in tex

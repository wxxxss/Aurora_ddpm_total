import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).with_name("generate_omni_2005_5min.py")
spec = importlib.util.spec_from_file_location("generate_omni_2005_5min", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_clean_preserves_rows_and_outputs_model_schema():
    times = pd.date_range("2005-01-01", periods=5, freq="5min")
    df = pd.DataFrame({
        "utc": times,
        "Bx": [1.0, 999.0, 3.0, 4.0, 5.0],
        "By": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Bz": [1.0, 2.0, np.nan, 4.0, 5.0],
        "V": [400.0, 410.0, 420.0, 430.0, 440.0],
        "P": [2.0, 2.1, 2.2, 2.3, 2.4],
    })
    cleaned, audit = mod.clean_model_input_dataframe(df)
    assert len(cleaned) == 5
    assert list(cleaned.columns) == ["utc", "Bx", "By", "Bz", "V", "P"]
    assert np.isclose(cleaned.loc[1, "Bx"], 2.0)
    assert np.isclose(cleaned.loc[2, "Bz"], 3.0)
    assert not cleaned[["Bx", "By", "Bz", "V", "P"]].isna().any().any()
    assert audit["invalid_before_interpolation"]["Bx"] == 1
    assert audit["invalid_before_interpolation"]["Bz"] == 1


def test_structured_array_field_names_match_final_model_loader():
    times = pd.date_range("2005-01-01", periods=2, freq="5min")
    df = pd.DataFrame({
        "utc": times,
        "Bx": [1, 2],
        "By": [3, 4],
        "Bz": [5, 6],
        "V": [400, 410],
        "P": [2.0, 2.1],
    })
    arr = mod.to_structured_array(df)
    assert arr.dtype.names == ("utc", "Bx", "By", "Bz", "V", "P")
    assert np.issubdtype(arr["utc"].dtype, np.datetime64)
    for field in ("Bx", "By", "Bz", "V", "P"):
        assert arr[field].dtype == np.float32


if __name__ == "__main__":
    test_clean_preserves_rows_and_outputs_model_schema()
    test_structured_array_field_names_match_final_model_loader()
    print("tests passed")

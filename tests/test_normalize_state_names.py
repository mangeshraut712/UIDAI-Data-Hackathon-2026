"""normalize_state_names should drop missing values instead of emitting Nan."""
import numpy as np
import pandas as pd

from src.preprocessing.data_loader import normalize_state_names


def test_normalize_state_names_drops_nan_none_and_numeric_codes() -> None:
    original = pd.DataFrame({
        "state": ["Maharashtra", np.nan, None, "100000", "WESTBENGAL", "nan"],
        "count": [1, 2, 3, 4, 5, 6],
    })

    cleaned = normalize_state_names(original)

    assert list(cleaned["state"]) == ["Maharashtra", "West Bengal"]
    assert "Nan" not in set(cleaned["state"])
    # Caller is unchanged
    assert original["state"].isna().sum() == 2


def test_normalize_state_names_missing_column_raises() -> None:
    frame = pd.DataFrame({"region": ["MH"]})
    try:
        normalize_state_names(frame)
    except KeyError as exc:
        assert "state" in str(exc)
    else:
        raise AssertionError("expected KeyError")

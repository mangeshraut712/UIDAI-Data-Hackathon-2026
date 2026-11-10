"""fill_median should copy the frame and skip all-NA numeric columns."""
import pandas as pd

from src.preprocessing.data_loader import handle_missing_values


def test_fill_median_does_not_mutate_the_caller_and_fills_numeric_gaps() -> None:
    original = pd.DataFrame({"count": [1.0, None, 3.0], "state": ["MH", None, "KA"]})
    snapshot = original.copy()

    filled = handle_missing_values(original, strategy="fill_median")

    pd.testing.assert_frame_equal(original, snapshot)
    assert filled.loc[1, "count"] == 2.0
    assert pd.isna(filled.loc[1, "state"])


def test_fill_median_skips_all_na_numeric_columns() -> None:
    frame = pd.DataFrame({"count": [1.0, None], "empty": [None, None]})
    filled = handle_missing_values(frame, strategy="fill_median")
    assert filled.loc[1, "count"] == 1.0
    assert filled["empty"].isna().all()
    assert frame["count"].isna().sum() == 1

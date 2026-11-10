"""Tests for handle_missing_values(fill_mode)."""
import pandas as pd

from src.preprocessing.data_loader import handle_missing_values


def test_fill_mode_leaves_all_na_column_and_does_not_mutate_input():
    original = pd.DataFrame({"state": ["Maharashtra", None, "Maharashtra"], "note": [None, None, None]})
    before = original.copy()
    result = handle_missing_values(original, strategy="fill_mode")
    pd.testing.assert_frame_equal(original, before)
    assert result["state"].tolist() == ["Maharashtra", "Maharashtra", "Maharashtra"]
    assert result["note"].isna().all()

"""Tests for aggregate_by_time."""
import pandas as pd

from src.preprocessing.data_loader import aggregate_by_time


def test_aggregate_by_time_counts_rows_and_leaves_input_unchanged():
    original = pd.DataFrame({
        "event_date": ["2026-01-10", "2026-01-20", "2026-02-01"],
        "state": ["Maharashtra", "Maharashtra", "Kerala"],
    })
    before = original.copy()
    result = aggregate_by_time(original, "event_date", freq="M")
    pd.testing.assert_frame_equal(original, before)
    assert isinstance(result, pd.DataFrame)
    assert list(result["count"]) == [2, 1]
    assert str(result.index[0].month) == "1"
    assert str(result.index[1].month) == "2"

import pandas as pd

from src.analysis.iit_level_analytics import day_offsets


def test_day_offsets_count_calendar_days_across_gaps():
    dates = pd.Series(pd.to_datetime(["2025-03-02", "2025-03-03", "2025-03-10", "2025-04-01"]))
    assert list(day_offsets(dates, dates.min())) == [0, 1, 8, 30]


def test_future_offsets_continue_from_the_same_origin():
    history = pd.Series(pd.to_datetime(["2025-03-02", "2025-03-09"]))
    future = pd.Series(pd.date_range("2025-03-10", periods=3, freq="D"))
    assert list(day_offsets(future, history.min())) == [8, 9, 10]

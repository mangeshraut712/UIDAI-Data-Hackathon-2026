"""Tests for data_quality_report."""
import pandas as pd

from src.preprocessing.data_loader import data_quality_report


def test_data_quality_report_handles_an_empty_frame():
    report = data_quality_report(pd.DataFrame(), name="Empty")
    assert report["rows"] == 0
    assert report["columns"] == 0
    assert report["missing_percent"] == 0.0
    assert report["missing_cells"] == 0

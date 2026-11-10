"""Tests for UIDAI preprocessing helpers."""
import pandas as pd

from src.preprocessing.data_loader import create_age_groups


def test_age_65_is_in_65_plus_and_older_ages_are_kept():
    ages = pd.Series([0, 5, 17, 18, 55, 56, 64, 65, 100, 105])
    groups = create_age_groups(ages).astype(str)
    assert list(groups) == [
        "0-5",
        "0-5",
        "6-17",
        "18-25",
        "46-55",
        "56-65",
        "56-65",
        "65+",
        "65+",
        "65+",
    ]

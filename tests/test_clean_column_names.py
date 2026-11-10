"""clean_column_names should copy the frame and keep cleaned names unique."""
import pandas as pd

from src.preprocessing.data_loader import clean_column_names


def test_clean_column_names_does_not_mutate_the_caller() -> None:
    original = pd.DataFrame({"User Name": [1], "Age": [20]})
    snapshot = list(original.columns)

    cleaned = clean_column_names(original)

    assert list(original.columns) == snapshot
    assert list(cleaned.columns) == ["user_name", "age"]
    assert cleaned.loc[0, "user_name"] == 1


def test_clean_column_names_suffixes_colliding_slugs() -> None:
    original = pd.DataFrame({"User Name": [1], "user_name": [2], "User-Name": [3], "!!!": [4]})
    cleaned = clean_column_names(original)
    assert list(cleaned.columns) == ["user_name", "user_name_2", "user_name_3", "column"]
    assert list(original.columns) == ["User Name", "user_name", "User-Name", "!!!"]
    assert cleaned.iloc[0].tolist() == [1, 2, 3, 4]

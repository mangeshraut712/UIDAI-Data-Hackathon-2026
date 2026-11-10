"""load_data should accept mixed-case suffixes and pathlib paths."""
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.data_loader import load_data


def test_load_data_reads_uppercase_csv_suffix(tmp_path: Path) -> None:
    path = tmp_path / "enrolment.CSV"
    pd.DataFrame({"state": ["MH"], "count": [1]}).to_csv(path, index=False)

    loaded = load_data(path)

    assert list(loaded.columns) == ["state", "count"]
    assert loaded.iloc[0].to_dict() == {"state": "MH", "count": 1}


def test_load_data_reads_string_path_with_mixed_case_suffix(tmp_path: Path) -> None:
    path = tmp_path / "enrolment.Csv"
    pd.DataFrame({"state": ["KA"]}).to_csv(path, index=False)

    loaded = load_data(str(path))

    assert loaded.loc[0, "state"] == "KA"


def test_load_data_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "enrolment.txt"
    path.write_text("not a table", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(path)

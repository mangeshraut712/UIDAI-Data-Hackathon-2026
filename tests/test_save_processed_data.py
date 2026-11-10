"""save_processed_data should not pretend to succeed on an unknown format."""
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.data_loader import save_processed_data


def test_save_processed_data_writes_csv_regardless_of_format_case(tmp_path: Path) -> None:
    path = tmp_path / "enrolment.CSV"
    frame = pd.DataFrame({"state": ["MH"], "count": [1]})

    save_processed_data(frame, path, format="CSV")

    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["state", "count"]
    assert loaded.iloc[0].to_dict() == {"state": "MH", "count": 1}


def test_save_processed_data_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "enrolment.parquett"
    frame = pd.DataFrame({"state": ["KA"]})

    with pytest.raises(ValueError, match="Unsupported format"):
        save_processed_data(frame, path, format="parquett")

    assert not path.exists()

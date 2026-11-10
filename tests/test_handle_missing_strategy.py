from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_loader import handle_missing_values


def test_unknown_strategy_raises() -> None:
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    with pytest.raises(ValueError, match="Unknown missing-value strategy"):
        handle_missing_values(df, strategy="fill_mean")
    # documented strategies still work and do not mutate the caller
    out = handle_missing_values(df, strategy="fill_zero")
    assert out["a"].isna().sum() == 0
    assert df["a"].isna().sum() == 1

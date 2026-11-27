"""normalize_state_names should drop missing values instead of emitting Nan."""
import numpy as np
import pandas as pd

from src.preprocessing.data_loader import normalize_state_names


def test_normalize_state_names_drops_nan_none_and_numeric_codes() -> None:
    original = pd.DataFrame({
        "state": ["Maharashtra", np.nan, None, "100000", "WESTBENGAL", "nan"],
        "count": [1, 2, 3, 4, 5, 6],
    })

    cleaned = normalize_state_names(original)

    assert list(cleaned["state"]) == ["Maharashtra", "West Bengal"]
    assert "Nan" not in set(cleaned["state"])
    # Caller is unchanged
    assert original["state"].isna().sum() == 2


def test_normalize_state_names_missing_column_raises() -> None:
    frame = pd.DataFrame({"region": ["MH"]})
    try:
        normalize_state_names(frame)
    except KeyError as exc:
        assert "state" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_normalize_state_names_alias_is_case_insensitive() -> None:
    frame = pd.DataFrame({
        "state": ["Jammu & Kashmir", "jammu & kashmir", "Jammu and Kashmir", "Orissa", "Pondicherry"],
    })

    cleaned = normalize_state_names(frame)

    assert list(cleaned["state"]) == [
        "Jammu And Kashmir",
        "Jammu And Kashmir",
        "Jammu And Kashmir",
        "Odisha",
        "Puducherry",
    ]


def test_state_mapping_has_no_case_insensitive_conflicts() -> None:
    """Two aliases that differ only by case must resolve to the same canonical name.

    The lookup folds keys to upper case, so a conflict would silently keep only
    whichever entry is written last.
    """
    from src.preprocessing.data_loader import STATE_MAPPING

    folded: dict[str, set[str]] = {}
    for alias, canonical in STATE_MAPPING.items():
        folded.setdefault(alias.upper(), set()).add(canonical)

    conflicts = {key: values for key, values in folded.items() if len(values) > 1}
    assert not conflicts, f"aliases collide after upper-casing: {conflicts}"


def test_upper_state_mapping_is_built_once_and_matches_state_mapping() -> None:
    from src.preprocessing.data_loader import STATE_MAPPING, UPPER_STATE_MAPPING

    assert UPPER_STATE_MAPPING == {k.upper(): v for k, v in STATE_MAPPING.items()}
    # A second call must reuse the same object rather than rebuilding it.
    before = id(UPPER_STATE_MAPPING)
    normalize_state_names(pd.DataFrame({"state": ["karnataka"]}))
    from src.preprocessing.data_loader import UPPER_STATE_MAPPING as again

    assert id(again) == before


def test_normalize_state_names_is_case_insensitive_for_unlisted_casings() -> None:
    frame = pd.DataFrame({"state": ["KaRnAtAkA", "jAmMu & kAsHmIr", "oRiSsA"]})
    cleaned = normalize_state_names(frame)
    assert list(cleaned["state"]) == ["Karnataka", "Jammu And Kashmir", "Odisha"]

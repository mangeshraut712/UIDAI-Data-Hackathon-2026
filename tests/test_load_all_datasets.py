"""Tests for UIDAIAnalyzer.load_all_datasets date parsing and dataset wiring."""

import pandas as pd

from src.analysis.comprehensive_analysis import UIDAIAnalyzer


def _write_csv(path, header, rows):
    path.write_text("\n".join([header, *rows]) + "\n")


def test_load_all_datasets_parses_day_first_dates(tmp_path):
    """Raw UIDAI exports use DD-MM-YYYY; loading must parse explicitly day-first.

    '07-01-2026' is ambiguous (7 Jan vs 1 Jul) and must resolve to 2026-01-07;
    '13-01-2026' is unambiguously day-first and must parse without error.
    """
    raw = tmp_path / "data" / "raw"
    dirs = {
        "enrolment": raw / "api_data_aadhar_enrolment",
        "demographic": raw / "api_data_aadhar_demographic",
        "biometric": raw / "api_data_aadhar_biometric",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    _write_csv(
        dirs["enrolment"] / "chunk.csv",
        "date,state,district,pincode,age_0_5,age_5_17,age_18_greater",
        [
            "07-01-2026,Maharashtra,Pune,411001,1,2,3",
            "13-01-2026,Maharashtra,Pune,411001,4,5,6",
        ],
    )
    _write_csv(
        dirs["demographic"] / "chunk.csv",
        "date,state,district,pincode,demo_age_5_17,demo_age_17_",
        ["07-01-2026,Maharashtra,Pune,411001,7,8"],
    )
    _write_csv(
        dirs["biometric"] / "chunk.csv",
        "date,state,district,pincode,bio_age_5_17,bio_age_17_",
        ["07-01-2026,Maharashtra,Pune,411001,9,10"],
    )

    data = UIDAIAnalyzer(base_dir=tmp_path).load_all_datasets()

    assert set(data) == {"enrolment", "demographic", "biometric"}
    assert list(data["enrolment"]["date"]) == [
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-13"),
    ]
    assert data["demographic"]["date"].iloc[0] == pd.Timestamp("2026-01-07")
    assert data["biometric"]["date"].iloc[0] == pd.Timestamp("2026-01-07")

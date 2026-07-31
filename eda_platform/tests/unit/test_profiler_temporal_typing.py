"""Values beat naming when typing a column as temporal.

2026-07-22 audit: `TrainingTimesLastYear` (counts 0-6) was typed `datetime`
because its name contains the substring "time". That one mis-type produced a
fake trend question, a bogus time-coverage metric, and a line chart over a
non-time axis — all three downstream layers behaved correctly on bad input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eda_platform.schemas.artifacts import DatasetProfile
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


def _semantic_types(tmp_path: Path, frame: pd.DataFrame) -> dict[str, str]:
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    artifact = profile_dataset(load_csv(path, dataset_id="ds_x"), project_id="p", session_id="r")
    profile = DatasetProfile.model_validate(artifact.payload)
    return {column.name: column.semantic_type for column in profile.columns_detail}


@pytest.mark.parametrize(
    ("column", "values", "expected"),
    [
        # Counts and durations whose names merely contain a temporal word.
        ("TrainingTimesLastYear", [0, 1, 2, 3, 4, 5, 6] * 20, "numeric"),
        ("response_time_ms", [12, 45, 300, 150, 88] * 28, "numeric"),
        ("Time", [float(index) for index in range(140)], "numeric"),
        ("birth_year", [1980, 1990, 2000, 1975, 1966] * 28, "numeric"),
        # Real temporal encodings still type as datetime.
        ("event_ts", [1_751_000_000 + index for index in range(140)], "datetime"),
    ],
)
def test_numeric_columns_are_typed_by_their_values(
    tmp_path: Path, column: str, values: list, expected: str
) -> None:
    frame = pd.DataFrame({column: values, "amount": [1.5] * len(values)})
    assert _semantic_types(tmp_path, frame)[column] == expected


def test_genuine_date_strings_still_type_as_datetime(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "signup_date": pd.date_range("2026-01-01", periods=140, freq="D").astype(str),
            "order_purchase_timestamp": pd.date_range("2026-01-01", periods=140, freq="h").astype(
                str
            ),
            "amount": [1.5] * 140,
        }
    )
    types = _semantic_types(tmp_path, frame)
    assert types["signup_date"] == "datetime"
    assert types["order_purchase_timestamp"] == "datetime"

"""A 0/1 flag stored as an integer still groups.

2026-08-04 FIFA run 2: both tool-guard rejections were `run_stat_test` refusing
`is_knockout` and `is_starting_xi` as group columns because the guard re-derived
their type from the pandas dtype (Int64 -> numeric). The profiler had already
classified both as boolean, and the role set had called them dimensions; the
guard asked neither.
"""

from __future__ import annotations

import pandas as pd

from eda_platform.core.tool_guard import check_column_semantic_type, is_boolean_like


def _violation(frame: pd.DataFrame, column: str):
    return check_column_semantic_type(
        "group_column",
        column,
        frame,
        allowed_semantic_types=["categorical"],
    )


def test_an_integer_flag_is_accepted_as_a_grouping_key() -> None:
    frame = pd.DataFrame(
        {
            "is_knockout": pd.array([0, 1, 1, 0, 1], dtype="Int64"),
            "goals": [1, 2, 0, 3, 2],
        }
    )
    assert _violation(frame, "is_knockout") is None


def test_a_real_measure_is_still_refused() -> None:
    """Widening for flags must not open the door to every numeric column."""
    frame = pd.DataFrame({"elo_rating": [1775, 2100, 1856, 1937, 1510]})
    violation = _violation(frame, "elo_rating")
    assert violation is not None
    assert "numeric" in violation.problem


def test_a_two_valued_measure_is_not_a_flag() -> None:
    """Few distinct values is not the test; being 0/1 is."""
    frame = pd.DataFrame({"squad_size": [23, 26, 23, 26, 23]})
    assert _violation(frame, "squad_size") is not None


def test_the_predicate_matches_the_profilers_vocabulary() -> None:
    assert is_boolean_like(pd.Series([0, 1, 1, 0]))
    assert is_boolean_like(pd.Series(["Yes", "No", "Yes"]))
    assert is_boolean_like(pd.Series([True, False]))
    assert not is_boolean_like(pd.Series([0, 1, 2]))
    assert not is_boolean_like(pd.Series([], dtype="float64"))

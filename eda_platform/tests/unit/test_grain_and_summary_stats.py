"""Composite key detection, grain statement, and tail percentiles."""

from pathlib import Path

import pandas as pd

from eda_platform.schemas.artifacts import AnalysisTable, DatasetProfile
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


def _profile(path: Path) -> tuple[DatasetProfile, object]:
    loaded = load_csv(path)
    artifact = profile_dataset(loaded, project_id="p", session_id="s")
    return DatasetProfile.model_validate(artifact.payload), loaded


def test_composite_key_is_found_when_no_single_column_is_unique(tmp_path: Path) -> None:
    # A fact table keyed by (customer, day): neither column alone identifies a
    # row, so single-column detection reports no key at all.
    path = tmp_path / "visits.csv"
    rows = ["customer_id,visit_date,minutes"]
    for customer in range(20):
        for day in range(1, 6):
            rows.append(f"C{customer:03d},2026-03-0{day},{customer * day}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, _ = _profile(path)

    assert profile.primary_key_candidates == []
    assert ["customer_id", "visit_date"] in profile.composite_key_candidates
    assert profile.grain is not None
    assert "customer_id" in profile.grain and "visit_date" in profile.grain


def test_single_column_key_is_reported_as_the_grain(tmp_path: Path) -> None:
    path = tmp_path / "customers.csv"
    rows = ["customer_id,city"]
    for index in range(30):
        rows.append(f"C{index:03d},city_{index % 4}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, _ = _profile(path)

    assert profile.primary_key_candidates == ["customer_id"]
    assert profile.grain is not None and "customer_id" in profile.grain


def test_grain_says_so_when_no_key_identifies_a_row(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    rows = ["kind,amount"]
    for index in range(40):
        rows.append(f"kind_{index % 3},{index % 7}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    profile, _ = _profile(path)

    assert profile.composite_key_candidates == []
    assert profile.grain is not None
    assert "no" in profile.grain.lower() or "not" in profile.grain.lower()


def test_numeric_summary_carries_tail_percentiles(tmp_path: Path) -> None:
    # p95/p99 are what tell you a column is capped or long-tailed; quartiles
    # alone hide both.
    path = tmp_path / "spend.csv"
    values = [float(index) for index in range(1000)]
    pd.DataFrame({"spend": values, "other": values}).to_csv(path, index=False)
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")

    tables = create_analysis_tables(
        loaded, profile_artifact, project_id="p", session_id="s"
    )
    summary = next(
        AnalysisTable.model_validate(artifact.payload)
        for artifact in tables
        if artifact.payload["kind"] == "numeric_summary"
    )
    spend = next(row for row in summary.rows if row["column"] == "spend")

    assert spend["p1"] < spend["q1"] < spend["median"] < spend["q3"] < spend["p99"]
    assert spend["p95"] == pytest_approx(949.05)
    assert "kurtosis" in spend


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, rel=1e-3)

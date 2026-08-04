from pathlib import Path

import numpy as np
import pandas as pd

from eda_platform.schemas.artifacts import AnalysisTable, ArtifactType
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


def test_analysis_tables_include_summary_and_correlation_stats(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "region,revenue,profit,cost\n"
        "East,10,1,9\n"
        "East,20,2,18\n"
        "West,30,3,27\n"
        "West,40,4,36\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )
    tables = [AnalysisTable.model_validate(artifact.payload) for artifact in artifacts]
    tables_by_kind = {table.kind: table for table in tables}

    assert all(artifact.type is ArtifactType.TABLE for artifact in artifacts)
    assert {"numeric_summary", "correlation"}.issubset(tables_by_kind)
    assert "grouped_summary" not in tables_by_kind

    revenue_summary = next(
        row for row in tables_by_kind["numeric_summary"].rows if row["column"] == "revenue"
    )
    assert revenue_summary["dataset"] == "sales.csv"
    assert revenue_summary["mean"] == 25.0
    assert revenue_summary["min"] == 10.0
    assert revenue_summary["max"] == 40.0

    revenue_profit = next(
        row
        for row in tables_by_kind["correlation"].rows
        if {row["column_a"], row["column_b"]} == {"revenue", "profit"}
    )
    assert revenue_profit["dataset"] == "sales.csv"
    assert revenue_profit["pearson"] == 1.0


def test_ultra_wide_correlation_uses_bounded_screening_and_exact_values(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(600, 129))
    values[:, 128] = values[:, 0] * 3.0 + rng.normal(scale=0.001, size=600)
    csv_path = tmp_path / "wide.csv"
    pd.DataFrame(values, columns=[f"metric_{index}" for index in range(129)]).to_csv(
        csv_path,
        index=False,
    )
    loaded = load_csv(csv_path, dataset_id="ds_wide")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )
    correlation = next(
        AnalysisTable.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.payload["kind"] == "correlation"
    )

    strongest = correlation.rows[0]
    assert {strongest["column_a"], strongest["column_b"]} == {"metric_0", "metric_128"}
    assert strongest["pearson"] == 1.0
    assert strongest["pairwise_complete_n"] == 600
    assert strongest["selection_is_approximate"] is True
    assert strongest["selection_method"] == "sparse_random_projection_then_exact"


def _correlation_rows(tmp_path: Path, name: str, frame: pd.DataFrame) -> list[dict]:
    csv_path = tmp_path / name
    frame.to_csv(csv_path, index=False)
    loaded = load_csv(csv_path, dataset_id=f"ds_{csv_path.stem}")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    for artifact in create_analysis_tables(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    ):
        if artifact.payload.get("kind") == "correlation":
            return artifact.payload["rows"]
    return []


def _flag(rows: list[dict], left: str, right: str) -> bool:
    for row in rows:
        if {row["column_a"], row["column_b"]} == {left, right}:
            return bool(row["is_trivial_pair"])
    raise AssertionError(f"no correlation row for {left} ~ {right}")


def _pearson(rows: list[dict], left: str, right: str) -> float:
    for row in rows:
        if {row["column_a"], row["column_b"]} == {left, right}:
            return float(row["pearson"])
    raise AssertionError(f"no correlation row for {left} ~ {right}")


def test_a_part_of_a_whole_is_not_a_correlation_finding(tmp_path: Path) -> None:
    # The 2026-08-04 World Cup report led with "total_shots and shots_on_target
    # show a strong positive association (r=0.802)". Shots on target are a
    # subset of shots, so the number is arithmetic; at r<0.97 it slipped past
    # the rescale/complement rules entirely.
    total = np.array([(index * 5) % 17 + 4 for index in range(60)])
    on_target = (total * 0.42).astype(int)
    # Correlated with shots AND always larger, so only the shared subject
    # separates it from the nested pair.
    possession = 45.0 + total * 0.9 + np.array([(index % 5) - 2 for index in range(60)])
    frame = pd.DataFrame(
        {
            "total_shots": total,
            "shots_on_target": on_target,
            "possession_pct": possession,
        }
    )

    rows = _correlation_rows(tmp_path, "team_stats.csv", frame)

    # Guard the fixture: both pairs must clear the correlation floor, or the
    # assertions below would pass without reaching the rule at all.
    assert _pearson(rows, "total_shots", "shots_on_target") > 0.9
    assert _pearson(rows, "possession_pct", "total_shots") > 0.9
    assert frame["total_shots"].le(frame["possession_pct"]).all()

    assert _flag(rows, "total_shots", "shots_on_target") is True
    # A different quantity that merely happens to be larger is a real finding:
    # "more possession, more shots" must survive.
    assert _flag(rows, "possession_pct", "total_shots") is False


def test_sibling_slices_of_one_measure_stay_real_correlations(tmp_path: Path) -> None:
    # striker_shots and midfield_shots share a subject but neither contains the
    # other; a name-only rule would silence a genuine relationship. Home/away
    # pairs cannot test this -- _complement_names already handles those.
    base = np.array([(index * 7) % 23 + 6 for index in range(60)])
    striker = base + np.array([(index % 7) - 3 for index in range(60)])
    midfield = base + np.array([3 - (index % 7) for index in range(60)])
    frame = pd.DataFrame({"striker_shots": striker, "midfield_shots": midfield})

    rows = _correlation_rows(tmp_path, "shots_by_line.csv", frame)

    # Below the 0.97 gate the older rescale/complement rules never run, so the
    # component rule is the only thing that could reject this pair.
    assert 0.5 < _pearson(rows, "striker_shots", "midfield_shots") < 0.97
    assert bool((striker > midfield).any()) and bool((midfield > striker).any())

    assert _flag(rows, "striker_shots", "midfield_shots") is False

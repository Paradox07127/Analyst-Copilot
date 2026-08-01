"""Association between categorical fields, and between categories and measures."""

from pathlib import Path

import pandas as pd

from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.tools import analysis
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset


def _association_table(tmp_path: Path, rows: list[str]) -> AnalysisTable | None:
    path = tmp_path / "assoc.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path)
    profile_artifact = profile_dataset(loaded, project_id="p", session_id="s")
    tables = create_analysis_tables(
        loaded, profile_artifact, project_id="p", session_id="s"
    )
    for artifact in tables:
        if artifact.payload["kind"] == "association":
            return AnalysisTable.model_validate(artifact.payload)
    return None


def test_categorical_pair_association_is_measured(tmp_path: Path) -> None:
    # region determines country exactly; plan is independent of both. A purely
    # numeric correlation layer reports nothing at all for this table.
    rows = ["region,country,plan"]
    regions = ["emea", "amer", "apac"]
    countries = {"emea": "de", "amer": "us", "apac": "jp"}
    for index in range(180):
        region = regions[index % 3]
        rows.append(f"{region},{countries[region]},{'pro' if index % 2 == 0 else 'free'}")

    table = _association_table(tmp_path, rows)

    assert table is not None
    pairs = {
        frozenset((row["column_a"], row["column_b"])): row for row in table.rows
    }
    perfect = pairs[frozenset(("region", "country"))]
    assert perfect["method"] == "cramers_v"
    assert perfect["association"] > 0.99
    independent = pairs[frozenset(("region", "plan"))]
    assert independent["association"] < 0.3


def test_category_to_measure_association_uses_correlation_ratio(
    tmp_path: Path,
) -> None:
    # Salary is fully explained by band; tenure is not.
    rows = ["band,salary,tenure"]
    for index in range(150):
        band = ["junior", "mid", "senior"][index % 3]
        salary = {"junior": 50.0, "mid": 80.0, "senior": 120.0}[band]
        rows.append(f"{band},{salary + (index % 2) * 0.5},{float(index % 11)}")

    table = _association_table(tmp_path, rows)

    assert table is not None
    salary_row = next(
        row
        for row in table.rows
        if {row["column_a"], row["column_b"]} == {"band", "salary"}
    )
    assert salary_row["method"] == "correlation_ratio"
    assert salary_row["association"] > 0.95
    assert salary_row["pairwise_complete_n"] == 150


def test_association_reports_nothing_when_only_one_category_exists(
    tmp_path: Path,
) -> None:
    rows = ["only_group,amount"]
    for index in range(40):
        rows.append(f"g,{float(index)}")

    assert _association_table(tmp_path, rows) is None


def test_wide_large_association_is_column_and_row_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(analysis, "_MAX_ASSOCIATION_SAMPLE_ROWS", 50)
    row_count = 100
    frame = pd.DataFrame(
        {
            **{
                f"cat_{column:02d}": [f"g{(row + column) % 3}" for row in range(row_count)]
                for column in range(30)
            },
            **{
                f"measure_{column:02d}": [float(row + column) for row in range(row_count)]
                for column in range(30)
            },
        }
    )
    source = tmp_path / "wide.csv"
    source.write_text("placeholder", encoding="utf-8")
    loaded = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_wide",
            name="wide.csv",
            path=source,
            content_hash="hash",
        ),
        frame=frame,
    )
    profile = profile_dataset(loaded, project_id="p", session_id="s")

    artifacts = create_analysis_tables(
        loaded, profile, project_id="p", session_id="s"
    )
    table = next(
        AnalysisTable.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.payload["kind"] == "association"
    )

    assert "Evaluated 24 of 30 eligible categorical fields" in table.description
    assert "24 of 30 numeric fields" in table.description
    assert all(row["selection_is_approximate"] is True for row in table.rows)
    assert all(row["selection_method"] == "deterministic_row_sample" for row in table.rows)
    assert all(row["analysis_population_rows"] == row_count for row in table.rows)
    assert all(row["pairwise_complete_n"] <= 50 for row in table.rows)

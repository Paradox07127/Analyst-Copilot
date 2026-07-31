"""Association between categorical fields, and between categories and measures."""

from pathlib import Path

from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.loader import load_csv
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

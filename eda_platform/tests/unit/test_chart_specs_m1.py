from pathlib import Path

from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.charts import ChartSpec
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


def test_chart_specs_include_numeric_categorical_and_datetime(tmp_path: Path) -> None:
    # Enough distinct amounts that the column is genuinely continuous: a handful
    # of repeated values is a category set and is charted as one bar per value.
    csv_path = tmp_path / "sales.csv"
    rows = ["order_date,amount,region,notes"]
    for index in range(40):
        amount = "" if index == 2 else f"{10 + index * 3.5:.2f}"
        note = "" if index == 0 else "ok"
        rows.append(
            f"2026-01-{(index % 28) + 1:02d},{amount},"
            f"{'East' if index % 2 == 0 else 'West'},{note}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )
    specs = [ChartSpec.model_validate(artifact.payload) for artifact in artifacts]
    titles = {spec.title for spec in specs}

    assert all(artifact.type is ArtifactType.CHART_SPEC for artifact in artifacts)
    assert "Missing values by column" in titles
    assert "Missingness association" in titles
    assert "Distribution of amount" in titles
    assert "Top values in region" in titles
    assert "Records over order_date" in titles
    assert all(spec.data.get("values") for spec in specs)


def test_chart_spec_to_vegalite_includes_description_and_values() -> None:
    spec = ChartSpec(
        dataset_id="ds_sales",
        title="Missing values by column",
        description="Columns with missing values.",
        mark="bar",
        data={"values": [{"column": "amount", "missing": 1}]},
        encoding={
            "x": {"field": "column", "type": "nominal"},
            "y": {"field": "missing", "type": "quantitative"},
        },
    )

    vegalite = spec.to_vegalite()

    assert vegalite["description"] == "Columns with missing values."
    assert vegalite["data"]["values"][0]["missing"] == 1


def test_chart_specs_build_histogram_from_full_numeric_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "players.csv"
    rows = ["age,attendance"]
    for index in range(40):
        rows.append(f'{20 + index}-{index:03d},"{40_000 + index * 137:,}"')
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_players")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )
    specs = [ChartSpec.model_validate(artifact.payload) for artifact in artifacts]
    age_distribution = next(spec for spec in specs if spec.title == "Distribution of age")

    # Chart data is a full-column histogram (bin/count), not a 5-row sample.
    rows = age_distribution.data["values"]
    assert rows
    assert set(rows[0].keys()) == {"bin_start", "bin_end", "bin_label", "count"}
    assert sum(row["count"] for row in rows) == 40

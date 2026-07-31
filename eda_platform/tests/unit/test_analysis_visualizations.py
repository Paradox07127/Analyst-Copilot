from pathlib import Path

import pandas as pd
import pytest

from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import create_correlation_chart_specs
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.stat_tests import (
    create_anova_boxplot_artifact,
    run_stat_test,
)


def _loaded(frame: pd.DataFrame) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_visual",
            name="visual.csv",
            path=Path("visual.csv"),
            content_hash="test",
        ),
        frame=frame,
    )


def _correlation_artifact(rows: list[dict[str, object]]) -> Artifact:
    table = AnalysisTable(
        dataset_id="ds_visual",
        title="Visual correlations",
        kind="correlation",
        description="Test correlations.",
        rows=rows,
    )
    return Artifact(
        id="table_corr",
        type=ArtifactType.TABLE,
        project_id="project",
        session_id="run",
        payload=table.model_dump(mode="json"),
    )


def test_correlation_charts_exclude_trivial_pairs_and_enforce_caps() -> None:
    columns = [f"value_{index:02d}" for index in range(20)]
    frame = pd.DataFrame(
        {column: [float(row + index) for row in range(250)] for index, column in enumerate(columns)}
    )
    rows: list[dict[str, object]] = [
        {
            "column_a": "value_00",
            "column_b": "value_01",
            "pearson": 0.999,
            "is_trivial_pair": True,
        }
    ]
    rows.extend(
        {
            "column_a": columns[index],
            "column_b": columns[index + 1],
            "pearson": 0.95 - index / 100,
            "is_trivial_pair": False,
        }
        for index in range(1, 19)
    )

    artifacts = create_correlation_chart_specs(
        _loaded(frame),
        _correlation_artifact(rows),
        project_id="project",
        session_id="run",
    )
    specs = [ChartSpec.model_validate(artifact.payload) for artifact in artifacts]
    heatmap = next(spec for spec in specs if spec.mark == "rect")
    heatmap_values = heatmap.data["values"]
    heatmap_columns = {row["column_a"] for row in heatmap_values} | {
        row["column_b"] for row in heatmap_values
    }
    scatter_specs = [spec for spec in specs if spec.mark == "point"]

    assert heatmap.to_vegalite()["encoding"]["color"]["scale"] == {
        "domain": [-1, 1],
        "scheme": "redblue",
    }
    assert len(heatmap_columns) <= 15
    assert not any(
        {row["column_a"], row["column_b"]} == {"value_00", "value_01"}
        for row in heatmap_values
    )
    assert len(scatter_specs) == 3
    assert all(len(spec.data["values"]) <= 200 for spec in scatter_specs)
    assert all(spec.to_vegalite()["mark"] == "point" for spec in scatter_specs)
    assert all("not causation" in spec.description for spec in scatter_specs)
    assert all(artifact.code_ref and artifact.plain_language for artifact in artifacts)


def test_correlation_rows_include_dataset_sample_size() -> None:
    loaded = _loaded(pd.DataFrame({"left": [1, 2, 3], "right": [2, 4, 6]}))
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    artifacts = create_analysis_tables(
        loaded,
        profile,
        project_id="project",
        session_id="run",
    )
    correlation = next(
        artifact for artifact in artifacts if artifact.payload["kind"] == "correlation"
    )
    assert all(row["sample_size"] == 3 for row in correlation.payload["rows"])


@pytest.mark.parametrize(
    ("test_type", "frame", "kwargs"),
    [
        (
            "independent_t_test",
            pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": list(range(10))}),
            {"group_column": "group", "value_column": "value"},
        ),
        (
            "paired_t_test",
            pd.DataFrame(
                {
                    "group": ["a"] * 5 + ["b"] * 5,
                    "value": list(range(5)) + [2, 2, 5, 5, 8],
                    "pair": list(range(5)) * 2,
                }
            ),
            {
                "group_column": "group",
                "value_column": "value",
                "pair_column": "pair",
            },
        ),
        (
            "chi_square_independence",
            pd.DataFrame(
                {
                    "group": ["a"] * 20 + ["b"] * 20,
                    "category": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
                }
            ),
            {"group_column": "group", "category_column": "category"},
        ),
        (
            "one_way_anova",
            pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5 + ["c"] * 5, "value": list(range(15))}),
            {"group_column": "group", "value_column": "value"},
        ),
        (
            "mann_whitney_u",
            pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": list(range(10))}),
            {"group_column": "group", "value_column": "value"},
        ),
        (
            "kruskal_wallis",
            pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5 + ["c"] * 5, "value": list(range(15))}),
            {"group_column": "group", "value_column": "value"},
        ),
    ],
)
def test_every_stat_test_populates_effect_size(
    test_type: str,
    frame: pd.DataFrame,
    kwargs: dict[str, str],
) -> None:
    result = run_stat_test(
        frame,
        dataset_id="ds_stats",
        test_type=test_type,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    assert result.effect_size is not None


def test_tiny_p_value_is_not_rounded_to_zero() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a"] * 100 + ["b"] * 100,
            "value": [float(index) / 100 for index in range(100)]
            + [2.0 + float(index) / 100 for index in range(100)],
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="ds_stats",
        test_type="independent_t_test",
        group_column="group",
        value_column="value",
    )
    assert result.p_value is not None
    assert 0.0 < result.p_value < 1e-50


def test_anova_boxplot_is_valid_and_caps_groups_and_samples() -> None:
    frame = pd.DataFrame(
        [
            {"group": f"group_{group:02d}", "value": float(group + value / 100)}
            for group in range(25)
            for value in range(60)
        ]
    )
    result = run_stat_test(
        frame,
        dataset_id="ds_stats",
        test_type="one_way_anova",
        group_column="group",
        value_column="value",
    )
    artifact = create_anova_boxplot_artifact(
        frame,
        result,
        project_id="project",
        session_id="run",
    )

    assert artifact is not None
    spec = ChartSpec.model_validate(artifact.payload)
    values = spec.data["values"]
    group_counts = pd.Series([row["group"] for row in values]).value_counts()
    assert spec.to_vegalite()["mark"] == "boxplot"
    assert len(group_counts) == 20
    assert group_counts.max() == 50
    assert len(values) == 1_000
    assert "20 of 25 groups" in (artifact.plain_language or "")
    assert "sample sizes" in (artifact.plain_language or "")

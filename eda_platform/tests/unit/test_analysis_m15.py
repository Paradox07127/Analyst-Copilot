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

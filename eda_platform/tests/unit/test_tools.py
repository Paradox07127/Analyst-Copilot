from pathlib import Path

from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.exporter import export_markdown_report
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality


def _write_orders_csv(path: Path) -> None:
    path.write_text(
        "order_id,amount,region\n"
        "1,10,East\n"
        "2,,West\n"
        "3,,East\n",
        encoding="utf-8",
    )


def test_loader_profiler_quality_chart_and_report(tmp_path) -> None:
    csv_path = tmp_path / "orders.csv"
    _write_orders_csv(csv_path)

    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile_artifact = profile_dataset(
        loaded,
        project_id="project_demo",
        session_id="run_demo",
    )
    quality_artifact = scan_quality(
        profile_artifact,
        project_id="project_demo",
        session_id="run_demo",
    )
    chart_artifacts = create_chart_specs(
        loaded,
        profile_artifact,
        project_id="project_demo",
        session_id="run_demo",
    )
    report_artifact = export_markdown_report(
        [profile_artifact, quality_artifact, *chart_artifacts],
        project_id="project_demo",
        session_id="run_demo",
    )

    profile = profile_artifact.payload
    quality = quality_artifact.payload

    assert loaded.record.name == "orders.csv"
    assert profile_artifact.type is ArtifactType.DATASET_PROFILE
    assert profile["rows"] == 3
    assert profile["missing_values"]["amount"] == 2
    assert quality["issues"][0]["code"] == "high_missing"
    assert chart_artifacts[0].payload["mark"] == "bar"
    assert "prof_" in report_artifact.payload["markdown"]
    assert "quality_" in report_artifact.payload["markdown"]


def test_duckdb_query_engine_registers_csv_and_selects(tmp_path) -> None:
    csv_path = tmp_path / "orders.csv"
    _write_orders_csv(csv_path)
    engine = DuckDBQueryEngine()

    engine.register_csv("orders", csv_path)
    result = engine.execute_select("select count(*) as rows from orders")

    assert result.to_dict("records") == [{"rows": 3}]

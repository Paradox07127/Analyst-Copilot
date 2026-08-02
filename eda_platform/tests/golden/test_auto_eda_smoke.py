from inspect import signature
from pathlib import Path

from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.sessions import SessionManifest

GOLDEN_DATA = Path(__file__).parent / "data"


def test_auto_eda_does_not_expose_m2_enable_switch() -> None:
    assert "enable_llm_report" not in signature(run_auto_eda).parameters


def test_auto_eda_creates_manifest_artifacts_and_trace(tmp_path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "order_id,amount,region\n1,10,East\n2,,West\n3,,East\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    result = run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Regional order analysis",
    )

    artifact_types = {artifact.type for artifact in result.artifacts}
    session_dir = workspace / "projects/project_demo/sessions/run_demo"

    assert (session_dir / "manifest.json").exists()
    manifest = SessionManifest.model_validate_json(
        (session_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.code_version != "unknown"
    assert (session_dir / "trace.jsonl").exists()
    assert (workspace / "projects/project_demo/uploads").exists()
    assert ArtifactType.DATASET_PROFILE in artifact_types
    assert ArtifactType.QUALITY_ISSUE_SET in artifact_types
    assert ArtifactType.CHART_SPEC in artifact_types
    assert ArtifactType.TABLE in artifact_types
    assert ArtifactType.QUESTION_CANDIDATE_SET in artifact_types
    assert ArtifactType.REPORT_BUNDLE in artifact_types
    assert ArtifactType.REPORT_AUDIT in artifact_types
    assert ArtifactType.HTML_REPORT in artifact_types
    assert ArtifactType.MARKDOWN_REPORT in artifact_types
    assert "### Claim Ledger" in result.report_markdown
    assert "prof_" in result.report_markdown
    assert (session_dir / "report" / "report.html").exists()
    assert result.business_context == "Regional order analysis"
    assert len(result.loaded_datasets) == 1
    assert list(result.loaded_datasets[0].frame.columns) == ["order_id", "amount", "region"]


def test_auto_eda_emits_live_trace_callback(tmp_path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("order_id,amount\n1,10\n", encoding="utf-8")
    seen: list[str] = []

    run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="live_debug",
        on_trace_event=lambda event: seen.append(event.event_type),
    )

    assert "step_started" in seen
    assert "step_completed" in seen


def test_auto_eda_dirty_dataset_surfaces_quality_rules(tmp_path) -> None:
    result = run_auto_eda(
        [GOLDEN_DATA / "dirty_sales.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="dirty_run",
    )
    quality_artifacts = [
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.QUALITY_ISSUE_SET
    ]
    issue_codes = {
        issue["code"] for artifact in quality_artifacts for issue in artifact.payload["issues"]
    }

    assert {"high_missing", "duplicate_rows", "constant_column", "mixed_type_string"}.issubset(
        issue_codes
    )


def test_auto_eda_time_series_dataset_creates_trend_chart(tmp_path) -> None:
    result = run_auto_eda(
        [GOLDEN_DATA / "time_series_sales.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="time_run",
    )
    chart_titles = {
        artifact.payload["title"]
        for artifact in result.artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    }
    profile = next(
        artifact for artifact in result.artifacts if artifact.type is ArtifactType.DATASET_PROFILE
    )

    assert profile.payload["semantic_type_counts"]["datetime"] == 1
    assert profile.payload["semantic_type_counts"]["numeric"] == 1
    assert profile.payload["semantic_type_counts"]["categorical"] == 1
    assert "Records over order_date" in chart_titles


def test_auto_eda_blank_run_id_creates_fresh_runs_for_same_input(tmp_path) -> None:
    csv_path = tmp_path / "time_series_sales.csv"
    csv_path.write_text(
        "order_date,amount,region\n2026-01-01,10,East\n2026-01-02,20,West\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    first = run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="project_demo",
    )
    second = run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="project_demo",
    )

    assert first.session_id != second.session_id
    assert (workspace / f"projects/project_demo/sessions/{first.session_id}/manifest.json").exists()
    assert (
        workspace / f"projects/project_demo/sessions/{second.session_id}/manifest.json"
    ).exists()


def test_auto_eda_multi_dataset_run_creates_relationship_artifacts(tmp_path) -> None:
    result = run_auto_eda(
        [
            GOLDEN_DATA / "ecommerce_orders.csv",
            GOLDEN_DATA / "ecommerce_customers.csv",
            GOLDEN_DATA / "ecommerce_products.csv",
            GOLDEN_DATA / "ecommerce_marketing.csv",
        ],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="ecommerce_relationships",
        relationship_discovery="eager",
    )

    artifact_types = {artifact.type for artifact in result.artifacts}

    assert ArtifactType.RELATIONSHIP_CANDIDATE_SET in artifact_types
    assert ArtifactType.RELATIONSHIP_VALIDATION_SET in artifact_types
    assert ArtifactType.ER_DIAGRAM in artifact_types
    assert ArtifactType.QUESTION_CANDIDATE_SET in artifact_types
    assert ArtifactType.QUESTION_EXECUTION_RESULT in artifact_types
    assert ArtifactType.SQL_RESULT in artifact_types

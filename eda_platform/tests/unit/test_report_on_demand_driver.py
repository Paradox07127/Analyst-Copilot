"""generate_report_on_demand offline regression: terminates fast, persists the
report, and pulls in derived-run artifacts transitively (auto_eda closure loop)."""

from __future__ import annotations

import time
from pathlib import Path

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, generate_report_on_demand
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionExecutionResult
from eda_platform.schemas.sessions import SessionManifest
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset

PROJECT = "proj_on_demand"
PRIMARY = "run_primary"
DERIVED = "qsess_derived"


def _sql_artifact(session_id: str, artifact_id: str, parents: list[str]) -> Artifact:
    result = SqlResult(
        sql="select region, sum(revenue) as total from sales group by region",
        columns=["region", "total"],
        dtypes={"region": "varchar", "total": "double"},
        rows_preview=[{"region": "East", "total": 40.0}],
        row_count=1,
    )
    return Artifact(
        id=artifact_id,
        type=ArtifactType.SQL_RESULT,
        project_id=PROJECT,
        session_id=session_id,
        parents=parents,
        payload=result.model_dump(mode="json"),
    )


def _build_workspace(tmp_path: Path) -> AutoEDAResult:
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.ensure_project(PROJECT, name=PROJECT)
    store.start_session(PROJECT, PRIMARY)
    store.start_session(PROJECT, DERIVED)

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,revenue,region\n1,10,East\n2,30,East\n", encoding="utf-8"
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id=PROJECT, session_id=PRIMARY)

    qexec = QuestionExecutionResult(
        question_id="q_demo",
        question="Which region drives revenue?",
        origin="template",
        status="succeeded",
    )
    qexec_artifact = Artifact(
        id="qexec_demo",
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id=PROJECT,
        session_id=PRIMARY,
        payload=qexec.model_dump(mode="json"),
    )
    primary_artifacts = [profile, qexec_artifact]

    # Derived run: a two-hop parent chain off a primary artifact, so the
    # transitive-closure loop needs more than one pass to converge.
    hop1 = _sql_artifact(DERIVED, "sql_hop1", parents=[qexec_artifact.id])
    hop2 = _sql_artifact(DERIVED, "sql_hop2", parents=["sql_hop1"])
    unrelated = _sql_artifact(DERIVED, "sql_unrelated", parents=["someone_else"])

    for artifact in [*primary_artifacts, hop1, hop2, unrelated]:
        store.save_artifact(artifact)
    store.write_manifest(
        SessionManifest(session_id=PRIMARY, project_id=PROJECT, input_hashes={}, code_version="test")
    )
    store.write_manifest(
        SessionManifest(
            session_id=DERIVED,
            project_id=PROJECT,
            input_hashes={},
            code_version="test",
            source_session_id=PRIMARY,
        )
    )

    return AutoEDAResult(
        project_id=PROJECT,
        session_id=PRIMARY,
        business_context="",
        artifacts=primary_artifacts,
        report_markdown="",
        workspace=workspace,
        loaded_datasets=[],
    )


def test_offline_on_demand_report_terminates_and_persists(tmp_path: Path) -> None:
    result = _build_workspace(tmp_path)

    started = time.perf_counter()
    generated = generate_report_on_demand(result, llm=None)
    elapsed = time.perf_counter() - started

    # The pre-pipeline scan plus offline generation is file I/O only; a minute
    # is far beyond any healthy bound and still catches a real stall/loop.
    assert elapsed < 60.0

    types = {artifact.type for artifact in generated.artifacts}
    assert ArtifactType.MARKDOWN_REPORT in types
    assert ArtifactType.REPORT_BUNDLE in types
    assert generated.report_markdown.strip()

    report_file = tmp_path / "ws" / "projects" / PROJECT / "sessions" / PRIMARY / "report" / "report.md"
    assert report_file.is_file()

    ids = {artifact.id for artifact in generated.artifacts}
    assert {"sql_hop1", "sql_hop2"} <= ids, "transitive derived artifacts must be included"
    assert "sql_unrelated" not in ids

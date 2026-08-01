"""ArtifactService: index-only pagination, type filter, detail read with
containment, payload path relativization."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import eda_platform.application.services.artifact_service as artifact_service_module
from eda_platform.application.services.artifact_service import (
    AgentHandoffNotFoundError,
    AgentHandoffNotReadyError,
    AgentHandoffTerminalError,
    ArtifactNotFoundError,
    ArtifactService,
    ArtifactTooLargeError,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    SessionNotFoundError,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.tools.agent_handoff import create_agent_handoff_artifact

PROJECT = "demo"
RUN = "run_1"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    return store


@pytest.fixture
def service(store: ArtifactStore) -> ArtifactService:
    return ArtifactService(store)


def _save(store: ArtifactStore, artifact_id: str, artifact_type: ArtifactType, **payload) -> None:
    store.save_artifact(
        Artifact(
            id=artifact_id,
            type=artifact_type,
            project_id=PROJECT,
            session_id=RUN,
            payload=payload or {"k": "v"},
        )
    )


def _seed_five(store: ArtifactStore) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC)
    _save(store, "chart_2", ArtifactType.CHART_SPEC)
    _save(store, "prof_1", ArtifactType.DATASET_PROFILE)
    _save(store, "quality_1", ArtifactType.QUALITY_ISSUE_SET)
    _save(store, "chart_3", ArtifactType.CHART_SPEC)


def test_paginates_in_insertion_order(store: ArtifactStore, service: ArtifactService) -> None:
    _seed_five(store)

    first = service.list_artifacts(RUN, limit=2)
    assert [item.artifact_id for item in first.items] == ["chart_1", "chart_2"]
    assert first.next_cursor

    second = service.list_artifacts(RUN, limit=2, cursor=first.next_cursor)
    assert [item.artifact_id for item in second.items] == ["prof_1", "quality_1"]
    assert second.next_cursor

    third = service.list_artifacts(RUN, limit=2, cursor=second.next_cursor)
    assert [item.artifact_id for item in third.items] == ["chart_3"]
    assert third.next_cursor is None


def test_type_filter(store: ArtifactStore, service: ArtifactService) -> None:
    _seed_five(store)

    page = service.list_artifacts(RUN, artifact_type=ArtifactType.CHART_SPEC.value)
    assert [item.artifact_id for item in page.items] == ["chart_1", "chart_2", "chart_3"]
    assert all(item.type == "ChartSpec" for item in page.items)

    assert service.list_artifacts(RUN, artifact_type="NoSuchType").items == []


def test_summaries_never_carry_payload(store: ArtifactStore, service: ArtifactService) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC, huge="x" * 1000)
    page = service.list_artifacts(RUN)
    assert "payload" not in page.items[0].model_dump()


def test_list_is_pure_index_even_without_payload_files(
    store: ArtifactStore, service: ArtifactService
) -> None:
    _seed_five(store)
    for name in ["chart_1", "chart_2", "prof_1", "quality_1", "chart_3"]:
        store.artifact_path(PROJECT, RUN, name).unlink()

    page = service.list_artifacts(RUN)

    assert [item.artifact_id for item in page.items] == [
        "chart_1",
        "chart_2",
        "prof_1",
        "quality_1",
        "chart_3",
    ]


def test_list_unknown_or_internal_run_raises(service: ArtifactService) -> None:
    with pytest.raises(SessionNotFoundError):
        service.list_artifacts("missing_run")
    with pytest.raises(SessionNotFoundError):
        service.list_artifacts("qsess__internal_1")


def test_invalid_cursor_raises(store: ArtifactStore, service: ArtifactService) -> None:
    _seed_five(store)
    with pytest.raises(InvalidCursorError):
        service.list_artifacts(RUN, cursor="not-base64!!")


def test_cursor_is_bound_to_type_filter(store: ArtifactStore, service: ArtifactService) -> None:
    _seed_five(store)
    unfiltered = service.list_artifacts(RUN, limit=2)
    assert unfiltered.next_cursor

    with pytest.raises(InvalidCursorError):
        service.list_artifacts(
            RUN,
            artifact_type=ArtifactType.CHART_SPEC.value,
            cursor=unfiltered.next_cursor,
        )

    filtered = service.list_artifacts(RUN, artifact_type=ArtifactType.CHART_SPEC.value, limit=2)
    assert filtered.next_cursor
    with pytest.raises(InvalidCursorError):
        service.list_artifacts(RUN, cursor=filtered.next_cursor)


def test_cursor_is_bound_to_run(store: ArtifactStore, service: ArtifactService) -> None:
    _seed_five(store)
    other_run = "run_2"
    store.start_session(PROJECT, other_run)
    store.save_artifact(
        Artifact(
            id="other_1",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=other_run,
            payload={"k": "v"},
        )
    )
    first = service.list_artifacts(RUN, limit=2)
    assert first.next_cursor

    with pytest.raises(InvalidCursorError):
        service.list_artifacts(other_run, cursor=first.next_cursor)


def test_get_artifact_detail(store: ArtifactStore, service: ArtifactService) -> None:
    store.save_artifact(
        Artifact(
            id="chart_1",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=RUN,
            parents=["profile_1"],
            payload={"title": "Revenue by month"},
            evidence=[EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")],
            code_ref="eda.chart",
            plain_language="Revenue chart",
        )
    )

    detail = service.get_artifact(RUN, "chart_1")

    assert detail.artifact_id == "chart_1"
    assert detail.type == "ChartSpec"
    assert detail.session_id == RUN
    assert detail.payload == {"title": "Revenue by month"}
    assert detail.parents == ["profile_1"]
    assert detail.evidence[0].artifact_id == "table_1"
    assert detail.env_digest
    assert detail.code_ref == "eda.chart"
    assert detail.plain_language == "Revenue chart"


def test_get_artifact_detail_is_bound_to_run_partition(
    store: ArtifactStore, service: ArtifactService
) -> None:
    _save(store, "shared_1", ArtifactType.TABLE, owner="run_1")
    store.start_session(PROJECT, "run_other")
    store.save_artifact(
        Artifact(
            id="shared_1",
            type=ArtifactType.TABLE,
            project_id=PROJECT,
            session_id="run_other",
            payload={"owner": "run_other"},
        )
    )
    assert service.get_artifact(RUN, "shared_1").payload == {"owner": "run_1"}


def test_get_artifact_missing_raises(service: ArtifactService) -> None:
    with pytest.raises(ArtifactNotFoundError):
        service.get_artifact(RUN, "nope")


def test_agent_handoff_requires_completed_publish_barrier(
    store: ArtifactStore, service: ArtifactService
) -> None:
    with pytest.raises(AgentHandoffNotReadyError):
        service.get_agent_handoff(RUN)

    handoff = create_agent_handoff_artifact(
        [],
        project_id=PROJECT,
        session_id=RUN,
        producer_version="test",
        execution_fingerprint="fingerprint",
        input_hashes={},
    )
    store.save_artifact(handoff)
    with pytest.raises(AgentHandoffNotReadyError):
        service.get_agent_handoff(RUN)

    store.mark_session_status(PROJECT, RUN, "completed")
    detail = service.get_agent_handoff(RUN)
    assert detail.artifact_id == handoff.id
    assert detail.payload.contract_version == "3.0"


def test_agent_handoff_waits_for_active_source_lane_job(
    store: ArtifactStore, service: ArtifactService
) -> None:
    store.mark_session_status(PROJECT, RUN, "completed")
    store.create_job(
        job_id="job_report",
        session_id="rpsess_report_1",
        project_id=PROJECT,
        kind="report_generate",
        lane_key=RUN,
    )

    with pytest.raises(AgentHandoffNotReadyError, match="job_report"):
        service.get_agent_handoff(RUN)

    store.mark_job_status("job_report", "completed")
    with pytest.raises(AgentHandoffNotFoundError):
        service.get_agent_handoff(RUN)


def test_completed_legacy_session_has_typed_handoff_not_found(
    store: ArtifactStore, service: ArtifactService
) -> None:
    store.mark_session_status(PROJECT, RUN, "completed")
    with pytest.raises(AgentHandoffNotFoundError):
        service.get_agent_handoff(RUN)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_session_does_not_publish_agent_handoff(
    store: ArtifactStore, service: ArtifactService, status: str
) -> None:
    store.mark_session_status(PROJECT, RUN, status)

    with pytest.raises(AgentHandoffTerminalError):
        service.get_agent_handoff(RUN)


def test_internal_run_artifact_hidden(store: ArtifactStore, service: ArtifactService) -> None:
    store.start_session(PROJECT, "qsess__internal_1")
    store.save_artifact(
        Artifact(
            id="internal_1",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id="qsess__internal_1",
            payload={},
        )
    )
    with pytest.raises(ArtifactNotFoundError):
        service.get_artifact(RUN, "internal_1")


def test_payload_workspace_paths_are_relativized(
    tmp_path: Path, store: ArtifactStore, service: ArtifactService
) -> None:
    absolute = str((tmp_path / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    _save(
        store,
        "sql_1",
        ArtifactType.SQL_RESULT,
        source=absolute,
        nested={"paths": [absolute, "unrelated"]},
    )

    payload = service.get_artifact(RUN, "sql_1").payload

    assert payload["source"] == f"projects/{PROJECT}/uploads/a.csv"
    assert payload["nested"]["paths"] == [f"projects/{PROJECT}/uploads/a.csv", "unrelated"]


def test_payload_sibling_prefix_and_prose_are_preserved(
    tmp_path: Path, store: ArtifactStore, service: ArtifactService
) -> None:
    root = str(tmp_path.resolve())
    _save(
        store,
        "sql_1",
        ArtifactType.SQL_RESULT,
        sibling=f"{root}_backup/x",
        prose=f"{root} is configured",
        exact=root,
    )

    payload = service.get_artifact(RUN, "sql_1").payload

    assert payload["sibling"] == f"{root}_backup/x"
    assert payload["prose"] == f"{root} is configured"
    assert payload["exact"] == "."


def test_warnings_workspace_paths_are_relativized(
    tmp_path: Path, store: ArtifactStore, service: ArtifactService
) -> None:
    absolute = str((tmp_path / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    store.save_artifact(
        Artifact(
            id="warned_1",
            type=ArtifactType.CHART_SPEC,
            project_id=PROJECT,
            session_id=RUN,
            payload={},
            warnings=[absolute, "plain warning"],
        )
    )

    warnings = service.get_artifact(RUN, "warned_1").warnings

    assert warnings == [f"projects/{PROJECT}/uploads/a.csv", "plain warning"]


def test_evidence_workspace_paths_are_relativized(
    tmp_path: Path, store: ArtifactStore, service: ArtifactService
) -> None:
    absolute = str((tmp_path / "projects" / PROJECT / "uploads" / "a.csv").resolve())
    store.save_artifact(
        Artifact(
            id="evidence_1",
            type=ArtifactType.TABLE,
            project_id=PROJECT,
            session_id=RUN,
            payload={},
            evidence=[
                EvidenceRef(
                    kind="artifact",
                    locator=absolute,
                    value=absolute,
                    unit_reference=absolute,
                )
            ],
        )
    )

    evidence = service.get_artifact(RUN, "evidence_1").evidence[0]
    relative = f"projects/{PROJECT}/uploads/a.csv"

    assert evidence.locator == relative
    assert evidence.value == relative
    assert evidence.unit_reference == relative


def test_identity_mismatch_is_refused(store: ArtifactStore, service: ArtifactService) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC)
    _save(store, "chart_2", ArtifactType.CHART_SPEC)
    other_path = store.artifact_path(PROJECT, RUN, "chart_2")
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute(
            "update artifacts set path = ? where artifact_id = ?",
            (str(other_path), "chart_1"),
        )

    with pytest.raises(ArtifactNotFoundError):
        service.get_artifact(RUN, "chart_1")


def test_oversized_artifact_is_refused(
    store: ArtifactStore,
    service: ArtifactService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC, title="Revenue")
    monkeypatch.setattr(artifact_service_module, "MAX_ARTIFACT_PAYLOAD_BYTES", 16)

    with pytest.raises(ArtifactTooLargeError):
        service.get_artifact(RUN, "chart_1")


def test_artifact_read_cap_is_inclusive(
    store: ArtifactStore,
    service: ArtifactService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC, title="Revenue")
    size = store.artifact_path(PROJECT, RUN, "chart_1").stat().st_size

    monkeypatch.setattr(artifact_service_module, "MAX_ARTIFACT_PAYLOAD_BYTES", size)
    assert service.get_artifact(RUN, "chart_1").artifact_id == "chart_1"

    monkeypatch.setattr(artifact_service_module, "MAX_ARTIFACT_PAYLOAD_BYTES", size - 1)
    with pytest.raises(ArtifactTooLargeError):
        service.get_artifact(RUN, "chart_1")


def test_detail_path_escaping_workspace_is_refused(
    tmp_path: Path, store: ArtifactStore, service: ArtifactService
) -> None:
    _save(store, "chart_1", ArtifactType.CHART_SPEC)
    row = store.artifact_index_row("chart_1")
    assert row is not None
    outside = tmp_path.parent / "escape_artifact.json"
    outside.write_text(row["path"].read_text(encoding="utf-8"), encoding="utf-8")
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute(
            "update artifacts set path = ? where artifact_id = ?",
            (str(outside), "chart_1"),
        )

    with pytest.raises(ArtifactNotFoundError):
        service.get_artifact(RUN, "chart_1")

"""Jobs table CRUD/idempotency, kernel cancel_check, and JobService semantics."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, TypedDict

import pytest

from eda_platform.application.dto import JobStatus
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.investigation_service import (
    APPROVAL_KIND_EXECUTE,
    InvestigationService,
)
from eda_platform.application.services.job_service import (
    SUPPORTED_JOB_KINDS,
    JobConflictError,
    JobIdempotencyMismatchError,
    JobNotFoundError,
    JobService,
    JobValidationError,
    reap_orphan_jobs,
)
from eda_platform.core.kernel import SessionCancelled, SessionContext, run_pipeline
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import trace_job_scope
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    return store


def _seed_csv(root: Path) -> Path:
    path = root / "seed" / "orders.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "region,amount\n" + "\n".join(f"r{i % 3},{100 + i}" for i in range(30)) + "\n",
        encoding="utf-8",
    )
    return path


class _RecordingBackend:
    def __init__(self, store: ArtifactStore | None = None) -> None:
        self.commands: list[JobCommand] = []
        self._store = store

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        # Mirrors LocalProcessJobBackend: the service cancels via the backend
        # port now (review O3), so the fake must flip the flag itself.
        if self._store is not None:
            self._store.request_cancel(job_id)

    def status(self, job_id: str) -> str:
        return "queued"


def test_job_crud_and_status_timestamps(store: ArtifactStore) -> None:
    job = store.create_job(
        job_id="job_1", session_id="run_1", project_id="demo", kind="auto_eda"
    )
    assert job["status"] == "queued"
    assert job["cancel_requested"] is False
    assert job["created_at"]
    assert job["started_at"] is None

    store.mark_job_status("job_1", "running")
    running = store.get_job("job_1")
    assert running is not None
    assert running["status"] == "running"
    assert running["started_at"]
    assert running["finished_at"] is None

    store.mark_job_status("job_1", "failed", error_code="ValueError", error_message="boom")
    failed = store.get_job("job_1")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["finished_at"]
    assert failed["error_code"] == "ValueError"
    assert failed["error_message"] == "boom"

    assert store.get_job("job_missing") is None


def test_request_cancel_sets_flag(store: ArtifactStore) -> None:
    store.create_job(job_id="job_1", session_id="run_1", project_id="demo", kind="auto_eda")
    assert store.request_cancel("job_1") is True
    job = store.get_job("job_1")
    assert job is not None
    assert job["cancel_requested"] is True
    assert store.request_cancel("job_missing") is False


def test_idempotency_key_lookup_and_uniqueness(store: ArtifactStore) -> None:
    store.create_job(
        job_id="job_1", session_id="run_1", project_id="demo", kind="auto_eda", idempotency_key="k1"
    )
    found = store.find_by_idempotency_key("k1")
    assert found is not None
    assert found["job_id"] == "job_1"
    assert store.find_by_idempotency_key("k2") is None
    with pytest.raises(sqlite3.IntegrityError):
        store.create_job(
            job_id="job_2", session_id="run_1", project_id="demo", kind="auto_eda", idempotency_key="k1"
        )
    # NULL idempotency keys never collide; active lane identity remains
    # independent and therefore uses distinct runs here.
    store.create_job(job_id="job_3", session_id="run_2", project_id="demo", kind="auto_eda")
    store.create_job(job_id="job_4", session_id="run_3", project_id="demo", kind="auto_eda")


class _NoopStep:
    name: ClassVar[str] = "noop"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = ()

    def __init__(self) -> None:
        self.ran = False

    def run(self, ctx: SessionContext) -> list[Artifact]:
        self.ran = True
        return []


def test_kernel_cancel_check_raises_at_step_boundary(store: ArtifactStore) -> None:
    step = _NoopStep()
    ctx = SessionContext(
        project_id="demo", session_id="run_cancel", store=store, cancel_check=lambda: True
    )
    with pytest.raises(SessionCancelled):
        run_pipeline([step], ctx)
    assert step.ran is False


def test_kernel_without_cancel_runs_step(store: ArtifactStore) -> None:
    step = _NoopStep()
    ctx = SessionContext(
        project_id="demo", session_id="run_go", store=store, cancel_check=lambda: False
    )
    run_pipeline([step], ctx)
    assert step.ran is True


def test_driver_cancel_mid_run_keeps_artifacts(store: ArtifactStore) -> None:
    csv_path = _seed_csv(store.root)
    checks = {"count": 0}

    def cancel_after_two() -> bool:
        checks["count"] += 1
        return checks["count"] > 2

    with pytest.raises(SessionCancelled):
        run_auto_eda(
            [csv_path],
            workspace=store.root,
            project_id="demo",
            session_id="run_mid_cancel",
            generate_report=False,
            cancel_check=cancel_after_two,
        )
    # Steps that finished before the cancel boundary keep their artifacts.
    artifacts, _warnings = store.list_artifacts_safe(project_id="demo", session_id="run_mid_cancel")
    assert len(artifacts) >= 1


def test_service_create_job_is_idempotent(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    backend = _RecordingBackend()
    service = JobService(store, backend)
    first = service.create_job(
        "run_1",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        llm="offline",
        idempotency_key="key-1",
    )
    second = service.create_job(
        "run_1",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        llm="offline",
        idempotency_key="key-1",
    )
    assert first.job_id == second.job_id
    assert len(backend.commands) == 1
    stored = store.get_job(first.job_id)
    assert stored is not None
    assert stored["request_digest"]
    params = json.loads(backend.commands[0].params_json)
    assert params["dataset_paths"] == ["seed/orders.csv"]
    assert params["llm"] == "offline"
    assert first.events_url == f"/api/v1/jobs/{first.job_id}/events"


def test_service_resolves_uploaded_dataset_id(store: ArtifactStore) -> None:
    upload_dir = store.project_dir("demo") / "uploads" / "orders-abc123" / "v1"
    upload_dir.mkdir(parents=True)
    (upload_dir / "orders.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    service = JobService(store, _RecordingBackend())
    status = service.create_job(
        "run_ds", kind="auto_eda", project_id="demo", datasets=["orders-abc123"]
    )
    assert status.status == "queued"


def test_service_rejects_bad_dataset_refs(store: ArtifactStore) -> None:
    service = JobService(store, _RecordingBackend())
    with pytest.raises(JobValidationError):
        service.create_job("run_1", kind="auto_eda", project_id="demo", datasets=["nope.csv"])
    with pytest.raises(JobValidationError):
        service.create_job("run_1", kind="auto_eda", project_id="demo", datasets=[])
    with pytest.raises(JobValidationError):
        service.create_job(
            "run_1", kind="auto_eda", project_id="demo", datasets=["../outside.csv"]
        )


def test_service_cancel_emits_trace_event_and_is_terminal_noop(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend(store))
    created = service.create_job(
        "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    cancelled = service.cancel_job(created.job_id)
    assert cancelled.cancel_requested is True
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert store.get_session_status("run_1") == "cancelled"
    events = store.list_trace_events(project_id="demo", session_id="run_1")
    assert [e.event_type for e in events] == ["job.queued", "job.cancelled"]

    after_terminal = service.cancel_job(created.job_id)
    assert after_terminal.status == "cancelled"
    events = store.list_trace_events(project_id="demo", session_id="run_1")
    assert [e.event_type for e in events] == ["job.queued", "job.cancelled"]

    with pytest.raises(JobNotFoundError):
        service.cancel_job("job_missing")


def _trace(
    store: ArtifactStore,
    session_id: str,
    event_type: str,
    name: str,
    *,
    job_id: str | None = None,
    job_generation: int | None = None,
) -> None:
    store.append_trace(
        "demo",
        TraceEvent(
            session_id=session_id,
            event_type=event_type,
            name=name,
            job_id=job_id,
            job_generation=job_generation,
            finished_at=datetime.now(UTC),
        ),
    )


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


# Review F3: one active job per run.
def test_second_active_job_for_run_conflicts(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    with pytest.raises(JobConflictError) as excinfo:
        service.create_job(
            "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
        )
    assert excinfo.value.job_id == first.job_id
    assert first.job_id in str(excinfo.value)

    # A terminal job frees the run for a new one.
    store.mark_job_status(first.job_id, "completed")
    second = service.create_job(
        "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    assert second.job_id != first.job_id


# Review O5: an idempotency key is bound to its original run/kind.
def test_idempotency_key_content_mismatch_conflicts(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_1",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        idempotency_key="key-1",
    )
    with pytest.raises(JobIdempotencyMismatchError) as excinfo:
        service.create_job(
            "run_other",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            idempotency_key="key-1",
        )
    assert excinfo.value.job_id == first.job_id


def test_idempotency_key_same_run_kind_but_different_body_is_rejected(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_1",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        business_context="first request",
        idempotency_key="key-content",
    )

    with pytest.raises(
        JobIdempotencyMismatchError, match="different canonical request"
    ):
        service.create_job(
            "run_1",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            business_context="different request",
            idempotency_key="key-content",
        )

    existing = store.find_by_idempotency_key("key-content")
    assert existing is not None
    assert existing["job_id"] == first.job_id


def test_legacy_idempotency_row_without_digest_fails_closed(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    legacy = store.create_job(
        job_id="job_legacy",
        session_id="run_legacy",
        project_id="demo",
        kind="auto_eda",
        idempotency_key="legacy-key",
    )
    assert legacy["request_digest"] is None

    with pytest.raises(JobIdempotencyMismatchError):
        JobService(store, _RecordingBackend()).create_job(
            "run_legacy",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            idempotency_key="legacy-key",
        )


def test_pre_scope_schema_migrates_request_scope_as_legacy_null(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "legacy_scope_workspace"
    store = ArtifactStore(workspace)
    store.ensure_project("demo", name="Demo")
    store.create_job(
        job_id="job_pre_scope",
        session_id="rpsess_legacy",
        project_id="demo",
        kind="report_generate",
        idempotency_key="pre-scope-key",
        request_digest="a" * 64,
    )
    with sqlite3.connect(store.db_path) as conn:
        # Simulate a pre-foundation schema: that legacy database did not yet
        # have the immutable-identity trigger referencing request_scope.
        conn.execute("drop trigger trg_jobs_request_identity_immutable")
        conn.execute("alter table jobs drop column request_scope")

    migrated = ArtifactStore(workspace)
    row = migrated.get_job("job_pre_scope")
    assert row is not None
    assert row["request_digest"] == "a" * 64
    assert row["request_scope"] is None
    with pytest.raises(JobIdempotencyMismatchError):
        JobService(migrated, _RecordingBackend()).create_report_generate_job(
            "rpsess_fresh",
            project_id="demo",
            source_session_id="source_run",
            llm="offline",
            idempotency_key="pre-scope-key",
        )


def test_investigation_rearm_with_different_approved_llm_rejects_same_key(
    store: ArtifactStore,
) -> None:
    store.start_session("demo", "source_run")
    store.mark_session_status("demo", "source_run", "completed")
    approvals = ApprovalService(store)
    action = {
        "type": "investigation_execute",
        "source_session_id": "source_run",
        "plan_session_id": "plan_run",
        "plan_ids": ["plan_1"],
        "plan_fingerprints": {"plan_1": "fingerprint"},
    }
    offline_payload = {
        "project_id": "demo",
        "source_session_id": "source_run",
        "plan_session_id": "plan_run",
        "plan_ids": ["plan_1"],
        "plan_fingerprints": {"plan_1": "fingerprint"},
        "llm": "offline",
    }
    action_hash, offline_token, _expires = approvals.register(
        kind=APPROVAL_KIND_EXECUTE,
        session_id="source_run",
        project_id="demo",
        action=action,
        payload=offline_payload,
    )
    approvals.validate_and_consume(
        action_hash,
        kind=APPROVAL_KIND_EXECUTE,
        session_id="source_run",
        generation=offline_token,
        idempotency_key="investigation-key",
    )
    backend = _RecordingBackend()
    jobs = JobService(store, backend)
    first = jobs.create_investigation_execute_job(
        "ixsess_offline",
        project_id="demo",
        source_session_id="source_run",
        plan_session_id="plan_run",
        plan_ids=["plan_1"],
        plan_fingerprints={"plan_1": "fingerprint"},
        llm="offline",
        idempotency_key="investigation-key",
        idempotency_content={
            "source_session_id": "source_run",
            "action_hash": action_hash,
            "approval_payload_digest": payload_digest(offline_payload),
            "payload_policy": None,
        },
    )
    store.mark_job_status(first.job_id, "completed")

    env_payload = {**offline_payload, "llm": "env"}
    same_hash, env_token, _expires = approvals.register(
        kind=APPROVAL_KIND_EXECUTE,
        session_id="source_run",
        project_id="demo",
        action=action,
        payload=env_payload,
    )
    assert same_hash == action_hash

    with pytest.raises(JobIdempotencyMismatchError):
        InvestigationService(store, approvals, jobs).execute(
            "source_run",
            action_hash=action_hash,
            approval_token=env_token,
            idempotency_key="investigation-key",
        )


def test_secret_env_is_hashed_and_content_order_is_canonical(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    backend = _RecordingBackend()
    service = JobService(store, backend)
    first = service.create_job(
        "run_secret",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        precleaning={"enabled": True, "options": {"trim": True, "fill": "median"}},
        llm_env={"OPENAI_API_KEY": "raw-secret"},
        idempotency_key="secret-key",
    )
    replay = service.create_job(
        "run_secret",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        precleaning={"options": {"fill": "median", "trim": True}, "enabled": True},
        llm_env={"OPENAI_API_KEY": "raw-secret"},
        idempotency_key="secret-key",
    )
    assert replay.job_id == first.job_id
    stored = store.get_job(first.job_id)
    assert stored is not None
    assert "raw-secret" not in json.dumps(stored)

    with pytest.raises(JobIdempotencyMismatchError):
        service.create_job(
            "run_secret",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            precleaning={"enabled": True, "options": {"trim": True, "fill": "median"}},
            llm_env={"OPENAI_API_KEY": "different-secret"},
            idempotency_key="secret-key",
        )


def test_mutated_persisted_digest_fails_closed(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_mutated",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        idempotency_key="mutated-key",
    )
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="request identity"):
            conn.execute(
                "update jobs set request_digest = ? where job_id = ?",
                ("0" * 64, first.job_id),
            )
    persisted = store.get_job(first.job_id)
    assert persisted is not None
    assert persisted["request_digest"] != "0" * 64
    store.create_job(
        job_id="job_malicious_digest",
        session_id="run_malicious_digest",
        project_id="demo",
        kind="auto_eda",
        idempotency_key="malicious-digest-key",
        request_digest="0" * 64,
        request_scope="run_malicious_digest",
    )

    with pytest.raises(JobIdempotencyMismatchError):
        service.create_job(
            "run_malicious_digest",
            kind="auto_eda",
            project_id="demo",
            datasets=["seed/orders.csv"],
            idempotency_key="malicious-digest-key",
        )


def test_mutated_persisted_request_scope_fails_closed(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_report_generate_job(
        "rpsess_original",
        project_id="demo",
        source_session_id="source_run",
        llm="offline",
        idempotency_key="mutated-scope-key",
    )
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="request identity"):
            conn.execute(
                "update jobs set request_scope = ? where job_id = ?",
                ("different_source_run", first.job_id),
            )
    persisted = store.get_job(first.job_id)
    assert persisted is not None
    assert persisted["request_scope"] == "source_run"
    store.create_job(
        job_id="job_malicious_scope",
        session_id="rpsess_malicious",
        project_id="demo",
        kind="report_generate",
        idempotency_key="malicious-scope-key",
        request_digest=str(persisted["request_digest"]),
        request_scope="different_source_run",
    )

    with pytest.raises(JobIdempotencyMismatchError):
        service.create_report_generate_job(
            "rpsess_retry",
            project_id="demo",
            source_session_id="source_run",
            llm="offline",
            idempotency_key="malicious-scope-key",
        )


class _CommonJobArgs(TypedDict):
    project_id: str
    idempotency_key: str


def _create_kind_for_idempotency_matrix(
    service: JobService,
    kind: str,
    *,
    execution_session_id: str,
    idempotency_key: str,
) -> JobStatus:
    common: _CommonJobArgs = {
        "project_id": "demo",
        "idempotency_key": idempotency_key,
    }
    if kind == "auto_eda":
        return service.create_job(
            "logical_auto_run",
            kind=kind,
            datasets=["seed/orders.csv"],
            **common,
        )
    if kind == "report_generate":
        return service.create_report_generate_job(
            execution_session_id, source_session_id="source_run", llm="offline", **common
        )
    if kind == "session_fork":
        return service.create_run_fork_job(
            execution_session_id,
            source_session_id="source_run",
            decision_kind="ml_target",
            llm="offline",
            **common,
        )
    if kind == "question_exec":
        return service.create_question_exec_job(
            execution_session_id,
            source_session_id="source_run",
            question_id="question_1",
            candidate_fingerprint="fingerprint",
            llm="offline",
            **common,
        )
    if kind == "question_draft":
        return service.create_question_draft_job(
            execution_session_id,
            source_session_id="source_run",
            question="Which region leads?",
            llm="offline",
            **common,
        )
    if kind == "investigation_plan":
        return service.create_investigation_plan_job(
            execution_session_id,
            source_session_id="source_run",
            question_ids=["question_1"],
            deep=True,
            **common,
        )
    if kind == "investigation_execute":
        return service.create_investigation_execute_job(
            execution_session_id,
            source_session_id="source_run",
            plan_session_id="plan_run",
            plan_ids=["plan_1"],
            plan_fingerprints={"plan_1": "fingerprint"},
            llm="offline",
            **common,
        )
    if kind == "macro_loop":
        return service.create_macro_loop_job(
            execution_session_id,
            source_session_id="source_run",
            plan_session_id="plan_run",
            depth=4,
            llm="offline",
            **common,
        )
    if kind == "skill_replay":
        return service.create_skill_replay_job(
            execution_session_id,
            source_session_id="source_run",
            skill_id="skill_1",
            skill={"skill_id": "skill_1", "sql": "select 1"},
            dataset_ids=["dataset_1"],
            **common,
        )
    if kind == "relationship_validate":
        return service.create_relationship_validate_job(
            execution_session_id,
            source_session_id="source_run",
            relationship_id="rel_1",
            pair_label="orders.customer_id → customers.id",
            candidate_fingerprint="fingerprint",
            **common,
        )
    if kind == "relationship_discover":
        return service.create_relationship_discover_job(
            execution_session_id, source_session_id="source_run", **common
        )
    if kind == "synthesis_brief_create":
        return service.create_synthesis_brief_job(
            execution_session_id,
            source_session_id="source_run",
            finding_artifact_ids=["finding_1"],
            finding_session_ids={"finding_1": "finding_run"},
            business_context="Board review",
            **common,
        )
    if kind == "decision_report_generate":
        return service.create_decision_report_job(
            execution_session_id,
            source_session_id="source_run",
            brief_artifact_id="brief_1",
            brief_session_id="brief_run",
            **common,
        )
    if kind in {
        "cleaning_preview",
        "cleaning_apply",
        "dataset_distributions",
        "custom_chart",
    }:
        return service.create_data_operation_job(
            execution_session_id,
            kind=kind,
            source_session_id="source_run",
            params={"dataset_id": "dataset_1"},
            idempotency_content={"dataset_id": "dataset_1"},
            **common,
        )
    raise AssertionError(f"Unhandled job kind: {kind}")


@pytest.mark.parametrize("kind", sorted(SUPPORTED_JOB_KINDS))
def test_idempotency_matrix_separates_logical_scope_from_execution_run(
    store: ArtifactStore,
    kind: str,
) -> None:
    _seed_csv(store.root)
    backend = _RecordingBackend()
    service = JobService(store, backend)
    key = f"matrix-{kind}"

    first = _create_kind_for_idempotency_matrix(
        service, kind, execution_session_id=f"execution_a_{kind}", idempotency_key=key
    )
    replay = _create_kind_for_idempotency_matrix(
        service, kind, execution_session_id=f"execution_b_{kind}", idempotency_key=key
    )

    assert replay.job_id == first.job_id
    assert replay.session_id == first.session_id
    assert len(backend.commands) == 1
    stored = store.get_job(first.job_id)
    assert stored is not None
    assert stored["request_scope"]


# Review F4: enqueue failure must not strand a queued row.
def test_enqueue_failure_marks_job_failed(store: ArtifactStore) -> None:
    _seed_csv(store.root)

    class _BrokenBackend(_RecordingBackend):
        def enqueue(self, command: JobCommand) -> JobRef:
            raise RuntimeError("spawn blew up")

    service = JobService(store, _BrokenBackend())
    with pytest.raises(RuntimeError):
        service.create_job(
            "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
        )
    assert store.find_active_job_for_session("run_1") is None
    events = store.list_trace_events(project_id="demo", session_id="run_1")
    assert [e.event_type for e in events] == ["job.queued", "job.failed"]
    assert store.get_session_status("run_1") == "failed"
    job = store.get_job(events[0].name)
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_code"] == "enqueue_failed"


# Review F1: SSE reads are scoped to the requesting job.
def test_events_after_scopes_to_requesting_job(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    _trace(store, "run_1", "profile", "step_profile", job_id=first.job_id)
    _trace(store, "run_1", "job.completed", first.job_id, job_id=first.job_id)
    store.mark_job_status(first.job_id, "completed")

    second = service.create_job(
        "run_1", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    _trace(store, "run_1", "insight", "step_insight", job_id=second.job_id)
    _trace(store, "run_1", "job.completed", second.job_id, job_id=second.job_id)
    store.mark_job_status(second.job_id, "completed")

    page = service.events_after(second.job_id, 0)
    types_names = [(e.type, e.name) for e in page.events]
    # Clamped to job2's own queued event: nothing from job1's lifetime replays.
    assert types_names == [
        ("job.queued", second.job_id),
        ("insight", "step_insight"),
        ("job.completed", second.job_id),
    ]
    assert page.exhausted is True

    # Replaying the first job never leaks a later job's ordinary trace.
    page_first = service.events_after(first.job_id, 0)
    assert [(e.type, e.name) for e in page_first.events] == [
        ("job.queued", first.job_id),
        ("profile", "step_profile"),
        ("job.completed", first.job_id),
    ]


def test_events_after_uses_persisted_job_correlation_across_restart(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_restart", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    _trace(
        store,
        "run_restart",
        "step_started",
        "first_step",
        job_id=first.job_id,
        job_generation=1,
    )
    _trace(
        store, "run_restart", "job.completed", first.job_id, job_id=first.job_id
    )
    store.mark_job_status(first.job_id, "completed")
    second = service.create_job(
        "run_restart", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    _trace(
        store,
        "run_restart",
        "step_started",
        "second_step",
        job_id=second.job_id,
        job_generation=1,
    )
    _trace(store, "run_restart", "legacy_unowned", "ambiguous_legacy_event")
    _trace(
        store, "run_restart", "job.completed", second.job_id, job_id=second.job_id
    )
    store.mark_job_status(second.job_id, "completed")

    restarted = JobService(ArtifactStore(store.root), _RecordingBackend())
    first_page = restarted.events_after(first.job_id, 0)
    second_page = restarted.events_after(second.job_id, 0)

    assert [event.name for event in first_page.events] == [
        first.job_id,
        "first_step",
        first.job_id,
    ]
    assert [event.name for event in second_page.events] == [
        second.job_id,
        "second_step",
        second.job_id,
    ]
    assert all(
        event.name != "ambiguous_legacy_event"
        for event in [*first_page.events, *second_page.events]
    )


def test_events_after_pagination_ignores_unrelated_job_rows(
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    first = service.create_job(
        "run_page", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    _trace(store, "run_page", "profile", "first_profile", job_id=first.job_id)
    _trace(store, "run_page", "job.completed", first.job_id, job_id=first.job_id)
    store.mark_job_status(first.job_id, "completed")
    second = service.create_job(
        "run_page", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    for index in range(8):
        _trace(
            store,
            "run_page",
            "insight",
            f"second_{index}",
            job_id=second.job_id,
        )

    monkeypatch.setattr(
        "eda_platform.application.services.job_service.EVENTS_PAGE_LIMIT", 2
    )
    page = service.events_after(first.job_id, 0)

    assert [(event.type, event.name) for event in page.events] == [
        ("job.queued", first.job_id),
        ("profile", "first_profile"),
    ]
    assert page.exhausted is False
    next_page = service.events_after(first.job_id, page.cursor)
    assert [(event.type, event.name) for event in next_page.events] == [
        ("job.completed", first.job_id)
    ]
    assert next_page.exhausted is True


def test_trace_job_scope_isolates_concurrent_emitters_on_one_run(
    store: ArtifactStore,
) -> None:
    store.start_session("demo", "run_concurrent_trace")
    store.create_job(
        job_id="job_trace_a",
        session_id="run_concurrent_trace",
        project_id="demo",
        kind="auto_eda",
        lane_key="lane_trace_a",
    )
    store.create_job(
        job_id="job_trace_b",
        session_id="run_concurrent_trace",
        project_id="demo",
        kind="auto_eda",
        lane_key="lane_trace_b",
    )
    barrier = threading.Barrier(2)

    def emit(job_id: str, prefix: str) -> None:
        with trace_job_scope(job_id, 3):
            barrier.wait()
            for index in range(20):
                _trace(
                    store,
                    "run_concurrent_trace",
                    "step_progress",
                    f"{prefix}_{index}",
                )

    first = threading.Thread(target=emit, args=("job_trace_a", "a"))
    second = threading.Thread(target=emit, args=("job_trace_b", "b"))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()

    service = JobService(ArtifactStore(store.root), _RecordingBackend())
    first_names = [event.name for event in service.events_after("job_trace_a", 0).events]
    second_names = [event.name for event in service.events_after("job_trace_b", 0).events]
    assert first_names == [f"a_{index}" for index in range(20)]
    assert second_names == [f"b_{index}" for index in range(20)]


def test_trace_job_scope_rejects_explicit_cross_job_mislabel(
    store: ArtifactStore,
) -> None:
    store.start_session("demo", "run_trace_conflict")
    with trace_job_scope("job_owner", 2), pytest.raises(
        ValueError, match="conflicts with the active job scope"
    ):
        _trace(
            store,
            "run_trace_conflict",
            "step_progress",
            "wrong_owner",
            job_id="job_sibling",
            job_generation=2,
        )
    assert store.list_trace_events(
        project_id="demo", session_id="run_trace_conflict"
    ) == []


# Review F4: orphaned rows are failed at startup, live workers are untouched.
def test_reap_orphan_jobs(store: ArtifactStore) -> None:
    store.create_job(job_id="job_dead", session_id="run_a", project_id="demo", kind="auto_eda")
    store.mark_job_status("job_dead", "running")
    store.set_job_pid("job_dead", _dead_pid())

    store.create_job(job_id="job_nopid", session_id="run_b", project_id="demo", kind="auto_eda")

    store.create_job(job_id="job_live", session_id="run_c", project_id="demo", kind="auto_eda")
    store.mark_job_status("job_live", "running")
    store.set_job_pid("job_live", os.getpid())

    store.create_job(job_id="job_done", session_id="run_d", project_id="demo", kind="auto_eda")
    store.mark_job_status("job_done", "completed")

    assert reap_orphan_jobs(store) == 2

    dead = store.get_job("job_dead")
    assert dead is not None and dead["status"] == "failed"
    assert dead["error_code"] == "orphaned"
    nopid = store.get_job("job_nopid")
    assert nopid is not None and nopid["status"] == "failed"
    live = store.get_job("job_live")
    assert live is not None and live["status"] == "running"
    done = store.get_job("job_done")
    assert done is not None and done["status"] == "completed"

    # The reap emits a terminal trace event so an open SSE stream can close.
    events = store.list_trace_events(project_id="demo", session_id="run_a")
    assert [e.event_type for e in events] == ["job.failed"]
    assert events[0].summary["error_code"] == "orphaned"

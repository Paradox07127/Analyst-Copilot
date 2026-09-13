"""T5 failure paths: worker error translation, retry, and degraded marking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.agents.narrator import NarrationOutcome, narrate_report
from eda_platform.agents.reporting import AgenticReportResult
from eda_platform.application.dto import ReportView
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.job_service import (
    JobConflictError,
    JobService,
    JobValidationError,
)
from eda_platform.application.services.report_service import ReportService
from eda_platform.core.kernel import SessionContext
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import trace_job_scope
from eda_platform.drivers import auto_eda
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportSection,
    ReportStatus,
)
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.evidence import EvidencePack
from eda_platform.worker.error_translation import (
    LLMNotConfiguredError,
    describe_worker_failure,
)
from eda_platform.worker.runner import _degraded_completion_summary


class _RecordingBackend:
    def __init__(self) -> None:
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def status(self, job_id: str) -> str:
        return "queued"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    return store


def _seed_csv(root: Path) -> None:
    path = root / "seed" / "orders.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("region,amount\nr1,100\nr2,200\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Worker error translation


def test_missing_column_reads_as_a_sentence_not_a_bare_keyerror() -> None:
    failure = describe_worker_failure(KeyError("order_purchase_timestamp"))
    assert failure.error_code == "KeyError"
    assert "order_purchase_timestamp" in failure.message
    assert "column" in failure.message
    assert not failure.message.startswith("KeyError")
    assert failure.detail == "KeyError: 'order_purchase_timestamp'"


def test_file_memory_timeout_and_llm_config_map_to_actionable_text() -> None:
    cases = {
        FileNotFoundError("/tmp/x.csv"): "could not be found",
        MemoryError(): "ran out of memory",
        TimeoutError("read timed out"): "timed out",
        LLMNotConfiguredError("refusing to run silently offline"): "provider",
        RuntimeError("provider rejected the API key"): "credentials",
    }
    for exc, fragment in cases.items():
        failure = describe_worker_failure(exc)
        assert fragment in failure.message, (exc, failure.message)
        assert failure.detail is not None


def test_llm_config_failure_keeps_a_machine_readable_code() -> None:
    failure = describe_worker_failure(LLMNotConfiguredError("no provider"))
    assert failure.error_code == "llm_not_configured"


def test_unmapped_exception_gets_generic_text_and_keeps_raw_detail() -> None:
    failure = describe_worker_failure(ZeroDivisionError("division by zero"))
    assert failure.message.startswith("Analysis failed unexpectedly")
    assert "Trace" in failure.message
    assert failure.detail == "ZeroDivisionError: division by zero"


def test_platform_valueerror_sentences_pass_through() -> None:
    failure = describe_worker_failure(ValueError("question source changed since approval"))
    assert failure.message == "question source changed since approval"


# ---------------------------------------------------------------------------
# Retry (Run again)


def test_retry_recreates_an_auto_eda_job_with_identical_params(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    backend = _RecordingBackend()
    service = JobService(store, backend)
    first = service.create_job(
        "run_retry",
        kind="auto_eda",
        project_id="demo",
        datasets=["seed/orders.csv"],
        business_context="Revenue",
    )
    lifecycle = JobLifecycleRepository(store)
    lifecycle.fail_active(
        first.job_id, error_code="KeyError", error_message="boom"
    )

    retried = service.retry_job(first.job_id, llm_env={"EDA_LLM_PROVIDER": "openai"})

    assert retried.job_id != first.job_id
    assert retried.kind == "auto_eda"
    assert retried.session_id == first.session_id
    assert retried.status == "queued"
    first_params = json.loads(lifecycle.params_json(first.job_id) or "{}")
    retry_params = json.loads(lifecycle.params_json(retried.job_id) or "{}")
    assert retry_params == first_params
    # The env overlay is re-resolved at retry time, never persisted.
    assert backend.commands[-1].env == {"EDA_LLM_PROVIDER": "openai"}


def test_retry_refuses_active_jobs_and_non_auto_eda_kinds(
    store: ArtifactStore,
) -> None:
    _seed_csv(store.root)
    store.start_session("demo", "run_src")
    service = JobService(store, _RecordingBackend())
    active = service.create_job(
        "run_active", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    with pytest.raises(JobConflictError):
        service.retry_job(active.job_id)

    question = service.create_question_exec_job(
        "qsess_retry",
        project_id="demo",
        source_session_id="run_src",
        question_id="q1",
        candidate_fingerprint="f" * 32,
    )
    JobLifecycleRepository(store).fail_active(
        question.job_id, error_code="ValueError", error_message="boom"
    )
    with pytest.raises(JobValidationError):
        service.retry_job(question.job_id)


def test_retry_refuses_completed_jobs(store: ArtifactStore) -> None:
    _seed_csv(store.root)
    service = JobService(store, _RecordingBackend())
    job = service.create_job(
        "run_done", kind="auto_eda", project_id="demo", datasets=["seed/orders.csv"]
    )
    lifecycle = JobLifecycleRepository(store)
    claim = lifecycle.claim_launch(job.job_id, owner="test")
    lifecycle.acknowledge_spawn(claim, pid=1, birth_identity="test")
    lifecycle.child_start(claim)
    lifecycle.finish(claim, "completed")
    with pytest.raises(JobValidationError):
        service.retry_job(job.job_id)


# ---------------------------------------------------------------------------
# Degraded completion marking


def _queued_job(store: ArtifactStore, session_id: str, params: dict) -> dict:
    lifecycle = JobLifecycleRepository(store)
    return lifecycle.create_queued_job(
        job_id=f"job_{session_id}",
        session_id=session_id,
        project_id="demo",
        kind="auto_eda",
        params_json=json.dumps(params),
        idempotency_key=None,
        lane_key=session_id,
        request_digest="d" * 64,
        request_scope=session_id,
    )


def test_degraded_summary_collects_llm_fallback_events(store: ArtifactStore) -> None:
    store.start_session("demo", "run_deg")
    job = _queued_job(store, "run_deg", {"llm": "env"})
    with trace_job_scope(str(job["job_id"]), 1):
        store.append_trace(
            "demo",
            TraceEvent(
                session_id="run_deg",
                event_type="question_llm_skipped",
                name="m4_question_discovery",
                summary={"error": "boom", "degraded": True},
            ),
        )
        store.append_trace(
            "demo",
            TraceEvent(
                session_id="run_deg",
                event_type="report_degraded",
                name="export_agentic_report",
                summary={"degraded": True, "reason": "Deterministic report fallback."},
            ),
        )

    summary = _degraded_completion_summary(store, job, {"llm": "env"})
    assert summary == {
        "degraded": True,
        "degraded_reasons": [
            "Question discovery ran without the language model.",
            "Deterministic report fallback.",
        ],
    }
    # A deliberately offline job is not degraded by running offline.
    assert _degraded_completion_summary(store, job, {"llm": "offline"}) is None


def test_degraded_summary_ignores_clean_runs(store: ArtifactStore) -> None:
    store.start_session("demo", "run_clean")
    job = _queued_job(store, "run_clean", {"llm": "env"})
    with trace_job_scope(str(job["job_id"]), 1):
        store.append_trace(
            "demo",
            TraceEvent(
                session_id="run_clean",
                event_type="step_completed",
                name="profile_dataset",
                summary={},
            ),
        )
    assert _degraded_completion_summary(store, job, {"llm": "env"}) is None


def test_finish_merges_degraded_summary_and_error_detail_into_job_events(
    store: ArtifactStore,
) -> None:
    store.start_session("demo", "run_fin")
    job = _queued_job(store, "run_fin", {"llm": "env"})
    lifecycle = JobLifecycleRepository(store)
    claim = lifecycle.claim_launch(str(job["job_id"]), owner="test")
    lifecycle.acknowledge_spawn(claim, pid=1, birth_identity="test")
    lifecycle.child_start(claim)
    lifecycle.finish(
        claim,
        "completed",
        summary={"degraded": True, "degraded_reasons": ["No model reached the run."]},
    )
    rows = store.list_job_trace_rows_after(
        job_id=str(job["job_id"]), after_id=0, limit=100
    )
    events = [TraceEvent.model_validate_json(payload) for _, payload in rows]
    completed = next(e for e in events if e.event_type == "job.completed")
    assert completed.summary["degraded"] is True
    assert completed.summary["degraded_reasons"] == ["No model reached the run."]

    store.start_session("demo", "run_fin2")
    job2 = _queued_job(store, "run_fin2", {"llm": "env"})
    claim2 = lifecycle.claim_launch(str(job2["job_id"]), owner="test")
    lifecycle.acknowledge_spawn(claim2, pid=1, birth_identity="test")
    lifecycle.child_start(claim2)
    lifecycle.finish(
        claim2,
        "failed",
        error_code="KeyError",
        error_message="The analysis referenced a missing column.",
        error_detail="KeyError: 'order_purchase_timestamp'",
    )
    rows2 = store.list_job_trace_rows_after(
        job_id=str(job2["job_id"]), after_id=0, limit=100
    )
    events2 = [TraceEvent.model_validate_json(payload) for _, payload in rows2]
    failed = next(e for e in events2 if e.event_type == "job.failed")
    assert failed.summary["error_message"] == "The analysis referenced a missing column."
    assert failed.summary["error_detail"] == "KeyError: 'order_purchase_timestamp'"


# ---------------------------------------------------------------------------
# Deterministic-fallback report marks the run degraded and the view


class _LiveLookingLLM:
    """Quacks like a live client; the step only asks is_offline_client of it."""

    def structured(self, **_: object) -> object:  # pragma: no cover - unused
        raise AssertionError("not called")

    def text(self, **_: object) -> str:  # pragma: no cover - unused
        return ""

    def last_usage(self) -> None:
        return None


def _fake_report(*, used_fallback: bool, discards: list[dict[str, str]]) -> AgenticReportResult:
    return AgenticReportResult(
        bundle=ReportBundle.empty(project_id="demo", session_id="run_rep"),
        audit=ReportAudit(status=ReportStatus.VALIDATED),
        evidence_pack=EvidencePack(payload_policy="schema+aggregates"),
        used_fallback=used_fallback,
        narration_discards=discards,
    )


def test_export_step_emits_report_degraded_on_live_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auto_eda,
        "generate_agentic_report",
        lambda *a, **k: _fake_report(
            used_fallback=True,
            discards=[{"section": "Executive Summary", "reason": "llm_error"}],
        ),
    )
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="demo", session_id="run_rep", store=store)
    step = auto_eda.ExportAgenticReportStep(
        [],
        business_context="",
        llm=_LiveLookingLLM(),
        payload_policy="schema+aggregates",
    )
    step.run(ctx)

    events = store.list_trace_events(project_id="demo", session_id="run_rep")
    degraded = next(e for e in events if e.event_type == "report_degraded")
    assert degraded.summary["degraded"] is True
    assert "deterministically" in str(degraded.summary["reason"])
    discarded = next(e for e in events if e.event_type == "narration_discarded")
    assert discarded.summary["discard_count"] == 1
    assert discarded.summary["discards"] == [
        {"section": "Executive Summary", "reason": "llm_error"}
    ]


def test_export_step_stays_quiet_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        auto_eda,
        "generate_agentic_report",
        lambda *a, **k: _fake_report(used_fallback=False, discards=[]),
    )
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="demo", session_id="run_rep", store=store)
    step = auto_eda.ExportAgenticReportStep(
        [], business_context="", llm=_LiveLookingLLM(), payload_policy="schema+aggregates"
    )
    step.run(ctx)
    events = store.list_trace_events(project_id="demo", session_id="run_rep")
    assert not [e for e in events if e.event_type == "report_degraded"]
    assert not [e for e in events if e.event_type == "narration_discarded"]


def test_report_view_carries_the_deterministic_fallback_note(
    store: ArtifactStore,
) -> None:
    store.start_session("demo", "run_view")
    note = "Deterministic fallback (LLM unavailable or repeatedly invalid)."
    store.save_artifact(
        Artifact(
            id="art_audit_1",
            type=ArtifactType.REPORT_AUDIT,
            project_id="demo",
            session_id="run_view",
            payload={"status": "validated", "semantic_notes": [note]},
        )
    )
    store.save_artifact(
        Artifact(
            id="art_report_1",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id="demo",
            session_id="run_view",
            payload={"markdown": "# Report\nbody"},
        )
    )
    view = ReportService(store).get_report("run_view")
    assert isinstance(view, ReportView)
    assert view.degraded is True
    assert view.degraded_reason == note


def test_report_view_is_not_degraded_without_the_note(store: ArtifactStore) -> None:
    store.start_session("demo", "run_view2")
    store.save_artifact(
        Artifact(
            id="art_audit_2",
            type=ArtifactType.REPORT_AUDIT,
            project_id="demo",
            session_id="run_view2",
            payload={"status": "validated", "semantic_notes": ["all good"]},
        )
    )
    store.save_artifact(
        Artifact(
            id="art_report_2",
            type=ArtifactType.MARKDOWN_REPORT,
            project_id="demo",
            session_id="run_view2",
            payload={"markdown": "# Report\nbody"},
        )
    )
    view = ReportService(store).get_report("run_view2")
    assert view.degraded is False
    assert view.degraded_reason is None


# ---------------------------------------------------------------------------
# Narrator discard observability


class _RejectingNarratorLLM:
    def structured(self, **_: object) -> object:
        raise RuntimeError("provider down")

    def text(self, **_: object) -> str:  # pragma: no cover - unused
        return ""

    def last_usage(self) -> None:
        return None


def test_narrator_records_why_each_draft_was_discarded() -> None:
    bundle = ReportBundle.empty(project_id="demo", session_id="run_narr")
    bundle.sections = [
        ReportSection(
            title="Findings",
            claims=[
                ReportClaim(id="c1", text="Revenue rose 10%."),
                ReportClaim(id="c2", text="Orders fell 3%."),
            ],
        )
    ]
    outcome = narrate_report(bundle, llm=_RejectingNarratorLLM())
    assert isinstance(outcome, NarrationOutcome)
    assert outcome.rejected == 1
    assert outcome.discards == [{"section": "Findings", "reason": "llm_error"}]
    # Behavior unchanged: the section keeps its bullets, no narrative written.
    assert bundle.sections[0].narrative == ""

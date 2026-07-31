from __future__ import annotations

import sqlite3
import threading
from typing import ClassVar

import pytest

import eda_platform.core.kernel as kernel_module
from eda_platform.core.budget import BudgetExceeded, SessionBudgetExceeded, SessionBudgetPolicy
from eda_platform.core.kernel import SessionContext, StepContractError, run_pipeline
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace_correlation import trace_job_scope
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class FakeStep:
    name: ClassVar[str] = "fake_step"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SESSION_SUMMARY,)

    def __init__(self, artifact_id: str, calls: list[str]) -> None:
        self.artifact_id = artifact_id
        self.calls = calls

    def run(self, ctx: SessionContext) -> list[Artifact]:
        self.calls.append(self.artifact_id)
        return [
            Artifact(
                id=self.artifact_id,
                type=ArtifactType.SESSION_SUMMARY,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                payload={"step": self.name, "artifact": self.artifact_id},
            )
        ]


def test_pipeline_runs_steps_in_order_and_saves_artifacts(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_demo", store=store)
    calls: list[str] = []

    result = run_pipeline(
        [FakeStep("summary_a", calls), FakeStep("summary_b", calls)],
        ctx,
    )

    assert calls == ["summary_a", "summary_b"]
    assert [artifact.id for artifact in result.artifacts] == ["summary_a", "summary_b"]
    assert store.get_artifact("summary_a").payload["artifact"] == "summary_a"


def test_parallel_pipeline_computes_concurrently_but_commits_in_order(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_parallel", store=store)
    barrier = threading.Barrier(2, timeout=2)

    class ParallelStep(FakeStep):
        parallel_safe: ClassVar[bool] = True

        def run(self, ctx: SessionContext) -> list[Artifact]:
            barrier.wait()
            return super().run(ctx)

    result = run_pipeline(
        [ParallelStep("summary_a", []), ParallelStep("summary_b", [])],
        ctx,
        max_workers=2,
    )

    assert [artifact.id for artifact in result.artifacts] == ["summary_a", "summary_b"]
    events = store.list_trace_events(project_id="project_demo", session_id="run_parallel")
    assert [event.event_type for event in events] == [
        "step_started",
        "step_started",
        "step_completed",
        "step_completed",
    ]


def test_parallel_pipeline_rejects_steps_without_explicit_safety_marker(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_unsafe", store=store)

    with pytest.raises(ValueError, match="parallel_safe=True"):
        run_pipeline(
            [FakeStep("summary_a", []), FakeStep("summary_b", [])],
            ctx,
            max_workers=2,
        )


def test_pipeline_records_step_started_before_completion(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_trace", store=store)

    run_pipeline([FakeStep("summary_a", [])], ctx)

    events = store.list_trace_events(project_id="project_demo", session_id="run_trace")
    assert [(event.event_type, event.name) for event in events] == [
        ("step_started", "fake_step"),
        ("step_completed", "fake_step"),
    ]


def test_pipeline_emits_trace_callback_for_live_debugging(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    seen: list[tuple[str, str]] = []
    ctx = SessionContext(
        project_id="project_demo",
        session_id="run_live",
        store=store,
        on_trace_event=lambda event: seen.append((event.event_type, event.name)),
    )

    run_pipeline([FakeStep("summary_a", [])], ctx)

    assert seen == [
        ("step_started", "fake_step"),
        ("step_completed", "fake_step"),
    ]


def test_pipeline_skips_completed_checkpoint_on_rerun(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    first_ctx = SessionContext(project_id="project_demo", session_id="run_demo", store=store)
    first_calls: list[str] = []
    run_pipeline([FakeStep("summary_a", first_calls)], first_ctx)

    second_ctx = SessionContext(project_id="project_demo", session_id="run_demo", store=store)
    second_calls: list[str] = []
    result = run_pipeline([FakeStep("summary_a", second_calls)], second_ctx)

    assert second_calls == []
    assert result.skipped_steps == ["fake_step"]


def test_budget_raises_when_elapsed_time_is_exhausted(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(
        project_id="project_demo",
        session_id="run_demo",
        store=store,
        max_seconds=0,
    )

    with pytest.raises(BudgetExceeded):
        run_pipeline([FakeStep("summary_a", [])], ctx)


def test_run_context_bridges_legacy_and_explicit_run_budget_limits(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    legacy = SessionContext(
        project_id="project_demo",
        session_id="run_legacy_budget",
        store=store,
        max_tokens=10,
    )
    legacy.session_budget.reserve("call_1", input_tokens=3, output_tokens=3)
    legacy.session_budget.settle(
        "call_1",
        input_tokens=3,
        output_tokens=3,
        total_tokens=6,
        cost_usd=0,
    )
    with pytest.raises(SessionBudgetExceeded):
        legacy.session_budget.reserve("call_2", input_tokens=3, output_tokens=2)

    explicit = SessionContext(
        project_id="project_demo",
        session_id="run_explicit_budget",
        store=store,
        budget_policy=SessionBudgetPolicy(max_requests=0),
    )
    with pytest.raises(SessionBudgetExceeded) as raised:
        explicit.session_budget.reserve("call_1")
    assert raised.value.dimension == "requests"

    wall_limited = SessionContext(
        project_id="project_demo",
        session_id="run_wall_budget",
        store=store,
        budget_policy=SessionBudgetPolicy(max_wall_seconds=0),
    )
    with pytest.raises(SessionBudgetExceeded) as wall_raised:
        run_pipeline([FakeStep("summary_wall", [])], wall_limited)
    assert wall_raised.value.dimension == "wall_seconds"


class ExplodingStep:
    name: ClassVar[str] = "exploding_step"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SESSION_SUMMARY,)

    def run(self, ctx: SessionContext) -> list[Artifact]:
        raise ValueError("boom")


def test_failed_step_marks_run_and_records_trace(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_fail", store=store)

    with pytest.raises(ValueError, match="boom"):
        run_pipeline([ExplodingStep()], ctx)

    assert store.get_session_status("run_fail") == "failed"
    trace_lines = (
        (store.session_dir("project_demo", "run_fail") / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert any('"step_failed"' in line for line in trace_lines)


@pytest.mark.parametrize(
    "fault_point",
    [
        "checkpoint_read",
        "step_started_trace",
        "artifact_save",
        "checkpoint_write",
        "step_completed_trace",
    ],
)
def test_complete_step_boundary_records_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id=f"run_{fault_point}", store=store)
    expected = OSError(f"{fault_point} unavailable")

    if fault_point == "checkpoint_read":
        monkeypatch.setattr(
            kernel_module,
            "_read_checkpoint",
            lambda _path: (_ for _ in ()).throw(expected),
        )
    elif fault_point == "artifact_save":
        monkeypatch.setattr(
            store,
            "save_artifact",
            lambda _artifact: (_ for _ in ()).throw(expected),
        )
    elif fault_point == "checkpoint_write":
        monkeypatch.setattr(
            kernel_module,
            "_write_checkpoint",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(expected),
        )
    else:
        original_emit = ctx.emit_trace
        target_type = (
            "step_started" if fault_point == "step_started_trace" else "step_completed"
        )

        def fail_selected_trace(event) -> None:
            if event.event_type == target_type:
                raise expected
            original_emit(event)

        monkeypatch.setattr(ctx, "emit_trace", fail_selected_trace)

    with pytest.raises(OSError, match=fault_point):
        run_pipeline([FakeStep("summary_fault", [])], ctx)

    assert store.get_session_status(ctx.session_id) == "failed"
    events = store.list_trace_events(project_id=ctx.project_id, session_id=ctx.session_id)
    failures = [event for event in events if event.event_type == "step_failed"]
    assert len(failures) == 1
    assert failures[0].name == "fake_step"
    assert failures[0].summary["error_type"] == "OSError"


def test_failure_reporting_failure_uses_minimal_durable_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="project_demo", session_id="run_reporting_fault", store=store)
    original = RuntimeError("primary step failure")

    class PrimaryFailure(FakeStep):
        def run(self, ctx: SessionContext) -> list[Artifact]:
            raise original

    monkeypatch.setattr(
        store,
        "mark_session_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("status reporter failed")),
    )
    original_emit = ctx.emit_trace

    def fail_failure_trace(event) -> None:
        if event.event_type == "step_failed":
            raise OSError("trace reporter failed")
        original_emit(event)

    monkeypatch.setattr(ctx, "emit_trace", fail_failure_trace)

    with pytest.raises(RuntimeError, match="primary step failure") as raised:
        run_pipeline([PrimaryFailure("unused", [])], ctx)
    assert raised.value is original

    with sqlite3.connect(store.db_path) as conn:
        status = conn.execute(
            "select status from sessions where session_id = ?", (ctx.session_id,)
        ).fetchone()
        rows = conn.execute(
            """
            select event_type, name, payload from trace_events
            where project_id = ? and session_id = ? and event_type = 'step_failed'
            """,
            (ctx.project_id, ctx.session_id),
        ).fetchall()
    assert status == ("failed",)
    assert len(rows) == 1
    assert rows[0][1] == "fake_step"
    assert "primary step failure" in rows[0][2]


def test_failure_callback_after_durable_append_does_not_duplicate_fallback(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path)

    def callback(event) -> None:
        if event.event_type == "step_failed":
            raise OSError("observer failed after durable append")

    ctx = SessionContext(
        project_id="project_demo",
        session_id="run_callback_failure",
        store=store,
        on_trace_event=callback,
    )

    with pytest.raises(ValueError, match="boom"):
        run_pipeline([ExplodingStep()], ctx)

    with sqlite3.connect(store.db_path) as conn:
        failures = conn.execute(
            """
            select event_key, count(*) from trace_events
            where project_id = ? and session_id = ? and event_type = 'step_failed'
            group by event_key
            """,
            (ctx.project_id, ctx.session_id),
        ).fetchall()
    assert len(failures) == 1
    assert failures[0][0].startswith("kernel-step-failed:")
    assert failures[0][1] == 1


def test_kernel_failure_fallback_keeps_job_correlation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)
    with trace_job_scope("job_kernel_failure", 4):
        ctx = SessionContext(
            project_id="project_demo",
            session_id="run_job_failure",
            store=store,
        )
    monkeypatch.setattr(
        store,
        "mark_session_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("status failed")),
    )
    original_emit = ctx.emit_trace

    def fail_failure_trace(event) -> None:
        if event.event_type == "step_failed":
            raise OSError("trace failed")
        original_emit(event)

    monkeypatch.setattr(ctx, "emit_trace", fail_failure_trace)
    with pytest.raises(ValueError, match="boom"):
        run_pipeline([ExplodingStep()], ctx)

    with sqlite3.connect(store.db_path) as conn:
        correlation = conn.execute(
            """
            select job_id, job_generation from trace_events
            where session_id = ? and event_type = 'step_failed'
            """,
            (ctx.session_id,),
        ).fetchone()
    assert correlation == ("job_kernel_failure", 4)


def test_checkpoint_reruns_when_cache_key_changes(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    class KeyedStep:
        name: ClassVar[str] = "keyed_step"
        requires: ClassVar[tuple[ArtifactType, ...]] = ()
        produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SESSION_SUMMARY,)

        def __init__(self, key: str, calls: list[str]) -> None:
            self.key = key
            self.calls = calls

        def cache_key(self, ctx: SessionContext) -> str:
            return self.key

        def run(self, ctx: SessionContext) -> list[Artifact]:
            self.calls.append(self.key)
            return [
                Artifact(
                    id=f"summary_{self.key}",
                    type=ArtifactType.SESSION_SUMMARY,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    payload={"key": self.key},
                )
            ]

    calls: list[str] = []
    ctx1 = SessionContext(project_id="p", session_id="run_keyed", store=store)
    run_pipeline([KeyedStep("v1", calls)], ctx1)
    ctx2 = SessionContext(project_id="p", session_id="run_keyed", store=store)
    run_pipeline([KeyedStep("v2", calls)], ctx2)

    assert calls == ["v1", "v2"]  # changed cache key forces a rerun


def test_pipeline_rejects_missing_declared_input_before_running_step(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="p", session_id="run_contract", store=store)
    calls: list[str] = []

    class NeedsProfile(FakeStep):
        requires: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)

    with pytest.raises(StepContractError, match="missing_required_types"):
        run_pipeline([NeedsProfile("summary_a", calls)], ctx)

    assert calls == []
    assert store.get_session_status("run_contract") == "failed"
    events = store.list_trace_events(project_id="p", session_id="run_contract")
    assert [event.event_type for event in events] == [
        "step_started",
        "step_contract_violation",
        "step_failed",
    ]


def test_pipeline_rejects_unexpected_output_without_persisting_it(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="p", session_id="run_output_contract", store=store)

    class WrongOutput(FakeStep):
        produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.DATASET_PROFILE,)

    with pytest.raises(StepContractError, match="unexpected_types"):
        run_pipeline([WrongOutput("summary_wrong", [])], ctx)

    with pytest.raises(KeyError):
        store.get_artifact("summary_wrong")


def test_pipeline_rejects_artifact_bound_to_another_run(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="p", session_id="run_expected", store=store)

    class WrongRun(FakeStep):
        def run(self, ctx: SessionContext) -> list[Artifact]:
            return [
                Artifact(
                    id="summary_wrong_run",
                    type=ArtifactType.SESSION_SUMMARY,
                    project_id=ctx.project_id,
                    session_id="run_other",
                    payload={},
                )
            ]

    with pytest.raises(StepContractError, match="misbound_artifact_ids"):
        run_pipeline([WrongRun("unused", [])], ctx)


def test_pipeline_recovers_from_checkpoint_with_missing_artifact(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ctx = SessionContext(project_id="p", session_id="run_recover", store=store)
    run_pipeline([FakeStep("summary_missing", [])], ctx)
    store.artifact_path("p", "run_recover", "summary_missing").unlink()
    calls: list[str] = []

    result = run_pipeline([FakeStep("summary_rebuilt", calls)], ctx)

    assert calls == ["summary_rebuilt"]
    assert [artifact.id for artifact in result.artifacts] == ["summary_rebuilt"]
    events = store.list_trace_events(project_id="p", session_id="run_recover")
    assert any(event.event_type == "checkpoint_invalid" for event in events)

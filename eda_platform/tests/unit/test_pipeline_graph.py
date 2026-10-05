from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest

from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.kernel import SessionCancelled, SessionContext, run_pipeline
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType


class RecoverableStep:
    name: ClassVar[str] = "recoverable"
    requires: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SESSION_SUMMARY,)
    parallel_safe: ClassVar[bool] = True

    def __init__(self, key: str, calls: list[str], *, fail: bool = False) -> None:
        self.key = key
        self.calls = calls
        self.fail = fail

    def cache_key(self, ctx: SessionContext) -> str:
        return self.key

    def run(self, ctx: SessionContext) -> list[Artifact]:
        self.calls.append(self.key)
        if self.fail:
            raise RuntimeError("computation failed")
        return [
            Artifact(
                id=self.key,
                type=ArtifactType.SESSION_SUMMARY,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                payload={},
            )
        ]


def context(root: Path, **kwargs) -> SessionContext:
    return SessionContext(project_id="p", session_id="run", store=ArtifactStore(root), **kwargs)


def test_parallel_recovery_reuses_successful_sibling_pending_writes(tmp_path: Path) -> None:
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="computation failed"):
        run_pipeline(
            [RecoverableStep("first", calls), RecoverableStep("second", calls, fail=True)],
            context(tmp_path),
            max_workers=2,
        )
    result = run_pipeline(
        [RecoverableStep("first", calls), RecoverableStep("second", calls)],
        context(tmp_path),
        max_workers=2,
    )
    assert calls.count("first") == 1
    assert calls.count("second") == 2
    assert [a.id for a in result.artifacts] == ["first", "second"]


@pytest.mark.parametrize("boundary", ["compute", "commit"])
@pytest.mark.parametrize("stop", ["cancel", "budget"])
def test_restored_nodes_enforce_current_cancellation_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    stop: str,
) -> None:
    calls: list[str] = []
    first = context(tmp_path)
    with monkeypatch.context() as patch:
        if boundary == "commit":

            def fail_save(_artifacts: list[Artifact], **_kwargs) -> None:
                raise RuntimeError("commit failed")

            patch.setattr(first.store, "save_artifacts", fail_save)
        with pytest.raises(RuntimeError, match="failed"):
            run_pipeline([RecoverableStep("first", calls, fail=boundary == "compute")], first)

    restored = context(
        tmp_path,
        cancel_check=lambda: stop == "cancel",
        max_seconds=0 if stop == "budget" else None,
    )
    with pytest.raises(SessionCancelled if stop == "cancel" else BudgetExceeded):
        run_pipeline([RecoverableStep("first", calls)], restored)
    assert calls == ["first"]
    assert restored.store.list_artifacts(project_id="p", session_id="run") == []
    if stop == "budget":
        failures = [
            event
            for event in restored.store.list_trace_events(
                project_id="p",
                session_id="run",
            )
            if event.event_type == "step_failed"
        ]
        assert failures[-1].summary["error_type"] == "BudgetExceeded"

    result = run_pipeline([RecoverableStep("first", calls)], context(tmp_path))
    assert [a.id for a in result.artifacts] == ["first"]
    assert calls.count("first") == (2 if boundary == "compute" else 1)


def test_fanout_never_exceeds_two_and_commit_order_is_stable(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class MeasuredStep(RecoverableStep):
        def run(self, ctx: SessionContext) -> list[Artifact]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return super().run(ctx)
            finally:
                with lock:
                    active -= 1

    names = [f"step_{i}" for i in range(5)]
    ctx = context(tmp_path)
    result = run_pipeline([MeasuredStep(name, []) for name in names], ctx, max_workers=10)
    assert peak == 2
    assert [artifact.id for artifact in result.artifacts] == names
    assert [
        event.summary["artifact_count"]
        for event in ctx.store.list_trace_events(
            project_id="p",
            session_id="run",
        )
        if event.event_type == "step_completed"
    ] == [1] * 5


_CRASH_SCRIPT = """\
import os
import sys
from pathlib import Path
from langgraph.cache.sqlite import SqliteCache
from eda_platform.core.kernel import SessionContext, run_pipeline
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType

root, boundary = Path(sys.argv[1]), sys.argv[2]
class Step:
    name = "crash_step"
    requires = ()
    produces = (ArtifactType.SESSION_SUMMARY,)
    def run(self, ctx):
        with (root / "calls").open("a") as file:
            file.write("compute\\n")
        return [Artifact(id="result", type=ArtifactType.SESSION_SUMMARY,
                         project_id="p", session_id="run", payload={"answer": 42})]

store = ArtifactStore(root)
ctx = SessionContext(project_id="p", session_id="run", store=store)
if boundary == "artifact":
    original = store.save_artifacts
    def save(artifacts, **kwargs):
        original(artifacts, **kwargs)
        os._exit(77)
    store.save_artifacts = save
elif boundary == "cache":
    original = SqliteCache.set
    def save(self, pairs):
        original(self, pairs)
        os._exit(77)
    SqliteCache.set = save
elif boundary == "trace":
    def callback(event):
        if event.event_type == "step_completed":
            os._exit(77)
    ctx.on_trace_event = callback
result = run_pipeline([Step()], ctx)
assert result.artifacts[0].payload == {"answer": 42}
"""


@pytest.mark.parametrize("boundary", ["artifact", "cache", "trace"])
def test_process_exit_during_commit_reopens_sqlite_without_recomputing_or_duplicate_events(
    tmp_path: Path,
    boundary: str,
) -> None:
    script = tmp_path / "crash.py"
    script.write_text(_CRASH_SCRIPT, encoding="utf-8")
    root = tmp_path / "workspace"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    command = [sys.executable, str(script), str(root)]
    crashed = subprocess.run([*command, boundary], env=env, capture_output=True, timeout=30)
    assert crashed.returncode == 77, crashed.stderr.decode()
    restored = subprocess.run([*command, "resume"], env=env, capture_output=True, timeout=30)
    assert restored.returncode == 0, restored.stderr.decode()
    assert (root / "calls").read_text() == "compute\n"
    store = ArtifactStore(root)
    events = store.list_trace_events(project_id="p", session_id="run")
    completed = [event for event in events if event.event_type == "step_completed"]
    assert len(completed) == 1
    assert completed[0].event_key is not None
    assert len(store.list_artifacts(project_id="p", session_id="run")) == 1


def test_resumed_all_cache_hit_batch_still_honors_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eda_platform.core.pipeline_graph as pipeline

    calls: list[str] = []
    steps = (RecoverableStep("first", calls),)
    pipeline.run_pipeline_graph(steps, context(tmp_path), width=1)
    with monkeypatch.context() as patch:

        def fail_commit(state, runtime):
            raise RuntimeError("before empty commit")

        patch.setattr(pipeline, "_commit", fail_commit)
        with pytest.raises(RuntimeError, match="before empty commit"):
            pipeline.run_pipeline_graph(steps, context(tmp_path), width=2)
    with pytest.raises(SessionCancelled):
        pipeline.run_pipeline_graph(steps, context(tmp_path, cancel_check=lambda: True), width=2)
    assert calls == ["first"]


def test_same_id_artifact_content_drift_invalidates_checkpoint_and_step_cache(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    first = context(tmp_path)
    run_pipeline([RecoverableStep("first", calls)], first)
    changed = first.store.get_artifact("first", project_id="p", session_id="run")
    changed.payload = {"modified": "after checkpoint"}
    first.store.save_artifact(changed)

    result = run_pipeline([RecoverableStep("first", calls)], context(tmp_path))
    assert calls == ["first", "first"]
    assert result.artifacts[0].payload == {}
    assert result.skipped_steps == []
    events = first.store.list_trace_events(project_id="p", session_id="run")
    assert any(event.event_type == "checkpoint_invalid" for event in events)


def test_equivalent_artifact_recreation_does_not_invalidate_content_cache(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    calls: list[str] = []
    ctx = context(tmp_path)
    run_pipeline([RecoverableStep("first", calls)], ctx)
    artifact = ctx.store.get_artifact("first", project_id="p", session_id="run")
    artifact.created_at = datetime.now(UTC) + timedelta(seconds=10)
    ctx.store.save_artifact(artifact)
    result = run_pipeline([RecoverableStep("first", calls)], context(tmp_path))
    assert calls == ["first"]
    assert result.skipped_steps == ["recoverable"]


def test_same_id_input_change_invalidates_pipeline_but_timestamp_does_not(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from eda_platform.drivers.auto_eda import ScanQualityStep
    from eda_platform.schemas.artifacts import DatasetProfile

    ctx = context(tmp_path)
    profile = DatasetProfile(
        dataset_id="d", name="data", rows=100, columns=1, column_names=["value"],
        dtypes={"value": "float64"}, missing_values={"value": 0},
        missing_percent={"value": 0.0}, numeric_columns=["value"], categorical_columns=[],
    )
    source = Artifact(
        id="fixed_profile", type=ArtifactType.DATASET_PROFILE,
        project_id="p", session_id="run", payload=profile.model_dump(mode="json"),
    )
    ctx.store.save_artifact(source)
    run_pipeline([ScanQualityStep(source.id)], ctx)
    source.created_at = datetime.now(UTC) + timedelta(seconds=1)
    ctx.store.save_artifact(source)
    assert run_pipeline([ScanQualityStep(source.id)], ctx).skipped_steps == ["scan_quality"]
    source.payload["missing_percent"]["value"] = 75.0
    source.payload["missing_values"]["value"] = 75
    ctx.store.save_artifact(source)
    refreshed = run_pipeline([ScanQualityStep(source.id)], ctx)
    assert refreshed.skipped_steps == []
    assert refreshed.artifacts[0].payload == ScanQualityStep(source.id).run(ctx)[0].payload
    assert refreshed.artifacts[0].payload["issues"][0]["code"] == "high_missing"

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# The whole module needs the optional [observability] extra; environments
# without it (e.g. CI's dev-only install) skip these tests at collection.
pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    InMemorySpanExporter,
)

from eda_platform.core import observability  # noqa: E402
from eda_platform.core.observability import (  # noqa: E402
    ObservabilityConfig,
    flush_run,
    mirror_trace_event,
)
from eda_platform.core.store import ArtifactStore  # noqa: E402
from eda_platform.schemas.sessions import TraceEvent  # noqa: E402

_T0 = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_observability_state() -> Iterator[None]:
    observability.reset()
    yield
    observability.reset()


def _event(event_type: str, name: str, *, session_id: str = "run_x", **summary: object) -> TraceEvent:
    return TraceEvent(
        session_id=session_id,
        event_type=event_type,
        name=name,
        started_at=_T0,
        finished_at=_T0 + timedelta(milliseconds=5),
        summary=dict(summary),
    )


# --- disabled default: zero-cost, no spans ---------------------------------


def test_config_from_env_defaults_to_disabled() -> None:
    assert ObservabilityConfig.from_env({}).enabled is False
    assert ObservabilityConfig.from_env({"EDA_OBSERVABILITY": "1"}).enabled is True


def test_disabled_mirror_is_noop() -> None:
    # No exporter forced and env not enabled → disabled, builds nothing.
    assert mirror_trace_event(_event("llm_call", "m3_build_plan")) == "disabled"
    assert observability._STATE is None


def test_import_survives_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if the OTel SDK were absent, resolving to enabled must degrade to a
    # no-op exporter rather than raising into the trace-persistence path.
    monkeypatch.setattr(observability, "_build_exporter", lambda config: None)
    monkeypatch.setattr(observability, "_env_config", lambda: ObservabilityConfig(enabled=True))
    assert mirror_trace_event(_event("step", "profile")) == "disabled"


# --- enabled: real span tree via an in-memory exporter ---------------------


def test_enabled_mirror_builds_openinference_span_tree() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    assert mirror_trace_event(_event("step", "profile_dataset")) == "mirrored"
    assert (
        mirror_trace_event(
            _event(
                "llm_call",
                "m3_build_plan",
                model="deepseek-chat",
                total_tokens=1234,
                prompt_tokens=1000,
                completion_tokens=234,
            )
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert "run:run_x" in spans
    assert "profile_dataset" in spans
    assert "m3_build_plan" in spans

    root = spans["run:run_x"]
    llm = spans["m3_build_plan"]
    # Child events are parented under the per-run root span.
    assert llm.parent is not None
    assert root.context is not None
    assert llm.parent.span_id == root.context.span_id
    # LLM events carry OpenInference LLM conventions pulled from the summary.
    llm_attrs = dict(llm.attributes or {})
    assert llm_attrs["openinference.span.kind"] == "LLM"
    assert llm_attrs["llm.model_name"] == "deepseek-chat"
    assert llm_attrs["llm.token_count.total"] == 1234
    assert dict(spans["profile_dataset"].attributes or {})["openinference.span.kind"] == "CHAIN"


# --- store hook: append_trace mirrors, source of truth unchanged -----------


def test_append_trace_mirrors_when_enabled(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)
    store = ArtifactStore(tmp_path / "ws")
    store.ensure_project("proj", "Project")
    store.start_session("proj", "run_hook")

    store.append_trace("proj", _event("tool_completed", "run_sql", session_id="run_hook"))
    flush_run("run_hook")

    names = {span.name for span in exporter.get_finished_spans()}
    assert "run:run_hook" in names
    assert "run_sql" in names
    # Source of truth still written and readable.
    events = store.list_trace_events(project_id="proj", session_id="run_hook")
    assert [e.name for e in events] == ["run_sql"]


def test_append_trace_emits_no_spans_when_disabled(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    # Deliberately do NOT configure_for_test → disabled path.
    store = ArtifactStore(tmp_path / "ws")
    store.ensure_project("proj", "Project")
    store.start_session("proj", "run_off")

    store.append_trace("proj", _event("tool_completed", "run_sql", session_id="run_off"))

    assert exporter.get_finished_spans() == ()
    events = store.list_trace_events(project_id="proj", session_id="run_off")
    assert [e.name for e in events] == ["run_sql"]

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from eda_platform.core import observability
from eda_platform.core.observability import (
    ObservabilityConfig,
    flush_run,
    mirror_trace_event,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import TraceEvent

_T0 = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_observability_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("EDA_OBSERVABILITY", raising=False)
    monkeypatch.delenv("EDA_LLM_DEBUG_FULL", raising=False)
    observability.reset()
    yield
    observability.reset()


def _event(
    event_type: str, name: str, *, session_id: str = "run_x", **summary: object
) -> TraceEvent:
    return TraceEvent(
        session_id=session_id,
        event_type=event_type,
        name=name,
        started_at=_T0,
        finished_at=_T0 + timedelta(milliseconds=5),
        summary=dict(summary),
    )


def _attribute_text(spans: Sequence[object]) -> str:
    """Flatten every attribute key/value for content-leak assertions."""
    parts: list[str] = []
    for span in spans:
        for key, value in dict(getattr(span, "attributes", None) or {}).items():
            parts.append(f"{key}={value!r}")
    return "\n".join(parts)


# --- disabled default: zero-cost, no spans ---------------------------------


def test_config_from_env_defaults_to_disabled() -> None:
    assert ObservabilityConfig.from_env({}).enabled is False
    assert ObservabilityConfig.from_env({"EDA_OBSERVABILITY": "1"}).enabled is True


def test_disabled_mirror_is_noop() -> None:
    # No exporter forced and env not enabled → disabled, builds nothing.
    assert mirror_trace_event(_event("llm_call", "m3_build_plan")) == "disabled"
    assert observability._STATE is None


def test_enabled_but_unbuildable_exporter_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "_build_exporter", lambda config: None)
    monkeypatch.setattr(observability, "_env_config", lambda: ObservabilityConfig(enabled=True))
    assert mirror_trace_event(_event("step", "profile")) == "disabled"


def test_mirror_degrades_when_otel_not_installed() -> None:
    # A subprocess with opentelemetry blocked at import proves the module
    # imports and no-ops without the optional dependency.
    code = textwrap.dedent(
        """
        import sys

        class _BlockOtel:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "opentelemetry":
                    raise ImportError("opentelemetry blocked for test")
                return None

        sys.meta_path.insert(0, _BlockOtel())
        from eda_platform.core import observability

        result = observability.mirror_trace_event(
            {
                "session_id": "run_sub",
                "event_type": "llm_call",
                "name": "m3_build_plan",
                "summary": {"model": "deepseek-chat", "prompt": "text"},
            }
        )
        assert result == "disabled", result
        observability.flush_run("run_sub")
        print("OK-no-otel", result)
        """
    )
    src = Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ, PYTHONPATH=str(src), EDA_OBSERVABILITY="1")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK-no-otel disabled" in proc.stdout


# --- enabled: GenAI span mapping via an in-memory exporter ------------------


def test_chat_event_maps_genai_attributes_under_invoke_agent_root() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    assert (
        mirror_trace_event(
            _event(
                "llm_call",
                "m3_build_plan",
                provider="deepseek",
                model="deepseek-chat",
                prompt_tokens=1000,
                completion_tokens=234,
                total_tokens=1234,
                estimated_cost_usd=0.0012,
            )
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"invoke_agent run_x", "chat deepseek-chat"}

    root = spans["invoke_agent run_x"]
    root_attrs = dict(root.attributes or {})
    assert root_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert root_attrs["gen_ai.conversation.id"] == "run_x"
    assert root_attrs["openinference.span.kind"] == "AGENT"

    chat = spans["chat deepseek-chat"]
    assert chat.parent is not None and root.context is not None
    assert chat.parent.span_id == root.context.span_id
    attrs = dict(chat.attributes or {})
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "deepseek-chat"
    assert attrs["gen_ai.provider.name"] == "deepseek"
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["gen_ai.usage.output_tokens"] == 234
    assert attrs["eda.total_tokens"] == 1234
    assert attrs["eda.estimated_cost_usd"] == 0.0012
    assert attrs["eda.task"] == "m3_build_plan"
    assert attrs["openinference.span.kind"] == "LLM"


def test_tool_events_map_execute_tool() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    assert mirror_trace_event(_event("tool_completed", "run_sql", status="success")) == "mirrored"
    assert (
        mirror_trace_event(
            _event("tool_failed", "correlate_columns", status="error: ValueError: cell secret 42")
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    ok = dict(spans["execute_tool run_sql"].attributes or {})
    assert ok["gen_ai.operation.name"] == "execute_tool"
    assert ok["gen_ai.tool.name"] == "run_sql"
    assert ok["eda.status"] == "success"
    assert ok["openinference.span.kind"] == "TOOL"

    failed = dict(spans["execute_tool correlate_columns"].attributes or {})
    # Only the outcome token and exception class survive; the message does not.
    assert failed["eda.status"] == "error"
    assert failed["error.type"] == "ValueError"
    assert "cell secret 42" not in _attribute_text(exporter.get_finished_spans())


def test_usage_settlement_maps_tokens_without_operation_name() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    assert (
        mirror_trace_event(
            _event(
                "llm_usage",
                "m3_build_plan",
                provider="deepseek",
                model="deepseek-chat",
                prompt_tokens=1000,
                completion_tokens=234,
                total_tokens=1234,
                cached_tokens=100,
                estimated_cost_usd=0.0012,
                finish_reason="stop",
            )
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    attrs = dict(spans["m3_build_plan"].attributes or {})
    assert "gen_ai.operation.name" not in attrs  # settlement is not a second chat
    assert attrs["gen_ai.usage.input_tokens"] == 1000
    assert attrs["gen_ai.usage.output_tokens"] == 234
    assert attrs["eda.cached_tokens"] == 100
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["eda.event_type"] == "llm_usage"


def test_evaluation_event_maps_evaluator_span() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    assert (
        mirror_trace_event(
            _event(
                "validator_result",
                "report_gate",
                score=0.87,
                passed=True,
                detail="verbatim finding text with cell values",
            )
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    attrs = dict(spans["evaluation report_gate"].attributes or {})
    assert attrs["openinference.span.kind"] == "EVALUATOR"
    assert attrs["eda.evaluation.name"] == "report_gate"
    assert attrs["eda.evaluation.score"] == 0.87
    assert attrs["eda.evaluation.passed"] is True
    assert "verbatim finding text" not in _attribute_text(exporter.get_finished_spans())


# --- redaction: content never enters spans by default -----------------------


def test_default_redaction_keeps_content_out_of_spans() -> None:
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    prompt = "Summarize patients table -- SECRET_PROMPT_TEXT"
    rows = [{"patient": "Ada Lovelace", "ssn": "078-05-1120"}]
    assert (
        mirror_trace_event(
            _event(
                "llm_call",
                "m3_build_plan",
                model="deepseek-chat",
                prompt=prompt,
                response_text="SECRET_COMPLETION_TEXT",
                rows=rows,
            )
        )
        == "mirrored"
    )
    flush_run("run_x")

    spans = exporter.get_finished_spans()
    text = _attribute_text(spans)
    for secret in ("SECRET_PROMPT_TEXT", "SECRET_COMPLETION_TEXT", "Ada Lovelace", "078-05-1120"):
        assert secret not in text

    attrs = dict({span.name: span for span in spans}["chat deepseek-chat"].attributes or {})
    assert attrs["eda.redacted.prompt.chars"] == len(prompt)
    expected_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    assert attrs["eda.redacted.prompt.sha256"] == expected_digest
    assert "eda.redacted.response_text.sha256" in attrs
    assert "eda.redacted.rows.sha256" in attrs
    assert not any(key.startswith("eda.content.") for key in attrs)


def test_full_capture_opt_in_emits_bounded_excerpt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDA_LLM_DEBUG_FULL", "1")
    exporter = InMemorySpanExporter()
    observability.configure_for_test(exporter)

    prompt = "P" * 3000
    assert mirror_trace_event(_event("llm_call", "task", model="m", prompt=prompt)) == "mirrored"
    flush_run("run_x")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    attrs = dict(spans["chat m"].attributes or {})
    assert attrs["eda.redacted.prompt.chars"] == 3000
    assert attrs["eda.content.prompt"] == "P" * 2000  # excerpt stays bounded


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
    assert "invoke_agent run_hook" in names
    assert "execute_tool run_sql" in names
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

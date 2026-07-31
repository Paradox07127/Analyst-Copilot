"""Dev-log panel core: event formatting, ring buffer, LLM capture (DI dev-log)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from eda_platform.core.dev_log import (
    LLM_DEBUG_FILENAME,
    PREVIEW_CHARS,
    DevLogBuffer,
    InstrumentedLLMClient,
    format_event_line,
    read_llm_debug,
)
from eda_platform.core.llm import LLMResultMetadata, LLMUsage, OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import TraceEvent

_T0 = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _event(
    event_type: str = "llm_call",
    name: str = "m4_question_discovery",
    *,
    duration_s: float | None = 2.5,
    summary: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        session_id="r",
        event_type=event_type,
        name=name,
        started_at=_T0,
        finished_at=None if duration_s is None else _T0 + timedelta(seconds=duration_s),
        summary=summary or {},
    )


# --- formatting -------------------------------------------------------------


def test_format_event_line_fixed_columns_with_metrics_and_duration() -> None:
    line = format_event_line(
        _event(summary={"total_tokens": 1240, "estimated_cost_usd": 0.004, "noise": "x"})
    )
    assert line.startswith("12:00:00.000  llm_call")
    assert "total_tokens=1240" in line
    assert "cost=$0.004" in line
    assert line.endswith("[2.50s]")
    assert "noise" not in line  # non-metric summary keys stay in trace.jsonl


def test_format_event_line_without_finish_has_no_duration() -> None:
    line = format_event_line(_event("step_started", "profile", duration_s=None))
    assert "[" not in line


# --- ring buffer + throttle -------------------------------------------------


def test_buffer_truncates_to_max_lines_and_throttles(monkeypatch: pytest.MonkeyPatch) -> None:
    # A small monotonic value models a freshly started process.  The first
    # event must flush by contract, regardless of host uptime.
    monkeypatch.setattr("eda_platform.core.dev_log.time.monotonic", lambda: 1.0)
    buffer = DevLogBuffer(max_lines=5, flush_every=3, flush_interval=999.0)
    flushes = [buffer.add(_event(name=f"step_{i}")) for i in range(7)]
    # The very first add always flushes (the panel shows something immediately);
    # after that only every 3rd add does — the huge interval disables the
    # time-based path for the rest of the test.
    assert flushes[0] is True
    assert flushes.count(True) == 3
    assert len(buffer.lines) == 5
    assert "step_6" in buffer.render()
    assert "step_0" not in buffer.render()


# --- instrumented client ----------------------------------------------------


class _Out(BaseModel):
    answer: str


_T = TypeVar("_T")


class _FakeLLM:
    """Mimics the real client's metering contract: ``last_usage`` starts None
    and each SUCCESSFUL call assigns a fresh metadata object (failed calls
    leave the marker untouched — the wrapper relies on that identity)."""

    settings = None

    def __init__(self) -> None:
        self._usage: LLMResultMetadata | None = None

    def _meter(self) -> None:
        self._usage = LLMResultMetadata(
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=LLMUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120, cached_tokens=30
            ),
            estimated_cost_usd=0.001,
        )

    def structured(self, *, task: str, schema: type[_T], payload: dict) -> _T:
        self._meter()
        return cast("_T", _Out(answer="ok"))

    def text(self, *, task: str, payload: dict) -> str:
        self._meter()
        return "text-answer"

    def last_usage(self) -> LLMResultMetadata | None:
        return self._usage


class _FlakyLLM(_FakeLLM):
    """One good call, then failures that do NOT touch the usage marker."""

    def text(self, *, task: str, payload: dict) -> str:
        if self._usage is not None:
            raise RuntimeError("boom")
        return super().text(task=task, payload=payload)


def test_instrumented_client_records_structured_call(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    session_dir = store.start_session("demo", "run")
    client = InstrumentedLLMClient(_FakeLLM(), session_dir=session_dir)
    out = client.structured(task="m4", schema=_Out, payload={"question": "q"})

    assert out.answer == "ok"
    assert len(client.records) == 1
    record = client.records[0]
    assert record["task"] == "m4"
    assert record["status"] == "success"
    assert record["total_tokens"] == 120
    # Prompt-cache metering flows through to the persisted record (Phase 0).
    assert record["cached_tokens"] == 30
    assert record["cache_hit_rate"] == 0.3
    assert record["model"] == "deepseek-v4-flash"
    assert "question" in record["payload_preview"]
    assert "ok" in record["response_preview"]
    # Persisted to llm_debug.jsonl (dir auto-created).
    on_disk = read_llm_debug(session_dir)
    assert len(on_disk) == 1
    assert on_disk[0]["task"] == "m4"


def test_instrumented_client_truncates_large_payload_previews(tmp_path: Path) -> None:
    client = InstrumentedLLMClient(_FakeLLM(), session_dir=None)
    client.text(task="t", payload={"blob": "x" * (PREVIEW_CHARS * 2)})

    preview = client.records[0]["payload_preview"]
    assert len(preview) < PREVIEW_CHARS * 2
    assert "truncated" in preview


def test_instrumented_client_records_errors_and_reraises() -> None:
    client = InstrumentedLLMClient(OfflineLLMClient(), session_dir=None)
    with pytest.raises(RuntimeError):
        client.structured(task="m4", schema=_Out, payload={})
    assert client.records[0]["status"].startswith("error: RuntimeError")


def test_error_record_does_not_inherit_previous_calls_usage() -> None:
    # Providers only refresh last_usage() on success; a failed call's record
    # must not carry the PREVIOUS call's tokens/cost/model.
    client = InstrumentedLLMClient(_FlakyLLM(), session_dir=None)
    client.text(task="ok", payload={})
    with pytest.raises(RuntimeError):
        client.text(task="fails", payload={})
    ok_record, error_record = client.records
    assert ok_record["total_tokens"] == 120
    assert error_record["status"].startswith("error: RuntimeError")
    assert "total_tokens" not in error_record
    assert "model" not in error_record


def test_read_llm_debug_tolerates_missing_and_partial_files(tmp_path: Path) -> None:
    assert read_llm_debug(tmp_path) == []
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    (session_dir / LLM_DEBUG_FILENAME).write_text(
        json.dumps({"task": "a"}) + "\nnot-json\n", encoding="utf-8"
    )
    records = read_llm_debug(session_dir)
    assert len(records) == 1 and records[0]["task"] == "a"

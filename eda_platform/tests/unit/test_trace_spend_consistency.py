"""C2: the Trace page must show ONE spend story.

SessionMetrics already prefers the authoritative ``llm_usage`` ledger over the
incomplete per-driver ``llm_call`` events; the trace UI caption and the debug
tables must use the same selection, or the same page shows two contradicting
call/token counts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.debug import llm_call_rows

_T0 = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)
_PROJECT = "p"
_RUN = "r"


def _event(event_type: str, name: str, *, offset_s: float, summary: dict) -> TraceEvent:
    started = _T0 + timedelta(seconds=offset_s)
    return TraceEvent(
        session_id=_RUN,
        event_type=event_type,
        name=name,
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        summary=summary,
    )


def _mixed_ledger_events() -> list[TraceEvent]:
    """3 authoritative llm_usage events (100 tok each) + 1 stale llm_call (42518 tok)."""
    events = [
        _event(
            "llm_usage",
            f"task_{i}",
            offset_s=float(i),
            summary={
                "task": f"task_{i}",
                "status": "success",
                "provider": "fake",
                "model": "fake-1",
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
                "estimated_cost_usd": 0.01,
            },
        )
        for i in range(3)
    ]
    events.append(
        _event(
            "llm_call",
            "narrative_only",
            offset_s=10.0,
            summary={"total_tokens": 42_518, "estimated_cost_usd": 0.5},
        )
    )
    return events


def _store_with(tmp_path: Path, events: list[TraceEvent]) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(_PROJECT, name=_PROJECT)
    store.start_session(_PROJECT, _RUN)
    for event in events:
        store.append_trace(_PROJECT, event)
    return store


def test_session_metrics_use_ledger_when_present(tmp_path: Path) -> None:
    events = _mixed_ledger_events()
    store = _store_with(tmp_path, events)
    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.llm_calls == 3
    assert metrics.total_tokens == 300


def test_llm_call_table_lists_ledger_rows_when_present() -> None:
    rows = llm_call_rows(_mixed_ledger_events())
    assert len(rows) == 3
    assert {row["task"] for row in rows} == {"task_0", "task_1", "task_2"}
    assert all(row["total_tokens"] == 100 for row in rows)


def test_pre_ledger_runs_still_fall_back_to_llm_call(tmp_path: Path) -> None:
    """Control group: without llm_usage events, the old llm_call counting stands."""
    events = [
        _event("llm_call", "legacy_task", offset_s=0.0, summary={"total_tokens": 55}),
        _event("llm_error", "legacy_task", offset_s=1.0, summary={"total_tokens": 5}),
        _event("tool_completed", "run_sql_tool", offset_s=2.0, summary={}),
    ]
    store = _store_with(tmp_path, events)
    metrics = summarize_session(store, _PROJECT, _RUN)
    assert metrics.llm_calls == 2
    assert metrics.total_tokens == 60
    assert len(llm_call_rows(events)) == 2

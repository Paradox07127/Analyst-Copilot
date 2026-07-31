from __future__ import annotations

from typing import Any

from eda_platform.core.session_metrics import spend_events
from eda_platform.schemas.artifacts import Artifact
from eda_platform.schemas.sessions import TraceEvent


def timeline_rows(events: list[TraceEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_type": event.event_type,
            "name": event.name,
            "started_at": event.started_at.isoformat(),
            "duration_ms": _duration_ms(event),
            "summary": _summary_preview(event.summary),
        }
        for event in events
    ]


def llm_call_rows(events: list[TraceEvent]) -> list[dict[str, Any]]:
    """One row per billed call: the llm_usage ledger when present, else llm_call."""
    rows: list[dict[str, Any]] = []
    for event in spend_events(events):
        rows.append(
            {
                "task": event.name,
                "provider": event.summary.get("provider", ""),
                "model": event.summary.get("model", ""),
                "prompt_tokens": _int_value(event.summary.get("prompt_tokens")),
                "completion_tokens": _int_value(event.summary.get("completion_tokens")),
                "total_tokens": _int_value(event.summary.get("total_tokens")),
                "cached_tokens": _int_value(event.summary.get("cached_tokens")),
                "cache_creation_tokens": _int_value(
                    event.summary.get("cache_creation_tokens")
                ),
                "reasoning_tokens": _int_value(event.summary.get("reasoning_tokens")),
                "estimated_cost_usd": event.summary.get("estimated_cost_usd"),
                "cost_basis": event.summary.get("cost_basis", ""),
                "pricing_version": event.summary.get("pricing_version", ""),
                # Ledger events written before usage_known existed carry no flag;
                # infer it from whether any token field was recorded.
                "usage_known": (
                    event.summary.get("usage_known") is True
                    if "usage_known" in event.summary
                    else any(
                        key in event.summary
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    )
                ),
                "request_id": event.summary.get("request_id", ""),
                "response_id": event.summary.get("response_id", ""),
                "finish_reason": event.summary.get("finish_reason", ""),
                "endpoint_host": event.summary.get("endpoint_host", ""),
                "schema": event.summary.get("schema", ""),
                "duration_ms": _duration_ms(event),
                "status": event.summary.get(
                    "status", "error" if event.event_type == "llm_error" else "success"
                ),
                "attempt": event.summary.get("attempt", ""),
                "error_type": event.summary.get("error_type", ""),
                "error": event.summary.get("error", ""),
            }
        )
    return rows


def tool_call_rows(events: list[TraceEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.event_type not in {"tool_started", "tool_completed", "tool_failed"}:
            continue
        rows.append(
            {
                "event_type": event.event_type,
                "tool": event.name,
                "duration_ms": _duration_ms(event),
                "row_count": event.summary.get("row_count", ""),
                "truncated": event.summary.get("truncated", ""),
                "artifact_id": event.summary.get("artifact_id", ""),
                "summary": _summary_preview(event.summary),
            }
        )
    return rows


def _has_error_detail(summary: dict[str, Any]) -> bool:
    """Successful LLM calls still carry empty ``error``/``error_type`` keys, so
    membership alone reported every one of them as a failure."""
    return any(str(summary.get(key) or "").strip() for key in ("error", "error_type"))


def error_rows(events: list[TraceEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if not (
            event.event_type.endswith("_failed")
            or event.event_type.endswith("_error")
            or _has_critical_report_validation(event)
            or _has_error_detail(event.summary)
        ):
            continue
        rows.append(
            {
                "event_type": event.event_type,
                "name": event.name,
                "error_type": event.summary.get("error_type", ""),
                "error": event.summary.get("error", _summary_preview(event.summary)),
            }
        )
    return rows


def artifact_rows(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": artifact.id,
            "type": artifact.type.value,
            "parents": len(artifact.parents),
            "warnings": len(artifact.warnings),
        }
        for artifact in artifacts
    ]


def _has_critical_report_validation(event: TraceEvent) -> bool:
    return (
        event.event_type == "report_validation"
        and _int_value(event.summary.get("critical_count")) > 0
    )


def _duration_ms(event: TraceEvent) -> int | None:
    if event.finished_at is None:
        return None
    return int((event.finished_at - event.started_at).total_seconds() * 1000)


def _summary_preview(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    items = []
    for key, value in summary.items():
        if key in {"sql", "prompt", "payload"}:
            value = _truncate(str(value), 80)
        items.append(f"{key}={value}")
    return "; ".join(items)


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 1]}..."


def _int_value(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

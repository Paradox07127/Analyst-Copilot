from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from eda_platform.schemas.sessions import TraceEvent

# Investigation-loop anti-repetition events.
PROBE_REPEATED_REJECTED = "probe_repeated_rejected"
LOOP_FAILURE_HISTORY_INJECTED = "loop_failure_history_injected"

# Finding-cluster deduplication event.
FINDINGS_DEDUPLICATED = "findings_deduplicated"

# Write-time evidence-interleaving events.
EVIDENCE_INTERLEAVE_REQUEST = "evidence_interleave_request"
EVIDENCE_INTERLEAVE_GRANTED = "evidence_interleave_granted"
EVIDENCE_INTERLEAVE_REJECTED = "evidence_interleave_rejected"


def trace_event(
    *,
    session_id: str,
    event_type: str,
    name: str,
    trial_id: str | None = None,
    investigation_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    call_id: str | None = None,
    attempt_id: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    summary: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(
        session_id=session_id,
        event_type=event_type,
        name=name,
        trial_id=trial_id,
        investigation_id=investigation_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        call_id=call_id,
        attempt_id=attempt_id,
        started_at=started_at or datetime.now(UTC),
        finished_at=finished_at,
        summary=summary or {},
    )

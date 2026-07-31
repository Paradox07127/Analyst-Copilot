"""A successful LLM call must not be reported as an error just because its
summary carries empty error placeholders (seen on a real run: 3 reported, 1 real)."""

from __future__ import annotations

from datetime import UTC, datetime

from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.debug import error_rows


def _event(event_type: str, name: str, summary: dict[str, object]) -> TraceEvent:
    now = datetime.now(UTC)
    return TraceEvent(
        session_id="r1",
        event_type=event_type,
        name=name,
        started_at=now,
        finished_at=now,
        summary=summary,
    )


def test_successful_call_with_empty_error_placeholders_is_not_an_error() -> None:
    events = [
        _event(
            "llm_call",
            "m2_report_claim_plan",
            {"status": "success", "error": "", "error_type": "", "total_tokens": 32273},
        )
    ]
    assert error_rows(events) == []


def test_call_with_a_real_error_message_is_still_reported() -> None:
    events = [
        _event(
            "llm_call",
            "m2_report_claim_plan",
            {"status": "error", "error": "rate limited", "error_type": "RateLimit"},
        )
    ]
    rows = error_rows(events)
    assert len(rows) == 1
    assert rows[0]["error"] == "rate limited"
    assert rows[0]["error_type"] == "RateLimit"


def test_error_type_alone_is_enough_to_report() -> None:
    events = [_event("llm_call", "x", {"error": "", "error_type": "TimeoutError"})]
    assert len(error_rows(events)) == 1


def test_failed_event_types_are_unaffected() -> None:
    events = [_event("step_failed", "profile_dataset", {"reason": "boom"})]
    assert len(error_rows(events)) == 1

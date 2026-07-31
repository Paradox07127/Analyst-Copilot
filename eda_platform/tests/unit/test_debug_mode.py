from __future__ import annotations

from datetime import UTC, datetime, timedelta

from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.debug import (
    artifact_rows,
    error_rows,
    llm_call_rows,
    timeline_rows,
    tool_call_rows,
)


def test_debug_helpers_shape_timeline_llm_tool_and_error_rows() -> None:
    started = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
    events = [
        TraceEvent(
            session_id="run_demo",
            event_type="step_completed",
            name="profile_dataset",
            started_at=started,
            finished_at=started + timedelta(milliseconds=250),
            summary={"artifact_count": 1},
        ),
        TraceEvent(
            session_id="run_demo",
            event_type="llm_call",
            name="m3_build_plan",
            started_at=started + timedelta(seconds=1),
            finished_at=started + timedelta(seconds=2),
            summary={
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "estimated_cost_usd": 0.0012,
                "schema": "AnalysisPlan",
            },
        ),
        TraceEvent(
            session_id="run_demo",
            event_type="llm_error",
            name="m2_report_claim_plan",
            started_at=started + timedelta(seconds=2),
            finished_at=started + timedelta(seconds=3),
            summary={
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "attempt": 1,
                "prompt_tokens": 60,
                "completion_tokens": 20,
                "total_tokens": 80,
                "error_type": "RuntimeError",
                "error": "LLM provider returned HTTP 400",
            },
        ),
        TraceEvent(
            session_id="run_demo",
            event_type="report_validation",
            name="m2_report_validator",
            started_at=started + timedelta(seconds=2),
            finished_at=started + timedelta(seconds=2),
            summary={
                "attempt": 2,
                "status": "needs_revision",
                "finding_count": 1,
                "critical_count": 1,
                "findings": ["critical:missing_evidence:Claim has no evidence references."],
            },
        ),
        TraceEvent(
            session_id="run_demo",
            event_type="tool_completed",
            name="run_sql",
            started_at=started + timedelta(seconds=3),
            finished_at=started + timedelta(seconds=4),
            summary={
                "sql": "select region, sum(amount) from orders group by region",
                "row_count": 2,
                "truncated": False,
            },
        ),
        TraceEvent(
            session_id="run_demo",
            event_type="step_failed",
            name="export_report",
            summary={"error_type": "RuntimeError", "error": "boom"},
        ),
    ]

    rows = timeline_rows(events)
    assert rows[0]["event_type"] == "step_completed"
    assert rows[0]["duration_ms"] == 250
    assert llm_call_rows(events) == [
        {
            "task": "m3_build_plan",
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.0012,
            "cost_basis": "",
            "pricing_version": "",
            "usage_known": True,
            "request_id": "",
            "response_id": "",
            "finish_reason": "",
            "endpoint_host": "",
            "schema": "AnalysisPlan",
            "duration_ms": 1000,
            "status": "success",
            "attempt": "",
            "error_type": "",
            "error": "",
        },
        {
            "task": "m2_report_claim_plan",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "prompt_tokens": 60,
            "completion_tokens": 20,
            "total_tokens": 80,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": None,
            "cost_basis": "",
            "pricing_version": "",
            "usage_known": True,
            "request_id": "",
            "response_id": "",
            "finish_reason": "",
            "endpoint_host": "",
            "schema": "",
            "duration_ms": 1000,
            "status": "error",
            "attempt": 1,
            "error_type": "RuntimeError",
            "error": "LLM provider returned HTTP 400",
        }
    ]
    assert tool_call_rows(events)[0]["tool"] == "run_sql"
    assert tool_call_rows(events)[0]["row_count"] == 2
    assert error_rows(events) == [
        {
            "event_type": "llm_error",
            "name": "m2_report_claim_plan",
            "error_type": "RuntimeError",
            "error": "LLM provider returned HTTP 400",
        },
        {
            "event_type": "report_validation",
            "name": "m2_report_validator",
            "error_type": "",
            "error": "attempt=2; status=needs_revision; finding_count=1; critical_count=1; "
            "findings=['critical:missing_evidence:Claim has no evidence references.']",
        },
        {
            "event_type": "step_failed",
            "name": "export_report",
            "error_type": "RuntimeError",
            "error": "boom",
        }
    ]


def test_debug_helpers_summarize_artifacts() -> None:
    artifacts = [
        Artifact(
            id="sql_1",
            type=ArtifactType.SQL_RESULT,
            project_id="project_demo",
            session_id="run_demo",
            payload={"row_count": 2, "truncated": False},
            warnings=["preview only"],
        )
    ]

    assert artifact_rows(artifacts) == [
        {
            "artifact_id": "sql_1",
            "type": "SqlResult",
            "parents": 0,
            "warnings": 1,
        }
    ]

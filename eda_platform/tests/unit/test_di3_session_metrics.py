from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eda_platform.core.llm_ledger import LLM_USAGE_EVENT
from eda_platform.core.session_metrics import persist_run_metrics, summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.sessions import TraceEvent

_T0 = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
_PROJECT = "p"
_RUN = "r"


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project(_PROJECT, name=_PROJECT)
    store.start_session(_PROJECT, _RUN)
    return store


def _event(
    event_type: str,
    name: str,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
    summary: dict | None = None,
) -> TraceEvent:
    started = _T0 + timedelta(seconds=offset_s)
    return TraceEvent(
        session_id=_RUN,
        event_type=event_type,
        name=name,
        started_at=started,
        finished_at=None if duration_s is None else started + timedelta(seconds=duration_s),
        summary=summary or {},
    )


def _artifact(artifact_id: str, artifact_type: ArtifactType, payload: dict) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=_PROJECT,
        session_id=_RUN,
        payload=payload,
    )


# --- aggregation math from synthetic trace events ---------------------------
def test_summarize_run_aggregates_llm_tool_tokens_and_steps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    events = [
        _event("step_started", "profile", offset_s=0.0),
        _event(
            "llm_call",
            "m2_report_claim_plan",
            offset_s=0.5,
            summary={
                "total_tokens": 100,
                "prompt_tokens": 80,
                "cached_tokens": 20,
                "estimated_cost_usd": 0.01,
            },
        ),
        _event("tool_completed", "run_sql_tool", offset_s=1.0),
        _event("run_sql", "analysis_query", offset_s=1.5),
        _event("step_completed", "profile", offset_s=0.0, duration_s=2.0),
        _event(
            "llm_error",
            "m2_report_claim_plan",
            offset_s=2.5,
            summary={"total_tokens": 50, "prompt_tokens": 40, "cached_tokens": 10},
        ),
        _event("step_started", "report", offset_s=3.0),
        _event("step_completed", "report", offset_s=3.0, duration_s=1.0),
    ]
    for event in events:
        store.append_trace(_PROJECT, event)

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.schema_version == 5
    assert metrics.session_id == _RUN
    assert metrics.cost_estimate_status in {
        "complete_estimate",
        "partial_estimate",
        "unavailable",
    }
    assert metrics.llm_calls == 2  # llm_call + llm_error
    assert metrics.tool_calls == 2  # tool_completed + run_sql (started not counted)
    assert metrics.total_tokens == 150
    # Prompt-cache aggregation (Phase 0): summed across llm_call + llm_error.
    assert metrics.prompt_tokens == 120  # 80 + 40
    assert metrics.cached_tokens == 30  # 20 + 10
    assert metrics.cache_hit_rate == 0.25  # 30 / 120
    assert metrics.est_cost_usd == 0.01
    # Wall clock: first start at t0, last timestamp = report step end at t0+4s.
    assert metrics.duration_seconds == 4.0
    # llm_error ends with _error -> counted as a failure.
    assert metrics.failures_count == 1

    steps = {step.step_name: step for step in metrics.steps}
    assert set(steps) == {"profile", "report"}
    assert steps["profile"].llm_calls == 1
    assert steps["profile"].tool_calls == 2
    assert steps["profile"].tokens == 100
    assert steps["profile"].duration_seconds == 2.0
    assert steps["report"].llm_calls == 0
    assert steps["report"].duration_seconds == 1.0


def test_step_metrics_attribute_ledger_usage_events(tmp_path: Path) -> None:
    # Per-step spend must count the same ledger events as the run headline:
    # drivers that never emit narrative llm_call events (9 of 15 calls in the
    # audited session) otherwise report llm_calls=0 for their step.
    store = _store(tmp_path)
    events = [
        _event("step_started", "execute_top_questions", offset_s=0.0),
        _event(
            LLM_USAGE_EVENT,
            "question_exec",
            offset_s=0.5,
            summary={"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
        ),
        # Narrative twin of the same provider call: must not double count.
        _event("llm_call", "question_exec", offset_s=0.6, summary={"total_tokens": 100}),
        _event(
            LLM_USAGE_EVENT,
            "question_exec",
            offset_s=1.0,
            summary={"total_tokens": 200, "prompt_tokens": 150, "completion_tokens": 50},
        ),
        _event("step_completed", "execute_top_questions", offset_s=0.0, duration_s=2.0),
        _event("step_started", "profile", offset_s=3.0),
        _event("step_completed", "profile", offset_s=3.0, duration_s=1.0),
    ]
    for event in events:
        store.append_trace(_PROJECT, event)

    metrics = summarize_session(store, _PROJECT, _RUN)

    steps = {step.step_name: step for step in metrics.steps}
    assert steps["execute_top_questions"].llm_calls == 2
    assert steps["execute_top_questions"].tokens == 300
    # A step with zero LLM calls still reports 0.
    assert steps["profile"].llm_calls == 0
    assert steps["profile"].tokens == 0
    # Totals already used the ledger and stay unchanged.
    assert metrics.llm_calls == 2
    assert metrics.total_tokens == 300


def test_summarize_run_repeated_step_names_merge(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for offset in (0.0, 5.0):
        store.append_trace(_PROJECT, _event("step_started", "profile", offset_s=offset))
        store.append_trace(
            _PROJECT,
            _event(
                "llm_call",
                "task",
                offset_s=offset + 0.5,
                summary={"total_tokens": 10},
            ),
        )
        store.append_trace(
            _PROJECT,
            _event("step_completed", "profile", offset_s=offset, duration_s=2.0),
        )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert len(metrics.steps) == 1
    step = metrics.steps[0]
    assert step.step_name == "profile"
    assert step.llm_calls == 2
    assert step.tokens == 20
    assert step.duration_seconds == 4.0


def test_summarize_run_cost_is_none_when_no_call_reported_cost(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_trace(_PROJECT, _event("llm_call", "task", summary={"total_tokens": 42}))

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.llm_calls == 1
    assert metrics.total_tokens == 42
    assert metrics.est_cost_usd is None


def test_summarize_run_counts_failed_and_failure_recorded_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_trace(_PROJECT, _event("step_failed", "profile", duration_s=0.1))
    store.append_trace(_PROJECT, _event("chat_turn_failed", "chat"))
    store.append_trace(_PROJECT, _event("failure_recorded", "ui.some_widget"))
    store.append_trace(_PROJECT, _event("step_completed", "ok_step", duration_s=0.1))

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.failures_count == 3


# --- artifact_counts + findings_count from the store ------------------------
def test_summarize_run_artifact_counts_and_findings(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_artifact(_artifact("prof_1", ArtifactType.DATASET_PROFILE, {"name": "a"}))
    store.save_artifact(_artifact("prof_2", ArtifactType.DATASET_PROFILE, {"name": "b"}))
    store.save_artifact(_artifact("vf_1", ArtifactType.VALIDATED_FINDING, {"text": "finding"}))
    store.save_artifact(
        _artifact(
            "qexec_1",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {"findings": [{"text": "f1"}, {"text": "f2"}]},
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.artifact_counts == {
        "DatasetProfile": 2,
        "ValidatedFinding": 1,
        "QuestionExecutionResult": 1,
    }
    # 1 ValidatedFinding artifact + 2 findings inside the execution result.
    assert metrics.findings_count == 3


def test_summarize_run_rolls_up_question_outcomes_and_contract_codes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_artifact(
        _artifact(
            "qexec_answered",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {
                "question_id": "q1",
                "question": "What is the answer?",
                "origin": "template",
                "status": "succeeded",
                "outcome": "answered",
                "interpretation_status": "validated",
                "findings": [{"text": "ok"}],
            },
        )
    )
    store.save_artifact(
        _artifact(
            "qexec_abstained",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {
                "status": "failed",
                "outcome": "abstained",
                "abstention_code": "hhi_out_of_range",
                "interpretation_status": "fallback",
                "findings": [],
            },
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.question_answered == 1
    assert metrics.question_abstained == 1
    assert metrics.question_failed == 0
    assert metrics.result_contract_failures == {"hhi_out_of_range": 1}
    assert metrics.interpretation_validated == 1
    assert metrics.interpretation_fallbacks == 1
    assert metrics.degraded is True
    assert metrics.coverage_limited is True


def test_safe_abstention_limits_coverage_without_marking_execution_degraded(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_artifact(
        _artifact(
            "qexec_abstained",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {
                "status": "failed",
                "outcome": "abstained",
                "abstention_code": "answer_schema_mismatch",
                "interpretation_status": "absent",
                "findings": [],
            },
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.degraded is False
    assert metrics.coverage_limited is True
    assert metrics.question_abstention_rate == 1.0
    assert metrics.question_answer_rate == 0.0


def test_efficiency_rollup_exposes_report_share_and_cost_per_answer(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.append_trace(_PROJECT, _event("step_started", "export_agentic_report"))
    store.append_trace(
        _PROJECT,
        _event("llm_call", "report", offset_s=1, summary={"total_tokens": 400}),
    )
    store.append_trace(
        _PROJECT,
        _event("step_completed", "export_agentic_report", duration_s=4),
    )
    store.save_artifact(
        _artifact(
            "qexec_answered",
            ArtifactType.QUESTION_EXECUTION_RESULT,
            {
                "question_id": "q_efficiency",
                "question": "What is the answer?",
                "origin": "template",
                "status": "succeeded",
                "outcome": "answered",
                "findings": [{"text": "answer"}],
            },
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.tokens_per_answered_question == 400
    assert metrics.seconds_per_answered_question == 4
    assert metrics.report_token_share == 1.0
    assert metrics.report_duration_share == 1.0
    assert metrics.publication_readiness == "analysis_available"


def test_summarize_run_tolerates_missing_run(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")

    metrics = summarize_session(store, "nope", "missing_run")

    assert metrics.session_id == "missing_run"
    assert metrics.llm_calls == 0
    assert metrics.tool_calls == 0
    assert metrics.total_tokens == 0
    assert metrics.est_cost_usd is None
    assert metrics.duration_seconds == 0.0
    assert metrics.steps == []
    assert metrics.artifact_counts == {}
    assert metrics.findings_count == 0
    assert metrics.failures_count == 0
    assert metrics.trace_status == "verified"


def test_summarize_run_marks_trace_read_failure_unverifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    def fail_trace_read(*_args: object, **_kwargs: object) -> list[TraceEvent]:
        raise ValueError("corrupt trace")

    monkeypatch.setattr(store, "list_trace_events", fail_trace_read)

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.trace_status == "unverifiable"
    assert metrics.degraded is True


def test_summarize_run_blocks_unverifiable_published_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_artifact(
        Artifact(
            id="decision_legacy",
            type=ArtifactType.DECISION_REPORT,
            project_id=_PROJECT,
            session_id=_RUN,
            parents=["missing_finding"],
            payload={
                "report_id": "report_legacy",
                "brief_id": "brief_legacy",
                "project_id": _PROJECT,
                "title": "Decision Report",
                "scqa": {
                    "situation": "A result exists.",
                    "complication": "Its source is missing.",
                    "question": "Can it be reused?",
                    "answer": "Freshness must be verified first.",
                },
                "sections": [{"title": "Finding", "body": "Source unavailable."}],
                "report_readiness": "eligible",
                "source_finding_artifact_ids": ["missing_finding"],
            },
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.publication_readiness == "published"
    assert metrics.publication_freshness == "unverifiable"
    assert metrics.publication_blocked is True


# --- persistence -------------------------------------------------------------
def test_persist_run_metrics_saves_run_metrics_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_trace(
        _PROJECT,
        _event("llm_call", "task", summary={"total_tokens": 7, "estimated_cost_usd": 0.002}),
    )

    artifact_id = persist_run_metrics(store, _PROJECT, _RUN)

    artifact = store.get_artifact(artifact_id)
    assert artifact.type is ArtifactType.SESSION_METRICS
    payload = SessionMetrics.model_validate(artifact.payload)
    assert payload.schema_version == 5
    assert payload.session_id == _RUN
    assert payload.llm_calls == 1
    assert payload.total_tokens == 7
    assert payload.est_cost_usd == 0.002


def test_persist_run_metrics_is_idempotent_per_run(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = persist_run_metrics(store, _PROJECT, _RUN)
    second = persist_run_metrics(store, _PROJECT, _RUN)

    assert first == second
    artifacts, _ = store.list_artifacts_safe(project_id=_PROJECT, session_id=_RUN)
    rollups = [a for a in artifacts if a.type is ArtifactType.SESSION_METRICS]
    assert len(rollups) == 1


# --- driver-level hook: auto_eda persists the rollup -------------------------
def test_run_auto_eda_persists_run_metrics_artifact(tmp_path: Path) -> None:
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n", encoding="utf-8")
    workspace = tmp_path / "workspace"

    result = run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="p",
        session_id="r",
    )

    store = ArtifactStore(workspace)
    artifacts, _ = store.list_artifacts_safe(project_id="p", session_id=result.session_id)
    rollups = [a for a in artifacts if a.type is ArtifactType.SESSION_METRICS]
    assert len(rollups) == 1
    payload = SessionMetrics.model_validate(rollups[0].payload)
    assert payload.session_id == result.session_id
    # The run produced real steps and artifacts; the rollup must reflect them.
    assert payload.steps
    assert payload.duration_seconds >= 0.0
    assert sum(payload.artifact_counts.values()) == len(result.artifacts)


# --- DI8-D three-tier report gate rollup ------------------------------------
def test_summarize_run_report_gate_rollup_prefers_standalone_audits(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    audit_payload = {
        "gate_verdict": "degraded",
        "degraded_claim_count": 2,
        "time_boundary_truncations": 1,
    }
    store.save_artifact(_artifact("audit_1", ArtifactType.REPORT_AUDIT, audit_payload))
    # The bundle embeds the SAME audit; it must not be double-counted.
    store.save_artifact(_artifact("bundle_1", ArtifactType.REPORT_BUNDLE, {"audit": audit_payload}))

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.semantic_degraded_claims == 2
    assert metrics.time_boundary_truncations == 1
    assert metrics.report_gate_verdict == "degraded"


def test_summarize_run_report_gate_rollup_falls_back_to_bundle_audit(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_artifact(
        _artifact(
            "bundle_1",
            ArtifactType.REPORT_BUNDLE,
            {
                "audit": {
                    "gate_verdict": "rejected",
                    "degraded_claim_count": 0,
                    "time_boundary_truncations": 0,
                }
            },
        )
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.report_gate_verdict == "rejected"
    assert metrics.semantic_degraded_claims == 0


def test_summarize_run_report_gate_verdict_none_without_reports(tmp_path: Path) -> None:
    store = _store(tmp_path)

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.report_gate_verdict is None
    assert metrics.semantic_degraded_claims == 0
    assert metrics.time_boundary_truncations == 0


# --- DI9 rollups: interleave / dedup / domain metrics ------------------------
def test_summarize_run_di9_rollups(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_artifact(
        _artifact(
            "ilv_1",
            ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT,
            {"exchanges": [], "granted_count": 3, "rejected_count": 1},
        )
    )
    store.save_artifact(
        _artifact(
            "qcand_1",
            ArtifactType.QUESTION_CANDIDATE_SET,
            {
                "candidates": [
                    {"template_id": "domain_metric"},
                    {"template_id": "domain_metric"},
                    {"template_id": "trend"},
                    {"template_id": None},
                ]
            },
        )
    )
    events = [
        _event(
            "findings_deduplicated",
            "synthesis_orchestrator",
            summary={"clusters": 4, "merged_supporting": 2},
        ),
        _event(
            "domain_metrics_skipped",
            "discover_questions",
            summary={"resolved_count": 5, "skipped_count": 3},
        ),
    ]
    for event in events:
        store.append_trace(_PROJECT, event)

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.evidence_interleave_granted == 3
    assert metrics.evidence_interleave_rejected == 1
    assert metrics.domain_metric_questions == 2
    assert metrics.findings_dedup_clusters == 4
    assert metrics.findings_dedup_merged == 2
    assert metrics.domain_metrics_skipped == 3


def test_summarize_run_di9_rollups_default_to_zero(tmp_path: Path) -> None:
    store = _store(tmp_path)

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.evidence_interleave_granted == 0
    assert metrics.findings_dedup_clusters == 0
    assert metrics.domain_metric_questions == 0
    assert metrics.domain_metrics_skipped == 0


def test_summarize_run_relationship_budget_rollups(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_trace(
        _PROJECT,
        _event(
            "relationship_discovery_bounded",
            "discover_relationships",
            summary={
                "overlap_pairs_evaluated": 12,
                "overlap_pairs_prefiltered": 40,
                "full_validations_completed": 3,
                "coverage_status": "limited",
                "candidate_payload_bytes": 4096,
            },
        ),
    )
    store.append_trace(
        _PROJECT,
        _event(
            "relationship_discovery_deferred",
            "discover_relationships",
            summary={"reason": "default_on_demand_policy"},
        ),
    )
    store.append_trace(
        _PROJECT,
        _event(
            "join_authorization_freshness",
            "discover_questions",
            summary={"fresh": 2, "stale": 1, "unverifiable": 3},
        ),
    )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.relationship_overlap_pairs_evaluated == 12
    assert metrics.relationship_overlap_pairs_prefiltered == 40
    assert metrics.relationship_full_validations == 3
    assert metrics.relationship_coverage_limited is True
    assert metrics.coverage_limited is True
    assert metrics.relationship_candidate_payload_bytes == 4096
    assert metrics.relationship_discovery_deferred is True
    assert metrics.join_authorizations_fresh == 2
    assert metrics.join_authorizations_stale == 1
    assert metrics.join_authorizations_unverifiable == 3


def test_a_v4_metrics_payload_still_loads_after_the_v5_bump() -> None:
    """Metrics are persisted artifacts, so a workspace written before the bump
    must keep opening; the new fields default rather than reject the payload."""
    stored_v4 = {
        "schema_version": 4,
        "session_id": _RUN,
        "llm_calls": 3,
        "total_tokens": 150,
        "prompt_tokens": 120,
        "cached_tokens": 20,
        "est_cost_usd": 0.004,
    }

    metrics = SessionMetrics.model_validate(stored_v4)

    assert metrics.schema_version == 4  # preserved, not silently rewritten
    assert metrics.est_cost_usd == 0.004
    assert metrics.completion_tokens == 0
    assert metrics.reasoning_tokens == 0
    assert metrics.cost_estimate_status == "not_applicable"
    assert metrics.llm_calls_by_model == {}


def test_cost_estimate_status_marks_a_partially_priced_session(tmp_path: Path) -> None:
    """est_cost_usd sums only priceable calls, so a total covering 1 of 2 calls
    must not present itself the same way as a complete one."""
    store = _store(tmp_path)
    for cost in (0.002, None):
        store.append_trace(
            _PROJECT,
            _event(
                LLM_USAGE_EVENT,
                "m3_build_plan",
                summary={"total_tokens": 10, "estimated_cost_usd": cost, "usage_known": True},
            ),
        )

    metrics = summarize_session(store, _PROJECT, _RUN)

    assert metrics.cost_estimate_status == "partial_estimate"
    assert (metrics.costed_calls, metrics.uncosted_calls) == (1, 1)

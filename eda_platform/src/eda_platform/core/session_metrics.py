"""Aggregate a run's persisted trace events + artifacts into SessionMetrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from eda_platform.core.finding_freshness import assess_decision_report_freshness
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.llm_ledger import (
    BUDGET_REJECTED_EVENT,
    BUDGET_RESERVED_EVENT,
    BUDGET_SETTLED_EVENT,
    LLM_USAGE_EVENT,
)
from eda_platform.core.publication_state import derive_publication_state
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.publication import PublicationFreshness
from eda_platform.schemas.resource_metrics import (
    AutoEdaResourceUsage,
    EdaArtifactMetrics,
    EdaDataFootprint,
    EdaInputMetrics,
    EdaMemoryMetrics,
    EdaResourcePreflight,
)
from eda_platform.schemas.session_metrics import SessionMetrics, StepMetric

if TYPE_CHECKING:
    from datetime import datetime

    from eda_platform.core.store import ArtifactStore
    from eda_platform.schemas.sessions import TraceEvent

_LLM_EVENT_TYPES = frozenset({"llm_call", "llm_error"})
# Ledger events are emitted once per real provider call at a single seam; the
# narrative llm_call events are per-driver and were historically incomplete.
_LEDGER_EVENT_TYPES = frozenset({LLM_USAGE_EVENT})
_TOOL_EVENT_TYPES = frozenset({"tool_completed", "tool_failed", "run_sql", "code_agent_attempt"})
_FAILURE_EVENT_TYPE = "failure_recorded"
_QUESTION_ROUTE_EVENT_TYPES = frozenset({"question_llm_skipped", "llm_call"})
_SEMANTIC_BOOTSTRAP_EVENT_TYPE = "semantic_bootstrap"
_TEMPLATE_BACKSTOP_EVENT_TYPE = "template_backstop"
_JOIN_PROPOSED_EVENT_TYPE = "join_candidates_proposed"
_JOIN_FRESHNESS_EVENT_TYPE = "join_authorization_freshness"
_RELATIONSHIP_BOUNDED_EVENT_TYPE = "relationship_discovery_bounded"
_RELATIONSHIP_ON_DEMAND_EVENT_TYPE = "relationship_validation_on_demand"
_RELATIONSHIP_DEFERRED_EVENT_TYPE = "relationship_discovery_deferred"


def summarize_session(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    artifact_snapshot: Sequence[Artifact] | None = None,
) -> SessionMetrics:
    """Build metrics for a persisted run, treating missing data as empty."""
    events, trace_unverifiable = _load_events(store, project_id, session_id)
    if artifact_snapshot is None:
        artifacts, _warnings = store.list_artifacts_safe(
            project_id=project_id, session_id=session_id
        )
    else:
        artifacts = list(artifact_snapshot)

    question_route = _question_route_rollup(events)
    billed = spend_events(events)
    metrics = SessionMetrics(
        session_id=session_id,
        llm_calls=len(billed),
        tool_calls=sum(1 for e in events if e.event_type in _TOOL_EVENT_TYPES),
        total_tokens=sum(_as_int(e.summary.get("total_tokens")) for e in billed),
        prompt_tokens=_prompt_tokens_total(billed),
        completion_tokens=sum(_as_int(e.summary.get("completion_tokens")) for e in billed),
        cached_tokens=_cached_tokens_total(billed),
        cache_creation_tokens=sum(_as_int(e.summary.get("cache_creation_tokens")) for e in billed),
        reasoning_tokens=sum(_as_int(e.summary.get("reasoning_tokens")) for e in billed),
        cache_hit_rate=_cache_hit_rate(billed),
        est_cost_usd=_total_cost(billed),
        costed_calls=sum(e.summary.get("estimated_cost_usd") is not None for e in billed),
        uncosted_calls=sum(e.summary.get("estimated_cost_usd") is None for e in billed),
        usage_known_calls=sum(e.summary.get("usage_known") is True for e in billed),
        usage_unknown_calls=sum(e.summary.get("usage_known") is not True for e in billed),
        cost_estimate_status=_cost_estimate_status(billed),
        llm_calls_by_model=dict(Counter(str(e.summary.get("model") or "unknown") for e in billed)),
        llm_calls_by_kind=dict(
            Counter(
                str(e.summary.get("transport_kind") or e.summary.get("kind") or "unknown")
                for e in billed
            )
        ),
        llm_calls_by_task=dict(
            Counter(str(e.summary.get("task") or e.name or "unknown") for e in billed)
        ),
        llm_calls_by_status=dict(
            Counter(str(e.summary.get("status") or "unknown") for e in billed)
        ),
        budget_reserved_calls=sum(event.event_type == BUDGET_RESERVED_EVENT for event in events),
        budget_settled_calls=sum(event.event_type == BUDGET_SETTLED_EVENT for event in events),
        budget_rejected_calls=sum(event.event_type == BUDGET_REJECTED_EVENT for event in events),
        budget_uncertain_calls=sum(
            event.event_type == BUDGET_SETTLED_EVENT and event.summary.get("status") == "uncertain"
            for event in events
        ),
        budget_total_tokens=sum(
            _as_int(event.summary.get("total_tokens"))
            for event in events
            if event.event_type == BUDGET_SETTLED_EVENT
        ),
        budget_est_cost_usd=_budget_total_cost(events),
        budget_reconciliation=_budget_reconciliation(events, billed),
        duration_seconds=_run_duration_seconds(events),
        resource_usage=_resource_usage(events, artifacts),
        steps=_step_metrics(events),
        artifact_counts=dict(Counter(a.type.value for a in artifacts)),
        findings_count=_findings_count(artifacts),
        failures_count=sum(1 for e in events if _is_failure(e)),
        trace_status="unverifiable" if trace_unverifiable else "verified",
        question_llm_skipped=question_route.skipped,
        question_proposals_dropped=question_route.dropped,
        question_dataset_names_resolved=question_route.resolved,
        question_list_coercions=question_route.coercions,
        degraded=question_route.degraded,
        semantic_bootstrap_degraded=any(
            bool(e.summary.get("degraded"))
            for e in events
            if e.event_type == _SEMANTIC_BOOTSTRAP_EVENT_TYPE
        ),
        column_roles_unverified=sum(
            _as_int(e.summary.get("unverified_count"))
            for e in events
            if e.event_type == _SEMANTIC_BOOTSTRAP_EVENT_TYPE
        ),
        template_backstop_used=sum(
            _as_int(e.summary.get("backstop_count"))
            for e in events
            if e.event_type == _TEMPLATE_BACKSTOP_EVENT_TYPE
        ),
        join_candidates_proposed=sum(
            _as_int(e.summary.get("proposed_count"))
            for e in events
            if e.event_type == _JOIN_PROPOSED_EVENT_TYPE
        ),
        join_authorizations_fresh=sum(
            _as_int(e.summary.get("fresh"))
            for e in events
            if e.event_type == _JOIN_FRESHNESS_EVENT_TYPE
        ),
        join_authorizations_stale=sum(
            _as_int(e.summary.get("stale"))
            for e in events
            if e.event_type == _JOIN_FRESHNESS_EVENT_TYPE
        ),
        join_authorizations_unverifiable=sum(
            _as_int(e.summary.get("unverifiable"))
            for e in events
            if e.event_type == _JOIN_FRESHNESS_EVENT_TYPE
        ),
        relationship_overlap_pairs_evaluated=sum(
            _as_int(e.summary.get("overlap_pairs_evaluated"))
            for e in events
            if e.event_type == _RELATIONSHIP_BOUNDED_EVENT_TYPE
        ),
        relationship_overlap_pairs_prefiltered=sum(
            _as_int(e.summary.get("overlap_pairs_prefiltered"))
            for e in events
            if e.event_type == _RELATIONSHIP_BOUNDED_EVENT_TYPE
        ),
        relationship_full_validations=sum(
            _as_int(e.summary.get("full_validations_completed"))
            for e in events
            if e.event_type
            in {_RELATIONSHIP_BOUNDED_EVENT_TYPE, _RELATIONSHIP_ON_DEMAND_EVENT_TYPE}
        ),
        relationship_coverage_limited=any(
            e.summary.get("coverage_status") == "limited"
            for e in events
            if e.event_type == _RELATIONSHIP_BOUNDED_EVENT_TYPE
        ),
        relationship_candidate_payload_bytes=sum(
            _as_int(e.summary.get("candidate_payload_bytes"))
            for e in events
            if e.event_type == _RELATIONSHIP_BOUNDED_EVENT_TYPE
        ),
        relationship_discovery_deferred=any(
            e.event_type == _RELATIONSHIP_DEFERRED_EVENT_TYPE for e in events
        ),
    )
    report_gate = _report_gate_rollup(artifacts)
    metrics.semantic_degraded_claims = report_gate.degraded_claims
    metrics.time_boundary_truncations = report_gate.time_boundary_truncations
    metrics.numeric_unverified_claims = report_gate.numeric_unverified_claims
    metrics.quantitative_coverage_gaps = report_gate.quantitative_coverage_gaps
    metrics.report_gate_verdict = report_gate.verdict
    _apply_di9_rollups(metrics, events, artifacts)
    _apply_question_quality_rollups(metrics, artifacts)
    _apply_macro_loop_rollups(metrics, artifacts)
    report_freshness: dict[str, PublicationFreshness] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.DECISION_REPORT:
            report_freshness[artifact.id] = assess_decision_report_freshness(
                store,
                project_id,
                artifact.id,
                report_session_id=artifact.session_id,
            ).status
    publication = derive_publication_state(artifacts, decision_report_freshness=report_freshness)
    metrics.publication_readiness = publication.readiness
    metrics.publication_freshness = publication.publication_freshness
    metrics.report_eligible_findings = publication.report_eligible_findings
    _apply_efficiency_rollups(metrics)
    metrics.degraded = bool(
        metrics.degraded or metrics.interpretation_fallbacks or trace_unverifiable
    )
    metrics.coverage_limited = bool(
        metrics.question_abstained > 0
        or metrics.relationship_discovery_deferred
        or metrics.relationship_coverage_limited
    )
    metrics.publication_blocked = bool(
        metrics.report_gate_verdict == "rejected"
        or publication.technical_report_status == "blocked_for_review"
        or publication.publication_freshness in {"stale", "unverifiable"}
    )
    return metrics


def persist_run_metrics(store: ArtifactStore, project_id: str, session_id: str) -> str:
    """Summarize the run and save the rollup as a SESSION_METRICS artifact."""
    artifact = create_run_metrics_artifact(store, project_id, session_id)
    store.save_artifact(artifact)
    return artifact.id


def create_run_metrics_artifact(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    artifact_snapshot: Sequence[Artifact] | None = None,
) -> Artifact:
    """Build, but do not publish, a metrics artifact from one explicit inventory."""
    metrics = summarize_session(
        store,
        project_id,
        session_id,
        artifact_snapshot=artifact_snapshot,
    )
    return Artifact(
        id=make_artifact_id(
            "session_metrics", {"project_id": project_id, "session_id": session_id}
        ),
        type=ArtifactType.SESSION_METRICS,
        project_id=project_id,
        session_id=session_id,
        payload=metrics.model_dump(mode="json"),
    )


# --------------------------------------------------------------------------- #
# Aggregation internals
# --------------------------------------------------------------------------- #
def _load_events(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
) -> tuple[list[TraceEvent], bool]:
    try:
        return store.list_trace_events(project_id=project_id, session_id=session_id), False
    except Exception:  # noqa: BLE001 - preserve a typed unverifiable rollup
        return [], True


def spend_events(events: list[TraceEvent]) -> list[TraceEvent]:
    """The events that count as spend: ledger if the run has one, else llm_call.

    Public so UI surfaces (trace captions, debug tables) count the same events
    as the SessionMetrics headline instead of re-deriving their own tally.
    """
    ledger = [e for e in events if e.event_type in _LEDGER_EVENT_TYPES]
    if ledger:
        return ledger
    return [e for e in events if e.event_type in _LLM_EVENT_TYPES]


def _prompt_tokens_total(billed: list[TraceEvent]) -> int:
    return sum(_as_int(e.summary.get("prompt_tokens")) for e in billed)


def _cached_tokens_total(billed: list[TraceEvent]) -> int:
    return sum(_as_int(e.summary.get("cached_tokens")) for e in billed)


def _cache_hit_rate(billed: list[TraceEvent]) -> float:
    """Run-wide cached / prompt token ratio; 0.0 when no prompt tokens were metered."""
    prompt = _prompt_tokens_total(billed)
    if prompt <= 0:
        return 0.0
    cached = min(_cached_tokens_total(billed), prompt)
    return round(cached / prompt, 6)


def _total_cost(billed: list[TraceEvent]) -> float | None:
    """Sum of per-call cost estimates; None when no call reported a cost."""
    costs = [
        _as_float(e.summary.get("estimated_cost_usd"))
        for e in billed
        if e.summary.get("estimated_cost_usd") is not None
    ]
    if not costs:
        return None
    return round(sum(costs), 6)


def _cost_estimate_status(
    billed: list[TraceEvent],
) -> Literal["complete_estimate", "partial_estimate", "unavailable", "not_applicable"]:
    """Whether est_cost_usd covers every call, some of them, or none."""
    if not billed:
        return "not_applicable"
    costed = sum(event.summary.get("estimated_cost_usd") is not None for event in billed)
    if costed == len(billed):
        return "complete_estimate"
    if costed:
        return "partial_estimate"
    return "unavailable"


def _budget_total_cost(events: list[TraceEvent]) -> float | None:
    settled = [event for event in events if event.event_type == BUDGET_SETTLED_EVENT]
    if not settled:
        return None
    if any(event.summary.get("estimated_cost_usd") is None for event in settled):
        return None
    return round(
        sum(_as_float(event.summary.get("estimated_cost_usd")) for event in settled),
        6,
    )


def _budget_reconciliation(
    events: list[TraceEvent],
    billed: list[TraceEvent],
) -> Literal["verified", "unverifiable", "not_applicable"]:
    budget_events = [
        event
        for event in events
        if event.event_type in {BUDGET_RESERVED_EVENT, BUDGET_SETTLED_EVENT}
    ]
    if not budget_events:
        return "not_applicable"
    ledger_events = [event for event in billed if event.call_id]
    reserved_events = [
        event
        for event in budget_events
        if event.event_type == BUDGET_RESERVED_EVENT and event.call_id
    ]
    settled = [
        event
        for event in budget_events
        if event.event_type == BUDGET_SETTLED_EVENT and event.call_id
    ]
    ledger_by_call = {event.call_id: event for event in ledger_events}
    reserved_by_call = {event.call_id: event for event in reserved_events}
    settled_by_call = {event.call_id: event for event in settled}
    ledger_call_ids = set(ledger_by_call)
    reserved_call_ids = set(reserved_by_call)
    settled_call_ids = set(settled_by_call)
    if (
        not ledger_call_ids
        or len(ledger_by_call) != len(ledger_events)
        or len(reserved_by_call) != len(reserved_events)
        or len(settled_by_call) != len(settled)
        or ledger_call_ids != reserved_call_ids
        or ledger_call_ids != settled_call_ids
    ):
        return "unverifiable"
    if any(event.summary.get("usage_known") is not True for event in settled):
        return "unverifiable"
    token_pairs = (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    )
    for call_id in ledger_call_ids:
        ledger = ledger_by_call[call_id]
        budget = settled_by_call[call_id]
        if any(
            _as_int(ledger.summary.get(ledger_name)) != _as_int(budget.summary.get(budget_name))
            for ledger_name, budget_name in token_pairs
        ):
            return "unverifiable"
        ledger_cost = ledger.summary.get("estimated_cost_usd")
        budget_cost = budget.summary.get("estimated_cost_usd")
        if ledger_cost is None or budget_cost is None:
            return "unverifiable"
        if round(_as_float(ledger_cost), 9) != round(_as_float(budget_cost), 9):
            return "unverifiable"
    return "verified"


def _run_duration_seconds(events: list[TraceEvent]) -> float:
    """Wall-clock span from the first event start to the last known timestamp."""
    starts = [e.started_at for e in events]
    ends: list[datetime] = [e.finished_at or e.started_at for e in events]
    if not starts:
        return 0.0
    span = (max(ends) - min(starts)).total_seconds()
    return max(round(span, 6), 0.0)


def _latest_event_summary(events: list[TraceEvent], event_type: str) -> dict[str, object]:
    for event in reversed(events):
        if event.event_type == event_type:
            return dict(event.summary)
    return {}


def _resource_usage(events: list[TraceEvent], artifacts: list[Artifact]) -> AutoEdaResourceUsage:
    inputs_summary = _latest_event_summary(events, "eda_inputs_loaded")
    runtime_summary = _latest_event_summary(events, "eda_resource_usage")
    preflight_artifact = next(
        (
            artifact
            for artifact in reversed(artifacts)
            if artifact.type is ArtifactType.RESOURCE_PREFLIGHT
        ),
        None,
    )
    preflight: EdaResourcePreflight | None = None
    if preflight_artifact is not None:
        try:
            preflight = EdaResourcePreflight.model_validate(preflight_artifact.payload)
        except ValueError:
            preflight = None

    analysis = _data_footprint(inputs_summary.get("analysis"))
    raw_lineage = _data_footprint(inputs_summary.get("raw_lineage"))
    handoff = next(
        (
            artifact
            for artifact in reversed(artifacts)
            if artifact.type is ArtifactType.AGENT_HANDOFF
        ),
        None,
    )
    context = handoff.payload.get("context_policy", {}) if handoff is not None else {}
    non_metrics = [
        artifact for artifact in artifacts if artifact.type is not ArtifactType.SESSION_METRICS
    ]
    canonical_bytes = sum(
        len(artifact.model_dump_json().encode("utf-8")) for artifact in non_metrics
    )
    storage_bytes = sum(
        len(artifact.model_dump_json(indent=2).encode("utf-8")) for artifact in non_metrics
    )
    baseline = _optional_int(runtime_summary.get("baseline_peak_rss_bytes"))
    peak = _optional_int(runtime_summary.get("peak_rss_bytes"))
    method = str(runtime_summary.get("peak_rss_method") or "unavailable")
    if method not in {
        "getrusage_ru_maxrss",
        "get_process_memory_info_peak_working_set",
        "unavailable",
    }:
        method = "unavailable"
    estimated_working_set = preflight.estimated_working_set_bytes if preflight is not None else 0
    verified_working_set = preflight.verified_working_set_bytes if preflight is not None else None
    budget = preflight.policy.max_working_set_bytes if preflight is not None else 0
    preflight_status = preflight.status if preflight is not None else "unavailable"
    processing_mode = preflight.compute_mode if preflight is not None else "unknown"
    requested_workers = preflight.requested_dataset_workers if preflight is not None else 1
    effective_workers = preflight.effective_dataset_workers if preflight is not None else 0
    has_runtime = bool(inputs_summary or runtime_summary)
    return AutoEdaResourceUsage(
        measurement_status=(
            "verified"
            if has_runtime and preflight is not None and preflight.phase == "verified"
            else "partial"
            if has_runtime or preflight is not None
            else "unavailable"
        ),
        wall_duration_seconds=max(0.0, _as_float(runtime_summary.get("wall_duration_seconds"))),
        preprocessing_duration_seconds=max(
            0.0, _as_float(runtime_summary.get("preprocessing_duration_seconds"))
        ),
        ingest_duration_seconds=max(0.0, _as_float(inputs_summary.get("ingest_duration_seconds"))),
        processing_mode=processing_mode,
        preflight_status=preflight_status,
        requested_dataset_workers=requested_workers,
        effective_dataset_workers=effective_workers,
        worker_adjustment_reason=(
            preflight.worker_adjustment_reason if preflight is not None else None
        ),
        inputs=EdaInputMetrics(
            analysis=analysis,
            raw_lineage=raw_lineage,
            unique_file_bytes=max(0, _as_int(inputs_summary.get("unique_file_bytes"))),
        ),
        memory=EdaMemoryMetrics(
            baseline_peak_rss_bytes=baseline,
            peak_rss_bytes=peak,
            peak_rss_delta_bytes=(
                max(0, peak - baseline) if peak is not None and baseline is not None else None
            ),
            peak_rss_method=method,
            working_set_budget_bytes=budget,
            estimated_working_set_bytes=estimated_working_set,
            verified_working_set_bytes=verified_working_set,
        ),
        artifacts=EdaArtifactMetrics(
            artifact_count=len(artifacts),
            storage_bytes_excluding_session_metrics=storage_bytes,
            canonical_json_bytes_excluding_session_metrics=canonical_bytes,
            agent_handoff_payload_bytes=max(0, _as_int(context.get("serialized_bytes"))),
            default_context_bytes=max(0, _as_int(context.get("initial_context_bytes"))),
            default_context_estimated_tokens=max(
                0, _as_int(context.get("initial_context_estimated_tokens"))
            ),
        ),
    )


def _data_footprint(value: object) -> EdaDataFootprint:
    if not isinstance(value, dict):
        return EdaDataFootprint()
    try:
        return EdaDataFootprint.model_validate(value)
    except ValueError:
        return EdaDataFootprint()


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _step_metrics(events: list[TraceEvent]) -> list[StepMetric]:
    """Aggregate kernel step brackets into per-step-name metrics."""
    # Per-step spend must count the same events as the run headline: ledger
    # events when the run has them (only some drivers emit narrative llm_call
    # events, so counting those under-attributed steps), llm_call otherwise.
    billed_types = (
        _LEDGER_EVENT_TYPES
        if any(event.event_type in _LEDGER_EVENT_TYPES for event in events)
        else _LLM_EVENT_TYPES
    )
    by_name: dict[str, StepMetric] = {}
    current: StepMetric | None = None
    for event in events:
        if event.event_type == "step_started":
            current = by_name.setdefault(event.name, StepMetric(step_name=event.name))
            continue
        if event.event_type in {"step_completed", "step_failed"}:
            metric = by_name.setdefault(event.name, StepMetric(step_name=event.name))
            if event.finished_at is not None:
                elapsed = (event.finished_at - event.started_at).total_seconds()
                metric.duration_seconds = round(metric.duration_seconds + max(elapsed, 0.0), 6)
            if current is not None and current.step_name == event.name:
                current = None
            continue
        if current is None:
            continue
        if event.event_type in billed_types:
            current.llm_calls += 1
            current.tokens += _as_int(event.summary.get("total_tokens"))
        elif event.event_type in _TOOL_EVENT_TYPES:
            current.tool_calls += 1
    return list(by_name.values())


@dataclass(frozen=True)
class _QuestionRouteRollup:
    """DI8-A yellow-light rollup of the LLM question route for one run."""

    skipped: bool
    dropped: int
    resolved: int
    coercions: int

    @property
    def degraded(self) -> bool:
        # Degraded = the LLM question route was skipped entirely, or delivered
        # only partially (malformed proposals dropped). Auto-resolved dataset
        # names and list coercions are repairs, not degradation — they are
        # still surfaced as counters above.
        return self.skipped or self.dropped > 0


def _question_route_rollup(events: list[TraceEvent]) -> _QuestionRouteRollup:
    route_events = [e for e in events if e.event_type in _QUESTION_ROUTE_EVENT_TYPES]
    return _QuestionRouteRollup(
        skipped=any(e.event_type == "question_llm_skipped" for e in route_events),
        dropped=sum(_as_int(e.summary.get("proposals_dropped")) for e in route_events),
        resolved=sum(_as_int(e.summary.get("dataset_names_resolved")) for e in route_events),
        coercions=sum(_as_int(e.summary.get("list_coercions")) for e in route_events),
    )


def _apply_di9_rollups(
    metrics: SessionMetrics, events: list[TraceEvent], artifacts: list[Artifact]
) -> None:
    """Aggregate evidence interleaving, finding deduplication, and domain-metric counters."""
    for artifact in artifacts:
        if artifact.type is ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT:
            metrics.evidence_interleave_granted += _as_int(artifact.payload.get("granted_count"))
            metrics.evidence_interleave_rejected += _as_int(artifact.payload.get("rejected_count"))
        elif artifact.type is ArtifactType.QUESTION_CANDIDATE_SET:
            candidates = artifact.payload.get("candidates")
            if isinstance(candidates, list):
                metrics.domain_metric_questions += sum(
                    1
                    for candidate in candidates
                    if isinstance(candidate, dict)
                    and candidate.get("template_id") == "domain_metric"
                )
    for event in events:
        if event.event_type == "findings_deduplicated":
            metrics.findings_dedup_clusters += _as_int(event.summary.get("clusters"))
            metrics.findings_dedup_merged += _as_int(event.summary.get("merged_supporting"))
        elif event.event_type == "domain_metrics_skipped":
            metrics.domain_metrics_skipped += _as_int(event.summary.get("skipped_count"))


@dataclass(frozen=True)
class _ReportGateRollup:
    """DI8-D three-tier gate rollup across the run's persisted report audits."""

    degraded_claims: int
    time_boundary_truncations: int
    numeric_unverified_claims: int
    quantitative_coverage_gaps: int
    verdict: str | None


_GATE_VERDICT_SEVERITY = {"pass": 0, "degraded": 1, "rejected": 2}


def _report_gate_rollup(artifacts: list[Artifact]) -> _ReportGateRollup:
    # Every persisted report writes BOTH a standalone REPORT_AUDIT artifact and
    # a REPORT_BUNDLE embedding the same audit; counting both would double the
    # totals, so standalone audits win and bundle audits are only a fallback.
    audits = [
        artifact.payload
        for artifact in artifacts
        if artifact.type is ArtifactType.REPORT_AUDIT and isinstance(artifact.payload, dict)
    ]
    if not audits:
        audits = [
            audit
            for artifact in artifacts
            if artifact.type is ArtifactType.REPORT_BUNDLE
            for audit in [artifact.payload.get("audit")]
            if isinstance(audit, dict)
        ]
    degraded = 0
    truncations = 0
    numeric_unverified = 0
    coverage_gaps = 0
    verdict: str | None = None
    for audit in audits:
        degraded += _as_int(audit.get("degraded_claim_count"))
        truncations += _as_int(audit.get("time_boundary_truncations"))
        numeric_unverified += _as_int(audit.get("numeric_unverified_claim_count"))
        coverage_gaps += _as_int(audit.get("quantitative_coverage_gap_count"))
        candidate = audit.get("gate_verdict")
        if isinstance(candidate, str) and candidate in _GATE_VERDICT_SEVERITY:
            if verdict is None or (
                _GATE_VERDICT_SEVERITY[candidate] > _GATE_VERDICT_SEVERITY[verdict]
            ):
                verdict = candidate
    return _ReportGateRollup(
        degraded_claims=degraded,
        time_boundary_truncations=truncations,
        numeric_unverified_claims=numeric_unverified,
        quantitative_coverage_gaps=coverage_gaps,
        verdict=verdict,
    )


def _findings_count(artifacts: list[Artifact]) -> int:
    count = 0
    for artifact in artifacts:
        if artifact.type is ArtifactType.VALIDATED_FINDING:
            count += 1
        elif artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            findings = artifact.payload.get("findings")
            if isinstance(findings, list):
                count += len(findings)
    return count


def _apply_question_quality_rollups(metrics: SessionMetrics, artifacts: list[Artifact]) -> None:
    outcome_counts: Counter[str] = Counter()
    contract_failures: Counter[str] = Counter()
    interpretation_counts: Counter[str] = Counter()
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        status = str(artifact.payload.get("status", "failed"))
        outcome = str(
            artifact.payload.get("outcome", "answered" if status == "succeeded" else "failed")
        )
        outcome_counts[outcome] += 1
        code = artifact.payload.get("abstention_code")
        if outcome == "abstained" and isinstance(code, str) and code:
            contract_failures[code] += 1
        interpretation_counts[str(artifact.payload.get("interpretation_status", "absent"))] += 1
    metrics.question_answered = outcome_counts["answered"]
    metrics.question_abstained = outcome_counts["abstained"]
    metrics.question_failed = outcome_counts["failed"]
    metrics.question_awaiting_approval = outcome_counts["awaiting_approval"]
    metrics.result_contract_failures = dict(contract_failures)
    metrics.interpretation_validated = interpretation_counts["validated"]
    metrics.interpretation_fallbacks = interpretation_counts["fallback"]


def _apply_macro_loop_rollups(metrics: SessionMetrics, artifacts: list[Artifact]) -> None:
    """Round-level macro-loop rollup derived from the run's LOOP_LEDGER (design §5.2).

    Re-running the loop persists a fresh ledger; only the newest one counts,
    otherwise repeated invocations double the round totals.
    """
    ledgers = [a for a in artifacts if a.type is ArtifactType.LOOP_LEDGER]
    if not ledgers:
        return
    latest = max(ledgers, key=lambda artifact: artifact.created_at)
    rounds = latest.payload.get("rounds")
    if not isinstance(rounds, list):
        return
    for row in rounds:
        if not isinstance(row, dict):
            continue
        metrics.macro_loop_rounds += 1
        metrics.macro_loop_new_findings += _as_int(row.get("new_validated_findings"))
        if row.get("disposition") == "discard":
            metrics.macro_loop_discard_rounds += 1


def _apply_efficiency_rollups(metrics: SessionMetrics) -> None:
    """Quality/latency/token balance indicators; zero-safe and deterministic."""
    terminal = metrics.question_answered + metrics.question_abstained + metrics.question_failed
    if terminal:
        metrics.question_answer_rate = round(metrics.question_answered / terminal, 6)
        metrics.question_abstention_rate = round(metrics.question_abstained / terminal, 6)
    if metrics.question_answered:
        metrics.tokens_per_answered_question = round(
            metrics.total_tokens / metrics.question_answered, 3
        )
        metrics.seconds_per_answered_question = round(
            metrics.duration_seconds / metrics.question_answered, 3
        )
    report_steps = [step for step in metrics.steps if step.step_name == "export_agentic_report"]
    report_tokens = sum(step.tokens for step in report_steps)
    report_seconds = sum(step.duration_seconds for step in report_steps)
    if metrics.total_tokens:
        metrics.report_token_share = round(report_tokens / metrics.total_tokens, 6)
    if metrics.duration_seconds:
        metrics.report_duration_share = round(report_seconds / metrics.duration_seconds, 6)


def _is_failure(event: TraceEvent) -> bool:
    return (
        event.event_type == _FAILURE_EVENT_TYPE
        or event.event_type.endswith("_failed")
        or event.event_type.endswith("_error")
    )


def _as_int(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0

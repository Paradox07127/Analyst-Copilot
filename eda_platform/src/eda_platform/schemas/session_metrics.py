"""Typed per-run observability rollup derived from traces and artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from eda_platform.schemas.publication import PublicationFreshness, PublicationReadiness
from eda_platform.schemas.resource_metrics import AutoEdaResourceUsage


class StepMetric(BaseModel):
    """Per-pipeline-step slice of the run rollup."""

    step_name: str
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    duration_seconds: float = 0.0


class SessionMetrics(BaseModel):
    """Whole-run observability rollup; artifact prefix ``session_metrics``."""

    schema_version: int = 6
    session_id: str
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    # Run-wide prompt-cache rollup: cache_hit_rate = cached_tokens / prompt_tokens.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    # A subset of completion_tokens; never add it to total_tokens.
    reasoning_tokens: int = 0
    cache_hit_rate: float = 0.0
    est_cost_usd: float | None = None
    # est_cost_usd sums only the calls that could be priced. Without these a
    # partial total is indistinguishable from a complete one.
    costed_calls: int = 0
    uncosted_calls: int = 0
    usage_known_calls: int = 0
    usage_unknown_calls: int = 0
    cost_estimate_status: Literal[
        "complete_estimate", "partial_estimate", "unavailable", "not_applicable"
    ] = "not_applicable"
    llm_calls_by_model: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_kind: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_task: dict[str, int] = Field(default_factory=dict)
    llm_calls_by_status: dict[str, int] = Field(default_factory=dict)
    budget_reserved_calls: int = 0
    budget_settled_calls: int = 0
    budget_rejected_calls: int = 0
    budget_uncertain_calls: int = 0
    budget_total_tokens: int = 0
    budget_est_cost_usd: float | None = None
    budget_reconciliation: Literal["verified", "unverifiable", "not_applicable"] = (
        "not_applicable"
    )
    duration_seconds: float = 0.0
    resource_usage: AutoEdaResourceUsage = Field(default_factory=AutoEdaResourceUsage)
    steps: list[StepMetric] = Field(default_factory=list)
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    findings_count: int = 0
    failures_count: int = 0
    trace_status: Literal["verified", "unverifiable"] = "verified"
    # Question-route degradation and repair metrics.
    question_llm_skipped: bool = False
    question_proposals_dropped: int = 0
    question_dataset_names_resolved: int = 0
    question_list_coercions: int = 0
    # True when question generation was skipped or partially accepted.
    degraded: bool = False
    # Semantic bootstrap, backstop, and relation-discovery metrics.
    semantic_bootstrap_degraded: bool = False
    column_roles_unverified: int = 0
    template_backstop_used: int = 0
    join_candidates_proposed: int = 0
    join_authorizations_fresh: int = 0
    join_authorizations_stale: int = 0
    join_authorizations_unverifiable: int = 0
    relationship_overlap_pairs_evaluated: int = 0
    relationship_overlap_pairs_prefiltered: int = 0
    relationship_full_validations: int = 0
    relationship_coverage_limited: bool = False
    relationship_candidate_payload_bytes: int = 0
    relationship_discovery_deferred: bool = False
    # Worst report-gate verdict and semantic degradation totals.
    semantic_degraded_claims: int = 0
    time_boundary_truncations: int = 0
    numeric_unverified_claims: int = 0
    quantitative_coverage_gaps: int = 0
    report_gate_verdict: str | None = None
    # Evidence interleave, deduplication, and domain-metric totals.
    evidence_interleave_granted: int = 0
    evidence_interleave_rejected: int = 0
    findings_dedup_clusters: int = 0
    findings_dedup_merged: int = 0
    domain_metric_questions: int = 0
    domain_metrics_skipped: int = 0
    # Macro-loop round rollup derived from the run's LOOP_LEDGER artifact.
    macro_loop_rounds: int = 0
    macro_loop_new_findings: int = 0
    macro_loop_discard_rounds: int = 0
    # Result-quality outcomes derived from typed execution artifacts.
    question_answered: int = 0
    question_abstained: int = 0
    question_failed: int = 0
    question_awaiting_approval: int = 0
    result_contract_failures: dict[str, int] = Field(default_factory=dict)
    interpretation_validated: int = 0
    interpretation_fallbacks: int = 0
    publication_readiness: PublicationReadiness = "draft"
    publication_freshness: PublicationFreshness = "not_applicable"
    report_eligible_findings: int = 0
    question_answer_rate: float = 0.0
    question_abstention_rate: float = 0.0
    tokens_per_answered_question: float = 0.0
    seconds_per_answered_question: float = 0.0
    report_token_share: float = 0.0
    report_duration_share: float = 0.0
    coverage_limited: bool = False
    publication_blocked: bool = False

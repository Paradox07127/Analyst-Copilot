from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError

from eda_platform.agents.evidence_interleave import (
    EvidenceInterleaveSession,
    InMemoryEvidenceResolver,
)
from eda_platform.agents.narrative_reviewer import review_narrative
from eda_platform.agents.narrator import narrate_report
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.core.llm import LLMClient, LLMResultMetadata, is_offline_client
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.questions import QuestionExecutionResult
from eda_platform.schemas.reports import (
    EvidenceRequest,
    InterleaveExchange,
    InterleaveTranscript,
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportFocusItem,
    ReportPlanClaim,
    ReportPlanDraft,
    ReportSection,
    ReportSeverity,
    required_report_sections,
)
from eda_platform.tools.domain_metrics import background_section_for
from eda_platform.tools.evidence import (
    EvidenceArtifactSummary,
    EvidencePack,
    PayloadPolicy,
    build_evidence_pack,
)
from eda_platform.tools.report_validator import apply_semantic_gate, validate_report_bundle

_MAX_ATTEMPTS = 3
# F5 budget breaker: stop rewriting once repair rounds have spent more than
# this multiple of the initial draft attempt's total tokens. Spend sums every
# provider call in an attempt (interleave rounds and error retries included);
# calls with missing usage count as 0 without disarming the breaker.
_REPAIR_TOKEN_BUDGET_RATIO = 1.5
_MAX_LLM_CLAIMS = 16
_REPORT_PLAN_TASK = "m2_report_claim_plan"
_BODY_NUMBER_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")
_REPAIRABLE_QUALITY_CODES = {"high_missing", "mixed_type_string", "outlier_detected"}
_SELECTED_FOCUS_SECTION = "Selected Analysis Focus"
_AGENT_ANALYSIS_SECTION = "Agent-Performed Analysis"
_BUSINESS_FINDINGS_SECTION = "Business Findings"
# Bound the model-facing digest size.
_MAX_DIGEST_QUESTIONS = 12
_MAX_DIGEST_FINDINGS_PER_QUESTION = 4
_MAX_BUSINESS_FINDING_FALLBACKS = 3
# Bound read-only evidence requests made while writing the claim plan.
_PLAN_INTERLEAVE_LIMIT = 4
_PLAN_INTERLEAVE_ROUNDS = 3
_PLAN_INTERLEAVE_SECTION = "plan"
# Keep forced numeric-repair reads separate from voluntary plan reads.
_FORCED_INTERLEAVE_LIMIT = 6
_FORCED_INTERLEAVE_SECTION = "numeric_repair"
# Executive summaries omit shape claims and prioritize scored business findings.
_EXEC_SUMMARY_CLAIM_LIMIT = 3
_SHAPE_CLAIM_PATTERN = re.compile(r"\bhas\s+\d[\d,]*\s+(?:rows?|columns?)\b", re.IGNORECASE)
_QUOTED_SPAN_PATTERN = re.compile(r'"[^"]*"')
_UNSCORED_CLAIM_RANK = 0.5
_METRIC_EVIDENCE_KINDS = {"stat", "table", "sql"}


@dataclass(frozen=True)
class LLMTraceEvent:
    task: str
    status: Literal["success", "error"]
    attempt: int
    started_at: datetime
    finished_at: datetime
    usage: LLMResultMetadata | None = None
    error_type: str = ""
    error: str = ""


@dataclass(frozen=True)
class ReportValidationTraceEvent:
    attempt: int
    status: str
    finding_count: int
    critical_count: int
    normalized_body_count: int = 0
    pruned_claim_count: int = 0
    deterministic_repair_count: int = 0
    section_coverage: float = 0.0
    claim_section_coverage: float = 0.0
    claim_survival_rate: float = 1.0
    stopped_no_progress: bool = False
    # F5: rewriting stopped because repair rounds exceeded the token budget.
    budget_stopped: bool = False
    # F5: attempt whose bundle was published after best-attempt selection
    # (0 on events that did not close the repair loop).
    selected_attempt: int = 0
    # F4: LLM claims aimed at the app-owned Selected Analysis Focus section
    # are dropped at plan intake; the drop is traced here, never silent.
    dropped_focus_claim_count: int = 0
    findings: list[str] = field(default_factory=list)
    # Full JSON dumps of each finding, including numeric_details.
    structured_findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgenticReportResult:
    bundle: ReportBundle
    audit: ReportAudit
    evidence_pack: EvidencePack
    llm_calls: list[LLMResultMetadata] = field(default_factory=list)
    llm_events: list[LLMTraceEvent] = field(default_factory=list)
    validation_events: list[ReportValidationTraceEvent] = field(default_factory=list)
    used_fallback: bool = False
    # Full record of evidence requests, when any were made.
    interleave_transcript: InterleaveTranscript | None = None
    # Evidence requests issued automatically for numeric repair.
    forced_evidence_requests: int = 0


def _forced_interleave_note(exchanges: list[InterleaveExchange]) -> str | None:
    """User-facing audit note for the forced evidence channel; only granted
    requests were auto-resolved, so rejected ones are disclosed separately."""
    if not exchanges:
        return None
    granted = sum(1 for exchange in exchanges if exchange.grant is not None)
    rejected = len(exchanges) - granted
    note = (
        f"Forced evidence interleave: {granted} evidence "
        "request(s) auto-resolved for numeric_mismatch repair"
    )
    if rejected:
        note += f"; {rejected} request(s) rejected"
    return note + "."


def generate_agentic_report(
    artifacts: list[Artifact],
    *,
    project_id: str,
    session_id: str,
    business_context: str,
    llm: LLMClient,
    payload_policy: PayloadPolicy = "schema+aggregates",
    enable_semantic_audit: bool = False,
    narrator_llm: LLMClient | None = None,
) -> AgenticReportResult:
    evidence_pack = build_evidence_pack(artifacts, payload_policy=payload_policy)
    question_results, sql_results = _extract_question_evidence(artifacts)
    _register_question_artifacts(evidence_pack, question_results, sql_results)
    llm_calls: list[LLMResultMetadata] = []
    llm_events: list[LLMTraceEvent] = []
    validation_events: list[ReportValidationTraceEvent] = []
    # Both sessions read only persisted artifacts and never run computation.
    interleave_session: EvidenceInterleaveSession | None = None
    forced_session: EvidenceInterleaveSession | None = None
    if not is_offline_client(llm):
        resolver = InMemoryEvidenceResolver(artifacts)
        interleave_session = EvidenceInterleaveSession(
            resolver,
            per_section_limit=_PLAN_INTERLEAVE_LIMIT,
            total_limit=_PLAN_INTERLEAVE_LIMIT,
        )
        forced_session = EvidenceInterleaveSession(
            resolver,
            per_section_limit=_FORCED_INTERLEAVE_LIMIT,
            total_limit=_FORCED_INTERLEAVE_LIMIT,
        )

    bundle, audit, used_fallback = _generate_with_repair(
        evidence_pack,
        project_id=project_id,
        session_id=session_id,
        business_context=business_context,
        llm=llm,
        question_results=question_results,
        sql_results=sql_results,
        llm_calls=llm_calls,
        llm_events=llm_events,
        validation_events=validation_events,
        interleave=interleave_session,
        forced_interleave=forced_session,
    )

    business_injected = _apply_business_findings_fallback(bundle, question_results)
    dataset_injected = _apply_dataset_overview_fallback(bundle, evidence_pack)
    executive_injected = _apply_executive_summary_fallback(bundle, question_results)
    if business_injected or dataset_injected or executive_injected:
        existing_semantic_notes = list(audit.semantic_notes)
        audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
        bundle, audit = _apply_hard_gate(
            bundle,
            audit,
            evidence_pack=evidence_pack,
            attempt=_MAX_ATTEMPTS,
            validation_events=validation_events,
            sql_results=sql_results,
        )
        for note in existing_semantic_notes:
            if note not in audit.semantic_notes:
                audit.semantic_notes.append(note)
        business_injected = sum(
            1
            for claim in _section(bundle, _BUSINESS_FINDINGS_SECTION).claims
            if claim.id.startswith("qbiz_")
        )
        dataset_injected = sum(
            1
            for claim in _section(bundle, "Dataset Overview").claims
            if claim.id.startswith("dataset_overview_")
        )
        executive_injected = sum(
            1
            for claim in _section(bundle, "Executive Summary").claims
            if claim.id.startswith("exec_summary_")
        )
        bundle.status = audit.status

    semantic_notes = list(audit.semantic_notes)
    if used_fallback:
        semantic_notes.append("Deterministic fallback (LLM unavailable or repeatedly invalid).")
    if business_injected:
        semantic_notes.append(
            f"Injected {business_injected} executed finding(s) into Business Findings "
            "(no LLM claim)."
        )
    if dataset_injected:
        semantic_notes.append(
            f"Injected {dataset_injected} profile-backed claim(s) into Dataset Overview."
        )
    if executive_injected:
        semantic_notes.append(
            f"Injected {executive_injected} surviving claim(s) into Executive Summary."
        )
    dropped_focus_claims = sum(
        event.dropped_focus_claim_count for event in validation_events
    )
    if dropped_focus_claims:
        semantic_notes.append(
            f"Dropped {dropped_focus_claims} LLM claim(s) targeting Selected "
            "Analysis Focus across plan attempts (the app owns that section)."
        )
    forced_exchanges = (
        list(forced_session.transcript.exchanges) if forced_session is not None else []
    )
    forced_evidence_requests = len(forced_exchanges)
    forced_note = _forced_interleave_note(forced_exchanges)
    if forced_note:
        semantic_notes.append(forced_note)
    audit.semantic_notes = semantic_notes
    _apply_section_coverage(bundle)
    if enable_semantic_audit:
        bundle = _apply_narrative_review(
            bundle,
            evidence_pack=evidence_pack,
            sql_results=sql_results,
            llm=llm,
            audit=audit,
        )
    # The final semantic gate grades surviving claims without pruning them.
    apply_semantic_gate(
        bundle,
        audit,
        evidence_pack=evidence_pack,
        role_sets=_extract_column_role_sets(artifacts),
        sql_results=sql_results,
        platform_sql_ids=_registry_sql_ids(question_results),
    )
    bundle.status = audit.status
    # Narration runs last, over claims that already cleared every gate, and may
    # only reuse their figures. A narrator failure costs prose, never a claim.
    narration = narrate_report(bundle, llm=narrator_llm or llm)
    if narration.attempted:
        note = (
            f"Wrote a connective narrative for {narration.written} of "
            f"{narration.attempted} eligible section(s), from claims that had "
            "already passed the gates."
        )
        if narration.rejected:
            note += (
                f" {narration.rejected} draft(s) were discarded for carrying a "
                "figure the claims do not state, or for citing a claim that "
                "does not exist; those sections keep their bullets."
            )
        audit.semantic_notes.append(note)
    bundle.audit = audit
    # Persist all write-time evidence reads in one transcript.
    exchanges: list[InterleaveExchange] = []
    if interleave_session is not None:
        exchanges.extend(interleave_session.transcript.exchanges)
    if forced_session is not None:
        exchanges.extend(forced_session.transcript.exchanges)
    interleave_transcript: InterleaveTranscript | None = None
    if exchanges:
        granted_count = sum(1 for exchange in exchanges if exchange.grant is not None)
        interleave_transcript = InterleaveTranscript(
            exchanges=exchanges,
            per_section_limit=_PLAN_INTERLEAVE_LIMIT,
            total_limit=_PLAN_INTERLEAVE_LIMIT + _FORCED_INTERLEAVE_LIMIT,
            granted_count=granted_count,
            rejected_count=len(exchanges) - granted_count,
        )
    return AgenticReportResult(
        bundle=bundle,
        audit=audit,
        evidence_pack=evidence_pack,
        llm_calls=llm_calls,
        llm_events=llm_events,
        validation_events=validation_events,
        used_fallback=used_fallback,
        interleave_transcript=interleave_transcript,
        forced_evidence_requests=forced_evidence_requests,
    )


def _generate_with_repair(
    evidence_pack: EvidencePack,
    *,
    project_id: str,
    session_id: str,
    business_context: str,
    llm: LLMClient,
    question_results: list[QuestionExecutionResult],
    sql_results: dict[str, SqlResult],
    llm_calls: list[LLMResultMetadata],
    llm_events: list[LLMTraceEvent],
    validation_events: list[ReportValidationTraceEvent],
    interleave: EvidenceInterleaveSession | None = None,
    forced_interleave: EvidenceInterleaveSession | None = None,
) -> tuple[ReportBundle, ReportAudit, bool]:
    if is_offline_client(llm):
        bundle = _deterministic_report_bundle(
            evidence_pack, project_id=project_id, session_id=session_id
        )
        _inject_question_claims(bundle, question_results)
        audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
        return bundle, audit, True

    prior_error: str | None = None
    prior_findings: list[str] = []
    prior_bundle: ReportBundle | None = None
    best: _AttemptCandidate | None = None
    forced_evidence: list[dict[str, Any]] = []
    previous_repair_signature: tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None = None
    truncation_retry = False
    raised_completion_budget = False
    draft_total_tokens: int | None = None
    repair_spent_tokens = 0
    for attempt in range(_MAX_ATTEMPTS):
        attempt_number = attempt + 1
        # F5: the budget gate runs before every plan call, so error and
        # truncation retries cannot burn tokens past the breaker.
        if (
            best is not None
            and draft_total_tokens is not None
            and repair_spent_tokens > draft_total_tokens * _REPAIR_TOKEN_BUDGET_RATIO
        ):
            validation_events[-1] = replace(
                validation_events[-1], budget_stopped=True
            )
            gated_bundle, gated_audit = _publish_best_attempt(
                best,
                evidence_pack=evidence_pack,
                validation_events=validation_events,
                sql_results=sql_results,
            )
            gated_audit.semantic_notes.append(
                "Report repair stopped early after exceeding the rewrite token budget."
            )
            return gated_bundle, gated_audit, False
        llm_started_at = datetime.now(UTC)
        attempt_usages: list[LLMResultMetadata] = []
        try:
            draft = _request_plan(
                evidence_pack,
                business_context=business_context,
                llm=llm,
                prior_error=prior_error,
                prior_findings=prior_findings,
                prior_bundle=prior_bundle,
                question_results=question_results,
                truncation_retry=truncation_retry,
                interleave=interleave,
                forced_evidence=forced_evidence,
                usages=attempt_usages,
            )
        except BudgetExceeded:
            raise
        except (RuntimeError, ValidationError) as exc:
            llm_finished_at = datetime.now(UTC)
            usage = attempt_usages[-1] if attempt_usages else None
            completion_cap = _completion_cap(llm)
            truncated = _completion_was_capped(usage, completion_cap)
            prior_error = _retry_error_message(
                exc,
                usage=usage,
                completion_cap=completion_cap,
                truncated=truncated,
            )
            if truncated and not raised_completion_budget and completion_cap is not None:
                raised_completion_budget = _raise_completion_budget(llm, completion_cap)
            truncation_retry = truncated
            if not truncated:
                prior_findings = []
                prior_bundle = None
                forced_evidence = []
            llm_calls.extend(attempt_usages)
            if draft_total_tokens is not None:
                repair_spent_tokens += _attempt_spend(attempt_usages)
            llm_events.append(
                LLMTraceEvent(
                    task=_REPORT_PLAN_TASK,
                    status="error",
                    attempt=attempt_number,
                    started_at=llm_started_at,
                    finished_at=llm_finished_at,
                    usage=usage,
                    error_type="truncation" if truncated else type(exc).__name__,
                    error=prior_error,
                )
            )
            continue

        llm_finished_at = datetime.now(UTC)
        truncation_retry = False
        usage = attempt_usages[-1] if attempt_usages else None
        llm_calls.extend(attempt_usages)
        if draft_total_tokens is None:
            # Missing usage counts as 0 so the breaker stays armed.
            draft_total_tokens = sum(
                item.usage.total_tokens for item in attempt_usages
            )
        else:
            repair_spent_tokens += _attempt_spend(attempt_usages)
        llm_events.append(
            LLMTraceEvent(
                task=_REPORT_PLAN_TASK,
                status="success",
                attempt=attempt_number,
                started_at=llm_started_at,
                finished_at=llm_finished_at,
                usage=usage,
            )
        )

        normalized_body_count = 0
        bundle, dropped_focus_claims = _bundle_from_plan(
            draft, project_id=project_id, session_id=session_id
        )
        # Deterministic question claims never depend on the LLM; inject before
        # validation so they pass the same evidence gate as authored claims.
        _inject_question_claims(bundle, question_results)
        audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
        deterministic_repair_count = _apply_deterministic_repairs(
            bundle, audit, evidence_pack=evidence_pack
        )
        if deterministic_repair_count:
            audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
        validation_events.append(
            _validation_trace_event(
                audit,
                bundle=bundle,
                attempt=attempt_number,
                normalized_body_count=normalized_body_count,
                deterministic_repair_count=deterministic_repair_count,
                dropped_focus_claim_count=len(dropped_focus_claims),
            )
        )
        candidate = _AttemptCandidate(
            attempt=attempt_number,
            bundle=bundle,
            audit=audit,
            sort_key=_attempt_sort_key(bundle, audit),
        )
        # Strict comparison keeps the earliest attempt on ties (stable pick).
        if best is None or candidate.sort_key < best.sort_key:
            best = candidate
        if not audit.has_critical_findings or not _requires_llm_retry(audit):
            # Converged, or only deterministic/prune findings remain: no more
            # LLM rewrites; publish the best attempt seen so far.
            gated_bundle, gated_audit = _publish_best_attempt(
                best,
                evidence_pack=evidence_pack,
                validation_events=validation_events,
                sql_results=sql_results,
            )
            return gated_bundle, gated_audit, False
        if attempt == _MAX_ATTEMPTS - 1:
            gated_bundle, gated_audit = _publish_best_attempt(
                best,
                evidence_pack=evidence_pack,
                validation_events=validation_events,
                sql_results=sql_results,
            )
            return gated_bundle, gated_audit, False
        repair_signature = _report_repair_signature(bundle, audit)
        # Stop when both the plan and validation result repeat unchanged.
        if repair_signature == previous_repair_signature:
            validation_events[-1] = replace(
                validation_events[-1], stopped_no_progress=True
            )
            gated_bundle, gated_audit = _publish_best_attempt(
                best,
                evidence_pack=evidence_pack,
                validation_events=validation_events,
                sql_results=sql_results,
            )
            gated_audit.semantic_notes.append(
                "Report repair stopped early after an identical no-progress retry."
            )
            return gated_bundle, gated_audit, False
        # F5 budget breaker: repair rounds (prompt+completion of every
        # provider call after the first successful draft attempt) must not
        # exceed 1.5x the draft attempt's total tokens.
        if (
            draft_total_tokens is not None
            and repair_spent_tokens > draft_total_tokens * _REPAIR_TOKEN_BUDGET_RATIO
        ):
            validation_events[-1] = replace(
                validation_events[-1], budget_stopped=True
            )
            gated_bundle, gated_audit = _publish_best_attempt(
                best,
                evidence_pack=evidence_pack,
                validation_events=validation_events,
                sql_results=sql_results,
            )
            gated_audit.semantic_notes.append(
                "Report repair stopped early after exceeding the rewrite token budget."
            )
            return gated_bundle, gated_audit, False
        previous_repair_signature = repair_signature
        prior_error = None
        prior_findings = [f"{f.code}: {f.message}" for f in audit.findings]
        prior_bundle = bundle
        # DI10 W3 forced interleave: resolve the authoritative values behind
        # every numeric_mismatch claim now, so the repair round rewrites with
        # the correct numbers instead of guessing (voluntary requests remain).
        forced_evidence = _force_numeric_mismatch_evidence(bundle, audit, forced_interleave)

    if best is not None:
        # Later attempts all raised; publish the best validated attempt.
        gated_bundle, gated_audit = _publish_best_attempt(
            best,
            evidence_pack=evidence_pack,
            validation_events=validation_events,
            sql_results=sql_results,
        )
        return gated_bundle, gated_audit, False

    # Every attempt raised (transport/parse errors) -> deterministic fallback.
    bundle = _deterministic_report_bundle(
        evidence_pack, project_id=project_id, session_id=session_id
    )
    _inject_question_claims(bundle, question_results)
    audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
    return bundle, audit, True


@dataclass(frozen=True)
class _AttemptCandidate:
    attempt: int
    bundle: ReportBundle
    audit: ReportAudit
    sort_key: tuple[int, int, int, float]


def _attempt_sort_key(
    bundle: ReportBundle,
    audit: ReportAudit,
) -> tuple[int, int, int, float]:
    """F5 best-attempt ordering, minimized lexicographically: fewest CRITICAL
    findings, then most verified numeric tokens, most claims, highest claim
    section coverage. Ties are broken by keeping the earlier attempt."""
    critical_count = sum(
        1 for finding in audit.findings if finding.severity is ReportSeverity.CRITICAL
    )
    verified_token_count = sum(
        1
        for section in bundle.sections
        for claim in section.claims
        for status in claim.numeric_statuses
        if status.status == "number_verified"
    )
    return (
        critical_count,
        -verified_token_count,
        -_claim_count(bundle),
        -_claim_section_coverage(bundle),
    )


def _publish_best_attempt(
    best: _AttemptCandidate,
    *,
    evidence_pack: EvidencePack,
    validation_events: list[ReportValidationTraceEvent],
    sql_results: dict[str, SqlResult],
) -> tuple[ReportBundle, ReportAudit]:
    """Hard-gate and publish the selected attempt, stamping the trace."""
    gated_bundle, gated_audit = _apply_hard_gate(
        best.bundle,
        best.audit,
        evidence_pack=evidence_pack,
        attempt=best.attempt,
        validation_events=validation_events,
        sql_results=sql_results,
    )
    validation_events[-1] = replace(
        validation_events[-1], selected_attempt=best.attempt
    )
    return gated_bundle, gated_audit


def _report_repair_signature(
    bundle: ReportBundle,
    audit: ReportAudit,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return the exact observable state that another LLM repair must change."""
    critical_findings = tuple(
        finding.model_dump(mode="json")
        for finding in audit.findings
        if finding.severity is ReportSeverity.CRITICAL and finding.repair_mode == "llm"
    )
    return bundle.model_dump(mode="json"), critical_findings


def _request_plan(
    evidence_pack: EvidencePack,
    *,
    business_context: str,
    llm: LLMClient,
    prior_error: str | None,
    prior_findings: list[str],
    prior_bundle: ReportBundle | None,
    question_results: list[QuestionExecutionResult],
    truncation_retry: bool = False,
    interleave: EvidenceInterleaveSession | None = None,
    forced_evidence: list[dict[str, Any]] | None = None,
    usages: list[LLMResultMetadata] | None = None,
) -> ReportPlanDraft:
    max_claims = 8 if truncation_retry else 12
    instructions = (
        "Produce a compact evidence-grounded EDA claim plan, not a full report. "
        "Return only English claims. Do not write section bodies. The app will "
        "inject all required report sections. Each quantitative or causal statement "
        "must be a claim that cites existing evidence artifact ids. Do not invent "
        "tables, columns, numeric values, totals, ratios, or percentages. EvidenceRef.kind "
        "must be one of stat, profile_field, table, chart, artifact, sql, or code. "
        "Use table for analysis_tables, chart for charts, stat/profile_field for dataset "
        "profile facts, and artifact for quality issues. Use only artifact_id values "
        f"present in evidence_pack.artifact_index. Produce at most {max_claims} claims total; "
        "prefer fewer, stronger claims over completeness. Write every natural-language field "
        "in English. Never write claims for the 'Selected Analysis Focus' section: the app "
        "fills it from executed questions and any claim targeting it is dropped."
    )
    if truncation_retry:
        instructions += (
            " The previous completion reached the token cap and was truncated. "
            "Return fewer and shorter claims, one sentence each, and omit optional detail."
        )
    if question_results:
        instructions += (
            " question_results lists executed analyses; propose 'Business Findings' and "
            "'Business Recommendations' claims from them, citing each finding's "
            "sql_result_artifact_id (kind table) or the qexec artifact_id (kind artifact)."
        )
    payload: dict[str, Any] = {
        "business_context": business_context,
        "allowed_section_titles": required_report_sections(),
        "max_claims": max_claims,
        "instructions": instructions,
    }
    if prior_findings and prior_bundle is not None:
        payload.update(
            {
                "previous_claim_plan": _claim_plan_from_bundle(prior_bundle),
                "previous_validator_findings": prior_findings,
                "repair_instructions": (
                    "Revise only the claims affected by the validator findings. "
                    "Keep supported claims unchanged. Do not add new artifact ids. "
                    "Return the complete revised claim plan."
                ),
            }
        )
    else:
        payload["evidence_manifest"] = _evidence_manifest(evidence_pack)
        if question_results:
            payload["question_results"] = _question_digest(question_results)
    if prior_error:
        payload["previous_error"] = prior_error
    if prior_findings and "previous_validator_findings" not in payload:
        payload["previous_validator_findings"] = prior_findings
    if forced_evidence:
        # DI10 W3 forced interleave: authoritative values pre-resolved from the
        # artifacts cited by claims that failed numeric validation.
        payload["granted_repair_evidence"] = list(forced_evidence)
        payload["granted_repair_instructions"] = (
            "granted_repair_evidence lists values resolved deterministically from "
            "the artifacts cited by the claims that failed numeric validation. "
            "Rewrite those claims using exactly these values; do not invent or "
            "keep unsupported numbers."
        )
    if interleave is not None and interleave.remaining_total > 0:
        payload["evidence_interleave"] = {
            "instructions": (
                "Before finalizing the plan you may return evidence_requests "
                "(artifact_id + locator, from the evidence manifest) instead of "
                "claims; the resolved detail values will be appended as "
                "requested_evidence on the next attempt. At most "
                f"{interleave.remaining_total} request(s) remain for this report."
            ),
        }
    # Resolve bounded persisted-evidence requests before finalizing the plan.
    requested_evidence: list[dict[str, Any]] = []
    draft = ReportPlanDraft()
    for _ in range(_PLAN_INTERLEAVE_ROUNDS):
        try:
            draft = llm.structured(
                task=_REPORT_PLAN_TASK,
                schema=ReportPlanDraft,
                payload=payload,
            )
        finally:
            # F5: every provider call in the attempt is metered, not only the
            # last one; the raising call's usage (when available) counts too.
            _record_call_usage(llm, usages)
        if (
            interleave is None
            or not draft.evidence_requests
            or interleave.remaining_total <= 0
        ):
            return draft
        for request in draft.evidence_requests:
            outcome = interleave.request(request, section=_PLAN_INTERLEAVE_SECTION)
            requested_evidence.append(outcome.model_dump(mode="json"))
        payload = dict(payload)
        payload["requested_evidence"] = list(requested_evidence)
    return draft


def _record_call_usage(
    llm: LLMClient, usages: list[LLMResultMetadata] | None
) -> None:
    if usages is None:
        return
    usage = llm.last_usage()
    # Identity guard: a transport error can leave last_usage() at the previous
    # call's object; never double-count it.
    if usage is not None and (not usages or usages[-1] is not usage):
        usages.append(usage)


def _attempt_spend(usages: list[LLMResultMetadata]) -> int:
    return sum(
        item.usage.prompt_tokens + item.usage.completion_tokens for item in usages
    )


def _completion_cap(llm: LLMClient) -> int | None:
    settings = getattr(llm, "settings", None)
    value = getattr(settings, "max_tokens", None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _completion_was_capped(
    usage: LLMResultMetadata | None,
    completion_cap: int | None,
) -> bool:
    if usage is None or completion_cap is None:
        return False
    return usage.usage.completion_tokens >= completion_cap


def _raise_completion_budget(llm: LLMClient, current_cap: int) -> bool:
    settings = getattr(llm, "settings", None)
    if settings is None:
        return False
    new_cap = min(12_000, max(current_cap + 1, int(current_cap * 1.5)))
    if new_cap <= current_cap:
        return False
    try:
        settings.max_tokens = new_cap
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _retry_error_message(
    exc: RuntimeError | ValidationError,
    *,
    usage: LLMResultMetadata | None,
    completion_cap: int | None,
    truncated: bool,
) -> str:
    base = f"{type(exc).__name__}: {str(exc)[:600]}"
    if not truncated:
        return base[:800]
    completion_tokens = usage.usage.completion_tokens if usage is not None else 0
    return (
        f"Completion truncated at {completion_tokens}/{completion_cap} tokens; "
        "retry with a larger completion budget and fewer, shorter claims. "
        f"Original error: {base}"
    )[:800]


def _bundle_from_plan(
    draft: ReportPlanDraft,
    *,
    project_id: str,
    session_id: str,
) -> tuple[ReportBundle, list[str]]:
    """Build a bundle from the plan; returns dropped focus-section claim ids (F4)."""
    bundle = ReportBundle.empty(project_id=project_id, session_id=session_id)
    section_map = {section.title: section for section in bundle.sections}
    fallback_section = section_map["Appendix: Charts and Technical Summary"]
    dropped_focus_claims: list[str] = []
    for index, plan_claim in enumerate(draft.claims[:_MAX_LLM_CLAIMS], start=1):
        if plan_claim.section_title.casefold() == _SELECTED_FOCUS_SECTION.casefold():
            dropped_focus_claims.append(plan_claim.id or f"claim_{index}")
            continue
        target = section_map.get(plan_claim.section_title, fallback_section)
        claim = _report_claim_from_plan(plan_claim, index=index)
        target.claims.append(claim)
    return bundle, dropped_focus_claims


def _report_claim_from_plan(plan_claim: ReportPlanClaim, *, index: int) -> ReportClaim:
    return ReportClaim(
        id=plan_claim.id or f"claim_{index}",
        text=plan_claim.text,
        evidence=plan_claim.evidence,
        referenced_datasets=plan_claim.referenced_datasets,
        referenced_columns=plan_claim.referenced_columns,
        quality_issue_refs=plan_claim.quality_issue_refs,
        confidence=plan_claim.confidence,
    )


def _validation_trace_event(
    audit: ReportAudit,
    *,
    bundle: ReportBundle,
    attempt: int,
    normalized_body_count: int = 0,
    pruned_claim_count: int = 0,
    deterministic_repair_count: int = 0,
    dropped_focus_claim_count: int = 0,
) -> ReportValidationTraceEvent:
    critical_count = sum(
        1 for finding in audit.findings if finding.severity is ReportSeverity.CRITICAL
    )
    findings = [
        f"{finding.severity.value}:{finding.code}:{finding.message}"
        for finding in audit.findings
    ]
    return ReportValidationTraceEvent(
        attempt=attempt,
        status=audit.status.value,
        finding_count=len(audit.findings),
        critical_count=critical_count,
        normalized_body_count=normalized_body_count,
        pruned_claim_count=pruned_claim_count,
        deterministic_repair_count=deterministic_repair_count,
        dropped_focus_claim_count=dropped_focus_claim_count,
        section_coverage=_rendered_section_coverage(bundle),
        claim_section_coverage=_claim_section_coverage(bundle),
        claim_survival_rate=_claim_survival_rate(bundle, pruned_claim_count),
        findings=findings,
        structured_findings=[
            finding.model_dump(mode="json") for finding in audit.findings
        ],
    )


def _evidence_manifest(evidence_pack: EvidencePack) -> dict[str, Any]:
    return {
        "payload_policy": evidence_pack.payload_policy,
        "artifact_index": {
            artifact_id: summary.model_dump(mode="json")
            for artifact_id, summary in evidence_pack.artifact_index.items()
        },
        "datasets": [
            {
                "artifact_id": dataset.artifact_id,
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "columns": _compact_columns(dataset.columns),
                "dtypes": _compact_mapping(dataset.dtypes),
                "semantic_type_counts": dataset.semantic_type_counts,
                "primary_key_candidates": dataset.primary_key_candidates[:10],
                "missing_percent": _top_mapping(dataset.missing_percent, limit=20),
            }
            for dataset in evidence_pack.datasets
        ],
        "quality_issues": [
            issue.model_dump(mode="json") for issue in evidence_pack.quality_issues[:60]
        ],
        "analysis_tables": [
            {
                "artifact_id": table.artifact_id,
                "dataset_id": table.dataset_id,
                "title": table.title,
                "kind": table.kind,
                "description": table.description,
                "columns": list(table.rows[0].keys()) if table.rows else [],
                "rows_preview": table.rows[:3],
            }
            for table in evidence_pack.analysis_tables[:20]
        ],
        "charts": [chart.model_dump(mode="json") for chart in evidence_pack.charts[:30]],
        "stat_tests": [
            stat_test.model_dump(mode="json") for stat_test in evidence_pack.stat_tests[:20]
        ],
        "model_cards": [
            model_card.model_dump(mode="json") for model_card in evidence_pack.model_cards[:10]
        ],
    }


def _compact_columns(columns: list[str], *, limit: int = 40) -> list[str] | dict[str, Any]:
    if len(columns) <= limit:
        return columns
    return {
        "total": len(columns),
        "head": columns[:20],
        "tail": columns[-10:],
        "omitted_count": len(columns) - 30,
    }


def _compact_mapping(values: dict[str, Any], *, limit: int = 60) -> dict[str, Any]:
    if len(values) <= limit:
        return values
    selected = list(values.items())[:limit]
    compact = dict(selected)
    compact["_omitted_count"] = len(values) - limit
    return compact


def _top_mapping(values: dict[str, float], *, limit: int) -> dict[str, float]:
    return dict(sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit])


def _claim_plan_from_bundle(bundle: ReportBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in bundle.sections:
        for claim in section.claims:
            payload = claim.model_dump(mode="json")
            payload["section_title"] = section.title
            # Validator-derived state stays out of the repair prompt: the LLM
            # must not react to unverified claims no finding asked it to touch.
            payload["numeric_statuses"] = []
            payload["numeric_rollup"] = "not_evaluated"
            payload["quantitative_coverage_gap"] = False
            payload["deterministic_source"] = False
            rows.append(payload)
    return rows


def _apply_deterministic_repairs(
    bundle: ReportBundle,
    audit: ReportAudit,
    *,
    evidence_pack: EvidencePack,
) -> int:
    repair_count = 0
    repair_count += _dedupe_claim_evidence(bundle)
    for finding in audit.findings:
        if finding.claim_id is None and finding.code in {
            "unsupported_section_body",
        }:
            repair_count += _clear_section_body(bundle, finding.section_title)
            continue
        if finding.code != "missing_quality_warning":
            continue
        claim = _find_claim(bundle, finding.section_title, finding.claim_id)
        if claim is None:
            continue
        repair_count += _attach_quality_warnings(claim, evidence_pack)
    return repair_count


def _clear_section_body(bundle: ReportBundle, section_title: str | None) -> int:
    if section_title is None:
        return 0
    for section in bundle.sections:
        if section.title == section_title and section.body:
            section.body = ""
            return 1
    return 0


def _dedupe_claim_evidence(bundle: ReportBundle) -> int:
    removed = 0
    for section in bundle.sections:
        for claim in section.claims:
            seen: set[tuple[str, str | None, str, str, str, str | None, str | None]] = set()
            deduped: list[EvidenceRef] = []
            for evidence in claim.evidence:
                key = (
                    evidence.kind,
                    evidence.artifact_id,
                    evidence.locator,
                    repr(evidence.value),
                    evidence.unit,
                    evidence.unit_label,
                    evidence.unit_reference,
                )
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                deduped.append(evidence)
            claim.evidence = deduped
    return removed


def _attach_quality_warnings(
    claim: ReportClaim,
    evidence_pack: EvidencePack,
) -> int:
    repairs = 0
    risky_issues = [
        issue
        for issue in evidence_pack.quality_issues
        if issue.column in claim.referenced_columns and issue.code in _REPAIRABLE_QUALITY_CODES
    ]
    for issue in risky_issues:
        if issue.artifact_id not in claim.quality_issue_refs:
            claim.quality_issue_refs.append(issue.artifact_id)
            repairs += 1
        if not any(
            ref.kind == "artifact"
            and ref.artifact_id == issue.artifact_id
            and ref.locator == f"quality_issue:{issue.code}:{issue.column or ''}"
            for ref in claim.evidence
        ):
            claim.evidence.append(
                EvidenceRef(
                    kind="artifact",
                    artifact_id=issue.artifact_id,
                    locator=f"quality_issue:{issue.code}:{issue.column or ''}",
                )
            )
            repairs += 1
    if risky_issues and "Quality caveat:" not in claim.text:
        claim.text = (
            f"{claim.text} Quality caveat: referenced high-risk columns have "
            "validator quality warnings."
        )
        repairs += 1
    return repairs


def _find_claim(
    bundle: ReportBundle,
    section_title: str | None,
    claim_id: str | None,
) -> ReportClaim | None:
    if section_title is None or claim_id is None:
        return None
    for section in bundle.sections:
        if section.title != section_title:
            continue
        for claim_index, claim in enumerate(section.claims):
            if (claim.id or f"{section.title}:{claim_index}") == claim_id:
                return claim
    return None


def _force_numeric_mismatch_evidence(
    bundle: ReportBundle,
    audit: ReportAudit,
    session: EvidenceInterleaveSession | None,
) -> list[dict[str, Any]]:
    """Resolve cited evidence for numeric-mismatch claims within a bounded session."""
    if session is None:
        return []
    outcomes: list[dict[str, Any]] = []
    requested: set[tuple[str, str]] = set()
    for finding in audit.findings:
        if finding.code != "numeric_mismatch" or finding.severity is not ReportSeverity.CRITICAL:
            continue
        claim = _find_claim(bundle, finding.section_title, finding.claim_id)
        if claim is None:
            continue
        for reference in claim.evidence:
            if not reference.artifact_id:
                continue
            key = (reference.artifact_id, reference.locator)
            if key in requested:
                continue
            if session.remaining_total <= 0:
                return outcomes
            requested.add(key)
            outcome = session.request(
                EvidenceRequest(
                    artifact_id=reference.artifact_id,
                    locator=reference.locator,
                    section=_FORCED_INTERLEAVE_SECTION,
                    reason=f"forced numeric_mismatch repair for claim {finding.claim_id}",
                ),
                section=_FORCED_INTERLEAVE_SECTION,
            )
            entry = outcome.model_dump(mode="json")
            entry["claim_id"] = finding.claim_id
            outcomes.append(entry)
    return outcomes


def _requires_llm_retry(audit: ReportAudit) -> bool:
    return any(
        finding.severity is ReportSeverity.CRITICAL and finding.repair_mode == "llm"
        for finding in audit.findings
    )


def _apply_hard_gate(
    bundle: ReportBundle,
    audit: ReportAudit,
    *,
    evidence_pack: EvidencePack,
    attempt: int,
    validation_events: list[ReportValidationTraceEvent],
    sql_results: dict[str, SqlResult] | None = None,
) -> tuple[ReportBundle, ReportAudit]:
    pruned_count = _prune_critical_claims(bundle, audit)
    if pruned_count == 0:
        return bundle, audit

    gated_audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
    gated_audit.semantic_notes.append(
        f"Hard validator removed {pruned_count} unsupported claim(s)."
    )
    validation_events.append(
        _validation_trace_event(
            gated_audit,
            bundle=bundle,
            attempt=attempt,
            pruned_claim_count=pruned_count,
        )
    )
    return bundle, gated_audit


def _prune_critical_claims(bundle: ReportBundle, audit: ReportAudit) -> int:
    targets = {
        (finding.section_title, finding.claim_id)
        for finding in audit.findings
        if finding.severity is ReportSeverity.CRITICAL
        and finding.section_title is not None
        and finding.claim_id is not None
    }
    if not targets:
        return 0

    pruned_count = 0
    for section in bundle.sections:
        retained: list[ReportClaim] = []
        for claim_index, claim in enumerate(section.claims):
            claim_id = claim.id or f"{section.title}:{claim_index}"
            if (section.title, claim_id) in targets:
                pruned_count += 1
            else:
                retained.append(claim)
        section.claims = retained
    return pruned_count


def _claim_count(bundle: ReportBundle) -> int:
    return sum(len(section.claims) for section in bundle.sections)


def _apply_section_coverage(bundle: ReportBundle) -> None:
    for section in bundle.sections:
        section.body = section.structural_body()


def _apply_narrative_review(
    bundle: ReportBundle,
    *,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
    llm: LLMClient,
    audit: ReportAudit,
) -> ReportBundle:
    """Review business prose after the hard gate without changing structured claims."""
    if is_offline_client(llm):
        audit.semantic_notes.append(
            "Narrative review skipped: LLM unavailable; hard validator remains release gate."
        )
        return bundle
    result = review_narrative(
        bundle,
        evidence_pack=evidence_pack,
        llm=llm,
        sql_results=sql_results,
    )
    reviewed = sum(1 for event in result.events if event.status == "reviewed")
    reverted = sum(1 for event in result.events if event.status == "reverted")
    skipped = sum(1 for event in result.events if event.status == "skipped")
    audit.semantic_notes.append(
        f"Narrative review: {reviewed} deepened, {reverted} reverted, {skipped} skipped; "
        "hard validator remains release gate."
    )
    return result.bundle


def _rendered_section_coverage(bundle: ReportBundle) -> float:
    required = required_report_sections()
    if not required:
        return 0.0
    present = {section.title for section in bundle.sections}
    return round(len([title for title in required if title in present]) / len(required), 4)


def _claim_section_coverage(bundle: ReportBundle) -> float:
    if not bundle.sections:
        return 0.0
    covered = sum(1 for section in bundle.sections if section.claims)
    return round(covered / len(bundle.sections), 4)


def _claim_survival_rate(bundle: ReportBundle, pruned_claim_count: int) -> float:
    retained = _claim_count(bundle)
    total = retained + pruned_claim_count
    if total == 0:
        return 1.0
    return round(retained / total, 4)


def _deterministic_report_bundle(
    evidence_pack: EvidencePack,
    *,
    project_id: str,
    session_id: str,
) -> ReportBundle:
    bundle = ReportBundle.empty(project_id=project_id, session_id=session_id)
    # Cover every dataset while preserving the first dataset's legacy IDs.
    for dataset_index, dataset in enumerate(evidence_pack.datasets):
        _section(bundle, "Dataset Overview").claims.append(
            ReportClaim(
                id=(
                    "dataset_row_count"
                    if dataset_index == 0
                    else f"dataset_row_count_{dataset.dataset_id}"
                ),
                text=f"{dataset.name} has {dataset.row_count} rows.",
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=dataset.artifact_id,
                        locator="rows",
                        value=dataset.row_count,
                    )
                ],
                referenced_datasets=[dataset.name],
                confidence="high",
            )
        )
        _section(bundle, "File-by-File EDA Summary").claims.append(
            ReportClaim(
                id=(
                    "dataset_column_count"
                    if dataset_index == 0
                    else f"dataset_column_count_{dataset.dataset_id}"
                ),
                text=f"{dataset.name} has {dataset.column_count} columns.",
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=dataset.artifact_id,
                        locator="columns",
                        value=dataset.column_count,
                    )
                ],
                referenced_datasets=[dataset.name],
                confidence="high",
            )
        )
    if evidence_pack.quality_issues:
        # Scope quality counts to their source artifact.
        issues_by_artifact: dict[str, list[Any]] = {}
        for issue in evidence_pack.quality_issues:
            issues_by_artifact.setdefault(issue.artifact_id, []).append(issue)
        dataset_names = {
            dataset.dataset_id: dataset.name for dataset in evidence_pack.datasets
        }
        for issue_index, (artifact_id, issues) in enumerate(issues_by_artifact.items()):
            dataset_id = issues[0].dataset_id
            dataset_name = dataset_names.get(dataset_id) or dataset_id
            issue_count = len(issues)
            _section(bundle, "Data Quality Findings").claims.append(
                ReportClaim(
                    id=(
                        "quality_issue_count"
                        if issue_index == 0
                        else f"quality_issue_count_{dataset_id}"
                    ),
                    text=(
                        f"{dataset_name} quality scan found {issue_count} "
                        f"{'issue' if issue_count == 1 else 'issues'}."
                    ),
                    evidence=[
                        EvidenceRef(
                            kind="artifact",
                            artifact_id=artifact_id,
                            locator="issues",
                            value=issue_count,
                        )
                    ],
                    referenced_datasets=[dataset_name],
                    quality_issue_refs=[artifact_id],
                    confidence="high",
                    # F3 exemption kept for legacy artifacts only: their
                    # QualityIssue prose resolves no numbers (F1). Structured
                    # sets (§11.3) resolve the "issues" locator to the set
                    # cardinality, so on new runs this count verifies and the
                    # exemption becomes a no-op.
                    deterministic_source=True,
                )
            )
    if evidence_pack.analysis_tables:
        table = evidence_pack.analysis_tables[0]
        _section(bundle, "Agent-Performed Analysis").claims.append(
            ReportClaim(
                id="analysis_table_available",
                text=f"Analysis table {table.title} is available.",
                evidence=[
                    EvidenceRef(
                        kind="artifact",
                        artifact_id=table.artifact_id,
                        locator="table",
                    )
                ],
                confidence="high",
            )
        )
    if evidence_pack.stat_tests:
        stat_test = evidence_pack.stat_tests[0]
        if stat_test.p_value is None:
            p_value_text = "not available"
        elif stat_test.p_value <= 0:
            p_value_text = "below floating-point resolution"
        else:
            p_value_text = f"{stat_test.p_value:g}"
        _section(bundle, "Agent-Performed Analysis").claims.append(
            ReportClaim(
                id="stat_test_available",
                text=(
                    f"{stat_test.test_type} ran on {stat_test.dataset_id} "
                    f"with p-value {p_value_text}."
                ),
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=stat_test.artifact_id,
                        locator="p_value",
                        value=stat_test.p_value,
                    )
                ],
                referenced_columns=[
                    column
                    for column in [stat_test.group_column, stat_test.value_column]
                    if column is not None
                ],
                confidence="high",
            )
        )
    if evidence_pack.model_cards:
        card = evidence_pack.model_cards[0]
        metric_name, metric_value = next(iter(card.metrics.items()), ("metric", None))
        _section(bundle, "Agent-Performed Analysis").claims.append(
            ReportClaim(
                id="model_card_available",
                text=(
                    f"{card.model_type} baseline for {card.target_column} used "
                    f"{card.split_strategy} split"
                    + (
                        f" and recorded {metric_name} {metric_value:g}."
                        if isinstance(metric_value, int | float)
                        else "."
                    )
                ),
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=card.artifact_id,
                        locator=f"metrics.{metric_name}",
                        value=metric_value if isinstance(metric_value, int | float) else None,
                    )
                ],
                referenced_columns=[card.target_column],
                confidence="high",
            )
        )
    if evidence_pack.charts:
        chart = evidence_pack.charts[0]
        _section(bundle, "Appendix: Charts and Technical Summary").claims.append(
            ReportClaim(
                id="chart_available",
                text=f"Chart {chart.title} is available.",
                evidence=[
                    EvidenceRef(
                        kind="chart",
                        artifact_id=chart.artifact_id,
                        locator="chart",
                    )
                ],
                confidence="high",
            )
        )
    return bundle


def _section(bundle: ReportBundle, title: str):
    return next(section for section in bundle.sections if section.title == title)


def _extract_column_role_sets(artifacts: list[Artifact]) -> list[ColumnRoleSet]:
    """Parse valid column-role artifacts, tolerating missing or malformed payloads."""
    role_sets: list[ColumnRoleSet] = []
    for artifact in artifacts:
        if artifact.type is not ArtifactType.COLUMN_ROLE_SET:
            continue
        try:
            role_sets.append(ColumnRoleSet.model_validate(artifact.payload))
        except (ValidationError, TypeError):
            continue
    return role_sets


def _extract_question_evidence(
    artifacts: list[Artifact],
) -> tuple[list[QuestionExecutionResult], dict[str, SqlResult]]:
    """Extract valid question-execution and SQL-result payloads from run artifacts."""
    question_results: list[QuestionExecutionResult] = []
    sql_results: dict[str, SqlResult] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            try:
                question_results.append(QuestionExecutionResult.model_validate(artifact.payload))
            except (ValidationError, TypeError):
                continue
        elif artifact.type is ArtifactType.SQL_RESULT:
            try:
                sql_results[artifact.id] = SqlResult.model_validate(artifact.payload)
            except (ValidationError, TypeError):
                continue
    return question_results, sql_results


def _register_question_artifacts(
    evidence_pack: EvidencePack,
    question_results: list[QuestionExecutionResult],
    sql_results: dict[str, SqlResult],
) -> None:
    """Register question-chain artifacts required by evidence existence checks."""
    known = evidence_pack.artifact_index

    def _register(artifact_id: str | None) -> None:
        if artifact_id and artifact_id not in known:
            known[artifact_id] = EvidenceArtifactSummary(
                artifact_id=artifact_id,
                artifact_type=ArtifactType.SQL_RESULT.value,
                title="Question analysis artifact",
            )

    for artifact_id in sql_results:
        _register(artifact_id)
    for question in question_results:
        _register(question.sql_result_artifact_id)
        for finding in question.findings:
            for ref in finding.evidence:
                _register(ref.artifact_id)


def _inject_question_claims(
    bundle: ReportBundle,
    question_results: list[QuestionExecutionResult],
) -> None:
    """Inject structured focus items and evidence-backed finding claims (F4)."""
    if not question_results:
        return
    focus_section = _section(bundle, _SELECTED_FOCUS_SECTION)
    analysis_section = _section(bundle, _AGENT_ANALYSIS_SECTION)
    for question in question_results:
        # A background metric describes the data; filing it as an analysis focus
        # and promoting it into the summary is how a barely-off-even HHI became
        # the World Cup report's headline (2026-08-04). `qbg_` keeps it out of
        # the Executive Summary picker, which selects on the `qfind_`/`qbiz_`
        # prefixes.
        background = background_section_for(question.metric_id)
        if background is None:
            _upsert_focus_item(
                focus_section,
                ReportFocusItem(
                    question=question.question,
                    outcome=question.outcome or question.status,
                    question_id=question.question_id,
                    reason=question.failure_reason,
                ),
            )
        target = _section(bundle, background) if background else analysis_section
        prefix = "qbg" if background else "qfind"
        for f_index, finding in enumerate(question.findings):
            if finding.dedup_role == "supporting":
                continue
            _upsert_claim(
                target,
                ReportClaim(
                    id=f"{prefix}_{question.question_id}_{f_index}",
                    text=finding.text,
                    evidence=list(finding.evidence),
                    confidence="low" if question.exploratory else "high",
                ),
            )


def _upsert_claim(section: ReportSection, claim: ReportClaim) -> None:
    if claim.id is not None:
        for index, existing in enumerate(section.claims):
            if existing.id == claim.id:
                section.claims[index] = claim
                return
    section.claims.append(claim)


def _upsert_focus_item(section: ReportSection, item: ReportFocusItem) -> None:
    key = item.question_id or item.question
    for index, existing in enumerate(section.focus_items):
        if (existing.question_id or existing.question) == key:
            section.focus_items[index] = item
            return
    section.focus_items.append(item)


def _registry_sql_ids(question_results: list[QuestionExecutionResult]) -> set[str]:
    """SQL results the platform's own metric templates produced.

    A registry metric carries a `metric_id` and its SQL comes from the template
    that metric owns, so unlike a planned query its coverage is known.
    """
    return {
        result.sql_result_artifact_id
        for result in question_results
        if result.metric_id and result.sql_result_artifact_id
    }


def _apply_business_findings_fallback(
    bundle: ReportBundle,
    question_results: list[QuestionExecutionResult],
) -> int:
    """Inject top evidence-backed findings when no business claims were authored."""
    section = _section(bundle, _BUSINESS_FINDINGS_SECTION)
    if section.claims:
        return 0
    # Every executed finding is already published verbatim under Agent-Performed
    # Analysis, so copying one here produces a duplicate the exporter replaces
    # with `See "Agent-Performed Analysis" ...`. Across three live runs the
    # section held nothing but those stubs (2026-08-05). Filling an empty
    # section with pointers to the next one is not filling it.
    published = {
        claim.text for claim in _section(bundle, _AGENT_ANALYSIS_SECTION).claims
    }
    findings = [
        (question, finding)
        for question in question_results
        if question.status == "succeeded"
        # A background metric is not a business finding, and `qbiz_` is a prefix
        # the Executive Summary picker selects on: the 2026-08-05 offline run
        # filed time coverage under Dataset Overview and this fallback put it on
        # the cover anyway, through the other door.
        and background_section_for(question.metric_id) is None
        for finding in question.findings
        if finding.dedup_role != "supporting" and finding.text not in published
    ][:_MAX_BUSINESS_FINDING_FALLBACKS]
    for index, (question, finding) in enumerate(findings):
        section.claims.append(
            ReportClaim(
                id=f"qbiz_{question.question_id}_{index}",
                text=finding.text,
                evidence=list(finding.evidence),
                confidence="low" if question.exploratory else "medium",
            )
        )
    return len(findings)


def _apply_dataset_overview_fallback(
    bundle: ReportBundle,
    evidence_pack: EvidencePack,
) -> int:
    section = _section(bundle, "Dataset Overview")
    # A background metric's line is filed here but is not an authored summary,
    # so it must not stand in for one — otherwise routing time coverage into
    # this section silently costs the reader the row and column counts.
    authored = [claim for claim in section.claims if not (claim.id or "").startswith("qbg_")]
    if authored or not evidence_pack.datasets:
        return 0
    for dataset in evidence_pack.datasets:
        section.claims.append(
            ReportClaim(
                id=f"dataset_overview_{dataset.dataset_id}",
                text=(
                    f"{dataset.name} has {dataset.row_count} rows and "
                    f"{dataset.column_count} columns."
                ),
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=dataset.artifact_id,
                        locator="rows",
                        value=dataset.row_count,
                    ),
                    EvidenceRef(
                        kind="stat",
                        artifact_id=dataset.artifact_id,
                        locator="columns",
                        value=dataset.column_count,
                    ),
                ],
                referenced_datasets=[dataset.name],
                confidence="high",
            )
        )
    return len(evidence_pack.datasets)


def _apply_executive_summary_fallback(
    bundle: ReportBundle,
    question_results: list[QuestionExecutionResult],
) -> int:
    """Inject a ranked deterministic summary when no summary was authored."""
    section = _section(bundle, "Executive Summary")
    if section.claims:
        return 0
    source_claims = [
        claim
        for source_section in bundle.sections
        if source_section.title != "Executive Summary"
        for claim in source_section.claims
    ]
    qualified = [
        claim for claim in source_claims if not _is_shape_claim_text(claim.text)
    ]
    # Prefer answered business questions over generic technical claims.
    question_qualified = [
        claim
        for claim in qualified
        if (claim.id or "").startswith(("qfind_", "qbiz_"))
    ]
    if question_qualified:
        qualified = question_qualified
    if qualified:
        selected = _select_executive_claims(qualified, question_results)
    else:
        selected = source_claims[:_EXEC_SUMMARY_CLAIM_LIMIT]
    injected = 0
    for index, claim in enumerate(selected, start=1):
        section.claims.append(
            ReportClaim(
                id=f"exec_summary_{claim.id or index}",
                text=f"Summary: {claim.text}",
                evidence=list(claim.evidence),
                referenced_datasets=list(claim.referenced_datasets),
                referenced_columns=list(claim.referenced_columns),
                quality_issue_refs=list(claim.quality_issue_refs),
                confidence=claim.confidence,
            )
        )
        injected += 1
    return injected


def _select_executive_claims(
    qualified: list[ReportClaim],
    question_results: list[QuestionExecutionResult],
) -> list[ReportClaim]:
    """Rank qualified claims for the executive summary."""
    ranked = sorted(
        enumerate(qualified),
        key=lambda item: (
            -_question_rank_for_claim(item[1], question_results),
            0 if _claim_text_has_number(item[1].text) else 1,
            0 if _has_metric_backed_evidence(item[1]) else 1,
            item[0],
        ),
    )
    selected: list[ReportClaim] = []
    seen_texts: set[str] = set()
    for _, claim in ranked:
        text_key = " ".join(claim.text.split()).lower()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        selected.append(claim)
        if len(selected) == _EXEC_SUMMARY_CLAIM_LIMIT:
            break
    if selected and not any(_claim_text_has_number(claim.text) for claim in selected):
        numeric = next(
            (claim for _, claim in ranked if _claim_text_has_number(claim.text)),
            None,
        )
        if numeric is not None and numeric not in selected:
            selected[-1] = numeric
    return selected


def _is_shape_claim_text(text: str) -> bool:
    """Return whether text is a row- or column-count claim."""
    return bool(_SHAPE_CLAIM_PATTERN.search(text))


def _claim_text_has_number(text: str) -> bool:
    """Return whether a claim asserts a number outside quoted citations."""
    unquoted = _QUOTED_SPAN_PATTERN.sub(" ", text)
    return bool(_BODY_NUMBER_PATTERN.search(unquoted))


def _has_metric_backed_evidence(claim: ReportClaim) -> bool:
    return any(reference.kind in _METRIC_EVIDENCE_KINDS for reference in claim.evidence)


def _question_rank_for_claim(
    claim: ReportClaim,
    question_results: list[QuestionExecutionResult],
) -> float:
    """Return the best finding score for a question-derived claim, or a neutral rank."""
    claim_id = claim.id or ""
    for question in question_results:
        question_id = question.question_id
        if claim_id.startswith((f"qfind_{question_id}_", f"qbiz_{question_id}_")):
            scores = [
                finding.score.final
                for finding in question.findings
                if finding.score is not None
            ]
            return max(scores) if scores else _UNSCORED_CLAIM_RANK
    return _UNSCORED_CLAIM_RANK


def _question_digest(
    question_results: list[QuestionExecutionResult],
) -> list[dict[str, Any]]:
    """Compact per-question digest for the LLM manifest (bounded to stay small)."""
    digest: list[dict[str, Any]] = []
    for question in question_results[:_MAX_DIGEST_QUESTIONS]:
        findings = [
            {
                "text": finding.text,
                "evidence": [
                    {
                        "artifact_id": ref.artifact_id,
                        "locator": ref.locator,
                        "value": ref.value,
                        "unit": ref.unit,
                        **({"unit_label": ref.unit_label} if ref.unit_label else {}),
                        **(
                            {"unit_reference": ref.unit_reference}
                            if ref.unit_reference
                            else {}
                        ),
                    }
                    for ref in finding.evidence[:3]
                ],
            }
            for finding in question.findings[:_MAX_DIGEST_FINDINGS_PER_QUESTION]
        ]
        digest.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "origin": question.origin,
                "status": question.status,
                "outcome": question.outcome,
                "abstention_code": question.abstention_code,
                "sql_result_artifact_id": question.sql_result_artifact_id,
                "findings": findings,
            }
        )
    return digest

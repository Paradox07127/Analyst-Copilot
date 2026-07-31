"""Assemble an evidence-bounded SCQA decision report from a synthesis brief."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

from pydantic import BaseModel, Field, ValidationError

from eda_platform.agents.evidence_interleave import (
    EvidenceInterleaveSession,
    StoreEvidenceResolver,
)
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.claim_language import CAUSAL_PHRASES, contains_causal_phrase
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.core.publication_fingerprint import (
    DECISION_REPORT_POLICY_VERSION,
    decision_report_input_fingerprint,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.trace import trace_event
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.decision_report import (
    DecisionReport,
    DecisionReportSection,
    MetaInsightSkeleton,
    SCQAFrame,
)
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.quality_context import QualityContext
from eda_platform.schemas.reports import EvidenceRequest
from eda_platform.schemas.synthesis import SynthesisBrief

_NUMBER_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")
_CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?:"
    r"(?P<prefix_code>[A-Z]{3})\s+(?P<prefix_value>-?\d+(?:\.\d+)?)"
    r"|"
    r"(?P<suffix_value>-?\d+(?:\.\d+)?)\s+(?P<suffix_code>[A-Z]{3})"
    r"(?:/order|\s+per\s+order)?"
    r")"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# Shared causal family (core.claim_language) plus the two bare-"cause" forms this
# report-level gate scans for on top of the shared phrases.
_CAUSAL_PHRASES = (*CAUSAL_PHRASES, "cause ", "cause.")
_REFINEMENT_TASK = "di4_decision_report_scqa_refinement"

# Bounds for evidence requests made during report rewriting.
_INTERLEAVE_PER_SECTION_LIMIT = 2
_INTERLEAVE_TOTAL_LIMIT = 8
_INTERLEAVE_MAX_ROUNDS = 5
_SCQA_SECTIONS = ("situation", "complication", "answer")

# Deterministic markers used to identify exceptions to the shared pattern.
_EXCEPTION_TEXT_MARKERS = (
    "anomal",
    "outlier",
    "flagged observation",
    "deviat",
    "exception",
    "contrary",
    "runs against",
)
_META_TOP_K = 5
# A neutral fallback preserves relative order for unscored findings.
_UNSCORED_RANK = 0.5

# Aggregate repeated non-critical quality conditions without hiding critical items.
_QUALITY_CONDITION_AGGREGATE_THRESHOLD = 3
_QUALITY_CONDITION_AGGREGATE_TEMPLATE = (
    "the {code} condition was observed across multiple columns; interpret "
    "affected values with care (details on the Quality page)"
)


class _SCQARewrite(BaseModel):
    """The deliberately narrow output surface available to the LLM."""

    situation: str
    complication: str
    answer: str


class _SCQAInterleavedRewrite(BaseModel):
    """Interleaved-writing output surface."""

    situation: str = ""
    complication: str = ""
    answer: str = ""
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


def create_decision_report(
    store: ArtifactStore,
    *,
    project_id: str,
    brief_artifact_id: str,
    brief_session_id: str | None = None,
    llm: LLMClient | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Create and persist a DecisionReport, returning its artifact id."""
    raise_if_cancelled(cancel_check, operation="decision report")
    exact_brief_session_id = brief_session_id or _unique_artifact_run(
        store,
        project_id=project_id,
        artifact_id=brief_artifact_id,
    )
    brief_artifact = store.get_artifact(
        brief_artifact_id,
        project_id=project_id,
        session_id=exact_brief_session_id,
    )
    if brief_artifact.project_id != project_id:
        raise ValueError("The synthesis brief belongs to a different project.")
    if brief_artifact.type is not ArtifactType.SYNTHESIS_BRIEF:
        raise ValueError("brief_artifact_id must identify a SynthesisBrief artifact.")
    brief = SynthesisBrief.model_validate(brief_artifact.payload)
    if brief.project_id != project_id:
        raise ValueError("The synthesis brief payload belongs to a different project.")
    if not brief.report_eligible:
        raise ValueError("Only a report-eligible synthesis brief can become a decision report.")

    finding_artifacts, findings = _load_findings(
        store,
        project_id=project_id,
        artifact_ids=brief.selected_finding_artifact_ids,
        artifact_session_ids=brief.selected_finding_session_ids,
    )
    raise_if_cancelled(cancel_check, operation="decision report")
    finding_artifacts, findings = _rank_by_impact_significance(finding_artifacts, findings)
    evidence = _evidence_values(findings)
    report = _deterministic_report(
        brief,
        findings,
        evidence,
        ordered_finding_ids=[artifact.id for artifact in finding_artifacts],
    )
    if llm is not None and not is_offline_client(llm):
        # Fall back to the deterministic report if evidence interleaving fails.
        session = _interleave_session(
            store,
            project_id=project_id,
            session_id=brief_artifact.session_id,
            finding_artifacts=finding_artifacts,
            findings=findings,
        )
        deterministic_report = report
        report = _refine_scqa(
            report,
            llm=llm,
            evidence=evidence,
            session=session,
            cancel_check=cancel_check,
        )
        raise_if_cancelled(cancel_check, operation="decision report")
        transcript_persisted = _persist_interleave_transcript(
            store,
            report=report,
            session=session,
            project_id=project_id,
            session_id=brief_artifact.session_id,
            brief_artifact_id=brief_artifact_id,
        )
        if not transcript_persisted:
            # A refined report must not outlive the persisted provenance for
            # evidence fetched during interleave. Fall back to the fully
            # evidence-bounded deterministic report.
            report = deterministic_report

    report.publication_input_fingerprint = decision_report_input_fingerprint(finding_artifacts)
    report.report_policy_version = DECISION_REPORT_POLICY_VERSION
    report.source_finding_session_ids = {
        artifact.id: artifact.session_id for artifact in finding_artifacts
    }

    payload = report.model_dump(mode="json")
    parents = [brief_artifact_id, *[artifact.id for artifact in finding_artifacts]]
    if report.interleave_transcript_artifact_id is not None:
        parents.append(report.interleave_transcript_artifact_id)
    artifact = Artifact(
        id=make_artifact_id(
            "dreport",
            {
                "brief_artifact_id": brief_artifact_id,
                "finding_artifact_ids": report.source_finding_artifact_ids,
                "report": payload,
            },
        ),
        type=ArtifactType.DECISION_REPORT,
        project_id=project_id,
        session_id=brief_artifact.session_id,
        parents=parents,
        payload=payload,
        plain_language=(
            "An SCQA decision report assembled from selected validated findings; "
            "every rendered number is checked against finding evidence."
        ),
    )
    raise_if_cancelled(cancel_check, operation="decision report")
    store.save_artifact(artifact)
    return artifact.id


def _interleave_session(
    store: ArtifactStore,
    *,
    project_id: str,
    session_id: str,
    finding_artifacts: Sequence[Artifact],
    findings: Sequence[ValidatedFinding],
) -> EvidenceInterleaveSession:
    """Build the bounded evidence session used by the report writer."""
    catalog_ids = [
        *[artifact.id for artifact in finding_artifacts],
        *[source_id for finding in findings for source_id in finding.source_artifact_ids],
        *[
            reference.artifact_id
            for finding in findings
            for claim in finding.findings
            for reference in claim.evidence
            if reference.artifact_id
        ],
    ]
    artifact_session_ids = {
        artifact.id: artifact.session_id for artifact in finding_artifacts
    }
    for finding in findings:
        artifact_session_ids.update(finding.source_artifact_session_ids)

    def _sink(event_type: str, summary: dict[str, object]) -> None:
        store.append_trace(
            project_id,
            trace_event(
                session_id=session_id,
                event_type=event_type,
                name=str(summary.get("artifact_id", "")),
                summary=dict(summary),
            ),
        )

    return EvidenceInterleaveSession(
        StoreEvidenceResolver(
            store,
            project_id=project_id,
            artifact_session_ids=artifact_session_ids,
            catalog_artifact_ids=catalog_ids,
        ),
        per_section_limit=_INTERLEAVE_PER_SECTION_LIMIT,
        total_limit=_INTERLEAVE_TOTAL_LIMIT,
        trace_sink=_sink,
    )


def _persist_interleave_transcript(
    store: ArtifactStore,
    *,
    report: DecisionReport,
    session: EvidenceInterleaveSession,
    project_id: str,
    session_id: str,
    brief_artifact_id: str,
) -> bool:
    """Persist the request/grant/rejection transcript and link it on the report."""
    transcript = session.transcript
    if not transcript.exchanges:
        return True
    payload = transcript.model_dump(mode="json")
    artifact = Artifact(
        id=make_artifact_id(
            "einterleave", {"brief_artifact_id": brief_artifact_id, "transcript": payload}
        ),
        type=ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT,
        project_id=project_id,
        session_id=session_id,
        parents=[brief_artifact_id, *session.granted_artifact_ids()],
        payload=payload,
        plain_language=(
            "The write-time evidence request/grant transcript of the decision "
            "report: every value the writer fetched, and every rejected request."
        ),
    )
    try:
        store.save_artifact(artifact)
    except OSError:
        return False
    report.interleave_transcript_artifact_id = artifact.id
    report.granted_evidence_artifact_ids = session.granted_artifact_ids()
    return True


def _load_findings(
    store: ArtifactStore,
    *,
    project_id: str,
    artifact_ids: Sequence[str],
    artifact_session_ids: dict[str, str],
) -> tuple[list[Artifact], list[ValidatedFinding]]:
    artifacts: list[Artifact] = []
    findings: list[ValidatedFinding] = []
    for artifact_id in dict.fromkeys(artifact_ids):
        session_id = artifact_session_ids.get(artifact_id)
        if session_id is None:
            session_id = _unique_artifact_run(
                store,
                project_id=project_id,
                artifact_id=artifact_id,
            )
        artifact = store.get_artifact(
            artifact_id,
            project_id=project_id,
            session_id=session_id,
        )
        if artifact.type is not ArtifactType.VALIDATED_FINDING:
            raise ValueError(f"Selected artifact is not a ValidatedFinding: {artifact_id}")
        finding = ValidatedFinding.model_validate(artifact.payload)
        if not finding.report_eligible:
            raise ValueError(f"Selected finding is not report-eligible: {artifact_id}")
        artifacts.append(artifact)
        findings.append(finding)
    return artifacts, findings


def _unique_artifact_run(
    store: ArtifactStore,
    *,
    project_id: str,
    artifact_id: str,
) -> str:
    """Resolve legacy briefs only when the project identity is unambiguous."""
    try:
        row = store.artifact_index_row(
            artifact_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise ValueError(
            f"Selected finding partition is ambiguous or missing: {artifact_id}"
        ) from exc
    if row is None:
        raise ValueError(
            f"Selected finding partition is ambiguous or missing: {artifact_id}"
        )
    return str(row["session_id"])


def _rank_by_impact_significance(
    artifacts: list[Artifact],
    findings: list[ValidatedFinding],
) -> tuple[list[Artifact], list[ValidatedFinding]]:
    """Order findings by impact and significance, highest first."""
    order = sorted(
        range(len(findings)),
        key=lambda index: (-_finding_rank(findings[index]), index),
    )
    return [artifacts[i] for i in order], [findings[i] for i in order]


def _finding_rank(finding: ValidatedFinding) -> float:
    scores = [claim.score.final for claim in finding.findings if claim.score is not None]
    if not scores:
        return _UNSCORED_RANK
    return max(scores)


def _is_exception_finding(finding: ValidatedFinding) -> bool:
    if finding.claim_class == "inconclusive":
        return True
    for claim in finding.findings:
        lowered = claim.text.lower()
        if any(marker in lowered for marker in _EXCEPTION_TEXT_MARKERS):
            return True
    return False


def _meta_insight_skeleton(
    findings: Sequence[ValidatedFinding],
    ordered_finding_ids: Sequence[str],
    evidence: Sequence[tuple[float, str, str | None]],
    *,
    top_k: int = _META_TOP_K,
) -> MetaInsightSkeleton:
    """Group the highest-ranked findings into a deterministic meta-insight outline."""
    skeleton = MetaInsightSkeleton()
    for finding, artifact_id in list(zip(findings, ordered_finding_ids, strict=False))[:top_k]:
        statement = next(
            (
                claim.text.strip()
                for claim in finding.findings
                if claim.text.strip()
                and _text_numbers_resolve(claim.text, evidence)
                and not _contains_causal_language(claim.text)
            ),
            "",
        )
        if not statement:
            continue
        if _is_exception_finding(finding):
            skeleton.exception_statements.append(statement)
            skeleton.exception_finding_artifact_ids.append(artifact_id)
        else:
            skeleton.commonality_statements.append(statement)
            skeleton.commonality_finding_artifact_ids.append(artifact_id)
    return skeleton


def _deterministic_report(
    brief: SynthesisBrief,
    findings: Sequence[ValidatedFinding],
    evidence: Sequence[tuple[float, str, str | None]],
    *,
    ordered_finding_ids: Sequence[str],
) -> DecisionReport:
    questions = _unique(finding.question.strip() for finding in findings)
    datasets = _unique(
        context.dataset_name.strip() for finding in findings for context in finding.quality_context
    )
    scope = "; ".join(questions)
    if datasets:
        situation = (
            f"The situation covers {', '.join(datasets)} and the analysis scope "
            f"defined by: {scope}."
        )
    else:
        situation = f"The situation covers the analysis scope defined by: {scope}."

    conditions = _unique(
        [
            *(limitation.strip() for finding in findings for limitation in finding.limitations),
            *_aggregated_quality_conditions(findings),
        ]
    )
    complication = (
        "These observations come with conditions: " + "; ".join(conditions) + "."
        if conditions
        else "These observations come with conditions that should be reviewed."
    )

    claim_texts = [item.text.strip() for finding in findings for item in finding.findings]
    headline = brief.headline.strip()
    # The executive answer is organized as a meta-insight
    # "commonality + exception" over the top-ranked findings. The skeleton is
    # deterministic; an LLM may later reword it (``_refine_scqa``) but never
    # adds facts, and every sentence still passes ``_safe_text``.
    meta_insight = _meta_insight_skeleton(findings, ordered_finding_ids, evidence)
    answer_parts = [headline]
    commonality_statements = [
        text for text in _unique(meta_insight.commonality_statements) if text != headline
    ]
    if commonality_statements:
        answer_parts.append(
            "Shared pattern across the validated findings: " + " ".join(commonality_statements)
        )
    exception_statements = _unique(meta_insight.exception_statements)
    if exception_statements:
        answer_parts.append(
            "Exceptions that run against the shared pattern: " + " ".join(exception_statements)
        )
    if commonality_statements or exception_statements:
        answer_parts.append(
            "Recommendation: weigh the shared pattern and its exceptions in the decision review."
        )
    else:
        recommendation_claims = _unique(
            text for text in claim_texts if text != headline and not _contains_causal_language(text)
        ) or _unique(text for text in claim_texts if not _contains_causal_language(text))
        if recommendation_claims:
            answer_parts.append(
                "Recommendation: use these validated observations in decision review: "
                + " ".join(recommendation_claims)
            )

    sections = [
        DecisionReportSection(
            title=finding.question,
            body=_safe_text(
                " ".join(
                    [
                        *(item.text.strip() for item in finding.findings),
                        *(
                            [finding.interpretation.strip()]
                            if finding.interpretation_status == "validated"
                            and finding.interpretation.strip()
                            else []
                        ),
                    ]
                ),
                evidence,
                fallback="No numerically resolvable claim sentence is available for this finding.",
            ),
            finding_artifact_ids=[ordered_finding_ids[index]],
        )
        for index, finding in enumerate(findings)
    ]
    scqa = SCQAFrame(
        situation=_safe_text(
            situation,
            evidence,
            fallback="The selected analyses define the evidence scope for this decision.",
        ),
        complication=_safe_text(
            complication,
            evidence,
            fallback="These observations come with conditions that should be reviewed.",
        ),
        question=_safe_text(
            brief.decision_context,
            evidence,
            fallback="What decision should the selected evidence inform?",
        ),
        answer=_safe_text(
            " ".join(answer_parts),
            evidence,
            reject_causal=True,
            fallback="Review the validated observations before making the decision.",
        ),
    )
    report = DecisionReport(
        report_id=make_artifact_id(
            "decision",
            {
                "brief_id": brief.brief_id,
                "finding_ids": brief.selected_finding_artifact_ids,
            },
        ),
        brief_id=brief.brief_id,
        project_id=brief.project_id,
        title="Decision Report",
        scqa=scqa,
        sections=sections,
        limitations=list(brief.limitations),
        investigation_gaps=list(brief.investigation_gaps),
        report_readiness=brief.report_readiness,
        narrative_status="deterministic",
        source_finding_artifact_ids=list(ordered_finding_ids),
        meta_insight=meta_insight,
    )
    if not _report_numbers_resolve(report, evidence):
        raise ValueError("Decision report contains a number that does not resolve to evidence.")
    return report


def _aggregated_quality_conditions(findings: Sequence[ValidatedFinding]) -> list[str]:
    """Aggregate quality-condition text by issue code."""
    itemized: list[str] = []
    by_code: dict[str, list[QualityContext]] = {}
    for finding in findings:
        for context in finding.quality_context:
            if context.severity == "critical" or not context.column:
                itemized.append(context.report_limitation.strip())
            else:
                by_code.setdefault(context.issue_code, []).append(context)
    texts: list[str] = []
    for code, group in by_code.items():
        columns = {(context.dataset_id, context.column) for context in group}
        if len(columns) > _QUALITY_CONDITION_AGGREGATE_THRESHOLD:
            texts.append(_QUALITY_CONDITION_AGGREGATE_TEMPLATE.format(code=code))
        else:
            texts.extend(context.report_limitation.strip() for context in group)
    texts.extend(itemized)
    return texts


def _refine_scqa(
    report: DecisionReport,
    *,
    llm: LLMClient,
    evidence: Sequence[tuple[float, str, str | None]],
    session: EvidenceInterleaveSession | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> DecisionReport:
    # The LLM may arrange grounded content but cannot introduce facts or numbers.
    payload: dict[str, object] = {
        "instructions": (
            "Rewrite only situation, complication, and answer. Preserve all facts; "
            "introduce no numbers and make no causal claims. Keep the answer "
            "organized as the shared pattern first, then the exceptions that run "
            "against it, using only the provided statements."
        ),
        "situation": report.scqa.situation,
        "complication": report.scqa.complication,
        "answer": report.scqa.answer,
    }
    if report.meta_insight is not None and not report.meta_insight.is_empty:
        payload["commonality_exception_skeleton"] = {
            "commonalities": list(report.meta_insight.commonality_statements),
            "exceptions": list(report.meta_insight.exception_statements),
        }
    raise_if_cancelled(cancel_check, operation="decision report rewrite")
    try:
        rewrite = (
            _request_scqa_rewrite(payload, llm=llm)
            if session is None
            else _request_interleaved_rewrite(
                payload,
                llm=llm,
                session=session,
                cancel_check=cancel_check,
            )
        )
    except (BudgetExceeded, SessionCancelled):
        raise
    except (RuntimeError, ValidationError, ValueError, TypeError, AttributeError, OSError) as exc:
        return _narrative_fallback(report, f"rewrite_call_failed:{type(exc).__name__}")
    if rewrite is None:
        return _narrative_fallback(report, "rewrite_unavailable")

    gate_evidence: list[tuple[float, str, str | None]] = list(evidence)
    if session is not None:
        gate_evidence.extend(session.granted_values())

    values = (rewrite.situation.strip(), rewrite.complication.strip(), rewrite.answer.strip())
    if not all(values):
        return _narrative_fallback(report, "rewrite_incomplete")
    if any(_contains_causal_language(value) for value in values):
        return _narrative_fallback(report, "rewrite_causal_language")
    if any(not _text_numbers_resolve(value, gate_evidence) for value in values):
        return _narrative_fallback(report, "rewrite_number_unsupported")

    candidate = report.model_copy(deep=True)
    candidate.scqa.situation = values[0]
    candidate.scqa.complication = values[1]
    candidate.scqa.answer = values[2]
    candidate.narrative_status = "llm_refined"
    candidate.narrative_fallback_reason = ""
    if not _report_numbers_resolve(candidate, gate_evidence):
        return _narrative_fallback(report, "report_number_unsupported")
    return candidate


def _narrative_fallback(report: DecisionReport, reason: str) -> DecisionReport:
    """Keep the deterministic narrative, but record why the rewrite was dropped."""
    fallen_back = report.model_copy(deep=True)
    fallen_back.narrative_status = "deterministic"
    fallen_back.narrative_fallback_reason = reason
    return fallen_back


def _request_scqa_rewrite(payload: dict[str, object], *, llm: LLMClient) -> _SCQARewrite:
    raw_rewrite = llm.structured(task=_REFINEMENT_TASK, schema=_SCQARewrite, payload=payload)
    return _SCQARewrite.model_validate(raw_rewrite)


def _request_interleaved_rewrite(
    payload: dict[str, object],
    *,
    llm: LLMClient,
    session: EvidenceInterleaveSession,
    cancel_check: Callable[[], bool] | None = None,
) -> _SCQARewrite | None:
    """Run the bounded evidence-request and rewrite loop."""
    payload = dict(payload)
    payload["evidence_interleave"] = {
        "instructions": (
            "Before writing a section you may request evidence from persisted "
            "artifacts: return evidence_requests items (artifact_id, locator, "
            "section) and leave the text fields empty; the resolved values will "
            "be provided for the next attempt. Requests are limited to "
            f"{_INTERLEAVE_PER_SECTION_LIMIT} per section and "
            f"{_INTERLEAVE_TOTAL_LIMIT} per report; sections are "
            f"{', '.join(_SCQA_SECTIONS)}. When you have enough evidence, "
            "return the final text with no evidence_requests. Only numbers "
            "present in granted evidence or the provided statements may appear."
        ),
        "available_artifacts": session.catalog(),
    }
    exchanges: list[dict[str, object]] = []
    rewrite: _SCQAInterleavedRewrite | None = None
    for _ in range(_INTERLEAVE_MAX_ROUNDS):
        raise_if_cancelled(cancel_check, operation="decision report rewrite")
        raw = llm.structured(task=_REFINEMENT_TASK, schema=_SCQAInterleavedRewrite, payload=payload)
        raise_if_cancelled(cancel_check, operation="decision report rewrite")
        rewrite = _SCQAInterleavedRewrite.model_validate(raw)
        requests = rewrite.evidence_requests if session.remaining_total > 0 else []
        if not requests:
            break
        for request in requests:
            outcome = session.request(request)
            exchanges.append(outcome.model_dump(mode="json"))
        payload = dict(payload)
        payload["granted_evidence"] = list(exchanges)
    if rewrite is None:
        return None
    return _SCQARewrite(
        situation=rewrite.situation,
        complication=rewrite.complication,
        answer=rewrite.answer,
    )


def _safe_text(
    text: str,
    evidence: Sequence[tuple[float, str, str | None]],
    *,
    fallback: str,
    reject_causal: bool = False,
) -> str:
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    accepted = [
        sentence
        for sentence in sentences
        if _text_numbers_resolve(sentence, evidence)
        and (not reject_causal or not _contains_causal_language(sentence))
    ]
    return " ".join(accepted) or fallback


def _report_numbers_resolve(
    report: DecisionReport,
    evidence: Sequence[tuple[float, str, str | None]],
) -> bool:
    texts = [
        report.scqa.situation,
        report.scqa.complication,
        report.scqa.question,
        report.scqa.answer,
        *(section.body for section in report.sections),
    ]
    return all(_text_numbers_resolve(text, evidence) for text in texts)


def _text_numbers_resolve(
    text: str,
    evidence: Sequence[tuple[float, str, str | None]],
) -> bool:
    for number, is_percent in _numbers_from_text(text):
        pool = [
            value
            for value, unit, _unit_label in evidence
            if (unit == "percent" if is_percent else unit in {"raw", "currency"})
        ]
        if not any(_number_matches(number, value) for value in pool):
            return False
    currency_evidence = [
        (value, unit_label.removesuffix("/order"))
        for value, unit, unit_label in evidence
        if unit == "currency" and unit_label is not None
    ]
    if currency_evidence:
        currency_amounts = _currency_amounts(text)
        for evidence_value, evidence_code in currency_evidence:
            if not any(
                _number_matches(number, evidence_value)
                for number, is_percent in _numbers_from_text(text)
                if not is_percent
            ):
                continue
            if not any(
                code == evidence_code and _number_matches(value, evidence_value)
                for value, code in currency_amounts
            ):
                return False
    return True


def _currency_amounts(text: str) -> list[tuple[float, str]]:
    amounts: list[tuple[float, str]] = []
    for match in _CURRENCY_AMOUNT_PATTERN.finditer(text):
        value = match.group("prefix_value") or match.group("suffix_value")
        code = match.group("prefix_code") or match.group("suffix_code")
        if value is not None and code is not None:
            amounts.append((float(value), code))
    return amounts


def _number_matches(number: float, evidence: float) -> bool:
    if number.is_integer():
        return number == evidence
    tolerance = max(abs(evidence) * 0.01, 0.01)
    return abs(number - evidence) <= tolerance


def _numbers_from_text(text: str) -> list[tuple[float, bool]]:
    numbers: list[tuple[float, bool]] = []
    for match in _NUMBER_PATTERN.finditer(text):
        token = match.group(0)
        try:
            numbers.append((float(token.removesuffix("%")), token.endswith("%")))
        except ValueError:
            continue
    return numbers


def _evidence_values(
    findings: Sequence[ValidatedFinding],
) -> list[tuple[float, str, str | None]]:
    values: list[tuple[float, str, str | None]] = []
    for finding in findings:
        for claim in finding.findings:
            for reference in claim.evidence:
                values.extend(_reference_values(reference))
    return values


def _reference_values(reference: EvidenceRef) -> list[tuple[float, str, str | None]]:
    if isinstance(reference.value, bool) or reference.value is None:
        return []
    if isinstance(reference.value, int | float):
        return [(float(reference.value), reference.unit, reference.unit_label)]
    parsed = _numbers_from_text(reference.value)
    return [
        (
            value,
            "percent" if is_percent else reference.unit,
            reference.unit_label,
        )
        for value, is_percent in parsed
    ]


def _contains_causal_language(text: str) -> bool:
    return contains_causal_phrase(text, phrases=_CAUSAL_PHRASES)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))

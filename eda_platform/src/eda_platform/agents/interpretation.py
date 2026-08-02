"""Validate model-written interpretations against deterministic finding evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.claim_language import (
    MODEL_ASSERTION_TERMS,
    asserts_model_capability,
    implies_causation,
)
from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.core.semantic import SemanticSeeds, pinned_context_block
from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSection
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.report_validator import (
    extract_numbers,
    full_coverage_evidence_refs,
    validate_report_bundle,
)

_TASK = "di4_l1_interpretation"

# User-confirmed definitions are context and never widen the admissible number set.
_PINNED_DEFINITIONS_INSTRUCTION = (
    "The following are established, user-confirmed definitions. Treat them as fixed "
    "facts: interpret each metric and field exactly as defined, and never redefine "
    "or reinterpret their meaning."
)

_MAX_INTERPRETATION_CHARS = 600

# Relative tolerance with an absolute floor for small values.
_NUMERIC_TOLERANCE = 0.01

class InterpretationResult(BaseModel):
    """Outcome of an interpretation attempt; only validated text is returned."""

    text: str = ""
    status: Literal["validated", "fallback", "absent"] = "absent"
    reject_reason: str = ""


class _InterpretationDraft(BaseModel):
    """Strict parse target for the model's reply."""

    interpretation: str = Field(default="")


def interpret_findings(
    llm: LLMClient,
    *,
    question: str,
    findings: list[QuestionFinding],
    method_context: str = "",
    limitations: Sequence[str] = (),
    seeds: SemanticSeeds | None = None,
    ranking_basis: dict | None = None,
) -> InterpretationResult:
    """Generate an interpretation and admit it only after deterministic validation."""
    if is_offline_client(llm):
        return InterpretationResult(status="absent")
    if not findings:
        return InterpretationResult(status="absent")

    allowed_numbers = _allowed_numbers(findings)
    pinned_block = pinned_context_block(seeds) if seeds is not None else ""
    payload = _build_payload(
        question=question,
        findings=findings,
        method_context=method_context,
        limitations=limitations,
        allowed_numbers=allowed_numbers,
        pinned_block=pinned_block,
        ranking_basis=ranking_basis,
    )

    draft: _InterpretationDraft | None = None
    last_error = ""
    for attempt in range(2):
        try:
            draft = llm.structured(task=_TASK, schema=_InterpretationDraft, payload=payload)
            break
        except BudgetExceeded:
            raise
        except (ValidationError, RuntimeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt == 0:
                continue
    if draft is None:
        return InterpretationResult(status="fallback", reject_reason=f"llm_error: {last_error}")

    text = draft.interpretation.strip()
    reject_reason = _gate(text, allowed_numbers)
    if reject_reason is not None:
        return InterpretationResult(status="fallback", reject_reason=reject_reason)
    return InterpretationResult(text=text, status="validated")


def _gate(text: str, allowed_numbers: Sequence[tuple[float, bool]]) -> str | None:
    """Return a rejection reason string, or ``None`` when the text is admissible."""
    if not text:
        return "empty_interpretation"
    if len(text) > _MAX_INTERPRETATION_CHARS:
        return f"too_long: {len(text)} > {_MAX_INTERPRETATION_CHARS} chars"
    if implies_causation(text):
        return "causal_language: interpretation asserts causation"
    for number, is_percent in extract_numbers(text):
        if not _matches_any(number, is_percent, allowed_numbers):
            unit = "%" if is_percent else ""
            return f"unsupported_number: {number:g}{unit} not traceable to evidence"
    return None


def validate_interpretation_text(
    text: str,
    findings: Sequence[QuestionFinding],
) -> tuple[bool, str]:
    """Validate free text against the numeric, causal, and length gates."""
    reason = _gate(text.strip(), _allowed_numbers(findings))
    if reason is None:
        return True, ""
    return False, reason


def _allowed_numbers(findings: Sequence[QuestionFinding]) -> list[tuple[float, bool]]:
    """Return evidence-backed numbers, preserving raw-versus-percent units.

    Finding text is not a source. Every reducer-written figure carries a paired
    EvidenceRef, but each finding text also opens with `question_en`, which is
    model-authored for LLM-origin candidates — admitting its digits let a model
    approve its own interpretation by first writing the number into the question.
    The cost is that an interpretation quoting a parameter from the question
    itself now falls back; discarding text is the safe direction.
    """
    allowed: list[tuple[float, bool]] = []
    for finding in findings:
        for reference in finding.evidence:
            value = reference.value
            if isinstance(value, int | float) and not isinstance(value, bool):
                allowed.append((float(value), reference.unit == "percent"))
    return allowed


# An agent answer summarises several tool calls, so it needs more room than the
# single-SQL interpretation gate allows.
_MAX_AGENT_ANSWER_CHARS = 2_000


def validate_agent_answer(
    answer: str,
    evidence_artifacts: Sequence[Artifact],
) -> tuple[bool, str]:
    """Gate a tool-loop answer against the typed parse of its own evidence.

    The bounded agent loop has no planned SQL to validate against, so the pool
    is rebuilt here from persisted tool payloads only. The answer never
    contributes to the set of numbers that may verify it.
    """
    text = answer.strip()
    if not text:
        return False, "empty_answer"
    if len(text) > _MAX_AGENT_ANSWER_CHARS:
        return False, f"too_long: {len(text)} > {_MAX_AGENT_ANSWER_CHARS} chars"
    if implies_causation(text):
        return False, "causal_language: answer asserts causation"
    if not evidence_artifacts:
        return False, "no_evidence: the answer cites no tool-produced artifact"
    has_model_card = any(
        artifact.type is ArtifactType.MODEL_CARD for artifact in evidence_artifacts
    )
    if not has_model_card and asserts_model_capability(text, terms=MODEL_ASSERTION_TERMS):
        return False, "unsupported_model_claim: no ModelCard artifact in evidence"
    return _agent_number_gate(text, evidence_artifacts)


def _agent_number_gate(
    text: str,
    evidence_artifacts: Sequence[Artifact],
) -> tuple[bool, str]:
    """Run the answer through the report validator's F0-F2 numeric chain."""
    bundle = ReportBundle(
        project_id="agent_answer_gate",
        session_id="agent_answer_gate",
        sections=[
            ReportSection(
                title="answer",
                claims=[
                    ReportClaim(
                        text=text,
                        evidence=full_coverage_evidence_refs(evidence_artifacts),
                    )
                ],
            )
        ],
    )
    sql_results = {
        artifact.id: SqlResult.model_validate(artifact.payload)
        for artifact in evidence_artifacts
        if artifact.type is ArtifactType.SQL_RESULT
    }
    validate_report_bundle(
        bundle,
        build_evidence_pack(list(evidence_artifacts)),
        sql_results=sql_results,
    )
    claim = bundle.sections[0].claims[0]
    if claim.numeric_rollup in {"number_verified", "no_numbers"}:
        return True, ""
    unverified = [
        status for status in claim.numeric_statuses if status.status != "number_verified"
    ]
    listed = ", ".join(
        f"{status.number:.10g}%" if status.is_percent else f"{status.number:.10g}"
        for status in unverified[:5]
    )
    return False, f"unsupported_number: {listed} not traceable to tool evidence"


def _matches_any(
    number: float,
    is_percent: bool,
    allowed_numbers: Sequence[tuple[float, bool]],
) -> bool:
    for target, target_is_percent in allowed_numbers:
        if is_percent != target_is_percent:
            continue
        tolerance = max(abs(target) * _NUMERIC_TOLERANCE, _NUMERIC_TOLERANCE)
        if abs(number - target) <= tolerance:
            return True
    return False


def _build_payload(
    *,
    question: str,
    findings: Sequence[QuestionFinding],
    method_context: str,
    limitations: Sequence[str],
    allowed_numbers: Sequence[tuple[float, bool]],
    pinned_block: str = "",
    ranking_basis: dict | None = None,
) -> dict:
    claims = [
        {
            "claim": finding.text,
            "evidence": [
                {"locator": reference.locator, "value": reference.value}
                for reference in finding.evidence
            ],
        }
        for finding in findings
    ]
    payload: dict = {
        "instructions": (
            "You are given the verified findings of a deterministic analysis. Write a "
            "short business interpretation of what these findings mean for the decision "
            "at hand: which result stands out, how large it is, and the practical "
            "so-what. Hard rules: (1) do NOT introduce any number that is not already in "
            "the findings or their evidence values — cite only the numbers listed in "
            "allowed_numbers; (2) make NO causal claims — these are observed/associative "
            "results, not causes; (3) at most three sentences; (4) 'ranking_basis' names "
            "the column and direction the result rows are verifiably ordered by — when "
            "ranking_basis is null the rows have no proven ordering, so do NOT use "
            "ranking or superlative wording (strongest, most, highest, top, leading). "
            "Put the interpretation in the 'interpretation' field."
        ),
        "question": question,
        "claims": claims,
        "ranking_basis": ranking_basis,
        "allowed_numbers": [
            f"{value:g}%" if is_percent else f"{value:g}"
            for value, is_percent in allowed_numbers
        ],
        "method_context": method_context,
        "limitations": list(limitations),
    }
    if pinned_block:
        # Prepend established definitions as fixed context. Context only — it never
        # widens allowed_numbers, so the number gate is unchanged.
        payload = {
            "pinned_definitions": f"{_PINNED_DEFINITIONS_INSTRUCTION}\n{pinned_block}",
            **payload,
        }
    return payload

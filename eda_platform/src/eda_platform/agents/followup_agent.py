"""Follow-up question generator for the analysis macro-loop (L2, design §2).

One LLM call turns the run ledger (validated findings + round history) into a
FollowUpProposalSet; the LLM may explicitly conclude (the InsightPilot bottom
symbol). Proposals then convert into QuestionCandidates with a DETERMINISTIC
score — the LLM never authors scores — and cross-round question-fingerprint
dedup happens on the deterministic side, never in the prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from eda_platform.core.llm import LLMClient, is_offline_client
from eda_platform.core.loop_fingerprint import question_fingerprint
from eda_platform.core.loop_ledger import is_duplicate_question
from eda_platform.core.trace import trace_event
from eda_platform.schemas.loop import FollowUpProposal, FollowUpProposalSet, LoopLedger
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.question_discovery import make_question_id

_TASK = "l2_followup_generation"

FOLLOWUP_GENERATION_FAILED = "followup_generation_failed"

# One structured call plus at most one repair retry on schema violations
# (question_agent pattern, tightened to the L2 budget).
_MAX_REPAIR_RETRIES = 1

_MAX_FOLLOWUP_QUESTIONS = 5

# Heuristic follow-up scoring constants (design §2: scores stay deterministic).
# Baseline = parent question's deterministic score decayed per round, nudged by
# the LLM's priority hint; recalibrate once depth >=2 has live data (design §7).
FOLLOWUP_ROUND_DECAY = 0.9
FOLLOWUP_PRIORITY_ADJUST: dict[str, float] = {"high": 0.05, "medium": 0.0, "low": -0.05}

TraceSink = Callable[[TraceEvent], None]

_INSTRUCTIONS = (
    "You continue an automated data-analysis loop. Based ONLY on the validated "
    "findings listed in the payload, propose follow-up analysis questions worth "
    "executing next round. Return at most max_questions proposals. Every proposal "
    "MUST set parent_finding_id to the finding_id of the validated finding it "
    "digs into. Ask business-language questions; never compute numbers or author "
    "scores. Do not re-ask questions this run already explored: the payload "
    "reports how many question fingerprints exist, and duplicates are dropped "
    "deterministically downstream, so repeats only waste your proposal slots. "
    "If nothing left is worth pursuing, return concluded=true with a short "
    "conclusion_reason and an empty proposals list."
)

_SCHEMA_EXAMPLE: dict[str, Any] = {
    "concluded": False,
    "conclusion_reason": "",
    "proposals": [
        {
            "question_text": "Why did region A revenue drop in Q3?",
            "rationale": "Largest validated decline; the driver is unknown.",
            "parent_finding_id": "finding-1",
            "priority_hint": "high",
        }
    ],
}


class _RawFollowUpProposal(BaseModel):
    question_text: str = ""
    rationale: str = ""
    parent_finding_id: str = ""
    priority_hint: str = "medium"


class _RawFollowUpResponse(BaseModel):
    concluded: bool = False
    conclusion_reason: str = ""
    proposals: list[_RawFollowUpProposal] = Field(default_factory=list)


@dataclass(frozen=True)
class ParentFinding:
    """Ledger-side context for one validated finding a follow-up may target."""

    finding_id: str
    statements: list[str]
    score: QuestionScore
    target_datasets: list[str]
    dataset_display_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FollowUpGenerationResult:
    proposal_set: FollowUpProposalSet
    error: str | None = None
    dropped_invalid_proposals: int = 0
    degraded: bool = False


@dataclass(frozen=True)
class FollowUpConversionResult:
    candidates: list[QuestionCandidate]
    dropped_duplicate_count: int = 0
    dropped_invalid_count: int = 0


def generate_followup_proposals(
    llm: LLMClient | None,
    *,
    ledger: LoopLedger,
    parent_findings: Sequence[ParentFinding],
    data_summary: str,
    round_id: int,
    max_questions: int = _MAX_FOLLOWUP_QUESTIONS,
    session_id: str = "",
    trace_sink: TraceSink | None = None,
) -> FollowUpGenerationResult:
    """One LLM call: ledger summary in, FollowUpProposalSet out; fail-safe to ⊥."""
    if llm is None or is_offline_client(llm):
        return _fail_safe(
            round_id,
            reason="LLM unavailable: follow-up generation skipped.",
            session_id=session_id,
            trace_sink=trace_sink,
        )

    payload = _payload(
        ledger,
        parent_findings=parent_findings,
        data_summary=data_summary,
        round_id=round_id,
        max_questions=max_questions,
    )
    previous_error: str | None = None
    raw: _RawFollowUpResponse | None = None
    for attempt in range(_MAX_REPAIR_RETRIES + 1):
        attempt_payload = dict(payload)
        if previous_error is not None:
            attempt_payload["previous_error"] = previous_error
            attempt_payload["schema_example"] = _SCHEMA_EXAMPLE
            attempt_payload["repair_attempt"] = attempt
        try:
            raw = llm.structured(task=_TASK, schema=_RawFollowUpResponse, payload=attempt_payload)
            break
        except ValidationError as exc:
            previous_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        except (RuntimeError, ValueError) as exc:
            # Transport/provider errors: a retry cannot repair these.
            return _fail_safe(
                round_id,
                reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                session_id=session_id,
                trace_sink=trace_sink,
            )
    if raw is None:
        return _fail_safe(
            round_id,
            reason=f"parse failed after retry: {previous_error or 'unknown'}"[:300],
            session_id=session_id,
            trace_sink=trace_sink,
        )

    known_finding_ids = {parent.finding_id for parent in parent_findings}
    accepted: list[FollowUpProposal] = []
    dropped = 0
    for proposal in raw.proposals:
        text = proposal.question_text.strip()
        if not text or proposal.parent_finding_id not in known_finding_ids:
            dropped += 1
            continue
        hint = (
            proposal.priority_hint
            if proposal.priority_hint in FOLLOWUP_PRIORITY_ADJUST
            else "medium"
        )
        accepted.append(
            FollowUpProposal(
                question_text=text,
                rationale=proposal.rationale.strip(),
                parent_finding_id=proposal.parent_finding_id,
                priority_hint=hint,  # type: ignore[arg-type]
            )
        )
    accepted = accepted[:max_questions]

    if raw.concluded:
        return FollowUpGenerationResult(
            proposal_set=FollowUpProposalSet(
                round_id=round_id,
                concluded=True,
                conclusion_reason=raw.conclusion_reason.strip(),
            ),
            dropped_invalid_proposals=dropped,
            degraded=dropped > 0,
        )
    if not accepted:
        return _fail_safe(
            round_id,
            reason="no valid proposals returned.",
            session_id=session_id,
            trace_sink=trace_sink,
            dropped=dropped,
        )
    return FollowUpGenerationResult(
        proposal_set=FollowUpProposalSet(round_id=round_id, proposals=accepted),
        dropped_invalid_proposals=dropped,
        degraded=dropped > 0,
    )


def followup_question_candidates(
    proposal_set: FollowUpProposalSet,
    *,
    ledger: LoopLedger,
    parent_findings: Sequence[ParentFinding],
) -> FollowUpConversionResult:
    """Convert proposals into deterministically scored QuestionCandidates.

    Cross-round (and in-batch) question-fingerprint dedup happens here, on the
    deterministic side; drops are counted, never silent.
    """
    parents = {parent.finding_id: parent for parent in parent_findings}
    seen_fingerprints: set[str] = set()
    candidates: list[QuestionCandidate] = []
    dropped_duplicates = 0
    dropped_invalid = 0
    for proposal in proposal_set.proposals:
        text = proposal.question_text.strip()
        parent = parents.get(proposal.parent_finding_id)
        if not text or parent is None or not parent.target_datasets:
            dropped_invalid += 1
            continue
        fingerprint = question_fingerprint(text)
        if is_duplicate_question(ledger, fingerprint) or fingerprint in seen_fingerprints:
            dropped_duplicates += 1
            continue
        seen_fingerprints.add(fingerprint)
        candidates.append(_candidate(proposal, parent=parent, round_id=proposal_set.round_id))
    return FollowUpConversionResult(
        candidates=candidates,
        dropped_duplicate_count=dropped_duplicates,
        dropped_invalid_count=dropped_invalid,
    )


def _candidate(
    proposal: FollowUpProposal, *, parent: ParentFinding, round_id: int
) -> QuestionCandidate:
    adjust = FOLLOWUP_PRIORITY_ADJUST.get(proposal.priority_hint, 0.0)
    deterministic = round(
        max(0.0, min(1.0, parent.score.deterministic_score * FOLLOWUP_ROUND_DECAY + adjust)), 6
    )
    score = parent.score.model_copy(
        update={
            "deterministic_score": deterministic,
            # Parent LLM display scores do not transfer to the follow-up.
            "llm_business_relevance": None,
            "llm_actionability": None,
        }
    )
    targets = list(parent.target_datasets)
    display_names = {
        dataset: label
        for dataset, label in parent.dataset_display_names.items()
        if dataset in set(targets) and label.strip()
    }
    return QuestionCandidate(
        question_id=make_question_id(
            origin="llm", question_en=proposal.question_text, target_datasets=targets
        ),
        question_en=proposal.question_text,
        origin="llm",
        target_datasets=targets,
        dataset_display_names=display_names,
        sql_template=None,
        score=score,
        exploratory=True,
        value_hypothesis=proposal.rationale,
        data_signal=(
            f"Follow-up on validated finding {proposal.parent_finding_id} (round {round_id})."
        ),
    )


def _payload(
    ledger: LoopLedger,
    *,
    parent_findings: Sequence[ParentFinding],
    data_summary: str,
    round_id: int,
    max_questions: int,
) -> dict[str, Any]:
    previous_round = ledger.rounds[-1].model_dump(mode="json") if ledger.rounds else None
    return {
        "instructions": _INSTRUCTIONS,
        "max_questions": max_questions,
        "round_id": round_id,
        "validated_findings": [
            {"finding_id": parent.finding_id, "statements": parent.statements}
            for parent in parent_findings
        ],
        "explored_question_fingerprints": len(ledger.question_fingerprints),
        "explored_finding_fingerprints": len(ledger.finding_fingerprints),
        "previous_round": previous_round,
        "data_summary": data_summary,
    }


def _fail_safe(
    round_id: int,
    *,
    reason: str,
    session_id: str,
    trace_sink: TraceSink | None,
    dropped: int = 0,
) -> FollowUpGenerationResult:
    """Concluded empty set: the loop terminates instead of blocking the run."""
    if trace_sink is not None:
        trace_sink(
            trace_event(
                session_id=session_id,
                event_type=FOLLOWUP_GENERATION_FAILED,
                name="followup_agent",
                summary={"round_id": round_id, "reason": reason[:300]},
            )
        )
    return FollowUpGenerationResult(
        proposal_set=FollowUpProposalSet(
            round_id=round_id,
            concluded=True,
            conclusion_reason=f"fail-safe: {reason}"[:300],
        ),
        error=reason,
        dropped_invalid_proposals=dropped,
        degraded=True,
    )

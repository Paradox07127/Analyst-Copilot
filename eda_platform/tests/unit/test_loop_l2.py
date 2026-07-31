"""L2 of the analysis macro-loop: follow-up generator, candidate conversion, depth config.

Design source: docs/archive/2026-07/base/eda-agent-platform-analysis-loop-design-2026-07-23.md
(§2 return-edge follow-up generator, §3 depth tiers, §8 pre-implementation rulings).
"""

from __future__ import annotations

from typing import Any

from eda_platform.agents.followup_agent import (
    FOLLOWUP_PRIORITY_ADJUST,
    FOLLOWUP_ROUND_DECAY,
    ParentFinding,
    followup_question_candidates,
    generate_followup_proposals,
)
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.loop_fingerprint import question_fingerprint
from eda_platform.schemas.loop import DEPTH_PROFILES, LoopDepthProfile, LoopLedger, LoopRoundRecord
from eda_platform.schemas.questions import QuestionScore
from eda_platform.schemas.sessions import TraceEvent

# ------------------------------------------------------------------ helpers


class _MockLLM:
    """Structured-output stub: returns a canned payload or raises."""

    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self._response = response
        self.calls = 0

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        # Invalid payloads raise pydantic.ValidationError here, like a real client.
        return schema.model_validate(self._response)

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


def _score(deterministic: float = 0.7) -> QuestionScore:
    return QuestionScore(
        data_availability=0.9,
        statistical_signal=0.6,
        quality_risk=0.2,
        join_risk=0.0,
        deterministic_score=deterministic,
        llm_business_relevance=0.8,
        llm_actionability=0.7,
    )


def _parent(finding_id: str = "finding-1") -> ParentFinding:
    return ParentFinding(
        finding_id=finding_id,
        statements=["Region A revenue dropped 12% in Q3."],
        score=_score(),
        target_datasets=["orders.csv"],
        dataset_display_names={"orders.csv": "Orders"},
    )


_LEDGER = LoopLedger(
    depth=2,
    finding_fingerprints=["aa11"],
    validated_finding_ids=["finding-1"],
    rounds=[LoopRoundRecord(round_id=1, new_validated_findings=1, disposition="keep")],
)


def _generate(llm: Any, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "ledger": _LEDGER,
        "parent_findings": [_parent()],
        "data_summary": "orders.csv: 100 rows x 4 columns",
        "round_id": 2,
    }
    kwargs.update(overrides)
    return generate_followup_proposals(llm, **kwargs)


# ------------------------------------------------- R1: proposal generation


def test_generate_parses_valid_proposals_with_parent_ids() -> None:
    llm = _MockLLM(
        {
            "concluded": False,
            "proposals": [
                {
                    "question_text": "Why did region A revenue drop in Q3?",
                    "rationale": "Largest validated drop; cause unknown.",
                    "parent_finding_id": "finding-1",
                    "priority_hint": "high",
                },
                {
                    "question_text": "Did discounting drive the Q3 revenue drop?",
                    "parent_finding_id": "finding-1",
                },
            ],
        }
    )
    result = _generate(llm)
    assert result.error is None
    assert result.degraded is False
    proposal_set = result.proposal_set
    assert proposal_set.round_id == 2
    assert proposal_set.concluded is False
    assert len(proposal_set.proposals) == 2
    assert all(p.parent_finding_id == "finding-1" for p in proposal_set.proposals)
    assert llm.calls == 1


def test_generate_drops_proposals_with_unknown_or_missing_parent() -> None:
    llm = _MockLLM(
        {
            "proposals": [
                {"question_text": "Valid follow-up?", "parent_finding_id": "finding-1"},
                {"question_text": "Orphan follow-up?", "parent_finding_id": "nope"},
                {"question_text": "No parent at all?"},
            ]
        }
    )
    result = _generate(llm)
    assert len(result.proposal_set.proposals) == 1
    assert result.dropped_invalid_proposals == 2
    assert result.degraded is True


def test_generate_caps_proposals_at_max_questions() -> None:
    llm = _MockLLM(
        {
            "proposals": [
                {"question_text": f"Follow-up {i}?", "parent_finding_id": "finding-1"}
                for i in range(8)
            ]
        }
    )
    result = _generate(llm)
    assert len(result.proposal_set.proposals) == 5  # N=5 default cap


# ------------------------------------- R2: bottom symbol and fail-safe paths


def test_generate_concluded_bottom_returns_empty_set() -> None:
    llm = _MockLLM({"concluded": True, "conclusion_reason": "Nothing left worth pursuing."})
    result = _generate(llm)
    assert result.error is None
    assert result.proposal_set.concluded is True
    assert result.proposal_set.proposals == []


def test_generate_offline_or_missing_llm_fails_safe() -> None:
    for llm in (None, OfflineLLMClient()):
        result = _generate(llm)
        assert result.proposal_set.concluded is True
        assert result.proposal_set.proposals == []
        assert result.error is not None
        assert result.degraded is True


def test_generate_runtime_error_fails_safe_without_retry() -> None:
    llm = _MockLLM(RuntimeError("provider down"))
    events: list[TraceEvent] = []
    result = _generate(llm, session_id="run-1", trace_sink=events.append)
    assert result.proposal_set.concluded is True
    assert result.proposal_set.proposals == []
    assert result.error is not None
    assert result.degraded is True
    assert llm.calls == 1  # transport errors are not retried
    assert len(events) == 1
    assert events[0].event_type == "followup_generation_failed"


def test_generate_bad_payload_retries_once_then_fails_safe() -> None:
    llm = _MockLLM({"proposals": 123})  # schema violation -> ValidationError
    events: list[TraceEvent] = []
    result = _generate(llm, session_id="run-1", trace_sink=events.append)
    assert result.proposal_set.concluded is True
    assert result.proposal_set.proposals == []
    assert result.error is not None
    assert llm.calls == 2  # one retry budget
    assert len(events) == 1


# ------------------------------- R3: conversion, dedup, deterministic score


def _proposal_set(texts_and_hints: list[tuple[str, str]]) -> Any:
    from eda_platform.schemas.loop import FollowUpProposal, FollowUpProposalSet

    return FollowUpProposalSet(
        round_id=2,
        proposals=[
            FollowUpProposal(
                question_text=text,
                parent_finding_id="finding-1",
                priority_hint=hint,  # type: ignore[arg-type]
            )
            for text, hint in texts_and_hints
        ],
    )


def test_conversion_drops_ledger_duplicates_and_counts() -> None:
    duplicate_text = "Why did region A revenue drop in Q3?"
    ledger = _LEDGER.model_copy(
        update={"question_fingerprints": [question_fingerprint(duplicate_text)]}
    )
    proposal_set = _proposal_set(
        [(duplicate_text, "medium"), ("Did discounting drive the drop?", "high")]
    )
    result = followup_question_candidates(
        proposal_set, ledger=ledger, parent_findings=[_parent()]
    )
    assert len(result.candidates) == 1
    assert result.dropped_duplicate_count == 1
    candidate = result.candidates[0]
    assert candidate.question_en == "Did discounting drive the drop?"
    assert candidate.origin == "llm"
    assert candidate.exploratory is True
    assert candidate.target_datasets == ["orders.csv"]


def test_conversion_score_is_deterministic_and_uses_decay_plus_hint() -> None:
    proposal_set = _proposal_set(
        [("High priority follow-up?", "high"), ("Low priority follow-up?", "low")]
    )
    kwargs: dict[str, Any] = {"ledger": _LEDGER, "parent_findings": [_parent()]}
    first = followup_question_candidates(proposal_set, **kwargs)
    second = followup_question_candidates(proposal_set, **kwargs)
    assert [c.score for c in first.candidates] == [c.score for c in second.candidates]
    assert [c.question_id for c in first.candidates] == [
        c.question_id for c in second.candidates
    ]
    parent_score = _score().deterministic_score
    high, low = first.candidates
    assert high.score.deterministic_score == round(
        parent_score * FOLLOWUP_ROUND_DECAY + FOLLOWUP_PRIORITY_ADJUST["high"], 6
    )
    assert low.score.deterministic_score == round(
        parent_score * FOLLOWUP_ROUND_DECAY + FOLLOWUP_PRIORITY_ADJUST["low"], 6
    )
    # LLM display scores are not inherited from the parent question.
    assert high.score.llm_business_relevance is None
    assert high.score.llm_actionability is None


def test_conversion_dedupes_within_batch_and_flags_unknown_parent() -> None:
    from eda_platform.schemas.loop import FollowUpProposal, FollowUpProposalSet

    proposal_set = FollowUpProposalSet(
        round_id=2,
        proposals=[
            FollowUpProposal(question_text="Same question?", parent_finding_id="finding-1"),
            FollowUpProposal(question_text="  same QUESTION!?", parent_finding_id="finding-1"),
            FollowUpProposal(question_text="Orphan question?", parent_finding_id="ghost"),
        ],
    )
    result = followup_question_candidates(
        proposal_set, ledger=LoopLedger(), parent_findings=[_parent()]
    )
    assert len(result.candidates) == 1
    assert result.dropped_duplicate_count == 1
    assert result.dropped_invalid_count == 1


# ----------------------------------------------------- R4: depth configuration


def test_depth_profiles_table_is_complete() -> None:
    assert set(DEPTH_PROFILES) == {0, 1, 2, 3}
    assert DEPTH_PROFILES[0] == LoopDepthProfile(rounds=0, per_round_questions=0)
    assert DEPTH_PROFILES[1] == LoopDepthProfile(rounds=0, per_round_questions=0)
    assert DEPTH_PROFILES[2] == LoopDepthProfile(rounds=1, per_round_questions=4)
    assert DEPTH_PROFILES[3] == LoopDepthProfile(rounds=3, per_round_questions=4)

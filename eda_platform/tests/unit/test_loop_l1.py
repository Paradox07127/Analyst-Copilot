"""L1 foundations of the analysis macro-loop: schemas, fingerprints, ledger ops.

Design source: docs/archive/2026-07/base/eda-agent-platform-analysis-loop-design-2026-07-23.md
(§2 ledger/return-edge, §5.2 keep/discard, §8.1 fingerprint rules).
"""

from __future__ import annotations

import pytest

from eda_platform.core.loop_fingerprint import finding_fingerprint, question_fingerprint
from eda_platform.core.loop_ledger import (
    admit_finding,
    is_duplicate_finding,
    is_duplicate_question,
    keep_or_discard,
    record_round,
)
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.loop import (
    FollowUpProposal,
    FollowUpProposalSet,
    LoopLedger,
    LoopRoundRecord,
)

# ---------------------------------------------------------------- schemas


def test_follow_up_proposal_set_round_trip() -> None:
    proposal_set = FollowUpProposalSet(
        round_id=2,
        concluded=False,
        proposals=[
            FollowUpProposal(
                question_text="Why did region A revenue drop in Q3?",
                rationale="Largest residual after seasonal adjustment.",
                parent_finding_id="finding-1",
                priority_hint="high",
            )
        ],
    )
    restored = FollowUpProposalSet.model_validate(proposal_set.model_dump())
    assert restored == proposal_set
    assert restored.schema_version == 1
    assert restored.conclusion_reason == ""


def test_follow_up_proposal_set_concluded_bottom() -> None:
    bottom = FollowUpProposalSet(
        round_id=3, concluded=True, conclusion_reason="No unexplained variance remains."
    )
    assert bottom.proposals == []
    assert FollowUpProposalSet.model_validate(bottom.model_dump()).concluded is True


def test_loop_ledger_round_trip_and_defaults() -> None:
    ledger = LoopLedger()
    assert ledger.depth == 0
    assert ledger.rounds == []
    populated = LoopLedger(
        depth=2,
        finding_fingerprints=["aa"],
        question_fingerprints=["bb"],
        validated_finding_ids=["finding-1"],
        rounds=[
            LoopRoundRecord(
                round_id=1,
                new_validated_findings=1,
                executed_questions=3,
                tokens=1200,
                exit_reason="continue",
                disposition="keep",
            )
        ],
    )
    assert LoopLedger.model_validate(populated.model_dump()) == populated


def test_artifact_type_gains_loop_members() -> None:
    assert ArtifactType.FOLLOW_UP_PROPOSAL_SET.value == "FollowUpProposalSet"
    assert ArtifactType.LOOP_LEDGER.value == "LoopLedger"


# ---------------------------------------------------------- fingerprints


_EVIDENCE = [
    ("SqlResult", "rows[3].revenue", 1234.5),
    ("Table", "summary.total", 99.0),
]


def test_finding_fingerprint_ignores_wording_given_same_evidence() -> None:
    fp_a = finding_fingerprint(["Revenue dropped sharply in Q3."], _EVIDENCE)
    fp_b = finding_fingerprint(
        ["Q3 saw a steep revenue decline across the board."], list(reversed(_EVIDENCE))
    )
    assert fp_a == fp_b


def test_finding_fingerprint_changes_when_value_changes() -> None:
    changed = [("SqlResult", "rows[3].revenue", 1234.6), ("Table", "summary.total", 99.0)]
    assert finding_fingerprint(["same text"], _EVIDENCE) != finding_fingerprint(
        ["same text"], changed
    )


def test_finding_fingerprint_empty_evidence_fallback_is_stable() -> None:
    fp_a = finding_fingerprint(["Revenue dropped 12% in Q3"], [])
    fp_b = finding_fingerprint(["  REVENUE dropped 99% in  q3 "], [])
    assert fp_a == fp_b  # casefold + digits and whitespace stripped
    assert fp_a != finding_fingerprint(["Margin rose in Q4"], [])


def test_finding_fingerprint_family_key_separates_questions() -> None:
    """§8.1: the family key joins the hash, so same evidence from different
    questions no longer collides, while wording changes within one family
    still map to the same fingerprint."""
    fp_q1 = finding_fingerprint(["Revenue dropped."], _EVIDENCE, family_key="q_1")
    fp_q2 = finding_fingerprint(["Revenue dropped."], _EVIDENCE, family_key="q_2")
    assert fp_q1 != fp_q2
    rephrased = finding_fingerprint(
        ["Q3 revenue saw a decline."], list(reversed(_EVIDENCE)), family_key="q_1"
    )
    assert rephrased == fp_q1


def test_finding_fingerprint_empty_family_key_is_backward_compatible() -> None:
    assert finding_fingerprint(["stmt"], _EVIDENCE, family_key="") == finding_fingerprint(
        ["stmt"], _EVIDENCE
    )
    # The text fallback keys on the family too.
    no_evidence_q1 = finding_fingerprint(["Margin rose in Q4"], [], family_key="q_1")
    no_evidence_q2 = finding_fingerprint(["Margin rose in Q4"], [], family_key="q_2")
    assert no_evidence_q1 != no_evidence_q2
    assert finding_fingerprint(["Margin rose in Q4"], [], family_key="") == finding_fingerprint(
        ["Margin rose in Q4"], []
    )


def test_question_fingerprint_normalizes_case_punctuation_whitespace() -> None:
    fp_a = question_fingerprint("Why did Region A's revenue drop?!")
    fp_b = question_fingerprint("  why did region a s revenue   drop ")
    assert fp_a == fp_b
    assert fp_a != question_fingerprint("Why did region B revenue drop?")


# --------------------------------------------------------------- ledger


def test_record_round_appends_and_is_pure() -> None:
    ledger = LoopLedger()
    first = LoopRoundRecord(round_id=1)
    updated = record_round(ledger, first)
    assert ledger.rounds == []  # input untouched
    assert [r.round_id for r in updated.rounds] == [1]
    updated2 = record_round(updated, LoopRoundRecord(round_id=2))
    assert [r.round_id for r in updated2.rounds] == [1, 2]


def test_record_round_rejects_non_increasing_round_id() -> None:
    ledger = record_round(LoopLedger(), LoopRoundRecord(round_id=2))
    with pytest.raises(ValueError):
        record_round(ledger, LoopRoundRecord(round_id=2))
    with pytest.raises(ValueError):
        record_round(ledger, LoopRoundRecord(round_id=1))


def test_admit_finding_appends_new_and_rejects_duplicate() -> None:
    ledger = LoopLedger()
    fp = finding_fingerprint(["stmt"], _EVIDENCE)
    admitted = admit_finding(ledger, "finding-1", fp)
    assert admitted.validated_finding_ids == ["finding-1"]
    assert admitted.finding_fingerprints == [fp]
    assert is_duplicate_finding(admitted, fp)
    assert not is_duplicate_finding(admitted, "other")
    # Duplicate fingerprint: ledger returned unchanged, no second admission.
    rejected = admit_finding(admitted, "finding-2", fp)
    assert rejected == admitted
    assert rejected.validated_finding_ids == ["finding-1"]


def test_is_duplicate_question() -> None:
    fp = question_fingerprint("Why did revenue drop?")
    ledger = LoopLedger(question_fingerprints=[fp])
    assert is_duplicate_question(ledger, fp)
    assert not is_duplicate_question(ledger, question_fingerprint("Different question"))


# §5.2: new_validated_findings counts only non-redundant admissions, so a round
# whose findings were ALL redundant has new_validated_findings == 0 by
# construction — "1 new but all redundant" cannot be represented.
def test_keep_or_discard_boundaries() -> None:
    keep = LoopRoundRecord(round_id=1, new_validated_findings=1, redundant_findings=0)
    assert keep_or_discard(keep) == "keep"

    mixed = LoopRoundRecord(round_id=2, new_validated_findings=1, redundant_findings=2)
    assert keep_or_discard(mixed) == "keep"  # some new, some redundant: not all-redundant

    nothing_new = LoopRoundRecord(round_id=3, new_validated_findings=0, redundant_findings=0)
    assert keep_or_discard(nothing_new) == "discard"

    all_redundant = LoopRoundRecord(round_id=4, new_validated_findings=0, redundant_findings=3)
    assert keep_or_discard(all_redundant) == "discard"


def test_keep_or_discard_crash_exit_maps_to_crash() -> None:
    crashed = LoopRoundRecord(round_id=5, new_validated_findings=1, exit_reason="crash")
    assert keep_or_discard(crashed) == "crash"

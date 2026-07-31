"""Pure ledger operations for the analysis macro-loop (design doc §2/§5.2).

Every function returns a new LoopLedger (or the input unchanged on rejection);
no IO here — persistence belongs to the L3 orchestrator.
"""

from __future__ import annotations

from eda_platform.schemas.loop import LoopLedger, LoopRoundRecord, RoundDisposition


def record_round(ledger: LoopLedger, record: LoopRoundRecord) -> LoopLedger:
    """Append a round record; round_id must be strictly increasing."""
    if ledger.rounds and record.round_id <= ledger.rounds[-1].round_id:
        raise ValueError(
            f"round_id must be strictly increasing: got {record.round_id} "
            f"after {ledger.rounds[-1].round_id}."
        )
    return ledger.model_copy(update={"rounds": [*ledger.rounds, record]})


def is_duplicate_finding(ledger: LoopLedger, fingerprint: str) -> bool:
    return fingerprint in ledger.finding_fingerprints


def is_duplicate_question(ledger: LoopLedger, fingerprint: str) -> bool:
    return fingerprint in ledger.question_fingerprints


def admit_finding(ledger: LoopLedger, finding_id: str, fingerprint: str) -> LoopLedger:
    """Admit a validated finding unless its fingerprint is already on the ledger."""
    if is_duplicate_finding(ledger, fingerprint):
        return ledger
    return ledger.model_copy(
        update={
            "finding_fingerprints": [*ledger.finding_fingerprints, fingerprint],
            "validated_finding_ids": [*ledger.validated_finding_ids, finding_id],
        }
    )


def keep_or_discard(record: LoopRoundRecord) -> RoundDisposition:
    """§5.2: keep iff the round admitted at least one new (non-redundant) finding.

    new_validated_findings counts only non-redundant admissions, so an
    all-redundant round has it at 0 and is discarded. Crashed rounds keep the
    crash disposition regardless of counts.
    """
    if record.exit_reason == "crash":
        return "crash"
    return "keep" if record.new_validated_findings >= 1 else "discard"

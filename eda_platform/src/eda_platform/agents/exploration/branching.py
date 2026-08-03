"""Deterministic branch-abandonment constraint derivation (E6).

The same recipe runs live at abandonment time and inside the evidence issuer
at each ``branch_abandoned`` event position during replay; both sides must
produce identical constraints or the root is rejected. Inputs are therefore
restricted to durable material: committed receipts, journaled gate verdicts,
gate reports and candidate seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from eda_platform.agents.exploration.candidates import CandidateSeed
from eda_platform.core.claim_gates import GateReport
from eda_platform.schemas.exploration import (
    BranchConstraint,
    BranchConstraintReason,
    ExplorationLoopEvent,
    GateVerdictEvent,
    ReceiptCommittedEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.receipts import EvidenceReceipt


def receipt_hypothesis_id(receipt: EvidenceReceipt) -> str:
    """Executor-minted identity parsed from run_id; free-form fields are not trusted."""
    execution = receipt.execution
    if execution is None:
        raise ValueError("receipt lacks executor identity.")
    segments = execution.run_id.split(":")
    if (
        len(segments) < 6
        or segments[-1] != "execute_probes"
        or segments[-3] != "hypothesis"
        or not segments[-2]
    ):
        raise ValueError(
            f"receipt run_id {execution.run_id!r} has no hypothesis binding."
        )
    hypothesis_id = segments[-2]
    statistics = receipt.statistics
    if (
        statistics is not None
        and statistics.hypothesis_id is not None
        and statistics.hypothesis_id != hypothesis_id
    ):
        raise ValueError("receipt adjudication conflicts with executor identity.")
    return hypothesis_id


def bundle_hypotheses_from_events(
    events: Sequence[ExplorationLoopEvent],
    committed_receipts: Mapping[str, EvidenceReceipt],
) -> dict[str, str]:
    """Positional gate-order attribution, identical to the issuer's bundle replay."""
    mapping: dict[str, str] = {}
    in_round = False
    gate_order: list[str] = []
    gate_cursor = 0
    for event in events:
        if isinstance(event, RoundStartedEvent):
            in_round = True
            gate_order = []
            gate_cursor = 0
        elif isinstance(event, ReceiptCommittedEvent) and in_round:
            receipt = committed_receipts.get(event.receipt_id)
            if receipt is None:
                raise ValueError(
                    f"journal receipt {event.receipt_id!r} is not committed."
                )
            if receipt.facts:
                hypothesis_id = receipt_hypothesis_id(receipt)
                if hypothesis_id not in gate_order:
                    gate_order.append(hypothesis_id)
        elif isinstance(event, GateVerdictEvent):
            if not in_round or gate_cursor >= len(gate_order):
                raise ValueError("gate verdict lacks its deterministic receipt group.")
            mapping[event.claim_bundle_id] = gate_order[gate_cursor]
            gate_cursor += 1
    return mapping


def derive_branch_constraints(
    *,
    candidates: Mapping[str, CandidateSeed],
    committed_receipts: Mapping[str, EvidenceReceipt],
    gate_reports: Mapping[str, GateReport],
    bundle_hypotheses: Mapping[str, str],
    prior: Sequence[BranchConstraint],
) -> tuple[BranchConstraint, ...]:
    """New "tried + why it failed" entries for one abandonment event."""
    negatives: dict[tuple[str, BranchConstraintReason], BranchConstraint] = {}

    def note(
        hypothesis_id: str, reason: BranchConstraintReason, detail_code: str
    ) -> None:
        candidate = candidates.get(hypothesis_id)
        if candidate is None:
            raise ValueError(
                f"no candidate seed for abandoned hypothesis {hypothesis_id!r}."
            )
        entry = BranchConstraint(
            hypothesis_fingerprint=candidate.hypothesis_fingerprint,
            coverage_key=candidate.coverage_key,
            family=candidate.proposal.family,
            reason=reason,
            detail_code=detail_code,
        )
        key = (entry.hypothesis_fingerprint, reason)
        current = negatives.get(key)
        if current is None or entry.detail_code < current.detail_code:
            negatives[key] = entry

    for receipt_id in sorted(committed_receipts):
        receipt = committed_receipts[receipt_id]
        if not receipt.facts:
            continue
        hypothesis_id = receipt_hypothesis_id(receipt)
        statistics = receipt.statistics
        if statistics is not None and statistics.hypothesis_outcome == "contradicts":
            note(hypothesis_id, "refuted", receipt_id)
        elif statistics is None or statistics.hypothesis_outcome is None:
            note(hypothesis_id, "inconclusive", receipt_id)

    for bundle_id in sorted(gate_reports):
        report = gate_reports[bundle_id]
        if report.passed:
            continue
        hypothesis_id = bundle_hypotheses.get(bundle_id)
        if hypothesis_id is None:
            raise ValueError(
                f"rejected bundle {bundle_id!r} has no candidate hypothesis attribution."
            )
        failing = next(
            (verdict.gate for verdict in report.verdicts if not verdict.passed), None
        )
        if failing is None:
            raise ValueError(f"rejected report {bundle_id!r} has no failing gate.")
        note(hypothesis_id, "gate_rejected", f"{bundle_id}:{failing}")

    prior_keys = {(item.hypothesis_fingerprint, item.reason) for item in prior}
    fresh = [entry for key, entry in negatives.items() if key not in prior_keys]
    return tuple(
        sorted(
            fresh,
            key=lambda item: (
                item.hypothesis_fingerprint,
                item.reason,
                item.detail_code,
            ),
        )
    )

"""Issuer tolerance for runs interrupted mid-round (seed6 budget_exhausted shape).

A run stopped between round_started and round_settled has journal-committed
receipts that never reached a reduce, so the workflow-state snapshot lacks
them. Those trailing orphans are tolerated; every other discrepancy still
raises the original anti-fabrication error.
"""

from __future__ import annotations

import pytest

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.agents.receipts import build_receipt
from eda_platform.drivers.exploration_evidence_issuer import (
    _verify_committed_receipts,
)
from eda_platform.schemas.exploration import (
    ExplorationLoopEvent,
    ExplorationLoopState,
    ReceiptCommittedEvent,
    RoundSettledEvent,
    RoundStartedEvent,
)
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
)

_XPL = "xpl-interrupted"
_WITNESS = "witness-interruption-v1"


def _receipt(name: str, execution: object | None = None) -> EvidenceReceipt:
    return build_receipt(
        execution=execution,
        tool_call_id=f"call-{name}",
        tool_name="profile_slice",
        tool_version="1",
        arguments={"call": name},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds-1",),
            columns=("region", "revenue"),
            scope_resolution="explicit",
        ),
        facts=(
            ReceiptFact(
                fact_id="rows_in_slice",
                name="rows_in_slice",
                value=42,
                value_type="count",
                support_type="direct",
            ),
        ),
        method=ReceiptMethod(family="profile_slice"),
        data_state_witness=_WITNESS,
        created_at="2026-08-03T00:00:00Z",
    )


def _bound_receipt(name: str, hypothesis_id: str) -> EvidenceReceipt:
    """A receipt carrying the executor identity the issuer replays from."""
    from eda_platform.schemas.receipts import ReceiptExecution

    return _receipt(
        name,
        execution=ReceiptExecution(
            run_id=f"{_XPL}:round:0:hypothesis:{hypothesis_id}:execute_probes",
            provider_call_id=f"call-{name}",
            logical_step_id="step_" + name.ljust(24, "x")[:24],
            sequence_index=1,
        ),
    )


def _workflow(receipts: tuple[EvidenceReceipt, ...]) -> ExplorationWorkflowState:
    state = ExplorationWorkflowState()
    for receipt in receipts:
        state.committed_receipts[receipt.receipt_id] = receipt
    return state


def _terminal(receipts: tuple[EvidenceReceipt, ...]) -> ExplorationLoopState:
    return ExplorationLoopState(
        exploration_id=_XPL,
        policy_fingerprint="policy-v1",
        effective_policy_fingerprint="policy-v1",
        code_fingerprint="code-v1",
        data_state_witness=_WITNESS,
        attempt_epoch=0,
        max_successful_tool_calls=10,
        max_rounds=5,
        tool_calls_committed=len(receipts),
        tool_calls_by_kind={"profile_slice": len(receipts)} if receipts else {},
        remaining_tool_call_budget=10 - len(receipts),
        remaining_round_budget=5,
        last_seq=100,
        step_receipt_refs={
            f"step-{receipt.receipt_id}": receipt.receipt_id for receipt in receipts
        },
    )


def _committed(seq: int, receipt: EvidenceReceipt) -> ReceiptCommittedEvent:
    return ReceiptCommittedEvent(
        seq=seq,
        exploration_id=_XPL,
        logical_step_id=f"step-{receipt.receipt_id}",
        receipt_id=receipt.receipt_id,
    )


def _interrupted_run() -> tuple[
    ExplorationWorkflowState,
    ExplorationLoopState,
    list[ExplorationLoopEvent],
    tuple[EvidenceReceipt, ...],
]:
    """Round 0 settled with receipts a+b; round 1 committed c+d and never settled."""
    a, b, c, d = (_receipt(name) for name in ("a", "b", "c", "d"))
    events: list[ExplorationLoopEvent] = [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        _committed(2, a),
        _committed(3, b),
        RoundSettledEvent(seq=4, exploration_id=_XPL, round_index=0, progress=True),
        RoundStartedEvent(seq=5, exploration_id=_XPL, round_index=1),
        _committed(6, c),
        _committed(7, d),
    ]
    return _workflow((a, b)), _terminal((a, b, c, d)), events, (a, b, c, d)


def test_trailing_unsettled_round_orphans_are_tolerated() -> None:
    workflow, terminal, events, _ = _interrupted_run()
    _verify_committed_receipts(workflow, terminal, events)


def test_exact_receipt_match_still_passes() -> None:
    workflow, terminal, events, receipts = _interrupted_run()
    a, b, c, d = receipts
    workflow.committed_receipts[c.receipt_id] = c
    workflow.committed_receipts[d.receipt_id] = d
    _verify_committed_receipts(workflow, terminal, events)


def test_journal_only_receipt_before_last_settle_still_raises() -> None:
    """An orphan committed inside an already-settled round is not interruption."""
    a, b, c = (_receipt(name) for name in ("a", "b", "c"))
    events: list[ExplorationLoopEvent] = [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        _committed(2, a),
        _committed(3, b),
        RoundSettledEvent(seq=4, exploration_id=_XPL, round_index=0, progress=True),
        RoundStartedEvent(seq=5, exploration_id=_XPL, round_index=1),
        _committed(6, c),
    ]
    # b is journal-only but was committed before the last round_settled.
    workflow = _workflow((a,))
    terminal = _terminal((a, b, c))
    with pytest.raises(ValueError, match="do not exactly match journal commits"):
        _verify_committed_receipts(workflow, terminal, events)


def test_workflow_receipt_missing_from_journal_still_raises() -> None:
    workflow, terminal, events, receipts = _interrupted_run()
    fabricated = _receipt("fabricated")
    workflow.committed_receipts[fabricated.receipt_id] = fabricated
    with pytest.raises(ValueError, match="do not exactly match journal commits"):
        _verify_committed_receipts(workflow, terminal, events)


def test_orphans_without_any_settled_round_are_tolerated() -> None:
    """Interruption during round 0: no round_settled exists yet."""
    a, b = (_receipt(name) for name in ("a", "b"))
    events: list[ExplorationLoopEvent] = [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        _committed(2, a),
        _committed(3, b),
    ]
    _verify_committed_receipts(_workflow(()), _terminal((a, b)), events)


# --- concurrent probe sessions interleave their commits ----------------------


def test_gate_verdicts_match_by_content_not_by_commit_order() -> None:
    """Concurrency regression (gpt-5.6-luna seed 202): the issuer paired gate
    verdicts with hypotheses positionally, by first-receipt-commit order. The
    reducer gates in selection order, so interleaved commits shifted every
    pairing and a healthy root was rejected as forged."""
    from eda_platform.agents.exploration.candidates import candidate_seed
    from eda_platform.drivers.exploration_evidence_issuer import (
        _canonical_claim_bundle,
        _rebuild_canonical_bundles,
    )
    from eda_platform.schemas.exploration import GateVerdictEvent
    from eda_platform.schemas.hypotheses import HypothesisPredicate, HypothesisProposal
    from eda_platform.schemas.exploration import InsightFamily

    def _proposal(name: str) -> HypothesisProposal:
        return HypothesisProposal(
            statement=f"Does {name} matter?",
            rationale="Concurrency ordering fixture.",
            expected_evidence="A slice profile.",
            falsification_conditions=("It does not.",),
            family=InsightFamily.DIAGNOSTIC,
            method_family="profile_slice",
            dataset_ids=("ds-1",),
            columns=(name,),
            probe_kind="profile",
            predicate=HypothesisPredicate(metric=name, operator="exists"),
        )

    seeds = {
        name: candidate_seed(_proposal(name), sequence_index=index + 1)
        for index, name in enumerate(("alpha", "beta"))
    }
    receipts = {
        name: _bound_receipt(name, seed.hypothesis_id)
        for name, seed in seeds.items()
    }

    workflow = ExplorationWorkflowState()
    for receipt in receipts.values():
        workflow.committed_receipts[receipt.receipt_id] = receipt
    candidates_by_round = {0: {seed.hypothesis_id: seed for seed in seeds.values()}}
    bundles = {
        name: _canonical_claim_bundle(seeds[name], (receipts[name],))
        for name in seeds
    }

    # alpha commits first, but the reducer gated beta first.
    events: list[ExplorationLoopEvent] = [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        _committed(2, receipts["alpha"]),
        _committed(3, receipts["beta"]),
        GateVerdictEvent(
            seq=4,
            exploration_id=_XPL,
            claim_bundle_id=bundles["beta"].claim_bundle_id,
            verdict="passed",
        ),
        GateVerdictEvent(
            seq=5,
            exploration_id=_XPL,
            claim_bundle_id=bundles["alpha"].claim_bundle_id,
            verdict="passed",
        ),
        RoundSettledEvent(seq=6, exploration_id=_XPL, round_index=0, progress=True),
    ]

    rebuilt = _rebuild_canonical_bundles(
        events, workflow, candidates_by_round=candidates_by_round
    )
    assert set(rebuilt) == {
        bundles["alpha"].claim_bundle_id,
        bundles["beta"].claim_bundle_id,
    }


def test_a_gate_id_no_group_can_recompute_still_raises() -> None:
    """Control: content matching must not become 'accept any id'."""
    from eda_platform.agents.exploration.candidates import candidate_seed
    from eda_platform.drivers.exploration_evidence_issuer import _rebuild_canonical_bundles
    from eda_platform.schemas.exploration import GateVerdictEvent, InsightFamily
    from eda_platform.schemas.hypotheses import HypothesisPredicate, HypothesisProposal

    seed = candidate_seed(
        HypothesisProposal(
            statement="Does alpha matter?",
            rationale="Control fixture.",
            expected_evidence="A slice profile.",
            falsification_conditions=("It does not.",),
            family=InsightFamily.DIAGNOSTIC,
            method_family="profile_slice",
            dataset_ids=("ds-1",),
            columns=("alpha",),
            probe_kind="profile",
            predicate=HypothesisPredicate(metric="alpha", operator="exists"),
        ),
        sequence_index=1,
    )
    bound = _bound_receipt("alpha", seed.hypothesis_id)
    workflow = ExplorationWorkflowState()
    workflow.committed_receipts[bound.receipt_id] = bound
    events: list[ExplorationLoopEvent] = [
        RoundStartedEvent(seq=1, exploration_id=_XPL, round_index=0),
        _committed(2, bound),
        GateVerdictEvent(
            seq=3, exploration_id=_XPL, claim_bundle_id="cb_" + "0" * 24, verdict="passed"
        ),
        RoundSettledEvent(seq=4, exploration_id=_XPL, round_index=0, progress=True),
    ]
    with pytest.raises(ValueError, match="differs from canonical reducer bundle"):
        _rebuild_canonical_bundles(
            events,
            workflow,
            candidates_by_round={0: {seed.hypothesis_id: seed}},
        )

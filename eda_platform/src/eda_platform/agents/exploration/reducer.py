"""Deterministic insight-state transitions (doc §4.5).

The model submits a `TransitionProposal`; this module decides the status. The
proposed status is deliberately never read — it exists so the loop can record
what the model believed and compare it against what the evidence supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from eda_platform.schemas.claims import ClaimBundle, split_evidence_ref
from eda_platform.schemas.insights import (
    InsightProof,
    InsightRecord,
    InsightStatus,
    InsightTrustLevel,
    TransitionProposal,
)
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptScope,
    verify_receipt_digest,
)

_INDEPENDENT_REPLICATION_KINDS = frozenset({"holdout", "external_replication"})
_SAME_SNAPSHOT_LIMITATION = (
    "Additional same-snapshot corroboration; not an independent replication."
)


def reduce_insight(
    proposal: TransitionProposal,
    *,
    prior: InsightRecord | None,
    committed_receipts: Mapping[str, EvidenceReceipt],
    admitted_claim_bundles: Mapping[str, ClaimBundle],
    expected_witness: str,
    round_index: int,
    require_typed_hypothesis_outcome: bool = False,
    statement: str | None = None,
    rationale: str | None = None,
) -> InsightRecord:
    """Fold one proposal into the insight's next state.

    Raises rather than downgrading when cited evidence is missing or forged: a
    proposal citing evidence that does not exist is a control-plane fault, not a
    weaker finding.
    """
    if not expected_witness.strip():
        raise ValueError("expected_witness must be non-empty.")
    supporting = _resolve(
        proposal.supporting_receipt_ids,
        committed_receipts,
        expected_witness=expected_witness,
    )
    contradicting = _resolve(
        proposal.contradicting_receipt_ids,
        committed_receipts,
        expected_witness=expected_witness,
    )
    bundle = admitted_claim_bundles.get(proposal.claim_bundle_id)
    if bundle is None:
        raise ValueError(
            f"claim bundle {proposal.claim_bundle_id!r} has not passed the claim gates."
        )
    if bundle.hypothesis_id != proposal.hypothesis_id:
        raise ValueError("claim bundle hypothesis does not match the transition proposal.")
    assigned = set(supporting) | set(contradicting)
    cited = set(bundle.referenced_receipt_ids())
    if not assigned:
        raise ValueError("an insight transition requires committed evidence.")
    # The bundle also carries the round's unadjudicated receipts, which are
    # legitimate exploratory claims but never evidence for or against the
    # hypothesis; every side receipt must still be in it.
    if not assigned <= cited:
        raise ValueError(
            "claim bundle receipt references must cover the proposal's "
            "supporting and contradicting evidence."
        )
    new_supporting = supporting
    new_contradicting = contradicting

    if prior is not None:
        if prior.insight_id != proposal.insight_id:
            raise ValueError(
                f"proposal {proposal.insight_id!r} cannot update insight "
                f"{prior.insight_id!r}."
            )
        if prior.hypothesis_id != proposal.hypothesis_id or prior.family != proposal.family:
            raise ValueError("an insight update cannot change hypothesis or family.")
        _resolve(
            prior.supporting_receipt_ids,
            committed_receipts,
            expected_witness=expected_witness,
        )
        _resolve(
            prior.contradicting_receipt_ids,
            committed_receipts,
            expected_witness=expected_witness,
        )
        new_supporting = tuple(
            receipt_id
            for receipt_id in supporting
            if receipt_id not in prior.supporting_receipt_ids
        )
        new_contradicting = tuple(
            receipt_id
            for receipt_id in contradicting
            if receipt_id not in prior.contradicting_receipt_ids
        )
        if not new_supporting and not new_contradicting:
            raise ValueError("an insight update requires novel committed evidence.")
        # Refutation is sticky: evidence against a claim is not erased by a
        # later round that simply declines to cite it again.
        contradicting = _merge(prior.contradicting_receipt_ids, contradicting)
        supporting = _merge(prior.supporting_receipt_ids, supporting)

    # A narrower-scope contradiction is real exploratory evidence, but it does
    # not get to overturn a broader prior finding — "scan the whole table,
    # then recheck on a slice" is exactly the pattern this loop encourages, so
    # a scope mismatch downgrades to a limitation instead of aborting the
    # round (U7).
    narrow_contradicting: tuple[str, ...] = ()
    if new_contradicting and prior is not None:
        prior_scopes = [
            committed_receipts[receipt_id].scope
            for receipt_id in prior.supporting_receipt_ids
        ]
        if prior_scopes:
            narrow_contradicting = tuple(
                receipt_id
                for receipt_id in new_contradicting
                if not all(
                    _scope_covers(committed_receipts[receipt_id].scope, prior_scope)
                    for prior_scope in prior_scopes
                )
            )
    scope_limitations = tuple(
        _narrow_scope_limitation(receipt_id, committed_receipts[receipt_id])
        for receipt_id in narrow_contradicting
    )
    if narrow_contradicting:
        contradicting = tuple(
            receipt_id for receipt_id in contradicting if receipt_id not in narrow_contradicting
        )

    if require_typed_hypothesis_outcome:
        _assert_typed_sides(
            supporting=supporting,
            contradicting=contradicting,
            receipts=committed_receipts,
            hypothesis_id=proposal.hypothesis_id,
        )

    status, trust, extra_limitations = _verdict(
        supporting=supporting,
        contradicting=contradicting,
        is_update=prior is not None,
        new_supporting=new_supporting,
        receipts=committed_receipts,
        prior_status=prior.status if prior is not None else None,
        prior_trust=prior.trust_level if prior is not None else None,
        had_downgraded_contradiction=bool(narrow_contradicting),
        require_typed_hypothesis_outcome=require_typed_hypothesis_outcome,
    )
    proof = _proof(
        bundle,
        proposal,
        committed_receipts,
        downgraded_contradicting_receipt_ids=frozenset(narrow_contradicting),
    )
    # The hypothesis text is immutable once set: a prior's statement/rationale
    # survive every later update.
    if prior is not None and prior.statement is not None:
        statement = prior.statement
    if prior is not None and prior.rationale is not None:
        rationale = prior.rationale
    return InsightRecord(
        insight_id=proposal.insight_id,
        hypothesis_id=proposal.hypothesis_id,
        family=proposal.family,
        status=status,
        trust_level=trust,
        statement=statement,
        rationale=rationale,
        claim_bundle_id=proposal.claim_bundle_id,
        supporting_receipt_ids=supporting,
        contradicting_receipt_ids=contradicting,
        proof=_merge_proof(prior.proof if prior is not None else (), proof),
        limitations=_merge(
            prior.limitations if prior is not None else (),
            (*proposal.limitations, *extra_limitations, *scope_limitations),
        ),
        created_round=prior.created_round if prior is not None else round_index,
        last_updated_round=round_index,
    )


def _assert_typed_sides(
    *,
    supporting: Sequence[str],
    contradicting: Sequence[str],
    receipts: Mapping[str, EvidenceReceipt],
    hypothesis_id: str,
) -> None:
    """Typed runs: the executor's outcome, not the proposer, owns side placement."""
    for expected, receipt_ids in (
        ("supports", supporting),
        ("contradicts", contradicting),
    ):
        for receipt_id in receipt_ids:
            receipt = receipts[receipt_id]
            if receipt.method.hypothesis_evidence_is_explicitly_invalid():
                # The executor already spoke: this method cannot address the
                # hypothesis. _verdict downgrades it rather than raising, and
                # the workflow never puts such a receipt on a side anyway
                # (adjudicate_receipt_hypothesis clears its outcome).
                continue
            statistics = receipt.statistics
            outcome = None if statistics is None else statistics.hypothesis_outcome
            if statistics is None or outcome is None:
                raise ValueError(
                    f"receipt {receipt_id!r} carries no typed outcome and cannot "
                    "sit on an evidence side."
                )
            if outcome != expected:
                raise ValueError(
                    f"receipt {receipt_id!r} has typed outcome {outcome!r} and "
                    f"cannot sit on the {expected} side."
                )
            if statistics.hypothesis_id != hypothesis_id:
                raise ValueError(
                    f"receipt {receipt_id!r} adjudicates hypothesis "
                    f"{statistics.hypothesis_id!r}, not {hypothesis_id!r}."
                )


def _verdict(
    *,
    supporting: Sequence[str],
    contradicting: Sequence[str],
    is_update: bool,
    new_supporting: Sequence[str],
    receipts: Mapping[str, EvidenceReceipt],
    prior_status: InsightStatus | None,
    prior_trust: InsightTrustLevel | None,
    had_downgraded_contradiction: bool,
    require_typed_hypothesis_outcome: bool,
) -> tuple[InsightStatus, InsightTrustLevel, tuple[str, ...]]:
    if any(
        _receipt_is_inconclusive(
            receipts[receipt_id],
            require_typed_hypothesis_outcome=require_typed_hypothesis_outcome,
        )
        for receipt_id in (*supporting, *contradicting)
    ):
        return "inconclusive", "unsupported", ()
    if supporting and contradicting:
        return "inconclusive", "contested", ()
    if contradicting:
        return "refuted", "refuted", ()
    if not supporting:
        return "inconclusive", "unsupported", ()
    if not is_update:
        return "new", "supported", ()
    if not new_supporting:
        if had_downgraded_contradiction:
            # The only new evidence this round was a narrower-scope
            # contradiction, already downgraded to a limitation above — that
            # is not a reinforcement attempt, so the prior verdict stands.
            return (prior_status or "new"), (prior_trust or "supported"), ()
        raise ValueError("reinforcement requires novel supporting evidence.")
    independence_keys = {
        receipts[receipt_id].evidence_independence_key
        for receipt_id in new_supporting
        if receipts[receipt_id].evidence_independence_key
        and receipts[receipt_id].replication_kind
        in _INDEPENDENT_REPLICATION_KINDS
    }
    prior_keys = {
        receipts[receipt_id].evidence_independence_key
        for receipt_id in supporting
        if receipt_id not in new_supporting
        and receipts[receipt_id].evidence_independence_key
    }
    if not independence_keys - prior_keys:
        # R3 (user decision 2026-08-03): another query against the same snapshot
        # is neither an upgrade nor a reason to walk back what already held.
        return (prior_status or "new"), "supported", (_SAME_SNAPSHOT_LIMITATION,)
    return "reinforced", "supported", ()


def _resolve(
    receipt_ids: Sequence[str],
    committed: Mapping[str, EvidenceReceipt],
    *,
    expected_witness: str,
) -> tuple[str, ...]:
    if len(receipt_ids) != len(set(receipt_ids)):
        raise ValueError("receipt ids must be unique within one evidence side.")
    for receipt_id in receipt_ids:
        receipt = committed.get(receipt_id)
        if receipt is None:
            raise ValueError(f"receipt {receipt_id!r} is not committed.")
        if receipt.receipt_id != receipt_id or not verify_receipt_digest(receipt):
            raise ValueError(f"receipt {receipt_id!r} fails its content digest.")
        if receipt.data_state_witness != expected_witness:
            raise ValueError(
                f"receipt {receipt_id!r} does not match the run data-state witness."
            )
    return tuple(receipt_ids)


def _merge(existing: Sequence[str], incoming: Sequence[str]) -> tuple[str, ...]:
    """Union preserving first-seen order — the record must stay replayable."""
    merged = list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _proof(
    bundle: ClaimBundle,
    proposal: TransitionProposal,
    committed: Mapping[str, EvidenceReceipt],
    *,
    downgraded_contradicting_receipt_ids: frozenset[str] = frozenset(),
) -> tuple[InsightProof, ...]:
    fact_ids_by_receipt: dict[str, set[str]] = {}
    for claim in bundle.claims:
        for ref in claim.evidence_fact_ids:
            receipt_id, fact_id = split_evidence_ref(ref)
            fact_ids_by_receipt.setdefault(receipt_id, set()).add(fact_id)
        for ref in claim.derivation_ids:
            receipt_id, derivation_id = split_evidence_ref(ref)
            fact_ids_by_receipt.setdefault(receipt_id, set()).add(derivation_id)
        for receipt_id in claim.statistics_receipt_ids:
            receipt = committed[receipt_id]
            evidence_ids = {fact.fact_id for fact in receipt.facts}
            if not evidence_ids:
                raise ValueError(
                    f"statistics receipt {receipt_id!r} has no fact-level proof node."
                )
            fact_ids_by_receipt.setdefault(receipt_id, set()).update(evidence_ids)
    supporting = set(proposal.supporting_receipt_ids)
    # A receipt downgraded for scope reasons sits in neither side this round:
    # it does not support, and it was disqualified from contradicting.
    contradicting = set(proposal.contradicting_receipt_ids) - downgraded_contradicting_receipt_ids
    # Bundle receipts the executor never adjudicated get no proof edge: an edge
    # must say "supports" or "contradicts", and neither would be true.
    return tuple(
        InsightProof(
            receipt_id=receipt_id,
            fact_ids=tuple(sorted(fact_ids)),
            comparison="supports" if receipt_id in supporting else "contradicts",
            evidence_independence_key=committed[receipt_id].evidence_independence_key,
        )
        for receipt_id, fact_ids in sorted(fact_ids_by_receipt.items())
        if receipt_id in supporting or receipt_id in contradicting
    )


def _merge_proof(
    existing: Sequence[InsightProof], incoming: Sequence[InsightProof]
) -> tuple[InsightProof, ...]:
    by_key = {
        (proof.receipt_id, proof.comparison, proof.fact_ids): proof
        for proof in (*existing, *incoming)
    }
    return tuple(by_key[key] for key in sorted(by_key))


def _receipt_is_inconclusive(
    receipt: EvidenceReceipt, *, require_typed_hypothesis_outcome: bool
) -> bool:
    if receipt.method.hypothesis_evidence_is_explicitly_invalid():
        return True
    statistics = receipt.statistics
    if statistics is not None and not statistics.has_valid_numeric_values():
        return True
    # Execution adjacency binds evidence to the probe, but it is not semantic
    # adjudication. Only an executor-authored typed outcome may support/refute
    # the original hypothesis; every other observation remains inconclusive.
    if require_typed_hypothesis_outcome and (
        statistics is None or statistics.hypothesis_outcome is None
    ):
        return True
    # Typed outcomes are an executor-owned semantic decision. Informational
    # method warnings still belong in limitations, but do not override that
    # decision. Legacy/untyped callers retain the conservative warning rule.
    if not require_typed_hypothesis_outcome and receipt.method.warnings:
        return True
    if statistics is None or statistics.sample_size is None:
        return False
    minimum = receipt.method.parameters.get("min_sample_size")
    if minimum is None:
        minimum = receipt.method.parameters.get("min_n")
    return isinstance(minimum, (int, float)) and statistics.sample_size < minimum


def _scope_covers(candidate: ReceiptScope, target: ReceiptScope) -> bool:
    if not set(target.dataset_ids).issubset(candidate.dataset_ids):
        return False
    if (
        target.scope_resolution == "whole_dataset"
        and candidate.scope_resolution != "whole_dataset"
    ):
        return False
    if candidate.scope_resolution != "whole_dataset" and not set(target.columns).issubset(
        candidate.columns
    ):
        return False
    if candidate.filters != target.filters:
        return False
    if candidate.time_range != target.time_range:
        return False
    return True


def _narrow_scope_limitation(receipt_id: str, receipt: EvidenceReceipt) -> str:
    scope = receipt.scope
    return (
        f"Receipt {receipt_id!r} found contradicting evidence on a narrower scope "
        f"(filters={scope.filters!r}, time_range={scope.time_range!r}, "
        f"columns={scope.columns!r}) than the evidence that established this "
        "insight; it does not refute the broader finding, so review that receipt "
        "before trusting the insight beyond that narrower scope."
    )

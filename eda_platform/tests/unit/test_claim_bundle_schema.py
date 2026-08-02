"""ClaimBundle schema: strict shape so the gates never see ambiguous input.

Claims are the only path from agent output to the report (plan §4.6), so the
schema must refuse anything the reachability gate could not resolve: unqualified
fact references, claims with no evidence, duplicate ids, unknown fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eda_platform.schemas.claims import (
    Claim,
    ClaimBundle,
    ClaimScope,
    split_evidence_ref,
)

_RCPT = "rcpt_" + "0" * 24
_RCPT_B = "rcpt_" + "1" * 24


def _claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_id": "c1",
        "claim_type": "observation",
        "claim_text": "There are 42 orders.",
        "support_type": "direct",
        "evidence_fact_ids": (f"{_RCPT}:f1",),
    }
    fields.update(overrides)
    return Claim.model_validate(fields)


def _bundle(**overrides: object) -> ClaimBundle:
    fields: dict[str, object] = {
        "claim_bundle_id": "clb_1",
        "hypothesis_id": "hyp_1",
        "evidence_lane": "exploratory",
        "claims": (_claim(),),
    }
    fields.update(overrides)
    return ClaimBundle.model_validate(fields)


def test_a_minimal_bundle_validates_and_is_frozen() -> None:
    bundle = _bundle()
    assert bundle.claims[0].claim_id == "c1"
    with pytest.raises(ValidationError):
        bundle.claims[0].claim_text = "rewritten"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        bundle.evidence_lane = "confirmatory"  # type: ignore[misc]


def test_unknown_fields_are_rejected_at_both_levels() -> None:
    with pytest.raises(ValidationError):
        _bundle(surprise=1)
    with pytest.raises(ValidationError):
        _claim(surprise=1)


def test_claim_type_enum_is_the_0731_family_set() -> None:
    """§5.6 verbatim; a silent enum drift would desynchronise gates and renderer."""
    for claim_type in (
        "observation",
        "comparison",
        "prediction",
        "model",
        "absence",
        "causal",
        "recommendation",
    ):
        support = "absence" if claim_type == "absence" else "direct"
        assert _claim(claim_type=claim_type, support_type=support).claim_type == claim_type
    with pytest.raises(ValidationError):
        _claim(claim_type="trend")


def test_evidence_references_must_be_qualified() -> None:
    with pytest.raises(ValidationError, match="receipt_id"):
        _claim(evidence_fact_ids=("f1",))
    with pytest.raises(ValidationError, match="receipt_id"):
        _claim(derivation_ids=("rcpt_NOTHEX:d1",))
    with pytest.raises(ValidationError, match="rcpt_"):
        _claim(statistics_receipt_ids=("stat_1",))


def test_a_claim_without_any_evidence_reference_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cites no evidence"):
        _claim(evidence_fact_ids=(), derivation_ids=())


def test_absence_claim_type_and_support_type_are_coupled() -> None:
    with pytest.raises(ValidationError, match="absence"):
        _claim(claim_type="absence", support_type="direct")
    with pytest.raises(ValidationError, match="absence"):
        _claim(claim_type="observation", support_type="absence")
    claim = _claim(
        claim_type="absence",
        support_type="absence",
        scope=ClaimScope(dataset_ids=("ds_orders",), columns=("amount",)),
    )
    assert claim.scope is not None


def test_duplicate_claim_ids_and_empty_bundles_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _bundle(claims=(_claim(), _claim()))
    with pytest.raises(ValidationError):
        _bundle(claims=())


def test_bundle_identity_fields_are_required() -> None:
    with pytest.raises(ValidationError):
        _bundle(claim_bundle_id="")
    with pytest.raises(ValidationError):
        _bundle(hypothesis_id="")
    with pytest.raises(ValidationError):
        _bundle(evidence_lane="anecdotal")


def test_split_evidence_ref_and_referenced_receipt_ids() -> None:
    assert split_evidence_ref(f"{_RCPT}:pair0.pearson") == (_RCPT, "pair0.pearson")
    with pytest.raises(ValueError):
        split_evidence_ref("not-a-ref")
    bundle = _bundle(
        claims=(
            _claim(
                evidence_fact_ids=(f"{_RCPT}:f1", f"{_RCPT_B}:f2"),
                derivation_ids=(f"{_RCPT}:d1",),
                statistics_receipt_ids=(_RCPT_B,),
            ),
        )
    )
    assert bundle.referenced_receipt_ids() == (_RCPT, _RCPT_B)

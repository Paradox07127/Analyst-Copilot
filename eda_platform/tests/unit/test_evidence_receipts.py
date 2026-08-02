"""EvidenceReceipt: the typed envelope every analysis tool must emit.

The gate design (exploration plan §6) only admits claims whose numbers resolve
against receipt facts, so the receipt itself must be tamper-evident and
self-consistent: the content digest covers every field, derivations may only
reference facts that exist, and an absence fact is only constructible on an
empty result. NabaOS's published signature formula omits result_count/facts
from the MAC input while claiming they are protected — these tests pin the
corrected behaviour (digest covers everything) so we cannot inherit that bug.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eda_platform.agents.receipts import build_receipt
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    verify_receipt_digest,
)


def _receipt(**overrides: object) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id="call_1",
        tool_name="run_sql",
        tool_version="1",
        arguments={"sql": "SELECT count(*) AS n FROM orders", "purpose": "count rows"},
        raw_output={"rows": [{"n": 42}]},
        artifact_ids=("sql_abc",),
        result_count=1,
        scope=ReceiptScope(dataset_ids=("ds_orders",), columns=("n",)),
        facts=(
            ReceiptFact(
                fact_id="f1",
                name="n",
                value=42,
                value_type="count",
                unit="raw",
                support_type="direct",
            ),
        ),
        method=ReceiptMethod(family="sql_aggregation"),
        data_state_witness="witness_1",
        created_at="2026-08-01T00:00:00Z",
        **overrides,  # type: ignore[arg-type]
    )


def test_the_digest_covers_every_field() -> None:
    """Tampering with any field — including result_count and facts — must show."""
    receipt = _receipt()
    assert verify_receipt_digest(receipt)

    for field, value in [
        ("result_count", 99),
        ("tool_name", "another_tool"),
        ("data_state_witness", "witness_2"),
        (
            "facts",
            (
                ReceiptFact(
                    fact_id="f1",
                    name="n",
                    value=777,
                    value_type="count",
                    unit="raw",
                    support_type="direct",
                ),
            ),
        ),
    ]:
        tampered = receipt.model_copy(update={field: value})
        assert not verify_receipt_digest(tampered), f"tampered {field} went undetected"


def test_the_same_inputs_produce_the_same_receipt_identity() -> None:
    """Receipt ids and digests are content-derived, not random."""
    first, second = _receipt(), _receipt()
    assert first.receipt_id == second.receipt_id
    assert first.content_digest == second.content_digest


def test_a_derivation_may_only_reference_existing_facts() -> None:
    receipt = _receipt()
    with pytest.raises(ValidationError, match="unknown fact"):
        EvidenceReceipt.model_validate(
            {
                **receipt.model_dump(),
                "derivations": [
                    {
                        "derived_fact_id": "d1",
                        "operator": "percentage",
                        "input_fact_ids": ["f1", "f_missing"],
                    }
                ],
            }
        )


def test_an_absence_fact_requires_an_empty_result() -> None:
    """`result_count == 0` is the only ground for an absence fact."""
    receipt = _receipt()
    with pytest.raises(ValidationError, match="absence"):
        EvidenceReceipt.model_validate(
            {
                **receipt.model_dump(),
                "facts": [
                    {
                        "fact_id": "f1",
                        "name": "no_nulls",
                        "value": None,
                        "value_type": "null",
                        "unit": None,
                        "support_type": "absence",
                    }
                ],
            }
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceReceipt.model_validate({**_receipt().model_dump(), "surprise": 1})


def test_the_artifact_type_and_catalog_semantics_exist() -> None:
    """A receipt artifact must have explicit handoff catalog semantics, not the
    unknown-type fallback, so its priority is a decision rather than an accident."""
    from eda_platform.tools.agent_handoff import _catalog_semantics

    assert ArtifactType.EVIDENCE_RECEIPT.value == "EvidenceReceipt"
    stage, role, priority = _catalog_semantics(ArtifactType.EVIDENCE_RECEIPT)
    assert (stage, role, priority) == ("exploration", "evidence", "on_demand")

"""EvidenceReceipt — the typed, tamper-evident envelope of one tool call.

Only receipt facts (never tool prose or model text) feed the claim gates, so
the envelope must be self-consistent by construction: derivations reference
real facts, absence facts require an empty result, and `content_digest`
covers every other field. There is deliberately no HMAC in v1 — the threat
model is model fabrication, not a hostile local administrator.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

FactSupportType = Literal["direct", "compression", "inference", "absence"]
FactValueType = Literal["number", "percent", "count", "string", "bool", "null"]
DerivationOperator = Literal[
    "percentage", "difference", "relative_change", "weighted_average", "ratio"
]


class ReceiptFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    name: str
    value: float | int | str | bool | None
    value_type: FactValueType
    unit: str | None = None
    support_type: FactSupportType = "direct"


class ReceiptDerivation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    derived_fact_id: str
    operator: DerivationOperator
    # Inputs only; the validator recomputes the result and never trusts a
    # model-supplied value for it.
    input_fact_ids: tuple[str, ...]


class ReceiptScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_ids: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    filters: str | None = None
    time_range: str | None = None


class ReceiptMethod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    parameters: dict[str, str | int | float | bool | None] = {}
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ReceiptStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str | None = None
    test_name: str
    test_statistic: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    sample_size: int | None = None
    sequence_index: int | None = None


class EvidenceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str
    tool_call_id: str
    tool_name: str
    tool_version: str
    input_digest: str
    output_digest: str
    artifact_ids: tuple[str, ...]
    result_count: int
    scope: ReceiptScope
    facts: tuple[ReceiptFact, ...]
    derivations: tuple[ReceiptDerivation, ...] = ()
    method: ReceiptMethod
    statistics: ReceiptStatistics | None = None
    data_state_witness: str
    created_at: str
    content_digest: str

    @model_validator(mode="after")
    def _internally_consistent(self) -> EvidenceReceipt:
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique within a receipt.")
        known = set(fact_ids)
        for derivation in self.derivations:
            missing = [ref for ref in derivation.input_fact_ids if ref not in known]
            if missing:
                raise ValueError(
                    f"derivation {derivation.derived_fact_id!r} references "
                    f"unknown fact ids: {missing}."
                )
            if derivation.derived_fact_id in known:
                raise ValueError(
                    f"derived fact id {derivation.derived_fact_id!r} collides with a fact."
                )
        if self.result_count < 0:
            raise ValueError("result_count cannot be negative.")
        if self.result_count != 0:
            offenders = [f.fact_id for f in self.facts if f.support_type == "absence"]
            if offenders:
                # An absence fact asserts "nothing matched"; a non-empty result
                # contradicts it by construction (NabaOS abhava rule).
                raise ValueError(
                    f"absence facts {offenders} require result_count == 0, "
                    f"got {self.result_count}."
                )
        return self


def receipt_content_digest(receipt_fields: dict[str, object]) -> str:
    """Digest over every field except content_digest itself.

    NabaOS's published signature leaves result_count and facts uncovered while
    claiming they are protected; covering the full canonical body closes that
    hole rather than reproducing it.
    """
    body = {key: value for key, value in receipt_fields.items() if key != "content_digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_receipt_digest(receipt: EvidenceReceipt) -> bool:
    return receipt.content_digest == receipt_content_digest(receipt.model_dump(mode="json"))

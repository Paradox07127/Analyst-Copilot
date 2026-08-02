"""Construct EvidenceReceipts from tool executions.

Fact extraction stays in each tool's adapter (tool-specific and deterministic);
this module owns the envelope: stable identity, input/output digests, and the
content digest that makes tampering visible.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptDerivation,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
    receipt_content_digest,
)


def _sha256_canonical(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_receipt(
    *,
    tool_call_id: str,
    tool_name: str,
    tool_version: str,
    arguments: dict[str, Any],
    raw_output: Any,
    artifact_ids: tuple[str, ...],
    result_count: int,
    scope: ReceiptScope,
    facts: tuple[ReceiptFact, ...],
    method: ReceiptMethod,
    data_state_witness: str,
    created_at: str,
    derivations: tuple[ReceiptDerivation, ...] = (),
    statistics: ReceiptStatistics | None = None,
) -> EvidenceReceipt:
    """Assemble a receipt whose identity and digest are content-derived."""
    input_digest = _sha256_canonical(arguments)
    output_digest = _sha256_canonical(raw_output)
    receipt_id = "rcpt_" + stable_hash(
        {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "input_digest": input_digest,
            "output_digest": output_digest,
        },
        length=24,
    )
    fields: dict[str, Any] = {
        "receipt_id": receipt_id,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_version": tool_version,
        "input_digest": input_digest,
        "output_digest": output_digest,
        "artifact_ids": artifact_ids,
        "result_count": result_count,
        "scope": scope.model_dump(mode="json"),
        "facts": [fact.model_dump(mode="json") for fact in facts],
        "derivations": [derivation.model_dump(mode="json") for derivation in derivations],
        "method": method.model_dump(mode="json"),
        "statistics": statistics.model_dump(mode="json") if statistics is not None else None,
        "data_state_witness": data_state_witness,
        "created_at": created_at,
    }
    # Digest the serialized view — the same view verify_receipt_digest rebuilds.
    digest_view = dict(fields)
    digest_view["artifact_ids"] = list(artifact_ids)
    fields["content_digest"] = receipt_content_digest(digest_view)
    return EvidenceReceipt.model_validate(fields)

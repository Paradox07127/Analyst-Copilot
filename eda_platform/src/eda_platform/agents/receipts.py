"""Construct EvidenceReceipts from tool executions.

Fact extraction stays in each tool's adapter (tool-specific and deterministic);
this module owns the envelope: stable identity, input/output digests, and the
content digest that makes tampering visible.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from eda_platform.agents.tool_context import HypothesisExecutionBinding
from eda_platform.core.ids import stable_hash
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptDerivation,
    ReceiptExecution,
    ReceiptFact,
    ReceiptFactManifest,
    ReceiptMethod,
    ReceiptScope,
    ReceiptStatistics,
    ReplicationKind,
    receipt_content_digest,
)


def _sha256_canonical(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_recorded_scope(scope: ReceiptScope) -> None:
    """A new receipt must state what an empty column list means.

    `columns=()` with no `scope_resolution` cannot be told apart from "scope was
    never recorded" — the distinction a coverage gate has to make. Enforced here
    rather than on the model so receipts persisted before the rule still load.
    """
    if scope.columns or scope.scope_resolution is not None:
        return
    raise ValueError(
        "ReceiptScope.scope_resolution must be recorded when no columns are "
        "listed: use 'whole_dataset' for a whole-dataset scan, or 'resolved' "
        "with the columns actually scanned."
    )


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
    execution: ReceiptExecution | None = None,
    fact_manifest: ReceiptFactManifest | None = None,
    evidence_independence_key: str | None = None,
    replication_kind: ReplicationKind | None = None,
) -> EvidenceReceipt:
    """Assemble a receipt whose identity and digest are content-derived."""
    _require_recorded_scope(scope)
    input_digest = _sha256_canonical(arguments)
    output_digest = _sha256_canonical(raw_output)
    receipt_id = "rcpt_" + stable_hash(
        {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "input_digest": input_digest,
            "output_digest": output_digest,
            # Executor-owned position of the call; tool_call_id alone is derived
            # from provider text that repeats within a run. Stable across a
            # replay of the same logical step, so the id does not drift.
            "logical_step_id": execution.logical_step_id if execution else None,
        },
        length=24,
    )
    fields: dict[str, Any] = {
        "receipt_id": receipt_id,
        "tool_call_id": tool_call_id,
        "execution": execution.model_dump(mode="json") if execution is not None else None,
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
        "fact_manifest": (
            fact_manifest.model_dump(mode="json") if fact_manifest is not None else None
        ),
        "evidence_independence_key": evidence_independence_key,
        "replication_kind": replication_kind,
        "data_state_witness": data_state_witness,
        "created_at": created_at,
    }
    # Digest the serialized view — the same view verify_receipt_digest rebuilds.
    digest_view = dict(fields)
    digest_view["artifact_ids"] = list(artifact_ids)
    fields["content_digest"] = receipt_content_digest(digest_view)
    return EvidenceReceipt.model_validate(fields)


def adjudicate_receipt_hypothesis(
    receipt: EvidenceReceipt,
    binding: HypothesisExecutionBinding | None,
) -> EvidenceReceipt:
    """Bind a typed predicate to tool facts using versioned deterministic rules.

    The binding is minted by the exploration control plane and is absent for
    ordinary agent tool calls. The model never authors the outcome.
    """
    if binding is None:
        return receipt
    outcome = _predicate_outcome(receipt, binding)
    if (
        receipt.method.hypothesis_evidence_is_explicitly_invalid()
        or (
            receipt.statistics is not None
            and not receipt.statistics.has_valid_numeric_values()
        )
    ):
        outcome = None
    statistics = receipt.statistics
    if statistics is None:
        statistics = ReceiptStatistics(
            hypothesis_id=binding.hypothesis_id,
            hypothesis_outcome=outcome,
            test_name="system_predicate_adjudication_v1",
        )
    else:
        statistics = statistics.model_copy(
            update={
                "hypothesis_id": binding.hypothesis_id,
                "hypothesis_outcome": outcome,
            }
        )
    fields = receipt.model_dump(mode="json", exclude={"content_digest"})
    fields["statistics"] = statistics.model_dump(mode="json")
    fields["content_digest"] = receipt_content_digest(fields)
    return EvidenceReceipt.model_validate(fields)


def _predicate_outcome(
    receipt: EvidenceReceipt,
    binding: HypothesisExecutionBinding,
) -> Literal["supports", "contradicts"] | None:
    if not _method_matches_tool(binding.method_family, receipt.tool_name):
        return None
    if not set(binding.dataset_ids).issubset(receipt.scope.dataset_ids):
        return None
    if receipt.scope.scope_resolution != "whole_dataset" and not set(
        binding.columns
    ).issubset(receipt.scope.columns):
        return None
    predicate = binding.predicate
    operator = predicate.operator
    statistics = receipt.statistics
    if statistics is not None and operator in {"differs", "associated_with"}:
        p_value = (
            statistics.adjusted_p_value
            if statistics.adjusted_p_value is not None
            else statistics.p_value
        )
        if p_value is not None:
            if p_value > 0.05:
                # The test ran and found no effect: a direct negation of
                # differs/associated_with, not an unadjudicated case.
                return "contradicts"
            effect = statistics.effect_size
            materiality = predicate.threshold or 0.0
            if effect is not None:
                return "supports" if abs(effect) > materiality else "contradicts"
            if predicate.threshold:
                # Materiality was required but no effect size exists to check it.
                return None
            return "supports"

    # A statistical group-comparison effect is a standardized difference,
    # rank effect, eta-squared, odds ratio, or another test-family quantity. It
    # is not the predicate metric in that metric's own units. Therefore
    # greater_than/less_than must fall through to a recorded numeric fact named
    # by predicate.metric; a run_stat_test receipt does not currently publish
    # such a fact and correctly remains unadjudicated instead of turning the
    # sign of an effect into an absolute-threshold claim.

    facts = {fact.fact_id: fact.value for fact in receipt.facts}
    if operator == "associated_with" and receipt.tool_name == "diagnose_missingness":
        threshold = predicate.threshold
        if threshold is None or predicate.right_operand is None:
            return None
        target_pair = {
            predicate.metric.casefold(),
            predicate.right_operand.casefold(),
        }
        for fact_id, value in facts.items():
            if not fact_id.startswith("group_range") or not fact_id.endswith(".columns"):
                continue
            if not isinstance(value, str) or {
                item.strip().casefold() for item in value.split("~")
            } != target_pair:
                continue
            prefix = fact_id.removesuffix(".columns")
            delta = facts.get(prefix + ".percentage_points")
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                return "supports" if abs(float(delta)) >= threshold else "contradicts"
        return None

    if operator == "associated_with" and receipt.tool_name == "correlate_columns":
        if predicate.right_operand is None:
            return None
        target_pair = {
            predicate.metric.casefold(),
            predicate.right_operand.casefold(),
        }
        for fact_id, value in facts.items():
            if not fact_id.startswith("pair") or not fact_id.endswith(".columns"):
                continue
            if not isinstance(value, str) or {
                item.strip().casefold() for item in value.split("~")
            } != target_pair:
                continue
            prefix = fact_id.removesuffix(".columns")
            coefficient = _finite_fact_number(facts.get(prefix + ".coefficient"))
            p_value = _finite_fact_number(facts.get(prefix + ".adjusted_p"))
            if coefficient is None or p_value is None:
                return None
            if p_value > 0.05:
                return "contradicts"
            materiality = predicate.threshold or 0.0
            return "supports" if abs(coefficient) > materiality else "contradicts"
        return None

    if operator == "has_spike" and receipt.tool_name == "analyze_time_series":
        detected = facts.get("spike_detected")
        if isinstance(detected, bool):
            return "supports" if detected else "contradicts"
        return None

    if operator == "has_spike" and receipt.tool_name == "screen_anomalies":
        # The scanned column must be the predicate's target; binding.columns
        # may be empty and the subset gate above would then pass vacuously.
        if receipt.scope.scope_resolution != "whole_dataset" and predicate.metric.casefold() not in {
            column.casefold() for column in receipt.scope.columns
        }:
            return None
        outlier_count = _finite_fact_number(facts.get("outlier_count"))
        if outlier_count is None:
            return None
        return "supports" if outlier_count > 0 else "contradicts"

    if operator == "associated_with" and receipt.tool_name == "run_baseline_model":
        target = facts.get("target_column")
        if not isinstance(target, str):
            return None
        operands = {predicate.metric.casefold()}
        if predicate.right_operand is not None:
            operands.add(predicate.right_operand.casefold())
        if target.casefold() not in operands:
            return None
        task_type = facts.get("task_type")
        if task_type == "classification":
            accuracy = _finite_fact_number(facts.get("metric.accuracy"))
            baseline = _finite_fact_number(facts.get("baseline_accuracy"))
            if accuracy is None or baseline is None:
                return None
            skill = accuracy - baseline
        elif task_type == "regression":
            # R^2 is skill over the mean-predictor baseline by definition.
            r2 = _finite_fact_number(facts.get("metric.r2"))
            if r2 is None:
                return None
            skill = r2
        else:
            return None
        materiality = predicate.threshold or 0.0
        return "supports" if skill > materiality else "contradicts"

    if operator in {"exists", "absent"}:
        exists = receipt.result_count > 0
        return "supports" if exists == (operator == "exists") else "contradicts"

    value = _predicate_numeric_value(facts, predicate.metric)
    threshold = predicate.threshold
    if value is None or threshold is None:
        return None
    if operator == "greater_than":
        return "supports" if value > threshold else "contradicts"
    if operator == "less_than":
        return "supports" if value < threshold else "contradicts"
    if operator == "equal_to":
        return "supports" if value == threshold else "contradicts"
    if operator == "not_equal_to":
        return "supports" if value != threshold else "contradicts"
    return None


def _finite_fact_number(value: float | int | str | bool | None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _method_matches_tool(method_family: str, tool_name: str) -> bool:
    """Resolve the small system-owned conceptual-method alias set fail closed."""
    normalized_method = method_family.strip().casefold()
    normalized_tool = tool_name.strip().casefold()
    aliases = {
        "compare_groups": frozenset({"run_stat_test"}),
    }
    allowed = aliases.get(normalized_method, frozenset({normalized_method}))
    return normalized_tool in allowed


def _predicate_numeric_value(
    facts: dict[str, float | int | str | bool | None], metric: str
) -> float | None:
    normalized = metric.casefold()
    for fact_id, value in facts.items():
        if fact_id.casefold() == normalized and isinstance(value, (int, float)) and not isinstance(
            value, bool
        ):
            return float(value)
    for fact_id, value in facts.items():
        if not fact_id.endswith(".name") or not isinstance(value, str):
            continue
        if value.casefold() != normalized:
            continue
        prefix = fact_id.removesuffix(".name")
        for suffix in (".missing_percent", ".value", ".mean"):
            candidate = facts.get(prefix + suffix)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
    return None

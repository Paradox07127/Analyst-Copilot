"""Six deterministic exit gates over a ClaimBundle (plan §6.1-§6.4).

Gate order is fixed: structure, reachability, numeric, entity, statistical,
state. A structure failure short-circuits (nothing downstream is resolvable);
every other gate runs to completion so the retry feedback carries the full
failure list. The statistical gate registers violations without blocking (v1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from eda_platform.core.claim_language import (
    asserts_model_capability,
    implies_causation,
)
from eda_platform.schemas.claims import (
    Claim,
    ClaimBundle,
    ClaimScope,
    split_evidence_ref,
)
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptDerivation,
    ReceiptFact,
    ReceiptScope,
    verify_receipt_digest,
)

# Canonical token extraction and exact/half-ULP tolerance ladder (analysis-v3
# §5.2); promoted to public names in E3-B, aliased so call sites stay unchanged.
from eda_platform.tools.report_validator import (
    numeric_tokens_from_text as _numeric_tokens_from_text,
    satisfies_threshold as _satisfies_threshold,
    value_supports_token as _value_supports_token,
)
from eda_platform.tools.stat_tests import TEST_PUBLISHABILITY

GATE_RETRY_BUDGET = 1
ABSTAINED_STATUS: Literal["abstained"] = "abstained"

GateName = Literal[
    "structure", "reachability", "numeric", "entity", "statistical", "state"
]
GATE_ORDER: tuple[GateName, ...] = (
    "structure",
    "reachability",
    "numeric",
    "entity",
    "statistical",
    "state",
)

# Receipts that license model/prediction claims. run_baseline_model is not an
# agent tool yet; this constant is the contract its future adapter must meet.
MODEL_EVIDENCE_TOOL_NAMES = frozenset({"run_baseline_model"})
MODEL_EVIDENCE_METHOD_FAMILIES = frozenset({"ml_baseline", "model_card"})

# R4: only these replication kinds license an independent-replication claim.
_INDEPENDENCE_LICENSING_KINDS = frozenset({"holdout", "external_replication"})
INDEPENDENT_REPLICATION_PHRASES: tuple[str, ...] = (
    "independent replication",
    "independently replicated",
    "replicated independently",
    "external replication",
    "externally replicated",
)

_POOL_SUMMARY_CAP = 12
_STRUCTURE_ERROR_CAP = 20


class GateViolation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    claim_id: str | None = None
    # Set when the failure is attributable to one tool call (e.g. a derivation
    # recompute failure); build_gate_feedback routes those to the tool channel.
    tool_call_id: str | None = None


class GateVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate: GateName
    passed: bool
    violations: tuple[GateViolation, ...] = ()


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_bundle_id: str
    passed: bool
    verdicts: tuple[GateVerdict, ...]
    # GroundEval Eq.1 multiplier (1 - v)^2 — a run-health metric, not a gate.
    health_score: float


class GateFeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: Literal["tool", "user"]
    tool_call_id: str | None = None
    message: str

    @model_validator(mode="after")
    def _tool_channel_is_bound(self) -> GateFeedbackItem:
        if self.channel == "tool" and not self.tool_call_id:
            raise ValueError("tool-channel feedback requires a tool_call_id.")
        return self


class DerivationRecomputeError(ValueError):
    """A whitelist derivation could not be recomputed from its input facts."""


def retry_decision(gate_failure_count: int) -> Literal["retry", "abstained"]:
    """One retry (plan §6.4); a second failure abstains, never ships text."""
    return "retry" if gate_failure_count <= GATE_RETRY_BUDGET else ABSTAINED_STATUS


def recompute_derivation(
    derivation: ReceiptDerivation,
    facts_by_id: Mapping[str, ReceiptFact],
) -> tuple[float, bool]:
    """Recompute a whitelist operator from input fact values only.

    Returns (value, is_percent). The stored derivation carries no result field
    by design; this recomputation is the value the numeric gate admits.
    """
    values: list[float] = []
    for ref in derivation.input_fact_ids:
        fact = facts_by_id.get(ref)
        if fact is None:
            raise DerivationRecomputeError(
                f"derivation {derivation.derived_fact_id!r}: input fact {ref!r} is missing."
            )
        if isinstance(fact.value, bool) or not isinstance(fact.value, (int, float)):
            raise DerivationRecomputeError(
                f"derivation {derivation.derived_fact_id!r}: input fact {ref!r} is not numeric."
            )
        values.append(float(fact.value))
    operator = derivation.operator
    if operator in {"percentage", "relative_change", "ratio"} and values[1] == 0.0:
        raise DerivationRecomputeError(
            f"derivation {derivation.derived_fact_id!r}: division by zero."
        )
    if operator == "percentage":
        return values[0] / values[1] * 100.0, True
    if operator == "difference":
        return values[0] - values[1], False
    if operator == "relative_change":
        return (values[0] - values[1]) / values[1] * 100.0, True
    if operator == "ratio":
        return values[0] / values[1], False
    # weighted_average: alternating (value, weight) pairs; even arity is
    # already enforced by the receipt schema.
    pairs = list(zip(values[0::2], values[1::2], strict=True))
    total_weight = sum(weight for _value, weight in pairs)
    if total_weight == 0.0:
        raise DerivationRecomputeError(
            f"derivation {derivation.derived_fact_id!r}: weights sum to zero."
        )
    return sum(value * weight for value, weight in pairs) / total_weight, False


@dataclass
class _ClaimEvidence:
    """Per-claim resolution result shared by the numeric/entity/state gates."""

    fact_pairs: list[tuple[EvidenceReceipt, ReceiptFact]] = field(default_factory=list)
    derivation_pairs: list[tuple[EvidenceReceipt, ReceiptDerivation]] = field(
        default_factory=list
    )
    stats_receipts: list[EvidenceReceipt] = field(default_factory=list)
    fact_receipts: dict[str, EvidenceReceipt] = field(default_factory=dict)
    receipts: dict[str, EvidenceReceipt] = field(default_factory=dict)


def run_claim_gates(
    bundle: ClaimBundle | Mapping[str, Any],
    *,
    committed_receipts: Mapping[str, EvidenceReceipt],
    run_witness: str,
) -> GateReport:
    """Run the six exit gates and return one structured verdict per gate."""
    # Revalidate even an already-typed bundle: model_construct/model_copy skip
    # the validators, so a forged instance could cite no evidence at all and
    # still be reported as structurally sound (same reasoning as
    # exploration_budget.effective_policy).
    payload: Any = bundle.model_dump() if isinstance(bundle, ClaimBundle) else bundle
    try:
        bundle = ClaimBundle.model_validate(payload)
    except ValidationError as exc:
        return _structure_failure_report(payload, exc)
    structure = GateVerdict(gate="structure", passed=True)

    reach_violations, evidence = _resolve_references(bundle, committed_receipts)
    numeric_violations = _numeric_gate(bundle, evidence)
    entity_violations = _entity_gate(bundle, evidence)
    statistical_violations = _statistical_gate(bundle, evidence)
    state_violations = _state_gate(evidence, run_witness)

    verdicts = (
        structure,
        GateVerdict(
            gate="reachability",
            passed=not reach_violations,
            violations=tuple(reach_violations),
        ),
        GateVerdict(
            gate="numeric",
            passed=not numeric_violations,
            violations=tuple(numeric_violations),
        ),
        GateVerdict(
            gate="entity",
            passed=not entity_violations,
            violations=tuple(entity_violations),
        ),
        # Registering-only in v1: violations are recorded, never blocking.
        GateVerdict(
            gate="statistical", passed=True, violations=tuple(statistical_violations)
        ),
        GateVerdict(
            gate="state", passed=not state_violations, violations=tuple(state_violations)
        ),
    )
    all_violations = [v for verdict in verdicts for v in verdict.violations]
    return GateReport(
        claim_bundle_id=bundle.claim_bundle_id,
        passed=all(verdict.passed for verdict in verdicts),
        verdicts=verdicts,
        health_score=_health_score(len(bundle.claims), all_violations),
    )


def build_gate_feedback(report: GateReport) -> list[GateFeedbackItem]:
    """Two-channel retry feedback (plan §6.4).

    Violations bound to one tool call go back on the tool channel; everything
    else becomes one user-channel message. Messages carry the failure list and
    allowed-pool summaries only — never the failed claim text.
    """
    if report.passed:
        return []
    blocking = [
        violation
        for verdict in report.verdicts
        if not verdict.passed
        for violation in verdict.violations
    ]
    by_tool: dict[str, list[GateViolation]] = {}
    general: list[GateViolation] = []
    for violation in blocking:
        if violation.tool_call_id is not None:
            by_tool.setdefault(violation.tool_call_id, []).append(violation)
        else:
            general.append(violation)
    items = [
        GateFeedbackItem(
            channel="tool",
            tool_call_id=tool_call_id,
            message="\n".join(_bullet(violation) for violation in violations),
        )
        for tool_call_id, violations in by_tool.items()
    ]
    if general:
        body = "\n".join(_bullet(violation) for violation in general)
        items.append(
            GateFeedbackItem(
                channel="user",
                message=f"Validation feedback:\n{body}\nFix the errors and try again.",
            )
        )
    return items


def _bullet(violation: GateViolation) -> str:
    prefix = f"claim {violation.claim_id}: " if violation.claim_id else ""
    return f"- {prefix}[{violation.code}] {violation.message}"


def _structure_failure_report(
    payload: Mapping[str, Any] | Any, exc: ValidationError
) -> GateReport:
    # include_input=False: schema errors must not echo model-authored text.
    violations = tuple(
        GateViolation(
            code="schema_invalid",
            message=f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}",
        )
        for error in exc.errors(include_url=False, include_input=False)[
            :_STRUCTURE_ERROR_CAP
        ]
    )
    bundle_id = ""
    if isinstance(payload, Mapping):
        raw = payload.get("claim_bundle_id")
        if isinstance(raw, str):
            bundle_id = raw
    return GateReport(
        claim_bundle_id=bundle_id or "<invalid>",
        passed=False,
        verdicts=(
            GateVerdict(gate="structure", passed=False, violations=violations),
        ),
        health_score=0.0,
    )


def _resolve_references(
    bundle: ClaimBundle,
    committed_receipts: Mapping[str, EvidenceReceipt],
) -> tuple[list[GateViolation], dict[str, _ClaimEvidence]]:
    """Reachability gate: every reference must land on a digest-verified,
    committed receipt (load_verified_receipt semantics — NabaOS lookup plus
    GroundEval cited∩fetched reconciliation)."""
    violations: list[GateViolation] = []
    digest_ok: dict[str, bool] = {}
    evidence: dict[str, _ClaimEvidence] = {}

    def _verified(receipt_id: str, claim_id: str) -> EvidenceReceipt | None:
        receipt = committed_receipts.get(receipt_id)
        if receipt is None:
            violations.append(
                GateViolation(
                    code="fabricated_receipt",
                    message=f"receipt {receipt_id} was never committed in this run.",
                    claim_id=claim_id,
                )
            )
            return None
        if receipt_id not in digest_ok:
            digest_ok[receipt_id] = (
                receipt.receipt_id == receipt_id and verify_receipt_digest(receipt)
            )
        if not digest_ok[receipt_id]:
            violations.append(
                GateViolation(
                    code="receipt_digest_mismatch",
                    message=f"receipt {receipt_id} fails content-digest verification.",
                    claim_id=claim_id,
                    tool_call_id=receipt.tool_call_id,
                )
            )
            return None
        return receipt

    for claim in bundle.claims:
        resolved = _ClaimEvidence()
        evidence[claim.claim_id] = resolved
        for ref in claim.evidence_fact_ids:
            receipt_id, fact_id = split_evidence_ref(ref)
            receipt = _verified(receipt_id, claim.claim_id)
            if receipt is None:
                continue
            resolved.receipts[receipt_id] = receipt
            fact = next((f for f in receipt.facts if f.fact_id == fact_id), None)
            if fact is None:
                violations.append(
                    GateViolation(
                        code="unknown_fact",
                        message=(
                            f"fact {fact_id!r} does not exist in receipt {receipt_id}."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
                continue
            resolved.fact_pairs.append((receipt, fact))
            resolved.fact_receipts[receipt_id] = receipt
        for ref in claim.derivation_ids:
            receipt_id, derived_id = split_evidence_ref(ref)
            receipt = _verified(receipt_id, claim.claim_id)
            if receipt is None:
                continue
            resolved.receipts[receipt_id] = receipt
            derivation = next(
                (d for d in receipt.derivations if d.derived_fact_id == derived_id),
                None,
            )
            if derivation is None:
                violations.append(
                    GateViolation(
                        code="unknown_derivation",
                        message=(
                            f"derivation {derived_id!r} does not exist in receipt "
                            f"{receipt_id}."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
                continue
            resolved.derivation_pairs.append((receipt, derivation))
        for receipt_id in claim.statistics_receipt_ids:
            receipt = _verified(receipt_id, claim.claim_id)
            if receipt is None:
                continue
            resolved.receipts[receipt_id] = receipt
            if receipt.statistics is None:
                violations.append(
                    GateViolation(
                        code="missing_statistics",
                        message=(
                            f"receipt {receipt_id} carries no statistics but is "
                            "cited as statistical backing."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
                continue
            resolved.stats_receipts.append(receipt)
    return violations, evidence


_NUMERIC_FACT_TYPES = {"number", "count", "percent"}


def _numeric_gate(
    bundle: ClaimBundle, evidence: dict[str, _ClaimEvidence]
) -> list[GateViolation]:
    """§4.6 hard constraint: every claim-text number resolves to a cited fact
    value or a recomputed whitelist derivation. The claim never feeds its own
    pool; count facts match exactly, floats by half-ULP of displayed digits."""
    violations: list[GateViolation] = []
    for claim in bundle.claims:
        resolved = evidence[claim.claim_id]
        raw_pool: list[tuple[float, str]] = []
        percent_pool: list[tuple[float, str]] = []
        for _receipt, fact in resolved.fact_pairs:
            if fact.value_type not in _NUMERIC_FACT_TYPES:
                continue
            if isinstance(fact.value, bool) or not isinstance(fact.value, (int, float)):
                continue
            policy = (
                "exact"
                if fact.value_type == "count" or isinstance(fact.value, int)
                else "rounded"
            )
            pool = percent_pool if fact.value_type == "percent" else raw_pool
            pool.append((float(fact.value), policy))
        for receipt, derivation in resolved.derivation_pairs:
            try:
                value, is_percent = recompute_derivation(
                    derivation, {fact.fact_id: fact for fact in receipt.facts}
                )
            except DerivationRecomputeError as exc:
                violations.append(
                    GateViolation(
                        code="derivation_recompute_failed",
                        message=str(exc),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
                continue
            (percent_pool if is_percent else raw_pool).append((value, "rounded"))

        tokens = _numeric_tokens_from_text(claim.claim_text)
        if not tokens:
            continue
        if not raw_pool and not percent_pool:
            violations.append(
                GateViolation(
                    code="no_evidence_values",
                    message=(
                        f"{len(tokens)} numeric token(s) cannot be verified: the "
                        "cited evidence resolves no numeric values."
                    ),
                    claim_id=claim.claim_id,
                )
            )
            continue
        # Threshold tokens (p < 0.05) may verify only against the cited
        # statistic they name — pooling the three fields let "losses were
        # < 1000000" verify against a p-value of 0.003.
        eligible: dict[str, list[float]] = {}
        for receipt in resolved.stats_receipts:
            statistics = receipt.statistics
            if statistics is None:
                continue
            for subject, value in (
                ("p_value", statistics.p_value),
                ("test_statistic", statistics.test_statistic),
                ("effect_size", statistics.effect_size),
            ):
                if value is not None:
                    eligible.setdefault(subject, []).append(float(value))
        for token in tokens:
            pool = percent_pool if token.is_percent else raw_pool
            subject_values = (
                eligible.get(token.threshold_subject, [])
                if token.threshold_subject is not None
                else []
            )
            if token.threshold_op is not None and not token.is_percent and subject_values:
                verified = _satisfies_threshold(token, subject_values)
            else:
                verified = any(
                    _value_supports_token(token, value, policy)
                    for value, policy in pool
                )
            if not verified:
                violations.append(
                    GateViolation(
                        code="numeric_mismatch",
                        message=(
                            f"number {_format_token(token.value, token.is_percent)} is "
                            "not supported by the cited evidence; "
                            + _pool_summary(raw_pool, percent_pool)
                        ),
                        claim_id=claim.claim_id,
                    )
                )
    return violations


def _format_token(value: float, is_percent: bool) -> str:
    return f"{value:g}%" if is_percent else f"{value:g}"


def _pool_summary(
    raw_pool: list[tuple[float, str]], percent_pool: list[tuple[float, str]]
) -> str:
    def _fmt(pool: list[tuple[float, str]], suffix: str) -> str:
        values = sorted({value for value, _policy in pool})[:_POOL_SUMMARY_CAP]
        return "[" + ", ".join(f"{value:g}{suffix}" for value in values) + "]"

    return (
        f"allowed raw values: {_fmt(raw_pool, '')}; "
        f"allowed percent values: {_fmt(percent_pool, '%')}"
    )


def _entity_gate(
    bundle: ClaimBundle, evidence: dict[str, _ClaimEvidence]
) -> list[GateViolation]:
    violations: list[GateViolation] = []
    independence_licensed = any(
        receipt.replication_kind in _INDEPENDENCE_LICENSING_KINDS
        for resolved in evidence.values()
        for receipt in resolved.receipts.values()
    )
    for claim in bundle.claims:
        resolved = evidence[claim.claim_id]
        needs_model = claim.claim_type in {"model", "prediction"} or (
            asserts_model_capability(claim.claim_text)
        )
        if needs_model and not any(
            receipt.tool_name in MODEL_EVIDENCE_TOOL_NAMES
            or receipt.method.family in MODEL_EVIDENCE_METHOD_FAMILIES
            for receipt in resolved.receipts.values()
        ):
            violations.append(
                GateViolation(
                    code="model_claim_without_model_evidence",
                    message=(
                        "the claim asserts model/prediction capability but cites "
                        "no model-card receipt."
                    ),
                    claim_id=claim.claim_id,
                )
            )
        if claim.claim_type == "causal":
            violations.append(
                GateViolation(
                    code="causal_claim_rejected",
                    message=(
                        "causal claims are rejected by default in v1; rephrase "
                        "as an observed association."
                    ),
                    claim_id=claim.claim_id,
                )
            )
        elif implies_causation(claim.claim_text):
            violations.append(
                GateViolation(
                    code="causal_language",
                    message=(
                        "the claim text asserts causation; only associative "
                        "wording is admissible."
                    ),
                    claim_id=claim.claim_id,
                )
            )
        if claim.claim_type == "recommendation":
            if claim.support_type != "inference":
                violations.append(
                    GateViolation(
                        code="recommendation_not_marked_inference",
                        message="recommendation claims must use support_type 'inference'.",
                        claim_id=claim.claim_id,
                    )
                )
            if not claim.assumptions:
                violations.append(
                    GateViolation(
                        code="recommendation_without_assumptions",
                        message="recommendation claims must list their assumptions.",
                        claim_id=claim.claim_id,
                    )
                )
        if claim.claim_type == "absence":
            violations.extend(_absence_gate(claim, resolved))
        if not independence_licensed and _declares_independence_in_text(claim):
            violations.append(
                GateViolation(
                    code="unlicensed_independence_claim",
                    message=(
                        "independent replication is claimed, but every cited "
                        "receipt is same-snapshot evidence (R4: a second query "
                        "on the same snapshot is corroboration, not replication)."
                    ),
                    claim_id=claim.claim_id,
                )
            )
    if bundle.declares_independent_replication and not independence_licensed:
        violations.append(
            GateViolation(
                code="unlicensed_independence_claim",
                message=(
                    "the bundle declares independent replication, but no cited "
                    "receipt carries replication_kind holdout/external_replication."
                ),
            )
        )
    return violations


def _declares_independence_in_text(claim: Claim) -> bool:
    lowered = claim.claim_text.lower()
    return any(phrase in lowered for phrase in INDEPENDENT_REPLICATION_PHRASES)


def _absence_gate(claim: Claim, resolved: _ClaimEvidence) -> list[GateViolation]:
    """GroundEval must-check-set gate (§6.3): a correct 'no' is not sufficient
    — the coverage receipts must have actually scanned the declared scope, and
    result_count == 0 is the only admissible ground."""
    violations: list[GateViolation] = []
    coverage = [
        receipt
        for receipt in resolved.fact_receipts.values()
        if receipt.result_count == 0
    ]
    for receipt in resolved.fact_receipts.values():
        if receipt.result_count > 0:
            violations.append(
                GateViolation(
                    code="false_absence",
                    message=(
                        f"receipt {receipt.receipt_id} returned "
                        f"{receipt.result_count} row(s); an absence claim requires "
                        "result_count == 0."
                    ),
                    claim_id=claim.claim_id,
                    tool_call_id=receipt.tool_call_id,
                )
            )
    if not coverage:
        violations.append(
            GateViolation(
                code="absence_without_coverage",
                message="an absence claim must cite an empty-result coverage receipt.",
                claim_id=claim.claim_id,
            )
        )
        return violations
    if claim.scope is None or not claim.scope.dataset_ids:
        violations.append(
            GateViolation(
                code="absence_scope_missing",
                message="an absence claim must declare the scope it checked.",
                claim_id=claim.claim_id,
            )
        )
        return violations
    # Per receipt, not two independent unions: unioning datasets and columns
    # separately let "A scanned ds_orders.amount" plus "B scanned ds_users.email"
    # license ds_orders.email, a pair no receipt ever touched.
    usable = [
        receipt
        for receipt in coverage
        if _narrowing_covers(claim.scope, receipt.scope)
    ]
    uncovered = sorted(
        f"{dataset_id}.{column}" if column else dataset_id
        for dataset_id, column in _required_scope_pairs(claim.scope)
        if not any(
            dataset_id in receipt.scope.dataset_ids
            and (column is None or column in receipt.scope.columns)
            for receipt in usable
        )
    )
    if uncovered:
        narrowed = len(usable) < len(coverage)
        violations.append(
            GateViolation(
                code="absence_scope_exceeds_coverage",
                message=(
                    "the declared absence scope was not actually scanned: "
                    f"{uncovered}."
                    + (
                        " Some coverage receipts were restricted by filters or a "
                        "time range the claim does not declare."
                        if narrowed
                        else ""
                    )
                ),
                claim_id=claim.claim_id,
            )
        )
    return violations


def _required_scope_pairs(
    scope: ClaimScope,
) -> list[tuple[str, str | None]]:
    columns: tuple[str | None, ...] = scope.columns or (None,)
    return [(dataset_id, column) for dataset_id in scope.dataset_ids for column in columns]


def _normalized_restriction(value: str | None) -> str | None:
    return " ".join(value.split()) if value and value.strip() else None


def _narrowing_covers(claim_scope: ClaimScope, receipt_scope: ReceiptScope) -> bool:
    """An unrestricted scan covers any declared sub-slice; a restricted one only
    covers the identical restriction. Filter strings are free-form SQL, so
    subsumption cannot be proven — identity is the only sound test, and the
    unsound direction (receipt narrower than the claim) is what let a 3-day
    single-region scan license an 'anywhere, full history' absence."""
    for receipt_value, claim_value in (
        (receipt_scope.filters, claim_scope.filters),
        (receipt_scope.time_range, claim_scope.time_range),
    ):
        restriction = _normalized_restriction(receipt_value)
        if restriction is None:
            continue
        if restriction != _normalized_restriction(claim_value):
            return False
    return True


def _statistical_gate(
    bundle: ClaimBundle, evidence: dict[str, _ClaimEvidence]
) -> list[GateViolation]:
    """v1 registers statistical-completeness violations without blocking."""
    violations: list[GateViolation] = []
    confirmatory = bundle.evidence_lane == "confirmatory"
    for claim in bundle.claims:
        stats_receipts = evidence[claim.claim_id].stats_receipts
        if confirmatory and not stats_receipts:
            violations.append(
                GateViolation(
                    code="confirmatory_without_statistics",
                    message=(
                        "a confirmatory-lane claim cites no statistics receipt."
                    ),
                    claim_id=claim.claim_id,
                )
            )
        for receipt in stats_receipts:
            statistics = receipt.statistics
            assert statistics is not None  # enforced by the reachability gate
            if statistics.p_value is not None and any(
                value is None
                for value in (
                    statistics.effect_size,
                    statistics.ci_low,
                    statistics.ci_high,
                    statistics.sample_size,
                )
            ):
                violations.append(
                    GateViolation(
                        code="p_value_without_effect_ci_n",
                        message=(
                            f"receipt {receipt.receipt_id} reports a p-value "
                            "without effect size, CI and sample size."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
            if statistics.sequence_index is None:
                violations.append(
                    GateViolation(
                        code="sequence_index_missing",
                        message=(
                            f"receipt {receipt.receipt_id} has no registry "
                            "sequence_index."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
            if (
                confirmatory
                and TEST_PUBLISHABILITY.get(statistics.test_name)
                != "confirmatory_ready"
            ):
                violations.append(
                    GateViolation(
                        code="test_not_confirmatory_ready",
                        message=(
                            f"test {statistics.test_name!r} is not publishable "
                            "in the confirmatory lane."
                        ),
                        claim_id=claim.claim_id,
                        tool_call_id=receipt.tool_call_id,
                    )
                )
    return violations


def _state_gate(
    evidence: dict[str, _ClaimEvidence], run_witness: str
) -> list[GateViolation]:
    violations: list[GateViolation] = []
    seen: set[str] = set()
    for resolved in evidence.values():
        for receipt in resolved.receipts.values():
            if receipt.receipt_id in seen:
                continue
            seen.add(receipt.receipt_id)
            if receipt.data_state_witness != run_witness:
                violations.append(
                    GateViolation(
                        code="witness_mismatch",
                        message=(
                            f"receipt {receipt.receipt_id} was produced against a "
                            "different data state than this run's frozen witness."
                        ),
                    )
                )
    return violations


def _health_score(claim_count: int, violations: list[GateViolation]) -> float:
    """GroundEval Eq.1: (1 - v)^2 where v is the violated share of claims;
    bundle-level violations each count as one violated unit."""
    if claim_count <= 0:
        return 0.0
    affected = {v.claim_id for v in violations if v.claim_id is not None}
    bundle_level = sum(1 for v in violations if v.claim_id is None)
    rate = min(1.0, (len(affected) + bundle_level) / claim_count)
    return (1.0 - rate) ** 2

"""Deterministic E5 six-section report over reducer and gate outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from eda_platform.agents.exploration.workflow import ExplorationWorkflowState
from eda_platform.core.claim_gates import claim_bundle_digest
from eda_platform.core.claim_renderer import assert_rendered_numbers, numeric_pool_from_texts
from eda_platform.schemas.claims import split_evidence_ref
from eda_platform.schemas.insights import InsightRecord
from eda_platform.schemas.receipts import verify_receipt_digest

_EMPTY = "(none)"


class ExplorationRenderedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str
    rendered_insight_ids: tuple[str, ...]
    coverage_gaps: tuple[str, ...]


def render_exploration_report(
    state: ExplorationWorkflowState,
    *,
    run_metadata: Mapping[str, str],
    coverage_targets: Sequence[str],
    budget_summary: Mapping[str, object],
    stop_reason: str,
) -> ExplorationRenderedReport:
    """Render only gate-passed reducer records in the fixed E5 section order."""
    required = ("exploration_id", "policy_fingerprint", "witness")
    if any(not run_metadata.get(key, "").strip() for key in required):
        raise ValueError("exploration report metadata is incomplete.")
    if not stop_reason.strip():
        raise ValueError("exploration report requires a structured stop reason.")
    if any(not key or not _is_scalar(value) for key, value in budget_summary.items()):
        raise ValueError("budget summary requires non-empty keys and scalar values.")

    insights = tuple(sorted(state.insights.values(), key=lambda item: item.insight_id))
    for insight in insights:
        bundle = state.admitted_bundles.get(insight.claim_bundle_id)
        report = state.gate_reports.get(insight.claim_bundle_id)
        if bundle is None or report is None or not report.passed:
            raise ValueError(
                f"insight {insight.insight_id!r} is not backed by a passed claim bundle."
            )
        if report.claim_bundle_id != bundle.claim_bundle_id:
            raise ValueError("gate report does not match its admitted claim bundle.")
        if report.claim_bundle_digest != claim_bundle_digest(bundle):
            raise ValueError("gate report does not match the exact admitted claim body.")
        if report.run_witness != run_metadata["witness"]:
            raise ValueError("gate report was evaluated against a different run witness.")
        if bundle.hypothesis_id != insight.hypothesis_id:
            raise ValueError("insight hypothesis does not match its admitted claim bundle.")
        _validate_insight_proof(state, insight, bundle.referenced_receipt_ids(), report.run_witness)

    coverage_gaps = tuple(
        sorted(set(coverage_targets) - set(state.coverage_completed))
    )
    lines = ["# Exploration report", ""]
    lines.extend(f"- {key}: {run_metadata[key]}" for key in required)
    lines.extend(["", "## Supported insights", ""])
    lines.extend(_insight_lines(state, insights, statuses={"new", "reinforced"}))
    lines.extend(["", "## Refuted hypotheses", ""])
    lines.extend(_insight_lines(state, insights, statuses={"refuted"}))
    lines.extend(["", "## Inconclusive questions", ""])
    lines.extend(_insight_lines(state, insights, statuses={"inconclusive"}))
    lines.extend(["", "## Data and method limitations", ""])
    limitations = sorted(
        {
            limitation
            for insight in insights
            for limitation in insight.limitations
            if limitation.strip()
        }
    )
    lines.extend([f"- {item}" for item in limitations] or [_EMPTY])
    lines.extend(["", "## Coverage gaps / not explored", ""])
    lines.extend([f"- {item}" for item in coverage_gaps] or [_EMPTY])
    lines.extend(["", "## Cost and structured stop reason", ""])
    lines.append(f"- stop_reason: {stop_reason}")
    lines.extend(
        f"- {key}: {_scalar_text(value)}"
        for key, value in sorted(budget_summary.items())
    )
    markdown = "\n".join(lines) + "\n"

    evidence_texts = [*run_metadata.values(), stop_reason]
    evidence_texts.extend(_scalar_text(value) for value in budget_summary.values())
    evidence_texts.extend(coverage_targets)
    evidence_texts.extend(state.coverage_completed)
    for insight in insights:
        evidence_texts.extend(
            (
                insight.insight_id,
                insight.hypothesis_id,
                insight.claim_bundle_id,
                *insight.limitations,
            )
        )
        if insight.statement:
            evidence_texts.append(insight.statement)
        if insight.rationale:
            evidence_texts.append(insight.rationale)
        evidence_texts.append(str(len(insight.supporting_receipt_ids)))
        evidence_texts.append(str(len(insight.contradicting_receipt_ids)))
        stats_line = _key_statistics_line(state, insight)
        if stats_line is not None:
            evidence_texts.append(stats_line)
        bundle = state.admitted_bundles[insight.claim_bundle_id]
        for claim in bundle.claims:
            evidence_texts.extend((claim.claim_id, claim.claim_text))
        for proof in insight.proof:
            evidence_texts.append(proof.receipt_id)
            evidence_texts.extend(proof.fact_ids)
    offenders = assert_rendered_numbers(markdown, numeric_pool_from_texts(evidence_texts))
    if offenders:
        raise ValueError(
            "exploration report introduced numbers outside the evidence/budget pool: "
            f"{offenders}"
        )
    return ExplorationRenderedReport(
        markdown=markdown,
        rendered_insight_ids=tuple(item.insight_id for item in insights),
        coverage_gaps=coverage_gaps,
    )


def _insight_lines(
    state: ExplorationWorkflowState,
    insights: Sequence[InsightRecord],
    *,
    statuses: set[str],
) -> list[str]:
    """One compact human-readable block per insight; the full claim bundle
    stays in workflow-state.json for audit."""
    lines: list[str] = []
    for insight in insights:
        if insight.status not in statuses:
            continue
        statement = insight.statement or "(no statement recorded)"
        lines.append(f"- **{statement}** — {insight.status}/{insight.trust_level}")
        summary = (
            f"{len(insight.supporting_receipt_ids)} supporting, "
            f"{len(insight.contradicting_receipt_ids)} contradicting receipt(s)"
        )
        edges = ", ".join(
            f"{edge.receipt_id} ({edge.comparison})" for edge in insight.proof
        )
        lines.append(f"  - evidence: {summary}" + (f"; {edges}" if edges else ""))
        stats = _key_statistics_line(state, insight)
        if stats is not None:
            lines.append(f"  - {stats}")
        if insight.rationale and insight.rationale.strip():
            lines.append(f"  - why: {insight.rationale}")
    return lines or [_EMPTY]


def _key_statistics_line(
    state: ExplorationWorkflowState, insight: InsightRecord
) -> str | None:
    """Key numbers from up to two adjudicating receipts' statistics."""
    fragments: list[str] = []
    for receipt_id in (*insight.supporting_receipt_ids, *insight.contradicting_receipt_ids):
        receipt = state.committed_receipts.get(receipt_id)
        statistics = receipt.statistics if receipt is not None else None
        if statistics is None:
            continue
        parts = [
            f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in (
                ("p_value", statistics.p_value),
                ("effect_size", statistics.effect_size),
                ("sample_size", statistics.sample_size),
            )
            if value is not None
        ]
        if not parts:
            continue
        fragments.append(f"{receipt_id}: " + ", ".join(parts))
        if len(fragments) == 2:
            break
    if not fragments:
        return None
    return "key stats: " + "; ".join(fragments)


def _validate_insight_proof(
    state: ExplorationWorkflowState,
    insight: InsightRecord,
    bundle_receipt_ids: Sequence[str],
    run_witness: str,
) -> None:
    assigned = set(insight.supporting_receipt_ids) | set(
        insight.contradicting_receipt_ids
    )
    # Since R1 an admitted bundle may also carry claims from receipts that were
    # never adjudicated; those are not evidence for either side and carry no
    # proof edge. Evidence must still be drawn from the gated bundle.
    if not assigned <= set(bundle_receipt_ids):
        raise ValueError("insight evidence assignment does not match its claim bundle.")
    if set(insight.supporting_receipt_ids) & set(insight.contradicting_receipt_ids):
        raise ValueError("one receipt cannot both support and contradict an insight.")

    # Only the adjudicated receipts owe proof edges; the bundle's other claims
    # are exploratory context that no insight side rests on.
    expected_facts: dict[str, set[str]] = {}
    bundle = state.admitted_bundles[insight.claim_bundle_id]
    for claim in bundle.claims:
        for reference in (*claim.evidence_fact_ids, *claim.derivation_ids):
            receipt_id, fact_id = split_evidence_ref(reference)
            expected_facts.setdefault(receipt_id, set()).add(fact_id)
        for receipt_id in claim.statistics_receipt_ids:
            receipt = state.committed_receipts.get(receipt_id)
            if receipt is None:
                raise ValueError("insight cites a statistics receipt that is not committed.")
            expected_facts.setdefault(receipt_id, set()).update(
                fact.fact_id for fact in receipt.facts
            )

    actual_facts: dict[str, set[str]] = {}
    for edge in insight.proof:
        receipt = state.committed_receipts.get(edge.receipt_id)
        if receipt is None or not verify_receipt_digest(receipt):
            raise ValueError("insight proof cites an absent or invalid receipt.")
        if receipt.data_state_witness != run_witness:
            raise ValueError("insight proof receipt belongs to a different run witness.")
        expected_comparison = (
            "supports"
            if edge.receipt_id in insight.supporting_receipt_ids
            else "contradicts"
        )
        if edge.comparison != expected_comparison:
            raise ValueError("insight proof comparison conflicts with its evidence side.")
        actual_facts.setdefault(edge.receipt_id, set()).update(edge.fact_ids)
    proved_expectation = {
        receipt_id: facts
        for receipt_id, facts in expected_facts.items()
        if receipt_id in assigned
    }
    if actual_facts != proved_expectation:
        raise ValueError("insight proof nodes do not match the gated claim references.")


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _scalar_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False)

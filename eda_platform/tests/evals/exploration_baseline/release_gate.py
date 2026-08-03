"""Eval-0 adapter for the core E4a release-gate contract."""

from __future__ import annotations

from eda_platform.core.exploration_release_gate import (
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aReleaseCertificate,
    E4aReleaseGateClosedError,
    E4aReleaseReport,
    E4aTrialEvidence,
    E4aTrialUsage,
    evaluate_e4a_contract,
    evaluate_e4a_production_release,
    issue_e4a_release_certificate,
)

from .harness import ItemResult


def evaluate_e4a_release(
    *,
    baseline: list[ItemResult],
    treatment: list[ItemResult],
    hard_caps: E4aHardCaps,
    minimum_seeds: int = 5,
) -> E4aReleaseReport:
    """Evaluate scripted harness results as contract-only, never as a certificate."""
    return evaluate_e4a_contract(
        baseline=[_from_item_result(result) for result in baseline],
        treatment=[_from_item_result(result) for result in treatment],
        hard_caps=hard_caps,
        minimum_seeds=minimum_seeds,
    )


def _from_item_result(result: ItemResult) -> E4aTrialEvidence:
    """Normalize legacy harness output while marking its identities as non-production."""
    return E4aTrialEvidence(
        trial_id=f"{result.item_id}:{result.model}:{result.tier}:{result.seed}",
        item_id=result.item_id,
        bucket=result.bucket,
        model=result.model,
        provider="scripted" if result.model == "scripted" else "unknown",
        tier=result.tier,
        seed=result.seed,
        status=result.status,
        passed=result.passed,
        scores=result.scores,
        usage=E4aTrialUsage(
            llm_requests=result.usage.llm_requests,
            total_tokens=result.usage.total_tokens,
            estimated_cost_usd=result.usage.estimated_cost_usd,
            wall_clock_seconds=result.usage.wall_clock_seconds,
            tool_calls=result.usage.tool_calls,
            rows_scanned=result.usage.rows_scanned,
            cells_scanned=result.usage.cells_scanned,
        ),
        checker_version=result.checker_version,
        code_fingerprint="contract-only",
        tool_capability_digest=result.capability_catalog_version or "contract-only",
    )


__all__ = [
    "E4aEvidenceBindings",
    "E4aHardCaps",
    "E4aReleaseCertificate",
    "E4aReleaseGateClosedError",
    "E4aReleaseReport",
    "E4aTrialEvidence",
    "E4aTrialUsage",
    "evaluate_e4a_production_release",
    "evaluate_e4a_release",
    "issue_e4a_release_certificate",
]

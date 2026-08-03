"""Shared production-certificate fixture for E4b tests."""

from __future__ import annotations

from datetime import UTC, datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eda_platform.application.services.exploration_service import (
    ExplorationRuntimeIdentity,
)
from eda_platform.core.exploration_profiles import (
    EXPLORATION_PROFILE_VERSION,
    EXPLORATION_STATISTICAL_POLICY_VERSION,
)
from eda_platform.core.exploration_release_gate import (
    E4A_RELEASE_GATE_VERSION,
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aReleaseCertificate,
    E4aTrialEvidence,
    E4aTrialUsage,
    attest_e4a_test_trial_evidence,
    issue_e4a_test_release_certificate,
)
from eda_platform.core.ids import stable_hash

TEST_EVIDENCE_KEY_ID = "test-evaluator-v1"
TEST_RELEASE_KEY_ID = "test-release-v1"
TEST_EVIDENCE_SIGNING_KEY = bytes.fromhex("11" * 32)
TEST_RELEASE_SIGNING_KEY = bytes.fromhex("22" * 32)


def _public_key(private_key: bytes) -> bytes:
    return (
        Ed25519PrivateKey.from_private_bytes(private_key)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


TEST_EVIDENCE_PUBLIC_KEYS = {TEST_EVIDENCE_KEY_ID: _public_key(TEST_EVIDENCE_SIGNING_KEY)}
TEST_TRUSTED_RELEASE_PUBLIC_KEYS = {TEST_RELEASE_KEY_ID: _public_key(TEST_RELEASE_SIGNING_KEY)}

TEST_BINDINGS = E4aEvidenceBindings(
    checker_version="checker-v1",
    code_fingerprint="code-v1",
    tool_capability_digest="tools-v1",
    evidence_key_id=TEST_EVIDENCE_KEY_ID,
)
TEST_CAPS = E4aHardCaps(
    max_wall_seconds=1_800,
    max_llm_requests=36,
    max_total_tokens=375_000,
    max_cost_usd=15,
    max_tool_calls=72,
    max_rows_scanned=30_000_000,
    max_cells_scanned=1_500_000,
)


def runtime_identity(
    *,
    bindings: E4aEvidenceBindings = TEST_BINDINGS,
    hard_caps: E4aHardCaps = TEST_CAPS,
) -> ExplorationRuntimeIdentity:
    return ExplorationRuntimeIdentity(
        release_gate_version=E4A_RELEASE_GATE_VERSION,
        bindings=bindings,
        hard_caps=hard_caps,
        scoring_policy_version=EXPLORATION_PROFILE_VERSION,
        statistical_policy_version=EXPLORATION_STATISTICAL_POLICY_VERSION,
    )


TEST_RUNTIME_IDENTITY = runtime_identity()


def release_certificate(
    *,
    bindings: E4aEvidenceBindings = TEST_BINDINGS,
    hard_caps: E4aHardCaps = TEST_CAPS,
) -> E4aReleaseCertificate:
    treatment = [
        _trial(tier=tier, seed=seed, bindings=bindings)
        for tier in ("quick", "standard", "deep")
        for seed in range(1, 6)
    ]
    baseline = [
        _trial(
            tier="standard",
            seed=0,
            provider="baseline",
            bindings=bindings,
            trial_id="frozen-baseline-0",
            precision=0.9,
        )
    ]
    return issue_e4a_test_release_certificate(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        bindings=bindings,
        issued_at=datetime(2026, 8, 2, tzinfo=UTC),
        evidence_public_keys={
            bindings.evidence_key_id: TEST_EVIDENCE_PUBLIC_KEYS[TEST_EVIDENCE_KEY_ID]
        },
        release_signing_key=TEST_RELEASE_SIGNING_KEY,
        release_key_id=TEST_RELEASE_KEY_ID,
    )


def _trial(
    *,
    tier: str,
    seed: int,
    provider: str = "openai",
    bindings: E4aEvidenceBindings = TEST_BINDINGS,
    trial_id: str | None = None,
    precision: float = 1.0,
) -> E4aTrialEvidence:
    raw = E4aTrialEvidence(
        trial_id=trial_id or f"provider-run-{tier}-{seed}",
        item_id="planted-retail-v1",
        bucket="planted",
        model="gpt-5.6-terra",
        provider=provider,
        tier=tier,
        seed=seed,
        status="scored",
        passed=True,
        scores={
            "precision": precision,
            "recall": 0.25,
            "grounding_rate": 1.0,
            "fabricated_receipt_rate": 0.0,
            "region_difference_recall": 1.0,
            "missingness_mechanism_recall": 0.0,
            "spike_day_recall": 0.0,
            "spam_fixture_input_count": 124.0,
            "spam_fixture_canonical_groups": 1.0,
            "no_information_rounds": 2.0,
            "no_information_stopped": 1.0,
            "proof_reachability_rate": 1.0,
            "journal_provenance_rate": 1.0,
        },
        usage=E4aTrialUsage(
            llm_requests=4,
            total_tokens=4_000,
            estimated_cost_usd=0.2,
            wall_clock_seconds=2,
            tool_calls=8,
            rows_scanned=10_000,
            cells_scanned=100_000,
        ),
        checker_version=bindings.checker_version,
        code_fingerprint=bindings.code_fingerprint,
        tool_capability_digest=bindings.tool_capability_digest,
    )
    return attest_e4a_test_trial_evidence(
        raw,
        signing_key=TEST_EVIDENCE_SIGNING_KEY,
        key_id=bindings.evidence_key_id,
        source_manifest_digest=stable_hash(
            {
                "trial_id": raw.trial_id,
                "tier": tier,
                "seed": seed,
                "durable_trial_record": True,
            },
            length=64,
        ),
    )

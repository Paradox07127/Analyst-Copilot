"""Issue a local-only E4a release certificate so E4b can be run on this machine.

The certificate this writes is backed by SYNTHETIC trial evidence. It unlocks
the exploration API for local testing and proves nothing about exploration
quality -- a production release must come from
``drivers.exploration_evidence_issuer.issue_e4a_release_from_evidence_roots``
over real, attested trial runs.

The two identities that still bite are computed from the live code: the tool
capability digest is rebuilt from the actual read-only tool inventory (the
worker recomputes it and refuses to run on a mismatch), and the gate/scoring/
statistical policy versions come from this build.

Usage:
    uv run python scripts/issue_local_exploration_certificate.py
    # then export the two printed variables and restart the API
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eda_platform.agents.data_tools import DataToolContext, build_data_tools
from eda_platform.application.services.exploration_service import (
    EXPLORATION_RELEASE_CERTIFICATE_ENV,
    EXPLORATION_RELEASE_TRUSTED_KEYS_ENV,
)
from eda_platform.core.config import resolve_workspace_path
from eda_platform.core.exploration_profiles import (
    EXPLORATION_PROFILE_VERSION,
    EXPLORATION_STATISTICAL_POLICY_VERSION,
    build_read_only_exploration_toolset,
    exploration_budget_profile,
)
from eda_platform.core.exploration_release_gate import (
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aTrialEvidence,
    E4aTrialUsage,
    attest_e4a_test_trial_evidence,
    issue_e4a_test_release_certificate,
)
from eda_platform.core.exploration_tiers import ExplorationTier
from eda_platform.core.ids import stable_hash
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.core.stat_registry import StatTestRegistry
from eda_platform.drivers.exploration import exploration_tool_capability_digest
from eda_platform.tools.sql_runner import build_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS_PER_TIER = 5
_TIERS: tuple[ExplorationTier, ...] = ("quick", "standard", "deep")


def live_tool_capability_digest() -> str:
    """Rebuild the digest from the same allowlist the worker will construct."""
    context = DataToolContext(
        datasets=[],
        catalog=build_catalog([]),
        project_id="local-issuer",
        session_id="local-issuer",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
        stat_registry=StatTestRegistry(None),
    )
    registered = {tool.name: tool for tool in build_data_tools(context)}
    return exploration_tool_capability_digest(
        build_read_only_exploration_toolset(registered)
    )


def hard_caps_covering_every_tier() -> E4aHardCaps:
    """Take the ceiling of all three profiles; a smaller cap rejects that tier."""
    budgets = [exploration_budget_profile(tier) for tier in _TIERS]
    return E4aHardCaps(
        max_wall_seconds=max(float(b.llm.max_wall_seconds or 0) for b in budgets),
        max_llm_requests=max(int(b.llm.max_requests or 0) for b in budgets),
        max_total_tokens=max(int(b.llm.max_total_tokens or 0) for b in budgets),
        max_cost_usd=max(float(b.llm.max_cost_usd or 0) for b in budgets),
        max_tool_calls=max(int(b.max_successful_tool_calls) for b in budgets),
        max_rows_scanned=max(int(b.max_rows_scanned) for b in budgets),
        max_cells_scanned=max(int(b.max_result_cells) for b in budgets),
    )


def local_code_fingerprint(tool_digest: str) -> str:
    """Identity for the journal. Not a source-tree hash -- it covers the tool
    contracts and policy versions only, which is what the certificate binds."""
    return "xplcode_local_" + stable_hash(
        {
            "tool_capability_digest": tool_digest,
            "scoring_policy_version": EXPLORATION_PROFILE_VERSION,
            "statistical_policy_version": EXPLORATION_STATISTICAL_POLICY_VERSION,
        },
        length=32,
    )


def _synthetic_trial(
    *,
    tier: str,
    seed: int,
    provider: str,
    bindings: E4aEvidenceBindings,
    evidence_signing_key: bytes,
    trial_id: str,
    precision: float,
) -> E4aTrialEvidence:
    raw = E4aTrialEvidence(
        trial_id=trial_id,
        item_id="local-synthetic-v1",
        bucket="planted",
        model="local-synthetic",
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
            llm_requests=1,
            total_tokens=1_000,
            estimated_cost_usd=0.01,
            wall_clock_seconds=1,
            tool_calls=1,
            rows_scanned=1_000,
            cells_scanned=1_000,
        ),
        checker_version=bindings.checker_version,
        code_fingerprint=bindings.code_fingerprint,
        tool_capability_digest=bindings.tool_capability_digest,
    )
    return attest_e4a_test_trial_evidence(
        raw,
        signing_key=evidence_signing_key,
        key_id=bindings.evidence_key_id,
        source_manifest_digest=stable_hash(
            {"trial_id": trial_id, "tier": tier, "seed": seed, "synthetic": True},
            length=64,
        ),
    )


def _public_key_hex(private_key: bytes) -> str:
    return (
        Ed25519PrivateKey.from_private_bytes(private_key)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="certificate path (default: <workspace>/.exploration-release/local.json)",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "comma-separated provider names the certificate covers "
            "(default: every non-offline provider in the registry)"
        ),
    )
    parser.add_argument(
        "--release-key-id",
        default="local-release-v1",
        help="key id recorded in the certificate and in the trusted-keys variable",
    )
    parser.add_argument(
        "--signing-key",
        default=None,
        help="32-byte hex Ed25519 release signing key (default: freshly generated)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.providers:
        providers = tuple(
            sorted({item.strip().casefold() for item in args.providers.split(",") if item.strip()})
        )
    else:
        providers = tuple(
            sorted(
                provider.value
                for provider in LLMProvider
                if provider is not LLMProvider.OFFLINE
            )
        )
    if not providers:
        raise SystemExit("at least one provider must be covered by the certificate")

    release_signing_key = (
        bytes.fromhex(args.signing_key) if args.signing_key else secrets.token_bytes(32)
    )
    evidence_signing_key = secrets.token_bytes(32)
    tool_digest = live_tool_capability_digest()
    bindings = E4aEvidenceBindings(
        checker_version="local-synthetic",
        code_fingerprint=local_code_fingerprint(tool_digest),
        tool_capability_digest=tool_digest,
        evidence_key_id="local-evidence-v1",
    )
    caps = hard_caps_covering_every_tier()

    treatment = [
        _synthetic_trial(
            tier=tier,
            seed=seed,
            provider=provider,
            bindings=bindings,
            evidence_signing_key=evidence_signing_key,
            trial_id=f"local-{provider}-{tier}-{seed}",
            precision=1.0,
        )
        for provider in providers
        for tier in _TIERS
        for seed in range(1, TRIALS_PER_TIER + 1)
    ]
    baseline = [
        _synthetic_trial(
            tier="standard",
            seed=0,
            provider="baseline",
            bindings=bindings,
            evidence_signing_key=evidence_signing_key,
            trial_id="local-frozen-baseline-0",
            precision=0.9,
        )
    ]

    certificate = issue_e4a_test_release_certificate(
        baseline=baseline,
        treatment=treatment,
        hard_caps=caps,
        bindings=bindings,
        minimum_trials_per_tier=TRIALS_PER_TIER,
        issued_at=datetime.now(UTC),
        evidence_public_keys={
            bindings.evidence_key_id: bytes.fromhex(_public_key_hex(evidence_signing_key))
        },
        release_signing_key=release_signing_key,
        release_key_id=args.release_key_id,
    )

    out = (
        Path(args.out).expanduser()
        if args.out
        else resolve_workspace_path(None) / ".exploration-release" / "local.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(certificate.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.chmod(out, 0o600)

    trusted = f"{args.release_key_id}:{_public_key_hex(release_signing_key)}"
    print(f"wrote local exploration certificate: {out}")
    print(f"  providers:        {', '.join(certificate.providers)}")
    print(f"  tool digest:      {tool_digest}")
    print(f"  code fingerprint: {bindings.code_fingerprint}")
    print(f"  certificate:      {certificate.certificate_digest}")
    print()
    print("Evidence is SYNTHETIC. This unlocks local E4b; it certifies nothing.")
    print("Export both variables, then restart the API:")
    print()
    print(f'export {EXPLORATION_RELEASE_CERTIFICATE_ENV}="{out}"')
    print(f"export {EXPLORATION_RELEASE_TRUSTED_KEYS_ENV}='{trusted}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eda_platform.core.exploration_release_gate import (
    E4aEvidenceBindings,
    E4aReleaseCertificate,
    E4aReleaseGateClosedError,
    E4aTrialEvidence,
    E4aTrialUsage,
    attest_e4a_test_trial_evidence,
    evaluate_e4a_production_release,
    issue_e4a_test_release_certificate,
    verify_e4a_release_certificate,
)

from .harness import ItemResult, RunUsage
from .release_gate import E4aHardCaps, evaluate_e4a_release

CAPS = E4aHardCaps(
    max_wall_seconds=60,
    max_llm_requests=10,
    max_total_tokens=20_000,
    max_cost_usd=1,
    max_tool_calls=20,
    max_rows_scanned=100_000,
    max_cells_scanned=1_000_000,
)
PRODUCTION_BINDINGS = E4aEvidenceBindings(
    checker_version="checker-sha256:abc",
    code_fingerprint="code-sha256:def",
    tool_capability_digest="tools-sha256:ghi",
    evidence_key_id="test-evidence-key",
)
_EVIDENCE_PRIVATE = bytes.fromhex("11" * 32)
_EVIDENCE_PUBLIC = (
    Ed25519PrivateKey.from_private_bytes(_EVIDENCE_PRIVATE).public_key().public_bytes_raw()
)
_EVIDENCE_KEYS = {PRODUCTION_BINDINGS.evidence_key_id: _EVIDENCE_PUBLIC}
_RELEASE_PRIVATE = bytes.fromhex("22" * 32)
_RELEASE_KEY_ID = "test-release-key"
_RELEASE_PUBLIC = (
    Ed25519PrivateKey.from_private_bytes(_RELEASE_PRIVATE).public_key().public_bytes_raw()
)


def _result(seed: int, **scores: float) -> ItemResult:
    defaults = {
        "precision": 1.0,
        "recall": 0.25,
        "grounding_rate": 1.0,
        "fabricated_receipt_rate": 0.0,
        "region_difference_recall": 1.0,
        "missingness_mechanism_recall": 0.0,
        "spike_day_recall": 0.0,
    }
    defaults.update(scores)
    return ItemResult(
        item_id="planted_retail_v1",
        bucket="planted",
        suite="capability",
        model="scripted",
        tier="standard",
        seed=seed,
        status="scored",
        passed=True,
        scores=defaults,
        usage=RunUsage(
            llm_requests=4,
            total_tokens=4_000,
            estimated_cost_usd=0.2,
            wall_clock_seconds=2,
        ),
    )


def test_five_repeatable_grounded_seeds_pass_the_e4a_gate() -> None:
    baseline = [_result(1, precision=0.0, recall=0.0, region_difference_recall=0.0)]
    treatment = [_result(seed) for seed in range(1, 6)]

    report = evaluate_e4a_release(baseline=baseline, treatment=treatment, hard_caps=CAPS)

    assert report.passed
    assert report.mode == "contract"
    assert report.repeatable_target_structures == ("region_difference_recall",)


def test_one_lucky_seed_cannot_pass_as_repeatable_recall() -> None:
    treatment = [_result(seed) for seed in range(1, 6)]
    treatment[-1] = _result(5, recall=0.0, region_difference_recall=0.0)
    report = evaluate_e4a_release(baseline=[], treatment=treatment, hard_caps=CAPS)
    assert not report.passed
    assert any("every treatment trial" in item for item in report.violations)
    assert any("at least one target structure" in item for item in report.violations)


def test_grounding_precision_and_caps_fail_closed() -> None:
    treatment = [_result(seed) for seed in range(1, 6)]
    treatment[0] = _result(
        1,
        precision=0.1,
        grounding_rate=0.5,
        fabricated_receipt_rate=0.5,
    ).model_copy(
        update={"usage": RunUsage(llm_requests=11, total_tokens=20_001, wall_clock_seconds=61)}
    )
    report = evaluate_e4a_release(
        baseline=[_result(1, precision=0.9)],
        treatment=treatment,
        hard_caps=CAPS,
    )
    assert not report.passed
    assert any("precision regressed" in item for item in report.violations)
    assert any("grounded" in item for item in report.violations)
    assert any("fabricated" in item for item in report.violations)
    assert any("request cap" in item for item in report.violations)
    assert any("token cap" in item for item in report.violations)
    assert any("wall-clock cap" in item for item in report.violations)


def _production_trial(
    tier: str,
    seed: int,
    *,
    provider: str = "openai",
    usage: E4aTrialUsage | None = None,
    score_updates: dict[str, float] | None = None,
    bindings: E4aEvidenceBindings = PRODUCTION_BINDINGS,
) -> E4aTrialEvidence:
    scores = {
        "precision": 1.0,
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
    }
    scores.update(score_updates or {})
    trial = E4aTrialEvidence(
        trial_id=f"provider-run-{tier}-{seed}",
        item_id="planted_retail_v1",
        bucket="planted",
        model="gpt-5.6-terra",
        provider=provider,
        tier=tier,
        seed=seed,
        status="scored",
        passed=True,
        scores=scores,
        usage=usage
        or E4aTrialUsage(
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
    return _attest(trial)


def _attest(trial: E4aTrialEvidence) -> E4aTrialEvidence:
    return attest_e4a_test_trial_evidence(
        trial,
        signing_key=_EVIDENCE_PRIVATE,
        key_id=PRODUCTION_BINDINGS.evidence_key_id,
        source_manifest_digest=sha256(trial.trial_id.encode()).hexdigest(),
    )


def _production_baseline() -> list[E4aTrialEvidence]:
    return [
        _attest(
            _production_trial("standard", 0, score_updates={"precision": 0.9}).model_copy(
                update={"trial_id": "frozen-baseline-0", "provider": "baseline"}
            )
        )
    ]


def _production_treatment() -> list[E4aTrialEvidence]:
    return [
        _production_trial(tier, seed)
        for tier in ("quick", "standard", "deep")
        for seed in range(1, 6)
    ]


def test_production_evidence_issues_a_digest_bound_certificate() -> None:
    certificate = issue_e4a_test_release_certificate(
        baseline=_production_baseline(),
        treatment=_production_treatment(),
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        issued_at=datetime(2026, 8, 2, tzinfo=UTC),
        evidence_public_keys=_EVIDENCE_KEYS,
        release_signing_key=_RELEASE_PRIVATE,
        release_key_id=_RELEASE_KEY_ID,
    )

    assert certificate.report.passed
    assert certificate.report.mode == "production"
    assert certificate.providers == ("openai",)
    assert all(len(ids) == 5 for ids in certificate.report.trial_ids_by_tier.values())
    assert len(certificate.evidence_digest) == 64
    assert len(certificate.certificate_digest) == 64
    assert (
        verify_e4a_release_certificate(
            certificate,
            trusted_public_keys={_RELEASE_KEY_ID: _RELEASE_PUBLIC},
        )
        == certificate
    )

    tampered = certificate.model_dump(mode="json")
    tampered["providers"] = ["anthropic"]
    with pytest.raises(ValueError, match="certificate digest"):
        E4aReleaseCertificate.model_validate(tampered)


@pytest.mark.parametrize("provider", ["scripted", "unknown", "offline", "vendor-x"])
def test_non_production_or_unknown_provider_cannot_receive_certificate(provider: str) -> None:
    treatment = _production_treatment()
    treatment[0] = treatment[0].model_copy(update={"provider": provider})

    with pytest.raises(E4aReleaseGateClosedError, match="provider"):
        issue_e4a_test_release_certificate(
            baseline=_production_baseline(),
            treatment=treatment,
            hard_caps=CAPS,
            bindings=PRODUCTION_BINDINGS,
            evidence_public_keys=_EVIDENCE_KEYS,
            release_signing_key=_RELEASE_PRIVATE,
            release_key_id=_RELEASE_KEY_ID,
        )


def test_scripted_model_cannot_be_certified_under_a_real_provider_name() -> None:
    treatment = _production_treatment()
    treatment[0] = treatment[0].model_copy(update={"model": "scripted"})

    with pytest.raises(E4aReleaseGateClosedError, match="provider"):
        issue_e4a_test_release_certificate(
            baseline=_production_baseline(),
            treatment=treatment,
            hard_caps=CAPS,
            bindings=PRODUCTION_BINDINGS,
            evidence_public_keys=_EVIDENCE_KEYS,
            release_signing_key=_RELEASE_PRIVATE,
            release_key_id=_RELEASE_KEY_ID,
        )


def test_empty_baseline_and_incomplete_tier_coverage_fail_closed() -> None:
    report = evaluate_e4a_production_release(
        baseline=[],
        treatment=_production_treatment()[:-1],
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        evidence_public_keys=_EVIDENCE_KEYS,
    )

    assert not report.passed
    assert any("baseline evidence must be non-empty" in item for item in report.violations)
    assert any(
        "deep treatment requires at least 5 unique trials" in item for item in report.violations
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("llm_requests", "unknown request usage"),
        ("total_tokens", "unknown token usage"),
        ("estimated_cost_usd", "unknown cost usage"),
        ("wall_clock_seconds", "unknown wall-clock usage"),
        ("tool_calls", "unknown tool-call usage"),
        ("rows_scanned", "unknown rows-scanned usage"),
        ("cells_scanned", "unknown cells-scanned usage"),
    ],
)
def test_unknown_usage_dimensions_fail_closed(field: str, message: str) -> None:
    treatment = _production_treatment()
    unknown_usage = treatment[0].usage.model_copy(update={field: None})
    treatment[0] = treatment[0].model_copy(update={"usage": unknown_usage})

    report = evaluate_e4a_production_release(
        baseline=_production_baseline(),
        treatment=treatment,
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        evidence_public_keys=_EVIDENCE_KEYS,
    )

    assert not report.passed
    assert any(message in item for item in report.violations)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("llm_requests", 11, "request cap"),
        ("total_tokens", 20_001, "token cap"),
        ("estimated_cost_usd", 1.01, "cost cap"),
        ("wall_clock_seconds", 60.01, "wall-clock cap"),
        ("tool_calls", 21, "tool-call cap"),
        ("rows_scanned", 100_001, "rows-scanned cap"),
        ("cells_scanned", 1_000_001, "cells-scanned cap"),
    ],
)
def test_every_hard_cap_dimension_is_enforced(field: str, value: int | float, message: str) -> None:
    treatment = _production_treatment()
    excessive_usage = treatment[0].usage.model_copy(update={field: value})
    treatment[0] = treatment[0].model_copy(update={"usage": excessive_usage})

    report = evaluate_e4a_production_release(
        baseline=_production_baseline(),
        treatment=treatment,
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        evidence_public_keys=_EVIDENCE_KEYS,
    )

    assert not report.passed
    assert any(message in item for item in report.violations)


@pytest.mark.parametrize(
    "metric",
    [
        "spam_fixture_input_count",
        "spam_fixture_canonical_groups",
        "no_information_rounds",
        "no_information_stopped",
        "proof_reachability_rate",
        "journal_provenance_rate",
    ],
)
def test_governance_metrics_are_mandatory(metric: str) -> None:
    treatment = _production_treatment()
    scores = dict(treatment[0].scores)
    scores.pop(metric)
    treatment[0] = treatment[0].model_copy(update={"scores": scores})

    report = evaluate_e4a_production_release(
        baseline=_production_baseline(),
        treatment=treatment,
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        evidence_public_keys=_EVIDENCE_KEYS,
    )

    assert not report.passed
    assert any(metric in item for item in report.violations)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("checker_version", "checker version"),
        ("code_fingerprint", "code fingerprint"),
        ("tool_capability_digest", "tool capability digest"),
    ],
)
def test_certificate_bindings_reject_mixed_evidence(field: str, message: str) -> None:
    treatment = _production_treatment()
    treatment[0] = treatment[0].model_copy(update={field: "different"})

    report = evaluate_e4a_production_release(
        baseline=_production_baseline(),
        treatment=treatment,
        hard_caps=CAPS,
        bindings=PRODUCTION_BINDINGS,
        evidence_public_keys=_EVIDENCE_KEYS,
    )

    assert not report.passed
    assert any(message in item for item in report.violations)

"""Fail-closed E4a exploration release evidence and certificate issuance.

The production certificate deliberately consumes a stricter evidence model than
the Eval-0 harness.  Harness results are useful for deterministic contract tests,
but they do not carry an authenticated provider or distinguish unknown usage from
a measured zero.  Treating those defaults as production evidence would make the
gate fail open.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from datetime import UTC, datetime
from hmac import compare_digest
from types import MappingProxyType
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eda_platform.core.ids import stable_hash
from eda_platform.core.provider_registry import LLMProvider

E4A_RELEASE_GATE_VERSION = "e4a-release-gate-v3"
E4A_EVIDENCE_ATTESTATION_VERSION = "e4a-evidence-attestation-v1"
EXPLORATION_TIERS = ("quick", "standard", "deep")

# Production images must pin operator-approved issuer keys here (or inject an
# equally immutable deployment-owned mapping). The default is deliberately
# empty: a workspace file or certificate may never nominate its own trust root.
TRUSTED_E4A_RELEASE_PUBLIC_KEYS: Mapping[str, bytes] = MappingProxyType({})

_TARGET_STRUCTURE_METRICS = (
    "region_difference_recall",
    "missingness_mechanism_recall",
    "spike_day_recall",
)
_GOVERNANCE_METRICS: tuple[tuple[str, float], ...] = (
    ("spam_fixture_input_count", 124.0),
    ("spam_fixture_canonical_groups", 1.0),
    ("no_information_rounds", 2.0),
    ("no_information_stopped", 1.0),
    ("proof_reachability_rate", 1.0),
    ("journal_provenance_rate", 1.0),
)
_PRODUCTION_PROVIDERS = frozenset(
    provider.value for provider in LLMProvider if provider is not LLMProvider.OFFLINE
)


class E4aHardCaps(BaseModel):
    """Per-trial limits. Every dimension is mandatory for production issuance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_wall_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_llm_requests: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    max_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_tool_calls: int = Field(gt=0)
    max_rows_scanned: int = Field(gt=0)
    max_cells_scanned: int = Field(gt=0)


class E4aTrialUsage(BaseModel):
    """Measured usage; ``None`` means unknown and therefore blocks a release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_requests: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    wall_clock_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tool_calls: int | None = Field(default=None, ge=0)
    rows_scanned: int | None = Field(default=None, ge=0)
    cells_scanned: int | None = Field(default=None, ge=0)


class E4aEvidenceBindings(BaseModel):
    """Immutable implementation identities covered by a certificate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checker_version: str = Field(min_length=1)
    code_fingerprint: str = Field(min_length=1)
    tool_capability_digest: str = Field(min_length=1)
    evidence_key_id: str = Field(min_length=1)


class E4aTrialEvidence(BaseModel):
    """Normalized evidence for either a baseline or a treatment trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    seed: int
    status: str = Field(min_length=1)
    passed: bool | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    usage: E4aTrialUsage = Field(default_factory=E4aTrialUsage)
    checker_version: str = Field(min_length=1)
    code_fingerprint: str = Field(min_length=1)
    tool_capability_digest: str = Field(min_length=1)
    source_manifest_digest: str | None = Field(default=None, min_length=64, max_length=64)
    provenance_key_id: str | None = Field(default=None, min_length=1)
    provenance_signature: str | None = Field(default=None, min_length=128, max_length=128)

    @field_validator("scores")
    @classmethod
    def _scores_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(score) for score in value.values()):
            raise ValueError("all E4a scores must be finite")
        return value


class E4aReleaseReport(BaseModel):
    """Auditable decision returned by contract and production evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["contract", "production"]
    passed: bool
    violations: tuple[str, ...] = ()
    planted_seeds: tuple[int, ...] = ()
    mean_baseline_precision: float = 0.0
    mean_treatment_precision: float = 0.0
    mean_treatment_recall: float = 0.0
    repeatable_target_structures: tuple[str, ...] = ()
    trial_ids_by_tier: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class E4aReleaseCertificate(BaseModel):
    """A self-checking certificate issued only after the production gate passes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_version: Literal["e4a-release-gate-v3"] = E4A_RELEASE_GATE_VERSION
    issued_at: datetime
    bindings: E4aEvidenceBindings
    hard_caps: E4aHardCaps
    report: E4aReleaseReport
    providers: tuple[str, ...]
    evidence_digest: str = Field(min_length=64, max_length=64)
    release_key_id: str = Field(min_length=1)
    certificate_digest: str = Field(min_length=64, max_length=64)
    certificate_signature: str = Field(min_length=128, max_length=128)

    @model_validator(mode="after")
    def _digest_must_match_payload(self) -> E4aReleaseCertificate:
        expected = _certificate_digest(self)
        if not compare_digest(self.certificate_digest, expected):
            raise ValueError("certificate digest does not match its payload")
        if not self.report.passed or self.report.mode != "production":
            raise ValueError("a certificate requires a passing production report")
        return self


class E4aReleaseGateClosedError(RuntimeError):
    """Raised when production evidence cannot receive a certificate."""

    def __init__(self, report: E4aReleaseReport) -> None:
        self.report = report
        detail = "; ".join(report.violations) or "production release gate did not pass"
        super().__init__(detail)


def evaluate_e4a_contract(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    hard_caps: E4aHardCaps,
    minimum_seeds: int = 5,
) -> E4aReleaseReport:
    """Evaluate deterministic harness evidence without making it certifiable."""
    return _evaluate(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        minimum_trials=minimum_seeds,
        production=False,
        bindings=None,
    )


def evaluate_e4a_production_release(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    hard_caps: E4aHardCaps,
    bindings: E4aEvidenceBindings,
    minimum_trials_per_tier: int = 5,
    evidence_public_keys: Mapping[str, bytes] | None = None,
) -> E4aReleaseReport:
    """Evaluate all certificate requirements and return every gate violation."""
    if minimum_trials_per_tier <= 0:
        raise ValueError("minimum_trials_per_tier must be positive")
    return _evaluate(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        minimum_trials=minimum_trials_per_tier,
        production=True,
        bindings=bindings,
        evidence_public_keys=evidence_public_keys,
    )


def issue_e4a_release_certificate(
    **_unsafe_self_reported_evidence: object,
) -> E4aReleaseCertificate:
    """Reject the retired self-reported production issuance surface.

    Production certificates are issued only by
    ``drivers.exploration_evidence_issuer.issue_e4a_release_from_evidence_roots``.
    Keeping this name as a rejecting shim prevents an older caller from silently
    treating caller-built scores, usage, or a caller-nominated trust root as
    authenticated evidence.
    """
    raise RuntimeError(
        "raw E4aTrialEvidence cannot receive a production certificate; "
        "issue from verified evidence roots"
    )


def issue_e4a_test_release_certificate(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    hard_caps: E4aHardCaps,
    bindings: E4aEvidenceBindings,
    minimum_trials_per_tier: int = 5,
    issued_at: datetime | None = None,
    evidence_public_keys: Mapping[str, bytes] | None = None,
    release_signing_key: bytes | None = None,
    release_key_id: str | None = None,
) -> E4aReleaseCertificate:
    """Build a certificate from synthetic trials for isolated tests only.

    This helper is intentionally named and documented as a test seam. Product
    code must use the evidence-root issuer, whose inputs contain no caller-made
    scores or usage totals.
    """
    return _issue_e4a_release_certificate_from_verified_trials(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        bindings=bindings,
        minimum_trials_per_tier=minimum_trials_per_tier,
        issued_at=issued_at,
        evidence_public_keys=evidence_public_keys,
        release_signing_key=release_signing_key,
        release_key_id=release_key_id,
    )


def _issue_e4a_release_certificate_from_verified_trials(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    hard_caps: E4aHardCaps,
    bindings: E4aEvidenceBindings,
    minimum_trials_per_tier: int = 5,
    issued_at: datetime | None = None,
    evidence_public_keys: Mapping[str, bytes] | None = None,
    release_signing_key: bytes | None = None,
    release_key_id: str | None = None,
) -> E4aReleaseCertificate:
    """Internal sink for evidence already verified by the privileged issuer."""
    report = evaluate_e4a_production_release(
        baseline=baseline,
        treatment=treatment,
        hard_caps=hard_caps,
        bindings=bindings,
        minimum_trials_per_tier=minimum_trials_per_tier,
        evidence_public_keys=evidence_public_keys,
    )
    signing_violation = release_signing_key is None or not (release_key_id or "").strip()
    if signing_violation:
        report = report.model_copy(
            update={
                "passed": False,
                "violations": (
                    *report.violations,
                    "release signing key and pinned key id are required",
                ),
            }
        )
    if not report.passed:
        raise E4aReleaseGateClosedError(report)

    planted_baseline = _planted_scored(baseline)
    planted_treatment = _planted_scored(treatment)
    evidence_digest = stable_hash(
        {
            "baseline": _canonical_trials(planted_baseline),
            "treatment": _canonical_trials(planted_treatment),
        },
        length=64,
    )
    payload = {
        "gate_version": E4A_RELEASE_GATE_VERSION,
        "issued_at": issued_at or datetime.now(UTC),
        "bindings": bindings,
        "hard_caps": hard_caps,
        "report": report,
        "providers": tuple(sorted({run.provider.casefold() for run in planted_treatment})),
        "evidence_digest": evidence_digest,
        "release_key_id": release_key_id,
    }
    draft = E4aReleaseCertificate.model_construct(
        **payload,
        certificate_digest="",
        certificate_signature="",
    )
    certificate_digest = _certificate_digest(draft)
    assert release_signing_key is not None and release_key_id is not None
    canonical_payload = draft.model_dump(
        mode="json", exclude={"certificate_digest", "certificate_signature"}
    )
    signature = Ed25519PrivateKey.from_private_bytes(release_signing_key).sign(
        _certificate_signature_payload(canonical_payload, certificate_digest)
    )
    return E4aReleaseCertificate(
        **payload,
        certificate_digest=certificate_digest,
        certificate_signature=signature.hex(),
    )


def _evaluate(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    hard_caps: E4aHardCaps,
    minimum_trials: int,
    production: bool,
    bindings: E4aEvidenceBindings | None,
    evidence_public_keys: Mapping[str, bytes] | None = None,
) -> E4aReleaseReport:
    base = _planted_scored(baseline)
    runs = _planted_scored(treatment)
    violations: list[str] = []

    if not base:
        violations.append("frozen planted baseline evidence must be non-empty")
    if any("precision" not in run.scores for run in base):
        violations.append("every planted baseline trial must report precision")

    if production:
        assert bindings is not None
        _check_production_evidence(
            baseline=baseline,
            treatment=treatment,
            runs=runs,
            bindings=bindings,
            minimum_trials_per_tier=minimum_trials,
            violations=violations,
            evidence_public_keys=evidence_public_keys,
        )
    else:
        seeds = {run.seed for run in runs}
        if len(seeds) < minimum_trials:
            violations.append(f"planted treatment requires at least {minimum_trials} seeds")
        if len(seeds) != len(runs):
            violations.append("planted treatment must contain exactly one scored run per seed")

    baseline_precision = _mean_score(base, "precision")
    treatment_precision = _mean_score(runs, "precision")
    treatment_recall = _mean_score(runs, "recall")
    if not runs or any(run.scores.get("recall", 0.0) <= 0 for run in runs):
        violations.append("planted recall must be greater than zero in every treatment trial")
    if any("precision" not in run.scores for run in runs):
        violations.append("every planted treatment trial must report precision")
    if treatment_precision < baseline_precision:
        violations.append("mean planted precision regressed from the frozen baseline")
    if any(run.scores.get("grounding_rate") != 1.0 for run in runs):
        violations.append("every published treatment claim must be grounded")
    if any(run.scores.get("fabricated_receipt_rate") != 0.0 for run in runs):
        violations.append("fabricated receipt rate must remain zero")

    repeatable = tuple(
        name
        for name in _TARGET_STRUCTURE_METRICS
        if runs and all(run.scores.get(name) == 1.0 for run in runs)
    )
    if not repeatable:
        violations.append(
            "at least one target structure must be recalled in every trial: "
            "region difference, missingness mechanism, or spike day"
        )

    if production:
        _check_governance_metrics(runs, violations)
    _check_usage(runs, hard_caps, production=production, violations=violations)

    tier_trials = {
        tier: tuple(sorted({run.trial_id for run in runs if run.tier == tier}))
        for tier in EXPLORATION_TIERS
    }
    return E4aReleaseReport(
        mode="production" if production else "contract",
        passed=not violations,
        violations=tuple(dict.fromkeys(violations)),
        planted_seeds=tuple(sorted({run.seed for run in runs})),
        mean_baseline_precision=baseline_precision,
        mean_treatment_precision=treatment_precision,
        mean_treatment_recall=treatment_recall,
        repeatable_target_structures=repeatable,
        trial_ids_by_tier=tier_trials,
    )


def _check_production_evidence(
    *,
    baseline: list[E4aTrialEvidence],
    treatment: list[E4aTrialEvidence],
    runs: list[E4aTrialEvidence],
    bindings: E4aEvidenceBindings,
    minimum_trials_per_tier: int,
    violations: list[str],
    evidence_public_keys: Mapping[str, bytes] | None,
) -> None:
    if not evidence_public_keys:
        violations.append("trusted evidence issuer public keys are required")
    for run in (*baseline, *treatment):
        if run.bucket != "planted":
            continue
        if run.provenance_key_id != bindings.evidence_key_id:
            violations.append(
                f"trial {run.trial_id} evidence key id does not match certificate bindings"
            )
            continue
        public_key = (evidence_public_keys or {}).get(bindings.evidence_key_id)
        if public_key is None or not verify_e4a_trial_evidence(run, public_key):
            violations.append(f"trial {run.trial_id} lacks a valid trusted provenance attestation")
    unscored = [
        run.trial_id
        for run in treatment
        if run.bucket == "planted" and (run.status != "scored" or run.passed is not True)
    ]
    if unscored:
        violations.append("every planted treatment trial must be scored and passing")

    for tier in EXPLORATION_TIERS:
        tier_runs = [run for run in runs if run.tier == tier]
        ids = [run.trial_id for run in tier_runs]
        if len(set(ids)) < minimum_trials_per_tier:
            violations.append(
                f"{tier} treatment requires at least {minimum_trials_per_tier} unique trials"
            )
        if len(set(ids)) != len(ids):
            violations.append(f"{tier} treatment contains duplicate trial ids")
    unknown_tiers = sorted({run.tier for run in runs} - set(EXPLORATION_TIERS))
    if unknown_tiers:
        unknown = ", ".join(unknown_tiers)
        violations.append(f"unknown treatment tiers are not certifiable: {unknown}")

    for run in runs:
        provider = run.provider.casefold()
        if provider not in _PRODUCTION_PROVIDERS or run.model.casefold() == "scripted":
            violations.append(
                f"trial {run.trial_id} has non-production or unknown provider {run.provider!r}"
            )
        if run.checker_version != bindings.checker_version:
            violations.append(f"trial {run.trial_id} checker version does not match certificate")
        if run.code_fingerprint != bindings.code_fingerprint:
            violations.append(f"trial {run.trial_id} code fingerprint does not match certificate")
        if run.tool_capability_digest != bindings.tool_capability_digest:
            violations.append(
                f"trial {run.trial_id} tool capability digest does not match certificate"
            )

    for run in baseline:
        if run.bucket == "planted" and run.checker_version != bindings.checker_version:
            violations.append(
                f"baseline trial {run.trial_id} checker version does not match certificate"
            )


def _check_governance_metrics(runs: list[E4aTrialEvidence], violations: list[str]) -> None:
    for metric, expected in _GOVERNANCE_METRICS:
        failing = [run.trial_id for run in runs if run.scores.get(metric) != expected]
        if failing:
            violations.append(
                f"governance metric {metric} must equal {expected:g} in every treatment trial"
            )


def _check_usage(
    runs: list[E4aTrialEvidence],
    caps: E4aHardCaps,
    *,
    production: bool,
    violations: list[str],
) -> None:
    dimensions = (
        ("wall_clock_seconds", "wall-clock", caps.max_wall_seconds),
        ("llm_requests", "request", caps.max_llm_requests),
        ("total_tokens", "token", caps.max_total_tokens),
        ("estimated_cost_usd", "cost", caps.max_cost_usd),
        ("tool_calls", "tool-call", caps.max_tool_calls),
        ("rows_scanned", "rows-scanned", caps.max_rows_scanned),
        ("cells_scanned", "cells-scanned", caps.max_cells_scanned),
    )
    for run in runs:
        for field, label, cap in dimensions:
            value = getattr(run.usage, field)
            if value is None:
                if production:
                    violations.append(f"trial {run.trial_id} has unknown {label} usage")
                continue
            if value > cap:
                violations.append(f"trial {run.trial_id} exceeded {label} cap")


def _planted_scored(results: list[E4aTrialEvidence]) -> list[E4aTrialEvidence]:
    return [
        result for result in results if result.bucket == "planted" and result.status == "scored"
    ]


def _mean_score(results: list[E4aTrialEvidence], name: str) -> float:
    values = [result.scores[name] for result in results if name in result.scores]
    return round(statistics.fmean(values), 6) if values else 0.0


def _canonical_trials(results: list[E4aTrialEvidence]) -> list[dict[str, object]]:
    return [
        run.model_dump(mode="json")
        for run in sorted(results, key=lambda item: (item.tier, item.trial_id))
    ]


def _certificate_digest(certificate: E4aReleaseCertificate) -> str:
    return stable_hash(
        certificate.model_dump(
            mode="json",
            exclude={"certificate_digest", "certificate_signature"},
        ),
        length=64,
    )


def attest_e4a_trial_evidence(
    *_unsafe_self_reported_evidence: object,
    **_unsafe_attestation_fields: object,
) -> E4aTrialEvidence:
    """Reject the retired public attestation surface for self-reported trials."""
    raise RuntimeError(
        "raw E4aTrialEvidence cannot be attested as production evidence; "
        "attest only a verified evidence-root projection"
    )


def attest_e4a_test_trial_evidence(
    trial: E4aTrialEvidence,
    *,
    signing_key: bytes,
    key_id: str,
    source_manifest_digest: str,
) -> E4aTrialEvidence:
    """Sign a synthetic trial for isolated certificate/verifier tests only."""
    return _attest_e4a_verified_trial_evidence(
        trial,
        signing_key=signing_key,
        key_id=key_id,
        source_manifest_digest=source_manifest_digest,
    )


def _attest_e4a_verified_trial_evidence(
    trial: E4aTrialEvidence,
    *,
    signing_key: bytes,
    key_id: str,
    source_manifest_digest: str,
) -> E4aTrialEvidence:
    """Internal signer used only after root verification or by the test seam."""
    unsigned = trial.model_copy(
        update={
            "source_manifest_digest": source_manifest_digest,
            "provenance_key_id": key_id,
            "provenance_signature": None,
        }
    )
    signature = Ed25519PrivateKey.from_private_bytes(signing_key).sign(
        _trial_signature_payload(unsigned)
    )
    return unsigned.model_copy(update={"provenance_signature": signature.hex()})


def verify_e4a_trial_evidence(trial: E4aTrialEvidence, public_key: bytes) -> bool:
    if (
        trial.source_manifest_digest is None
        or trial.provenance_key_id is None
        or trial.provenance_signature is None
    ):
        return False
    try:
        signature = bytes.fromhex(trial.provenance_signature)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            _trial_signature_payload(trial),
        )
    except (ValueError, InvalidSignature):
        return False
    return True


def verify_e4a_release_certificate(
    certificate: E4aReleaseCertificate,
    *,
    trusted_public_keys: Mapping[str, bytes] = TRUSTED_E4A_RELEASE_PUBLIC_KEYS,
) -> E4aReleaseCertificate:
    """Authenticate a certificate against an external, pinned trust root."""
    validated = E4aReleaseCertificate.model_validate(certificate.model_dump(mode="json"))
    public_key = trusted_public_keys.get(validated.release_key_id)
    if public_key is None:
        raise ValueError("release certificate key id is not trusted")
    payload = validated.model_dump(
        mode="json", exclude={"certificate_digest", "certificate_signature"}
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(validated.certificate_signature),
            _certificate_signature_payload(payload, validated.certificate_digest),
        )
    except (ValueError, InvalidSignature) as exc:
        raise ValueError("release certificate signature is invalid") from exc
    return validated


def _trial_signature_payload(trial: E4aTrialEvidence) -> bytes:
    return _canonical_bytes(
        {
            "attestation_version": E4A_EVIDENCE_ATTESTATION_VERSION,
            "trial": trial.model_dump(mode="json", exclude={"provenance_signature"}),
        }
    )


def _certificate_signature_payload(payload: Mapping[str, object], certificate_digest: str) -> bytes:
    return _canonical_bytes(
        {"certificate": dict(payload), "certificate_digest": certificate_digest}
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

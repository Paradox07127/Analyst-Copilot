"""E4b exploration API control plane over the authoritative JSONL journal."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from eda_platform.application.dto import JobStatus
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.job_service import JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.exploration_budget import apply_budget_increase
from eda_platform.core.exploration_journal import (
    ExplorationPolicyIntegrityError,
    JsonlExplorationJournal,
    assert_policy_sealed,
)
from eda_platform.core.exploration_profiles import (
    EXPLORATION_PROFILE_VERSION,
    EXPLORATION_STATISTICAL_POLICY_VERSION,
    build_exploration_policy,
)
from eda_platform.core.exploration_release_gate import (
    E4A_RELEASE_GATE_VERSION,
    TRUSTED_E4A_RELEASE_PUBLIC_KEYS,
    E4aEvidenceBindings,
    E4aHardCaps,
    E4aReleaseCertificate,
    verify_e4a_release_certificate,
)
from eda_platform.core.exploration_shadow_store import (
    ShadowExplorationStore,
    shadow_run_root,
    validate_shadow_run_path,
)
from eda_platform.core.exploration_tiers import ExplorationTier
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.llm_ledger import (
    budget_policy_fingerprint,
    restore_run_budget_state,
)
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.session_loader import load_run
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.claims import ClaimBundle
from eda_platform.schemas.exploration import BudgetAmendedEvent, ExplorationPolicy
from eda_platform.schemas.exploration_api import (
    ExplorationBudgetExtended,
    ExplorationBudgetView,
    ExplorationCostRange,
    ExplorationEventView,
    ExplorationEvidenceView,
    ExplorationFactView,
    ExplorationHypothesisView,
    ExplorationInsightView,
    ExplorationJobView,
    ExplorationPrepared,
    ExplorationReportView,
    ExplorationRunMetadata,
    ExplorationStarted,
    ExplorationStatisticsView,
    ExplorationView,
)
from eda_platform.schemas.exploration_budget import (
    BudgetAmendment,
    BudgetCapIncrease,
    ExplorationBudgetPolicy,
)
from eda_platform.schemas.insights import InsightRecord
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    data_state_witness_digest,
    load_verified_receipt,
)
from eda_platform.schemas.sessions import TraceEvent

APPROVAL_KIND_EXPLORATION_START = "exploration_start"
EXPLORATION_JOB_KIND = "exploration_run"
EXPLORATION_SESSION_PREFIX = "explsess_"
EXPLORATION_RELEASE_CERTIFICATE_ENV = "EDA_EXPLORATION_RELEASE_CERTIFICATE_PATH"
EXPLORATION_RELEASE_TRUSTED_KEYS_ENV = "EDA_EXPLORATION_RELEASE_TRUSTED_KEYS"
EXPLORATION_EVENTS_PAGE_LIMIT = 500
_ED25519_PUBLIC_KEY_BYTES = 32
_MAX_CERTIFICATE_BYTES = 2 << 20
_AMENDMENT_APPROVER = "system:e4b-api"


@dataclass(frozen=True, slots=True)
class ExplorationRuntimeIdentity:
    """Build-pinned identities that a signed release must match exactly."""

    release_gate_version: str
    bindings: E4aEvidenceBindings
    hard_caps: E4aHardCaps
    scoring_policy_version: str
    statistical_policy_version: str


# Production packaging must replace this with the image's pinned identity. A
# certificate or workspace file can never nominate its own trusted runtime.
TRUSTED_EXPLORATION_RUNTIME_IDENTITY: ExplorationRuntimeIdentity | None = None


class ExplorationServiceError(Exception):
    pass


class ExplorationReleaseUnavailableError(ExplorationServiceError):
    """The operator has not installed a valid production release certificate."""


class ExplorationNotFoundError(ExplorationServiceError):
    def __init__(self, exploration_id: str) -> None:
        super().__init__(f"Exploration not found: {exploration_id}")
        self.exploration_id = exploration_id


class ExplorationConflictError(ExplorationServiceError):
    pass


class ExplorationValidationError(ExplorationServiceError):
    pass


class ExplorationSourceChangedError(ExplorationServiceError):
    def __init__(self, exploration_id: str) -> None:
        super().__init__(
            f"Exploration {exploration_id} cannot resume because its data-state "
            "witness changed. The run was stopped fail closed."
        )
        self.exploration_id = exploration_id


@dataclass(frozen=True, slots=True)
class ExplorationSourceSnapshot:
    project_id: str
    dataset_ids: tuple[str, ...]
    data_state_witness: str


@dataclass(frozen=True, slots=True)
class ExplorationEventsPage:
    events: tuple[ExplorationEventView, ...]
    status: str
    cursor: int
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _ProductProjection:
    current_hypothesis: ExplorationHypothesisView | None = None
    evidence: tuple[ExplorationEvidenceView, ...] = ()
    insights: tuple[ExplorationInsightView, ...] = ()
    limitations: tuple[str, ...] = ()
    coverage_targets: tuple[str, ...] = ()
    coverage_completed: tuple[str, ...] = ()
    coverage_unexplored: tuple[str, ...] = ()


SourceSnapshotResolver = Callable[[str, tuple[str, ...]], ExplorationSourceSnapshot]

_WorkflowProjectionParts = tuple[
    tuple[EvidenceReceipt, ...],
    dict[str, ClaimBundle],
    tuple[InsightRecord, ...],
    tuple[str, ...],
]


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class ExplorationReleaseTrust:
    """What one composition root is allowed to trust for E4b, resolved once."""

    certificate: E4aReleaseCertificate | None
    public_keys: Mapping[str, bytes]
    runtime_identity: ExplorationRuntimeIdentity | None


def operator_pinned_release_public_keys() -> Mapping[str, bytes]:
    """Read issuer keys an operator pinned in the environment.

    The shipped map stays empty on purpose: neither a workspace file nor the
    certificate may nominate its own trust root. The process environment is a
    different class of input -- it is owned by whoever starts the server, the
    same authority that supplies the certificate path -- so a local operator can
    open E4b without patching the production constant. A malformed value raises
    instead of falling back to the empty map: a typo in a trust root must be
    loud, never a silent downgrade to "no trust".
    """
    raw = os.environ.get(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, "").strip()
    if not raw:
        return TRUSTED_E4A_RELEASE_PUBLIC_KEYS
    keys: dict[str, bytes] = {}
    for item in raw.split(","):
        key_id, separator, hex_key = item.strip().partition(":")
        if not separator or not key_id.strip():
            raise ValueError(
                f"{EXPLORATION_RELEASE_TRUSTED_KEYS_ENV} entries must be "
                "'<key_id>:<hex_public_key>'"
            )
        try:
            public_key = bytes.fromhex(hex_key.strip())
        except ValueError as exc:
            raise ValueError(
                f"{EXPLORATION_RELEASE_TRUSTED_KEYS_ENV} public key for "
                f"{key_id.strip()!r} is not hexadecimal"
            ) from exc
        if len(public_key) != _ED25519_PUBLIC_KEY_BYTES:
            raise ValueError(
                f"{EXPLORATION_RELEASE_TRUSTED_KEYS_ENV} public key for "
                f"{key_id.strip()!r} must be {_ED25519_PUBLIC_KEY_BYTES} bytes"
            )
        keys[key_id.strip()] = public_key
    return MappingProxyType(keys)


def _operator_pinned_runtime_identity(
    certificate: E4aReleaseCertificate,
) -> ExplorationRuntimeIdentity:
    """Bind the installed certificate to this build's own version constants.

    The bindings and caps come from the certificate because the operator that
    pinned its issuer key is vouching for them. The three version fields come
    from the running code, so a certificate issued under a different gate,
    scoring or statistical policy is still rejected -- and the worker still
    recomputes the tool digest against the live tool inventory.
    """
    return ExplorationRuntimeIdentity(
        release_gate_version=E4A_RELEASE_GATE_VERSION,
        bindings=certificate.bindings,
        hard_caps=certificate.hard_caps,
        scoring_policy_version=EXPLORATION_PROFILE_VERSION,
        statistical_policy_version=EXPLORATION_STATISTICAL_POLICY_VERSION,
    )


def resolve_configured_release_trust() -> ExplorationReleaseTrust:
    """Resolve certificate, issuer keys and runtime identity as one decision."""
    public_keys = operator_pinned_release_public_keys()
    pinned_by_operator = public_keys is not TRUSTED_E4A_RELEASE_PUBLIC_KEYS
    identity = TRUSTED_EXPLORATION_RUNTIME_IDENTITY
    raw = os.environ.get(EXPLORATION_RELEASE_CERTIFICATE_ENV, "").strip()
    if not raw:
        return ExplorationReleaseTrust(None, public_keys, identity)
    try:
        path = Path(raw).expanduser()
        if not path.is_file() or path.stat().st_size > _MAX_CERTIFICATE_BYTES:
            return ExplorationReleaseTrust(None, public_keys, identity)
        certificate = E4aReleaseCertificate.model_validate_json(path.read_bytes())
        verified = verify_e4a_release_certificate(
            certificate,
            trusted_public_keys=public_keys,
        )
        if pinned_by_operator and identity is None:
            identity = _operator_pinned_runtime_identity(verified)
        assert_certificate_matches_runtime(verified, identity)
        return ExplorationReleaseTrust(verified, public_keys, identity)
    except (OSError, ValueError):
        return ExplorationReleaseTrust(None, public_keys, identity)


def load_configured_release_certificate(
    *,
    trusted_release_public_keys: Mapping[str, bytes] | None = None,
    trusted_runtime_identity: ExplorationRuntimeIdentity | None = None,
) -> E4aReleaseCertificate | None:
    """Load the operator-installed certificate; any ambiguity leaves E4b closed."""
    if trusted_release_public_keys is None and trusted_runtime_identity is None:
        return resolve_configured_release_trust().certificate
    raw = os.environ.get(EXPLORATION_RELEASE_CERTIFICATE_ENV, "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_file() or path.stat().st_size > _MAX_CERTIFICATE_BYTES:
            return None
        certificate = E4aReleaseCertificate.model_validate_json(path.read_bytes())
        verified = verify_e4a_release_certificate(
            certificate,
            trusted_public_keys=(
                TRUSTED_E4A_RELEASE_PUBLIC_KEYS
                if trusted_release_public_keys is None
                else trusted_release_public_keys
            ),
        )
        assert_certificate_matches_runtime(verified, trusted_runtime_identity)
        return verified
    except (OSError, ValueError):
        return None


class ExplorationService:
    def __init__(
        self,
        store: ArtifactStore,
        approvals: ApprovalService,
        jobs: JobService,
        *,
        release_certificate: E4aReleaseCertificate | None,
        trusted_release_public_keys: Mapping[str, bytes] = (
            TRUSTED_E4A_RELEASE_PUBLIC_KEYS
        ),
        trusted_runtime_identity: ExplorationRuntimeIdentity | None = (
            TRUSTED_EXPLORATION_RUNTIME_IDENTITY
        ),
        source_snapshot_resolver: SourceSnapshotResolver | None = None,
    ) -> None:
        self._store = store
        self._approvals = approvals
        self._jobs = jobs
        self._release_certificate = release_certificate
        self._trusted_release_public_keys = MappingProxyType(
            dict(trusted_release_public_keys)
        )
        self._trusted_runtime_identity = trusted_runtime_identity
        self._source_snapshot_resolver = (
            source_snapshot_resolver or self._resolve_source_snapshot
        )
        self._workflow_cache: (
            tuple[Path, tuple[int, int, int, int], _WorkflowProjectionParts] | None
        ) = None

    def require_release_certificate(
        self, *, provider: str | None = None
    ) -> E4aReleaseCertificate:
        """Revalidate on every entry, including reads and each SSE poll."""
        certificate = self._release_certificate
        if certificate is None:
            raise ExplorationReleaseUnavailableError(
                "Exploration API is disabled until a production E4a release "
                "certificate is installed."
            )
        try:
            validated = verify_e4a_release_certificate(
                certificate,
                trusted_public_keys=self._trusted_release_public_keys,
            )
            assert_certificate_matches_runtime(
                validated, self._trusted_runtime_identity
            )
        except (TypeError, ValueError) as exc:
            raise ExplorationReleaseUnavailableError(
                "The installed E4a release certificate failed integrity validation."
            ) from exc
        providers = {item.casefold() for item in validated.providers}
        if not providers:
            raise ExplorationReleaseUnavailableError(
                "The installed E4a release certificate contains no production provider."
            )
        if provider is not None and provider.casefold() not in providers:
            raise ExplorationReleaseUnavailableError(
                f"Provider {provider!r} is not covered by the installed E4a release certificate."
            )
        return validated

    def prepare(
        self,
        session_id: str,
        *,
        mode: Literal["open", "goal_directed"],
        goal: str | None,
        dataset_ids: tuple[str, ...],
        thinking_level: ExplorationTier,
        provider: str,
    ) -> ExplorationPrepared:
        certificate = self.require_release_certificate(provider=provider)
        snapshot = self._source_snapshot_resolver(session_id, dataset_ids)
        policy = build_exploration_policy(
            tier=thinking_level,
            dataset_scope=snapshot.dataset_ids,
            tool_capability_digest=certificate.bindings.tool_capability_digest,
            mode=mode,
            goal=goal,
        )
        assert_policy_matches_runtime(policy, self._trusted_runtime_identity)
        assert_policy_covered_by_certificate(policy, certificate)
        exploration_id = f"expl_{uuid.uuid4().hex}"
        prepared_at = datetime.now(UTC)
        action = {
            "type": APPROVAL_KIND_EXPLORATION_START,
            "exploration_id": exploration_id,
            "source_session_id": session_id,
            "project_id": snapshot.project_id,
            "policy_fingerprint": policy.policy_fingerprint,
            "data_state_witness": snapshot.data_state_witness,
            "release_certificate_digest": certificate.certificate_digest,
            "provider": provider.casefold(),
            "prepared_at": prepared_at.isoformat(),
        }
        payload = {
            **action,
            "policy": policy.model_dump(mode="json"),
        }
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_EXPLORATION_START,
            session_id=session_id,
            project_id=snapshot.project_id,
            action=action,
            payload=payload,
        )
        maximum_cost = policy.budget.llm.max_cost_usd
        if maximum_cost is None:  # Profiles must never make cost unbounded.
            raise ExplorationValidationError(
                "The exploration policy has no cost hard cap and cannot be approved."
            )
        return ExplorationPrepared(
            exploration_id=exploration_id,
            session_id=session_id,
            project_id=snapshot.project_id,
            policy=policy,
            data_state_witness=snapshot.data_state_witness,
            cost_range=ExplorationCostRange(
                minimum_usd=Decimal("0"),
                maximum_usd=maximum_cost,
            ),
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            release_certificate_digest=certificate.certificate_digest,
        )

    def start(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        provider: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
    ) -> ExplorationStarted:
        certificate = self.require_release_certificate(provider=provider)
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._start_once(
                session_id,
                action_hash=action_hash,
                approval_token=approval_token,
                certificate=certificate,
                provider=provider,
                payload_policy=payload_policy,
                llm_env=llm_env,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _start_once(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        certificate: E4aReleaseCertificate,
        provider: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> ExplorationStarted:
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                payload, digest, _status = self._approvals.inspect_payload(
                    action_hash, session_id=session_id
                )
                exploration_id = str(payload.get("exploration_id", ""))
                content = self._start_idempotency_content(
                    session_id=session_id,
                    exploration_id=exploration_id,
                    action_hash=action_hash,
                    approval_payload_digest=digest,
                    release_certificate_digest=certificate.certificate_digest,
                    provider=provider,
                    payload_policy=payload_policy,
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=str(payload.get("project_id", "")),
                    kind=EXPLORATION_JOB_KIND,
                    content=content,
                    env=llm_env,
                )
                metadata = self._metadata(session_id, exploration_id)
                job = self._jobs.get_job(str(existing["job_id"]))
                return ExplorationStarted(
                    exploration=self._view(metadata, job=job),
                    job=_job_view(job),
                )

        def validate(payload: dict[str, Any]) -> tuple[ExplorationRunMetadata, str]:
            return self._validated_start_payload(
                session_id,
                action_hash=action_hash,
                payload=payload,
                certificate=certificate,
                provider=provider,
            )

        payload, (metadata, provider_name) = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_EXPLORATION_START,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        self._write_metadata(metadata)
        journal = self._journal(metadata.exploration_id)
        journal.initialize(
            exploration_id=metadata.exploration_id,
            policy=metadata.policy,
            code_fingerprint=certificate.bindings.code_fingerprint,
            data_state_witness=metadata.data_state_witness,
        )
        approval_payload_digest = payload_digest(payload)
        content = self._start_idempotency_content(
            session_id=session_id,
            exploration_id=metadata.exploration_id,
            action_hash=action_hash,
            approval_payload_digest=approval_payload_digest,
            release_certificate_digest=certificate.certificate_digest,
            provider=provider_name,
            payload_policy=payload_policy,
        )
        try:
            job = self._jobs.create_exploration_job(
                _execution_session_id(metadata.exploration_id),
                project_id=metadata.project_id,
                source_session_id=session_id,
                exploration_id=metadata.exploration_id,
                policy=metadata.policy.model_dump(mode="json"),
                data_state_witness=metadata.data_state_witness,
                code_fingerprint=certificate.bindings.code_fingerprint,
                release_certificate_digest=certificate.certificate_digest,
                provider=provider_name,
                payload_policy=payload_policy,
                llm_env=llm_env,
                operation="start",
                idempotency_key=idempotency_key,
                idempotency_content=content,
            )
        except Exception:
            self._stop_after_launch_failure(journal)
            raise
        return ExplorationStarted(
            exploration=self._view(metadata, job=job),
            job=_job_view(job),
        )

    def get(self, session_id: str, exploration_id: str) -> ExplorationView:
        certificate = self.require_release_certificate()
        metadata = self._metadata(
            session_id, exploration_id, certificate=certificate
        )
        return self._view(metadata)

    def pause(self, session_id: str, exploration_id: str) -> ExplorationView:
        metadata = self._metadata(session_id, exploration_id)
        journal = self._journal(exploration_id)
        state = self._required_state(journal, exploration_id)
        if state.status == "stopped":
            raise ExplorationConflictError("a stopped exploration cannot be paused")
        if state.status == "running":
            journal.append_new("pause_requested", reason="requested through E4b API")
        return self._view(metadata)

    def resume(
        self,
        session_id: str,
        exploration_id: str,
        *,
        provider: str,
        payload_policy: str | None,
        llm_env: dict[str, str] | None,
        idempotency_key: str | None,
    ) -> ExplorationStarted:
        certificate = self.require_release_certificate(provider=provider)
        metadata = self._metadata(session_id, exploration_id, certificate=certificate)
        content = {
            "operation": "resume",
            "source_session_id": session_id,
            "exploration_id": exploration_id,
            "release_certificate_digest": certificate.certificate_digest,
            "provider": provider.casefold(),
            "payload_policy": payload_policy,
        }
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=metadata.project_id,
                    kind=EXPLORATION_JOB_KIND,
                    content=content,
                    env=llm_env,
                )
                job = self._jobs.get_job(str(existing["job_id"]))
                return ExplorationStarted(
                    exploration=self._view(metadata, job=job),
                    job=_job_view(job),
                )

        journal = self._journal(exploration_id)
        state = self._required_state(journal, exploration_id)
        if state.status != "paused":
            raise ExplorationConflictError(
                f"resume requires status 'paused'; current status is {state.status!r}"
            )
        snapshot = self._source_snapshot_resolver(
            session_id, metadata.policy.dataset_scope
        )
        if (
            snapshot.project_id != metadata.project_id
            or snapshot.dataset_ids != metadata.policy.dataset_scope
            or snapshot.data_state_witness != metadata.data_state_witness
        ):
            journal.append_new(
                "exploration_stopped",
                stop_reason="state_witness_changed",
                final_report_ref=None,
            )
            raise ExplorationSourceChangedError(exploration_id)
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise ExplorationConflictError(
                f"exploration still has an active job: {active['job_id']}"
            )
        journal.append_new("resumed")
        try:
            job = self._jobs.create_exploration_job(
                _execution_session_id(exploration_id),
                project_id=metadata.project_id,
                source_session_id=session_id,
                exploration_id=exploration_id,
                policy=metadata.policy.model_dump(mode="json"),
                data_state_witness=metadata.data_state_witness,
                code_fingerprint=certificate.bindings.code_fingerprint,
                release_certificate_digest=certificate.certificate_digest,
                provider=provider.casefold(),
                payload_policy=payload_policy,
                llm_env=llm_env,
                operation="resume",
                idempotency_key=idempotency_key,
                idempotency_content=content,
            )
        except Exception:
            self._stop_after_launch_failure(journal)
            raise
        return ExplorationStarted(
            exploration=self._view(metadata, job=job),
            job=_job_view(job),
        )

    def cancel(self, session_id: str, exploration_id: str) -> ExplorationView:
        metadata = self._metadata(session_id, exploration_id)
        journal = self._journal(exploration_id)
        state = self._required_state(journal, exploration_id)
        if state.status != "stopped":
            journal.append_new(
                "exploration_stopped",
                stop_reason="cancelled",
                final_report_ref=None,
            )
        active = self._store.find_active_job_for_session(
            _execution_session_id(exploration_id)
        )
        if active is not None:
            self._jobs.cancel_job(str(active["job_id"]))
        return self._view(metadata)

    def extend_budget(
        self,
        session_id: str,
        exploration_id: str,
        *,
        increase: BudgetCapIncrease,
        reason: str,
        idempotency_key: str,
    ) -> ExplorationBudgetExtended:
        certificate = self.require_release_certificate()
        metadata = self._metadata(
            session_id, exploration_id, certificate=certificate
        )
        journal = self._journal(exploration_id)
        key = idempotency_key.strip()
        if not key:
            raise ExplorationValidationError(
                "Idempotency-Key is required for a budget amendment"
            )
        amendment_id = "xamend_" + stable_hash(
            {"exploration_id": exploration_id, "idempotency_key": key},
            length=24,
        )
        with session_key_lock(
            self._store.root, _execution_session_id(exploration_id)
        ):
            existing = self._existing_amendment(
                exploration_id,
                amendment_id,
                increase=increase,
                reason=reason,
            )
            if existing is not None:
                amendment, effective_fingerprint = existing
            else:
                state = self._required_state(journal, exploration_id)
                if state.status == "stopped":
                    raise ExplorationConflictError(
                        "a stopped exploration cannot extend its budget"
                    )
                events = journal.events()
                effective = _effective_budget(metadata.policy.budget, events)
                proposed = apply_budget_increase(effective, increase)
                assert_budget_covered_by_certificate(proposed, certificate)
                amended = journal.amend_budget(
                    amendment_id=amendment_id, increase=increase
                )
                previous = _previous_fingerprint(journal.events(), amendment_id)
                amendment = BudgetAmendment(
                    amendment_id=amendment_id,
                    previous_effective_fingerprint=previous,
                    increase=increase,
                    reason=reason,
                    approved_by=_AMENDMENT_APPROVER,
                    created_at=datetime.now(UTC).isoformat(),
                )
                self._write_amendment(exploration_id, amendment)
                effective_fingerprint = amended.effective_policy_fingerprint
        return ExplorationBudgetExtended(
            exploration=self._view(metadata),
            amendment=amendment,
            effective_policy_fingerprint=effective_fingerprint,
        )

    def events_after(
        self, session_id: str, exploration_id: str, after_seq: int
    ) -> ExplorationEventsPage:
        self._metadata(session_id, exploration_id)
        if after_seq < -1:
            raise ExplorationValidationError("exploration event cursor cannot be below -1")
        journal = self._journal(exploration_id)
        state = self._required_state(journal, exploration_id)
        waiting = [event for event in journal.events() if event.seq > after_seq]
        selected = waiting[:EXPLORATION_EVENTS_PAGE_LIMIT]
        events = tuple(_event_view(event) for event in selected)
        cursor = selected[-1].seq if selected else after_seq
        return ExplorationEventsPage(
            events=events,
            status=state.status,
            cursor=cursor,
            exhausted=len(waiting) <= EXPLORATION_EVENTS_PAGE_LIMIT,
        )

    def _validated_start_payload(
        self,
        session_id: str,
        *,
        action_hash: str,
        payload: dict[str, Any],
        certificate: E4aReleaseCertificate,
        provider: str,
    ) -> tuple[ExplorationRunMetadata, str]:
        if str(payload.get("source_session_id", "")) != session_id:
            raise ExplorationValidationError(
                "the approval was prepared for a different source session"
            )
        exploration_id = str(payload.get("exploration_id", ""))
        provider_name = str(payload.get("provider", "")).casefold()
        if provider_name != provider.casefold():
            raise ExplorationValidationError(
                "provider settings changed since exploration approval"
            )
        if str(payload.get("release_certificate_digest", "")) != (
            certificate.certificate_digest
        ):
            raise ExplorationReleaseUnavailableError(
                "the approval is bound to a different E4a release certificate"
            )
        try:
            policy = ExplorationPolicy.model_validate(payload.get("policy"))
            assert_policy_sealed(policy)
        except (ExplorationPolicyIntegrityError, TypeError, ValueError) as exc:
            raise ExplorationValidationError(
                "the approved exploration policy failed integrity validation"
            ) from exc
        if policy.policy_fingerprint != str(payload.get("policy_fingerprint", "")):
            raise ExplorationValidationError(
                "the approved exploration policy fingerprint does not match"
            )
        if policy.tool_capability_digest != certificate.bindings.tool_capability_digest:
            raise ExplorationReleaseUnavailableError(
                "the approved tool capability digest is not covered by the certificate"
            )
        assert_policy_covered_by_certificate(policy, certificate)
        assert_policy_matches_runtime(policy, self._trusted_runtime_identity)
        snapshot = self._source_snapshot_resolver(session_id, policy.dataset_scope)
        expected_witness = str(payload.get("data_state_witness", ""))
        if snapshot.data_state_witness != expected_witness:
            raise ExplorationValidationError(
                "source data changed since approval; prepare the exploration again"
            )
        project_id = str(payload.get("project_id", ""))
        if snapshot.project_id != project_id:
            raise ExplorationValidationError("the approved project identity changed")
        try:
            created_at = datetime.fromisoformat(str(payload.get("prepared_at", "")))
        except ValueError as exc:
            raise ExplorationValidationError(
                "the approval carries no valid preparation timestamp"
            ) from exc
        return (
            ExplorationRunMetadata(
                exploration_id=exploration_id,
                source_session_id=session_id,
                project_id=project_id,
                policy=policy,
                data_state_witness=expected_witness,
                release_certificate_digest=certificate.certificate_digest,
                approval_action_hash=action_hash,
                created_at=created_at,
            ),
            provider_name,
        )

    def _metadata(
        self,
        session_id: str,
        exploration_id: str,
        *,
        certificate: E4aReleaseCertificate | None = None,
    ) -> ExplorationRunMetadata:
        current = certificate or self.require_release_certificate()
        path = self._metadata_path(exploration_id)
        try:
            metadata = ExplorationRunMetadata.model_validate_json(path.read_bytes())
        except FileNotFoundError as exc:
            raise ExplorationNotFoundError(exploration_id) from exc
        except (OSError, ValueError) as exc:
            raise ExplorationConflictError(
                f"exploration metadata is unreadable: {exploration_id}"
            ) from exc
        if metadata.exploration_id != exploration_id or metadata.source_session_id != session_id:
            raise ExplorationNotFoundError(exploration_id)
        if metadata.release_certificate_digest != current.certificate_digest:
            raise ExplorationReleaseUnavailableError(
                "the exploration is bound to a different E4a release certificate"
            )
        if metadata.policy.tool_capability_digest != current.bindings.tool_capability_digest:
            raise ExplorationReleaseUnavailableError(
                "the exploration tool capability digest is not covered by the certificate"
            )
        assert_policy_covered_by_certificate(metadata.policy, current)
        assert_policy_matches_runtime(
            metadata.policy, self._trusted_runtime_identity
        )
        assert_budget_covered_by_certificate(
            _effective_budget(
                metadata.policy.budget,
                self._journal(exploration_id).events(),
            ),
            current,
        )
        try:
            assert_policy_sealed(metadata.policy)
        except (ExplorationPolicyIntegrityError, ValueError) as exc:
            raise ExplorationConflictError(
                "the stored exploration policy failed integrity validation"
            ) from exc
        return metadata

    def _view(
        self, metadata: ExplorationRunMetadata, *, job: JobStatus | None = None
    ) -> ExplorationView:
        state = self._required_state(
            self._journal(metadata.exploration_id), metadata.exploration_id
        )
        projection = self._product_projection(metadata, state.current_round_index)
        cost_usd, max_cost_usd = self._cost_projection(metadata)
        amendments = self._amendments(metadata.exploration_id, state.amendment_ids)
        selected_job = job or self._latest_job(metadata.exploration_id)
        return ExplorationView(
            exploration_id=metadata.exploration_id,
            session_id=metadata.source_session_id,
            project_id=metadata.project_id,
            goal=metadata.policy.goal or "Explore freely",
            thinking_level=metadata.policy.thinking_level,
            status=state.status,
            stop_reason=state.stop_reason,
            last_seq=state.last_seq,
            policy_fingerprint=state.policy_fingerprint,
            effective_policy_fingerprint=state.effective_policy_fingerprint,
            data_state_witness=state.data_state_witness,
            amendment_ids=tuple(state.amendment_ids),
            current_hypothesis=projection.current_hypothesis,
            current_evidence=projection.evidence,
            insights=projection.insights,
            limitations=projection.limitations,
            coverage_targets=projection.coverage_targets,
            coverage_completed=projection.coverage_completed,
            coverage_unexplored=projection.coverage_unexplored,
            report=ExplorationReportView(
                available=state.final_report_ref is not None,
                artifact_ref=state.final_report_ref,
            ),
            budget=ExplorationBudgetView(
                base=metadata.policy.budget,
                max_llm_requests=state.max_llm_requests,
                remaining_llm_requests=state.remaining_llm_call_budget,
                max_successful_tool_calls=state.max_successful_tool_calls,
                remaining_successful_tool_calls=state.remaining_tool_call_budget,
                max_rows_scanned=state.max_rows_scanned,
                rows_scanned=state.rows_scanned,
                max_result_cells=state.max_result_cells,
                result_cells=state.result_cells,
                max_rounds=state.max_rounds,
                remaining_rounds=state.remaining_round_budget,
                max_cost_usd=max_cost_usd,
                cost_usd=cost_usd,
                remaining_cost_usd=(
                    None
                    if max_cost_usd is None
                    else max(Decimal("0"), max_cost_usd - cost_usd)
                ),
                llm_requests_used=state.llm_calls_settled + state.llm_calls_uncertain,
                successful_tool_calls_used=state.tool_calls_committed,
                rounds_used=state.rounds_started,
                amendments=amendments,
            ),
            job=None if selected_job is None else _job_view(selected_job),
            events_url=_events_url(metadata.source_session_id, metadata.exploration_id),
        )

    def _product_projection(
        self,
        metadata: ExplorationRunMetadata,
        current_round_index: int | None,
    ) -> _ProductProjection:
        root = shadow_run_root(self._store.root, metadata.exploration_id)
        workflow_path = validate_shadow_run_path(
            self._store.root,
            metadata.exploration_id,
            root / "workflow-state.json",
        )
        receipts: tuple[EvidenceReceipt, ...] = ()
        bundles: dict[str, ClaimBundle] = {}
        records: tuple[InsightRecord, ...] = ()
        workflow_coverage: tuple[str, ...] = ()
        cached = self._cached_workflow_projection(workflow_path)
        if cached is not None:
            return self._projection_from_parts(
                metadata, current_round_index, *cached
            )
        try:
            identity = _file_identity(workflow_path)
            raw = json.loads(workflow_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            identity = None
            raw = None
        except (OSError, json.JSONDecodeError) as exc:
            raise ExplorationConflictError(
                "exploration workflow projection is unreadable"
            ) from exc
        if isinstance(raw, dict):
            try:
                receipts = tuple(
                    load_verified_receipt(item)
                    for item in _required_list(raw, "committed_receipts")
                )
                bundle_items = tuple(
                    ClaimBundle.model_validate(item)
                    for item in _required_list(raw, "admitted_bundles")
                )
                bundles = {item.claim_bundle_id: item for item in bundle_items}
                records = tuple(
                    InsightRecord.model_validate(item)
                    for item in _required_list(raw, "insights")
                )
                workflow_coverage = tuple(
                    sorted(str(item) for item in _required_list(raw, "coverage_completed"))
                )
            except (TypeError, ValueError) as exc:
                raise ExplorationConflictError(
                    "exploration workflow projection failed validation"
                ) from exc

        parts = (receipts, bundles, records, workflow_coverage)
        if identity is not None and _file_identity(workflow_path) == identity:
            # Re-stat after the read, not before: only an identity that still
            # holds afterwards proves the bytes we parsed are the whole file.
            self._workflow_cache = (workflow_path, identity, parts)
        return self._projection_from_parts(metadata, current_round_index, *parts)

    def _cached_workflow_projection(
        self, workflow_path: Path
    ) -> _WorkflowProjectionParts | None:
        """Reuse the parse when the file is byte-for-byte the one we read.

        Every SSE frame makes the client refetch this view, and a deep run's
        workflow-state.json reaches ~2 MB with 150+ receipts to re-verify. The
        journal is authoritative, so a stale cache is never a correctness
        risk -- but a mismatched stat identity must invalidate it anyway.
        """
        cached = self._workflow_cache
        if cached is None or cached[0] != workflow_path:
            return None
        if _file_identity(workflow_path) != cached[1]:
            return None
        return cached[2]

    def _projection_from_parts(
        self,
        metadata: ExplorationRunMetadata,
        current_round_index: int | None,
        receipts: tuple[EvidenceReceipt, ...],
        bundles: dict[str, ClaimBundle],
        records: tuple[InsightRecord, ...],
        workflow_coverage: tuple[str, ...],
    ) -> _ProductProjection:
        evidence = tuple(
            _evidence_view(receipt)
            for receipt in sorted(receipts, key=lambda item: item.receipt_id)
        )
        insights: list[ExplorationInsightView] = []
        for record in sorted(records, key=lambda item: item.insight_id):
            bundle = bundles.get(record.claim_bundle_id)
            if bundle is None:
                raise ExplorationConflictError(
                    f"insight has no admitted claim bundle: {record.insight_id}"
                )
            insights.append(
                ExplorationInsightView(
                    insight_id=record.insight_id,
                    hypothesis_id=record.hypothesis_id,
                    statement=(
                        record.statement
                        if (record.statement or "").strip()
                        else "; ".join(claim.claim_text for claim in bundle.claims)
                    ),
                    family=record.family.value,
                    status=record.status,
                    trust_level=record.trust_level,
                    evidence_lane=bundle.evidence_lane,
                    proof=record.proof,
                    limitations=record.limitations,
                )
            )
        limitations = tuple(
            sorted(
                {
                    limitation
                    for insight in insights
                    for limitation in insight.limitations
                    if limitation.strip()
                }
            )
        )
        try:
            shadow = ShadowExplorationStore(self._store.root).read(
                metadata.exploration_id
            )
        except (OSError, ValueError) as exc:
            raise ExplorationConflictError(
                "exploration shadow projection failed validation"
            ) from exc
        completed = (
            workflow_coverage
            if shadow is None
            else tuple(sorted(set(shadow.coverage_completed)))
        )
        unexplored = (
            () if shadow is None else tuple(sorted(set(shadow.coverage_unexplored)))
        )
        return _ProductProjection(
            current_hypothesis=self._current_hypothesis(
                metadata.exploration_id, current_round_index
            ),
            evidence=evidence,
            insights=tuple(insights),
            limitations=limitations,
            coverage_targets=tuple(sorted(set(completed) | set(unexplored))),
            coverage_completed=completed,
            coverage_unexplored=unexplored,
        )

    def _current_hypothesis(
        self, exploration_id: str, current_round_index: int | None
    ) -> ExplorationHypothesisView | None:
        if current_round_index is None:
            return None
        root = shadow_run_root(self._store.root, exploration_id)
        recovery_root = validate_shadow_run_path(
            self._store.root, exploration_id, root / "phase-responses"
        )
        candidates: list[dict[str, Any]] = []
        try:
            paths = tuple(recovery_root.glob("*.json"))
        except OSError as exc:
            raise ExplorationConflictError(
                "exploration recovery projection is unreadable"
            ) from exc
        marker = f"{exploration_id}:round:{current_round_index}:"
        for path in paths:
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExplorationConflictError(
                    "exploration recovery projection is unreadable"
                ) from exc
            if not isinstance(body, dict) or not str(
                body.get("logical_step_id", "")
            ).startswith(marker):
                continue
            result = body.get("result")
            if not isinstance(result, dict):
                continue
            if result.get("kind") == "reduction_outcome":
                frontier = result.get("frontier")
                items = frontier.get("items") if isinstance(frontier, dict) else None
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        payload = item.get("payload")
                        if isinstance(payload, dict):
                            candidates.append(payload)
            elif result.get("kind") == "candidate_batch":
                items = result.get("candidates")
                if isinstance(items, list):
                    candidates.extend(item for item in items if isinstance(item, dict))
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("priority", 0.0)),
                int(item.get("sequence_index", 0)),
            ),
        )
        proposal = selected.get("proposal")
        if not isinstance(proposal, dict):
            return None
        return ExplorationHypothesisView(
            hypothesis_id=str(selected.get("hypothesis_id", "")),
            statement=str(proposal.get("statement", "")),
            why_selected=str(proposal.get("rationale", "")),
            status=str(selected.get("status", "proposed")),
            priority=float(selected.get("priority", 0.0)),
        )

    def _cost_projection(
        self, metadata: ExplorationRunMetadata
    ) -> tuple[Decimal, Decimal | None]:
        journal_events = self._journal(metadata.exploration_id).events()
        effective = metadata.policy.budget
        accepted = {budget_policy_fingerprint(effective.llm.to_policy())}
        for event in journal_events:
            if isinstance(event, BudgetAmendedEvent):
                effective = apply_budget_increase(effective, event.increase)
                accepted.add(budget_policy_fingerprint(effective.llm.to_policy()))
        budget_events = self._budget_events(metadata.exploration_id)
        try:
            restored = restore_run_budget_state(
                effective.llm.to_policy(),
                budget_events,
                run_started_at=(journal_events[0].occurred_at if journal_events else None),
                accepted_policy_fingerprints=frozenset(accepted),
            )
        except Exception as exc:
            raise ExplorationConflictError(
                "exploration LLM budget ledger failed validation"
            ) from exc
        return restored.cost_usd_used, effective.llm.max_cost_usd

    def _budget_events(self, exploration_id: str) -> list[TraceEvent]:
        root = shadow_run_root(self._store.root, exploration_id)
        path = validate_shadow_run_path(
            self._store.root, exploration_id, root / "llm-budget.jsonl"
        )
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ExplorationConflictError(
                "exploration LLM budget ledger is unreadable"
            ) from exc
        complete = body if body.endswith(b"\n") else body.rpartition(b"\n")[0]
        try:
            return [
                TraceEvent.model_validate_json(line)
                for line in complete.splitlines()
                if line.strip()
            ]
        except ValueError as exc:
            raise ExplorationConflictError(
                "exploration LLM budget ledger failed validation"
            ) from exc

    def _amendments(
        self, exploration_id: str, amendment_ids: list[str]
    ) -> tuple[BudgetAmendment, ...]:
        items: list[BudgetAmendment] = []
        for amendment_id in amendment_ids:
            path = self._amendment_path(exploration_id, amendment_id)
            try:
                items.append(BudgetAmendment.model_validate_json(path.read_bytes()))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                raise ExplorationConflictError(
                    f"budget amendment artifact is unreadable: {amendment_id}"
                ) from exc
        return tuple(items)

    def _latest_job(self, exploration_id: str) -> JobStatus | None:
        row = self._store.latest_job_for_session(_execution_session_id(exploration_id))
        return None if row is None else self._jobs.get_job(str(row["job_id"]))

    def _resolve_source_snapshot(
        self, session_id: str, dataset_ids: tuple[str, ...]
    ) -> ExplorationSourceSnapshot:
        return resolve_exploration_source_snapshot(
            self._store, session_id, dataset_ids
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])

    def read_report(self, session_id: str, exploration_id: str) -> str:
        """Return the deterministic markdown report the run finalized.

        The report lives in the shadow run directory, never in the artifact
        store: the exploration lane is product-store blind by construction, so
        this is the only path by which a reader can reach it.
        """
        metadata = self._metadata(session_id, exploration_id)
        state = self._required_state(
            self._journal(exploration_id), exploration_id
        )
        reference = state.final_report_ref
        if not reference:
            raise ExplorationNotFoundError(f"{exploration_id}:report")
        path = validate_shadow_run_path(
            self._store.root, exploration_id, self._store.root / reference
        )
        del metadata
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ExplorationNotFoundError(f"{exploration_id}:report") from exc
        except OSError as exc:
            raise ExplorationConflictError(
                "the exploration report is unreadable"
            ) from exc

    def _journal(self, exploration_id: str) -> JsonlExplorationJournal:
        root = shadow_run_root(self._store.root, exploration_id)
        return JsonlExplorationJournal(
            validate_shadow_run_path(
                self._store.root, exploration_id, root / "journal.jsonl"
            )
        )

    @staticmethod
    def _required_state(
        journal: JsonlExplorationJournal, exploration_id: str
    ):
        state = journal.rebuild()
        if state is None:
            raise ExplorationNotFoundError(exploration_id)
        return state

    def _metadata_path(self, exploration_id: str) -> Path:
        root = shadow_run_root(self._store.root, exploration_id)
        return validate_shadow_run_path(
            self._store.root, exploration_id, root / "api-request.json"
        )

    def _write_metadata(self, metadata: ExplorationRunMetadata) -> None:
        path = self._metadata_path(metadata.exploration_id)
        if path.exists():
            try:
                existing = ExplorationRunMetadata.model_validate_json(path.read_bytes())
            except (OSError, ValueError) as exc:
                raise ExplorationConflictError(
                    "existing exploration metadata is unreadable"
                ) from exc
            if existing != metadata:
                raise ExplorationConflictError(
                    "exploration metadata is immutable and already differs"
                )
            return
        _write_json_atomic(path, metadata.model_dump_json(indent=2).encode("utf-8"))

    def _write_amendment(
        self, exploration_id: str, amendment: BudgetAmendment
    ) -> None:
        root = shadow_run_root(self._store.root, exploration_id)
        path = validate_shadow_run_path(
            self._store.root,
            exploration_id,
            root / "amendments" / f"{amendment.amendment_id}.json",
        )
        _write_json_atomic(path, amendment.model_dump_json(indent=2).encode("utf-8"))

    def _existing_amendment(
        self,
        exploration_id: str,
        amendment_id: str,
        *,
        increase: BudgetCapIncrease,
        reason: str,
    ) -> tuple[BudgetAmendment, str] | None:
        matching_event = next(
            (
                event
                for event in self._journal(exploration_id).events()
                if isinstance(event, BudgetAmendedEvent)
                and event.amendment_id == amendment_id
            ),
            None,
        )
        if matching_event is None:
            return None
        path = self._amendment_path(exploration_id, amendment_id)
        try:
            amendment = BudgetAmendment.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            amendment = BudgetAmendment(
                amendment_id=amendment_id,
                previous_effective_fingerprint=_previous_fingerprint(
                    self._journal(exploration_id).events(), amendment_id
                ),
                increase=matching_event.increase,
                reason=reason,
                approved_by=_AMENDMENT_APPROVER,
                created_at=datetime.now(UTC).isoformat(),
            )
            self._write_amendment(exploration_id, amendment)
        except (OSError, ValueError) as exc:
            raise ExplorationConflictError(
                f"budget amendment artifact is unreadable: {amendment_id}"
            ) from exc
        if amendment.increase != increase or amendment.reason != reason:
            raise ExplorationValidationError(
                "Idempotency-Key is already bound to a different budget amendment"
            )
        return amendment, matching_event.effective_policy_fingerprint

    def _amendment_path(self, exploration_id: str, amendment_id: str) -> Path:
        root = shadow_run_root(self._store.root, exploration_id)
        return validate_shadow_run_path(
            self._store.root,
            exploration_id,
            root / "amendments" / f"{amendment_id}.json",
        )

    @staticmethod
    def _start_idempotency_content(
        *,
        session_id: str,
        exploration_id: str,
        action_hash: str,
        approval_payload_digest: str,
        release_certificate_digest: str,
        provider: str,
        payload_policy: str | None,
    ) -> dict[str, object]:
        return {
            "operation": "start",
            "source_session_id": session_id,
            "exploration_id": exploration_id,
            "action_hash": action_hash,
            "approval_payload_digest": approval_payload_digest,
            "release_certificate_digest": release_certificate_digest,
            "provider": provider.casefold(),
            "payload_policy": payload_policy,
        }

    @staticmethod
    def _stop_after_launch_failure(journal: JsonlExplorationJournal) -> None:
        try:
            state = journal.rebuild()
            if state is not None and state.status != "stopped":
                journal.append_new(
                    "exploration_stopped",
                    stop_reason="failed",
                    final_report_ref=None,
                )
        except Exception:
            pass


def _event_view(event: Any) -> ExplorationEventView:
    body = event.model_dump(mode="json")
    for key in (
        "schema_version",
        "seq",
        "exploration_id",
        "event_type",
        "attempt_epoch",
        "occurred_at",
    ):
        body.pop(key, None)
    return ExplorationEventView(
        event_id=f"{event.exploration_id}:{event.seq}",
        exploration_id=event.exploration_id,
        seq=event.seq,
        type=event.event_type,
        occurred_at=event.occurred_at,
        data=body,
    )


def resolve_exploration_source_snapshot(
    store: ArtifactStore,
    session_id: str,
    dataset_ids: tuple[str, ...],
) -> ExplorationSourceSnapshot:
    if INTERNAL_SESSION_MARKER in session_id:
        raise SessionNotFoundError(session_id)
    row = store.get_session_index_row(session_id)
    if row is None:
        raise SessionNotFoundError(session_id)
    project_id = str(row["project_id"])
    if str(row.get("status", "")) != "completed":
        raise ExplorationConflictError("exploration requires a completed source session")
    loaded = load_run(project_id, session_id, workspace=store.root)
    if not loaded.ok or not loaded.datasets_available or loaded.result is None:
        raise ExplorationValidationError(
            "the source session datasets cannot be reloaded for exploration"
        )
    datasets = {item.record.dataset_id: item for item in loaded.result.loaded_datasets}
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in datasets]
    if missing:
        raise ExplorationValidationError(
            f"dataset is not part of the source session: {missing[0]}"
        )
    profile_ids: dict[str, str] = {}
    for artifact in loaded.result.artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            dataset_id = artifact.payload.get("dataset_id")
            if isinstance(dataset_id, str):
                profile_ids[dataset_id] = artifact.id
    entries = [
        (
            dataset_id,
            profile_ids.get(dataset_id),
            datasets[dataset_id].record.content_hash,
        )
        for dataset_id in dataset_ids
    ]
    return ExplorationSourceSnapshot(
        project_id=project_id,
        dataset_ids=dataset_ids,
        data_state_witness=data_state_witness_digest(entries),
    )


def assert_policy_covered_by_certificate(
    policy: ExplorationPolicy,
    certificate: E4aReleaseCertificate,
) -> None:
    """Refuse product caps larger than those exercised by certified evidence."""
    assert_budget_covered_by_certificate(policy.budget, certificate)


def assert_certificate_matches_runtime(
    certificate: E4aReleaseCertificate,
    trusted: ExplorationRuntimeIdentity | None,
) -> None:
    """Require the signed evidence to name the exact deployed build and caps."""
    if trusted is None:
        raise ValueError("trusted exploration runtime identity is unavailable")
    if trusted.release_gate_version != E4A_RELEASE_GATE_VERSION:
        raise ValueError("trusted runtime release-gate version is stale")
    if certificate.gate_version != trusted.release_gate_version:
        raise ValueError("certificate release-gate version does not match runtime")
    if certificate.bindings != trusted.bindings:
        raise ValueError("certificate evidence/build/tool bindings do not match runtime")
    if certificate.hard_caps != trusted.hard_caps:
        raise ValueError("certificate hard caps do not match runtime")
    if trusted.scoring_policy_version != EXPLORATION_PROFILE_VERSION:
        raise ValueError("trusted runtime exploration profile does not match code")
    if (
        trusted.statistical_policy_version
        != EXPLORATION_STATISTICAL_POLICY_VERSION
    ):
        raise ValueError("trusted runtime statistical policy does not match code")


def assert_policy_matches_runtime(
    policy: ExplorationPolicy,
    trusted: ExplorationRuntimeIdentity | None,
) -> None:
    if trusted is None:
        raise ExplorationReleaseUnavailableError(
            "The trusted exploration runtime identity is unavailable."
        )
    if (
        policy.tool_capability_digest != trusted.bindings.tool_capability_digest
        or policy.scoring_policy_version != trusted.scoring_policy_version
        or policy.statistical_policy_version != trusted.statistical_policy_version
    ):
        raise ExplorationReleaseUnavailableError(
            "The exploration policy does not match the trusted runtime identity."
        )


def assert_budget_covered_by_certificate(
    budget: ExplorationBudgetPolicy,
    certificate: E4aReleaseCertificate,
) -> None:
    """Refuse effective (including amended) caps beyond certified evidence."""
    caps = certificate.hard_caps
    checks: tuple[tuple[str, int | float | Decimal | None, Decimal], ...] = (
        ("llm requests", budget.llm.max_requests, Decimal(caps.max_llm_requests)),
        (
            "total tokens",
            budget.llm.max_total_tokens,
            Decimal(caps.max_total_tokens),
        ),
        (
            "cost",
            budget.llm.max_cost_usd,
            Decimal(str(caps.max_cost_usd)),
        ),
        (
            "wall time",
            budget.llm.max_wall_seconds,
            Decimal(str(caps.max_wall_seconds)),
        ),
        (
            "tool calls",
            budget.max_successful_tool_calls,
            Decimal(caps.max_tool_calls),
        ),
        (
            "rows scanned",
            budget.max_rows_scanned,
            Decimal(caps.max_rows_scanned),
        ),
        (
            "result cells",
            budget.max_result_cells,
            Decimal(caps.max_cells_scanned),
        ),
    )
    for name, value, certified_maximum in checks:
        if value is None or Decimal(str(value)) > certified_maximum:
            raise ExplorationReleaseUnavailableError(
                f"The {name} policy cap is not covered by the E4a release certificate."
            )


def _effective_budget(
    base: ExplorationBudgetPolicy,
    events: Sequence[object],
) -> ExplorationBudgetPolicy:
    effective = base
    for event in events:
        if isinstance(event, BudgetAmendedEvent):
            effective = apply_budget_increase(effective, event.increase)
    return effective


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"workflow-state {key} must be a list")
    return value


def _previous_fingerprint(events: list[Any], amendment_id: str) -> str:
    current = ""
    for event in events:
        if event.event_type == "exploration_started":
            current = event.policy_fingerprint
        elif isinstance(event, BudgetAmendedEvent):
            if event.amendment_id == amendment_id:
                if not current:
                    raise ExplorationConflictError(
                        "budget amendment chain has no base policy fingerprint"
                    )
                return current
            current = event.effective_policy_fingerprint
    raise ExplorationConflictError("budget amendment is missing from the journal")


def _execution_session_id(exploration_id: str) -> str:
    return f"{EXPLORATION_SESSION_PREFIX}{exploration_id.removeprefix('expl_')}"


def _events_url(session_id: str, exploration_id: str) -> str:
    return f"/api/v1/sessions/{session_id}/explorations/{exploration_id}/events"


_MAX_EVIDENCE_FACTS = 8


def _evidence_view(receipt: EvidenceReceipt) -> ExplorationEvidenceView:
    """Carry the adjudicating numbers, not just the receipt id.

    The panel previously rendered "<tool>: N result(s)" plus a row of opaque
    ids, so the one thing a reader needs -- what the probe measured -- was
    reachable only by reading the run directory by hand.
    """
    statistics = receipt.statistics
    view = (
        None
        if statistics is None
        else ExplorationStatisticsView(
            test_name=statistics.test_name,
            outcome=statistics.hypothesis_outcome,
            test_statistic=statistics.test_statistic,
            p_value=statistics.p_value,
            adjusted_p_value=statistics.adjusted_p_value,
            effect_size=statistics.effect_size,
            ci_low=statistics.ci_low,
            ci_high=statistics.ci_high,
            sample_size=statistics.sample_size,
        )
    )
    return ExplorationEvidenceView(
        receipt_id=receipt.receipt_id,
        tool_name=receipt.tool_name,
        summary=_evidence_summary(receipt),
        fact_ids=tuple(fact.fact_id for fact in receipt.facts),
        facts=tuple(
            ExplorationFactView(
                fact_id=fact.fact_id,
                name=fact.name,
                value=fact.value,
                unit=fact.unit,
            )
            for fact in receipt.facts[:_MAX_EVIDENCE_FACTS]
        ),
        statistics=view,
    )


def _evidence_summary(receipt: EvidenceReceipt) -> str:
    parts = [f"{receipt.tool_name}: {receipt.result_count} result(s)"]
    statistics = receipt.statistics
    if statistics is not None:
        numbers = [
            f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in (
                ("p", statistics.adjusted_p_value or statistics.p_value),
                ("effect", statistics.effect_size),
                ("n", statistics.sample_size),
            )
            if value is not None
        ]
        if numbers:
            parts.append(f"{statistics.test_name} · " + ", ".join(numbers))
    return " — ".join(parts)


def _job_view(job: JobStatus) -> ExplorationJobView:
    return ExplorationJobView(
        job_id=job.job_id,
        execution_session_id=job.session_id,
        status=job.status,
    )


def _write_json_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

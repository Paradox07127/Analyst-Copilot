"""Server-side pending-action approvals (§6.0).

The `action_hash` contract from `core.permissions` is unchanged; only the
pending state lives in the SQLite `pending_actions`
table, so stateless HTTP clients can preview on one request and approve on
another. Generic over action kinds: cleaning today, chat/investigation later.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from eda_platform.core.permissions import action_hash as compute_action_hash
from eda_platform.core.store import ArtifactStore

DEFAULT_APPROVAL_TTL_SECONDS = 30 * 60
DEFAULT_CONTENTION_TIMEOUT_SECONDS = 30.0
DEFAULT_CONTENTION_MAX_ATTEMPTS = 64

T = TypeVar("T")


def payload_digest(payload: dict[str, Any]) -> str:
    """Full sha256 over the canonical JSON of a pending payload (C4): a row
    whose payload_json was edited in the DB no longer matches and reads 404."""
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ApprovalServiceError(Exception):
    error_code = "approval_error"

    def __init__(self, message: str, action_hash: str) -> None:
        super().__init__(message)
        self.action_hash = action_hash


class ApprovalNotFoundError(ApprovalServiceError):
    error_code = "approval_not_found"

    def __init__(self, action_hash: str) -> None:
        super().__init__(
            "Approval not found for this action; run preview again to request a new one.",
            action_hash,
        )


class ApprovalExpiredError(ApprovalServiceError):
    error_code = "approval_expired"

    def __init__(self, action_hash: str) -> None:
        super().__init__(
            "Approval expired; run preview again and approve the fresh result.",
            action_hash,
        )


class ApprovalConsumedError(ApprovalServiceError):
    error_code = "approval_consumed"

    def __init__(self, action_hash: str) -> None:
        super().__init__(
            "Approval was already used; run preview again to approve another apply.",
            action_hash,
        )


class ApprovalIdempotencyRaceError(ApprovalConsumedError):
    """The same idempotent request lost the approval's atomic consume race."""


class ApprovalService:
    def __init__(
        self,
        store: ArtifactStore,
        *,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
        contention_timeout_seconds: float = DEFAULT_CONTENTION_TIMEOUT_SECONDS,
        contention_max_attempts: int = DEFAULT_CONTENTION_MAX_ATTEMPTS,
    ) -> None:
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._contention_timeout_seconds = max(0.0, contention_timeout_seconds)
        self._contention_max_attempts = max(1, contention_max_attempts)

    def register(
        self,
        *,
        kind: str,
        session_id: str,
        project_id: str,
        action: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[str, str, datetime]:
        """Persist a pending action and return (action_hash, generation, expires_at).

        The hash is computed by the existing `core.permissions.action_hash`,
        so the value handed to the client is exactly what the execution path
        will re-derive and verify. The generation is a fresh one-time token per
        register (C1): re-previewing rotates it, invalidating older tokens.
        """
        digest = compute_action_hash(action)
        generation = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        self._store.create_pending_action(
            action_hash=digest,
            session_id=session_id,
            project_id=project_id,
            kind=kind,
            payload_json=json.dumps(payload, ensure_ascii=False),
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            generation=generation,
            payload_digest=payload_digest(payload),
        )
        return digest, generation, expires_at

    def validate_and_consume(
        self,
        action_hash: str,
        *,
        kind: str,
        session_id: str,
        generation: str,
        idempotency_key: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Consume a pending approval exactly once and return its payload.

        Rows are keyed by (action_hash, session_id), so another run's approval
        with the same hash is simply invisible; a kind, generation, or payload
        digest mismatch also reads as not-found — a tampered hash/token must
        not leak that some other action with that hash exists.
        """
        absolute_deadline = (
            time.monotonic() + self._contention_timeout_seconds
            if deadline is None
            else deadline
        )
        for _attempt in range(self._contention_max_attempts):
            row = self._store.get_pending_action(action_hash, session_id=session_id)
            if (
                row is None
                or str(row["kind"]) != kind
                or str(row["generation"]) != generation
            ):
                raise ApprovalNotFoundError(action_hash)
            payload = json.loads(str(row["payload_json"]))
            payload = payload if isinstance(payload, dict) else {}
            if payload_digest(payload) != str(row["payload_digest"]):
                raise ApprovalNotFoundError(action_hash)
            if row["status"] == "consumed":
                if (
                    idempotency_key is not None
                    and row.get("consumed_idempotency_key") == idempotency_key
                ):
                    raise ApprovalIdempotencyRaceError(action_hash)
                raise ApprovalConsumedError(action_hash)
            # Aware-UTC isoformat strings share one format, so lexicographic
            # comparison matches chronological order (same convention as jobs).
            now = datetime.now(UTC).isoformat()
            if row["status"] == "expired" or str(row["expires_at"]) <= now:
                self._store.expire_pending_action(action_hash, session_id=session_id, now=now)
                raise ApprovalExpiredError(action_hash)
            if self._store.consume_pending_action(
                action_hash,
                session_id=session_id,
                generation=generation,
                now=now,
                idempotency_key=idempotency_key,
            ):
                try:
                    # C2: never return the pre-consume read — rebuild the payload
                    # from the row as it stands after the successful flip.
                    consumed = self._store.get_pending_action(action_hash, session_id=session_id)
                    if (
                        consumed is None
                        or str(consumed["generation"]) != generation
                    ):
                        raise ApprovalNotFoundError(action_hash)
                    payload = json.loads(str(consumed["payload_json"]))
                    payload = payload if isinstance(payload, dict) else {}
                    if payload_digest(payload) != str(consumed["payload_digest"]):
                        raise ApprovalNotFoundError(action_hash)
                    return payload
                except BaseException:
                    # The CAS belongs to this invocation, so a failure before
                    # handing the payload to the producer must not burn the
                    # one-time token.
                    with suppress(Exception):
                        self._store.restore_pending_action(
                            action_hash, session_id=session_id
                        )
                    raise

            # Lost a concurrent race after the reads above; classify from a
            # fresh row so the caller gets 404 vs 409 vs 410 correctly.
            fresh = self._store.get_pending_action(action_hash, session_id=session_id)
            if fresh is None or str(fresh["generation"]) != generation:
                raise ApprovalNotFoundError(action_hash)
            if fresh["status"] == "consumed":
                if (
                    idempotency_key is not None
                    and fresh.get("consumed_idempotency_key") == idempotency_key
                ):
                    raise ApprovalIdempotencyRaceError(action_hash)
                raise ApprovalConsumedError(action_hash)
            if fresh["status"] != "pending" or str(fresh["expires_at"]) <= now:
                raise ApprovalExpiredError(action_hash)
            if time.monotonic() >= absolute_deadline:
                break

        # Preserve the existing typed 409 approval-consumed contract when
        # contention never settles; importantly, never recurse or reset budget.
        if idempotency_key is not None:
            raise ApprovalIdempotencyRaceError(action_hash)
        raise ApprovalConsumedError(action_hash)

    def validate_then_consume(
        self,
        action_hash: str,
        *,
        kind: str,
        session_id: str,
        generation: str,
        validate: Callable[[dict[str, Any]], T],
        idempotency_key: str | None = None,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], T]:
        """Validate request/source identity before the one-way consume CAS.

        The callback must be read-only. Its return value carries any validated
        source objects into the producer so callers do not repeat a second,
        subtly different validation after consumption. A concurrent re-arm
        changes ``generation`` and makes the consume fail closed.
        """

        payload, _digest, status = self.inspect_payload(action_hash, session_id=session_id)
        row = self._store.get_pending_action(action_hash, session_id=session_id)
        if (
            row is None
            or str(row["kind"]) != kind
            or str(row["generation"]) != generation
        ):
            raise ApprovalNotFoundError(action_hash)
        now = datetime.now(UTC).isoformat()
        if status == "expired" or str(row["expires_at"]) <= now:
            self._store.expire_pending_action(action_hash, session_id=session_id, now=now)
            raise ApprovalExpiredError(action_hash)
        if status == "consumed":
            if (
                idempotency_key is not None
                and row.get("consumed_idempotency_key") == idempotency_key
            ):
                raise ApprovalIdempotencyRaceError(action_hash)
            raise ApprovalConsumedError(action_hash)

        validate(payload)
        consumed = self.validate_and_consume(
            action_hash,
            kind=kind,
            session_id=session_id,
            generation=generation,
            idempotency_key=idempotency_key,
            deadline=deadline,
        )
        try:
            # Close the validate→consume TOCTOU window for mutable source
            # fingerprints and logical lanes. The callback is required to be
            # read-only, so repeating it cannot publish a side effect.
            validated = validate(consumed)
        except BaseException:
            with suppress(Exception):
                self._store.restore_pending_action(action_hash, session_id=session_id)
            raise
        return consumed, validated

    @contextmanager
    def compensate_on_failure(
        self,
        action_hash: str,
        *,
        session_id: str,
    ) -> Iterator[None]:
        """Re-arm the same generation when no durable producer was committed."""

        try:
            yield
        except BaseException:
            with suppress(Exception):
                self._store.restore_pending_action(action_hash, session_id=session_id)
            raise

    def inspect_payload(
        self, action_hash: str, *, session_id: str
    ) -> tuple[dict[str, Any], str, str]:
        """Read a pending or consumed payload with the same integrity guard.

        Job idempotency checks use this before status-specific replay checks so
        a freshly re-armed approval with changed effective parameters is a
        content mismatch, not an ambiguous "approval not consumed" response.
        """
        row = self._store.get_pending_action(action_hash, session_id=session_id)
        if row is None:
            raise ApprovalNotFoundError(action_hash)
        payload = json.loads(str(row["payload_json"]))
        payload = payload if isinstance(payload, dict) else {}
        digest = payload_digest(payload)
        if digest != str(row["payload_digest"]):
            raise ApprovalNotFoundError(action_hash)
        return payload, digest, str(row["status"])

    def wait_for_idempotent_resolution(
        self,
        action_hash: str,
        *,
        session_id: str,
        idempotency_key: str,
        timeout_seconds: float = 30.0,
        deadline: float | None = None,
    ) -> bool:
        """Wait until a same-key consume winner either publishes or rolls back.

        Approval consumption precedes producer validation and job insertion.
        A race loser may replay once the winner's job exists, or retry ownership
        if the winner restores the approval after a failure.
        """
        absolute_deadline = (
            time.monotonic() + timeout_seconds if deadline is None else deadline
        )
        while time.monotonic() < absolute_deadline:
            if self._store.find_by_idempotency_key(idempotency_key) is not None:
                return True
            row = self._store.get_pending_action(action_hash, session_id=session_id)
            if row is not None and row["status"] == "pending":
                return True
            time.sleep(0.01)
        return self._store.find_by_idempotency_key(idempotency_key) is not None

    def run_idempotent_producer(
        self,
        action_hash: str,
        *,
        session_id: str,
        idempotency_key: str | None,
        operation: Callable[[float], T],
    ) -> T:
        """Run one approval-backed producer under one bounded retry budget."""
        deadline = time.monotonic() + self._contention_timeout_seconds
        if idempotency_key is None:
            return operation(deadline)
        last_race: ApprovalIdempotencyRaceError | None = None
        for _attempt in range(self._contention_max_attempts):
            try:
                return operation(deadline)
            except ApprovalIdempotencyRaceError as exc:
                last_race = exc
                if not self.wait_for_idempotent_resolution(
                    action_hash,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    deadline=deadline,
                ):
                    break
        assert last_race is not None
        raise last_race

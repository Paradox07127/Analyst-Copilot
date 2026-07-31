"""Recoverable cross-media JSON replacement operations.

SQLite owns concurrency and the committed resource version, while files remain
the portable representation.  A replacement therefore has two short database
transactions with filesystem work between them:

``reserve (DB) -> apply (filesystem) -> finalize (DB)``.

The schema is owned by :mod:`eda_platform.core.store`; this module deliberately
does not create or migrate tables.  It only uses ``resource_heads``,
``storage_operations`` and ``storage_operation_items``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.fs import BINARY_FLAG, fsync_directory

if TYPE_CHECKING:
    from eda_platform.core.store import ArtifactStore

OperationState = Literal[
    "prepared",
    "fs_applied",
    "db_committed",
    "done",
    "blocked",
    "aborted",
]
FaultHook = Callable[[str, str, int | None], None]

_ACTIVE_STATES = ("prepared", "fs_applied", "db_committed", "blocked")
_MISSING_DIGEST = hashlib.sha256(b"eda-platform:missing-json-resource:v1").hexdigest()
_CONTROL_ROOT = PurePosixPath(".storage-operations")
_DEFAULT_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class StorageOperationError(RuntimeError):
    """Base class for journal and resource invariant failures."""


class UnsafeStoragePathError(StorageOperationError):
    """A path was absolute, escaped the workspace, or traversed a symlink."""


class ResourceVersionConflictError(StorageOperationError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Resource version conflict: expected {expected_version}, current is "
            f"{current_version}."
        )
        self.expected_version = expected_version
        self.current_version = current_version


class ResourceDigestMismatchError(StorageOperationError):
    """The committed database digest and current filesystem content disagree."""


class ResourceOperationInProgressError(StorageOperationError):
    def __init__(self, op_id: str) -> None:
        super().__init__(f"Resource already has an unfinished storage operation: {op_id}.")
        self.op_id = op_id


class StorageOperationBlockedError(StorageOperationError):
    def __init__(self, op_id: str, reason: str) -> None:
        super().__init__(f"Storage operation {op_id} is blocked: {reason}")
        self.op_id = op_id
        self.reason = reason


class StorageOperationNotFoundError(StorageOperationError):
    def __init__(self, op_id: str) -> None:
        super().__init__(f"Unknown storage operation: {op_id}.")
        self.op_id = op_id


class StorageOperationNotAppliedError(StorageOperationError):
    """Finalize was requested while at least one file still had its base content."""


class StorageRequestKeyConflictError(StorageOperationError):
    """A request key was already bound to different replacement content."""


@dataclass(frozen=True)
class ResourceTarget:
    resource_kind: str
    project_id: str
    resource_key: str

    def __post_init__(self) -> None:
        for label, value in (
            ("resource_kind", self.resource_kind),
            ("project_id", self.project_id),
            ("resource_key", self.resource_key),
        ):
            if not value or "\x00" in value:
                raise ValueError(f"{label} must be a non-empty text value.")


@dataclass(frozen=True)
class ReplacementFile:
    relative_path: str
    payload: object


@dataclass(frozen=True)
class ResourceHead:
    target: ResourceTarget
    relative_path: str
    version: int
    content_digest: str
    updated_at: str


@dataclass(frozen=True)
class ReservedReplacement:
    op_id: str
    target: ResourceTarget
    expected_version: int
    target_version: int
    base_digest: str
    target_digest: str
    state: OperationState


@dataclass(frozen=True)
class ReplacementReplayItem:
    """Original immutable input stored for one replacement item."""

    relative_path: str
    payload: bytes


@dataclass(frozen=True)
class ReplacementReplayRecord:
    """Read-only operation record used to validate an idempotent replay."""

    op_id: str
    target: ResourceTarget
    expected_version: int
    target_version: int
    target_digest: str
    state: OperationState
    items: tuple[ReplacementReplayItem, ...]


@dataclass(frozen=True)
class _OperationItem:
    ordinal: int
    relative_path: str
    work_relative_path: str
    base_digest: str
    target_digest: str
    payload: bytes


@dataclass(frozen=True)
class _Operation:
    reservation: ReservedReplacement
    request_key: str | None
    error_message: str | None
    updated_at: str
    items: tuple[_OperationItem, ...]


@dataclass(frozen=True)
class PreparedReplacement:
    """Filesystem-digested replacement ready for reservation in a caller transaction."""

    op_id: str
    target: ResourceTarget
    expected_version: int
    target_version: int
    base_digest: str
    target_digest: str
    request_key: str | None
    items: tuple[_OperationItem, ...]


def canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic UTF-8 JSON bytes without persisting source formatting."""
    value: Any
    if isinstance(payload, bytes):
        raw = payload.decode("utf-8")
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    elif isinstance(payload, str):
        value = json.loads(payload, parse_constant=_reject_nonfinite)
    else:
        # Round-trip through the encoder so non-JSON values fail here, before a
        # reservation can block the resource.
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(payload: object) -> str:
    """SHA-256 of the canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def composite_digest(file_digests: Mapping[str, str]) -> str:
    """Digest a path-to-content-digest map using the journal's resource envelope."""

    return _resource_digest(file_digests)


def missing_resource_digest(relative_paths: Sequence[str]) -> str:
    """Composite digest for a resource whose tracked files are all absent."""

    if isinstance(relative_paths, (str, bytes)):
        raise ValueError("relative_paths must be a sequence of paths.")
    normalized = tuple(_normalize_resource_path(path) for path in relative_paths)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("relative_paths must be non-empty and unique.")
    return _resource_digest({path: _MISSING_DIGEST for path in normalized})


def staging_relative_path(op_id: str, ordinal: int) -> str:
    """Deterministic workspace-relative staging location for one operation item."""
    _validate_operation_id(op_id)
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative.")
    return str(_CONTROL_ROOT / "staging" / op_id / f"{ordinal:04d}.tmp")


def quarantine_relative_path(op_id: str, ordinal: int, source_relative_path: str) -> str:
    """Deterministic future run-delete quarantine path.

    Run deletion is intentionally not implemented in this slice, but it uses the
    same journal item schema and path policy.
    """
    _validate_operation_id(op_id)
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative.")
    source = _normalize_resource_path(source_relative_path)
    return str(
        _CONTROL_ROOT
        / "quarantine"
        / op_id
        / f"{ordinal:04d}-{PurePosixPath(source).name}"
    )


class StorageOperationJournal:
    """Reserve, apply, finalize and recover versioned JSON file replacements."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        fault_hook: FaultHook | None = None,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive.")
        self.root = store.root.resolve()
        self.db_path = store.db_path
        self._fault_hook = fault_hook
        self._max_payload_bytes = max_payload_bytes

    def get_head(self, target: ResourceTarget) -> ResourceHead | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select relative_path, version, content_digest, updated_at
                from resource_heads
                where resource_kind = ? and project_id = ? and resource_key = ?
                """,
                _target_params(target),
            ).fetchone()
        return None if row is None else _head_from_row(target, row)

    def bootstrap_head(
        self,
        target: ResourceTarget,
        *,
        primary_relative_path: str,
        version: int,
        tracked_relative_paths: Sequence[str] | None = None,
        expected_content_digest: str | None = None,
    ) -> ResourceHead:
        """Insert a committed head for legacy files, idempotently.

        Bootstrap is the one intentional exception to the rule that filesystem
        I/O stays outside a database transaction: the legacy files have no
        SQLite owner yet.  We read them once before ``BEGIN IMMEDIATE`` and once
        immediately before the head insert.  A change between those reads fails
        closed rather than publishing an already-stale committed digest.
        """
        if version < 0:
            raise ValueError("version must be non-negative.")
        primary = _normalize_resource_path(primary_relative_path)
        tracked = tuple(
            _normalize_resource_path(path)
            for path in (tracked_relative_paths or (primary,))
        )
        if not tracked or len(set(tracked)) != len(tracked):
            raise ValueError("tracked_relative_paths must be non-empty and unique.")
        digests = {path: self._digest_path(path) for path in tracked}
        content_digest = _resource_digest(digests)
        if (
            expected_content_digest is not None
            and content_digest != expected_content_digest
        ):
            raise ResourceDigestMismatchError(
                "Legacy resource files no longer match the caller's pre-read "
                "composite digest."
            )
        now = _now()

        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                self._fault("before_bootstrap_recheck", "bootstrap", None)
                rechecked = {path: self._digest_path(path) for path in tracked}
                rechecked_digest = _resource_digest(rechecked)
                if rechecked != digests or (
                    expected_content_digest is not None
                    and rechecked_digest != expected_content_digest
                ):
                    raise ResourceDigestMismatchError(
                        "Legacy resource files changed while their committed head "
                        "was being bootstrapped."
                    )
                row = conn.execute(
                    """
                    select relative_path, version, content_digest, updated_at
                    from resource_heads
                    where resource_kind = ? and project_id = ? and resource_key = ?
                    """,
                    _target_params(target),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        insert into resource_heads(
                            resource_kind, project_id, resource_key, relative_path,
                            version, content_digest, updated_at
                        ) values(?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            target.resource_kind,
                            target.project_id,
                            target.resource_key,
                            primary,
                            version,
                            content_digest,
                            now,
                        ),
                    )
                    conn.commit()
                    return ResourceHead(target, primary, version, content_digest, now)

                head = _head_from_row(target, row)
                if (
                    head.relative_path != primary
                    or head.version != version
                    or head.content_digest != content_digest
                ):
                    raise ResourceDigestMismatchError(
                        "Existing resource head does not match the legacy files "
                        "supplied for bootstrap."
                    )
                conn.commit()
                return head
            except Exception:
                conn.rollback()
                raise

    def get_replacement_replay(
        self, request_key: str
    ) -> ReplacementReplayRecord | None:
        """Return the original replacement request without mutating or recovering it."""

        if not request_key:
            raise ValueError("request_key must be non-empty.")
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select op_id from storage_operations
                where op_kind = 'replace_resource' and request_key = ?
                """,
                (request_key,),
            ).fetchone()
        if row is None:
            return None
        return self.get_replacement_replay_by_op(str(row["op_id"]))

    def get_replacement_replay_by_op(
        self, op_id: str
    ) -> ReplacementReplayRecord:
        """Return one immutable replacement request by durable operation id."""
        operation = self._load_operation(op_id)
        reservation = operation.reservation
        return ReplacementReplayRecord(
            op_id=reservation.op_id,
            target=reservation.target,
            expected_version=reservation.expected_version,
            target_version=reservation.target_version,
            target_digest=reservation.target_digest,
            state=reservation.state,
            items=tuple(
                ReplacementReplayItem(
                    relative_path=item.relative_path,
                    payload=item.payload,
                )
                for item in operation.items
            ),
        )

    def reserve_replace(
        self,
        target: ResourceTarget,
        *,
        expected_version: int,
        replacements: Sequence[ReplacementFile],
        request_key: str | None = None,
    ) -> ReservedReplacement:
        """CAS-reserve one resource generation without doing filesystem I/O in the txn."""
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative.")
        prepared = self.prepare_replace(
            target,
            expected_version=expected_version,
            replacements=replacements,
            request_key=request_key,
        )

        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                reservation = self.reserve_prepared_in_transaction(conn, prepared)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        self.reservation_committed(reservation.op_id)
        return reservation

    def reservation_committed(self, op_id: str) -> None:
        """Run the post-reservation fault boundary after a caller transaction commits."""
        self._fault("after_reserve", op_id, None)

    def replacement_is_reversible(self, op_id: str) -> bool:
        """True only while a prepared operation has not changed any target file."""
        operation = self._load_operation(op_id)
        if operation.reservation.state != "prepared":
            return False
        return all(
            item.base_digest == item.target_digest
            or self._operation_item_digest(op_id, item) == item.base_digest
            for item in operation.items
        )

    def prepare_replace(
        self,
        target: ResourceTarget,
        *,
        expected_version: int,
        replacements: Sequence[ReplacementFile],
        request_key: str | None = None,
    ) -> PreparedReplacement:
        """Digest files before a caller-owned ``BEGIN IMMEDIATE`` transaction."""
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative.")
        items = self._prepare_items(replacements)
        return PreparedReplacement(
            op_id=f"sop_{uuid4().hex}",
            target=target,
            expected_version=expected_version,
            target_version=expected_version + 1,
            base_digest=_resource_digest(
                {item.relative_path: item.base_digest for item in items}
            ),
            target_digest=_resource_digest(
                {item.relative_path: item.target_digest for item in items}
            ),
            request_key=request_key,
            items=items,
        )

    def reserve_prepared_in_transaction(
        self,
        conn: sqlite3.Connection,
        prepared: PreparedReplacement,
    ) -> ReservedReplacement:
        """Reserve a prepared operation inside an existing immediate transaction."""
        if prepared.request_key is not None:
            replay = self._reservation_for_request(conn, prepared.request_key)
            if replay is not None:
                if (
                    replay.target != prepared.target
                    or replay.expected_version != prepared.expected_version
                    or replay.target_digest != prepared.target_digest
                ):
                    raise StorageRequestKeyConflictError(
                        "The storage request key is already bound to different content."
                    )
                return replay

        active = conn.execute(
            f"""
            select op_id from storage_operations
            where resource_kind = ? and project_id = ? and resource_key = ?
              and state in ({_sql_placeholders(len(_ACTIVE_STATES))})
            limit 1
            """,
            (*_target_params(prepared.target), *_ACTIVE_STATES),
        ).fetchone()
        if active is not None:
            raise ResourceOperationInProgressError(str(active["op_id"]))

        head_row = conn.execute(
            """
            select relative_path, version, content_digest, updated_at
            from resource_heads
            where resource_kind = ? and project_id = ? and resource_key = ?
            """,
            _target_params(prepared.target),
        ).fetchone()
        if head_row is None:
            raise ResourceDigestMismatchError(
                "Resource has no committed head; bootstrap it before reserving a write."
            )
        head = _head_from_row(prepared.target, head_row)
        if head.version != prepared.expected_version:
            raise ResourceVersionConflictError(prepared.expected_version, head.version)
        if head.content_digest != prepared.base_digest:
            raise ResourceDigestMismatchError(
                "Current files do not match the committed resource digest."
            )

        now = _now()
        conn.execute(
            """
            insert into storage_operations(
                op_id, op_kind, resource_kind, project_id, resource_key,
                expected_version, target_version, base_digest, target_digest,
                request_key, state, error_code, error_message, created_at, updated_at
            ) values(
                ?, 'replace_resource', ?, ?, ?, ?, ?, ?, ?, ?,
                'prepared', null, null, ?, ?
            )
            """,
            (
                prepared.op_id,
                prepared.target.resource_kind,
                prepared.target.project_id,
                prepared.target.resource_key,
                prepared.expected_version,
                prepared.target_version,
                prepared.base_digest,
                prepared.target_digest,
                prepared.request_key,
                now,
                now,
            ),
        )
        for item in prepared.items:
            conn.execute(
                """
                insert into storage_operation_items(
                    op_id, ordinal, mode, source_relpath, work_relpath,
                    base_digest, target_digest, payload, required
                ) values(?, ?, 'replace_file', ?, ?, ?, ?, ?, 1)
                """,
                (
                    prepared.op_id,
                    item.ordinal,
                    item.relative_path,
                    staging_relative_path(prepared.op_id, item.ordinal),
                    item.base_digest,
                    item.target_digest,
                    item.payload,
                ),
            )
        return ReservedReplacement(
            op_id=prepared.op_id,
            target=prepared.target,
            expected_version=prepared.expected_version,
            target_version=prepared.target_version,
            base_digest=prepared.base_digest,
            target_digest=prepared.target_digest,
            state="prepared",
        )

    def apply_replace(self, op_id: str) -> ReservedReplacement:
        """Idempotently replace every file and mark the journal ``fs_applied``."""

        with self._operation_lock(op_id):
            return self._apply_replace_locked(op_id)

    def _apply_replace_locked(self, op_id: str) -> ReservedReplacement:
        operation = self._load_operation(op_id)
        if operation.reservation.state == "done":
            return operation.reservation
        if operation.reservation.state == "blocked":
            raise StorageOperationBlockedError(
                op_id, operation.error_message or "manual recovery is required"
            )
        if operation.reservation.state not in ("prepared", "fs_applied"):
            raise StorageOperationError(
                f"Operation {op_id} cannot apply files from state "
                f"{operation.reservation.state!r}."
            )

        for item in operation.items:
            current = self._operation_item_digest(op_id, item)
            if current == item.target_digest:
                self._remove_staging_if_present(item.work_relative_path)
                continue
            if current != item.base_digest:
                reason = (
                    f"unknown digest for {item.relative_path}: got {current}, expected base "
                    f"{item.base_digest} or target {item.target_digest}"
                )
                self._mark_blocked(op_id, "unknown_digest", reason)
                raise StorageOperationBlockedError(op_id, reason)
            self._replace_one(op_id, item)

        self._fault("after_apply_before_state", op_id, None)
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                cursor = conn.execute(
                    """
                    update storage_operations
                    set state = 'fs_applied', error_code = null, error_message = null,
                        updated_at = ?
                    where op_id = ? and state in ('prepared', 'fs_applied')
                    """,
                    (_now(), op_id),
                )
                if cursor.rowcount != 1:
                    raise StorageOperationError(
                        f"Operation {op_id} changed state while files were being applied."
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return _with_state(operation.reservation, "fs_applied")

    def finalize_replace(self, op_id: str) -> ResourceHead:
        """Commit the target generation after all files independently verify."""

        with self._operation_lock(op_id):
            return self._finalize_replace_locked(op_id)

    def _finalize_replace_locked(self, op_id: str) -> ResourceHead:
        operation = self._load_operation(op_id)
        if operation.reservation.state == "done":
            return self._completed_head(operation)
        if operation.reservation.state == "blocked":
            raise StorageOperationBlockedError(
                op_id, operation.error_message or "manual recovery is required"
            )

        actual: dict[str, str] = {}
        for item in operation.items:
            digest = self._operation_item_digest(op_id, item)
            actual[item.relative_path] = digest
            if (
                item.base_digest != item.target_digest
                and digest == item.base_digest
            ):
                raise StorageOperationNotAppliedError(
                    f"Operation {op_id} has not replaced {item.relative_path}."
                )
            if digest != item.target_digest:
                reason = (
                    f"{item.relative_path} changed after apply; got {digest}, "
                    f"expected {item.target_digest}"
                )
                self._mark_blocked(op_id, "unknown_digest", reason)
                raise StorageOperationBlockedError(op_id, reason)
        if _resource_digest(actual) != operation.reservation.target_digest:
            reason = "Applied files do not match the operation target digest."
            self._mark_blocked(op_id, "target_digest_mismatch", reason)
            raise StorageOperationBlockedError(op_id, reason)

        self._fault("before_finalize", op_id, None)
        reservation = operation.reservation
        now = _now()
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                cursor = conn.execute(
                    """
                    update resource_heads
                    set version = ?, content_digest = ?, updated_at = ?
                    where resource_kind = ? and project_id = ? and resource_key = ?
                      and version = ? and content_digest = ?
                    """,
                    (
                        reservation.target_version,
                        reservation.target_digest,
                        now,
                        reservation.target.resource_kind,
                        reservation.target.project_id,
                        reservation.target.resource_key,
                        reservation.expected_version,
                        reservation.base_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ResourceDigestMismatchError(
                        "Resource head changed before replacement finalization."
                    )
                op_cursor = conn.execute(
                    """
                    update storage_operations
                    set state = 'done', error_code = null, error_message = null, updated_at = ?
                    where op_id = ? and state in ('prepared', 'fs_applied')
                    """,
                    (now, op_id),
                )
                if op_cursor.rowcount != 1:
                    raise StorageOperationError(
                        f"Operation {op_id} changed state before finalization."
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        self._fault("after_finalize", op_id, None)
        head = self.get_head(reservation.target)
        if head is None:
            raise ResourceDigestMismatchError("Finalized resource head is missing.")
        return head

    def replace_resource(
        self,
        target: ResourceTarget,
        *,
        expected_version: int,
        replacements: Sequence[ReplacementFile],
        request_key: str | None = None,
    ) -> ResourceHead:
        reservation = self.reserve_replace(
            target,
            expected_version=expected_version,
            replacements=replacements,
            request_key=request_key,
        )
        with self._operation_lock(reservation.op_id):
            self._apply_replace_locked(reservation.op_id)
            return self._finalize_replace_locked(reservation.op_id)

    def recover_replace(self, op_id: str) -> ResourceHead:
        """Resume a prepared/applied operation by inspecting file digests."""

        with self._operation_lock(op_id):
            return self._recover_replace_locked(op_id)

    def _recover_replace_locked(self, op_id: str) -> ResourceHead:
        operation = self._load_operation(op_id)
        if operation.reservation.state == "done":
            return self._completed_head(operation)
        if operation.reservation.state == "aborted":
            raise StorageOperationError(f"Operation {op_id} was aborted.")
        if operation.reservation.state == "db_committed":
            raise StorageOperationError(
                "db_committed is a run-delete cleanup state, not a resource replacement state."
            )

        if operation.reservation.state == "blocked":
            unknown = self._unknown_item_digests(operation)
            if unknown:
                raise StorageOperationBlockedError(op_id, "; ".join(unknown))
            with closing(self._connect()) as conn:
                self._begin_immediate(conn)
                try:
                    conn.execute(
                        """
                        update storage_operations
                        set state = 'prepared', error_code = null, error_message = null,
                            updated_at = ?
                        where op_id = ? and state = 'blocked'
                        """,
                        (_now(), op_id),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        self._apply_replace_locked(op_id)
        return self._finalize_replace_locked(op_id)

    def recover_target(self, target: ResourceTarget) -> ResourceHead | None:
        """Recover the target's one active replacement, if present."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                select op_id from storage_operations
                where resource_kind = ? and project_id = ? and resource_key = ?
                  and op_kind = 'replace_resource'
                  and state in ({_sql_placeholders(len(_ACTIVE_STATES))})
                order by created_at, op_id limit 1
                """,
                (*_target_params(target), *_ACTIVE_STATES),
            ).fetchone()
        if row is None:
            return self.get_head(target)
        return self.recover_replace(str(row["op_id"]))

    def _prepare_items(self, replacements: Sequence[ReplacementFile]) -> tuple[_OperationItem, ...]:
        if not replacements:
            raise ValueError("At least one replacement file is required.")
        normalized: list[tuple[str, bytes]] = []
        for replacement in replacements:
            relative_path = _normalize_resource_path(replacement.relative_path)
            payload = _payload_bytes(replacement.payload)
            if len(payload) > self._max_payload_bytes:
                raise ValueError(
                    f"Replacement payload exceeds {self._max_payload_bytes} bytes."
                )
            # Validate JSON and digest its semantic content; preserve the caller's
            # formatting when writing the portable file.
            canonical_json_bytes(payload)
            normalized.append((relative_path, payload))
        paths = [path for path, _payload in normalized]
        if len(set(paths)) != len(paths):
            raise ValueError("Replacement relative paths must be unique.")
        normalized.sort(key=lambda item: item[0])
        return tuple(
            _OperationItem(
                ordinal=ordinal,
                relative_path=path,
                work_relative_path="",
                base_digest=self._digest_path(path),
                target_digest=canonical_digest(payload),
                payload=payload,
            )
            for ordinal, (path, payload) in enumerate(normalized)
        )

    def _replace_one(self, op_id: str, item: _OperationItem) -> None:
        target = self._safe_path(item.relative_path, create_parent=True)
        work = self._safe_path(item.work_relative_path, create_parent=True)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise UnsafeStoragePathError(
                f"Replacement target is not a regular file: {item.relative_path}."
            )
        self._fault("before_stage_write", op_id, item.ordinal)
        if work.is_symlink():
            raise UnsafeStoragePathError(
                f"Staging target cannot be a symlink: {item.work_relative_path}."
            )
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            | getattr(os, "O_NOFOLLOW", 0)
            | BINARY_FLAG
        )
        descriptor = os.open(work, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(item.payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._fault("after_stage_fsync", op_id, item.ordinal)
        os.replace(work, target)
        fsync_directory(target.parent)
        self._fault("after_replace", op_id, item.ordinal)

    def _digest_path(self, relative_path: str) -> str:
        path = self._safe_path(relative_path)
        if path.is_symlink():
            raise UnsafeStoragePathError(f"Storage target cannot be a symlink: {relative_path}.")
        if not path.exists():
            return _MISSING_DIGEST
        if not path.is_file():
            raise UnsafeStoragePathError(
                f"JSON storage target is not a regular file: {relative_path}."
            )
        payload = path.read_bytes()
        if len(payload) > self._max_payload_bytes:
            raise StorageOperationError(
                f"Stored JSON exceeds {self._max_payload_bytes} bytes: {relative_path}."
            )
        return canonical_digest(payload)

    def _safe_path(self, relative_path: str, *, create_parent: bool = False) -> Path:
        normalized = _normalize_relative_path(relative_path)
        candidate = self.root.joinpath(*PurePosixPath(normalized).parts)
        parent = candidate.parent
        if create_parent:
            _create_safe_directories(self.root, parent)
        else:
            _assert_safe_existing_parents(self.root, parent)
        try:
            resolved_parent = parent.resolve(strict=False)
        except OSError as exc:
            raise UnsafeStoragePathError(f"Cannot resolve storage path: {normalized}.") from exc
        if resolved_parent != self.root and not resolved_parent.is_relative_to(self.root):
            raise UnsafeStoragePathError(f"Storage path escapes workspace: {normalized}.")
        return candidate

    def _remove_staging_if_present(self, relative_path: str) -> None:
        path = self._safe_path(relative_path)
        if path.is_symlink():
            raise UnsafeStoragePathError(f"Staging path cannot be a symlink: {relative_path}.")
        if path.is_file():
            path.unlink()
            fsync_directory(path.parent)

    def _load_operation(self, op_id: str) -> _Operation:
        _validate_operation_id(op_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select op_id, resource_kind, project_id, resource_key,
                       expected_version, target_version, base_digest, target_digest,
                       request_key, state, error_message, updated_at
                from storage_operations
                where op_id = ? and op_kind = 'replace_resource'
                """,
                (op_id,),
            ).fetchone()
            if row is None:
                raise StorageOperationNotFoundError(op_id)
            item_rows = conn.execute(
                """
                select ordinal, source_relpath, work_relpath, base_digest,
                       target_digest, payload
                from storage_operation_items
                where op_id = ? and mode = 'replace_file'
                order by ordinal
                """,
                (op_id,),
            ).fetchall()
        if not item_rows:
            raise StorageOperationError(f"Replacement operation {op_id} has no items.")
        state = _operation_state(str(row["state"]))
        target = ResourceTarget(
            resource_kind=str(row["resource_kind"]),
            project_id=str(row["project_id"]),
            resource_key=str(row["resource_key"]),
        )
        reservation = ReservedReplacement(
            op_id=str(row["op_id"]),
            target=target,
            expected_version=int(row["expected_version"]),
            target_version=int(row["target_version"]),
            base_digest=str(row["base_digest"]),
            target_digest=str(row["target_digest"]),
            state=state,
        )
        items = tuple(
            _OperationItem(
                ordinal=int(item["ordinal"]),
                relative_path=_normalize_resource_path(str(item["source_relpath"])),
                work_relative_path=_normalize_relative_path(str(item["work_relpath"])),
                base_digest=str(item["base_digest"]),
                target_digest=str(item["target_digest"]),
                payload=_payload_bytes(item["payload"]),
            )
            for item in item_rows
        )
        return _Operation(
            reservation=reservation,
            request_key=None if row["request_key"] is None else str(row["request_key"]),
            error_message=(
                None if row["error_message"] is None else str(row["error_message"])
            ),
            updated_at=str(row["updated_at"]),
            items=items,
        )

    def _completed_head(self, operation: _Operation) -> ResourceHead:
        """Return the generation this operation committed, not a later head.

        A request-key replay may arrive after another writer has advanced the
        resource. Returning the current head would misreport the original
        operation's result even though no file was changed by the replay.
        """
        current = self.get_head(operation.reservation.target)
        if current is None:
            raise ResourceDigestMismatchError("Done operation has no resource head.")
        return ResourceHead(
            target=operation.reservation.target,
            relative_path=current.relative_path,
            version=operation.reservation.target_version,
            content_digest=operation.reservation.target_digest,
            updated_at=operation.updated_at,
        )

    def _reservation_for_request(
        self, conn: sqlite3.Connection, request_key: str
    ) -> ReservedReplacement | None:
        row = conn.execute(
            """
            select op_id, resource_kind, project_id, resource_key,
                   expected_version, target_version, base_digest, target_digest, state
            from storage_operations
            where op_kind = 'replace_resource' and request_key = ?
            """,
            (request_key,),
        ).fetchone()
        if row is None:
            return None
        return ReservedReplacement(
            op_id=str(row["op_id"]),
            target=ResourceTarget(
                str(row["resource_kind"]),
                str(row["project_id"]),
                str(row["resource_key"]),
            ),
            expected_version=int(row["expected_version"]),
            target_version=int(row["target_version"]),
            base_digest=str(row["base_digest"]),
            target_digest=str(row["target_digest"]),
            state=_operation_state(str(row["state"])),
        )

    def _unknown_item_digests(self, operation: _Operation) -> list[str]:
        unknown: list[str] = []
        for item in operation.items:
            try:
                current = self._digest_path(item.relative_path)
            except (StorageOperationError, UnicodeError, ValueError) as exc:
                unknown.append(f"{item.relative_path}: unreadable ({type(exc).__name__})")
                continue
            if current not in (item.base_digest, item.target_digest):
                unknown.append(f"{item.relative_path}: {current}")
        return unknown

    def _operation_item_digest(self, op_id: str, item: _OperationItem) -> str:
        try:
            return self._digest_path(item.relative_path)
        except StorageOperationBlockedError:
            raise
        except (StorageOperationError, UnicodeError, ValueError) as exc:
            reason = (
                f"unknown content for {item.relative_path}: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )
            self._mark_blocked(op_id, "unreadable_resource", reason)
            raise StorageOperationBlockedError(op_id, reason) from exc

    def _mark_blocked(self, op_id: str, error_code: str, reason: str) -> None:
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                conn.execute(
                    """
                    update storage_operations
                    set state = 'blocked', error_code = ?, error_message = ?, updated_at = ?
                    where op_id = ? and state not in ('done', 'aborted')
                    """,
                    (error_code, reason[:1000], _now(), op_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def _operation_lock(self, op_id: str) -> Iterator[None]:
        """Serialize filesystem recovery for one durable operation across processes."""

        _validate_operation_id(op_id)
        relative_path = str(_CONTROL_ROOT / "locks" / f"{op_id}.lock")
        lock_path = self._safe_path(relative_path, create_parent=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | BINARY_FLAG
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise UnsafeStoragePathError(
                f"Cannot open storage operation lock: {relative_path}."
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UnsafeStoragePathError(
                    f"Storage operation lock is not a regular file: {relative_path}."
                )
            lock_exclusive(descriptor)
            try:
                yield
            finally:
                unlock(descriptor)
        finally:
            os.close(descriptor)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 5000")
        conn.execute("pragma foreign_keys = on")
        return conn

    @staticmethod
    def _begin_immediate(conn: sqlite3.Connection) -> None:
        conn.execute("begin immediate")

    def _fault(self, stage: str, op_id: str, ordinal: int | None) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage, op_id, ordinal)


def _payload_bytes(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _resource_digest(digests: Mapping[str, str]) -> str:
    envelope = [
        {"relative_path": path, "digest": digest}
        for path, digest in sorted(digests.items())
    ]
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def _target_params(target: ResourceTarget) -> tuple[str, str, str]:
    return target.resource_kind, target.project_id, target.resource_key


def _head_from_row(target: ResourceTarget, row: sqlite3.Row) -> ResourceHead:
    return ResourceHead(
        target=target,
        relative_path=str(row["relative_path"]),
        version=int(row["version"]),
        content_digest=str(row["content_digest"]),
        updated_at=str(row["updated_at"]),
    )


def _with_state(
    reservation: ReservedReplacement, state: OperationState
) -> ReservedReplacement:
    return ReservedReplacement(
        op_id=reservation.op_id,
        target=reservation.target,
        expected_version=reservation.expected_version,
        target_version=reservation.target_version,
        base_digest=reservation.base_digest,
        target_digest=reservation.target_digest,
        state=state,
    )


def _normalize_relative_path(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise UnsafeStoragePathError("Storage path must be a non-empty POSIX relative path.")
    path = PurePosixPath(raw)
    if path.is_absolute() or path == PurePosixPath("."):
        raise UnsafeStoragePathError(f"Storage path must be relative: {raw!r}.")
    if any(part in ("", ".", "..") for part in path.parts):
        raise UnsafeStoragePathError(f"Storage path contains an unsafe segment: {raw!r}.")
    normalized = str(path)
    if PurePosixPath(normalized).parts[0] == _CONTROL_ROOT.parts[0] and not normalized.startswith(
        f"{_CONTROL_ROOT}/"
    ):
        raise UnsafeStoragePathError(f"Invalid storage control path: {raw!r}.")
    return normalized


def _normalize_resource_path(raw: str) -> str:
    normalized = _normalize_relative_path(raw)
    if PurePosixPath(normalized).parts[0] == _CONTROL_ROOT.parts[0]:
        raise UnsafeStoragePathError(
            "Resource paths cannot use the storage-operation control directory."
        )
    return normalized


def _validate_operation_id(op_id: str) -> None:
    if not op_id or any(not (char.isalnum() or char in "_-") for char in op_id):
        raise ValueError("Operation id contains unsafe path characters.")


def _assert_safe_existing_parents(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise UnsafeStoragePathError(f"Storage parent cannot be a symlink: {current.name}.")
        if current.exists() and not current.is_dir():
            raise UnsafeStoragePathError(
                f"Storage parent is not a directory: {current.name}."
            )


def _create_safe_directories(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise UnsafeStoragePathError(f"Storage parent cannot be a symlink: {current.name}.")
        if current.exists():
            if not current.is_dir():
                raise UnsafeStoragePathError(
                    f"Storage parent is not a directory: {current.name}."
                )
            continue
        current.mkdir(exist_ok=True)
        if current.is_symlink() or not current.is_dir():
            raise UnsafeStoragePathError(
                f"Storage parent is not a safe directory: {current.name}."
            )
        fsync_directory(current.parent)


def _operation_state(raw: str) -> OperationState:
    if raw not in {
        "prepared",
        "fs_applied",
        "db_committed",
        "done",
        "blocked",
        "aborted",
    }:
        raise StorageOperationError(f"Unknown storage operation state: {raw!r}.")
    return raw  # type: ignore[return-value]


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not supported: {value}.")


def _sql_placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _now() -> str:
    return datetime.now(UTC).isoformat()

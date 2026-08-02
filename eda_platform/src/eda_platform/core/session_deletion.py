"""Recoverable, cross-media run deletion.

SQLite owns the logical deletion.  Run files and chat transcripts are first
renamed into deterministic workspace-local quarantine paths, then one database
transaction removes the run-owned rows, inserts the durable audit event, and
marks the operation ``db_committed``.  The audit JSONL and quarantine purge are
recoverable mirrors/cleanup and never re-insert the SQLite event.

This module deliberately depends only on the frozen storage foundation schema.
Service/API exception mapping is left to the application layer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import NoReturn
from uuid import uuid4

from eda_platform.core.bounded_pagination import (
    jsonl_path_key,
    run_resource_scopes,
)
from eda_platform.core.dev_log import LLM_DEBUG_FILENAME
from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.fs import fsync_directory, remove_tree
from eda_platform.core.ids import AUDIT_SESSION_ID, is_safe_session_id
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.storage_operations import quarantine_relative_path
from eda_platform.core.store import ArtifactStore, session_results_relative_path
from eda_platform.schemas.sessions import TraceEvent

FaultHook = Callable[[str, str, int | None], None]

_ACTIVE_OPERATION_STATES = ("prepared", "fs_applied", "db_committed", "blocked")
_TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled")
_AUDIT_EVENT_TYPE = "session.deleted"


class SessionDeletionError(RuntimeError):
    """Base class for typed core deletion failures."""


class SessionDeletionNotFoundError(SessionDeletionError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class SessionDeletionBusyError(SessionDeletionError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(f"Run {session_id} has an active job: {job_id}")
        self.session_id = session_id
        self.job_id = job_id


class SessionDeletionRetryableError(SessionDeletionError):
    def __init__(self, op_id: str, reason: str) -> None:
        super().__init__(f"Run deletion {op_id} can be retried: {reason}")
        self.op_id = op_id
        self.reason = reason


class SessionDeletionBlockedError(SessionDeletionError):
    def __init__(self, op_id: str, reason: str) -> None:
        super().__init__(f"Run deletion {op_id} is blocked: {reason}")
        self.op_id = op_id
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SessionDeletionResult:
    session_id: str
    project_id: str
    op_id: str
    deleted: bool = True


@dataclass(frozen=True, slots=True)
class _DeletionOperation:
    op_id: str
    project_id: str
    session_id: str
    state: str
    error_message: str | None


@dataclass(frozen=True, slots=True)
class _DeletionItem:
    ordinal: int
    source_relative_path: str
    work_relative_path: str
    payload: bytes | None
    required: bool


class SessionDeletionCoordinator:
    """Reserve, apply, finalize, and recover one run deletion."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self.root = store.root.resolve()
        self.db_path = store.db_path
        self._fault_hook = fault_hook

    def delete(self, session_id: str) -> SessionDeletionResult:
        """Delete a live run or resume its one active deletion operation."""

        if not is_safe_session_id(session_id):
            raise SessionDeletionNotFoundError(session_id)
        existing = self._active_operation_for_session(session_id)
        if existing is not None:
            return self.recover(existing.op_id)
        with session_key_lock(self.root, session_id):
            existing = self._active_operation_for_session(session_id)
            if existing is not None:
                return self._recover_locked(existing.op_id)
            op_id, created = self._reserve(session_id)
            if created:
                self._fault("after_reserve", op_id, None)
            return self._recover_locked(op_id)

    def recover(self, op_id: str) -> SessionDeletionResult:
        """Resume an operation from any non-terminal durable state."""

        operation = self._load_operation(op_id)
        with session_key_lock(self.root, operation.session_id):
            return self._recover_locked(op_id)

    def _recover_locked(self, op_id: str) -> SessionDeletionResult:
        """Recover while the caller holds the stable run-key lock."""
        with self._operation_lock(op_id):
            operation = self._load_operation(op_id)
            if operation.state == "done":
                return _result(operation)
            if operation.state == "blocked":
                raise SessionDeletionBlockedError(
                    op_id, operation.error_message or "manual recovery is required"
                )
            if operation.state == "prepared":
                self._apply_quarantine(operation)
                operation = self._load_operation(op_id)
            if operation.state == "fs_applied":
                self._commit_database_deletion(operation)
                operation = self._load_operation(op_id)
            if operation.state == "db_committed":
                self._mirror_audit(operation)
                self._fault("after_audit_mirror", op_id, None)
                self._purge_quarantine(operation)
                self._discard_sandbox_scratch(operation)
                self._mark_done(op_id)
                self._fault("after_done", op_id, None)
                operation = self._load_operation(op_id)
            if operation.state != "done":
                raise SessionDeletionError(
                    f"Run deletion {op_id} stopped in unexpected state {operation.state!r}."
                )
            return _result(operation)

    def _reserve(self, session_id: str) -> tuple[str, bool]:
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                row = conn.execute(
                    """
                    select project_id, artifact_count, storage_state, delete_op_id
                    from sessions where session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    active = self._active_operation_for_session(session_id)
                    if active is not None:
                        return active.op_id, False
                    raise SessionDeletionNotFoundError(session_id)
                project_id = str(row["project_id"])
                storage_state = str(row["storage_state"])
                delete_op_id = row["delete_op_id"]
                if storage_state == "deleting" and delete_op_id is not None:
                    conn.commit()
                    return str(delete_op_id), False
                if storage_state != "live":
                    raise SessionDeletionBlockedError(
                        str(delete_op_id or "unknown"),
                        f"run has unsupported storage state {storage_state!r}",
                    )

                active_job = conn.execute(
                    f"""
                    select job_id from jobs
                    where (session_id = ? or lane_key = ? or request_scope = ?)
                      and status not in ({_placeholders(len(_TERMINAL_JOB_STATUSES))})
                    order by created_at, job_id limit 1
                    """,
                    (session_id, session_id, session_id, *_TERMINAL_JOB_STATUSES),
                ).fetchone()
                if active_job is not None:
                    raise SessionDeletionBusyError(session_id, str(active_job["job_id"]))

                op_id = f"del_{uuid4().hex}"
                now = _now()
                # Clear terminal-job replay identity while the target is still
                # live. Once deletion is reserved, the raw jobs trigger rejects
                # every run/lane/status update, including terminal ones.
                conn.execute(
                    f"""
                    update jobs set idempotency_key = null
                    where project_id = ? and (session_id = ? or lane_key = ?)
                      and status in ({_placeholders(len(_TERMINAL_JOB_STATUSES))})
                    """,
                    (
                        project_id,
                        session_id,
                        session_id,
                        *_TERMINAL_JOB_STATUSES,
                    ),
                )
                source_paths = _source_relative_paths(project_id, session_id)
                conn.execute(
                    """
                    insert into storage_operations(
                        op_id, op_kind, resource_kind, project_id, resource_key,
                        request_key, state, created_at, updated_at
                    ) values(?, 'delete_session', 'session', ?, ?, ?, 'prepared', ?, ?)
                    """,
                    (
                        op_id,
                        project_id,
                        session_id,
                        op_id,
                        now,
                        now,
                    ),
                )
                metadata = json.dumps(
                    {"artifact_count": int(row["artifact_count"] or 0)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                for ordinal, source in enumerate(source_paths):
                    conn.execute(
                        """
                        insert into storage_operation_items(
                            op_id, ordinal, mode, source_relpath, work_relpath,
                            payload, required
                        ) values(?, ?, 'quarantine_path', ?, ?, ?, ?)
                        """,
                        (
                            op_id,
                            ordinal,
                            source,
                            quarantine_relative_path(op_id, ordinal, source),
                            metadata if ordinal == 0 else None,
                            1 if ordinal == 0 else 0,
                        ),
                    )
                updated = conn.execute(
                    """
                    update sessions
                    set storage_state = 'deleting', delete_op_id = ?,
                        state_version = state_version + 1, updated_at = ?
                    where session_id = ? and project_id = ? and storage_state = 'live'
                    """,
                    (op_id, now, session_id, project_id),
                )
                if updated.rowcount != 1:
                    raise SessionDeletionRetryableError(
                        op_id, "run changed while deletion was reserved"
                    )
                conn.commit()
                return op_id, True
            except Exception:
                conn.rollback()
                raise

    def _apply_quarantine(self, operation: _DeletionOperation) -> None:
        items = self._load_items(operation)
        try:
            for item in items:
                source = self._safe_path(item.source_relative_path)
                work = self._safe_path(item.work_relative_path, create_parent=True)
                source_exists = os.path.lexists(source)
                work_exists = os.path.lexists(work)
                if source_exists and source.is_symlink():
                    self._block(operation.op_id, f"source is a symlink: {source.name}")
                if work_exists and work.is_symlink():
                    self._block(operation.op_id, f"quarantine is a symlink: {work.name}")
                if source_exists and work_exists:
                    self._block(
                        operation.op_id,
                        f"source and quarantine both exist for item {item.ordinal}",
                    )
                if source_exists:
                    self._validate_item_type(operation.op_id, item.ordinal, source)
                    os.replace(source, work)
                    fsync_directory(source.parent)
                    fsync_directory(work.parent)
                    self._fault("after_quarantine_item", operation.op_id, item.ordinal)
                elif work_exists:
                    self._validate_item_type(operation.op_id, item.ordinal, work)
                elif item.required:
                    self._block(
                        operation.op_id,
                        f"required item {item.ordinal} is missing from source and quarantine",
                    )
            self._fault("after_quarantine_before_state", operation.op_id, None)
        except SessionDeletionBlockedError as exc:
            if exc.op_id != operation.op_id:
                self._block(operation.op_id, exc.reason)
            raise
        except OSError as exc:
            raise SessionDeletionRetryableError(operation.op_id, str(exc)) from exc

        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                updated = conn.execute(
                    """
                    update storage_operations
                    set state = 'fs_applied', error_code = null,
                        error_message = null, updated_at = ?
                    where op_id = ? and state = 'prepared'
                    """,
                    (_now(), operation.op_id),
                )
                if updated.rowcount != 1:
                    raise SessionDeletionRetryableError(
                        operation.op_id,
                        "operation changed while quarantine was applied",
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _commit_database_deletion(self, operation: _DeletionOperation) -> None:
        self._assert_sources_absent(operation)
        items = self._load_items(operation)
        try:
            artifact_count = _artifact_count(items)
        except SessionDeletionBlockedError as exc:
            self._block(operation.op_id, exc.reason)
        event_key = _audit_event_key(operation.op_id)
        event = _audit_event(operation, artifact_count, event_key)
        payload = event.model_dump_json()
        self._fault("before_db_commit", operation.op_id, None)

        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                current = conn.execute(
                    """
                    select storage_state, delete_op_id from sessions
                    where session_id = ? and project_id = ?
                    """,
                    (operation.session_id, operation.project_id),
                ).fetchone()
                if current is None:
                    state = conn.execute(
                        "select state from storage_operations where op_id = ?",
                        (operation.op_id,),
                    ).fetchone()
                    if state is not None and str(state["state"]) == "db_committed":
                        conn.commit()
                        return
                    raise SessionDeletionRetryableError(
                        operation.op_id, "run row vanished before delete commit"
                    )
                if (
                    str(current["storage_state"]) != "deleting"
                    or str(current["delete_op_id"]) != operation.op_id
                ):
                    raise SessionDeletionBlockedError(
                        operation.op_id,
                        "run deletion ownership no longer matches the operation",
                    )

                active_job = conn.execute(
                    f"""
                    select job_id from jobs
                    where (session_id = ? or lane_key = ? or request_scope = ?)
                      and status not in ({_placeholders(len(_TERMINAL_JOB_STATUSES))})
                    order by created_at, job_id limit 1
                    """,
                    (
                        operation.session_id,
                        operation.session_id,
                        operation.session_id,
                        *_TERMINAL_JOB_STATUSES,
                    ),
                ).fetchone()
                if active_job is not None:
                    raise SessionDeletionBusyError(operation.session_id, str(active_job["job_id"]))

                conn.execute(
                    "delete from artifacts where project_id = ? and session_id = ?",
                    (operation.project_id, operation.session_id),
                )
                conn.execute(
                    "delete from trace_events where project_id = ? and session_id = ?",
                    (operation.project_id, operation.session_id),
                )
                conn.execute(
                    "delete from pending_actions where project_id = ? and session_id = ?",
                    (operation.project_id, operation.session_id),
                )
                self._purge_page_indexes(conn, operation)
                conn.execute(
                    """
                    insert into trace_events(
                        session_id, project_id, event_type, name, payload, event_key
                    ) values(?, ?, ?, ?, ?, ?)
                    on conflict(event_key) where event_key is not null do nothing
                    """,
                    (
                        AUDIT_SESSION_ID,
                        operation.project_id,
                        event.event_type,
                        event.name,
                        payload,
                        event_key,
                    ),
                )
                audit = conn.execute(
                    "select payload from trace_events where event_key = ?",
                    (event_key,),
                ).fetchone()
                if audit is None or str(audit["payload"]) != payload:
                    raise SessionDeletionBlockedError(
                        operation.op_id, "audit event key is bound to different content"
                    )
                removed = conn.execute(
                    """
                    delete from sessions
                    where session_id = ? and project_id = ?
                      and storage_state = 'deleting' and delete_op_id = ?
                    """,
                    (operation.session_id, operation.project_id, operation.op_id),
                )
                if removed.rowcount != 1:
                    raise SessionDeletionRetryableError(operation.op_id, "run row was not deleted")
                advanced = conn.execute(
                    """
                    update storage_operations
                    set state = 'db_committed', error_code = null,
                        error_message = null, updated_at = ?
                    where op_id = ? and state = 'fs_applied'
                    """,
                    (_now(), operation.op_id),
                )
                if advanced.rowcount != 1:
                    raise SessionDeletionRetryableError(
                        operation.op_id, "operation did not reach db_committed"
                    )
                self._fault("after_db_changes_before_commit", operation.op_id, None)
                self._assert_sources_absent(operation, persist=False)
                conn.commit()
            except SessionDeletionBlockedError as exc:
                conn.rollback()
                self._block(operation.op_id, exc.reason)
            except Exception:
                conn.rollback()
                raise
        self._fault("after_db_commit", operation.op_id, None)

    def _mirror_audit(self, operation: _DeletionOperation) -> None:
        event_key = _audit_event_key(operation.op_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select payload from trace_events where event_key = ?",
                (event_key,),
            ).fetchone()
        if row is None:
            self._block(operation.op_id, "committed deletion has no audit event")
        payload = str(row["payload"])
        try:
            sqlite_canonical = _canonical_trace_event_payload(payload)
        except ValueError as exc:
            self._block(operation.op_id, f"committed audit event is invalid: {exc}")

        trace_relative = str(
            PurePosixPath("projects")
            / operation.project_id
            / "sessions"
            / AUDIT_SESSION_ID
            / "trace.jsonl"
        )
        try:
            trace_path = self._safe_path(trace_relative, create_parent=True)
        except SessionDeletionBlockedError as exc:
            self._block(operation.op_id, exc.reason)
        if os.path.lexists(trace_path) and trace_path.is_symlink():
            self._block(operation.op_id, "audit trace path is a symlink")
        audit_lock_relative = str(
            PurePosixPath(".storage-operations") / "locks" / f"audit-{operation.project_id}.lock"
        )
        try:
            with self._file_lock(audit_lock_relative, operation.op_id, "audit mirror lock"):
                existing = trace_path.read_bytes() if trace_path.exists() else b""
                if existing and not existing.endswith(b"\n"):
                    self._block(operation.op_id, "audit trace has an incomplete tail")
                matching = _jsonl_event_payloads(existing, event_key)
                if len(matching) > 1:
                    self._block(
                        operation.op_id,
                        "audit trace contains duplicate deletion events",
                    )
                if matching:
                    if matching[0] != sqlite_canonical:
                        self._block(
                            operation.op_id,
                            "audit trace event key is bound to different content",
                        )
                    return
                rendered = existing + payload.encode("utf-8") + b"\n"
                temporary = trace_path.with_name(f".{trace_path.name}.{operation.op_id}.tmp")
                if os.path.lexists(temporary) and temporary.is_symlink():
                    self._block(operation.op_id, "audit trace staging path is a symlink")
                with temporary.open("wb") as handle:
                    handle.write(rendered)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, trace_path)
                fsync_directory(trace_path.parent)
        except SessionDeletionBlockedError as exc:
            if exc.op_id != operation.op_id:
                self._block(operation.op_id, exc.reason)
            raise
        except OSError as exc:
            raise SessionDeletionRetryableError(operation.op_id, str(exc)) from exc

    def _purge_page_indexes(self, conn: sqlite3.Connection, operation: _DeletionOperation) -> None:
        """Drop the run's cached pagination projections.

        These tables are keyed by run scope and by workspace-relative file path,
        neither of which any other cleanup touches, so without this a deleted
        run leaves its full projection behind forever.
        """
        scopes = run_resource_scopes(operation.project_id, operation.session_id)
        # The index classes create their tables lazily, so a workspace that has
        # never served a page legitimately has none of them.
        present = {
            str(row[0])
            for row in conn.execute(
                """
                select name from sqlite_master
                where type = 'table' and name like '%_page_%'
                """
            ).fetchall()
        }
        placeholders = ",".join("?" for _ in scopes)
        for table in ("resource_page_entries", "resource_page_sources"):
            if table not in present:
                continue
            conn.execute(
                f"delete from {table} where scope in ({placeholders})",  # noqa: S608
                scopes,
            )
        # JsonlPageIndex keys by the workspace-relative path, which is exactly
        # what the deletion item set already enumerates.
        run_relpath, chat_relpath, _results_relpath = _source_relative_paths(
            operation.project_id, operation.session_id
        )
        path_keys = [
            jsonl_path_key(PurePosixPath(run_relpath) / LLM_DEBUG_FILENAME),
            jsonl_path_key(chat_relpath),
        ]
        placeholders = ",".join("?" for _ in path_keys)
        for table in ("jsonl_page_entries", "jsonl_page_sources"):
            if table not in present:
                continue
            conn.execute(
                f"delete from {table} where path_key in ({placeholders})",  # noqa: S608
                path_keys,
            )

    def _purge_quarantine(self, operation: _DeletionOperation) -> None:
        try:
            items = self._load_items(operation)
            for item in items:
                work = self._safe_path(item.work_relative_path)
                if not os.path.lexists(work):
                    continue
                if work.is_symlink():
                    self._block(operation.op_id, "quarantine path became a symlink")
                self._validate_item_type(operation.op_id, item.ordinal, work)
                if work.is_dir():
                    remove_tree(work)
                else:
                    work.unlink()
                fsync_directory(work.parent)
                self._fault("after_purge_item", operation.op_id, item.ordinal)
            if items:
                operation_dir = self._safe_path(items[0].work_relative_path).parent
                if operation_dir.is_dir() and not any(operation_dir.iterdir()):
                    operation_dir.rmdir()
                    fsync_directory(operation_dir.parent)
        except SessionDeletionBlockedError as exc:
            if exc.op_id != operation.op_id:
                self._block(operation.op_id, exc.reason)
            raise
        except OSError as exc:
            raise SessionDeletionRetryableError(operation.op_id, str(exc)) from exc

    def _discard_sandbox_scratch(self, operation: _DeletionOperation) -> None:
        """Drop the run's code-agent scratch tree, best effort.

        Deliberately not a quarantine item: a late tool process may recreate
        this directory, and `_assert_sources_absent` would then block a
        deletion that has already committed. A stray scratch tree is harmless;
        a blocked deletion is not.
        """
        scratch = self._safe_path(
            str(
                PurePosixPath("_sandbox")
                / "code_agent"
                / operation.project_id
                / operation.session_id
            )
        )
        if scratch.is_symlink() or not scratch.is_dir():
            return
        remove_tree(scratch, ignore_errors=True)

    def _mark_done(self, op_id: str) -> None:
        operation = self._load_operation(op_id)
        self._assert_sources_absent(operation)
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                updated = conn.execute(
                    """
                    update storage_operations
                    set state = 'done', error_code = null, error_message = null,
                        updated_at = ?
                    where op_id = ? and state = 'db_committed'
                    """,
                    (_now(), op_id),
                )
                if updated.rowcount != 1:
                    raise SessionDeletionRetryableError(op_id, "operation did not reach done")
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _assert_sources_absent(
        self,
        operation: _DeletionOperation,
        *,
        persist: bool = True,
    ) -> None:
        for source_relative in _source_relative_paths(operation.project_id, operation.session_id):
            source = self._safe_path(source_relative)
            if os.path.lexists(source):
                reason = f"source reappeared during deletion: {source_relative}"
                if persist:
                    self._block(operation.op_id, reason)
                raise SessionDeletionBlockedError(operation.op_id, reason)

    def _active_operation_for_session(self, session_id: str) -> _DeletionOperation | None:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                select op_id, project_id, resource_key, state, error_message
                from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = ?
                  and state in ({_placeholders(len(_ACTIVE_OPERATION_STATES))})
                order by created_at, op_id limit 2
                """,
                (session_id, *_ACTIVE_OPERATION_STATES),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise SessionDeletionError(f"Run {session_id} has multiple active delete operations.")
        return _operation_from_row(rows[0])

    def _load_operation(self, op_id: str) -> _DeletionOperation:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select op_id, project_id, resource_key, state, error_message
                from storage_operations
                where op_id = ? and op_kind = 'delete_session'
                  and resource_kind = 'session'
                """,
                (op_id,),
            ).fetchone()
        if row is None:
            raise SessionDeletionNotFoundError(op_id)
        return _operation_from_row(row)

    def _load_items(self, operation: _DeletionOperation) -> tuple[_DeletionItem, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select ordinal, source_relpath, work_relpath, payload, required
                from storage_operation_items
                where op_id = ? and mode = 'quarantine_path'
                order by ordinal
                """,
                (operation.op_id,),
            ).fetchall()
        expected_sources = _source_relative_paths(operation.project_id, operation.session_id)
        if len(rows) != len(expected_sources):
            self._block(operation.op_id, "delete operation item set is incomplete")
        items: list[_DeletionItem] = []
        for row, expected_source in zip(rows, expected_sources, strict=True):
            ordinal = int(row["ordinal"])
            source = str(row["source_relpath"])
            expected_work = quarantine_relative_path(operation.op_id, ordinal, expected_source)
            raw_required = row["required"]
            expected_required = ordinal == 0
            if (
                ordinal != len(items)
                or source != expected_source
                or str(row["work_relpath"]) != expected_work
                or raw_required not in (0, 1)
                or bool(raw_required) != expected_required
            ):
                self._block(
                    operation.op_id,
                    "delete operation contains an unexpected path",
                )
            raw_payload = row["payload"]
            items.append(
                _DeletionItem(
                    ordinal=ordinal,
                    source_relative_path=source,
                    work_relative_path=expected_work,
                    payload=(None if raw_payload is None else bytes(raw_payload)),
                    required=expected_required,
                )
            )
        return tuple(items)

    def _block(self, op_id: str, reason: str) -> NoReturn:
        with closing(self._connect()) as conn:
            self._begin_immediate(conn)
            try:
                conn.execute(
                    """
                    update storage_operations
                    set state = case
                            when state = 'db_committed' then state
                            else 'blocked'
                        end,
                        error_code = 'unsafe_state',
                        error_message = ?, updated_at = ?
                    where op_id = ? and state not in ('done', 'aborted')
                    """,
                    (reason[:1000], _now(), op_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        raise SessionDeletionBlockedError(op_id, reason)

    def _safe_path(self, relative_path: str, *, create_parent: bool = False) -> Path:
        normalized = _normalize_relative_path(relative_path)
        candidate = self.root.joinpath(*PurePosixPath(normalized).parts)
        parent = candidate.parent
        if create_parent:
            _create_safe_directories(self.root, parent)
        else:
            _assert_safe_parents(self.root, parent)
        resolved_parent = parent.resolve(strict=False)
        if resolved_parent != self.root and not resolved_parent.is_relative_to(self.root):
            raise SessionDeletionBlockedError("unknown", f"path escapes workspace: {normalized}")
        return candidate

    def _validate_item_type(self, op_id: str, ordinal: int, path: Path) -> None:
        expected = _SOURCE_ITEM_KINDS[ordinal]
        valid = path.is_dir() if expected == "directory" else path.is_file()
        if not valid:
            self._block(op_id, f"item {ordinal} is not the expected {expected}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 5000")
        conn.execute("pragma foreign_keys = on")
        return conn

    @staticmethod
    def _begin_immediate(conn: sqlite3.Connection) -> None:
        conn.execute("begin immediate")

    @contextmanager
    def _operation_lock(self, op_id: str) -> Iterator[None]:
        if not op_id or any(not (char.isalnum() or char in "_-") for char in op_id):
            raise SessionDeletionNotFoundError(op_id)
        relative = str(PurePosixPath(".storage-operations") / "locks" / f"{op_id}.lock")
        with self._file_lock(relative, op_id, "operation lock"):
            yield

    @contextmanager
    def _file_lock(self, relative_path: str, op_id: str, label: str) -> Iterator[None]:
        try:
            lock_path = self._safe_path(relative_path, create_parent=True)
        except SessionDeletionBlockedError as exc:
            self._block(op_id, exc.reason)
        if os.path.lexists(lock_path) and lock_path.is_symlink():
            self._block(op_id, f"{label} is a symlink")
        try:
            with lock_path.open("a+b") as handle:
                lock_exclusive(handle.fileno())
                try:
                    yield
                finally:
                    unlock(handle.fileno())
        except OSError as exc:
            raise SessionDeletionRetryableError(op_id, str(exc)) from exc

    def _fault(self, stage: str, op_id: str, ordinal: int | None) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage, op_id, ordinal)


# Positional, matching _source_relative_paths: run media, chat transcript,
# async job results. Item 0 stays the only required one.
_SOURCE_ITEM_KINDS = ("directory", "file", "directory")


def _source_relative_paths(project_id: str, session_id: str) -> tuple[str, str, str]:
    # Values are normalized again before use. Keeping the construction here
    # makes the item set deterministic and independently verifiable on resume.
    return (
        str(PurePosixPath("projects") / project_id / "sessions" / session_id),
        str(PurePosixPath("projects") / project_id / "chat" / f"{session_id}.jsonl"),
        session_results_relative_path(project_id, session_id),
    )


def _operation_from_row(row: sqlite3.Row) -> _DeletionOperation:
    return _DeletionOperation(
        op_id=str(row["op_id"]),
        project_id=str(row["project_id"]),
        session_id=str(row["resource_key"]),
        state=str(row["state"]),
        error_message=(None if row["error_message"] is None else str(row["error_message"])),
    )


def _result(operation: _DeletionOperation) -> SessionDeletionResult:
    return SessionDeletionResult(
        session_id=operation.session_id,
        project_id=operation.project_id,
        op_id=operation.op_id,
    )


def _artifact_count(items: tuple[_DeletionItem, ...]) -> int:
    if not items or items[0].payload is None:
        return 0
    try:
        payload = json.loads(items[0].payload)
        return int(payload.get("artifact_count", 0))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionDeletionBlockedError(
            "unknown", "delete operation metadata is invalid"
        ) from exc


def _audit_event_key(op_id: str) -> str:
    return f"run-delete:{op_id}"


def _audit_event(
    operation: _DeletionOperation,
    artifact_count: int,
    event_key: str,
) -> TraceEvent:
    return TraceEvent(
        session_id=AUDIT_SESSION_ID,
        event_type=_AUDIT_EVENT_TYPE,
        name=operation.session_id,
        finished_at=datetime.now(UTC),
        summary={
            "session_id": operation.session_id,
            "project_id": operation.project_id,
            "deleted": True,
            "artifact_count": artifact_count,
            "storage_operation_id": operation.op_id,
            "event_key": event_key,
        },
    )


def _canonical_trace_event_payload(payload: bytes | str) -> bytes:
    event = TraceEvent.model_validate_json(payload)
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _jsonl_event_payloads(payload: bytes, event_key: str) -> tuple[bytes, ...]:
    matching: list[bytes] = []
    for raw_line in payload.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = TraceEvent.model_validate_json(raw_line)
        except ValueError as exc:
            raise SessionDeletionBlockedError(
                "unknown", "audit trace contains invalid JSON"
            ) from exc
        if event.summary.get("event_key") == event_key:
            matching.append(
                json.dumps(
                    event.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return tuple(matching)


def _normalize_relative_path(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise SessionDeletionBlockedError("unknown", "path must be a non-empty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SessionDeletionBlockedError("unknown", "path contains an unsafe segment")
    return str(path)


def _assert_safe_parents(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as exc:
        raise SessionDeletionBlockedError("unknown", "path escapes workspace") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise SessionDeletionBlockedError(
                "unknown", f"path parent is a symlink: {current.name}"
            )
        if current.exists() and not current.is_dir():
            raise SessionDeletionBlockedError(
                "unknown", f"path parent is not a directory: {current.name}"
            )


def _create_safe_directories(root: Path, parent: Path) -> None:
    _assert_safe_parents(root, parent)
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if not current.exists():
            current.mkdir()
            fsync_directory(current.parent)


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _now() -> str:
    return datetime.now(UTC).isoformat()

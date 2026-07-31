"""Durable, CAS-based lifecycle operations for local worker processes."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from eda_platform.core.process_control import (
    force_kill,
    pid_is_alive,
    request_stop,
)
from eda_platform.core.process_identity import (
    ProcessIdentity,
    ProcessIdentityComparison,
    check_process_identity,
)
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.store import ArtifactStore, SessionStorageDeletingError
from eda_platform.schemas.sessions import TraceEvent

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
DEFAULT_LEASE_SECONDS = 30
DEFAULT_HEARTBEAT_SECONDS = 5.0


@dataclass(frozen=True)
class LaunchClaim:
    job_id: str
    token: str
    attempt: int


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def serialize_process_identity(identity: ProcessIdentity) -> str:
    return json.dumps(
        {
            "pid": identity.pid,
            "source": identity.source,
            "start_token": identity.start_token,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_process_identity(value: object) -> ProcessIdentity | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        payload = json.loads(value)
        return ProcessIdentity(
            pid=int(payload["pid"]),
            source=payload["source"],
            start_token=str(payload["start_token"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _wait_identity_exit(expected: ProcessIdentity, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        comparison = check_process_identity(expected)
        if comparison is ProcessIdentityComparison.MISMATCH:
            return True
        if comparison is ProcessIdentityComparison.UNKNOWN and not pid_is_alive(
            expected.pid
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


class JobLifecycleRepository:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.root = store.root
        self.db_path = store.db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                columns = {
                    str(row[1])
                    for row in conn.execute("pragma table_info(jobs)").fetchall()
                }
                if "params_json" not in columns:
                    conn.execute("alter table jobs add column params_json text")
                if "launch_ack_at" not in columns:
                    conn.execute("alter table jobs add column launch_ack_at text")
                if "critical_depth" not in columns:
                    conn.execute(
                        "alter table jobs add column critical_depth integer not null default 0"
                    )
                if "critical_owner_generation" not in columns:
                    conn.execute(
                        "alter table jobs add column critical_owner_generation integer"
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def create_queued_job(
        self,
        *,
        job_id: str,
        session_id: str,
        project_id: str,
        kind: str,
        params_json: str,
        idempotency_key: str | None,
        lane_key: str,
        request_digest: str,
        request_scope: str,
    ) -> dict:
        """Reserve the F-020 lane and append job.queued in one transaction."""
        with self._locked_job_targets(session_id, lane_key), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                deleting = self._deleting_target(conn, session_id, lane_key)
                if deleting is not None:
                    raise SessionStorageDeletingError(deleting)
                for target in dict.fromkeys((session_id, lane_key)):
                    tombstone = conn.execute(
                        """
                        select 1 from storage_operations
                        where op_kind = 'delete_session' and resource_kind = 'session'
                          and resource_key = ? and state != 'aborted'
                        limit 1
                        """,
                        (target,),
                    ).fetchone()
                    row = conn.execute(
                        "select project_id, storage_state from sessions where session_id = ?",
                        (target,),
                    ).fetchone()
                    if tombstone is not None or (
                        target == lane_key
                        and lane_key != session_id
                        and row is not None
                        and str(row[1]) != "live"
                    ):
                        raise SessionStorageDeletingError(target)
                conn.execute(
                    """
                    insert into sessions(session_id, project_id, path, status)
                    values(?, ?, ?, 'running')
                    on conflict(session_id) do nothing
                    """,
                    (
                        session_id,
                        project_id,
                        str(Path("projects") / project_id / "sessions" / session_id),
                    ),
                )
                created_at = _iso()
                conn.execute(
                    """
                    insert into jobs(
                        job_id, session_id, project_id, kind, status, created_at,
                        idempotency_key, lane_key, request_digest, request_scope,
                        params_json
                    ) values(?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        session_id,
                        project_id,
                        kind,
                        created_at,
                        idempotency_key,
                        lane_key,
                        request_digest,
                        request_scope,
                        params_json,
                    ),
                )
                # A pre-F022 writer may have terminalized the previous job
                # without releasing the run owner. Reconcile only a terminal
                # owner, in the same reservation transaction.
                conn.execute(
                    """
                    update sessions set active_job_id = null,
                        state_version = state_version + 1
                    where session_id = ? and active_job_id in (
                        select job_id from jobs
                        where session_id = ? and status in ('completed', 'failed', 'cancelled')
                    )
                    """,
                    (session_id, session_id),
                )
                owned = conn.execute(
                    """
                    update sessions set active_job_id = ?, state_version = state_version + 1
                    where session_id = ? and project_id = ? and storage_state = 'live'
                      and active_job_id is null
                    """,
                    (job_id, session_id, project_id),
                )
                if owned.rowcount != 1:
                    raise sqlite3.IntegrityError("run already has an active job owner")
                self._insert_event(
                    conn,
                    job_id=job_id,
                    session_id=session_id,
                    project_id=project_id,
                    generation=0,
                    event_type="job.queued",
                    status="queued",
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    def claim_launch(
        self,
        job_id: str,
        *,
        owner: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> LaunchClaim:
        job = self._require_job(job_id)
        with self._locked_job(job), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                token = uuid4().hex
                now = _now()
                cursor = conn.execute(
                    """
                    update jobs set
                        status = 'launching',
                        state_version = state_version + 1,
                        launch_attempt = launch_attempt + 1,
                        launch_token = ?,
                        lease_owner = ?,
                        lease_expires_at = ?,
                        heartbeat_at = ?,
                        pid = null,
                        pid_start_identity = null,
                        launch_ack_at = null
                    where job_id = ? and status = 'queued'
                      and cancel_requested = 0
                    """,
                    (
                        token,
                        owner,
                        _iso(now + timedelta(seconds=lease_seconds)),
                        _iso(now),
                        job_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Job is not launchable: {job_id}")
                attempt = int(
                    conn.execute(
                        "select launch_attempt from jobs where job_id = ?", (job_id,)
                    ).fetchone()[0]
                )
                conn.commit()
                return LaunchClaim(job_id=job_id, token=token, attempt=attempt)
            except Exception:
                conn.rollback()
                raise

    def acknowledge_spawn(
        self,
        claim: LaunchClaim,
        *,
        pid: int,
        birth_identity: str,
    ) -> None:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update jobs set
                    pid = ?, pid_start_identity = ?, launch_ack_at = ?,
                    heartbeat_at = ?, state_version = state_version + 1
                where job_id = ? and status = 'launching'
                  and launch_token = ? and launch_attempt = ?
                """,
                (
                    pid,
                    birth_identity,
                    _iso(),
                    _iso(),
                    claim.job_id,
                    claim.token,
                    claim.attempt,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Launch acknowledgement lost its CAS claim.")

    def child_start(self, claim: LaunchClaim) -> dict | None:
        job = self._require_job(claim.job_id)
        with self._locked_job(job), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                row = conn.execute(
                    """
                    select cancel_requested, project_id, session_id
                    from jobs
                    where job_id = ? and status in ('launching', 'cancelling')
                      and launch_token = ? and launch_attempt = ?
                      and launch_ack_at is not null
                    """,
                    (claim.job_id, claim.token, claim.attempt),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                if bool(row[0]):
                    self._terminal_in_transaction(
                        conn,
                        claim=claim,
                        status="cancelled",
                        error_code=None,
                        error_message=None,
                    )
                    conn.commit()
                    return None
                now = _now()
                cursor = conn.execute(
                    """
                    update jobs set
                        status = 'running',
                        state_version = state_version + 1,
                        started_at = coalesce(started_at, ?),
                        heartbeat_at = ?,
                        lease_expires_at = ?
                    where job_id = ? and status = 'launching'
                      and launch_token = ? and launch_attempt = ?
                    """,
                    (
                        _iso(now),
                        _iso(now),
                        _iso(now + timedelta(seconds=DEFAULT_LEASE_SECONDS)),
                        claim.job_id,
                        claim.token,
                        claim.attempt,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    return None
                self._insert_event(
                    conn,
                    job_id=claim.job_id,
                    session_id=str(row[2]),
                    project_id=str(row[1]),
                    generation=claim.attempt,
                    event_type="job.started",
                    status="running",
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return self.store.get_job(claim.job_id)

    def materialize_trace(self, job_id: str) -> None:
        """Rebuild trace.jsonl from DB truth; atomic replace makes recovery exact-once."""
        job = self._require_job(job_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select payload from trace_events
                where project_id = ? and session_id = ?
                order by id
                """,
                (str(job["project_id"]), str(job["session_id"])),
            ).fetchall()
        body = "".join(f"{str(row[0]).rstrip()}\n" for row in rows)
        self.store.write_session_text(
            str(job["project_id"]),
            str(job["session_id"]),
            "trace.jsonl",
            body,
        )

    def heartbeat(
        self,
        claim: LaunchClaim,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        now = _now()
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update jobs set heartbeat_at = ?, lease_expires_at = ?
                where job_id = ? and status in ('running', 'cancelling')
                  and launch_token = ? and launch_attempt = ?
                """,
                (
                    _iso(now),
                    _iso(now + timedelta(seconds=lease_seconds)),
                    claim.job_id,
                    claim.token,
                    claim.attempt,
                ),
            )
        return cursor.rowcount == 1

    def enter_critical(self, claim: LaunchClaim) -> bool:
        """Durably enter/nest a publish shield owned by one launch generation."""
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                nested = conn.execute(
                    """
                    update jobs set critical_depth = critical_depth + 1,
                        state_version = state_version + 1
                    where job_id = ? and launch_token = ? and launch_attempt = ?
                      and status in ('running', 'cancelling')
                      and critical_depth > 0 and critical_depth < 1000000
                      and critical_owner_generation = ?
                      and kill_fence_state = 'open'
                    """,
                    (
                        claim.job_id,
                        claim.token,
                        claim.attempt,
                        claim.attempt,
                    ),
                )
                if nested.rowcount == 1:
                    conn.commit()
                    return True
                outer = conn.execute(
                    """
                    update jobs set critical_depth = 1,
                        critical_owner_generation = ?,
                        state_version = state_version + 1
                    where job_id = ? and launch_token = ? and launch_attempt = ?
                      and status = 'running' and cancel_requested = 0
                      and critical_depth = 0
                      and critical_owner_generation is null
                      and kill_fence_state = 'open'
                    """,
                    (
                        claim.attempt,
                        claim.job_id,
                        claim.token,
                        claim.attempt,
                    ),
                )
                conn.commit()
                return outer.rowcount == 1
            except Exception:
                conn.rollback()
                raise

    def exit_critical(self, claim: LaunchClaim) -> bool:
        """Release only the same generation's shield; stale owners fail closed."""
        with closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                cursor = conn.execute(
                    """
                    update jobs set
                        critical_depth = critical_depth - 1,
                        critical_owner_generation = case
                            when critical_depth = 1 then null
                            else critical_owner_generation
                        end,
                        state_version = state_version + 1
                    where job_id = ? and launch_token = ? and launch_attempt = ?
                      and critical_depth > 0
                      and critical_owner_generation = ?
                      and kill_fence_state = 'open'
                    """,
                    (
                        claim.job_id,
                        claim.token,
                        claim.attempt,
                        claim.attempt,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1
            except Exception:
                conn.rollback()
                raise

    def finish(
        self,
        claim: LaunchClaim,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Invalid terminal status: {status}")
        job = self._require_job(claim.job_id)
        with self._locked_job(job), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                changed = self._terminal_in_transaction(
                    conn,
                    claim=claim,
                    status=status,
                    error_code=error_code,
                    error_message=error_message,
                )
                conn.commit()
                return changed
            except Exception:
                conn.rollback()
                raise

    def fail_active(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        clear_idempotency: bool = False,
    ) -> bool:
        job = self._require_job(job_id)
        with self._locked_job(job), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                row = conn.execute(
                    "select status, launch_token, launch_attempt from jobs where job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None or str(row[0]) in TERMINAL_STATUSES:
                    conn.rollback()
                    return False
                status = str(row[0])
                # Adopt a legacy, unowned active row before using the strict
                # terminal handshake. Never steal an owner from another job.
                if row[1] is None:
                    conn.execute(
                        """
                        update sessions set active_job_id = ?,
                            state_version = state_version + 1
                        where session_id = ? and active_job_id is null
                          and storage_state = 'live'
                        """,
                        (job_id, str(job["session_id"])),
                    )
                if status == "cancelling":
                    terminal = "cancelled"
                else:
                    terminal = "failed"
                now = _iso()
                cursor = conn.execute(
                    """
                    update jobs set
                        status = ?, state_version = state_version + 1,
                        finished_at = ?, error_code = ?, error_message = ?,
                        lease_expires_at = null,
                        critical_depth = 0, critical_owner_generation = null,
                        idempotency_key = case when ? then null else idempotency_key end
                    where job_id = ? and status = ?
                    """,
                    (
                        terminal,
                        now,
                        error_code,
                        error_message[:500],
                        int(clear_idempotency),
                        job_id,
                        status,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    return False
                self._update_run_terminal(
                    conn,
                    job_id=job_id,
                    session_id=str(job["session_id"]),
                    status=terminal,
                )
                self._insert_event(
                    conn,
                    job_id=job_id,
                    session_id=str(job["session_id"]),
                    project_id=str(job["project_id"]),
                    generation=int(row[2]),
                    event_type=f"job.{terminal}",
                    status=terminal,
                    summary={"error_code": error_code, "error_message": error_message[:500]},
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def request_cancel(self, job_id: str, *, grace_seconds: int = 2) -> dict:
        job = self._require_job(job_id)
        with self._locked_job(job), closing(self._connect()) as conn:
            conn.execute("begin immediate")
            try:
                current = conn.execute(
                    """
                    select status, launch_attempt, project_id, session_id
                    from jobs where job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if current is None:
                    raise KeyError(f"Job not found: {job_id}")
                status = str(current[0])
                generation = int(current[1])
                if status in TERMINAL_STATUSES:
                    conn.commit()
                    return self._require_job(job_id)
                now = _now()
                if status == "queued":
                    cursor = conn.execute(
                        """
                        update jobs set
                            cancel_requested = 1, cancel_requested_at = ?,
                            cancel_deadline_at = ?, status = 'cancelled',
                            state_version = state_version + 1, finished_at = ?
                        where job_id = ? and status = 'queued'
                        """,
                        (_iso(now), _iso(now), _iso(now), job_id),
                    )
                    terminal = cursor.rowcount == 1
                    if terminal:
                        self._update_run_terminal(
                            conn,
                            job_id=job_id,
                            session_id=str(current[3]),
                            status="cancelled",
                        )
                else:
                    cursor = conn.execute(
                        """
                        update jobs set
                            cancel_requested = 1, cancel_requested_at = ?,
                            cancel_deadline_at = ?, status = 'cancelling',
                            state_version = state_version + 1
                        where job_id = ? and status in ('launching', 'running')
                        """,
                        (
                            _iso(now),
                            _iso(now + timedelta(seconds=grace_seconds)),
                            job_id,
                        ),
                    )
                    terminal = False
                if cursor.rowcount == 1:
                    self._insert_event(
                        conn,
                        job_id=job_id,
                        session_id=str(current[3]),
                        project_id=str(current[2]),
                        generation=generation,
                        event_type=(
                            "job.cancelled" if terminal else "job.cancel_requested"
                        ),
                        status="cancelled" if terminal else "cancelling",
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return self._require_job(job_id)

    def cancellation_claim_due(self, job_id: str) -> LaunchClaim | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select launch_token, launch_attempt, cancel_deadline_at
                from jobs where job_id = ? and status = 'cancelling'
                  and cancel_requested = 1
                """,
                (job_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        deadline = _parse_time(row[2])
        if deadline is None or deadline > _now():
            return None
        return LaunchClaim(job_id=job_id, token=str(row[0]), attempt=int(row[1]))

    def cancellation_blocked_by_critical(self, claim: LaunchClaim) -> bool:
        """Whether the same due cancellation is waiting only on its shield."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                select 1 from jobs
                where job_id = ? and launch_token = ? and launch_attempt = ?
                  and status = 'cancelling' and cancel_requested = 1
                  and cancel_deadline_at <= ?
                  and critical_depth > 0
                  and critical_owner_generation = ?
                  and kill_fence_state = 'open'
                """,
                (
                    claim.job_id,
                    claim.token,
                    claim.attempt,
                    _iso(),
                    claim.attempt,
                ),
            ).fetchone()
        return row is not None

    def authorize_signal(
        self, job_id: str, *, claim: LaunchClaim | None = None
    ) -> tuple[int, str] | None:
        """Commit the kill fence only when PID and birth identity still match."""
        job = self._require_job(job_id)
        pid = job.get("pid")
        identity = job.get("pid_start_identity")
        expected = deserialize_process_identity(identity)
        if (
            expected is None
            or pid is None
            or expected.pid != int(pid)
            or check_process_identity(expected) is not ProcessIdentityComparison.MATCH
        ):
            return None
        with closing(self._connect()) as conn, conn:
            if claim is None:
                cursor = conn.execute(
                    """
                    update jobs set kill_fence_state = 'committed'
                    where job_id = ? and pid = ? and pid_start_identity = ?
                      and kill_fence_state = 'open'
                      and critical_depth = 0
                      and critical_owner_generation is null
                    """,
                    (job_id, int(pid), str(identity)),
                )
            else:
                cursor = conn.execute(
                    """
                    update jobs set kill_fence_state = 'committed'
                    where job_id = ? and pid = ? and pid_start_identity = ?
                      and kill_fence_state = 'open'
                      and status = 'cancelling' and cancel_requested = 1
                      and launch_token = ? and launch_attempt = ?
                      and critical_depth = 0
                      and critical_owner_generation is null
                      and cancel_deadline_at <= ?
                    """,
                    (
                        job_id,
                        int(pid),
                        str(identity),
                        claim.token,
                        claim.attempt,
                        _iso(),
                    ),
                )
        return (int(pid), str(identity)) if cursor.rowcount == 1 else None

    def recover_startup(self, *, fail_queued: bool = False) -> int:
        self._clear_stale_critical_sections()
        recovered = 0
        for job in self.store.list_active_jobs():
            job_id = str(job["job_id"])
            status = str(job["status"])
            if status == "queued":
                if fail_queued:
                    recovered += int(
                        self.fail_active(
                            job_id,
                            error_code="startup_unlaunched",
                            error_message="API restarted before the queued job was launched.",
                            clear_idempotency=True,
                        )
                    )
                continue
            if status == "cancelling":
                deadline = _parse_time(job.get("cancel_deadline_at"))
                if deadline is not None and deadline > _now():
                    continue
                claim = self.cancellation_claim_due(job_id)
                if claim is None:
                    continue
                if not self.terminate_identity_safe(job_id, claim=claim):
                    continue
                recovered += int(
                    self.finish(
                        claim,
                        "cancelled",
                        error_code="cancel_escalated",
                        error_message="Cancellation deadline expired at API startup.",
                    )
                )
                continue
            lease = _parse_time(job.get("lease_expires_at"))
            expected = deserialize_process_identity(job.get("pid_start_identity"))
            comparison = (
                ProcessIdentityComparison.UNKNOWN
                if expected is None
                else check_process_identity(expected)
            )
            if (
                comparison is ProcessIdentityComparison.MATCH
                and lease is not None
                and lease > _now()
            ):
                continue
            if (
                comparison is ProcessIdentityComparison.UNKNOWN
                and (
                    (
                        expected is None
                        and job.get("pid") is not None
                        and pid_is_alive(int(job["pid"]))
                    )
                    or (expected is not None and pid_is_alive(expected.pid))
                )
            ):
                # An unverifiable live PID is shielded, never guessed dead.
                self.authorize_signal(job_id)
                continue
            if (
                comparison is ProcessIdentityComparison.MATCH
                and not self.terminate_identity_safe(job_id)
            ):
                continue
            recovered += int(
                self.fail_active(
                    job_id,
                    error_code="orphaned",
                    error_message="Worker identity or lease was not live at API startup.",
                    clear_idempotency=True,
                )
            )
        return recovered

    def _clear_stale_critical_sections(self) -> int:
        """Clear only shields whose recorded generation no longer owns the row."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                update jobs set critical_depth = 0,
                    critical_owner_generation = null,
                    state_version = state_version + 1
                where critical_depth > 0
                  and critical_owner_generation is not launch_attempt
                  and kill_fence_state = 'open'
                """
            )
        return cursor.rowcount

    def terminate_identity_safe(
        self,
        job_id: str,
        *,
        grace_seconds: float = 1.0,
        claim: LaunchClaim | None = None,
    ) -> bool:
        """TERM→KILL only the persisted birth identity; return only after exit."""
        target = self.authorize_signal(job_id, claim=claim)
        if target is None:
            job = self._require_job(job_id)
            expected = deserialize_process_identity(job.get("pid_start_identity"))
            if expected is None:
                return False
            comparison = check_process_identity(expected)
            return (
                comparison is ProcessIdentityComparison.MISMATCH
                or (
                    comparison is ProcessIdentityComparison.UNKNOWN
                    and not pid_is_alive(expected.pid)
                )
            )
        pid, serialized = target
        expected = deserialize_process_identity(serialized)
        if expected is None:
            return False
        if check_process_identity(expected) is not ProcessIdentityComparison.MATCH:
            return False
        request_stop(pid)
        if _wait_identity_exit(expected, grace_seconds):
            return True
        if check_process_identity(expected) is not ProcessIdentityComparison.MATCH:
            return _wait_identity_exit(expected, 0)
        force_kill(pid)
        return _wait_identity_exit(expected, grace_seconds)

    def params_json(self, job_id: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select params_json from jobs where job_id = ?", (job_id,)
            ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def _terminal_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        claim: LaunchClaim,
        status: str,
        error_code: str | None,
        error_message: str | None,
    ) -> bool:
        row = conn.execute(
            """
            select status, project_id, session_id from jobs
            where job_id = ? and launch_token = ? and launch_attempt = ?
            """,
            (claim.job_id, claim.token, claim.attempt),
        ).fetchone()
        if row is None or str(row[0]) in TERMINAL_STATUSES:
            return False
        current = str(row[0])
        effective = "cancelled" if current == "cancelling" else status
        if current not in {"launching", "running", "cancelling"}:
            return False
        cursor = conn.execute(
            """
            update jobs set
                status = ?, state_version = state_version + 1,
                finished_at = ?, error_code = ?, error_message = ?,
                heartbeat_at = ?, lease_expires_at = null,
                critical_depth = 0, critical_owner_generation = null
            where job_id = ? and status = ?
              and launch_token = ? and launch_attempt = ?
            """,
            (
                effective,
                _iso(),
                error_code,
                error_message,
                _iso(),
                claim.job_id,
                current,
                claim.token,
                claim.attempt,
            ),
        )
        if cursor.rowcount != 1:
            return False
        self._update_run_terminal(
            conn,
            job_id=claim.job_id,
            session_id=str(row[2]),
            status=effective,
        )
        summary = (
            {}
            if error_code is None
            else {"error_code": error_code, "error_message": error_message}
        )
        self._insert_event(
            conn,
            job_id=claim.job_id,
            session_id=str(row[2]),
            project_id=str(row[1]),
            generation=claim.attempt,
            event_type=f"job.{effective}",
            status=effective,
            summary=summary,
        )
        return True

    @staticmethod
    def _update_run_terminal(
        conn: sqlite3.Connection,
        *,
        job_id: str,
        session_id: str,
        status: str,
    ) -> None:
        session_status = {
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }[status]
        cursor = conn.execute(
            """
            update sessions set
                status = ?, active_job_id = null,
                state_version = state_version + 1, updated_at = ?
            where session_id = ? and active_job_id = ? and storage_state = 'live'
            """,
            (session_status, _iso(), session_id, job_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Session terminal ownership handshake failed.")

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        session_id: str,
        project_id: str,
        generation: int,
        event_type: str,
        status: str,
        summary: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {"job_id": job_id, "status": status}
        if summary:
            payload.update(summary)
        event = TraceEvent(
            session_id=session_id,
            event_type=event_type,
            name=job_id,
            job_id=job_id,
            job_generation=generation,
            event_key=f"job:{job_id}:{generation}:{event_type}",
            finished_at=_now(),
            summary=payload,
        )
        conn.execute(
            """
            insert into trace_events(
                session_id, project_id, event_type, name, payload,
                job_id, job_generation, event_key
            ) values(?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(event_key) where event_key is not null do nothing
            """,
            (
                session_id,
                project_id,
                event_type,
                job_id,
                event.model_dump_json(),
                job_id,
                generation,
                f"job:{job_id}:{generation}:{event_type}",
            ),
        )

    @contextmanager
    def _locked_job_targets(self, session_id: str, lane_key: str) -> Iterator[None]:
        with ExitStack() as stack:
            for target in sorted({session_id, lane_key}):
                stack.enter_context(session_key_lock(self.root, target))
            yield

    @contextmanager
    def _locked_job(self, job: dict) -> Iterator[None]:
        with self._locked_job_targets(str(job["session_id"]), str(job["lane_key"])):
            yield

    def _require_job(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        return job

    def _deleting_target(
        self, conn: sqlite3.Connection, session_id: str, lane_key: str
    ) -> str | None:
        for target in dict.fromkeys((session_id, lane_key)):
            row = conn.execute(
                """
                select 1 from sessions where session_id = ? and storage_state = 'deleting'
                union all
                select 1 from storage_operations
                where op_kind = 'delete_session' and resource_kind = 'session'
                  and resource_key = ?
                  and state in ('prepared', 'fs_applied', 'db_committed', 'blocked')
                limit 1
                """,
                (target, target),
            ).fetchone()
            if row is not None:
                return target
        return None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 5000")
        conn.execute("pragma foreign_keys = on")
        return conn


class Heartbeat:
    def __init__(
        self,
        repository: JobLifecycleRepository,
        claim: LaunchClaim,
        interval: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self._repository = repository
        self._claim = claim
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"job-heartbeat-{claim.job_id}",
            daemon=True,
        )

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval * 2))

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if not self._repository.heartbeat(self._claim):
                    return
            except sqlite3.Error:
                continue

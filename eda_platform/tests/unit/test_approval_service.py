"""Pending-action lifecycle (§6.0): register → consume once → replay/expiry
guards, with the hash contract unchanged from core.permissions. Rows are keyed
by (action_hash, session_id) so identical content in two runs stays independent;
each register also mints a one-time generation token (C1) that a later
re-preview invalidates."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.application.services.approval_service import (
    ApprovalConsumedError,
    ApprovalExpiredError,
    ApprovalIdempotencyRaceError,
    ApprovalNotFoundError,
    ApprovalService,
    payload_digest,
)
from eda_platform.core.permissions import action_hash
from eda_platform.core.store import ArtifactStore

ACTION = {"type": "cleaning_apply", "dataset_id": "ds_1", "recipe_id": "r1"}
PAYLOAD = {"recipe": {"dataset_id": "ds_1"}, "dataset_id": "ds_1"}


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path)


def _register(
    service: ApprovalService, *, session_id: str = "run_1", project_id: str = "demo"
) -> tuple[str, str]:
    digest, generation, _expires = service.register(
        kind="cleaning_apply",
        session_id=session_id,
        project_id=project_id,
        action=ACTION,
        payload=PAYLOAD,
    )
    return digest, generation


def test_register_uses_core_permissions_hash(store: ArtifactStore) -> None:
    digest, generation = _register(ApprovalService(store))
    assert digest == action_hash(ACTION)
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending"
    assert row["kind"] == "cleaning_apply"
    assert row["session_id"] == "run_1"
    assert row["project_id"] == "demo"
    assert row["generation"] == generation and generation
    assert row["payload_digest"] == payload_digest(PAYLOAD)


def test_validate_and_consume_returns_payload_once(store: ArtifactStore) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    payload = service.validate_and_consume(
        digest, kind="cleaning_apply", session_id="run_1", generation=generation
    )
    assert payload == PAYLOAD
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None and row["status"] == "consumed"


def test_validate_then_consume_validation_failure_preserves_same_token(
    store: ArtifactStore,
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)

    def reject(_payload: dict) -> None:
        raise ValueError("path identity mismatch")

    with pytest.raises(ValueError, match="path identity mismatch"):
        service.validate_then_consume(
            digest,
            kind="cleaning_apply",
            session_id="run_1",
            generation=generation,
            validate=reject,
        )

    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == generation
    payload, validated = service.validate_then_consume(
        digest,
        kind="cleaning_apply",
        session_id="run_1",
        generation=generation,
        validate=lambda candidate: candidate["dataset_id"],
    )
    assert payload == PAYLOAD
    assert validated == "ds_1"


def test_validate_then_consume_orders_validator_before_store_cas(
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    calls: list[str] = []
    original = store.consume_pending_action

    def consume(
        action_hash: str,
        *,
        session_id: str,
        generation: str,
        now: str,
        idempotency_key: str | None = None,
    ) -> bool:
        assert calls == ["validated"]
        calls.append("consumed")
        return original(
            action_hash,
            session_id=session_id,
            generation=generation,
            now=now,
            idempotency_key=idempotency_key,
        )

    monkeypatch.setattr(store, "consume_pending_action", consume)
    service.validate_then_consume(
        digest,
        kind="cleaning_apply",
        session_id="run_1",
        generation=generation,
        validate=lambda _payload: calls.append("validated"),
    )
    assert calls == ["validated", "consumed", "validated"]


def test_compensation_context_rearms_same_generation_after_fault(
    store: ArtifactStore,
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    service.validate_then_consume(
        digest,
        kind="cleaning_apply",
        session_id="run_1",
        generation=generation,
        validate=lambda _payload: None,
    )

    with pytest.raises(OSError, match="injected durable producer fault"):
        with service.compensate_on_failure(digest, session_id="run_1"):
            raise OSError("injected durable producer fault")

    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == generation


def test_source_change_during_consume_is_revalidated_and_rearmed(
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    source_identity = "approved"
    original = store.consume_pending_action

    def consume(
        action_hash: str,
        *,
        session_id: str,
        generation: str,
        now: str,
        idempotency_key: str | None = None,
    ) -> bool:
        nonlocal source_identity
        consumed = original(
            action_hash,
            session_id=session_id,
            generation=generation,
            now=now,
            idempotency_key=idempotency_key,
        )
        source_identity = "changed"
        return consumed

    monkeypatch.setattr(store, "consume_pending_action", consume)

    def validate(_payload: dict) -> None:
        if source_identity != "approved":
            raise ValueError("source changed during reservation")

    with pytest.raises(ValueError, match="source changed during reservation"):
        service.validate_then_consume(
            digest,
            kind="cleaning_apply",
            session_id="run_1",
            generation=generation,
            validate=validate,
        )

    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == generation


def test_concurrent_validate_then_consume_allows_one_producer(
    store: ArtifactStore,
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    barrier = threading.Barrier(8)
    winners: list[str] = []
    failures: list[type[Exception]] = []

    def worker() -> None:
        barrier.wait()
        try:
            _payload, value = service.validate_then_consume(
                digest,
                kind="cleaning_apply",
                session_id="run_1",
                generation=generation,
                validate=lambda _payload: "validated",
            )
            winners.append(value)
        except Exception as exc:
            failures.append(type(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert winners == ["validated"]
    assert failures == [ApprovalConsumedError] * 7


def test_replay_after_consume_raises_consumed(store: ArtifactStore) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    service.validate_and_consume(
        digest, kind="cleaning_apply", session_id="run_1", generation=generation
    )
    with pytest.raises(ApprovalConsumedError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )


def test_expired_approval_raises_and_marks_row(store: ArtifactStore) -> None:
    service = ApprovalService(store, ttl_seconds=-1)
    digest, generation = _register(service)
    with pytest.raises(ApprovalExpiredError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None and row["status"] == "expired"


def test_unknown_hash_raises_not_found(store: ArtifactStore) -> None:
    with pytest.raises(ApprovalNotFoundError):
        ApprovalService(store).validate_and_consume(
            "0" * 64, kind="cleaning_apply", session_id="run_1", generation="tok"
        )


def test_kind_mismatch_reads_as_not_found(store: ArtifactStore) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    with pytest.raises(ApprovalNotFoundError):
        service.validate_and_consume(
            digest, kind="analysis_plan", session_id="run_1", generation=generation
        )
    # The row must survive untouched for the correct kind.
    assert (
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )
        == PAYLOAD
    )


def test_run_mismatch_reads_as_not_found(store: ArtifactStore) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    with pytest.raises(ApprovalNotFoundError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_other", generation=generation
        )


def test_stale_generation_reads_as_not_found(store: ArtifactStore) -> None:
    """C1: a re-preview rotates the generation; the token handed out with the
    older preview must read as not-found, never resurrect a consumed row."""
    service = ApprovalService(store)
    digest, old_generation = _register(service)
    digest2, new_generation = _register(service)
    assert digest2 == digest and new_generation != old_generation
    with pytest.raises(ApprovalNotFoundError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=old_generation
        )
    # The current token still works exactly once.
    assert (
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=new_generation
        )
        == PAYLOAD
    )
    # And after consumption the old token still reads not-found (no 409 leak).
    with pytest.raises(ApprovalNotFoundError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=old_generation
        )


def test_tampered_payload_reads_as_not_found(store: ArtifactStore) -> None:
    """C4: a payload_json edited behind the service's back no longer matches
    the stored digest, and the row is unusable."""
    service = ApprovalService(store)
    digest, generation = _register(service)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "update pending_actions set payload_json = ? where action_hash = ?",
            ('{"recipe": {"dataset_id": "ds_evil"}}', digest),
        )
    with pytest.raises(ApprovalNotFoundError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )
    # The tampered row was not consumed by the failed attempt.
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None and row["status"] == "pending"


def test_consume_returns_row_as_persisted_after_flip(store: ArtifactStore) -> None:
    """C2: the returned payload is rebuilt from a fresh post-consume read of
    the row, not from the pre-consume snapshot."""
    service = ApprovalService(store)
    digest, generation = _register(service)
    original_get = store.get_pending_action
    reads: list[str | None] = []

    def tracking_get(action_hash: str, *, session_id: str) -> dict | None:
        row = original_get(action_hash, session_id=session_id)
        reads.append(None if row is None else str(row["status"]))
        return row

    store.get_pending_action = tracking_get  # type: ignore[method-assign]
    try:
        payload = service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )
    finally:
        store.get_pending_action = original_get  # type: ignore[method-assign]
    assert payload == PAYLOAD
    # First read saw the pending row; the value-bearing read saw it consumed.
    assert reads[0] == "pending"
    assert reads[-1] == "consumed"


def test_post_consume_read_failure_rearms_same_generation(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ApprovalService(store)
    digest, generation = _register(service)
    original = store.get_pending_action
    reads = 0

    def fail_first_post_consume_read(
        action_hash: str, *, session_id: str
    ) -> dict[str, object] | None:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise RuntimeError("post-consume read failed")
        return original(action_hash, session_id=session_id)

    monkeypatch.setattr(store, "get_pending_action", fail_first_post_consume_read)
    with pytest.raises(RuntimeError, match="post-consume read failed"):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )

    row = original(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending"
    assert row["generation"] == generation


def test_reregister_rearms_a_consumed_hash(store: ArtifactStore) -> None:
    """A fresh preview re-arms the same hash; without it the replay guard holds."""
    service = ApprovalService(store)
    digest, generation = _register(service)
    service.validate_and_consume(
        digest, kind="cleaning_apply", session_id="run_1", generation=generation
    )
    with pytest.raises(ApprovalConsumedError):
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation
        )
    digest2, generation2 = _register(service)
    assert digest2 == digest
    assert (
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation2
        )
        == PAYLOAD
    )


def test_same_hash_in_two_runs_stays_independent(store: ArtifactStore) -> None:
    """Slice-E F2: identical content in two projects yields one action_hash;
    the second preview must not steal the first run's pending row."""
    service = ApprovalService(store)
    digest, generation1 = _register(service, session_id="run_1", project_id="demo")
    digest2, generation2 = _register(service, session_id="run_2", project_id="other")
    assert digest2 == digest
    # run_1's row survived run_2's register untouched.
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending" and row["project_id"] == "demo"
    assert row["generation"] == generation1
    # Each run consumes its own row; the other stays pending until consumed.
    assert (
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_1", generation=generation1
        )
        == PAYLOAD
    )
    row2 = store.get_pending_action(digest, session_id="run_2")
    assert row2 is not None and row2["status"] == "pending"
    assert (
        service.validate_and_consume(
            digest, kind="cleaning_apply", session_id="run_2", generation=generation2
        )
        == PAYLOAD
    )


def test_init_db_rebuilds_legacy_single_pk_table(tmp_path: Path) -> None:
    """Slice-E F2 migration: an old single-column-PK pending_actions table is
    dropped and recreated with the composite (action_hash, session_id) key."""
    db_path = tmp_path / "state.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table pending_actions (
                action_hash text primary key,
                session_id text not null,
                project_id text not null,
                kind text not null,
                payload_json text not null,
                created_at text not null,
                expires_at text not null,
                status text not null default 'pending'
            )
            """
        )
        conn.execute(
            "insert into pending_actions values"
            " ('h1', 'run_1', 'demo', 'cleaning_apply', '{}', 't', 't', 'pending')"
        )
    store = ArtifactStore(tmp_path)
    with sqlite3.connect(db_path) as conn:
        pk_columns = [
            row[1]
            for row in conn.execute("pragma table_info(pending_actions)")
            if row[5] > 0
        ]
        columns = {
            row[1] for row in conn.execute("pragma table_info(pending_actions)")
        }
    assert pk_columns == ["action_hash", "session_id"]
    assert {"generation", "payload_digest"} <= columns
    # Legacy TTL-scratch rows are discarded, not migrated.
    assert store.get_pending_action("h1", session_id="run_1") is None


def test_init_db_adds_token_columns_to_composite_pk_table(tmp_path: Path) -> None:
    """A pre-C1 composite-PK table gains the token columns; its rows keep an
    empty generation, which never matches a client-supplied token."""
    db_path = tmp_path / "state.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table pending_actions (
                action_hash text not null,
                session_id text not null,
                project_id text not null,
                kind text not null,
                payload_json text not null,
                created_at text not null,
                expires_at text not null,
                status text not null default 'pending',
                primary key (action_hash, session_id)
            )
            """
        )
        conn.execute(
            "insert into pending_actions values"
            " ('h1', 'run_1', 'demo', 'cleaning_apply', '{}', 't',"
            " '2999-01-01T00:00:00+00:00', 'pending')"
        )
    store = ArtifactStore(tmp_path)
    row = store.get_pending_action("h1", session_id="run_1")
    assert row is not None
    assert row["generation"] == "" and row["payload_digest"] == ""
    with pytest.raises(ApprovalNotFoundError):
        ApprovalService(store).validate_and_consume(
            "h1", kind="cleaning_apply", session_id="run_1", generation="anytoken"
        )


def test_atomic_consume_rowcount_guard(store: ArtifactStore) -> None:
    """Store-level: the UPDATE ... WHERE status='pending' flip fires exactly once."""
    service = ApprovalService(store)
    digest, generation = _register(service)
    now = "2999-01-01T00:00:00+00:00"
    assert (
        store.consume_pending_action(
            digest, session_id="run_1", generation=generation, now="2000-01-01T00:00:00+00:00"
        )
        is True
    )
    assert (
        store.consume_pending_action(
            digest, session_id="run_1", generation=generation, now="2000-01-01T00:00:00+00:00"
        )
        is False
    )
    assert store.expire_pending_action(digest, session_id="run_1", now=now) is False


def test_store_consume_requires_matching_generation(store: ArtifactStore) -> None:
    """C1 store-level: the UPDATE itself refuses a superseded token even when
    the service-layer pre-check is bypassed."""
    digest, generation = _register(ApprovalService(store))
    now = "2000-01-01T00:00:00+00:00"
    assert (
        store.consume_pending_action(digest, session_id="run_1", generation="stale", now=now)
        is False
    )
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None and row["status"] == "pending"
    assert (
        store.consume_pending_action(
            digest, session_id="run_1", generation=generation, now=now
        )
        is True
    )


def test_store_restore_pending_action_flips_consumed_back(store: ArtifactStore) -> None:
    """C6 store-level: compensation re-arms the consumed row with the same
    generation; a pending or missing row is untouched."""
    digest, generation = _register(ApprovalService(store))
    assert store.restore_pending_action(digest, session_id="run_1") is False  # still pending
    assert store.consume_pending_action(
        digest, session_id="run_1", generation=generation, now="2000-01-01T00:00:00+00:00"
    )
    assert store.restore_pending_action(digest, session_id="run_1") is True
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None
    assert row["status"] == "pending" and row["generation"] == generation


def test_same_key_race_waiter_unblocks_when_winner_restores(
    store: ArtifactStore,
) -> None:
    """A failed first owner re-arms the token; the race loser may take over."""
    service = ApprovalService(store)
    digest, generation = _register(service)
    service.validate_and_consume(
        digest,
        kind="cleaning_apply",
        session_id="run_1",
        generation=generation,
        idempotency_key="same-key",
    )
    waiting = threading.Event()
    results: list[bool] = []

    def wait_for_resolution() -> None:
        waiting.set()
        results.append(
            service.wait_for_idempotent_resolution(
                digest,
                session_id="run_1",
                idempotency_key="same-key",
                timeout_seconds=1,
            )
        )

    thread = threading.Thread(target=wait_for_resolution)
    thread.start()
    assert waiting.wait(timeout=1)
    assert store.restore_pending_action(digest, session_id="run_1")
    thread.join(timeout=1)
    assert results == [True]
    assert (
        service.validate_and_consume(
            digest,
            kind="cleaning_apply",
            session_id="run_1",
            generation=generation,
            idempotency_key="same-key",
        )
        == PAYLOAD
    )


def test_consume_false_with_pending_row_exits_within_shared_budget(
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ApprovalService(
        store,
        contention_timeout_seconds=0,
        contention_max_attempts=10_000,
    )
    digest, generation = _register(service)
    calls = 0

    def never_consume(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(store, "consume_pending_action", never_consume)
    with pytest.raises(ApprovalIdempotencyRaceError):
        service.validate_and_consume(
            digest,
            kind="cleaning_apply",
            session_id="run_1",
            generation=generation,
            idempotency_key="same-key",
        )
    assert calls == 1


def test_producer_reentry_reuses_one_deadline_and_succeeds_after_restores(
    store: ArtifactStore,
) -> None:
    service = ApprovalService(
        store,
        contention_timeout_seconds=1,
        contention_max_attempts=5,
    )
    digest, generation = _register(service)
    deadlines: list[float] = []
    side_effects: list[str] = []

    def operation(deadline: float) -> str:
        deadlines.append(deadline)
        service.validate_and_consume(
            digest,
            kind="cleaning_apply",
            session_id="run_1",
            generation=generation,
            idempotency_key="same-key",
            deadline=deadline,
        )
        if len(deadlines) <= 3:
            assert store.restore_pending_action(digest, session_id="run_1")
            raise ApprovalIdempotencyRaceError(digest)
        side_effects.append("committed")
        return "ok"

    assert (
        service.run_idempotent_producer(
            digest,
            session_id="run_1",
            idempotency_key="same-key",
            operation=operation,
        )
        == "ok"
    )
    assert len(deadlines) == 4
    assert len(set(deadlines)) == 1
    assert side_effects == ["committed"]


def test_producer_reentry_stops_at_max_attempts_without_resetting_deadline(
    store: ArtifactStore,
) -> None:
    service = ApprovalService(
        store,
        contention_timeout_seconds=1,
        contention_max_attempts=3,
    )
    digest, _generation = _register(service)
    deadlines: list[float] = []

    def always_race(deadline: float) -> None:
        deadlines.append(deadline)
        raise ApprovalIdempotencyRaceError(digest)

    with pytest.raises(ApprovalIdempotencyRaceError):
        service.run_idempotent_producer(
            digest,
            session_id="run_1",
            idempotency_key="same-key",
            operation=always_race,
        )
    assert len(deadlines) == 3
    assert len(set(deadlines)) == 1


def test_store_consume_refuses_expired_row(store: ArtifactStore) -> None:
    """Slice-E F5a: the store-level UPDATE itself must enforce the TTL — an
    already-expired pending row is unconsumable even when the service-layer
    pre-check is bypassed."""
    digest, generation = _register(ApprovalService(store, ttl_seconds=-1))
    row = store.get_pending_action(digest, session_id="run_1")
    assert row is not None and row["status"] == "pending"  # pre-check bypassed
    now = datetime.now(UTC).isoformat()
    assert (
        store.consume_pending_action(digest, session_id="run_1", generation=generation, now=now)
        is False
    )
    fresh = store.get_pending_action(digest, session_id="run_1")
    assert fresh is not None and fresh["status"] == "pending"


def test_concurrent_consume_has_exactly_one_winner(store: ArtifactStore) -> None:
    """Slice-E F5b: 16 threads racing the same (hash, run) — one True."""
    digest, generation = _register(ApprovalService(store))
    now = datetime.now(UTC).isoformat()
    results: list[bool] = []
    barrier = threading.Barrier(16)

    def worker() -> None:
        barrier.wait()
        results.append(
            store.consume_pending_action(
                digest, session_id="run_1", generation=generation, now=now
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 16
    assert sum(results) == 1


def test_all_approval_consumers_use_the_validate_before_consume_seam() -> None:
    service_root = (
        Path(__file__).parents[2]
        / "src"
        / "eda_platform"
        / "application"
        / "services"
    )
    expected_calls = {
        "chat_service.py": 1,
        "cleaning_service.py": 1,
        "question_service.py": 2,
        "relationship_service.py": 1,
        "skill_service.py": 1,
        "investigation_service.py": 3,
    }
    for filename, expected in expected_calls.items():
        source = (service_root / filename).read_text(encoding="utf-8")
        assert source.count(".validate_then_consume(") == expected, filename
        assert ".validate_and_consume(" not in source, filename

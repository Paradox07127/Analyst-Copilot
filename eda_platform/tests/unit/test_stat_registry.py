"""System-owned statistical registry: the sole allocator of family sequences.

Family ids, sequence indices and comparison counts are all derived here, every
attempt (including failures) is recorded, and a replayed journal reproduces
identical numbering. Forged or rolled-back sequences have no API to enter
through and fail closed on load, and concurrent writers are fenced by a file
lock rather than racing on the journal.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from eda_platform.core.stat_registry import (
    StatAttempt,
    StatRegistryError,
    StatTestRegistry,
    derive_family_id,
)


def _begin(
    registry: StatTestRegistry,
    family_id: str,
    step: str,
    *,
    digest: str = "digest_a",
) -> StatAttempt:
    return registry.begin_attempt(
        family_id=family_id,
        requested_test_type="independent_t_test",
        arguments_digest=digest,
        logical_step_id=step,
    )


def test_sequences_are_allocated_densely_per_family() -> None:
    registry = StatTestRegistry()
    a1 = _begin(registry, "fam_a", "step_1")
    b1 = _begin(registry, "fam_b", "step_2")
    a2 = _begin(registry, "fam_a", "step_3")
    a3 = _begin(registry, "fam_a", "step_4")
    assert [a1.sequence_index, a2.sequence_index, a3.sequence_index] == [1, 2, 3]
    assert b1.sequence_index == 1
    assert registry.comparison_count("fam_a") == 3
    assert registry.comparison_count("fam_b") == 1
    assert registry.comparison_count("fam_missing") == 0


def test_failed_attempts_still_consume_a_sequence() -> None:
    registry = StatTestRegistry()
    first = _begin(registry, "fam_a", "step_1")
    registry.record_failure(first.attempt_id, error="guard rejected the column")
    second = _begin(registry, "fam_a", "step_2")
    assert second.sequence_index == 2
    assert registry.comparison_count("fam_a") == 2
    attempts = registry.attempts("fam_a")
    assert [attempt.status for attempt in attempts] == ["failed", "running"]


def test_replaying_a_logical_step_returns_the_same_attempt() -> None:
    registry = StatTestRegistry()
    first = _begin(registry, "fam_a", "step_1")
    _begin(registry, "fam_a", "step_2")
    replay = _begin(registry, "fam_a", "step_1")
    assert replay.attempt_id == first.attempt_id
    assert replay.sequence_index == first.sequence_index == 1
    assert registry.comparison_count("fam_a") == 2, "a replay must not re-count"


def test_a_replay_with_different_arguments_is_rejected() -> None:
    registry = StatTestRegistry()
    _begin(registry, "fam_a", "step_1", digest="digest_a")
    with pytest.raises(StatRegistryError):
        _begin(registry, "fam_a", "step_1", digest="digest_FORGED")
    with pytest.raises(StatRegistryError):
        _begin(registry, "fam_other", "step_1", digest="digest_a")


def test_outcome_rollbacks_and_forgeries_are_rejected() -> None:
    registry = StatTestRegistry()
    attempt = _begin(registry, "fam_a", "step_1")
    registry.record_completion(attempt.attempt_id, receipt_id="rcpt_1")
    registry.record_completion(attempt.attempt_id, receipt_id="rcpt_1")  # idempotent
    with pytest.raises(StatRegistryError):
        registry.record_completion(attempt.attempt_id, receipt_id="rcpt_other")
    with pytest.raises(StatRegistryError):
        registry.record_failure(attempt.attempt_id, error="cannot unwind a completion")
    with pytest.raises(StatRegistryError):
        registry.record_completion("att_unknown", receipt_id="rcpt_x")


def test_journal_reload_preserves_every_sequence(tmp_path: Path) -> None:
    path = tmp_path / "stat_registry.jsonl"
    registry = StatTestRegistry(path)
    first = _begin(registry, "fam_a", "step_1")
    registry.record_failure(first.attempt_id, error="boom")
    second = _begin(registry, "fam_a", "step_2")
    registry.record_completion(second.attempt_id, receipt_id="rcpt_2")

    reloaded = StatTestRegistry(path)
    attempts = reloaded.attempts("fam_a")
    assert [attempt.sequence_index for attempt in attempts] == [1, 2]
    assert [attempt.status for attempt in attempts] == ["failed", "completed"]
    assert reloaded.comparison_count("fam_a") == 2
    third = _begin(reloaded, "fam_a", "step_3")
    assert third.sequence_index == 3, "numbering must continue, never restart"


def test_a_doctored_journal_with_a_rolled_back_sequence_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stat_registry.jsonl"
    registry = StatTestRegistry(path)
    _begin(registry, "fam_a", "step_1")
    _begin(registry, "fam_a", "step_2")
    forged = {
        "event": "attempt_started",
        "attempt_id": "att_forged",
        "family_id": "fam_a",
        "sequence_index": 1,  # rollback: the allocator already issued 1 and 2
        "logical_step_id": "step_9",
        "requested_test_type": "independent_t_test",
        "arguments_digest": "digest_z",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(forged) + "\n")
    with pytest.raises(StatRegistryError):
        StatTestRegistry(path).attempts("fam_a")


def test_a_torn_tail_is_ignored_on_reload(tmp_path: Path) -> None:
    path = tmp_path / "stat_registry.jsonl"
    registry = StatTestRegistry(path)
    _begin(registry, "fam_a", "step_1")
    with path.open("ab") as handle:
        handle.write(b'{"event": "attempt_started", "attempt_')
    reloaded = StatTestRegistry(path)
    assert reloaded.comparison_count("fam_a") == 1


# ---------------------------------------------------------------------------
# Family derivation: the model may not name its own family.
# ---------------------------------------------------------------------------


def test_family_ids_are_derived_from_the_data_under_test() -> None:
    base = derive_family_id(dataset_id="ds_a", columns=["segment", "revenue"])
    assert base.startswith("fam_")
    assert base == derive_family_id(dataset_id="ds_a", columns=["revenue", "segment"])
    assert base == derive_family_id(
        dataset_id="ds_a", columns=["revenue", "segment", "revenue"]
    )
    assert base != derive_family_id(dataset_id="ds_b", columns=["segment", "revenue"])
    assert base != derive_family_id(dataset_id="ds_a", columns=["segment", "cost"])


# ---------------------------------------------------------------------------
# Concurrency: the allocator must be fenced, like ReceiptOutbox.
# ---------------------------------------------------------------------------


def test_two_live_registries_cannot_duplicate_a_sequence(tmp_path: Path) -> None:
    path = tmp_path / "stat_registry.jsonl"
    first = StatTestRegistry(path)
    second = StatTestRegistry(path)
    left = _begin(first, "fam_a", "step_first")
    right = _begin(second, "fam_a", "step_second")
    assert {left.sequence_index, right.sequence_index} == {1, 2}
    reloaded = StatTestRegistry(path)
    assert reloaded.comparison_count("fam_a") == 2


def test_concurrent_allocations_are_serialised(tmp_path: Path) -> None:
    path = tmp_path / "stat_registry.jsonl"
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    allocated: list[int] = []
    failures: list[BaseException] = []
    guard = threading.Lock()

    def worker(index: int) -> None:
        try:
            registry = StatTestRegistry(path)
            barrier.wait(timeout=10)
            attempt = _begin(registry, "fam_a", f"step_{index}")
            with guard:
                allocated.append(attempt.sequence_index)
        except BaseException as exc:  # noqa: BLE001 - reported below
            with guard:
                failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not failures, f"allocation raised: {failures}"
    assert sorted(allocated) == list(range(1, worker_count + 1))
    assert StatTestRegistry(path).comparison_count("fam_a") == worker_count


def test_a_torn_tail_repair_cannot_erase_a_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stat_registry.jsonl"
    writer_a = StatTestRegistry(path)
    _begin(writer_a, "fam_a", "step_1")
    writer_b = StatTestRegistry(path)
    with path.open("ab") as handle:  # a crashed writer's partial line
        handle.write(b'{"event": "attempt_star')

    inside_repair = threading.Event()
    resume_repair = threading.Event()
    original = StatTestRegistry._truncate_torn_tail

    def slow_repair(self: StatTestRegistry) -> None:
        # Widen writer A's read/truncate window only; writer B must be fenced out.
        if self is not writer_a:
            original(self)
            return
        raw = path.read_bytes()
        if raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        inside_repair.set()
        resume_repair.wait(timeout=10)
        with path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())

    monkeypatch.setattr(StatTestRegistry, "_truncate_torn_tail", slow_repair)

    thread_a = threading.Thread(
        target=lambda: writer_a._append({"event": "attempt_failed", "attempt_id": "a"})
    )
    thread_a.start()
    assert inside_repair.wait(timeout=10)
    thread_b = threading.Thread(
        target=lambda: writer_b._append({"event": "attempt_failed", "attempt_id": "b"})
    )
    thread_b.start()
    time.sleep(0.3)  # give an unfenced writer B time to append and be erased
    resume_repair.set()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)

    written = path.read_text(encoding="utf-8")
    assert '"attempt_id": "a"' in written
    assert '"attempt_id": "b"' in written, "a committed record was truncated away"

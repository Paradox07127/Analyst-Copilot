"""Generic JSONL event journal: mechanism tests on a minimal toy domain.

The investigation and exploration journals are thin subclasses; these tests
pin the shared mechanics (fsync append, torn-tail recovery, attempt-epoch
fencing, snapshot cache) independently of any domain reducer.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import BaseModel, TypeAdapter

from eda_platform.core.event_journal import (
    EventJournalCorruptionError,
    EventTransitionError,
    JsonlEventJournal,
)


class TickEvent(BaseModel):
    schema_version: int = 1
    seq: int
    journal_id: str
    event_type: Literal["started", "attempt_started", "ticked", "finished"]
    attempt_epoch: int = 0
    note: str | None = None


class TickState(BaseModel):
    journal_id: str
    attempt_epoch: int = 0
    ticks: int = 0
    status: Literal["running", "finished"] = "running"
    last_seq: int = 0


def reduce_tick(state: TickState | None, event: TickEvent) -> TickState:
    if state is None:
        if event.event_type != "started" or event.seq != 0:
            raise EventTransitionError("the first event must be started with seq 0.")
        return TickState(journal_id=event.journal_id, attempt_epoch=event.attempt_epoch)
    if event.event_type == "started":
        raise EventTransitionError("started may only be the first event.")
    if event.journal_id != state.journal_id:
        raise EventTransitionError("event journal_id does not match journal state.")
    if event.seq != state.last_seq + 1:
        raise EventTransitionError(f"event seq must be {state.last_seq + 1}, got {event.seq}.")
    expected_epoch = (
        state.attempt_epoch + 1
        if event.event_type == "attempt_started"
        else state.attempt_epoch
    )
    if event.attempt_epoch != expected_epoch:
        raise EventTransitionError(
            f"event attempt_epoch must be {expected_epoch}, got {event.attempt_epoch}."
        )
    if state.status != "running":
        raise EventTransitionError("cannot append to a finished journal.")
    values = state.model_dump()
    values["last_seq"] = event.seq
    if event.event_type == "attempt_started":
        values["attempt_epoch"] = event.attempt_epoch
    elif event.event_type == "ticked":
        values["ticks"] = state.ticks + 1
    elif event.event_type == "finished":
        values["status"] = "finished"
    return TickState.model_validate(values)


def _empty_journal(tmp_path: Path) -> JsonlEventJournal[TickEvent, TickState]:
    return JsonlEventJournal(
        tmp_path / "ticks" / "tick.journal.jsonl",
        event_adapter=TypeAdapter(TickEvent),
        state_adapter=TypeAdapter(TickState),
        reducer=reduce_tick,
        id_field="journal_id",
        label="tick",
        executor_lock_prefix="tick-executor",
    )


def _journal(tmp_path: Path) -> JsonlEventJournal[TickEvent, TickState]:
    journal = _empty_journal(tmp_path)
    journal.append(TickEvent(seq=0, journal_id="tick_1", event_type="started"))
    return journal


def test_rebuild_replays_identical_state_from_every_complete_prefix(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("ticked")
    journal.append_new("ticked")
    final = journal.append_new("finished")

    events = journal.events()
    assert [event.seq for event in events] == [0, 1, 2, 3]
    state: TickState | None = None
    for event in events:
        state = reduce_tick(state, event)
    assert state == final
    assert journal.rebuild() == final
    assert final.ticks == 2
    assert final.status == "finished"


def test_append_new_requires_an_initialized_journal(tmp_path: Path) -> None:
    journal = _empty_journal(tmp_path)
    with pytest.raises(EventTransitionError, match="initialize the journal"):
        journal.append_new("ticked")
    with pytest.raises(EventTransitionError, match="initialize the journal"):
        journal.claim_attempt()


def test_torn_tail_is_ignored_then_truncated_on_next_append(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("ticked")
    before = journal.rebuild()

    with journal.path.open("ab") as handle:
        handle.write(b'{"seq":2,"journal_id"')

    resumed = _empty_journal(tmp_path)
    assert resumed.rebuild() == before

    after = resumed.append_new("ticked")
    assert after.ticks == 2
    assert after.last_seq == 2
    raw = resumed.path.read_bytes()
    assert raw.endswith(b"\n")
    assert b'{"seq":2,"journal_id"\n' not in raw


def test_valid_torn_tail_is_uncommitted_and_its_seq_is_reused(tmp_path: Path) -> None:
    """A crash after the record bytes but before its "\\n" must not brick the
    journal: the reader used to count that tail as committed while the next
    append deleted it, leaving a permanent seq gap no journal API could repair.
    """
    journal = _journal(tmp_path)
    journal.append_new("ticked")
    journal.path.write_bytes(journal.path.read_bytes().removesuffix(b"\n"))

    resumed = _empty_journal(tmp_path)
    torn = resumed.rebuild()
    assert torn is not None
    assert torn.last_seq == 0
    assert torn.ticks == 0

    after = resumed.append_new("ticked")
    assert after.last_seq == 1
    assert resumed.rebuild() == after
    assert [event.seq for event in resumed.events()] == [0, 1]


def test_append_new_inside_a_fenced_side_effect_does_not_deadlock(tmp_path: Path) -> None:
    """Recording a side effect in the same fence that authorized it is the whole
    point of the fence; re-entering the writer lock used to block forever."""
    journal = _journal(tmp_path)
    journal.claim_attempt()
    outcome: list[object] = []

    def body() -> None:
        try:
            with journal.fenced_side_effect() as epoch:
                journal.append_new("ticked", note=f"epoch-{epoch}")
            outcome.append("committed")
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "append_new inside fenced_side_effect deadlocked"
    assert outcome == ["committed"], outcome

    state = journal.rebuild()
    assert state is not None
    assert state.ticks == 1
    assert state.last_seq == 2


def test_concurrent_claim_attempt_hands_out_distinct_epochs_without_errors(
    tmp_path: Path,
) -> None:
    """Reading the current epoch outside the lock turned lock contention into a
    hard "attempt_epoch must be N" error that looked like data corruption."""
    _journal(tmp_path)
    rivals = 8
    ready = threading.Barrier(rivals)
    guard = threading.Lock()
    results: list[object] = []

    def claim() -> None:
        journal = _empty_journal(tmp_path)
        ready.wait()
        try:
            value: object = journal.claim_attempt().attempt_epoch
        except BaseException as exc:  # noqa: BLE001
            value = exc
        with guard:
            results.append(value)

    threads = [threading.Thread(target=claim) for _ in range(rivals)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)

    failures = [item for item in results if isinstance(item, BaseException)]
    assert not failures, failures
    assert sorted(cast(list[int], results)) == list(range(1, rivals + 1))


def test_middle_corruption_fails_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("ticked")
    lines = journal.path.read_bytes().splitlines(keepends=True)
    journal.path.write_bytes(lines[0] + b"{broken-json}\n" + lines[1])
    with pytest.raises(EventJournalCorruptionError, match="Invalid committed record"):
        journal.rebuild()


def test_blank_committed_record_fails_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.path.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(EventJournalCorruptionError, match="Blank committed record"):
        journal.rebuild()


def test_epoch_fence_blocks_a_stale_writer(tmp_path: Path) -> None:
    first = _journal(tmp_path)
    second = _empty_journal(tmp_path)
    assert first.claim_attempt().attempt_epoch == 1
    assert second.claim_attempt().attempt_epoch == 2

    with pytest.raises(EventTransitionError, match="stale tick executor"):
        first.append_new("ticked")
    with pytest.raises(EventTransitionError, match="stale tick executor"):
        with first.fenced_side_effect():
            raise AssertionError("stale executor entered the commit section")

    state = second.append_new("ticked")
    assert state.ticks == 1
    with second.fenced_side_effect() as epoch:
        assert epoch == 2


def test_fenced_side_effect_requires_a_claimed_attempt(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with pytest.raises(EventTransitionError, match="claim an executor attempt"):
        with journal.fenced_side_effect():
            raise AssertionError("unclaimed executor entered the commit section")


def test_snapshot_round_trip_and_corruption_isolation(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("ticked")
    state = journal.rebuild()
    snapshot_path = journal.write_snapshot(state)
    assert snapshot_path.exists()
    assert journal.read_snapshot() == state

    with pytest.raises(EventTransitionError, match="does not match the journal"):
        journal.write_snapshot(TickState(journal_id="tick_1", ticks=99, last_seq=1))

    lines = journal.path.read_bytes().splitlines(keepends=True)
    journal.path.write_bytes(lines[0] + b"{corrupt}\n")
    with pytest.raises(EventJournalCorruptionError):
        journal.rebuild()
    assert journal.read_snapshot() == state


def test_reducer_transitions_are_enforced_at_append(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    line_count = len(journal.events())

    with pytest.raises(EventTransitionError, match="seq must be"):
        journal.append(TickEvent(seq=9, journal_id="tick_1", event_type="ticked"))
    with pytest.raises(EventTransitionError, match="attempt_epoch must be"):
        journal.append(
            TickEvent(seq=1, journal_id="tick_1", event_type="ticked", attempt_epoch=7)
        )

    assert len(journal.events()) == line_count


def test_terminal_status_from_the_reducer_blocks_append(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append_new("finished")
    with pytest.raises(EventTransitionError, match="finished journal"):
        journal.append_new("ticked")

"""Receipt outbox/WAL: tool_prepared -> artifact -> receipt_committed -> reconcile.

Every step is idempotent, so a crash at any persistence boundary followed by a
replay of the same logical step recovers to exactly one committed logical
receipt: orphaned prepares are aborted, durable-but-uncommitted artifacts are
rolled forward.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eda_platform.core.receipt_outbox import (
    ReceiptOutbox,
    ReceiptOutboxError,
)


def _never_exists(_artifact_id: str) -> bool:
    return False


def test_prepare_write_commit_is_the_happy_path(tmp_path: Path) -> None:
    outbox = ReceiptOutbox(tmp_path / "outbox.jsonl")
    resolution = outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    assert resolution.phase == "prepared"
    assert not resolution.replayed
    outbox.mark_artifact_written("step_1")
    outbox.commit("step_1")
    state = outbox.state("step_1")
    assert state is not None
    assert state.phase == "committed"
    assert state.receipt_id == "rcpt_a"
    assert state.artifact_id == "art_a"


def test_replay_of_a_committed_step_returns_the_recorded_identity(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    outbox = ReceiptOutbox(path)
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    outbox.mark_artifact_written("step_1")
    outbox.commit("step_1")

    # A crash replay rebuilds the receipt with a drifted created_at, so the
    # candidate ids differ; the recorded identity must win.
    replay = ReceiptOutbox(path).prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_b",
        artifact_id="art_b",
        artifact_exists=lambda artifact_id: artifact_id == "art_a",
    )
    assert replay.phase == "committed"
    assert replay.replayed
    assert replay.receipt_id == "rcpt_a"
    assert replay.artifact_id == "art_a"


def test_crash_before_the_artifact_write_aborts_the_orphan(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    ReceiptOutbox(path).prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    # Crash: the content-addressed artifact never became durable.
    replay_outbox = ReceiptOutbox(path)
    resolution = replay_outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_b",
        artifact_id="art_b",
        artifact_exists=_never_exists,
    )
    assert resolution.phase == "prepared"
    assert resolution.receipt_id == "rcpt_b"
    replay_outbox.mark_artifact_written("step_1")
    replay_outbox.commit("step_1")
    state = replay_outbox.state("step_1")
    assert state is not None and state.phase == "committed"
    assert state.receipt_id == "rcpt_b"
    committed = [e for e in replay_outbox.events() if e["event"] == "receipt_committed"]
    assert len(committed) == 1, "recovery must leave exactly one logical receipt"


def test_crash_after_the_artifact_write_rolls_forward(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    first = ReceiptOutbox(path)
    first.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    first.mark_artifact_written("step_1")
    # Crash before receipt_committed. The artifact is durable, so replay must
    # adopt the crashed attempt instead of minting a second receipt.
    resolution = ReceiptOutbox(path).prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_b",
        artifact_id="art_b",
        artifact_exists=lambda artifact_id: artifact_id == "art_a",
    )
    assert resolution.phase == "committed"
    assert resolution.receipt_id == "rcpt_a"
    assert resolution.artifact_id == "art_a"


def test_prepared_step_with_a_durable_artifact_rolls_forward_without_marker(
    tmp_path: Path,
) -> None:
    """Crash between the artifact os.replace and the artifact_written record."""
    path = tmp_path / "outbox.jsonl"
    ReceiptOutbox(path).prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    resolution = ReceiptOutbox(path).prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_b",
        artifact_id="art_b",
        artifact_exists=lambda artifact_id: artifact_id == "art_a",
    )
    assert resolution.phase == "committed"
    assert resolution.receipt_id == "rcpt_a"


def test_reconcile_repairs_every_pending_step_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    outbox = ReceiptOutbox(path)
    outbox.prepare(
        logical_step_id="step_orphan",
        receipt_id="rcpt_o",
        artifact_id="art_o",
        artifact_exists=_never_exists,
    )
    outbox.prepare(
        logical_step_id="step_durable",
        receipt_id="rcpt_d",
        artifact_id="art_d",
        artifact_exists=_never_exists,
    )
    outbox.mark_artifact_written("step_durable")

    recovered = ReceiptOutbox(path)
    report = recovered.reconcile(lambda artifact_id: artifact_id == "art_d")
    assert report.rolled_forward == ["step_durable"]
    assert report.aborted == ["step_orphan"]
    durable = recovered.state("step_durable")
    orphan = recovered.state("step_orphan")
    assert durable is not None and durable.phase == "committed"
    assert orphan is not None and orphan.phase == "aborted"

    second = recovered.reconcile(lambda artifact_id: artifact_id == "art_d")
    assert second.rolled_forward == [] and second.aborted == []


def test_steps_out_of_order_are_rejected(tmp_path: Path) -> None:
    outbox = ReceiptOutbox(tmp_path / "outbox.jsonl")
    with pytest.raises(ReceiptOutboxError):
        outbox.commit("step_unknown")
    with pytest.raises(ReceiptOutboxError):
        outbox.mark_artifact_written("step_unknown")


def test_repeated_marks_and_commits_are_idempotent(tmp_path: Path) -> None:
    outbox = ReceiptOutbox(tmp_path / "outbox.jsonl")
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    outbox.mark_artifact_written("step_1")
    outbox.mark_artifact_written("step_1")
    outbox.commit("step_1")
    outbox.commit("step_1")
    committed = [e for e in outbox.events() if e["event"] == "receipt_committed"]
    assert len(committed) == 1


def test_a_worker_whose_step_was_stolen_cannot_mark_or_commit(tmp_path: Path) -> None:
    """The lock is per call, so two workers can interleave on one logical step.

    Worker A prepared, worker B took the step over, then A completed its own
    protocol and drove B's ids to committed while B's artifact was never
    written. The fence token makes A's late mark/commit fail instead.
    """
    outbox = ReceiptOutbox(tmp_path / "outbox.jsonl")
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_b",
        artifact_id="art_b",
        artifact_exists=_never_exists,
    )
    with pytest.raises(ReceiptOutboxError, match="rcpt_a"):
        outbox.mark_artifact_written("step_1", expected_receipt_id="rcpt_a")
    with pytest.raises(ReceiptOutboxError, match="rcpt_a"):
        outbox.commit("step_1", expected_receipt_id="rcpt_a")

    state = outbox.state("step_1")
    assert state is not None
    assert state.receipt_id == "rcpt_b"
    assert state.phase == "prepared", "the stolen step must not reach committed"
    assert not [e for e in outbox.events() if e["event"] == "receipt_committed"]


def test_the_fence_token_admits_the_worker_that_still_owns_the_step(
    tmp_path: Path,
) -> None:
    outbox = ReceiptOutbox(tmp_path / "outbox.jsonl")
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    outbox.mark_artifact_written("step_1", expected_receipt_id="rcpt_a")
    outbox.commit("step_1", expected_receipt_id="rcpt_a")
    state = outbox.state("step_1")
    assert state is not None and state.phase == "committed"
    # Idempotent replay of an already-committed step still passes the fence.
    outbox.mark_artifact_written("step_1", expected_receipt_id="rcpt_a")
    outbox.commit("step_1", expected_receipt_id="rcpt_a")
    assert len([e for e in outbox.events() if e["event"] == "receipt_committed"]) == 1


def test_a_torn_tail_record_is_discarded_on_recovery(tmp_path: Path) -> None:
    path = tmp_path / "outbox.jsonl"
    outbox = ReceiptOutbox(path)
    outbox.prepare(
        logical_step_id="step_1",
        receipt_id="rcpt_a",
        artifact_id="art_a",
        artifact_exists=_never_exists,
    )
    outbox.mark_artifact_written("step_1")
    outbox.commit("step_1")
    with path.open("ab") as handle:
        handle.write(b'{"event": "tool_prepared", "logical_st')  # torn write, no newline
    recovered = ReceiptOutbox(path)
    state = recovered.state("step_1")
    assert state is not None and state.phase == "committed"
    assert len(recovered.events()) == 3

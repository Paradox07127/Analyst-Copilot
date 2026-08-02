"""Append-only outbox making receipt persistence an exactly-once logical commit.

Protocol per logical step: ``tool_prepared`` -> content-addressed artifact
write -> ``artifact_written`` -> ``receipt_committed``. Recovery policy:
a prepare without a durable artifact is aborted and replaced; a durable
artifact without a commit is rolled forward. Every step is idempotent, so a
crash at any boundary followed by a replay of the same logical step converges
on exactly one committed receipt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from eda_platform.core.file_lock import lock_exclusive, unlock

OutboxPhase = Literal["prepared", "artifact_written", "committed", "aborted"]

_EVENT_TYPES = {
    "tool_prepared",
    "artifact_written",
    "receipt_committed",
    "prepare_aborted",
}


class ReceiptOutboxError(RuntimeError):
    """An outbox record sequence or artifact state is inconsistent."""


@dataclass(frozen=True, slots=True)
class OutboxEntry:
    logical_step_id: str
    receipt_id: str
    artifact_id: str
    phase: OutboxPhase


@dataclass(frozen=True, slots=True)
class OutboxResolution:
    """What the caller must do after prepare: proceed, or adopt the recorded ids.

    ``receipt_id`` doubles as the fence token: pass it back as
    ``expected_receipt_id`` so a step taken over by another worker fails loudly.
    """

    phase: Literal["prepared", "committed"]
    receipt_id: str
    artifact_id: str
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    rolled_forward: list[str]
    aborted: list[str]


class ReceiptOutbox:
    """A flushed, fsynced JSONL outbox with crash-tail truncation."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def events(self) -> list[dict[str, Any]]:
        return list(self._read_events())

    def state(self, logical_step_id: str) -> OutboxEntry | None:
        return self._fold().get(logical_step_id)

    def prepare(
        self,
        *,
        logical_step_id: str,
        receipt_id: str,
        artifact_id: str,
        artifact_exists: Callable[[str], bool],
    ) -> OutboxResolution:
        with self._locked():
            entries = self._fold()
            entry = entries.get(logical_step_id)
            if entry is not None and entry.phase != "aborted":
                resolved = self._resolve_pending(entry, artifact_exists)
                if resolved is not None:
                    return resolved
                # The prior prepare left no durable artifact: abort it and
                # let this attempt take over the logical step.
                self._append(
                    "prepare_aborted",
                    logical_step_id=logical_step_id,
                    receipt_id=entry.receipt_id,
                    artifact_id=entry.artifact_id,
                )
            self._append(
                "tool_prepared",
                logical_step_id=logical_step_id,
                receipt_id=receipt_id,
                artifact_id=artifact_id,
            )
            return OutboxResolution(
                phase="prepared",
                receipt_id=receipt_id,
                artifact_id=artifact_id,
                replayed=False,
            )

    def mark_artifact_written(
        self, logical_step_id: str, *, expected_receipt_id: str | None = None
    ) -> None:
        with self._locked():
            entry = self._require(logical_step_id, expected_receipt_id)
            if entry.phase in {"artifact_written", "committed"}:
                return
            if entry.phase != "prepared":
                raise ReceiptOutboxError(
                    f"cannot mark artifact for step in phase {entry.phase!r}."
                )
            self._append(
                "artifact_written",
                logical_step_id=logical_step_id,
                receipt_id=entry.receipt_id,
                artifact_id=entry.artifact_id,
            )

    def commit(
        self, logical_step_id: str, *, expected_receipt_id: str | None = None
    ) -> None:
        with self._locked():
            entry = self._require(logical_step_id, expected_receipt_id)
            if entry.phase == "committed":
                return
            if entry.phase != "artifact_written":
                raise ReceiptOutboxError(
                    f"cannot commit step in phase {entry.phase!r}; the artifact "
                    "must be durable first."
                )
            self._append(
                "receipt_committed",
                logical_step_id=logical_step_id,
                receipt_id=entry.receipt_id,
                artifact_id=entry.artifact_id,
            )

    def reconcile(self, artifact_exists: Callable[[str], bool]) -> ReconcileReport:
        """Repair every pending step: roll durable artifacts forward, abort orphans."""
        rolled_forward: list[str] = []
        aborted: list[str] = []
        with self._locked():
            for step_id, entry in self._fold().items():
                if entry.phase not in {"prepared", "artifact_written"}:
                    continue
                if self._resolve_pending(entry, artifact_exists) is not None:
                    rolled_forward.append(step_id)
                else:
                    self._append(
                        "prepare_aborted",
                        logical_step_id=step_id,
                        receipt_id=entry.receipt_id,
                        artifact_id=entry.artifact_id,
                    )
                    aborted.append(step_id)
        return ReconcileReport(rolled_forward=rolled_forward, aborted=aborted)

    def _resolve_pending(
        self,
        entry: OutboxEntry,
        artifact_exists: Callable[[str], bool],
    ) -> OutboxResolution | None:
        """Roll a pending entry forward if its artifact is durable; else None."""
        if entry.phase == "committed":
            return OutboxResolution(
                phase="committed",
                receipt_id=entry.receipt_id,
                artifact_id=entry.artifact_id,
                replayed=True,
            )
        if artifact_exists(entry.artifact_id):
            if entry.phase == "prepared":
                self._append(
                    "artifact_written",
                    logical_step_id=entry.logical_step_id,
                    receipt_id=entry.receipt_id,
                    artifact_id=entry.artifact_id,
                )
            self._append(
                "receipt_committed",
                logical_step_id=entry.logical_step_id,
                receipt_id=entry.receipt_id,
                artifact_id=entry.artifact_id,
            )
            return OutboxResolution(
                phase="committed",
                receipt_id=entry.receipt_id,
                artifact_id=entry.artifact_id,
                replayed=True,
            )
        if entry.phase == "artifact_written":
            raise ReceiptOutboxError(
                f"artifact {entry.artifact_id!r} was recorded durable but is "
                "missing; refusing to guess."
            )
        return None

    def _require(
        self, logical_step_id: str, expected_receipt_id: str | None = None
    ) -> OutboxEntry:
        entry = self._fold().get(logical_step_id)
        if entry is None or entry.phase == "aborted":
            raise ReceiptOutboxError(
                f"no pending prepare for logical step {logical_step_id!r}."
            )
        # The lock is held per call, not for the whole prepare -> write -> commit
        # sequence, so a second worker's prepare can take the step over in
        # between. Without this fence the first worker's late mark/commit drives
        # the *new* entry to committed and vouches for an artifact it never wrote.
        if expected_receipt_id is not None and entry.receipt_id != expected_receipt_id:
            raise ReceiptOutboxError(
                f"logical step {logical_step_id!r} was taken over: expected "
                f"receipt {expected_receipt_id!r}, found {entry.receipt_id!r}."
            )
        return entry

    def _fold(self) -> dict[str, OutboxEntry]:
        entries: dict[str, OutboxEntry] = {}
        for event in self._read_events():
            kind = event["event"]
            step_id = event["logical_step_id"]
            current = entries.get(step_id)
            if kind == "tool_prepared":
                if current is not None and current.phase != "aborted":
                    raise ReceiptOutboxError(
                        f"duplicate tool_prepared for step {step_id!r}."
                    )
                entries[step_id] = OutboxEntry(
                    logical_step_id=step_id,
                    receipt_id=event["receipt_id"],
                    artifact_id=event["artifact_id"],
                    phase="prepared",
                )
                continue
            if current is None or current.phase == "aborted":
                raise ReceiptOutboxError(
                    f"{kind} without an active prepare for step {step_id!r}."
                )
            if event.get("receipt_id") != current.receipt_id:
                raise ReceiptOutboxError(
                    f"{kind} receipt id does not match the prepared step {step_id!r}."
                )
            if kind == "artifact_written":
                if current.phase == "prepared":
                    entries[step_id] = replace(current, phase="artifact_written")
            elif kind == "receipt_committed":
                entries[step_id] = replace(current, phase="committed")
            elif kind == "prepare_aborted":
                if current.phase == "committed":
                    raise ReceiptOutboxError(
                        f"cannot abort committed step {step_id!r}."
                    )
                entries[step_id] = replace(current, phase="aborted")
            else:
                raise ReceiptOutboxError(f"unsupported outbox event {kind!r}.")
        return entries

    def _append(self, event: str, **fields: Any) -> None:
        if event not in _EVENT_TYPES:
            raise ReceiptOutboxError(f"unsupported outbox event {event!r}.")
        record = {
            "event": event,
            "created_at": datetime.now(UTC).isoformat(),
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._truncate_torn_tail()
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _truncate_torn_tail(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        with self.path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_events(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw:
            return
        lines = raw.split(b"\n")
        has_trailing_newline = raw.endswith(b"\n")
        last_index = len(lines) - (2 if has_trailing_newline else 1)
        for index, line in enumerate(lines):
            if has_trailing_newline and index == len(lines) - 1:
                continue
            is_torn_tail = not has_trailing_newline and index == last_index
            if not line:
                if is_torn_tail:
                    continue
                raise ReceiptOutboxError(
                    f"blank committed outbox record at line {index + 1} in {self.path}."
                )
            try:
                event = json.loads(line)
            except ValueError as exc:
                if is_torn_tail:
                    return
                raise ReceiptOutboxError(
                    f"invalid committed outbox record at line {index + 1} in "
                    f"{self.path}: {exc}"
                ) from exc
            yield event

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            lock_exclusive(handle.fileno())
            try:
                yield
            finally:
                unlock(handle.fileno())

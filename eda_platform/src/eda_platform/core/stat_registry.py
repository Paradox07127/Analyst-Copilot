"""System-owned allocator of statistical family sequence indices.

The registry is the only party that may assign ``sequence_index``, derive
comparison counts, or name a family. Every attempt — including failures — is
recorded, so selective reporting cannot shrink the multiplicity ledger.
Replaying the journal reproduces identical numbering; a journal whose numbering
was rolled back or forged fails closed on load. Allocation is fenced by an
exclusive file lock covering re-read, numbering and append together.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from eda_platform.core.file_lock import lock_exclusive, unlock
from eda_platform.core.ids import stable_hash

AttemptStatus = Literal["running", "completed", "failed"]


class StatRegistryError(RuntimeError):
    """An allocation, outcome or journal record is inconsistent."""


def derive_family_id(*, dataset_id: str, columns: Iterable[str]) -> str:
    """Family id for the comparisons on one dataset and one set of columns.

    Derived, never supplied: a model that could name its own family could mint a
    fresh one per test and drive every comparison count back to 1.
    """
    return "fam_" + stable_hash(
        {"dataset_id": dataset_id, "columns": sorted({str(c) for c in columns})},
        length=24,
    )


@dataclass(frozen=True, slots=True)
class StatAttempt:
    attempt_id: str
    family_id: str
    sequence_index: int
    logical_step_id: str
    requested_test_type: str
    arguments_digest: str
    status: AttemptStatus = "running"
    receipt_id: str | None = None
    error: str | None = None


class StatTestRegistry:
    """Append-only attempt ledger; in-memory when no path is given."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.lock_path = (
            None if self.path is None else self.path.with_suffix(self.path.suffix + ".lock")
        )
        self._attempts: dict[str, StatAttempt] = {}
        self._by_step: dict[str, str] = {}
        self._family_order: dict[str, list[str]] = {}
        self._lock_depth = 0
        self._thread_lock = threading.RLock()
        with self._locked():
            self._reload()

    def begin_attempt(
        self,
        *,
        family_id: str,
        requested_test_type: str,
        arguments_digest: str,
        logical_step_id: str,
    ) -> StatAttempt:
        with self._locked():
            self._reload()
            return self._begin_attempt_locked(
                family_id=family_id,
                requested_test_type=requested_test_type,
                arguments_digest=arguments_digest,
                logical_step_id=logical_step_id,
            )

    def _begin_attempt_locked(
        self,
        *,
        family_id: str,
        requested_test_type: str,
        arguments_digest: str,
        logical_step_id: str,
    ) -> StatAttempt:
        existing_id = self._by_step.get(logical_step_id)
        if existing_id is not None:
            existing = self._attempts[existing_id]
            if (
                existing.family_id != family_id
                or existing.arguments_digest != arguments_digest
                or existing.requested_test_type != requested_test_type
            ):
                raise StatRegistryError(
                    f"logical step {logical_step_id!r} was already allocated for "
                    "a different family or argument set."
                )
            return existing
        sequence_index = len(self._family_order.get(family_id, [])) + 1
        attempt_id = "att_" + stable_hash(
            {
                "family_id": family_id,
                "sequence_index": sequence_index,
                "logical_step_id": logical_step_id,
            },
            length=24,
        )
        event = {
            "event": "attempt_started",
            "attempt_id": attempt_id,
            "family_id": family_id,
            "sequence_index": sequence_index,
            "logical_step_id": logical_step_id,
            "requested_test_type": requested_test_type,
            "arguments_digest": arguments_digest,
        }
        self._apply(event)
        self._append(event)
        return self._attempts[attempt_id]

    def record_completion(self, attempt_id: str, *, receipt_id: str) -> StatAttempt:
        with self._locked():
            self._reload()
            return self._record_completion_locked(attempt_id, receipt_id=receipt_id)

    def _record_completion_locked(self, attempt_id: str, *, receipt_id: str) -> StatAttempt:
        attempt = self._require(attempt_id)
        if attempt.status == "completed":
            if attempt.receipt_id != receipt_id:
                raise StatRegistryError(
                    f"attempt {attempt_id!r} is already completed with a "
                    "different receipt."
                )
            return attempt
        if attempt.status == "failed":
            raise StatRegistryError(
                f"attempt {attempt_id!r} already failed; outcomes cannot be rewritten."
            )
        event = {
            "event": "attempt_completed",
            "attempt_id": attempt_id,
            "receipt_id": receipt_id,
        }
        self._apply(event)
        self._append(event)
        return self._attempts[attempt_id]

    def record_failure(self, attempt_id: str, *, error: str) -> StatAttempt:
        with self._locked():
            self._reload()
            return self._record_failure_locked(attempt_id, error=error)

    def _record_failure_locked(self, attempt_id: str, *, error: str) -> StatAttempt:
        attempt = self._require(attempt_id)
        if attempt.status == "failed":
            return attempt
        if attempt.status == "completed":
            raise StatRegistryError(
                f"attempt {attempt_id!r} is completed; a completion cannot be "
                "unwound into a failure."
            )
        event = {
            "event": "attempt_failed",
            "attempt_id": attempt_id,
            "error": error,
        }
        self._apply(event)
        self._append(event)
        return self._attempts[attempt_id]

    def comparison_count(self, family_id: str) -> int:
        """Every allocated attempt in the family, failures included."""
        with self._locked():
            self._reload()
            return len(self._family_order.get(family_id, []))

    def attempts(self, family_id: str | None = None) -> list[StatAttempt]:
        with self._locked():
            self._reload()
            if family_id is None:
                return list(self._attempts.values())
            return [
                self._attempts[attempt_id]
                for attempt_id in self._family_order.get(family_id, [])
            ]

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive fence over the whole read-number-append cycle.

        Re-entrant: flock is held per open file description, so a nested second
        open in this process would block against the outer hold forever.
        """
        with self._thread_lock:
            if self.lock_path is None or self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as handle:
                lock_exclusive(handle.fileno())
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    unlock(handle.fileno())

    def _reload(self) -> None:
        """Rebuild state from the journal; another writer may have appended."""
        if self.path is None:
            return
        self._attempts = {}
        self._by_step = {}
        self._family_order = {}
        if not self.path.exists():
            return
        for event in self._read_events():
            self._apply(event)

    def _require(self, attempt_id: str) -> StatAttempt:
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            raise StatRegistryError(f"unknown stat attempt {attempt_id!r}.")
        return attempt

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "attempt_started":
            attempt_id = event["attempt_id"]
            family_id = event["family_id"]
            if attempt_id in self._attempts:
                raise StatRegistryError(f"duplicate attempt id {attempt_id!r} in journal.")
            if event["logical_step_id"] in self._by_step:
                raise StatRegistryError(
                    f"duplicate logical step {event['logical_step_id']!r} in journal."
                )
            expected = len(self._family_order.get(family_id, [])) + 1
            if event["sequence_index"] != expected:
                raise StatRegistryError(
                    f"family {family_id!r} sequence must be {expected}, got "
                    f"{event['sequence_index']}; the journal was forged or rolled back."
                )
            attempt = StatAttempt(
                attempt_id=attempt_id,
                family_id=family_id,
                sequence_index=event["sequence_index"],
                logical_step_id=event["logical_step_id"],
                requested_test_type=event["requested_test_type"],
                arguments_digest=event["arguments_digest"],
            )
            self._attempts[attempt_id] = attempt
            self._by_step[attempt.logical_step_id] = attempt_id
            self._family_order.setdefault(family_id, []).append(attempt_id)
        elif kind == "attempt_completed":
            attempt = self._require(event["attempt_id"])
            self._attempts[attempt.attempt_id] = replace(
                attempt, status="completed", receipt_id=event["receipt_id"]
            )
        elif kind == "attempt_failed":
            attempt = self._require(event["attempt_id"])
            self._attempts[attempt.attempt_id] = replace(
                attempt, status="failed", error=event.get("error")
            )
        else:
            raise StatRegistryError(f"unsupported registry event {kind!r}.")

    def _append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self._locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._truncate_torn_tail()
            encoded = json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8")
            with self.path.open("ab") as handle:
                handle.write(encoded + b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _truncate_torn_tail(self) -> None:
        if self.path is None or not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        with self.path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_events(self) -> list[dict[str, Any]]:
        assert self.path is not None
        raw = self.path.read_bytes()
        if not raw:
            return []
        events: list[dict[str, Any]] = []
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
                raise StatRegistryError(
                    f"blank committed registry record at line {index + 1} in {self.path}."
                )
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                if is_torn_tail:
                    break
                raise StatRegistryError(
                    f"invalid committed registry record at line {index + 1} in "
                    f"{self.path}: {exc}"
                ) from exc
        return events

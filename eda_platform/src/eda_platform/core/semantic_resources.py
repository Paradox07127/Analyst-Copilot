"""Recoverable, cross-process repository for project semantic seeds.

SQLite owns the committed version and content digest. ``seeds.json`` and
``versions.json`` remain the portable representation, but every replacement is
one multi-file :class:`StorageOperationJournal` transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from eda_platform.core.meaning_proposals import (
    MeaningProposal,
    MeaningProposals,
)
from eda_platform.core.semantic import SemanticSeeds
from eda_platform.core.storage_operations import (
    ReplacementFile,
    ReplacementReplayRecord,
    ResourceDigestMismatchError,
    ResourceHead,
    ResourceOperationInProgressError,
    ResourceTarget,
    ResourceVersionConflictError,
    StorageOperationError,
    StorageOperationJournal,
    StorageRequestKeyConflictError,
    canonical_digest,
    canonical_json_bytes,
    composite_digest,
)
from eda_platform.core.store import ArtifactStore

_RESOURCE_KIND = "semantic_seeds"
_RESOURCE_KEY = "seeds"
_MAX_SEMANTIC_FILE_BYTES = 16 * 1024 * 1024
_MISSING_DIGEST = hashlib.sha256(
    b"eda-platform:missing-json-resource:v1"
).hexdigest()


class SemanticResourceStateError(StorageOperationError):
    """Legacy or committed semantic JSON is missing required structure."""


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    seeds: SemanticSeeds
    proposals: MeaningProposals
    version: int
    content_digest: str


@dataclass(frozen=True, slots=True)
class _SemanticFiles:
    seeds: SemanticSeeds
    proposals: MeaningProposals
    versions: dict[str, Any]
    version: int
    content_digest: str


class SemanticSeedsRepository:
    """Versioned repository for ``semantic/seeds.json`` + ``versions.json``."""

    def __init__(
        self,
        store: ArtifactStore,
        project_id: str,
        *,
        journal: StorageOperationJournal | None = None,
    ) -> None:
        project_dir = store.project_dir(project_id)
        relative_project = project_dir.resolve().relative_to(store.root.resolve())
        semantic_root = relative_project / "semantic"
        self._root = store.root.resolve()
        self._project_id = project_id
        self._seeds_relative = str(semantic_root / "seeds.json")
        self._versions_relative = str(semantic_root / "versions.json")
        self._proposals_relative = str(
            semantic_root / "meaning_proposals.json"
        )
        self._target = ResourceTarget(_RESOURCE_KIND, project_id, _RESOURCE_KEY)
        self._journal = journal or StorageOperationJournal(store)
        self._db_path = store.db_path
        self._ensure_command_schema()

    def source_fingerprint(self) -> tuple[tuple[str, int, int] | tuple[str], ...]:
        """Return the repository-owned files that invalidate semantic read pages."""
        stats: list[tuple[str, int, int] | tuple[str]] = []
        for relative in (
            self._seeds_relative,
            self._versions_relative,
            self._proposals_relative,
        ):
            path = self._root / relative
            try:
                stat = path.stat()
                stats.append((relative, stat.st_size, stat.st_mtime_ns))
            except OSError:
                stats.append((f"{relative}:missing",))
        return tuple(stats)

    def read(self) -> SemanticSnapshot:
        """Recover any interrupted write, then return one verified generation."""
        head = self._journal.recover_target(self._target)
        if head is None:
            legacy = self._read_files()
            head = self._journal.bootstrap_head(
                self._target,
                primary_relative_path=self._seeds_relative,
                version=legacy.version,
                tracked_relative_paths=(
                    self._seeds_relative,
                    self._versions_relative,
                    self._proposals_relative,
                ),
                expected_content_digest=legacy.content_digest,
            )
        return self._snapshot_for_head(head)

    def committed_head(self) -> ResourceHead | None:
        """Cheap source identity for read projections; does not parse JSON."""
        return self._journal.get_head(self._target)

    def replace_seeds(
        self,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        request_key: str,
    ) -> SemanticSnapshot:
        """CAS-replace seeds while preserving proposal state."""
        replay = self._journal.get_replacement_replay(
            _scoped_request_key(self._target, request_key)
        )
        if replay is not None:
            proposal_payload = {
                item.relative_path: item.payload for item in replay.items
            }.get(self._proposals_relative)
            if proposal_payload is None:
                raise SemanticResourceStateError(
                    "Semantic replay is missing proposal state."
                )
            try:
                proposals = MeaningProposals.model_validate_json(
                    proposal_payload
                )
            except ValidationError as exc:
                raise SemanticResourceStateError(
                    "Semantic replay has malformed proposal state."
                ) from exc
            return self.replace_state(
                expected_version=expected_version,
                new_seeds=new_seeds,
                new_proposals=proposals,
                request_key=request_key,
            )
        current = self.read()
        if current.version != expected_version:
            raise ResourceVersionConflictError(expected_version, current.version)
        return self.replace_state(
            expected_version=expected_version,
            new_seeds=new_seeds,
            new_proposals=current.proposals,
            request_key=request_key,
        )

    def replace_state(
        self,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        new_proposals: MeaningProposals,
        request_key: str,
    ) -> SemanticSnapshot:
        """CAS-replace seeds, versions and proposal revisions as one generation."""
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative.")
        if not request_key:
            raise ValueError("request_key must be non-empty.")

        scoped_request_key = _scoped_request_key(self._target, request_key)
        replay = self._journal.get_replacement_replay(scoped_request_key)
        if replay is not None:
            return self._replay_snapshot(
                replay,
                expected_version=expected_version,
                new_seeds=new_seeds,
                new_proposals=new_proposals,
            )

        # Bootstrap only when no committed owner exists. Once a head exists,
        # reserve_replace itself must arbitrate active operations; calling
        # recover_target here would let two writers concurrently "help" the
        # same operation and race its state transition.
        if self._journal.get_head(self._target) is None:
            self.read()
        files = self._read_files()
        versions = dict(files.versions)
        versions["seeds"] = expected_version + 1
        try:
            self._journal.replace_resource(
                self._target,
                expected_version=expected_version,
                replacements=(
                    ReplacementFile(
                        self._seeds_relative,
                        new_seeds.model_dump(mode="json"),
                    ),
                    ReplacementFile(self._versions_relative, versions),
                    ReplacementFile(
                        self._proposals_relative,
                        new_proposals.model_dump(mode="json"),
                    ),
                ),
                request_key=scoped_request_key,
            )
        except StorageRequestKeyConflictError:
            # A same-key writer may have reserved after our first read-only
            # lookup. Re-read the immutable operation record and apply the
            # semantic replay contract instead of comparing against current files.
            replay = self._journal.get_replacement_replay(scoped_request_key)
            if replay is None:
                raise
            return self._replay_snapshot(
                replay,
                expected_version=expected_version,
                new_seeds=new_seeds,
                new_proposals=new_proposals,
            )
        replay = self._journal.get_replacement_replay(scoped_request_key)
        if replay is None:
            raise SemanticResourceStateError(
                "Committed semantic replacement has no replay record."
            )
        return self._snapshot_from_replay_record(replay)

    def replay_seeds(self, *, request_key: str) -> SemanticSnapshot | None:
        """Recover and return a prior content-bound request without reading current files."""

        if not request_key:
            raise ValueError("request_key must be non-empty.")
        replay = self._journal.get_replacement_replay(
            _scoped_request_key(self._target, request_key)
        )
        if replay is None:
            return None
        if replay.target != self._target:
            raise SemanticResourceStateError(
                "Semantic replay record points at a different resource."
            )
        if replay.state in ("prepared", "fs_applied", "blocked"):
            self._journal.recover_replace(replay.op_id)
        elif replay.state != "done":
            raise SemanticResourceStateError(
                f"Semantic replay operation is not recoverable from {replay.state!r}."
            )
        return self._snapshot_from_replay_record(replay)

    def upsert_proposals(
        self,
        drafts: list[MeaningProposal],
        *,
        request_key: str,
    ) -> SemanticSnapshot:
        """Merge AutoEDA drafts through the same three-file CAS generation."""
        for _attempt in range(8):
            snapshot = self.read()
            proposals = snapshot.proposals
            changed = False
            for draft in drafts:
                if not draft.meaning.strip():
                    continue
                existing = proposals.find(draft.dataset, draft.column)
                incoming = draft.model_copy(
                    deep=True,
                    update={"status": "proposed"},
                )
                if existing is None:
                    proposals.proposals.append(incoming)
                    changed = True
                elif existing.status == "proposed" and _proposal_content(
                    existing
                ) != _proposal_content(incoming):
                    proposals.proposals[proposals.proposals.index(existing)] = incoming
                    changed = True
            if not changed:
                return snapshot
            try:
                return self.replace_state(
                    expected_version=snapshot.version,
                    new_seeds=snapshot.seeds,
                    new_proposals=proposals,
                    request_key=f"{request_key}:v{snapshot.version}",
                )
            except ResourceVersionConflictError:
                continue
        raise ResourceOperationInProgressError("semantic-proposal-upsert-contention")

    def execute_approved_answer(
        self,
        *,
        action_hash: str,
        session_id: str,
        generation: str,
        approval_kind: str,
        approval_payload: dict[str, Any],
        candidate: Any,
        request_digest: str,
    ) -> SemanticSnapshot:
        """Atomically bind an approval generation to one prepared semantic operation."""
        existing = self._command_row(action_hash, session_id, generation)
        if existing is not None:
            return self._resume_command(
                existing,
                approval_payload=approval_payload,
                request_digest=request_digest,
            )

        snapshot = self.read()
        seeds = snapshot.seeds
        for index, answer in enumerate(seeds.verified_answers):
            if answer.question == candidate.question:
                seeds.verified_answers[index] = candidate
                break
        else:
            seeds.verified_answers.append(candidate)
        files = self._read_files()
        versions = dict(files.versions)
        versions["seeds"] = snapshot.version + 1
        scoped_key = _scoped_request_key(
            self._target, f"approval:{generation}:{request_digest}"
        )
        prepared = self._journal.prepare_replace(
            self._target,
            expected_version=snapshot.version,
            replacements=(
                ReplacementFile(
                    self._seeds_relative, seeds.model_dump(mode="json")
                ),
                ReplacementFile(self._versions_relative, versions),
                ReplacementFile(
                    self._proposals_relative,
                    snapshot.proposals.model_dump(mode="json"),
                ),
            ),
            request_key=scoped_key,
        )
        now = datetime.now(UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            # A concurrent retry may have inserted after our initial lookup.
            row = self._command_row_in_connection(
                conn, action_hash, session_id, generation
            )
            if row is not None:
                conn.commit()
                return self._resume_command(
                    row,
                    approval_payload=approval_payload,
                    request_digest=request_digest,
                )
            pending = conn.execute(
                """
                select project_id, kind, payload_json, expires_at, status,
                       generation, payload_digest
                from pending_actions
                where action_hash = ? and session_id = ?
                """,
                (action_hash, session_id),
            ).fetchone()
            expected_payload_digest = _approval_payload_digest(approval_payload)
            if (
                pending is None
                or str(pending["project_id"]) != self._project_id
                or str(pending["kind"]) != approval_kind
                or str(pending["generation"]) != generation
                or str(pending["status"]) != "pending"
                or str(pending["expires_at"]) <= now
                or str(pending["payload_digest"]) != expected_payload_digest
                or json.loads(str(pending["payload_json"])) != approval_payload
            ):
                raise SemanticResourceStateError(
                    "Approval generation is not pending or its payload changed."
                )
            reservation = self._journal.reserve_prepared_in_transaction(
                conn, prepared
            )
            conn.execute(
                """
                insert into semantic_commands(
                    command_id, action_hash, session_id, project_id,
                    approval_generation, approval_payload_digest,
                    request_digest, op_id, state, response_json,
                    created_at, updated_at
                ) values(?, ?, ?, ?, ?, ?, ?, ?, 'executing', null, ?, ?)
                """,
                (
                    f"scmd_{uuid4().hex}",
                    action_hash,
                    session_id,
                    self._project_id,
                    generation,
                    expected_payload_digest,
                    request_digest,
                    reservation.op_id,
                    now,
                    now,
                ),
            )
            cursor = conn.execute(
                """
                update pending_actions
                set status = 'executing', consumed_idempotency_key = ?
                where action_hash = ? and session_id = ? and generation = ?
                  and status = 'pending'
                """,
                (reservation.op_id, action_hash, session_id, generation),
            )
            if cursor.rowcount != 1:
                raise SemanticResourceStateError(
                    "Approval changed while semantic command was reserved."
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        self._journal.reservation_committed(prepared.op_id)
        return self._finish_command(
            action_hash,
            session_id,
            generation,
            prepared.op_id,
        )

    def compensate_approved_answer(
        self,
        *,
        action_hash: str,
        session_id: str,
        generation: str,
        op_id: str,
    ) -> bool:
        """Rollback only a still-prepared command bound to this exact generation/op."""
        if not self._journal.replacement_is_reversible(op_id):
            return False
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            operation = conn.execute(
                "select state from storage_operations where op_id = ?",
                (op_id,),
            ).fetchone()
            if operation is None or str(operation["state"]) != "prepared":
                conn.rollback()
                return False
            cursor = conn.execute(
                """
                update semantic_commands
                set state = 'compensated', updated_at = ?
                where action_hash = ? and session_id = ?
                  and approval_generation = ? and op_id = ?
                  and state = 'executing'
                """,
                (
                    datetime.now(UTC).isoformat(),
                    action_hash,
                    session_id,
                    generation,
                    op_id,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return False
            conn.execute(
                """
                update storage_operations set state = 'aborted'
                where op_id = ? and state = 'prepared'
                """,
                (op_id,),
            )
            conn.execute(
                """
                update pending_actions
                set status = 'pending', consumed_idempotency_key = null
                where action_hash = ? and session_id = ? and generation = ?
                  and status = 'executing'
                  and consumed_idempotency_key = ?
                """,
                (action_hash, session_id, generation, op_id),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_command_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                create table if not exists semantic_commands (
                    command_id text primary key,
                    action_hash text not null,
                    session_id text not null,
                    project_id text not null,
                    approval_generation text not null,
                    approval_payload_digest text not null,
                    request_digest text not null,
                    op_id text not null,
                    state text not null,
                    response_json text,
                    created_at text not null,
                    updated_at text not null,
                    unique(action_hash, session_id, approval_generation),
                    unique(op_id),
                    foreign key(op_id) references storage_operations(op_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        return conn

    def _command_row(
        self, action_hash: str, session_id: str, generation: str
    ) -> sqlite3.Row | None:
        conn = self._connect()
        try:
            return self._command_row_in_connection(
                conn, action_hash, session_id, generation
            )
        finally:
            conn.close()

    @staticmethod
    def _command_row_in_connection(
        conn: sqlite3.Connection,
        action_hash: str,
        session_id: str,
        generation: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            select action_hash, session_id, project_id, approval_generation,
                   approval_payload_digest, request_digest, op_id, state,
                   response_json
            from semantic_commands
            where action_hash = ? and session_id = ? and approval_generation = ?
            """,
            (action_hash, session_id, generation),
        ).fetchone()

    def _resume_command(
        self,
        row: sqlite3.Row,
        *,
        approval_payload: dict[str, Any],
        request_digest: str,
    ) -> SemanticSnapshot:
        if (
            str(row["project_id"]) != self._project_id
            or str(row["approval_payload_digest"])
            != _approval_payload_digest(approval_payload)
            or str(row["request_digest"]) != request_digest
        ):
            raise StorageRequestKeyConflictError(
                "Approval generation is bound to a different semantic command."
            )
        state = str(row["state"])
        response_json = row["response_json"]
        if state == "committed" and isinstance(response_json, str):
            return _snapshot_from_response(response_json)
        if state != "executing":
            raise SemanticResourceStateError(
                f"Semantic command cannot resume from state {state!r}."
            )
        return self._finish_command(
            str(row["action_hash"]),
            str(row["session_id"]),
            str(row["approval_generation"]),
            str(row["op_id"]),
        )

    def _finish_command(
        self,
        action_hash: str,
        session_id: str,
        generation: str,
        op_id: str,
    ) -> SemanticSnapshot:
        self._journal.recover_replace(op_id)
        snapshot = self._snapshot_from_replay_record(
            self._journal.get_replacement_replay_by_op(op_id)
        )
        response_json = json.dumps(
            {
                "seeds": snapshot.seeds.model_dump(mode="json"),
                "proposals": snapshot.proposals.model_dump(mode="json"),
                "version": snapshot.version,
                "content_digest": snapshot.content_digest,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        now = datetime.now(UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            cursor = conn.execute(
                """
                update semantic_commands
                set state = 'committed', response_json = ?, updated_at = ?
                where action_hash = ? and session_id = ?
                  and approval_generation = ? and op_id = ?
                  and state in ('executing', 'committed')
                """,
                (
                    response_json,
                    now,
                    action_hash,
                    session_id,
                    generation,
                    op_id,
                ),
            )
            if cursor.rowcount != 1:
                raise SemanticResourceStateError(
                    "Semantic command binding changed before finalization."
                )
            conn.execute(
                """
                update pending_actions
                set status = 'consumed'
                where action_hash = ? and session_id = ? and generation = ?
                  and status = 'executing'
                  and consumed_idempotency_key = ?
                """,
                (action_hash, session_id, generation, op_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return snapshot

    def _replay_snapshot(
        self,
        replay: ReplacementReplayRecord,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        new_proposals: MeaningProposals,
    ) -> SemanticSnapshot:
        if replay.target != self._target or replay.expected_version != expected_version:
            raise StorageRequestKeyConflictError(
                "The semantic request key is already bound to a different target or version."
            )
        items = {item.relative_path: item.payload for item in replay.items}
        original_seeds = items.get(self._seeds_relative)
        original_proposals = items.get(self._proposals_relative)
        if original_seeds is None or canonical_digest(original_seeds) != canonical_digest(
            new_seeds.model_dump(mode="json")
        ):
            raise StorageRequestKeyConflictError(
                "The semantic request key is already bound to different seeds."
            )
        if (
            original_proposals is None
            or canonical_digest(original_proposals)
            != canonical_digest(new_proposals.model_dump(mode="json"))
        ):
            raise StorageRequestKeyConflictError(
                "The semantic request key is already bound to different proposals."
            )
        if replay.state in ("prepared", "fs_applied", "blocked"):
            self._journal.recover_replace(replay.op_id)
        elif replay.state != "done":
            raise SemanticResourceStateError(
                f"Semantic replay operation is not recoverable from {replay.state!r}."
            )
        return self._snapshot_from_replay_record(replay)

    def _snapshot_from_replay_record(
        self, replay: ReplacementReplayRecord
    ) -> SemanticSnapshot:
        items = {item.relative_path: item.payload for item in replay.items}
        if set(items) != {
            self._seeds_relative,
            self._versions_relative,
            self._proposals_relative,
        }:
            raise SemanticResourceStateError(
                "Semantic replay record has an unexpected replacement item set."
            )
        try:
            seeds = SemanticSeeds.model_validate_json(items[self._seeds_relative])
            versions = json.loads(items[self._versions_relative])
            proposals = MeaningProposals.model_validate_json(
                items[self._proposals_relative]
            )
        except (ValidationError, UnicodeError, ValueError) as exc:
            raise SemanticResourceStateError(
                "Semantic replay record contains malformed JSON."
            ) from exc
        if (
            not isinstance(versions, dict)
            or versions.get("seeds") != replay.target_version
        ):
            raise SemanticResourceStateError(
                "Semantic replay record has an inconsistent target version."
            )
        content_digest = composite_digest(
            {
                self._seeds_relative: canonical_digest(
                    items[self._seeds_relative]
                ),
                self._versions_relative: canonical_digest(
                    items[self._versions_relative]
                ),
                self._proposals_relative: canonical_digest(
                    items[self._proposals_relative]
                ),
            }
        )
        if content_digest != replay.target_digest:
            raise SemanticResourceStateError(
                "Semantic replay record does not match its target digest."
            )
        return SemanticSnapshot(
            seeds=SemanticSeeds.model_validate_json(seeds.model_dump_json()),
            proposals=MeaningProposals.model_validate_json(
                proposals.model_dump_json()
            ),
            version=replay.target_version,
            content_digest=replay.target_digest,
        )

    def _snapshot_for_head(self, head: ResourceHead) -> SemanticSnapshot:
        if head.relative_path != self._seeds_relative:
            raise SemanticResourceStateError(
                "Semantic resource head points at an unexpected primary path."
            )
        files = self._read_files()
        if files.content_digest != head.content_digest:
            raise ResourceDigestMismatchError(
                "Semantic files differ from their committed SQLite content digest."
            )
        if files.version != head.version:
            raise SemanticResourceStateError(
                "versions.json does not match the committed semantic version."
            )
        # Deep validation copy ensures an earlier snapshot never follows a
        # later repository generation through shared model/list references.
        seeds = SemanticSeeds.model_validate_json(files.seeds.model_dump_json())
        return SemanticSnapshot(
            seeds=seeds,
            proposals=MeaningProposals.model_validate_json(
                files.proposals.model_dump_json()
            ),
            version=head.version,
            content_digest=head.content_digest,
        )

    def _read_files(self) -> _SemanticFiles:
        seeds_payload, seeds_digest = self._read_json(
            self._seeds_relative,
            missing=SemanticSeeds().model_dump(mode="json"),
        )
        versions_payload, versions_digest = self._read_json(
            self._versions_relative,
            missing={},
        )
        proposals_payload, proposals_digest = self._read_json(
            self._proposals_relative,
            missing=MeaningProposals().model_dump(mode="json"),
        )
        try:
            seeds = SemanticSeeds.model_validate(seeds_payload)
            proposals = MeaningProposals.model_validate(proposals_payload)
        except ValidationError as exc:
            raise SemanticResourceStateError(
                "semantic seeds or meaning proposals are malformed."
            ) from exc
        if not isinstance(versions_payload, dict):
            raise SemanticResourceStateError(
                "semantic/versions.json must contain a JSON object."
            )
        raw_version = versions_payload.get("seeds", 0)
        if (
            not isinstance(raw_version, int)
            or isinstance(raw_version, bool)
            or raw_version < 0
        ):
            raise SemanticResourceStateError(
                "semantic/versions.json has an invalid seeds version."
            )
        if seeds_digest == _MISSING_DIGEST and raw_version > 0:
            raise SemanticResourceStateError(
                "semantic/seeds.json is missing for a non-zero committed version."
            )
        content_digest = composite_digest(
            {
                self._seeds_relative: seeds_digest,
                self._versions_relative: versions_digest,
                self._proposals_relative: proposals_digest,
            }
        )
        return _SemanticFiles(
            seeds=seeds,
            proposals=proposals,
            versions={str(key): value for key, value in versions_payload.items()},
            version=raw_version,
            content_digest=content_digest,
        )

    def _read_json(
        self,
        relative_path: str,
        *,
        missing: object,
    ) -> tuple[object, str]:
        path = self._root / relative_path
        if not path.exists():
            return missing, _MISSING_DIGEST
        if path.is_symlink() or not path.is_file():
            raise SemanticResourceStateError(
                f"{relative_path} is not a regular semantic JSON file."
            )
        try:
            payload = path.read_bytes()
            if len(payload) > _MAX_SEMANTIC_FILE_BYTES:
                raise SemanticResourceStateError(
                    f"{relative_path} exceeds the semantic JSON size limit."
                )
            digest = canonical_digest(payload)
            value = json.loads(payload)
        except (OSError, UnicodeError, ValueError) as exc:
            raise SemanticResourceStateError(
                f"{relative_path} is unreadable or malformed."
            ) from exc
        return value, digest


def _scoped_request_key(target: ResourceTarget, request_key: str) -> str:
    """Bind a caller key to one project resource before the global DB index sees it."""

    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "namespace": "semantic-seeds-request-v1",
                "resource_kind": target.resource_kind,
                "project_id": target.project_id,
                "resource_key": target.resource_key,
                "request_key": request_key,
            }
        )
    ).hexdigest()
    return f"semantic-seeds:{digest}"


def _proposal_content(proposal: MeaningProposal) -> dict[str, object]:
    payload = proposal.model_dump(mode="json")
    payload.pop("revision", None)
    payload.pop("status", None)
    return payload


def _approval_payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_from_response(response_json: str) -> SemanticSnapshot:
    try:
        payload = json.loads(response_json)
        return SemanticSnapshot(
            seeds=SemanticSeeds.model_validate(payload["seeds"]),
            proposals=MeaningProposals.model_validate(payload["proposals"]),
            version=int(payload["version"]),
            content_digest=str(payload["content_digest"]),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise SemanticResourceStateError(
            "Semantic command has a malformed persisted response."
        ) from exc


def load_semantic_seeds_safe(
    store: ArtifactStore, project_id: str
) -> SemanticSeeds | None:
    """Recover a semantic generation for runtime context, degrading corrupt state to absent."""

    try:
        return SemanticSeedsRepository(store, project_id).read().seeds
    except (OSError, UnicodeError, ValueError, StorageOperationError):
        return None

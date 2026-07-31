"""Semantic layer use cases (§7.5 / 阶段4 slice H): read the project's
semantic knowledge from one run's viewpoint, and edit it with optimistic
locking.

The seeds edit version is NOT `SemanticSeeds.version` — that field is a schema
tag. SQLite CAS plus the storage journal own seeds concurrency across processes.
A small process lock remains only for adjacent legacy metadata files such as
the join whitelist and proposal review status.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

from eda_platform.application.dto import (
    ColumnRoleRow,
    EntityNoteView,
    FieldMeaningView,
    JoinWhitelistEntryView,
    MeaningProposalView,
    MetricDefinitionView,
    ProposalReviewed,
    SemanticSeedsUpdated,
    SemanticView,
    VerifiedAnswerView,
    VerifiedRelationsUpdated,
    VerifiedRelationView,
)
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    SessionNotFoundError,
)
from eda_platform.core.bounded_pagination import (
    ResourcePageIndex,
    decode_bound_cursor,
    encode_bound_cursor,
    source_token,
)
from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.meaning_proposals import (
    MeaningProposal,
    MeaningProposals,
    apply_proposal_to_seeds,
)
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.semantic import (
    EntityNote,
    FieldMeaning,
    JoinWhitelist,
    JoinWhitelistEntry,
    MetricDefinition,
    SemanticSeeds,
    VerifiedAnswer,
    VerifiedRelation,
    confirm_join,
    load_join_whitelist,
    revoke_auto_confirmation,
    save_join_whitelist,
)
from eda_platform.core.semantic_resources import (
    SemanticResourceStateError,
    SemanticSeedsRepository,
    SemanticSnapshot,
)
from eda_platform.core.storage_operations import (
    ResourceDigestMismatchError,
    ResourceOperationInProgressError,
    ResourceVersionConflictError,
    StorageOperationBlockedError,
    StorageOperationError,
    StorageRequestKeyConflictError,
    canonical_digest,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType, DatasetProfile


class SemanticServiceError(Exception):
    pass


class SemanticVersionConflictError(SemanticServiceError):
    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"Semantic seeds changed since they were loaded: expected version "
            f"{expected_version}, current version is {current_version}. Reload and retry."
        )
        self.expected_version = expected_version
        self.current_version = current_version


class SemanticSeedsOutOfBandError(SemanticServiceError):
    """The seeds file changed on disk mid-save (a writer outside this service)."""

    def __init__(self) -> None:
        super().__init__(
            "Semantic seeds were changed on disk by another writer while saving. "
            "Reload and retry."
        )


class SemanticStateError(SemanticServiceError):
    """The service-owned semantic state on disk is corrupt."""


class SemanticJoinNotFoundError(SemanticServiceError):
    def __init__(self, label: str) -> None:
        super().__init__(f"Unknown join whitelist label: {label}")
        self.label = label


class SemanticJoinStateError(SemanticServiceError):
    pass


class SemanticJoinNotConfirmableError(SemanticServiceError):
    pass


class SemanticProposalNotFoundError(SemanticServiceError):
    def __init__(self, dataset: str, column: str) -> None:
        super().__init__(f"Unknown meaning proposal: {dataset}.{column}")


class SemanticProposalConflictError(SemanticServiceError):
    pass


class SemanticValidationError(SemanticServiceError):
    pass


class SemanticSeedsInvalidError(SemanticServiceError):
    pass


class SemanticService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._metadata_lock = threading.RLock()

    def get_view(
        self, session_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> SemanticView:
        limit = max(1, min(limit, 200))
        project_id = self._project_for_run(session_id)
        repository = self._repository(project_id)
        head = repository.committed_head()
        bootstrap_snapshot: SemanticSnapshot | None = None
        if head is None:
            bootstrap_snapshot = self._read_snapshot(project_id)
            head = repository.committed_head()
        if head is None:
            raise SemanticStateError("Semantic resource head could not be initialized.")
        version = self._semantic_source_version(
            project_id,
            session_id,
            repository=repository,
            head_version=head.version,
            head_digest=head.content_digest,
        )
        scope = f"semantic:{project_id}:{session_id}"
        offset = (
            decode_bound_cursor(cursor, scope=scope, source_version=version)
            if cursor
            else 0
        )
        index = ResourcePageIndex(self._store.db_path)
        if not index.is_current(scope, version):
            snapshot = bootstrap_snapshot or self._read_snapshot(project_id)
            collections = self._semantic_collections(project_id, session_id, snapshot)
            index.replace(
                scope,
                version,
                {
                    name: [item.model_dump_json() for item in items]
                    for name, items in collections.items()
                },
            )
            current_head = repository.committed_head()
            if (
                current_head is None
                or self._semantic_source_version(
                    project_id,
                    session_id,
                    repository=repository,
                    head_version=current_head.version,
                    head_digest=current_head.content_digest,
                )
                != version
            ):
                raise InvalidCursorError
        model_by_collection = {
            "field_meanings": FieldMeaningView,
            "metric_definitions": MetricDefinitionView,
            "entity_notes": EntityNoteView,
            "verified_answers": VerifiedAnswerView,
            "verified_relations": VerifiedRelationView,
            "column_roles": ColumnRoleRow,
            "join_whitelist": JoinWhitelistEntryView,
            "proposals": MeaningProposalView,
        }
        pages = {
            name: index.page(
                scope,
                version,
                name,
                offset=offset,
                limit=limit + 1,
            )
            for name in model_by_collection
        }
        current_head = repository.committed_head()
        if (
            current_head is None
            or self._semantic_source_version(
                project_id,
                session_id,
                repository=repository,
                head_version=current_head.version,
                head_digest=current_head.content_digest,
            )
            != version
        ):
            raise InvalidCursorError
        has_more = any(len(items) > limit for items in pages.values())
        parsed = {
            name: [
                model_by_collection[name].model_validate_json(item)
                for item in items[:limit]
            ]
            for name, items in pages.items()
        }
        consumed = offset + limit
        return SemanticView(
            session_id=session_id,
            project_id=project_id,
            seeds_version=head.version,
            field_meanings=parsed["field_meanings"],
            metric_definitions=parsed["metric_definitions"],
            entity_notes=parsed["entity_notes"],
            verified_answers=parsed["verified_answers"],
            verified_relations=parsed["verified_relations"],
            column_roles=parsed["column_roles"],
            join_whitelist=parsed["join_whitelist"],
            proposals=parsed["proposals"],
            next_cursor=(
                encode_bound_cursor(
                    consumed,
                    scope=scope,
                    source_version=version,
                )
                if has_more
                else None
            ),
        )

    def _semantic_collections(
        self,
        project_id: str,
        session_id: str,
        snapshot: SemanticSnapshot,
    ) -> dict[str, list]:
        project_dir = self._store.project_dir(project_id)
        seeds = snapshot.seeds
        dataset_ids = self._run_dataset_ids(project_id, session_id)
        try:
            whitelist_entries = load_join_whitelist(project_dir).entries
        except ValueError:
            whitelist_entries = []
        proposals = [
            MeaningProposalView(
                dataset=proposal.dataset,
                column=proposal.column,
                meaning=proposal.meaning,
                unit_guess=proposal.unit_guess,
                confidence=proposal.confidence,
                source=proposal.source,
            )
            for proposal in snapshot.proposals.proposals
            if proposal.status == "proposed"
        ]
        field_meanings = [_to_field_view(field) for field in seeds.field_meanings]
        metric_definitions = [
            _to_metric_view(metric) for metric in seeds.metric_definitions
        ]
        entity_notes = [_to_entity_view(note) for note in seeds.entity_notes]
        verified_answers = [
            _to_answer_view(answer) for answer in seeds.verified_answers
        ]
        verified_relations = [
            _to_relation_view(relation) for relation in seeds.verified_relations
        ]
        column_roles = self._column_roles(project_id, session_id)
        join_whitelist = [
            _to_join_view(entry, dataset_ids, seeds_version=snapshot.version)
            for entry in whitelist_entries
        ]
        return {
            "field_meanings": field_meanings,
            "metric_definitions": metric_definitions,
            "entity_notes": entity_notes,
            "verified_answers": verified_answers,
            "verified_relations": verified_relations,
            "column_roles": column_roles,
            "join_whitelist": join_whitelist,
            "proposals": proposals,
        }

    def _semantic_source_version(
        self,
        project_id: str,
        session_id: str,
        *,
        repository: SemanticSeedsRepository,
        head_version: int,
        head_digest: str,
    ) -> str:
        semantic_dir = self._store.project_dir(project_id) / "semantic"
        file_stats = list(repository.source_fingerprint())
        join_path = semantic_dir / "join_whitelist.json"
        try:
            join_stat = join_path.stat()
            file_stats.append(("join_whitelist.json", join_stat.st_size, join_stat.st_mtime_ns))
        except OSError:
            file_stats.append(("join_whitelist.json:missing",))
        # Run status, not the artifact index: a run writing profiles would bump
        # an index-derived token on every page and make pagination unreachable.
        # Status is stable while the run produces artifacts and flips once when
        # it settles, which is when the projection must be rebuilt.
        return source_token(
            "semantic-v1",
            project_id,
            session_id,
            head_version,
            head_digest,
            file_stats,
            self._store.get_session_status(session_id),
        )

    def update_seeds(
        self,
        session_id: str,
        *,
        expected_version: int,
        field_meanings: list[FieldMeaningView],
        metric_definitions: list[MetricDefinitionView] | None = None,
        entity_notes: list[EntityNoteView] | None = None,
        verified_answers: list[VerifiedAnswerView] | None = None,
    ) -> SemanticSeedsUpdated:
        """Replace the hand-editable seed classes of one project.

        `field_meanings` is always applied; the three optional classes are only
        touched when supplied, so a client editing one class cannot silently
        clear the others. All four share one edit counter because they share one
        seeds.json.
        """
        project_id = self._project_for_run(session_id)
        _validate_seed_payload(
            field_meanings=field_meanings,
            metric_definitions=metric_definitions,
            entity_notes=entity_notes,
            verified_answers=verified_answers,
        )
        request_key = _semantic_request_key(
            "update",
            {
                "expected_version": expected_version,
                "field_meanings": field_meanings,
                "metric_definitions": metric_definitions,
                "entity_notes": entity_notes,
                "verified_answers": verified_answers,
            },
        )
        snapshot = self._read_snapshot(project_id)
        if expected_version != snapshot.version:
            raise SemanticVersionConflictError(expected_version, snapshot.version)
        seeds = snapshot.seeds
        before = seeds.model_dump(mode="json")
        _apply_seed_payload(
            seeds,
            field_meanings=field_meanings,
            metric_definitions=metric_definitions,
            entity_notes=entity_notes,
            verified_answers=verified_answers,
        )
        if seeds.model_dump(mode="json") == before:
            return _to_seeds_updated(session_id, snapshot)
        committed = self._replace(
            project_id,
            expected_version=expected_version,
            new_seeds=seeds,
            request_key=request_key,
        )
        return _to_seeds_updated(session_id, committed)

    def confirm_whitelist_join(
        self,
        session_id: str,
        label: str,
        *,
        expected_version: int | None = None,
    ) -> JoinWhitelistEntryView:
        project_id = self._project_for_run(session_id)
        project_dir = self._store.project_dir(project_id)
        dataset_ids = self._run_dataset_ids(project_id, session_id)
        with self._project_metadata_fence(project_id):
            snapshot = self._checked_snapshot(project_id, expected_version)
            original = load_join_whitelist(project_dir)
            entry = original.entry(label)
            if entry is None:
                raise SemanticJoinNotFoundError(label)
            changed = entry.status != "confirmed"
            if entry.status != "confirmed":
                self._validate_join_confirmation(entry, dataset_ids)
                whitelist = confirm_join(project_dir, label)
                entry = whitelist.entry(label)
                assert entry is not None
            if changed:
                request_key = _semantic_request_key(
                    "join-confirm",
                    {
                        "expected_version": snapshot.version,
                        "label": label,
                    },
                )
                snapshot = self._replace_with_whitelist_compensation(
                    project_id,
                    expected_version=snapshot.version,
                    new_seeds=snapshot.seeds,
                    request_key=request_key,
                    project_dir=project_dir,
                    original_whitelist=original,
                )
        return _to_join_view(
            entry, dataset_ids, seeds_version=snapshot.version
        )

    def revoke_whitelist_join(
        self,
        session_id: str,
        label: str,
        *,
        expected_version: int | None = None,
    ) -> JoinWhitelistEntryView:
        project_id = self._project_for_run(session_id)
        project_dir = self._store.project_dir(project_id)
        dataset_ids = self._run_dataset_ids(project_id, session_id)
        with self._project_metadata_fence(project_id):
            snapshot = self._checked_snapshot(project_id, expected_version)
            original = load_join_whitelist(project_dir)
            entry = original.entry(label)
            if entry is None:
                raise SemanticJoinNotFoundError(label)
            if entry.status == "confirmed":
                # Existing rule: user-confirmed entries are never downgraded.
                raise SemanticJoinStateError(
                    f"Join {label} was confirmed by a user and cannot be revoked."
                )
            changed = entry.status == "auto_confirmed"
            if entry.status == "auto_confirmed":
                whitelist = revoke_auto_confirmation(project_dir, label)
                entry = whitelist.entry(label)
                assert entry is not None
            # status == "proposed": already revoked — idempotent no-op.
            if changed:
                request_key = _semantic_request_key(
                    "join-revoke",
                    {
                        "expected_version": snapshot.version,
                        "label": label,
                    },
                )
                snapshot = self._replace_with_whitelist_compensation(
                    project_id,
                    expected_version=snapshot.version,
                    new_seeds=snapshot.seeds,
                    request_key=request_key,
                    project_dir=project_dir,
                    original_whitelist=original,
                )
        return _to_join_view(
            entry, dataset_ids, seeds_version=snapshot.version
        )

    def confirm_join_and_sink_relation(
        self,
        session_id: str,
        *,
        label: str,
        left: str,
        right: str,
        cardinality: str | None,
        source_session_id: str | None,
        expected_version: int,
    ) -> JoinWhitelistEntryView:
        """Confirm a whitelist entry and seed its verified relation as one command.

        The whitelist remains a legacy adjacent JSON resource, while semantic
        seeds use the durable CAS journal. A project fence serializes both and
        the original whitelist is restored if the journal commit fails.
        """
        project_id = self._project_for_run(session_id)
        project_dir = self._store.project_dir(project_id)
        dataset_ids = self._run_dataset_ids(project_id, session_id)
        with self._project_metadata_fence(project_id):
            snapshot = self._checked_snapshot(project_id, expected_version)
            original = load_join_whitelist(project_dir)
            entry = original.entry(label)
            if entry is None:
                raise SemanticJoinNotFoundError(label)
            whitelist_changed = entry.status != "confirmed"
            if whitelist_changed:
                self._validate_join_confirmation(entry, dataset_ids)
                whitelist = confirm_join(project_dir, label)
                entry = whitelist.entry(label)
                assert entry is not None

            seeds = snapshot.seeds
            relation_cardinality = cardinality or entry.cardinality
            existing = _find_relation(seeds, left, right)
            relation_changed = not (
                existing is not None
                and existing.cardinality == relation_cardinality
                and existing.source_session_id == source_session_id
            )
            if relation_changed:
                relation = VerifiedRelation(
                    left=left,
                    right=right,
                    cardinality=relation_cardinality,
                    source_session_id=source_session_id,
                )
                if existing is None:
                    seeds.verified_relations.append(relation)
                else:
                    seeds.verified_relations[
                        seeds.verified_relations.index(existing)
                    ] = relation
            if whitelist_changed or relation_changed:
                request_key = _semantic_request_key(
                    "relationship-confirm",
                    {
                        "expected_version": expected_version,
                        "label": label,
                        "left": left,
                        "right": right,
                        "cardinality": relation_cardinality,
                        "source_session_id": source_session_id,
                    },
                )
                snapshot = self._replace_with_whitelist_compensation(
                    project_id,
                    expected_version=snapshot.version,
                    new_seeds=seeds,
                    request_key=request_key,
                    project_dir=project_dir,
                    original_whitelist=original,
                )
        return _to_join_view(
            entry, dataset_ids, seeds_version=snapshot.version
        )

    def sink_verified_relation(
        self,
        session_id: str,
        *,
        left: str,
        right: str,
        cardinality: str,
        source_session_id: str | None = None,
    ) -> VerifiedRelationsUpdated:
        """Persist a user-confirmed relation into the project seeds.

        Confirm a relation into project seeds:
        idempotent on (left, right), so re-confirming replaces rather than
        duplicates. Rewriting identical content does not bump the version, or a
        replayed confirm would 409 an unrelated in-flight seeds editor.
        """
        project_id = self._project_for_run(session_id)
        snapshot = self._read_snapshot(project_id)
        seeds = snapshot.seeds
        existing = _find_relation(seeds, left, right)
        if (
            existing is not None
            and existing.cardinality == cardinality
            and existing.source_session_id == source_session_id
        ):
            return _to_relations_updated(session_id, snapshot.version, seeds)
        relation = VerifiedRelation(
            left=left,
            right=right,
            cardinality=cardinality,
            source_session_id=source_session_id,
        )
        if existing is None:
            seeds.verified_relations.append(relation)
        else:
            seeds.verified_relations[seeds.verified_relations.index(existing)] = relation
        request_key = _semantic_request_key(
            "relation-sink",
            {
                "expected_version": snapshot.version,
                "left": left,
                "right": right,
                "cardinality": cardinality,
                "source_session_id": source_session_id,
            },
        )
        committed = self._replace(
            project_id,
            expected_version=snapshot.version,
            new_seeds=seeds,
            request_key=request_key,
        )
        return _to_relations_updated(session_id, committed.version, committed.seeds)

    def delete_verified_relation(
        self,
        session_id: str,
        *,
        left: str,
        right: str,
        expected_version: int | None = None,
    ) -> VerifiedRelationsUpdated:
        """Remove one relation by its (left, right) identity.

        Keyed on identity rather than list position
        (`seeds.verified_relations.pop(index)`): an index resolved against a
        stale render deletes whichever row happens to sit there now. Deleting a
        row that is already gone is a no-op, so a double submit is idempotent.
        """
        project_id = self._project_for_run(session_id)
        snapshot = self._checked_snapshot(project_id, expected_version)
        seeds = snapshot.seeds
        remaining = [
            relation
            for relation in seeds.verified_relations
            if not (relation.left == left and relation.right == right)
        ]
        if len(remaining) == len(seeds.verified_relations):
            return _to_relations_updated(session_id, snapshot.version, seeds)
        seeds.verified_relations = remaining
        request_key = _semantic_request_key(
            "relation-delete",
            {
                "expected_version": snapshot.version,
                "left": left,
                "right": right,
            },
        )
        committed = self._replace(
            project_id,
            expected_version=snapshot.version,
            new_seeds=seeds,
            request_key=request_key,
        )
        return _to_relations_updated(session_id, committed.version, committed.seeds)

    def accept_meaning_proposal(
        self,
        session_id: str,
        *,
        dataset: str,
        column: str,
        meaning: str | None = None,
        unit: str | None = None,
        expected_version: int | None = None,
    ) -> ProposalReviewed:
        project_id = self._project_for_run(session_id)
        snapshot = self._checked_snapshot(project_id, expected_version)
        proposals = snapshot.proposals
        proposal = proposals.find(dataset, column)
        if proposal is None:
            raise SemanticProposalNotFoundError(dataset, column)
        if proposal.status == "rejected":
            raise SemanticProposalConflictError(
                f"Proposal {dataset}.{column} was rejected and cannot be "
                "accepted; a fresh analysis run must propose it again."
            )
        final_meaning = (proposal.meaning if meaning is None else meaning).strip()
        if not final_meaning:
            raise SemanticValidationError(
                "A meaning is required to accept a proposal."
            )
        final_proposal = proposal.model_copy(deep=True)
        final_proposal.meaning = final_meaning
        if unit is not None:
            final_proposal.unit_guess = unit.strip()
        existing_seed = next(
            (
                field
                for field in snapshot.seeds.field_meanings
                if field.dataset == dataset and field.column == column
            ),
            None,
        )
        if (
            proposal.status == "accepted"
            and proposal.meaning == final_proposal.meaning
            and proposal.unit_guess == final_proposal.unit_guess
        ):
            expected_unit = final_proposal.unit_guess.strip() or None
            if (
                existing_seed is None
                or existing_seed.meaning != final_proposal.meaning
                or existing_seed.unit != expected_unit
            ):
                raise SemanticProposalConflictError(
                    f"Accepted proposal {dataset}.{column} no longer matches "
                    "the current seed generation."
                )
            return _proposal_reviewed(
                session_id, dataset, column, "accepted", snapshot.version
            )
        proposal_unit = proposal.unit_guess.strip() or None
        if existing_seed is not None and proposal.status != "accepted" and (
            existing_seed.meaning != proposal.meaning
            or existing_seed.unit != proposal_unit
        ):
            raise SemanticProposalConflictError(
                f"The seed for {dataset}.{column} was edited by hand and no "
                "longer matches the proposal; edit the field meaning "
                "directly instead of accepting the proposal."
            )
        apply_proposal_to_seeds(snapshot.seeds, final_proposal)
        _mark_proposal(proposal, final_proposal, status="accepted")
        committed = self._replace_state(
            project_id,
            expected_version=snapshot.version,
            new_seeds=snapshot.seeds,
            new_proposals=proposals,
            request_key=_proposal_request_key("accept", final_proposal),
        )
        return _proposal_reviewed(
            session_id, dataset, column, "accepted", committed.version
        )

    def reject_meaning_proposal(
        self,
        session_id: str,
        *,
        dataset: str,
        column: str,
        expected_version: int | None = None,
    ) -> ProposalReviewed:
        project_id = self._project_for_run(session_id)
        snapshot = self._checked_snapshot(project_id, expected_version)
        proposal = snapshot.proposals.find(dataset, column)
        if proposal is None:
            raise SemanticProposalNotFoundError(dataset, column)
        if proposal.status == "rejected":
            return _proposal_reviewed(
                session_id, dataset, column, "rejected", snapshot.version
            )
        if proposal.status == "accepted":
            snapshot.seeds.field_meanings = [
                field
                for field in snapshot.seeds.field_meanings
                if not (
                    field.dataset == dataset
                    and field.column == column
                    and field.meaning == proposal.meaning
                )
            ]
        proposal.status = "rejected"
        committed = self._replace_state(
            project_id,
            expected_version=snapshot.version,
            new_seeds=snapshot.seeds,
            new_proposals=snapshot.proposals,
            request_key=_proposal_request_key("reject", proposal),
        )
        return _proposal_reviewed(
            session_id, dataset, column, "rejected", committed.version
        )

    def accept_all_verified(self, session_id: str) -> int:
        """Accept verified pending drafts in one seeds generation."""
        project_id = self._project_for_run(session_id)
        snapshot = self._read_snapshot(project_id)
        pending = [
            proposal
            for proposal in snapshot.proposals.proposals
            if proposal.status == "proposed"
            and proposal.confidence == "verified"
            and proposal.meaning.strip()
        ]
        if not pending:
            return 0
        for proposal in pending:
            proposal.meaning = proposal.meaning.strip()
            apply_proposal_to_seeds(snapshot.seeds, proposal)
            proposal.status = "accepted"
        self._replace_state(
            project_id,
            expected_version=snapshot.version,
            new_seeds=snapshot.seeds,
            new_proposals=snapshot.proposals,
            request_key=_semantic_request_key(
                "proposal-accept-all",
                [proposal.model_dump(mode="json") for proposal in pending],
            ),
        )
        return len(pending)

    def read_project_snapshot(self, project_id: str) -> SemanticSnapshot:
        """Strict repository read for application services."""
        return self._read_snapshot(project_id)

    def promote_approved_answer(
        self,
        project_id: str,
        *,
        session_id: str,
        action_hash: str,
        generation: str,
        approval_payload: dict[str, object],
        candidate: VerifiedAnswer,
        request_digest: str,
    ) -> SemanticSnapshot:
        """Execute or recover one approval-generation-bound durable command."""
        try:
            return self._repository(project_id).execute_approved_answer(
                action_hash=action_hash,
                session_id=session_id,
                generation=generation,
                approval_kind="knowledge_promote",
                approval_payload=approval_payload,
                candidate=candidate,
                request_digest=request_digest,
            )
        except ResourceVersionConflictError as exc:
            raise SemanticVersionConflictError(
                exc.expected_version, exc.current_version
            ) from exc
        except (
            ResourceDigestMismatchError,
            ResourceOperationInProgressError,
            StorageRequestKeyConflictError,
        ) as exc:
            raise SemanticSeedsOutOfBandError() from exc
        except StorageOperationError as exc:
            raise SemanticStateError(str(exc)) from exc

    def _column_roles(self, project_id: str, session_id: str) -> list[ColumnRoleRow]:
        rows: list[ColumnRoleRow] = []
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.COLUMN_ROLE_SET,),
        ):
            with suppress(ValueError):
                role_set = ColumnRoleSet.model_validate(artifact.payload)
                rows.extend(
                    ColumnRoleRow(
                        dataset=role_set.dataset,
                        column=role.column,
                        role=str(role.role),
                        confidence=role.confidence,
                    )
                    for role in role_set.roles
                )
        return rows

    def _run_dataset_ids(self, project_id: str, session_id: str) -> dict[str, str]:
        """Current-run dataset name → content-derived id, for join freshness."""
        identities: dict[str, str] = {}
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.DATASET_PROFILE,),
        ):
            with suppress(ValueError):
                profile = DatasetProfile.model_validate(artifact.payload)
                if profile.name and profile.dataset_id:
                    identities[profile.name] = profile.dataset_id
        return identities

    def _repository(self, project_id: str) -> SemanticSeedsRepository:
        return SemanticSeedsRepository(self._store, project_id)

    def _read_snapshot(self, project_id: str) -> SemanticSnapshot:
        try:
            return self._repository(project_id).read()
        except StorageOperationError as exc:
            raise SemanticStateError(str(exc)) from exc

    def _checked_snapshot(
        self, project_id: str, expected_version: int | None
    ) -> SemanticSnapshot:
        snapshot = self._read_snapshot(project_id)
        if expected_version is not None and expected_version != snapshot.version:
            raise SemanticVersionConflictError(expected_version, snapshot.version)
        return snapshot

    @contextmanager
    def _project_metadata_fence(self, project_id: str) -> Iterator[None]:
        lock_id = f"semantic_{stable_hash({'project_id': project_id}, length=24)}"
        with session_key_lock(self._store.root, lock_id), self._metadata_lock:
            yield

    @staticmethod
    def _validate_join_confirmation(
        entry: JoinWhitelistEntry, dataset_ids: dict[str, str]
    ) -> None:
        if entry.status == "proposed" and not entry.validation_verified:
            raise SemanticJoinStateError(
                f"Join {entry.label()} has not passed full validation yet; it is "
                "validated automatically during analysis sessions."
            )
        freshness = entry.validation_freshness(dataset_ids)
        if freshness != "fresh":
            raise SemanticJoinNotConfirmableError(
                f"Join {entry.label()} validation is {freshness} for this run's "
                "datasets; re-run analysis on the current uploads first."
            )
        if entry.cardinality == "many_to_many":
            raise SemanticJoinNotConfirmableError(
                f"Join {entry.label()} is many-to-many and cannot be confirmed."
            )

    def _replay(
        self, project_id: str, request_key: str
    ) -> SemanticSnapshot | None:
        try:
            repository = self._repository(project_id)
            # Mutation entry points fail closed on an out-of-band current file,
            # even when their own earlier request has a valid replay record.
            repository.read()
            return repository.replay_seeds(request_key=request_key)
        except ResourceDigestMismatchError as exc:
            raise SemanticSeedsOutOfBandError() from exc
        except StorageOperationError as exc:
            raise SemanticStateError(str(exc)) from exc

    def _replace(
        self,
        project_id: str,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        request_key: str,
    ) -> SemanticSnapshot:
        try:
            return self._repository(project_id).replace_seeds(
                expected_version=expected_version,
                new_seeds=new_seeds,
                request_key=request_key,
            )
        except ResourceVersionConflictError as exc:
            raise SemanticVersionConflictError(
                expected_version, exc.current_version
            ) from exc
        except (
            ResourceDigestMismatchError,
            ResourceOperationInProgressError,
            StorageRequestKeyConflictError,
        ) as exc:
            raise SemanticSeedsOutOfBandError() from exc
        except (
            SemanticResourceStateError,
            StorageOperationBlockedError,
            StorageOperationError,
        ) as exc:
            raise SemanticStateError(str(exc)) from exc

    def _replace_with_whitelist_compensation(
        self,
        project_id: str,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        request_key: str,
        project_dir: Path,
        original_whitelist: JoinWhitelist,
    ) -> SemanticSnapshot:
        """Keep the legacy whitelist aligned with the journaled seed generation."""
        try:
            return self._replace(
                project_id,
                expected_version=expected_version,
                new_seeds=new_seeds,
                request_key=request_key,
            )
        except Exception:
            # A storage call may commit durably and then fail while returning
            # the result. In that case replay proves the new generation owns
            # the whitelist and compensation would itself corrupt the command.
            replay = self._replay(project_id, request_key)
            if replay is not None:
                return replay
            save_join_whitelist(project_dir, original_whitelist)
            raise

    def _replace_state(
        self,
        project_id: str,
        *,
        expected_version: int,
        new_seeds: SemanticSeeds,
        new_proposals: MeaningProposals,
        request_key: str,
    ) -> SemanticSnapshot:
        try:
            return self._repository(project_id).replace_state(
                expected_version=expected_version,
                new_seeds=new_seeds,
                new_proposals=new_proposals,
                request_key=request_key,
            )
        except ResourceVersionConflictError as exc:
            raise SemanticVersionConflictError(
                expected_version, exc.current_version
            ) from exc
        except (
            ResourceDigestMismatchError,
            ResourceOperationInProgressError,
            StorageRequestKeyConflictError,
        ) as exc:
            raise SemanticSeedsOutOfBandError() from exc
        except StorageOperationError as exc:
            raise SemanticStateError(str(exc)) from exc

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _validate_rows[RowT](
    rows: Sequence[RowT],
    *,
    complete: Callable[[RowT], bool],
    incomplete_message: str,
    key: Callable[[RowT], str],
    duplicate_label: str,
) -> None:
    """Reject blank required fields (422 semantic_invalid) and duplicate keys
    (422 semantic_seeds_invalid), the shape every seed class is checked to."""
    for row in rows:
        if not complete(row):
            raise SemanticValidationError(incomplete_message)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        value = key(row)
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise SemanticSeedsInvalidError(
            f"Duplicate {duplicate_label}: " + ", ".join(sorted(set(duplicates)))
        )


def _validate_seed_payload(
    *,
    field_meanings: list[FieldMeaningView],
    metric_definitions: list[MetricDefinitionView] | None,
    entity_notes: list[EntityNoteView] | None,
    verified_answers: list[VerifiedAnswerView] | None,
) -> None:
    _validate_rows(
        field_meanings,
        complete=lambda field: bool(
            field.dataset.strip() and field.column.strip() and field.meaning.strip()
        ),
        incomplete_message=(
            "Dataset, column and meaning are all required for every field meaning."
        ),
        key=lambda field: f"{field.dataset.strip()}.{field.column.strip()}",
        duplicate_label="(dataset, column) field meanings",
    )
    if metric_definitions is not None:
        _validate_rows(
            metric_definitions,
            complete=lambda metric: bool(metric.name.strip() and metric.definition.strip()),
            incomplete_message="Name and definition are required for every metric definition.",
            key=lambda metric: metric.name.strip(),
            duplicate_label="metric definition names",
        )
    if entity_notes is not None:
        _validate_rows(
            entity_notes,
            complete=lambda note: bool(note.name.strip() and note.note.strip()),
            incomplete_message="Entity and note are required for every entity note.",
            key=lambda note: note.name.strip(),
            duplicate_label="entity note names",
        )
    if verified_answers is not None:
        _validate_rows(
            verified_answers,
            complete=lambda answer: bool(answer.question.strip() and answer.answer.strip()),
            incomplete_message="Question and answer are required for every verified answer.",
            key=lambda answer: answer.question.strip(),
            duplicate_label="verified answer questions",
        )


def _apply_seed_payload(
    seeds: SemanticSeeds,
    *,
    field_meanings: list[FieldMeaningView],
    metric_definitions: list[MetricDefinitionView] | None,
    entity_notes: list[EntityNoteView] | None,
    verified_answers: list[VerifiedAnswerView] | None,
) -> None:
    seeds.field_meanings = [
        FieldMeaning(
            dataset=field.dataset.strip(),
            column=field.column.strip(),
            meaning=field.meaning.strip(),
            unit=(field.unit or "").strip() or None,
            aliases=[alias.strip() for alias in field.aliases if alias.strip()],
        )
        for field in field_meanings
    ]
    if metric_definitions is not None:
        seeds.metric_definitions = [
            MetricDefinition(
                name=metric.name.strip(),
                definition=metric.definition.strip(),
                formula=(metric.formula or "").strip() or None,
                caveats=(metric.caveats or "").strip() or None,
            )
            for metric in metric_definitions
        ]
    if entity_notes is not None:
        seeds.entity_notes = [
            EntityNote(name=note.name.strip(), note=note.note.strip())
            for note in entity_notes
        ]
    if verified_answers is not None:
        seeds.verified_answers = [
            VerifiedAnswer(
                question=answer.question.strip(),
                answer=answer.answer.strip(),
                evidence_note=(answer.evidence_note or "").strip() or None,
                # An edit round-trips the original date; a new answer is stamped here.
                verified_at=answer.verified_at or datetime.now(UTC),
            )
            for answer in verified_answers
        ]


def _semantic_request_key(operation: str, payload: object) -> str:
    return f"{operation}:{canonical_digest(_canonical_value(payload))}"


def _canonical_value(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _proposal_request_key(operation: str, proposal: MeaningProposal) -> str:
    return _semantic_request_key(
        f"proposal-{operation}",
        {
            "dataset": proposal.dataset,
            "column": proposal.column,
            "meaning": proposal.meaning,
            "unit_guess": proposal.unit_guess,
            "confidence": proposal.confidence,
            "source": proposal.source,
            "revision": proposal.revision,
        },
    )


def _mark_proposal(
    proposal: MeaningProposal,
    final: MeaningProposal,
    *,
    status: str,
) -> None:
    proposal.meaning = final.meaning
    proposal.unit_guess = final.unit_guess
    proposal.status = status  # type: ignore[assignment]


def _proposal_reviewed(
    session_id: str,
    dataset: str,
    column: str,
    status: str,
    version: int,
) -> ProposalReviewed:
    return ProposalReviewed(
        session_id=session_id,
        dataset=dataset,
        column=column,
        status=status,  # type: ignore[arg-type]
        seeds_version=version,
    )


def _to_seeds_updated(
    session_id: str, snapshot: SemanticSnapshot
) -> SemanticSeedsUpdated:
    seeds = snapshot.seeds
    return SemanticSeedsUpdated(
        session_id=session_id,
        version=snapshot.version,
        field_meanings=[_to_field_view(field) for field in seeds.field_meanings],
        metric_definitions=[
            _to_metric_view(metric) for metric in seeds.metric_definitions
        ],
        entity_notes=[_to_entity_view(note) for note in seeds.entity_notes],
        verified_answers=[
            _to_answer_view(answer) for answer in seeds.verified_answers
        ],
    )


def _to_field_view(field: FieldMeaning) -> FieldMeaningView:
    return FieldMeaningView(
        dataset=field.dataset,
        column=field.column,
        meaning=field.meaning,
        unit=field.unit,
        aliases=list(field.aliases),
    )


def _to_metric_view(metric: MetricDefinition) -> MetricDefinitionView:
    return MetricDefinitionView(
        name=metric.name,
        definition=metric.definition,
        formula=metric.formula,
        caveats=metric.caveats,
    )


def _to_entity_view(note: EntityNote) -> EntityNoteView:
    return EntityNoteView(name=note.name, note=note.note)


def _find_relation(seeds: SemanticSeeds, left: str, right: str) -> VerifiedRelation | None:
    return next(
        (
            relation
            for relation in seeds.verified_relations
            if relation.left == left and relation.right == right
        ),
        None,
    )


def _to_relation_view(relation: VerifiedRelation) -> VerifiedRelationView:
    return VerifiedRelationView(
        left=relation.left,
        right=relation.right,
        cardinality=relation.cardinality,
        confirmed_by=relation.confirmed_by,
        confirmed_at=relation.confirmed_at,
        source_session_id=relation.source_session_id,
    )


def _to_relations_updated(
    session_id: str, version: int, seeds: SemanticSeeds
) -> VerifiedRelationsUpdated:
    return VerifiedRelationsUpdated(
        session_id=session_id,
        seeds_version=version,
        verified_relations=[
            _to_relation_view(relation) for relation in seeds.verified_relations
        ],
    )


def _to_answer_view(answer: VerifiedAnswer) -> VerifiedAnswerView:
    return VerifiedAnswerView(
        question=answer.question,
        answer=answer.answer,
        evidence_note=answer.evidence_note,
        verified_at=answer.verified_at,
    )


def _to_join_view(
    entry: JoinWhitelistEntry,
    dataset_ids_by_name: dict[str, str],
    *,
    seeds_version: int = 0,
) -> JoinWhitelistEntryView:
    return JoinWhitelistEntryView(
        seeds_version=seeds_version,
        label=entry.label(),
        left_dataset=entry.left_dataset,
        left_columns=list(entry.left_columns),
        right_dataset=entry.right_dataset,
        right_columns=list(entry.right_columns),
        status=entry.status,
        cardinality=entry.cardinality,
        validation_verified=entry.validation_verified,
        freshness=entry.validation_freshness(dataset_ids_by_name),
        join_row_multiplier=entry.join_row_multiplier,
        usage_count=entry.usage_count,
        confidence_source=entry.confidence_source,
    )

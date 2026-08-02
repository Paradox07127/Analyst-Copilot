"""Relationship use cases (§7.5 / 页面迁移地图 §10.1).

`get_graph` shapes the run's existing RELATIONSHIP_* artifacts plus the
project's join whitelist into nodes and edges — it never recomputes anything,
so opening the page costs artifact reads only. Full validation does read both
source CSVs, so it goes through the same prepare → approve → job path as
question execution: the job runs on its own derived `rvsess_*` run, while the
validation artifact the driver writes still lands on the source run.

`discover` is the same job shape on a `rdsess_*` run, minus the approval:
`run_auto_eda` defers cross-table discovery by default, so most runs reach the
page with no candidates at all and discovery has to be triggerable from it.
It only reads CSVs and writes candidate artifacts — nothing is promoted to a
join — so an idempotency key is the whole ceremony.

Confirm/revoke are not reimplemented here: they forward to SemanticService,
which owns the join-whitelist rules ("confirmed is never downgraded", "only a
fresh, non-many-to-many validation may be confirmed").
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from eda_platform.application.dto import (
    JobCreated,
    JobStatus,
    RelationshipDiscoveryStarted,
    RelationshipEdge,
    RelationshipGraphView,
    RelationshipNode,
    RelationshipValidationPrepared,
    RelationshipValidationStarted,
)
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.semantic_service import SemanticService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.semantic import JoinWhitelistEntry, load_join_whitelist
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipValidation,
    RelationshipValidationSet,
)

APPROVAL_KIND_RELATIONSHIP = "relationship_validate"
DISCOVER_JOB_KIND = "relationship_discover"
MAX_DISCOVERY_DATASETS = 20
MAX_DISCOVERY_PAIRS = MAX_DISCOVERY_DATASETS * (MAX_DISCOVERY_DATASETS - 1) // 2
VALIDATE_SESSION_PREFIX = "rvsess_"
DISCOVER_SESSION_PREFIX = "rdsess_"
# Both kinds read-modify-write the same run's relationship artifacts, so the
# busy guard treats them as one lane per source run.
RELATIONSHIP_JOB_KINDS = frozenset({APPROVAL_KIND_RELATIONSHIP, DISCOVER_JOB_KIND})
MIN_DISCOVERY_DATASETS = 2
CONFIRMED_JOIN_STATUSES = frozenset({"confirmed", "auto_confirmed"})
# `validate_relationships` silently skips anything below medium, so offering
# Validate on a low-confidence edge would queue a job that can only fail.
VALIDATABLE_CONFIDENCES = frozenset({"high", "medium"})


def _candidate_seed_pair(candidate: RelationshipCandidate) -> tuple[str, str]:
    """Semantic-seed side keys (`dataset.col[+col...]`) for a candidate pair —
    the same shape the semantic seeds layer stores."""
    pair = candidate.pair
    return (
        f"{pair.left_dataset_name}.{'+'.join(pair.left_columns)}",
        f"{pair.right_dataset_name}.{'+'.join(pair.right_columns)}",
    )


def relationship_id_for(label: str) -> str:
    """URL-safe stable id for a pair label (which contains `.`, `+` and ` -> `)."""
    return stable_hash({"pair": label}, length=16)


def latest_candidate_set(
    store: ArtifactStore, project_id: str, session_id: str
) -> RelationshipCandidateSet | None:
    """Newest parseable RelationshipCandidateSet of the run. Shared with the
    worker runner so both sides select the exact same set."""
    artifacts = store.list_indexed_artifacts(
        project_id=project_id,
        session_id=session_id,
        artifact_types=(ArtifactType.RELATIONSHIP_CANDIDATE_SET,),
    )
    for artifact in reversed(artifacts):
        with suppress(ValueError):
            return RelationshipCandidateSet.model_validate(artifact.payload)
    return None


def candidate_fingerprint(candidate: RelationshipCandidate) -> str:
    """Hash over the validation-affecting fields only: a rescored candidate with
    the same columns must not invalidate an approval. Shared with the runner."""
    pair = candidate.pair
    return stable_hash(
        {
            "left_dataset_id": pair.left_dataset_id,
            "left_columns": list(pair.left_columns),
            "right_dataset_id": pair.right_dataset_id,
            "right_columns": list(pair.right_columns),
        },
        length=64,
    )


class RelationshipServiceError(Exception):
    pass


class RelationshipNotFoundError(RelationshipServiceError):
    def __init__(self, relationship_id: str, session_id: str) -> None:
        super().__init__(f"Relationship not found in run {session_id}: {relationship_id}")
        self.relationship_id = relationship_id
        self.session_id = session_id


class RelationshipNotValidatableError(RelationshipServiceError):
    def __init__(self, relationship_id: str, reason: str) -> None:
        super().__init__(f"Relationship {relationship_id} cannot be validated: {reason}")
        self.relationship_id = relationship_id


class RelationshipSourceChangedError(RelationshipServiceError):
    """The candidate changed since prepare; the approval no longer binds it."""

    def __init__(self, relationship_id: str) -> None:
        super().__init__(
            f"Relationship {relationship_id} changed since the approval was "
            "prepared; prepare again and approve the fresh content."
        )
        self.relationship_id = relationship_id


class RelationshipRunBusyError(RelationshipServiceError):
    def __init__(self, session_id: str, job_id: str) -> None:
        super().__init__(
            f"Run {session_id} has an active job ({job_id}); wait for it to finish "
            "before validating relationships."
        )
        self.session_id = session_id
        self.job_id = job_id


class RelationshipNotDiscoverableError(RelationshipServiceError):
    def __init__(self, session_id: str, reason: str) -> None:
        super().__init__(f"Relationship discovery cannot run for {session_id}: {reason}")
        self.session_id = session_id


class RelationshipValidationRequestError(RelationshipServiceError):
    pass


class RelationshipService:
    def __init__(
        self,
        store: ArtifactStore,
        datasets: DatasetService,
        approvals: ApprovalService,
        jobs: JobService,
        semantic: SemanticService,
    ) -> None:
        self._store = store
        self._datasets = datasets
        self._approvals = approvals
        self._jobs = jobs
        self._semantic = semantic

    def get_graph(self, session_id: str) -> RelationshipGraphView:
        project_id = self._project_for_run(session_id)
        seeds_version = self._semantic.read_project_snapshot(project_id).version
        handles = self._datasets.list_datasets(session_id)
        nodes = [
            RelationshipNode(
                dataset_id=handle.dataset_id,
                name=handle.display_name,
                row_count=handle.row_count,
                column_count=len(handle.schema_),
                source_available=handle.ingest_status == "ready",
            )
            for handle in handles
        ]
        candidate_set = self._latest_candidate_set(project_id, session_id)
        if candidate_set is None:
            return RelationshipGraphView(
                session_id=session_id,
                project_id=project_id,
                seeds_version=seeds_version,
                nodes=nodes,
            )
        candidate_artifact_id = self._candidate_artifact_id(project_id, session_id)
        validations, validation_artifact_ids = self._validations(project_id, session_id)
        entries = self._whitelist_entries(project_id)
        dataset_ids_by_name = {node.name: node.dataset_id for node in nodes}
        loadable = {node.dataset_id for node in nodes if node.source_available}
        edges = [
            _to_edge(
                candidate,
                validations.get(candidate.pair.label()),
                entries.get(candidate.pair.label()),
                candidate_artifact_id=candidate_artifact_id,
                validation_artifact_id=validation_artifact_ids.get(candidate.pair.label()),
                dataset_ids_by_name=dataset_ids_by_name,
                loadable=loadable,
            )
            for candidate in candidate_set.candidates
        ]
        edges.sort(key=lambda edge: (-edge.ensemble_score, edge.label))
        return RelationshipGraphView(
            session_id=session_id,
            project_id=project_id,
            seeds_version=seeds_version,
            discovered=True,
            nodes=nodes,
            edges=edges,
            coverage_status=candidate_set.coverage_status,
            coverage_reason=candidate_set.coverage_reason,
            overlap_pairs_evaluated=candidate_set.overlap_pairs_evaluated,
            overlap_pairs_prefiltered=candidate_set.overlap_pairs_prefiltered,
            truncated_pairs=candidate_set.truncated_pairs,
        )

    def discover(
        self, session_id: str, *, idempotency_key: str | None = None
    ) -> RelationshipDiscoveryStarted:
        """Queue on-demand discovery for a run whose analysis deferred it.

        No approval: discovery only reads the source CSVs and writes candidate
        artifacts, it never promotes a join. The heavy scan still goes through
        the job system so the request returns immediately.
        """
        project_id = self._project_for_run(session_id)
        idempotency_content = {"source_session_id": session_id}
        # Same ordering rationale as validate: an idempotent replay must answer
        # before the busy guard, or a retried discover would 409 against the
        # very job it already started.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._check_discover_replay_matches(existing, session_id=session_id)
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=DISCOVER_JOB_KIND,
                    content=idempotency_content,
                )
                return self._discover_replayed(session_id, existing)
        self._require_relationship_lane_free(project_id, session_id)
        self._require_discoverable(session_id)
        rerun = self._latest_candidate_set(project_id, session_id) is not None
        job = self._jobs.create_relationship_discover_job(
            generate_discover_session_id(session_id),
            project_id=project_id,
            source_session_id=session_id,
            rerun=rerun,
            idempotency_key=idempotency_key,
            idempotency_content=idempotency_content,
        )
        return RelationshipDiscoveryStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def prepare_validate(
        self, session_id: str, relationship_id: str
    ) -> RelationshipValidationPrepared:
        project_id = self._project_for_run(session_id)
        self._require_relationship_lane_free(project_id, session_id)
        candidate = self._require_candidate(project_id, session_id, relationship_id)
        label = candidate.pair.label()
        existing = self._validations(project_id, session_id)[0].get(label)
        if existing is not None and existing.verified:
            raise RelationshipNotValidatableError(relationship_id, "it is already fully validated")
        if candidate.confidence not in VALIDATABLE_CONFIDENCES:
            raise RelationshipNotValidatableError(
                relationship_id,
                f"its confidence is '{candidate.confidence}'; only high or medium "
                "candidates reach the DuckDB validation stage",
            )
        self._require_sources_loadable(session_id, candidate, relationship_id)
        fingerprint = candidate_fingerprint(candidate)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_RELATIONSHIP,
            session_id=session_id,
            project_id=project_id,
            action={
                "type": "relationship_validate",
                "source_session_id": session_id,
                "relationship_id": relationship_id,
                "candidate_fingerprint": fingerprint,
            },
            payload={
                "relationship_id": relationship_id,
                "project_id": project_id,
                "source_session_id": session_id,
                "pair_label": label,
                "candidate_fingerprint": fingerprint,
            },
        )
        return RelationshipValidationPrepared(
            session_id=session_id,
            relationship_id=relationship_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            label=label,
            left_dataset=candidate.pair.left_dataset_name,
            right_dataset=candidate.pair.right_dataset_name,
            confidence=candidate.confidence,
        )

    def validate(
        self,
        session_id: str,
        relationship_id: str,
        *,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None = None,
    ) -> RelationshipValidationStarted:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._validate_once(
                session_id,
                relationship_id,
                action_hash=action_hash,
                approval_token=approval_token,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _validate_once(
        self,
        session_id: str,
        relationship_id: str,
        *,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> RelationshipValidationStarted:
        project_id = self._project_for_run(session_id)
        # Idempotent replay must win before approval consumption, or a retried
        # validate would 409 on its own already-consumed hash (cleaning F1).
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                _payload, replay_payload_digest, _status = self._approvals.inspect_payload(
                    action_hash, session_id=session_id
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=APPROVAL_KIND_RELATIONSHIP,
                    content={
                        "source_session_id": session_id,
                        "relationship_id": relationship_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                    },
                )
                self._check_replay_matches(
                    existing,
                    session_id=session_id,
                    relationship_id=relationship_id,
                    action_hash=action_hash,
                )
                return self._replayed(session_id, relationship_id, existing)
        self._require_relationship_lane_free(project_id, session_id)

        def validate(payload: dict[str, Any]) -> tuple[Any, str]:
            if str(payload.get("relationship_id", "")) != relationship_id:
                raise RelationshipValidationRequestError(
                    "The approval was prepared for a different relationship than the request path."
                )
            candidate = self._require_candidate(project_id, session_id, relationship_id)
            fingerprint = candidate_fingerprint(candidate)
            if fingerprint != str(payload.get("candidate_fingerprint", "")):
                raise RelationshipSourceChangedError(relationship_id)
            return candidate, fingerprint

        payload, (candidate, fingerprint) = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_RELATIONSHIP,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            execution_session_id = generate_validate_session_id(session_id, relationship_id)
            job = self._jobs.create_relationship_validate_job(
                execution_session_id,
                project_id=project_id,
                source_session_id=session_id,
                relationship_id=relationship_id,
                pair_label=candidate.pair.label(),
                candidate_fingerprint=fingerprint,
                idempotency_key=idempotency_key,
                idempotency_content={
                    "source_session_id": session_id,
                    "relationship_id": relationship_id,
                    "action_hash": action_hash,
                    "approval_payload_digest": payload_digest(payload),
                },
            )
        return RelationshipValidationStarted(
            session_id=session_id,
            relationship_id=relationship_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def confirm(
        self, session_id: str, relationship_id: str, *, expected_version: int
    ) -> RelationshipEdge:
        """Confirm the join and sink the matching verified relation.

        Both writes are required:
        the seed is the class value_discovery and question_agent read as join
        context, so confirming without it silently narrows their input.
        """
        project_id = self._project_for_run(session_id)
        candidate = self._require_candidate(project_id, session_id, relationship_id)
        label = candidate.pair.label()
        left, right = _candidate_seed_pair(candidate)
        validation = self._validations(project_id, session_id)[0].get(label)
        self._semantic.confirm_join_and_sink_relation(
            session_id,
            label=label,
            left=left,
            right=right,
            cardinality=(validation.cardinality if validation is not None else None),
            source_session_id=session_id,
            expected_version=expected_version,
        )
        return self._edge_now(session_id, relationship_id)

    def revoke(
        self, session_id: str, relationship_id: str, *, expected_version: int
    ) -> RelationshipEdge:
        project_id = self._project_for_run(session_id)
        label = self._require_candidate(project_id, session_id, relationship_id).pair.label()
        self._semantic.revoke_whitelist_join(session_id, label, expected_version=expected_version)
        return self._edge_now(session_id, relationship_id)

    def _edge_now(self, session_id: str, relationship_id: str) -> RelationshipEdge:
        for edge in self.get_graph(session_id).edges:
            if edge.relationship_id == relationship_id:
                return edge
        raise RelationshipNotFoundError(relationship_id, session_id)

    def _latest_candidate_set(
        self, project_id: str, session_id: str
    ) -> RelationshipCandidateSet | None:
        return latest_candidate_set(self._store, project_id, session_id)

    def _candidate_artifact_id(self, project_id: str, session_id: str) -> str | None:
        artifacts = self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.RELATIONSHIP_CANDIDATE_SET,),
        )
        for artifact in reversed(artifacts):
            with suppress(ValueError):
                RelationshipCandidateSet.model_validate(artifact.payload)
                return artifact.id
        return None

    def _validations(
        self, project_id: str, session_id: str
    ) -> tuple[dict[str, RelationshipValidation], dict[str, str]]:
        """Merged validations plus the evidence artifact each winner came from,
        oldest artifact first so the newest one wins per pair."""
        merged: dict[str, RelationshipValidation] = {}
        artifact_ids: dict[str, str] = {}
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=(ArtifactType.RELATIONSHIP_VALIDATION_SET,),
        ):
            with suppress(ValueError):
                validation_set = RelationshipValidationSet.model_validate(artifact.payload)
                for validation in validation_set.validations:
                    merged[validation.pair.label()] = validation
                    artifact_ids[validation.pair.label()] = artifact.id
        return merged, artifact_ids

    def _whitelist_entries(self, project_id: str) -> dict[str, JoinWhitelistEntry]:
        project_dir = self._store.project_dir(project_id)
        try:
            entries = load_join_whitelist(project_dir).entries
        except ValueError:
            return {}
        return {entry.label(): entry for entry in entries}

    def _require_candidate(
        self, project_id: str, session_id: str, relationship_id: str
    ) -> RelationshipCandidate:
        candidate_set = self._latest_candidate_set(project_id, session_id)
        if candidate_set is None:
            raise RelationshipNotFoundError(relationship_id, session_id)
        for candidate in candidate_set.candidates:
            if relationship_id_for(candidate.pair.label()) == relationship_id:
                return candidate
        raise RelationshipNotFoundError(relationship_id, session_id)

    def _require_sources_loadable(
        self, session_id: str, candidate: RelationshipCandidate, relationship_id: str
    ) -> None:
        available = {
            handle.dataset_id
            for handle in self._datasets.list_datasets(session_id)
            if handle.ingest_status == "ready"
        }
        missing = sorted(
            {candidate.pair.left_dataset_id, candidate.pair.right_dataset_id} - available
        )
        if missing:
            raise RelationshipNotValidatableError(
                relationship_id,
                "both source datasets must still be readable; missing "
                f"dataset id(s): {', '.join(missing)}",
            )

    def _require_relationship_lane_free(self, project_id: str, session_id: str) -> None:
        """One discover or validate job per source run at a time.

        Both kinds read-modify-write the same run's relationship artifacts and
        the same project join whitelist from separate processes, so a second
        one would silently drop the first's result — and discovery can rewrite
        the very candidate set a validation was approved against. The source
        run itself must be idle too. Derived run ids end in a deterministic
        suffix over their originating request, which is what identifies a job
        as belonging to this source run before its manifest exists.
        """
        active = self._store.find_active_job_for_lane(session_id)
        if active is not None:
            raise RelationshipRunBusyError(session_id, str(active["job_id"]))
        suffixes = self._derived_run_suffixes(project_id, session_id)
        for job in self._store.list_active_jobs():
            if (
                str(job["kind"]) in RELATIONSHIP_JOB_KINDS
                and str(job["project_id"]) == project_id
                and str(job["session_id"]).rsplit("_", 1)[-1] in suffixes
            ):
                raise RelationshipRunBusyError(session_id, str(job["job_id"]))

    def _derived_run_suffixes(self, project_id: str, session_id: str) -> set[str]:
        """Every derived run id this source run could have minted, reduced to
        the deterministic suffix the ids end in."""
        suffixes = {generate_discover_session_id(session_id).rsplit("_", 1)[-1]}
        candidate_set = self._latest_candidate_set(project_id, session_id)
        if candidate_set is not None:
            suffixes |= {
                generate_validate_session_id(
                    session_id, relationship_id_for(candidate.pair.label())
                ).rsplit("_", 1)[-1]
                for candidate in candidate_set.candidates
            }
        return suffixes

    def _require_discoverable(self, session_id: str) -> None:
        """Discovery scans pairs of readable tables, so it needs two of them;
        mirrors the gate the Relationships page renders the button behind."""
        handles = self._datasets.list_datasets(session_id)
        readable = sum(1 for handle in handles if handle.ingest_status == "ready")
        if len(handles) < MIN_DISCOVERY_DATASETS:
            raise RelationshipNotDiscoverableError(
                session_id, "a run needs at least two datasets to relate"
            )
        if readable < MIN_DISCOVERY_DATASETS:
            raise RelationshipNotDiscoverableError(
                session_id,
                "at least two source tables must still be readable; "
                f"{readable} of {len(handles)} are",
            )
        if readable > MAX_DISCOVERY_DATASETS:
            raise RelationshipNotDiscoverableError(
                session_id,
                f"at most {MAX_DISCOVERY_DATASETS} readable datasets "
                f"({MAX_DISCOVERY_PAIRS} pairs) can be scanned at once; "
                f"this run has {readable}",
            )

    def _check_discover_replay_matches(self, job_row: dict, *, session_id: str) -> None:
        """A replay is only legitimate for a relationship_discover job in this
        run's own project, derived from this very run (mirrors validate)."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        expected_suffix = generate_discover_session_id(session_id).rsplit("_", 1)[-1]
        if (
            run_row is None
            or str(job_row["kind"]) != DISCOVER_JOB_KIND
            or str(job_row["project_id"]) != str(run_row["project_id"])
            or not str(job_row["session_id"]).endswith(f"_{expected_suffix}")
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a "
                "different project, kind or source run.",
            )

    def _discover_replayed(self, session_id: str, job_row: dict) -> RelationshipDiscoveryStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return RelationshipDiscoveryStarted(
            session_id=session_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _check_replay_matches(
        self, job_row: dict, *, session_id: str, relationship_id: str, action_hash: str
    ) -> None:
        """The idempotency fast path must not bypass the approval checks: a
        replay is only legitimate for a relationship_validate job in this run's
        own project whose action hash this run really consumed, derived from
        this very run (mirrors the questions/skills slices)."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != APPROVAL_KIND_RELATIONSHIP
            or str(job_row["project_id"]) != str(run_row["project_id"])
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different project or kind.",
            )
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None or str(pending["status"]) != "consumed":
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but the action "
                "hash was never consumed by this run.",
            )
        payload = json.loads(str(pending["payload_json"]))
        if (
            not isinstance(payload, dict)
            or str(payload.get("relationship_id", "")) != relationship_id
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} for a different relationship.",
            )
        expected_suffix = generate_validate_session_id(session_id, relationship_id).rsplit("_", 1)[
            -1
        ]
        if not str(job_row["session_id"]).endswith(f"_{expected_suffix}"):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} derived from a "
                "different source run.",
            )

    def _replayed(
        self, session_id: str, relationship_id: str, job_row: dict
    ) -> RelationshipValidationStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return RelationshipValidationStarted(
            session_id=session_id,
            relationship_id=relationship_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])


def _to_edge(
    candidate: RelationshipCandidate,
    validation: RelationshipValidation | None,
    entry: JoinWhitelistEntry | None,
    *,
    candidate_artifact_id: str | None,
    validation_artifact_id: str | None,
    dataset_ids_by_name: dict[str, str],
    loadable: set[str],
) -> RelationshipEdge:
    pair = candidate.pair
    signals = candidate.signals
    verified = validation is not None and validation.verified
    confirmed = entry is not None and entry.status in CONFIRMED_JOIN_STATUSES
    freshness = entry.validation_freshness(dataset_ids_by_name) if entry is not None else None
    sources_loadable = {pair.left_dataset_id, pair.right_dataset_id} <= loadable
    return RelationshipEdge(
        relationship_id=relationship_id_for(pair.label()),
        label=pair.label(),
        state="confirmed" if confirmed else ("validated" if verified else "candidate"),
        left_dataset_id=pair.left_dataset_id,
        left_dataset=pair.left_dataset_name,
        left_columns=list(pair.left_columns),
        right_dataset_id=pair.right_dataset_id,
        right_dataset=pair.right_dataset_name,
        right_columns=list(pair.right_columns),
        confidence=candidate.confidence,
        ensemble_score=candidate.ensemble_score,
        name_similarity=signals.name_similarity,
        overlap_left_in_right=signals.overlap_left_in_right,
        overlap_right_in_left=signals.overlap_right_in_left,
        right_unique_rate=signals.right_unique_rate,
        signals_sampled=signals.sampled,
        verified=verified,
        cardinality=validation.cardinality if validation is not None else None,
        join_row_multiplier=validation.join_row_multiplier if validation is not None else None,
        orphan_rate_left=validation.orphan_rate_left if validation is not None else None,
        orphan_rate_right=validation.orphan_rate_right if validation is not None else None,
        verification_sql=validation.verification_sql if validation is not None else "",
        validation_sampled=validation.sampled if validation is not None else False,
        warnings=list(validation.warnings) if validation is not None else [],
        join_status=entry.status if entry is not None else None,
        freshness=freshness,
        can_validate=(
            not verified
            and not confirmed
            and sources_loadable
            and candidate.confidence in VALIDATABLE_CONFIDENCES
        ),
        # Mirrors SemanticService.confirm_whitelist_join's own gate, so the
        # button is only offered when the POST would actually succeed.
        can_confirm=(
            entry is not None
            and entry.status == "proposed"
            and entry.validation_verified
            and freshness == "fresh"
            and entry.cardinality != "many_to_many"
        ),
        can_revoke=entry is not None and entry.status == "auto_confirmed",
        candidate_artifact_id=candidate_artifact_id,
        validation_artifact_id=validation_artifact_id if validation is not None else None,
    )


def generate_validate_session_id(source_session_id: str, relationship_id: str) -> str:
    """Derived run id for one validation; the suffix is deterministic so an
    idempotent retry can prove a job really came from this request."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash(
        {"source_session_id": source_session_id, "relationship_id": relationship_id}, length=6
    )
    return f"{VALIDATE_SESSION_PREFIX}{stamp}_{suffix}"


def generate_discover_session_id(source_session_id: str) -> str:
    """Derived run id for one discovery; same deterministic-suffix trick as
    ``generate_validate_session_id``, keyed on the source run alone."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash({"discover_source_session_id": source_session_id}, length=6)
    return f"{DISCOVER_SESSION_PREFIX}{stamp}_{suffix}"


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )

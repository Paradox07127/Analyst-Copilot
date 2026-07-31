"""Skills use cases (§10.3 P2).

List serves the project's saved skill library plus the builtin seed templates,
each with the parameter signature a replay must bind. Prepare validates the
bindings against the target run's datasets, freezes the resulting concrete
skill into a server-side pending approval, and returns the SQL that will run.
Execute consumes the approval once and queues a `skill_replay` job on a fresh
derived run, where the worker calls the existing `replay_skill` driver.

The approval payload carries the whole instantiated skill, not a reference to
one: nothing the library does afterwards can change what a consumed approval
executes, so no re-fingerprinting is needed at pickup time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from eda_platform.application.dto import (
    DatasetHandle,
    JobCreated,
    JobStatus,
    SkillParamSpec,
    SkillPlanCandidate,
    SkillReplayPrepared,
    SkillReplayStarted,
    SkillSummary,
    SkillsView,
    SkillTargetColumn,
    SkillTargetDataset,
)
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.dataset_service import DatasetService
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    ProjectNotFoundError,
    SessionNotFoundError,
)
from eda_platform.core.bounded_pagination import (
    ResourcePageIndex,
    decode_bound_cursor,
    encode_bound_cursor,
    source_token,
)
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, stable_hash
from eda_platform.core.query import UnsafeQueryError, validate_select_statement
from eda_platform.core.skills_store import (
    add_skill,
    import_seed,
    instantiate_seed,
    is_bindable_identifier,
    load_builtin_seeds,
    load_skills,
    save_skills,
    shown_identifier,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import AnalysisSkill, SeedSkillTemplate
from eda_platform.tools.sql_runner import relation_names_for, rewrite_relation_names

APPROVAL_KIND_SKILL = "skill_replay"
REPLAY_SESSION_PREFIX = "ssess_"
SOURCE_LIBRARY = "library"
SOURCE_SEED = "seed"
MAX_SKILL_NAME_CHARS = 120
MAX_SKILL_DESCRIPTION_CHARS = 1000
# Artifact types whose payload is a frozen, already-validated AnalysisPlan.
SAVABLE_PLAN_TYPES = (ArtifactType.CHAT_TURN_PLAN,)


class SkillServiceError(Exception):
    pass


class SkillNotFoundError(SkillServiceError):
    def __init__(self, skill_id: str, session_id: str) -> None:
        super().__init__(f"Skill not found for run {session_id}: {skill_id}")
        self.skill_id = skill_id
        self.session_id = session_id


class SkillBindingInvalidError(SkillServiceError):
    """The chosen datasets/columns cannot satisfy the skill's signature."""


class SkillValidationError(SkillServiceError):
    pass


class SkillSqlRejectedError(SkillServiceError):
    """The concrete SQL would not survive the read-only gate at execution."""


class SkillNotInLibraryError(SkillServiceError):
    def __init__(self, skill_id: str, project_id: str) -> None:
        super().__init__(f"Skill not in the library of project {project_id}: {skill_id}")
        self.skill_id = skill_id
        self.project_id = project_id


class SkillNotDeletableError(SkillServiceError):
    """Builtin seed templates ship with the package; only saved skills go away."""

    def __init__(self, skill_id: str) -> None:
        super().__init__(
            f"{skill_id} is a builtin seed template and cannot be deleted; "
            "delete the skill you imported from it instead."
        )
        self.skill_id = skill_id


class SkillPlanNotFoundError(SkillServiceError):
    def __init__(self, artifact_id: str, session_id: str) -> None:
        super().__init__(f"No savable analysis plan in run {session_id}: {artifact_id}")
        self.artifact_id = artifact_id
        self.session_id = session_id


class SkillService:
    def __init__(
        self,
        store: ArtifactStore,
        datasets: DatasetService,
        approvals: ApprovalService,
        jobs: JobService,
    ) -> None:
        self._store = store
        self._datasets = datasets
        self._approvals = approvals
        self._jobs = jobs

    def list_skills(
        self, session_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> SkillsView:
        limit = max(1, min(limit, 200))
        project_id = self._project_for_run(session_id)
        scope = f"skills:{project_id}:{session_id}"
        version = self._skills_source_version(project_id, session_id)
        offset = (
            decode_bound_cursor(cursor, scope=scope, source_version=version)
            if cursor
            else 0
        )
        index = ResourcePageIndex(self._store.db_path)
        if not index.is_current(scope, version):
            handles = self._datasets.list_datasets(session_id)
            summaries = [
                *(_library_summary(skill) for skill in self._library(project_id)),
                *(_seed_summary(seed) for seed in load_builtin_seeds()),
            ]
            plans = [
                _plan_candidate(artifact, plan)
                for artifact, plan in self._savable_plans(project_id, session_id)
            ]
            index.replace(
                scope,
                version,
                {
                    "skills": [item.model_dump_json() for item in summaries],
                    "datasets": [
                        _target_dataset(item).model_dump_json() for item in handles
                    ],
                    "plans": [item.model_dump_json() for item in plans],
                },
            )
            if self._skills_source_version(project_id, session_id) != version:
                raise InvalidCursorError
        skill_rows = index.page(
            scope, version, "skills", offset=offset, limit=limit + 1
        )
        dataset_rows = index.page(
            scope, version, "datasets", offset=offset, limit=limit + 1
        )
        plan_rows = index.page(
            scope, version, "plans", offset=offset, limit=limit + 1
        )
        if self._skills_source_version(project_id, session_id) != version:
            raise InvalidCursorError
        has_more = any(
            len(rows) > limit for rows in (skill_rows, dataset_rows, plan_rows)
        )
        consumed = offset + limit
        return SkillsView(
            session_id=session_id,
            project_id=project_id,
            skills=[
                SkillSummary.model_validate_json(item) for item in skill_rows[:limit]
            ],
            datasets=[
                SkillTargetDataset.model_validate_json(item)
                for item in dataset_rows[:limit]
            ],
            savable_plans=[
                SkillPlanCandidate.model_validate_json(item)
                for item in plan_rows[:limit]
            ],
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

    def _skills_source_version(self, project_id: str, session_id: str) -> str:
        project_dir = self._store.project_dir(project_id)
        paths = (
            project_dir / "skills" / "skills.json",
            self._store.session_dir(project_id, session_id) / "manifest.json",
        )
        stats: list[tuple[str, int, int] | tuple[str]] = []
        for path in paths:
            try:
                stat = path.stat()
                stats.append((str(path.name), stat.st_size, stat.st_mtime_ns))
            except OSError:
                stats.append((f"{path.name}:missing",))
        # Same reason as SemanticService._semantic_source_version: an
        # artifact-index token churns on every write during a run, so a second
        # page is never reachable. Run status settles once instead.
        return source_token(
            "skills-v1",
            project_id,
            session_id,
            stats,
            self._store.get_session_status(session_id),
        )

    def save_skill(
        self,
        session_id: str,
        *,
        source_artifact_id: str,
        name: str,
        description: str = "",
    ) -> SkillSummary:
        """Freeze one of this run's validated plan artifacts into a named skill."""
        # Local import: `drivers.analysis_skill` pulls the whole chat driver, and
        # the API process must not pay for it just to start.
        from eda_platform.drivers.analysis_skill import skill_from_plan

        project_id = self._project_for_run(session_id)
        name = name.strip()
        description = description.strip()
        if not name:
            raise SkillValidationError("A skill needs a name.")
        if len(name) > MAX_SKILL_NAME_CHARS:
            raise SkillValidationError(
                f"A skill name is at most {MAX_SKILL_NAME_CHARS} characters."
            )
        if len(description) > MAX_SKILL_DESCRIPTION_CHARS:
            raise SkillValidationError(
                f"A skill description is at most {MAX_SKILL_DESCRIPTION_CHARS} characters."
            )
        plan = next(
            (
                candidate
                for artifact, candidate in self._savable_plans(project_id, session_id)
                if artifact.id == source_artifact_id
            ),
            None,
        )
        if plan is None:
            raise SkillPlanNotFoundError(source_artifact_id, session_id)
        skill = skill_from_plan(plan, name, description, source_session_id=session_id)
        add_skill(self._store.project_dir(project_id), skill)
        return _library_summary(skill)

    def import_seed_skill(
        self,
        session_id: str,
        seed_id: str,
        *,
        dataset_ids: list[str],
        bindings: dict[str, str],
        name: str = "",
    ) -> SkillSummary:
        """Bind a builtin seed template to this run's data and save it as a
        library skill, so it survives the run it was bound in.

        Idempotent on the (seed, relation, bindings) triple, like ``import_seed``.
        """
        project_id = self._project_for_run(session_id)
        name = name.strip()
        if len(name) > MAX_SKILL_NAME_CHARS:
            raise SkillValidationError(
                f"A skill name is at most {MAX_SKILL_NAME_CHARS} characters."
            )
        seed = next((item for item in load_builtin_seeds() if item.seed_id == seed_id), None)
        if seed is None:
            raise SkillNotFoundError(seed_id, session_id)
        targets = self._resolve_targets(session_id, dataset_ids)
        relations = relation_names_for([handle.display_name for handle in targets])
        # Same arity/column/guard checks a replay of this seed would face, so an
        # unreplayable skill never reaches the library.
        instantiated = _instantiate_seed_for(seed, targets, relations, bindings)
        _validated_sql(instantiated)
        project_dir = self._store.project_dir(project_id)
        try:
            skill = import_seed(
                project_dir, seed, relation_name=relations[0], bindings=bindings
            )
        except (ToolGuardError, ValueError) as exc:
            raise SkillBindingInvalidError(str(exc)) from exc
        if name and name != skill.name:
            # add_skill replaces in place on skill_id, so the rename lands on the
            # row import_seed just wrote rather than duplicating it.
            skill = skill.model_copy(update={"name": name})
            add_skill(project_dir, skill)
        return _library_summary(skill)

    def delete_skill(self, project_id: str, skill_id: str) -> None:
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        if any(seed.seed_id == skill_id for seed in load_builtin_seeds()):
            raise SkillNotDeletableError(skill_id)
        project_dir = self._store.project_dir(project_id)
        skills = load_skills(project_dir)
        remaining = [skill for skill in skills if skill.skill_id != skill_id]
        if len(remaining) == len(skills):
            raise SkillNotInLibraryError(skill_id, project_id)
        save_skills(project_dir, remaining)

    def _savable_plans(
        self, project_id: str, session_id: str
    ) -> list[tuple[Artifact, AnalysisPlan]]:
        """This run's frozen plan artifacts, newest first. A plan artifact only
        exists once the plan passed the approval gate, so no extra check here."""
        pairs: list[tuple[Artifact, AnalysisPlan]] = []
        for artifact in self._store.list_indexed_artifacts(
            project_id=project_id,
            session_id=session_id,
            artifact_types=SAVABLE_PLAN_TYPES,
        ):
            plan = _plan_from_artifact(artifact)
            if plan is not None:
                pairs.append((artifact, plan))
        pairs.reverse()
        return pairs

    def prepare_replay(
        self,
        session_id: str,
        skill_id: str,
        *,
        dataset_ids: list[str],
        bindings: dict[str, str],
    ) -> SkillReplayPrepared:
        project_id = self._project_for_run(session_id)
        targets = self._resolve_targets(session_id, dataset_ids)
        instantiated = self._instantiate(project_id, session_id, skill_id, targets, bindings)
        concrete = _validated_sql(instantiated)
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_SKILL,
            session_id=session_id,
            project_id=project_id,
            # The fingerprint is what makes two bindings of one skill distinct
            # action hashes; pending rows are keyed by (action_hash, session_id).
            action={
                "type": "skill_replay",
                "source_session_id": session_id,
                "skill_id": skill_id,
                "skill_fingerprint": _skill_fingerprint(
                    concrete, [handle.dataset_id for handle in targets]
                ),
            },
            payload={
                "skill_id": skill_id,
                "source_session_id": session_id,
                "project_id": project_id,
                "dataset_ids": [handle.dataset_id for handle in targets],
                "bindings": dict(bindings),
                # The approval owns the content, not a library pointer: what was
                # reviewed is exactly what the worker replays.
                "skill": concrete.model_dump(mode="json"),
            },
        )
        return SkillReplayPrepared(
            session_id=session_id,
            skill_id=skill_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            name=concrete.name,
            question=concrete.plan.question,
            sql_preview=concrete.plan.sql,
            dataset_ids=[handle.dataset_id for handle in targets],
            dataset_names=[handle.display_name for handle in targets],
            bindings=dict(bindings),
        )

    def execute_replay(
        self,
        session_id: str,
        skill_id: str,
        *,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None = None,
    ) -> SkillReplayStarted:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._execute_replay_once(
                session_id,
                skill_id,
                action_hash=action_hash,
                approval_token=approval_token,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
            ),
        )

    def _execute_replay_once(
        self,
        session_id: str,
        skill_id: str,
        *,
        action_hash: str,
        approval_token: str,
        idempotency_key: str | None,
        contention_deadline: float,
    ) -> SkillReplayStarted:
        project_id = self._project_for_run(session_id)
        # Idempotent replay wins before approval consumption: a retried execute
        # must not 409 on its own already-consumed hash (questions/cleaning F1).
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                _payload, replay_payload_digest, _status = (
                    self._approvals.inspect_payload(action_hash, session_id=session_id)
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind=APPROVAL_KIND_SKILL,
                    content={
                        "source_session_id": session_id,
                        "skill_id": skill_id,
                        "action_hash": action_hash,
                        "approval_payload_digest": replay_payload_digest,
                    },
                )
                self._check_replay_matches(
                    existing, session_id=session_id, skill_id=skill_id, action_hash=action_hash
                )
                return self._replayed(session_id, skill_id, existing)
        def validate(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            if str(payload.get("skill_id", "")) != skill_id:
                raise SkillValidationError(
                    "The approval was prepared for a different skill than the request path."
                )
            skill = payload.get("skill")
            dataset_ids = payload.get("dataset_ids")
            if not isinstance(skill, dict) or not isinstance(dataset_ids, list):
                raise SkillValidationError("The approval payload is not a replayable skill.")
            return skill, [str(item) for item in dataset_ids]

        payload, validated = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_SKILL,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            skill, dataset_ids = validated
            execution_session_id = generate_replay_session_id(
                session_id, skill_id, dataset_ids
            )
            job = self._jobs.create_skill_replay_job(
                execution_session_id,
                project_id=project_id,
                source_session_id=session_id,
                skill_id=skill_id,
                skill=skill,
                dataset_ids=dataset_ids,
                idempotency_key=idempotency_key,
                idempotency_content={
                    "source_session_id": session_id,
                    "skill_id": skill_id,
                    "action_hash": action_hash,
                    "approval_payload_digest": payload_digest(payload),
                },
            )
        return SkillReplayStarted(
            session_id=session_id,
            skill_id=skill_id,
            execution_session_id=job.session_id,
            job=_to_created(job),
        )

    def _instantiate(
        self,
        project_id: str,
        session_id: str,
        skill_id: str,
        targets: list[DatasetHandle],
        bindings: dict[str, str],
    ) -> AnalysisSkill:
        """The concrete, replayable skill for these targets — library skills
        rebound onto them, seeds bound to real columns. Library ids win on
        collision."""
        relations = relation_names_for([handle.display_name for handle in targets])
        for skill in self._library(project_id):
            if skill.skill_id == skill_id:
                _require_no_bindings(bindings)
                _require_target_arity(skill, targets)
                _require_columns_present(skill.param_columns, targets)
                return _rebound_for_targets(skill, relations)
        for seed in load_builtin_seeds():
            if seed.seed_id == skill_id:
                return _instantiate_seed_for(seed, targets, relations, bindings)
        raise SkillNotFoundError(skill_id, session_id)

    def _library(self, project_id: str) -> list[AnalysisSkill]:
        return load_skills(self._store.project_dir(project_id))

    def _resolve_targets(self, session_id: str, dataset_ids: list[str]) -> list[DatasetHandle]:
        if not dataset_ids:
            raise SkillBindingInvalidError(
                "Select at least one dataset of this run to replay the skill on."
            )
        by_id = {handle.dataset_id: handle for handle in self._datasets.list_datasets(session_id)}
        resolved: list[DatasetHandle] = []
        seen: set[str] = set()
        for dataset_id in dataset_ids:
            handle = by_id.get(dataset_id)
            if handle is None:
                raise SkillBindingInvalidError(f"Dataset {dataset_id} is not part of run {session_id}.")
            if dataset_id in seen:
                raise SkillBindingInvalidError(f"Dataset {dataset_id} was selected twice.")
            seen.add(dataset_id)
            resolved.append(handle)
        return resolved

    def _check_replay_matches(
        self, job_row: dict, *, session_id: str, skill_id: str, action_hash: str
    ) -> None:
        """The idempotency fast path must not bypass the approval checks: a
        replay is only legitimate for a skill_replay job in this run's own
        project whose action hash this run really consumed, derived from this
        very run (mirrors the questions slice)."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != APPROVAL_KIND_SKILL
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
        if not isinstance(payload, dict) or str(payload.get("skill_id", "")) != skill_id:
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} for a different skill.",
            )
        dataset_ids = [str(item) for item in payload.get("dataset_ids", [])]
        expected_suffix = generate_replay_session_id(session_id, skill_id, dataset_ids).rsplit("_", 1)[-1]
        if not str(job_row["session_id"]).endswith(f"_{expected_suffix}"):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} derived from a "
                "different source run.",
            )

    def _replayed(self, session_id: str, skill_id: str, job_row: dict) -> SkillReplayStarted:
        job = self._jobs.get_job(str(job_row["job_id"]))
        return SkillReplayStarted(
            session_id=session_id,
            skill_id=skill_id,
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


def generate_replay_session_id(source_session_id: str, skill_id: str, dataset_ids: list[str]) -> str:
    """Derived run id for one replay; the suffix is deterministic so an
    idempotent retry can prove a job really came from this request."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash(
        {"source_session_id": source_session_id, "skill_id": skill_id, "dataset_ids": list(dataset_ids)},
        length=6,
    )
    return f"{REPLAY_SESSION_PREFIX}{stamp}_{suffix}"


def _rebound_for_targets(skill: AnalysisSkill, target_relations: list[str]) -> AnalysisSkill:
    """Bind a saved skill's relation names onto the chosen targets here, so the
    preview, the approved payload and the SQL the worker runs are one string.

    ``replay_skill`` rebinds too, but on an already-rebound skill its mapping is
    the identity and it changes nothing — that only holds while
    ``expected_datasets`` is kept in step with the rewritten SQL below.
    """
    mapping = _relation_mapping(skill.expected_datasets, target_relations)
    plan = skill.plan.model_copy(
        update={
            "sql": rewrite_relation_names(skill.plan.sql, mapping),
            "dataset_names": _remap_dataset_names(skill.plan.dataset_names, mapping),
        }
    )
    return skill.model_copy(update={"plan": plan, "expected_datasets": list(target_relations)})


def _relation_mapping(expected: list[str], target_relations: list[str]) -> dict[str, str]:
    """Same rule as the replay driver, already enforced by ``_require_target_arity``:
    one target absorbs the whole analysis, otherwise map one-to-one in order."""
    if len(target_relations) == 1:
        return {name: target_relations[0] for name in expected}
    return dict(zip(expected, target_relations, strict=True))


def _remap_dataset_names(dataset_names: list[str], mapping: dict[str, str]) -> list[str]:
    remapped: list[str] = []
    for name in dataset_names:
        target = mapping.get(name, name)
        if target not in remapped:
            remapped.append(target)
    return remapped


def _validated_sql(skill: AnalysisSkill) -> AnalysisSkill:
    """Refuse now what the read-only gate would refuse at execution, and freeze
    the gate's own normalisation so preview == approval == executed SQL."""
    try:
        statement = validate_select_statement(skill.plan.sql)
    except UnsafeQueryError as exc:
        raise SkillSqlRejectedError(
            f"This skill's SQL will not pass the read-only gate: {exc}"
        ) from exc
    return skill.model_copy(update={"plan": skill.plan.model_copy(update={"sql": statement})})


def _instantiate_seed_for(
    seed: SeedSkillTemplate,
    targets: list[DatasetHandle],
    target_relations: list[str],
    bindings: dict[str, str],
) -> AnalysisSkill:
    if len(targets) != 1:
        raise SkillBindingInvalidError(
            f"A seed template is instantiated on exactly one dataset; {len(targets)} were selected."
        )
    _require_columns_present(list(bindings.values()), targets)
    try:
        return instantiate_seed(seed, relation_name=target_relations[0], bindings=bindings)
    except ToolGuardError as exc:
        raise SkillBindingInvalidError(str(exc)) from exc


def _require_no_bindings(bindings: dict[str, str]) -> None:
    if bindings:
        raise SkillBindingInvalidError(
            "A saved skill has no placeholders; it only takes target datasets."
        )


def _require_target_arity(skill: AnalysisSkill, targets: list[DatasetHandle]) -> None:
    """Same rule the replay driver enforces: one target for the whole analysis,
    or exactly one per dataset the skill referenced."""
    expected = len(skill.expected_datasets)
    if len(targets) == 1 or len(targets) == expected:
        return
    raise SkillBindingInvalidError(
        f"Select 1 dataset to run the whole analysis on, or {expected} to map "
        f"onto the datasets this skill referenced; {len(targets)} were selected."
    )


def _require_columns_present(columns: list[str], targets: list[DatasetHandle]) -> None:
    """Reject bindings the replay gate would refuse anyway. A profile without
    recorded columns cannot answer, so it defers to the driver's own guard."""
    if any(not handle.schema_ for handle in targets):
        return
    available = {column.name for handle in targets for column in handle.schema_}
    missing = [column for column in columns if column not in available]
    if missing:
        # The binding value rides this message into the UI verbatim, so clip it.
        shown = ", ".join(shown_identifier(name) for name in sorted(set(missing)))
        raise SkillBindingInvalidError(
            f"Column(s) {shown} do not exist in the selected dataset(s) of this run."
        )


def _skill_fingerprint(skill: AnalysisSkill, dataset_ids: list[str]) -> str:
    return stable_hash(
        {
            "sql": skill.plan.sql,
            "question": skill.plan.question,
            "columns": list(skill.param_columns),
            "datasets": list(skill.expected_datasets),
            "targets": list(dataset_ids),
        },
        length=64,
    )


def _plan_from_artifact(artifact: Artifact) -> AnalysisPlan | None:
    """Rebuild the frozen plan from a ChatTurnPlan payload (drop raw_message)."""
    try:
        fields = {key: value for key, value in artifact.payload.items() if key != "raw_message"}
        return AnalysisPlan.model_validate(fields)
    except (ValueError, TypeError):
        return None


def _plan_candidate(artifact: Artifact, plan: AnalysisPlan) -> SkillPlanCandidate:
    return SkillPlanCandidate(
        artifact_id=artifact.id,
        question=plan.question,
        sql=plan.sql,
        method=plan.method,
        dataset_names=list(plan.dataset_names),
        columns=list(plan.columns),
        created_at=artifact.created_at,
    )


def _library_summary(skill: AnalysisSkill) -> SkillSummary:
    return SkillSummary(
        skill_id=skill.skill_id,
        source=SOURCE_LIBRARY,
        name=skill.name,
        description=skill.description,
        question=skill.plan.question,
        sql=skill.plan.sql,
        method=skill.plan.method,
        param_columns=list(skill.param_columns),
        expected_datasets=list(skill.expected_datasets),
        source_session_id=skill.source_session_id,
        created_at=skill.created_at,
    )


def _seed_summary(seed: SeedSkillTemplate) -> SkillSummary:
    return SkillSummary(
        skill_id=seed.seed_id,
        source=SOURCE_SEED,
        name=seed.name,
        description=seed.rationale,
        question=seed.question,
        sql=seed.sql,
        method=seed.method,
        params=[
            SkillParamSpec(name=param.name, role=param.role, description=param.description)
            for param in seed.params
        ],
    )


def _target_dataset(handle: DatasetHandle) -> SkillTargetDataset:
    return SkillTargetDataset(
        dataset_id=handle.dataset_id,
        name=handle.display_name,
        relation=relation_names_for([handle.display_name])[0],
        columns=[
            SkillTargetColumn(name=column.name, bindable=is_bindable_identifier(column.name))
            for column in handle.schema_
        ],
    )


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )

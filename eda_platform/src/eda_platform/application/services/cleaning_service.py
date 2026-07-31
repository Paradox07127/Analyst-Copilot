"""Cleaning use cases (§7.5 / 阶段4 slice E).

Preview computes the recipe diff with the existing driver and registers a
server-side pending approval; apply consumes the approval hash, writes the
cleaned version via the same permission-gated driver (hash math unchanged),
then forks a fresh auto_eda job onto the cleaned file through JobService —
the API path that forks analysis onto a cleaned dataset version.

The two read use cases (cleaning log, raw before-cleaning view) shape the same
rows the Cleaning page renders, via application.cleaning_view.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pandas.api.types import is_object_dtype, is_string_dtype

from eda_platform.application.cleaning_view import (
    cleaning_guardrail_rows,
    cleaning_operation_rows,
    cleaning_suggestion_rows,
    cleaning_summary_rows,
    dataset_names_from_profile_artifacts,
)
from eda_platform.application.dto import (
    CleaningApplied,
    CleaningColumnChange,
    CleaningLogGuardrailRow,
    CleaningLogOperationRow,
    CleaningLogSuggestionRow,
    CleaningLogSummaryRow,
    CleaningLogView,
    CleaningOperation,
    CleaningPreviewResult,
    CleaningPreviewView,
    CleaningRawView,
    DatasetProfileSummary,
    FieldProfileRow,
    JobCreated,
    JobStatus,
    RawChartView,
    RawDataPreviewView,
)
from eda_platform.application.services.approval_service import (
    ApprovalService,
    payload_digest,
)
from eda_platform.application.services.dataset_service import (
    DatasetNotFoundError,
    DatasetService,
    DatasetSourceMissingError,
)
from eda_platform.application.services.insight_service import _contains_vega_expression
from eda_platform.application.services.job_service import JobConflictError, JobService
from eda_platform.application.services.session_service import SessionNotFoundError
from eda_platform.application.workbench import dataset_display_rows, semantic_type_counts
from eda_platform.application.workspace_paths import relativize_workspace_paths
from eda_platform.core.fs import remove_tree
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, hash_file
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.cleaning_apply import (
    apply_cleaning,
    cleaning_action,
    preview_cleaning,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.cleaning import CleaningRecipe, CleaningTransform, transform_is_lossy
from eda_platform.tools.cleaning import AppliedCleaning, _next_free_version
from eda_platform.tools.loader import LoadedDataset, load_csv

APPROVAL_KIND_CLEANING = "cleaning_apply"
FORK_BUSINESS_CONTEXT = "Cleaned dataset variant."


class CleaningServiceError(Exception):
    pass


class CleaningValidationError(CleaningServiceError):
    error_code = "cleaning_invalid"


class CleaningApplyRefusedError(CleaningServiceError):
    error_code = "cleaning_refused"


class CleaningSourceChangedError(CleaningServiceError):
    """C4: the source CSV on disk no longer matches the previewed content."""

    error_code = "cleaning_source_changed"

    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Source data for {dataset_id} changed since the preview; "
            "run preview again and approve the fresh result."
        )
        self.dataset_id = dataset_id


class CleaningService:
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

    def get_log(self, session_id: str) -> CleaningLogView:
        """The four cleaning-transparency tables built from the run's recipes."""
        project_id = self._project_for_run(session_id)
        recipes = self._artifacts(project_id, session_id, ArtifactType.CLEANING_RECIPE)
        return CleaningLogView(
            session_id=session_id,
            recipe_count=len(recipes),
            summary=[
                CleaningLogSummaryRow.model_validate(row) for row in cleaning_summary_rows(recipes)
            ],
            deleted_data=[
                CleaningLogOperationRow.model_validate(row)
                for row in cleaning_operation_rows(recipes)
            ],
            protection_triggers=[
                CleaningLogGuardrailRow.model_validate(row)
                for row in cleaning_guardrail_rows(recipes)
            ],
            suggestions=[
                CleaningLogSuggestionRow.model_validate(row)
                for row in cleaning_suggestion_rows(recipes)
            ],
        )

    def get_raw(self, session_id: str) -> CleaningRawView:
        """Raw before-cleaning profiles, charts, and data previews for a run."""
        project_id = self._project_for_run(session_id)
        profiles = self._artifacts(project_id, session_id, ArtifactType.RAW_DATASET_PROFILE)
        charts = self._artifacts(project_id, session_id, ArtifactType.RAW_CHART_SPEC)
        previews = self._artifacts(project_id, session_id, ArtifactType.RAW_DATA_PREVIEW)
        recipes = self._artifacts(project_id, session_id, ArtifactType.CLEANING_RECIPE)
        dataset_names = dataset_names_from_profile_artifacts(profiles)
        return CleaningRawView(
            session_id=session_id,
            precleaning_recorded=bool(profiles or charts or previews or recipes),
            profiles=[_profile_summary(artifact) for artifact in profiles],
            charts=[
                view
                for view in (self._raw_chart(artifact, dataset_names) for artifact in charts)
                if view is not None
            ],
            previews=[self._raw_preview(artifact) for artifact in previews],
        )

    def _raw_chart(
        self, artifact: Artifact, dataset_names: dict[str, str]
    ) -> RawChartView | None:
        """None when the spec is unservable — same client-renderer boundary as
        InsightService.get_chart (inline data only, no vega expressions)."""
        try:
            spec = ChartSpec.model_validate(artifact.payload)
        except ValueError:
            return None
        if spec.data and set(spec.data) - {"values"}:
            return None
        vegalite = relativize_workspace_paths(spec.to_vegalite(), self._store.root)
        if _contains_vega_expression(vegalite):
            return None
        return RawChartView(
            artifact_id=artifact.id,
            title=spec.title,
            dataset_id=spec.dataset_id,
            dataset_name=dataset_names.get(spec.dataset_id, spec.dataset_id),
            description=str(vegalite.get("description") or spec.description),
            plain_language=artifact.plain_language,
            spec=vegalite,
        )

    def _raw_preview(self, artifact: Artifact) -> RawDataPreviewView:
        payload = relativize_workspace_paths(artifact.payload, self._store.root)
        dataset_id = str(payload.get("dataset_id") or "")
        rows_preview = payload.get("rows_preview")
        column_names = payload.get("column_names")
        return RawDataPreviewView(
            artifact_id=artifact.id,
            dataset_id=dataset_id,
            name=str(payload.get("name") or dataset_id or "Dataset"),
            rows=_int_or_zero(payload.get("rows")),
            columns=_int_or_zero(payload.get("columns")),
            column_names=[str(name) for name in column_names]
            if isinstance(column_names, list)
            else [],
            rows_preview=[row for row in rows_preview if isinstance(row, dict)]
            if isinstance(rows_preview, list)
            else [],
        )

    def _artifacts(
        self, project_id: str, session_id: str, artifact_type: ArtifactType
    ) -> list[Artifact]:
        return self._store.list_indexed_artifacts(
            project_id=project_id, session_id=session_id, artifact_types=(artifact_type,)
        )

    def _project_for_run(self, session_id: str) -> str:
        if INTERNAL_SESSION_MARKER in session_id:
            raise SessionNotFoundError(session_id)
        row = self._store.get_session_index_row(session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return str(row["project_id"])

    def preview(
        self,
        session_id: str,
        *,
        dataset_id: str,
        trim_whitespace: bool = True,
        drop_duplicate_rows: bool = True,
        drop_missing_rows: bool = False,
        drop_outlier_rows: bool = False,
        cancel_check: Callable[[], object] | None = None,
    ) -> CleaningPreviewResult:
        if cancel_check is not None:
            cancel_check()
        project_id, source_uri, content_hash, loaded = self._load(
            session_id,
            dataset_id,
            cancel_check=cancel_check,
        )
        if cancel_check is not None:
            cancel_check()
        recipe = _build_recipe(
            loaded,
            trim_whitespace=trim_whitespace,
            drop_duplicate_rows=drop_duplicate_rows,
            drop_missing_rows=drop_missing_rows,
            drop_outlier_rows=drop_outlier_rows,
        )
        if not recipe.transforms:
            raise CleaningValidationError(
                "No applicable cleaning operations: enable at least one option "
                "that matches the dataset (e.g. trim needs text columns)."
            )
        try:
            dispatch = preview_cleaning(loaded, recipe)
        except (ToolGuardError, ValueError) as exc:
            raise CleaningValidationError(str(exc)) from exc
        preview = dispatch.preview
        if preview is None:  # pragma: no cover - driver always returns a preview
            raise CleaningValidationError("Cleaning preview produced no diff.")
        # Slice-E F3: report the version apply will actually allocate (CL-2
        # skips occupied slots), not a blind source_version+1 — with v2 already
        # on disk the next apply writes v3 and the preview must say so.
        target_version, _, _ = _next_free_version(
            self._store.project_dir(project_id) / "cleaned",
            recipe.dataset_id,
            preview.target_version,
            loaded.record.name,
        )
        digest, approval_token, expires_at = self._approvals.register(
            kind=APPROVAL_KIND_CLEANING,
            session_id=session_id,
            project_id=project_id,
            action=cleaning_action(recipe),
            payload={
                "recipe": recipe.model_dump(mode="json"),
                "dataset_id": dataset_id,
                "project_id": project_id,
                "source_uri": source_uri,
                "content_hash": content_hash,
            },
        )
        return CleaningPreviewResult(
            session_id=session_id,
            dataset_id=dataset_id,
            action_hash=digest,
            approval_token=approval_token,
            expires_at=expires_at,
            operations=[
                CleaningOperation(
                    transform_id=transform.transform_id,
                    type=str(transform.type),
                    target_column=transform.target_column,
                    description=transform.description,
                    lossy=transform_is_lossy(transform),
                )
                for transform in recipe.transforms
            ],
            preview=CleaningPreviewView(
                dataset_id=preview.dataset_id,
                recipe_id=preview.recipe_id,
                source_version=preview.source_version,
                target_version=target_version,
                row_count_before=preview.row_count_before,
                row_count_after=preview.row_count_after,
                rows_dropped=preview.rows_dropped,
                rows_edited=preview.rows_edited,
                cells_changed=preview.cells_changed,
                column_changes=[
                    CleaningColumnChange(
                        column=diff.column,
                        before_dtype=diff.before_dtype,
                        after_dtype=diff.after_dtype,
                        changed_rows=diff.changed_rows,
                        before_missing=diff.before_missing,
                        after_missing=diff.after_missing,
                    )
                    for diff in preview.column_diffs
                ],
                warnings=list(preview.warnings),
            ),
        )

    def apply(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        llm: Literal["env", "offline"] = "env",
        idempotency_key: str | None = None,
        cancel_check: Callable[[], object] | None = None,
    ) -> CleaningApplied:
        return self._approvals.run_idempotent_producer(
            action_hash,
            session_id=session_id,
            idempotency_key=idempotency_key,
            operation=lambda deadline: self._apply_once(
                session_id,
                action_hash=action_hash,
                approval_token=approval_token,
                llm=llm,
                idempotency_key=idempotency_key,
                contention_deadline=deadline,
                cancel_check=cancel_check,
            ),
        )

    def _apply_once(
        self,
        session_id: str,
        *,
        action_hash: str,
        approval_token: str,
        llm: Literal["env", "offline"],
        idempotency_key: str | None,
        contention_deadline: float,
        cancel_check: Callable[[], object] | None,
    ) -> CleaningApplied:
        idempotency_content = {
            "source_session_id": session_id,
            "action_hash": action_hash,
            "llm": llm,
        }
        # Idempotent replay must win before approval consumption, or a retried
        # apply would 409 on its own already-consumed hash.
        if idempotency_key:
            existing = self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                project_id = self._project_for_run(session_id)
                _payload, replay_payload_digest, _status = (
                    self._approvals.inspect_payload(action_hash, session_id=session_id)
                )
                self._jobs.assert_idempotent_replay(
                    existing,
                    request_scope=session_id,
                    project_id=project_id,
                    kind="auto_eda",
                    content={
                        **idempotency_content,
                        "approval_payload_digest": replay_payload_digest,
                    },
                )
                self._check_replay_matches(existing, session_id=session_id, action_hash=action_hash)
                return self._replayed(session_id, action_hash, existing)
        def validate(payload: dict[str, Any]) -> tuple[CleaningRecipe, str, Any, Path]:
            if cancel_check is not None:
                cancel_check()
            recipe = CleaningRecipe.model_validate(payload.get("recipe"))
            project_id, loaded = self._reload_from_payload(
                payload,
                recipe,
                cancel_check=cancel_check,
            )
            if cancel_check is not None:
                cancel_check()
            output_dir = self._store.project_dir(project_id) / "cleaned"
            if not output_dir.resolve().is_relative_to(self._store.root.resolve()):
                raise CleaningValidationError(
                    "Cleaning output directory escapes the workspace root."
                )
            return recipe, project_id, loaded, output_dir

        payload, validated = self._approvals.validate_then_consume(
            action_hash,
            kind=APPROVAL_KIND_CLEANING,
            session_id=session_id,
            generation=approval_token,
            validate=validate,
            idempotency_key=idempotency_key,
            deadline=contention_deadline,
        )
        with self._approvals.compensate_on_failure(action_hash, session_id=session_id):
            recipe, project_id, loaded, output_dir = validated
            outcome = apply_cleaning(
                loaded,
                recipe,
                output_dir=output_dir,
                approved_hash=action_hash,
                store=self._store,
                project_id=project_id,
                session_id=session_id,
            )
            if cancel_check is not None:
                cancel_check()
            if outcome.status != "applied" or outcome.applied is None:
                raise CleaningApplyRefusedError(
                    outcome.message or "Cleaning apply was refused."
                )
        try:
            cleaned_rel = str(
                outcome.applied.result.output_path.resolve().relative_to(
                    self._store.root.resolve()
                )
            )
            new_session_id = _new_fork_session_id()
            job = self._jobs.create_job(
                new_session_id,
                kind="auto_eda",
                project_id=project_id,
                datasets=[cleaned_rel],
                business_context=FORK_BUSINESS_CONTEXT,
                llm=llm,
                idempotency_key=idempotency_key,
                idempotency_content={
                    **idempotency_content,
                    "approval_payload_digest": payload_digest(payload),
                },
                idempotency_scope=session_id,
            )
            return CleaningApplied(
                session_id=session_id,
                new_session_id=job.session_id,
                dataset_id=recipe.dataset_id,
                target_version=outcome.applied.result.target_version,
                job=_to_created(job),
            )
        except Exception:
            # C6: the approval was consumed and the version written, but no job
            # exists to answer for them — undo both so the same token can retry.
            self._rollback_apply(action_hash, session_id=session_id, applied=outcome.applied)
            raise

    def _rollback_apply(
        self, action_hash: str, *, session_id: str, applied: AppliedCleaning
    ) -> None:
        with suppress(Exception):
            version_dir = applied.result.output_path.parent
            if version_dir.resolve().is_relative_to(self._store.root.resolve()):
                remove_tree(version_dir, ignore_errors=True)
        with suppress(Exception):
            self._store.restore_pending_action(action_hash, session_id=session_id)

    def _check_replay_matches(self, job_row: dict, *, session_id: str, action_hash: str) -> None:
        """Slice-E F1: the idempotency fast path must not bypass the approval
        checks. Replaying is only legitimate when the stored job is a cleaning
        fork in this run's own project AND this run really consumed the hash;
        anything else means the key belongs to a different request."""
        job_id = str(job_row["job_id"])
        run_row = self._store.get_session_index_row(session_id)
        if (
            run_row is None
            or str(job_row["kind"]) != "auto_eda"
            or str(job_row["project_id"]) != str(run_row["project_id"])
        ):
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id} from a different "
                "project or kind.",
            )
        pending = self._store.get_pending_action(action_hash, session_id=session_id)
        if pending is None or str(pending["status"]) != "consumed":
            raise JobConflictError(
                job_id,
                f"Idempotency key already used by job {job_id}, but the action "
                "hash was never consumed by this run.",
            )

    def _replayed(self, session_id: str, action_hash: str, job_row: dict) -> CleaningApplied:
        dataset_id = ""
        row = self._store.get_pending_action(action_hash, session_id=session_id)
        if row is not None:
            with_payload = json.loads(str(row["payload_json"]))
            if isinstance(with_payload, dict):
                dataset_id = str(with_payload.get("dataset_id", ""))
        job = self._jobs.get_job(str(job_row["job_id"]))
        return CleaningApplied(
            session_id=session_id,
            new_session_id=job.session_id,
            dataset_id=dataset_id,
            target_version=None,
            job=_to_created(job),
        )

    def _load(
        self,
        session_id: str,
        dataset_id: str,
        *,
        cancel_check: Callable[[], object] | None = None,
    ) -> tuple[str, str, str, LoadedDataset]:
        handles = self._datasets.list_datasets(session_id)
        handle = next((item for item in handles if item.dataset_id == dataset_id), None)
        if handle is None:
            raise DatasetNotFoundError(dataset_id, session_id)
        if handle.ingest_status != "ready" or not handle.original_uri:
            raise DatasetSourceMissingError(dataset_id)
        # C4: hash the file as previewed (never a possibly stale manifest hash)
        # so apply can detect on-disk drift against exactly this content.
        loaded = load_csv(
            self._store.root / handle.original_uri,
            dataset_id=dataset_id,
            cancel_check=cancel_check,
        )
        return handle.project_id, handle.original_uri, loaded.record.content_hash, loaded

    def _reload_from_payload(
        self,
        payload: dict[str, Any],
        recipe: CleaningRecipe,
        *,
        cancel_check: Callable[[], object] | None = None,
    ) -> tuple[str, LoadedDataset]:
        source_uri = str(payload.get("source_uri", ""))
        source = (self._store.root / source_uri).resolve()
        # The uri came from our own DB, but containment stays cheap insurance.
        if not source_uri or not source.is_relative_to(self._store.root.resolve()):
            raise DatasetSourceMissingError(recipe.dataset_id)
        if not source.is_file():
            raise DatasetSourceMissingError(recipe.dataset_id)
        # C4: recompute the hash — never trust the stored value as current disk
        # state. A file rewritten since the preview must not be cleaned under
        # the old approval.
        actual_hash = hash_file(source, cancel_check=cancel_check)
        if actual_hash != str(payload.get("content_hash") or ""):
            raise CleaningSourceChangedError(recipe.dataset_id)
        project_id = str(payload.get("project_id") or "default")
        loaded = load_csv(
            source,
            dataset_id=recipe.dataset_id,
            content_hash=actual_hash,
            cancel_check=cancel_check,
        )
        return project_id, loaded


def _profile_summary(artifact: Artifact) -> DatasetProfileSummary:
    profile = DatasetProfile.model_validate(artifact.payload)
    return DatasetProfileSummary(
        dataset_id=profile.dataset_id,
        name=profile.name,
        rows=profile.rows,
        columns=profile.columns,
        semantic_type_counts=semantic_type_counts(artifact),
        fields=[FieldProfileRow.model_validate(row) for row in dataset_display_rows(artifact)],
    )


def _int_or_zero(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _build_recipe(
    loaded: LoadedDataset,
    *,
    trim_whitespace: bool,
    drop_duplicate_rows: bool,
    drop_missing_rows: bool,
    drop_outlier_rows: bool,
) -> CleaningRecipe:
    """Deterministic recipe from form options: same options + same dataset →
    same transform ids → same action_hash, which makes preview idempotent."""
    transforms: list[CleaningTransform] = []
    if trim_whitespace:
        # dtype check via pandas type predicates: pandas 3 strings are "str",
        # not "object", so a bare `dtype == object` misses every text column.
        transforms.extend(
            CleaningTransform(
                transform_id=f"trim_{index}",
                type="trim_whitespace",
                target_column=str(column),
                description=f"Trim leading/trailing whitespace in {column}.",
            )
            for index, column in enumerate(loaded.frame.columns)
            if is_object_dtype(loaded.frame[column]) or is_string_dtype(loaded.frame[column])
        )
    if drop_duplicate_rows:
        transforms.append(
            CleaningTransform(
                transform_id="dedupe",
                type="drop_duplicate_rows",
                description="Remove exact duplicate rows.",
            )
        )
    if drop_missing_rows:
        transforms.append(
            CleaningTransform(
                transform_id="drop_missing",
                type="drop_missing_rows",
                description="Drop every row that contains a missing value.",
            )
        )
    if drop_outlier_rows:
        transforms.append(
            CleaningTransform(
                transform_id="drop_outliers",
                type="drop_outlier_rows",
                description="Drop rows with IQR outliers in numeric columns.",
            )
        )
    return CleaningRecipe(
        dataset_id=loaded.record.dataset_id,
        source_version=loaded.record.version,
        recipe_id=f"api_{loaded.record.dataset_id}",
        transforms=transforms,
        created_by="user",
    )


def _new_fork_session_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_clean{uuid.uuid4().hex[:6]}"


def _to_created(job: JobStatus) -> JobCreated:
    return JobCreated(
        job_id=job.job_id,
        session_id=job.session_id,
        status=job.status,
        events_url=job.events_url,
    )

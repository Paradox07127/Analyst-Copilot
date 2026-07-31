"""Typed, workspace-scoped tools for autonomous data-analysis agents."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from eda_platform.agents.runtime import AgentTool, AgentToolResult
from eda_platform.core.permissions import PermissionTier, require_permission
from eda_platform.core.skills_store import load_skills
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import SqlCatalog, run_sql


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadArtifactArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)


class RunSqlArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=20_000)
    purpose: str = Field(min_length=1, max_length=500)


class RunSavedSkillArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1)
    target_dataset_ids: list[str] = Field(min_length=1, max_length=8)


class OpenAnalysisArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=2_000)


OpenAnalysisExecutor = Callable[[OpenAnalysisArguments], AgentToolResult]


@dataclass(slots=True)
class DataToolContext:
    """All values are session-local; tools never accept a host path from a model."""

    datasets: Sequence[LoadedDataset]
    catalog: SqlCatalog
    project_id: str
    session_id: str
    store: ArtifactStore | None
    payload_policy: PayloadPolicy
    artifacts: list[Artifact] = field(default_factory=list)
    open_analysis: OpenAnalysisExecutor | None = None
    _artifacts_by_id: dict[str, Artifact] = field(init=False, repr=False)
    _datasets_by_id: dict[str, LoadedDataset] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._artifacts_by_id = {artifact.id: artifact for artifact in self.artifacts}
        self._datasets_by_id = {dataset.record.dataset_id: dataset for dataset in self.datasets}

    def add_artifact(self, artifact: Artifact, *, persist: bool = True) -> None:
        self._artifacts_by_id[artifact.id] = artifact
        if not any(existing.id == artifact.id for existing in self.artifacts):
            self.artifacts.append(artifact)
        if persist and self.store is not None:
            self.store.save_artifact(artifact)

    def artifact(self, artifact_id: str) -> Artifact:
        artifact = self._artifacts_by_id.get(artifact_id)
        if artifact is None:
            raise ValueError("Artifact is not available in this analysis session.")
        return artifact

    def datasets_for(self, dataset_ids: list[str]) -> list[LoadedDataset]:
        selected: list[LoadedDataset] = []
        seen: set[str] = set()
        for dataset_id in dataset_ids:
            dataset = self._datasets_by_id.get(dataset_id)
            if dataset is None:
                raise ValueError("A selected dataset is not part of this analysis session.")
            if dataset_id in seen:
                raise ValueError("A saved skill target may be selected only once.")
            seen.add(dataset_id)
            selected.append(dataset)
        return selected


def build_data_tools(context: DataToolContext) -> list[AgentTool]:
    """Return the narrow, local-only capability set for one agent task."""
    tools = [
        AgentTool(
            name="inspect_data_catalog",
            description=(
                "Inspect the loaded datasets, their safe relation names, row counts and columns. "
                "Use this before writing SQL or choosing a saved skill."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _inspect_catalog(context),
        ),
        AgentTool(
            name="list_artifacts",
            description=(
                "List existing evidence artifacts from this session. Use it to find profiles, "
                "quality checks, findings and earlier query results before making a claim."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _list_artifacts(context),
        ),
        AgentTool(
            name="read_artifact",
            description=(
                "Read one listed artifact by id. The payload is constrained by this session's "
                "data-disclosure policy."
            ),
            args_schema=ReadArtifactArguments,
            execute=lambda args: _read_artifact(context, cast(ReadArtifactArguments, args)),
        ),
        AgentTool(
            name="run_sql",
            description=(
                "Run one read-only DuckDB SELECT/WITH query over the loaded relations. "
                "Never use file readers, mutation statements, network functions or unlisted tables."
            ),
            args_schema=RunSqlArguments,
            execute=lambda args: _run_sql(context, cast(RunSqlArguments, args)),
        ),
        AgentTool(
            name="list_saved_skills",
            description=(
                "List validated, project-local analysis skills that can be replayed against "
                "compatible currently loaded datasets."
            ),
            args_schema=_NoArguments,
            execute=lambda _args: _list_saved_skills(context),
        ),
        AgentTool(
            name="run_saved_skill",
            description=(
                "Replay one listed saved skill on selected loaded dataset ids. This executes "
                "only the skill's frozen read-only plan through the same local safety gates."
            ),
            args_schema=RunSavedSkillArguments,
            execute=lambda args: _run_saved_skill(context, cast(RunSavedSkillArguments, args)),
        ),
    ]
    open_analysis = context.open_analysis
    if open_analysis is not None:
        tools.append(
            AgentTool(
                name="run_open_analysis",
                description=(
                    "Ask the secured Python analysis tool to perform a custom analysis when "
                    "SQL is insufficient. It receives only mounted local data and runs inside "
                    "the configured sandbox; never request host, network or filesystem access."
                ),
                args_schema=OpenAnalysisArguments,
                execute=lambda args: _run_open_analysis(
                    context,
                    open_analysis,
                    cast(OpenAnalysisArguments, args),
                ),
            )
        )
    return tools


def _run_open_analysis(
    context: DataToolContext,
    execute: OpenAnalysisExecutor,
    args: OpenAnalysisArguments,
) -> AgentToolResult:
    result = execute(args)
    for artifact in result.artifacts:
        # The secured code executor persists its own artifacts. Register them
        # in the live context so a later agent step can inspect and cite them.
        context.add_artifact(artifact, persist=False)
    return result


def _inspect_catalog(context: DataToolContext) -> AgentToolResult:
    rows: list[dict[str, Any]] = []
    for dataset in context.datasets:
        relation = context.catalog.relations[dataset.record.name]
        rows.append(
            {
                "dataset_id": dataset.record.dataset_id,
                "name": dataset.record.name,
                "relation": relation,
                "rows": int(len(dataset.frame)),
                "columns": [str(column) for column in dataset.frame.columns],
            }
        )
    profiles = [
        artifact
        for artifact in context.artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    return AgentToolResult(
        content={
            "datasets": rows,
            "profile_artifact_ids": [artifact.id for artifact in profiles],
        },
        # Catalog claims can now be persisted as evidence even when the answer
        # needs no query beyond the already-produced dataset profiles.
        artifacts=profiles,
    )


def _list_artifacts(context: DataToolContext) -> AgentToolResult:
    return AgentToolResult(
        content={
            "artifacts": [
                {
                    "artifact_id": artifact.id,
                    "type": artifact.type.value,
                    "warnings": artifact.warnings[:5],
                    "evidence_count": len(artifact.evidence),
                }
                for artifact in context.artifacts
            ]
        }
    )


def _read_artifact(context: DataToolContext, args: ReadArtifactArguments) -> AgentToolResult:
    artifact = context.artifact(args.artifact_id)
    payload: dict[str, Any] | None
    if context.payload_policy == "schema_only":
        payload = None
    elif artifact.type is ArtifactType.RAW_DATA_PREVIEW:
        # Raw rows are intentionally never made available just because an agent
        # asked to inspect an artifact. Aggregates/profiles remain available.
        payload = {"notice": "Raw data preview content is withheld by the agent tool policy."}
    else:
        payload = _clip_json(artifact.payload)
    return AgentToolResult(
        content={
            "artifact_id": artifact.id,
            "type": artifact.type.value,
            "payload": payload,
            "warnings": artifact.warnings[:10],
            "evidence": [item.model_dump(mode="json") for item in artifact.evidence[:30]],
        },
        artifacts=[artifact],
    )


def _run_sql(context: DataToolContext, args: RunSqlArguments) -> AgentToolResult:
    decision = require_permission({"type": "duckdb_select", "sql": args.sql})
    if decision.tier is PermissionTier.DENY:
        raise ValueError(decision.feedback)
    artifact = run_sql(
        context.catalog,
        args.sql,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    context.add_artifact(artifact)
    payload = _clip_json(artifact.payload)
    return AgentToolResult(
        content={
            "purpose": args.purpose,
            "artifact_id": artifact.id,
            "result": payload,
        },
        artifacts=[artifact],
    )


def _list_saved_skills(context: DataToolContext) -> AgentToolResult:
    if context.store is None:
        return AgentToolResult(content={"skills": []})
    skills = load_skills(context.store.project_dir(context.project_id))
    return AgentToolResult(
        content={
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "question": skill.plan.question,
                    "expected_datasets": skill.expected_datasets,
                    "required_columns": skill.param_columns,
                }
                for skill in skills
            ]
        }
    )


def _run_saved_skill(
    context: DataToolContext,
    args: RunSavedSkillArguments,
) -> AgentToolResult:
    if context.store is None:
        raise ValueError("Saved skills are unavailable because this chat has no workspace store.")
    skill = next(
        (
            item
            for item in load_skills(context.store.project_dir(context.project_id))
            if item.skill_id == args.skill_id
        ),
        None,
    )
    if skill is None:
        raise ValueError("Saved skill was not found in this project.")
    # `analysis_skill` re-enters the chat driver for its frozen execution path,
    # so importing it only when the tool is actually invoked keeps the chat
    # driver's module graph acyclic.
    from eda_platform.drivers.analysis_skill import replay_skill

    result = replay_skill(
        skill,
        context.datasets_for(args.target_dataset_ids),
        store=context.store,
        project_id=context.project_id,
        session_id=context.session_id,
    )
    for artifact in result.artifacts:
        # replay_skill persists through the approved-plan execution path.
        context.add_artifact(artifact, persist=False)
    return AgentToolResult(
        content={
            "skill_id": skill.skill_id,
            "skill_name": skill.name,
            "status": result.status,
            "message": result.message,
            "sql": result.sql,
            "artifact_ids": [artifact.id for artifact in result.artifacts],
        },
        artifacts=list(result.artifacts),
    )


def _clip_json(value: Any, *, max_chars: int = 12_000) -> Any:
    """Keep tool observations bounded even when an artifact contains a table."""
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return value
    return {
        "truncated": True,
        "preview": encoded[:max_chars],
        "original_chars": len(encoded),
    }

"""Save/replay logic for :class:`AnalysisSkill`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from eda_platform.core.permissions import action_hash, analysis_plan_action
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_exists,
    raise_for_violations,
)
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.chat import ChatTurnResult
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.skills import AnalysisSkill
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog, rewrite_relation_names

_REPLAY_TOOL = "replay_analysis_skill"


def skill_from_plan(
    plan: AnalysisPlan,
    name: str,
    description: str = "",
    source_session_id: str | None = None,
) -> AnalysisSkill:
    """Freeze a validated plan into a named skill."""
    return AnalysisSkill(
        name=name,
        description=description,
        plan=plan,
        param_columns=list(plan.columns),
        expected_datasets=list(plan.dataset_names),
        source_session_id=source_session_id,
    )


def replay_skill(
    skill: AnalysisSkill,
    target_datasets: Sequence[LoadedDataset],
    *,
    store: ArtifactStore | None,
    project_id: str,
    session_id: str,
    cancel_check: Callable[[], bool] | None = None,
) -> ChatTurnResult:
    """Replay a skill's frozen plan against new datasets, deterministically."""
    raise_if_cancelled(cancel_check, operation="skill replay")
    _guard_columns_present(skill, target_datasets)
    rewritten = _rewrite_plan_for_targets(skill, target_datasets)
    approved_hash = action_hash(analysis_plan_action(rewritten))
    raise_if_cancelled(cancel_check, operation="skill replay")
    result = run_chat_turn(
        rewritten.question,
        datasets=target_datasets,
        project_id=project_id,
        session_id=session_id,
        llm=_ReplayPlaceholderLLM(),
        store=store,
        approved_plan=rewritten,
        approved_action_hash=approved_hash,
    )
    raise_if_cancelled(cancel_check, operation="skill replay")
    return result


def _guard_columns_present(
    skill: AnalysisSkill,
    target_datasets: Sequence[LoadedDataset],
) -> None:
    """Reject replay (teaching-style) unless every param column exists downstream."""
    available = _available_columns(target_datasets)
    violations = [
        check_column_exists(
            "param_columns",
            column,
            available,
            fix_hint=(
                "Replay this skill only against a dataset that has every column "
                "the saved plan reads, or edit the plan before saving it as a skill."
            ),
        )
        for column in skill.param_columns
    ]
    raise_for_violations(_REPLAY_TOOL, violations)


def _rewrite_plan_for_targets(
    skill: AnalysisSkill,
    target_datasets: Sequence[LoadedDataset],
) -> AnalysisPlan:
    """Rebind the frozen plan's relation names onto the replay targets."""
    mapping = _relation_mapping(skill, target_datasets)
    new_sql = _rewrite_relations(skill.plan.sql, mapping)
    new_dataset_names = _remap_dataset_names(skill.plan.dataset_names, mapping)
    return skill.plan.model_copy(update={"sql": new_sql, "dataset_names": new_dataset_names})


def _relation_mapping(
    skill: AnalysisSkill,
    target_datasets: Sequence[LoadedDataset],
) -> dict[str, str]:
    """Map each source relation name to a target relation name."""
    expected = skill.expected_datasets
    target_relations = _target_relation_names(target_datasets)
    if not target_relations:
        raise_for_violations(
            _REPLAY_TOOL,
            [
                GuardViolation(
                    field="target_datasets",
                    got=[],
                    allowed="at least one target dataset to replay against",
                    fix_hint="Select one or more datasets to replay this skill on.",
                    problem="no target datasets were provided.",
                )
            ],
        )
    if len(target_relations) == 1:
        sole = target_relations[0]
        return {name: sole for name in expected}
    if len(expected) == len(target_relations):
        return dict(zip(expected, target_relations, strict=True))
    raise_for_violations(
        _REPLAY_TOOL,
        [
            GuardViolation(
                field="target_datasets",
                got=len(target_relations),
                allowed=(
                    f"exactly 1 target dataset, or {len(expected)} to match the "
                    "datasets this skill referenced"
                ),
                fix_hint=(
                    "Pick a single target dataset to run the whole analysis on, or "
                    f"select {len(expected)} datasets to map one-to-one."
                ),
                problem=(
                    f"cannot map {len(expected)} referenced datasets onto "
                    f"{len(target_relations)} targets."
                ),
            )
        ],
    )
    raise AssertionError("unreachable: raise_for_violations raised above")


def _target_relation_names(target_datasets: Sequence[LoadedDataset]) -> list[str]:
    catalog = build_catalog(target_datasets)
    return [catalog.relations[dataset.record.name] for dataset in target_datasets]


def _rewrite_relations(sql: str, mapping: dict[str, str]) -> str:
    """Replace source relation tokens with their target names.

    Delegates to the shared scanner: the previous whole-word regex also
    rewrote inside string literals, so a skill selecting
    ``WHERE channel = 'orders'`` silently changed which rows it returned when
    replayed on another dataset (review J1).
    """
    return rewrite_relation_names(sql, mapping)


def _remap_dataset_names(dataset_names: list[str], mapping: dict[str, str]) -> list[str]:
    remapped: list[str] = []
    for name in dataset_names:
        target = mapping.get(name, name)
        if target not in remapped:
            remapped.append(target)
    return remapped


def _available_columns(target_datasets: Sequence[LoadedDataset]) -> set[str]:
    columns: set[str] = set()
    for dataset in target_datasets:
        columns.update(str(column) for column in dataset.frame.columns)
    return columns


class _ReplayPlaceholderLLM:
    """Satisfies ``run_chat_turn``'s required ``llm`` on the approved-plan path."""

    def structured(self, *, task: str, schema: Any, payload: dict) -> Any:
        raise AssertionError("LLM must not be called during deterministic skill replay.")

    def text(self, *, task: str, payload: dict) -> str:
        raise AssertionError("LLM must not be called during deterministic skill replay.")

"""Role 2 (Investigation) orchestration with rewritten trust boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eda_platform import __version__
from eda_platform.agents.followup_agent import (
    ParentFinding,
    followup_question_candidates,
    generate_followup_proposals,
)
from eda_platform.agents.interpretation import InterpretationResult, interpret_findings
from eda_platform.agents.investigation_loop import run_bounded_loop
from eda_platform.core.budget import (
    BudgetExceeded,
    SessionBudgetPolicy,
    SessionBudgetState,
)
from eda_platform.core.claim_language import contains_causal_phrase, implies_causation
from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.ids import INTERNAL_SESSION_MARKER, make_artifact_id, stable_hash
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import LLMClient, OfflineLLMClient, is_offline_client
from eda_platform.core.llm_ledger import (
    LLM_USAGE_EVENT,
    meter_llm_client,
    restore_run_budget_state,
)
from eda_platform.core.loop_fingerprint import finding_fingerprint, question_fingerprint
from eda_platform.core.loop_journal import JsonlLoopJournal, LoopTransitionError
from eda_platform.core.loop_ledger import (
    admit_finding,
    is_duplicate_finding,
    keep_or_discard,
    record_round,
)
from eda_platform.core.session_metrics import persist_run_metrics
from eda_platform.core.semantic import SemanticSeeds
from eda_platform.core.semantic_resources import load_semantic_seeds_safe
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import (
    ToolGuardError,
    infer_column_semantic_type,
)
from eda_platform.core.trace import trace_event
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.drivers.question_exec import (
    _load_source_datasets,
    execute_question_candidate,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, SqlResult
from eda_platform.schemas.deep_investigation import DeepInvestigationResult
from eda_platform.schemas.investigations import (
    ClaimClass,
    InvestigationApproval,
    InvestigationGate,
    InvestigationPlan,
    InvestigationRecord,
    ReliabilityRating,
    ValidatedFinding,
)
from eda_platform.schemas.loop import (
    DEPTH_PROFILES,
    LoopLedger,
    LoopRoundRecord,
    MacroLoopExitReason,
)
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.quality_context import QualityContext, QualityContextSet
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionExecutionResult,
    QuestionFinding,
)
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSection
from eda_platform.schemas.sessions import SessionManifest, TraceEvent
from eda_platform.schemas.stats import StatTestResult, StatTestType
from eda_platform.tools.evidence import EvidencePack, build_evidence_pack
from eda_platform.tools.investigation_methods import select_investigation_method
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.ml_baseline import (
    create_model_card_artifact,
    guard_baseline_model_params,
    run_baseline_model,
)
from eda_platform.tools.report_validator import extract_numbers, validate_report_bundle
from eda_platform.tools.sql_runner import build_catalog
from eda_platform.tools.stat_tests import (
    create_anova_boxplot_artifact,
    create_stat_test_artifact,
    guard_stat_test_params,
    run_stat_test,
)

# Method-family aliases accepted from persisted and current plans.
_GROUP_COMPARISON_FAMILIES = frozenset({"group_comparison"})
_PREDICTION_FAMILIES = frozenset({"predictive_modeling", "outcome_prediction"})
_ANOMALY_FAMILIES = frozenset({"anomaly_detection"})

# The marker binds deep investigation to the approved tool set.
_DEEP_MARKER_TOOL = "llm_probe_planner"
_DEEP_ASSUMPTION = (
    "Deep investigation enabled: up to 3 follow-up probes within the approved "
    "scope (LLM plans, deterministic execution)."
)
_DEEP_MAX_STEPS = 3
_DEEP_LLM_CALL_CAP = 8

# Deterministic claim caveats forced onto every method finding's limitations.
_OBSERVED_CAVEAT = "This is an observed result in the current data, not a causal claim."
_PREDICTIVE_BASELINE_CAVEAT = (
    "This is a baseline predictive estimate reported only within its measured "
    "performance metrics; it is associative, not a causal explanation of the outcome."
)


@dataclass(frozen=True)
class SkippedPlan:
    """A selected plan refused before execution, with the fail-closed reason."""

    plan_id: str
    investigation_id: str
    reason: str


@dataclass(frozen=True)
class InvestigationRunResult:
    project_id: str
    session_id: str
    source_session_id: str
    artifacts: list[Artifact]
    workspace: Path
    skipped: list[SkippedPlan] = field(default_factory=list)


def create_investigation_plans(
    *,
    project_id: str,
    source_session_id: str,
    question_ids: Sequence[str],
    workspace: Path | str,
    session_id: str | None = None,
    deep: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> InvestigationRunResult:
    """Create immutable, user-reviewable plans without executing analysis."""
    raise_if_cancelled(cancel_check, operation="investigation planning")
    workspace_path = require_absolute_workspace(workspace)
    store = ArtifactStore(workspace_path)
    source_artifacts = store.list_artifacts(project_id=project_id, session_id=source_session_id)
    source_qcand = _source_candidate_set(source_artifacts)
    candidate_set = QuestionCandidateSet.model_validate(source_qcand.payload)
    actual_session_id = session_id or _generate_plan_session_id(source_session_id, question_ids)
    store.start_session(project_id, actual_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=actual_session_id,
            project_id=project_id,
            input_hashes={source_session_id: "derived_question_cards"},
            code_version="investigation-orchestrator-v2",
            model_versions={"orchestrator": "deterministic"},
        )
    )

    available_datasets = _available_dataset_names(source_artifacts)
    profiles_by_name = _profiles_by_name(source_artifacts)
    quality_context_by_id = _quality_context_by_id(source_artifacts)
    candidates_by_id = {candidate.question_id: candidate for candidate in candidate_set.candidates}
    artifacts: list[Artifact] = []
    for question_id in dict.fromkeys(question_ids):
        raise_if_cancelled(cancel_check, operation="investigation planning")
        candidate = candidates_by_id.get(question_id)
        if candidate is None:
            artifacts.append(
                _unknown_question_record(
                    question_id=question_id,
                    project_id=project_id,
                    session_id=actual_session_id,
                    parent_ids=[source_qcand.id],
                )
            )
            continue
        plan = _build_plan(
            candidate,
            source_session_id=source_session_id,
            session_id=actual_session_id,
            available_datasets=available_datasets,
            profiles_by_name=profiles_by_name,
            quality_context_by_id=quality_context_by_id,
            source_qcand_id=source_qcand.id,
            deep=deep,
        )
        plan_artifact = _plan_artifact(plan, project_id=project_id, session_id=actual_session_id)
        artifacts.append(plan_artifact)
        if plan.status == "needs_data":
            artifacts.append(
                _record_artifact(
                    _plan_blocked_record(plan),
                    project_id=project_id,
                    session_id=actual_session_id,
                    parent_ids=[plan_artifact.id],
                )
            )

    raise_if_cancelled(cancel_check, operation="investigation planning")
    for artifact in artifacts:
        raise_if_cancelled(cancel_check, operation="investigation planning")
        store.save_artifact(artifact)
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=actual_session_id,
            event_type="investigation_plans_created",
            name="investigation_orchestrator",
            finished_at=datetime.now(UTC),
            summary={
                "requested_question_count": len(set(question_ids)),
                "plan_count": sum(
                    artifact.type is ArtifactType.INVESTIGATION_PLAN for artifact in artifacts
                ),
            },
        ),
    )
    store.mark_session_status(project_id, actual_session_id, "awaiting_approval")
    return InvestigationRunResult(
        project_id=project_id,
        session_id=actual_session_id,
        source_session_id=source_session_id,
        artifacts=artifacts,
        workspace=workspace_path,
    )


def approve_plan(
    *,
    project_id: str,
    plan_session_id: str,
    plan_id: str,
    workspace: Path | str,
    reason: str = "",
) -> Artifact:
    """Persist a user approval bound to the exact plan content (H5)."""
    return _record_decision(
        project_id=project_id,
        plan_session_id=plan_session_id,
        plan_id=plan_id,
        workspace=workspace,
        decision="approved",
        reason=reason,
    )[0]


def reject_plan(
    *,
    project_id: str,
    plan_session_id: str,
    plan_id: str,
    workspace: Path | str,
    reason: str,
) -> list[Artifact]:
    """Persist a user rejection approval plus a rejection investigation record."""
    return _record_decision(
        project_id=project_id,
        plan_session_id=plan_session_id,
        plan_id=plan_id,
        workspace=workspace,
        decision="rejected",
        reason=reason,
    )


def _record_decision(
    *,
    project_id: str,
    plan_session_id: str,
    plan_id: str,
    workspace: Path | str,
    decision: str,
    reason: str,
) -> list[Artifact]:
    store = ArtifactStore(Path(workspace))
    plan_artifact = store.get_artifact(
        plan_id,
        project_id=project_id,
        session_id=plan_session_id,
    )
    if plan_artifact.type is not ArtifactType.INVESTIGATION_PLAN:
        raise ValueError("Approval target must be an InvestigationPlan artifact.")
    plan = InvestigationPlan.model_validate(plan_artifact.payload)
    records_by_investigation = _records_by_investigation(
        store.list_artifacts(project_id=project_id, session_id=plan_session_id)
    )
    if decision == "rejected" and _has_execution_outcome(
        records_by_investigation.get(plan.investigation_id, [])
    ):
        raise ValueError("Cannot reject a plan after an execution outcome exists.")
    approval = InvestigationApproval(
        approval_id="appr_"
        + stable_hash(
            {
                "investigation_id": plan.investigation_id,
                "plan_fingerprint": _plan_fingerprint(plan),
                "decision": decision,
            },
            length=12,
        ),
        investigation_id=plan.investigation_id,
        plan_fingerprint=_plan_fingerprint(plan),
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
        decided_at=datetime.now(UTC).isoformat(),
    )
    approval_artifact = _approval_artifact(
        approval, project_id=project_id, session_id=plan_session_id, parent_ids=[plan_id]
    )
    store.save_artifact(approval_artifact)
    artifacts = [approval_artifact]
    if decision == "rejected":
        record = InvestigationRecord(
            record_id=_record_id(plan.investigation_id, "rejected"),
            investigation_id=plan.investigation_id,
            question_id=plan.question_id,
            status="rejected",
            reason_code="user_rejected_plan",
            reason=reason or "The user rejected this investigation plan before execution.",
            next_action="Revise the Question Card or select a different investigation.",
            validation_gates=plan.validation_gates,
            source_artifact_ids=[plan_id, approval_artifact.id],
        )
        record_artifact = _record_artifact(
            record,
            project_id=project_id,
            session_id=plan_session_id,
            parent_ids=[plan_id, approval_artifact.id],
        )
        store.save_artifact(record_artifact)
        artifacts.append(record_artifact)
    return artifacts


def execute_investigation_plans(
    *,
    project_id: str,
    plan_session_id: str,
    plan_ids: Sequence[str],
    workspace: Path | str,
    llm: LLMClient | None = None,
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    budget_policy: SessionBudgetPolicy | None = None,
    restored_session_budget: SessionBudgetState | None = None,
    llm_session_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> InvestigationRunResult:
    """Execute approved plans and close run lifecycle on hard terminal failures."""
    raise_if_cancelled(cancel_check, operation="investigation execution")
    try:
        return _execute_investigation_plans(
            project_id=project_id,
            plan_session_id=plan_session_id,
            plan_ids=plan_ids,
            workspace=workspace,
            llm=llm,
            preview_rows=preview_rows,
            timeout_seconds=timeout_seconds,
            budget_policy=budget_policy,
            restored_session_budget=restored_session_budget,
            llm_session_id=llm_session_id,
            cancel_check=cancel_check,
        )
    except (BudgetExceeded, LoopTransitionError):
        store = ArtifactStore(Path(workspace))
        store.mark_session_status(project_id, plan_session_id, "failed")
        try:
            persist_run_metrics(store, project_id, plan_session_id)
        except Exception:  # noqa: BLE001 - preserve the original terminal error
            pass
        raise


def _execute_investigation_plans(
    *,
    project_id: str,
    plan_session_id: str,
    plan_ids: Sequence[str],
    workspace: Path | str,
    llm: LLMClient | None = None,
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    budget_policy: SessionBudgetPolicy | None = None,
    restored_session_budget: SessionBudgetState | None = None,
    llm_session_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> InvestigationRunResult:
    """Execute approved plans and emit either a finding or an investigation record.

    Plans that are blocked (not planned, not execution-ready, or already resolved)
    are skipped with a reason in ``result.skipped`` instead of failing the batch.
    """
    raise_if_cancelled(cancel_check, operation="investigation execution")
    workspace_path = require_absolute_workspace(workspace)
    store = ArtifactStore(workspace_path)
    plan_artifacts = store.list_artifacts(project_id=project_id, session_id=plan_session_id)
    plans_by_id = {
        artifact.id: (artifact, InvestigationPlan.model_validate(artifact.payload))
        for artifact in plan_artifacts
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
    }
    if not plans_by_id:
        raise ValueError("Plan run does not contain InvestigationPlan artifacts.")
    selected_ids = list(dict.fromkeys(plan_ids))
    if not selected_ids:
        raise ValueError("Select at least one known InvestigationPlan artifact.")
    unknown_plan_ids = [plan_id for plan_id in selected_ids if plan_id not in plans_by_id]
    if unknown_plan_ids:
        raise ValueError("Selected InvestigationPlan artifacts are unavailable.")
    selected = [plans_by_id[plan_id] for plan_id in selected_ids]
    records_by_investigation = _records_by_investigation(plan_artifacts)
    # Fail-closed per plan: a blocked plan is skipped with its reason instead of
    # aborting the whole batch (a single not-ready plan used to kill every run).
    executable: list[tuple[Artifact, InvestigationPlan]] = []
    skipped: list[SkippedPlan] = []
    for plan_artifact, plan in selected:
        raise_if_cancelled(cancel_check, operation="investigation execution")
        blockers = _plan_execution_blockers(plan, records_by_investigation)
        if blockers:
            skipped.append(
                SkippedPlan(
                    plan_id=plan_artifact.id,
                    investigation_id=plan.investigation_id,
                    reason="; ".join(blockers),
                )
            )
        else:
            executable.append((plan_artifact, plan))
    approvals_by_investigation = _approvals_by_investigation(plan_artifacts)
    approval_blockers = _approval_blockers(executable, approvals_by_investigation)
    if approval_blockers:
        raise ValueError(
            "Execution requires a matching approval artifact: " + "; ".join(approval_blockers)
        )

    source_session_ids = {plan.source_session_id for _, plan in selected}
    if len(source_session_ids) != 1:
        raise ValueError("Selected plans must originate from the same source run.")
    source_session_id = source_session_ids.pop()
    source_artifacts = store.list_artifacts(project_id=project_id, session_id=source_session_id)
    role_sets_by_name = _role_sets_by_name(source_artifacts)
    source_qcand = _source_candidate_set(source_artifacts)
    candidate_set = QuestionCandidateSet.model_validate(source_qcand.payload)
    candidates_by_id = {candidate.question_id: candidate for candidate in candidate_set.candidates}
    datasets = _load_source_datasets(
        store,
        project_id=project_id,
        source_session_id=source_session_id,
        source_artifacts=source_artifacts,
    )
    effective_budget_policy = (
        restored_session_budget.policy
        if restored_session_budget is not None
        else budget_policy or SessionBudgetPolicy()
    )
    if budget_policy is not None and budget_policy != effective_budget_policy:
        raise ValueError("Restored run budget policy does not match budget_policy.")
    accounting_session_id = llm_session_id or plan_session_id
    session_budget = restored_session_budget or restore_run_budget_state(
        effective_budget_policy,
        store.list_trace_events(project_id=project_id, session_id=accounting_session_id),
    )

    def _emit_usage(event: TraceEvent) -> None:
        store.append_trace(project_id, event)

    run_llm = meter_llm_client(
        llm or OfflineLLMClient(),
        session_id=accounting_session_id,
        emit=_emit_usage,
        budget=session_budget,
        session_dir=store.session_dir(project_id, accounting_session_id),
    )
    seeds: SemanticSeeds | None = load_semantic_seeds_safe(store, project_id)
    artifacts: list[Artifact] = []
    for plan_artifact, plan in executable:
        raise_if_cancelled(cancel_check, operation="investigation execution")
        session_budget.check_wall_time()
        candidate = candidates_by_id.get(plan.question_id)
        if candidate is None:
            artifacts.append(
                _persist(
                    store,
                    _record_artifact(
                        _missing_candidate_record(plan),
                        project_id=project_id,
                        session_id=plan_session_id,
                        parent_ids=[plan_artifact.id, source_qcand.id],
                    ),
                )
            )
            continue
        if _candidate_fingerprint(candidate) != plan.candidate_fingerprint:
            artifacts.append(
                _persist(
                    store,
                    _record_artifact(
                        _stale_candidate_record(plan),
                        project_id=project_id,
                        session_id=plan_session_id,
                        parent_ids=[plan_artifact.id, source_qcand.id],
                        artifact_id=_outcome_artifact_id(plan.investigation_id),
                    ),
                )
            )
            continue
        out_of_scope = sorted(set(candidate.target_datasets) - set(plan.target_datasets))
        scoped_datasets = _scoped_datasets(datasets, plan)
        if out_of_scope or not scoped_datasets:
            artifacts.append(
                _persist(
                    store,
                    _record_artifact(
                        _scope_failed_record(plan, out_of_scope),
                        project_id=project_id,
                        session_id=plan_session_id,
                        parent_ids=[plan_artifact.id, source_qcand.id],
                        artifact_id=_outcome_artifact_id(plan.investigation_id),
                    ),
                )
            )
            continue
        marker_id = _outcome_artifact_id(plan.investigation_id)
        loop_journal: JsonlLoopJournal | None = None
        if _plan_requests_deep(plan) and not is_offline_client(run_llm):
            loop_journal = _initialize_loop_journal(
                store,
                project_id=project_id,
                session_id=plan_session_id,
                plan=plan,
            )
        execution_guard = (
            loop_journal.execution_lock()
            if loop_journal is not None
            else nullcontext()
        )
        with execution_guard:
            primary_checkpoint = (
                _load_primary_checkpoint(
                    store,
                    loop_journal,
                    project_id=project_id,
                    session_id=plan_session_id,
                )
                if loop_journal is not None
                else None
            )
            if primary_checkpoint is None:
                if loop_journal is not None:
                    _mark_primary_execution_started(loop_journal)
                # Persist only for a fresh primary attempt. Rewriting this ID
                # during resume would replace the checkpointed terminal record
                # with an executing marker before it could be loaded.
                _persist(
                    store,
                    _record_artifact(
                        _executing_marker_record(plan),
                        project_id=project_id,
                        session_id=plan_session_id,
                        parent_ids=[plan_artifact.id],
                        artifact_id=marker_id,
                    ),
                )
                store.append_trace(
                    project_id,
                    TraceEvent(
                        session_id=plan_session_id,
                        event_type="investigation_started",
                        name="investigation_orchestrator",
                        summary={"investigation_id": plan.investigation_id},
                    ),
                )
                raise_if_cancelled(cancel_check, operation="investigation execution")
                execution_artifacts, outcome_artifacts = _execute_by_method(
                    plan,
                    candidate=candidate,
                    scoped_datasets=scoped_datasets,
                    project_id=project_id,
                    session_id=plan_session_id,
                    plan_artifact_id=plan_artifact.id,
                    source_qcand_id=source_qcand.id,
                    marker_id=marker_id,
                    llm=run_llm,
                    preview_rows=preview_rows,
                    timeout_seconds=timeout_seconds,
                    seeds=seeds,
                    role_sets=role_sets_by_name,
                )
                raise_if_cancelled(cancel_check, operation="investigation execution")
                if loop_journal is not None:
                    for artifact in (*execution_artifacts, *outcome_artifacts):
                        store.save_artifact(artifact)
                    _write_primary_checkpoint(
                        loop_journal,
                        execution_artifacts,
                        outcome_artifacts,
                    )
            else:
                execution_artifacts, outcome_artifacts = primary_checkpoint
            if loop_journal is not None:

                def _deep_trace_sink(event: TraceEvent) -> None:
                    store.append_trace(project_id, event)

                raise_if_cancelled(cancel_check, operation="deep investigation")
                execution_artifacts, outcome_artifacts = _run_deep_investigation(
                    plan,
                    scoped_datasets=scoped_datasets,
                    execution_artifacts=execution_artifacts,
                    outcome_artifacts=outcome_artifacts,
                    project_id=project_id,
                    session_id=plan_session_id,
                    plan_artifact_id=plan_artifact.id,
                    marker_id=marker_id,
                    llm=run_llm,
                    trace_sink=_deep_trace_sink,
                    journal=loop_journal,
                )
                raise_if_cancelled(cancel_check, operation="deep investigation")
            for artifact in (*execution_artifacts, *outcome_artifacts):
                raise_if_cancelled(cancel_check, operation="investigation execution")
                store.save_artifact(artifact)
            artifacts.extend(execution_artifacts)
            artifacts.extend(outcome_artifacts)
            store.append_trace(
                project_id,
                TraceEvent(
                    session_id=plan_session_id,
                    event_type="investigation_completed",
                    name="investigation_orchestrator",
                    finished_at=datetime.now(UTC),
                    summary={
                        "investigation_id": plan.investigation_id,
                        "outcomes": [artifact.type.value for artifact in outcome_artifacts],
                    },
                ),
            )
    raise_if_cancelled(cancel_check, operation="investigation execution")
    if executable:
        store.mark_session_status(project_id, plan_session_id, "completed")
    try:
        persist_run_metrics(store, project_id, plan_session_id)
    except Exception as exc:  # noqa: BLE001 - metrics must not invalidate completed work
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=plan_session_id,
                event_type="run_metrics_failed",
                name="persist_run_metrics",
                finished_at=datetime.now(UTC),
                summary={"error": f"{type(exc).__name__}: {str(exc)[:300]}"},
            ),
        )
    return InvestigationRunResult(
        project_id=project_id,
        session_id=plan_session_id,
        source_session_id=source_session_id,
        artifacts=artifacts,
        workspace=workspace_path,
        skipped=skipped,
    )


def _execute_by_method(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    source_qcand_id: str,
    marker_id: str,
    llm: LLMClient,
    preview_rows: int,
    timeout_seconds: float,
    seeds: SemanticSeeds | None = None,
    role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> tuple[list[Artifact], list[Artifact]]:
    """Route an approved, in-scope plan to its real method executor."""
    family = plan.method_family
    if family in _GROUP_COMPARISON_FAMILIES:
        return _run_group_comparison(
            plan,
            candidate=candidate,
            scoped_datasets=scoped_datasets,
            project_id=project_id,
            session_id=session_id,
            plan_artifact_id=plan_artifact_id,
            source_qcand_id=source_qcand_id,
            marker_id=marker_id,
            llm=llm,
            seeds=seeds,
            role_sets=role_sets,
        )
    if family in _PREDICTION_FAMILIES:
        return _run_outcome_prediction(
            plan,
            candidate=candidate,
            scoped_datasets=scoped_datasets,
            project_id=project_id,
            session_id=session_id,
            plan_artifact_id=plan_artifact_id,
            source_qcand_id=source_qcand_id,
            marker_id=marker_id,
            llm=llm,
            seeds=seeds,
            role_sets=role_sets,
        )
    if family in _ANOMALY_FAMILIES:
        return _run_anomaly_detection(
            plan,
            candidate=candidate,
            scoped_datasets=scoped_datasets,
            project_id=project_id,
            session_id=session_id,
            plan_artifact_id=plan_artifact_id,
            source_qcand_id=source_qcand_id,
            marker_id=marker_id,
            llm=llm,
            seeds=seeds,
            role_sets=role_sets,
        )
    # Descriptive families use the read-only SQL path.
    execution_artifacts = execute_question_candidate(
        candidate,
        datasets=scoped_datasets,
        project_id=project_id,
        session_id=session_id,
        parent_ids=[plan_artifact_id, source_qcand_id],
        llm=llm,
        preview_rows=preview_rows,
        timeout_seconds=timeout_seconds,
        seeds=seeds,
    )
    outcome_artifacts = _validate_execution(
        plan,
        candidate=candidate,
        execution_artifacts=execution_artifacts,
        project_id=project_id,
        session_id=session_id,
        parent_ids=[plan_artifact_id, *[artifact.id for artifact in execution_artifacts]],
        record_artifact_id=marker_id,
        llm=llm,
        seeds=seeds,
    )
    return execution_artifacts, outcome_artifacts


def _run_group_comparison(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    source_qcand_id: str,
    marker_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
    role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> tuple[list[Artifact], list[Artifact]]:
    parents = [plan_artifact_id, source_qcand_id]
    selection = _select_group_comparison_columns(candidate, scoped_datasets)
    if selection is None:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(
                    "No target dataset in the approved scope exposes a usable grouping "
                    "column (2-20 groups) paired with a numeric measure, so a group "
                    "comparison cannot run."
                ),
                next_action=(
                    "Revise the Question Card to name a grouping column and numeric measure."
                ),
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    dataset, group_column, value_column, group_count = selection
    frame = _scoped_frame(scoped_datasets, dataset)
    test_type: StatTestType = "independent_t_test" if group_count == 2 else "welch_anova"
    try:
        guard_stat_test_params(
            frame,
            test_type=test_type,
            group_column=group_column,
            value_column=value_column,
            category_column=None,
            comparison_count=1,
        )
        result = run_stat_test(
            frame,
            dataset_id=dataset.record.dataset_id,
            test_type=test_type,
            group_column=group_column,
            value_column=value_column,
        )
    except (ToolGuardError, ValueError) as exc:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(
                    "The comparison could not be computed on the approved data "
                    f"({type(exc).__name__}). A constant measure or an empty group "
                    "leaves no valid statistic."
                ),
                next_action="Check the grouping and measure columns before retrying.",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    stat_artifact = create_stat_test_artifact(
        result, project_id=project_id, session_id=session_id, parents=parents
    )
    execution_artifacts: list[Artifact] = [stat_artifact]
    boxplot = create_anova_boxplot_artifact(
        frame, result, project_id=project_id, session_id=session_id, parents=[stat_artifact.id]
    )
    if boxplot is not None:
        execution_artifacts.append(boxplot)
    reducer = _load_reducer("stat_findings")
    if reducer is None:
        return execution_artifacts, [
            _method_unavailable_artifact(
                plan,
                what="group-comparison findings reducer",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    findings = reducer(
        result, stat_artifact.id, role_set=(role_sets or {}).get(dataset.record.name)
    )
    assumption_warnings = _assumption_warnings(result)
    warning_notes = [warning.message for warning in result.warnings]
    extra_limitations = list(assumption_warnings)
    if assumption_warnings:
        extra_limitations.append(
            "Statistical assumption checks flagged concerns; treat any significance "
            "statement as provisional pending validation."
        )
    extra_limitations.extend(warning_notes)
    outcome = _finalize_method_finding(
        plan,
        candidate=candidate,
        findings=findings,
        claim_class="observed",
        extra_limitations=extra_limitations,
        method_gates=[_assumption_method_gate(result)],
        execution_artifacts=execution_artifacts,
        source_artifact_id=stat_artifact.id,
        project_id=project_id,
        session_id=session_id,
        plan_artifact_id=plan_artifact_id,
        record_artifact_id=marker_id,
        llm=llm,
        seeds=seeds,
    )
    return execution_artifacts, outcome


def _run_outcome_prediction(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    source_qcand_id: str,
    marker_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
    role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> tuple[list[Artifact], list[Artifact]]:
    parents = [plan_artifact_id, source_qcand_id]
    target_column = _prediction_target_column(candidate)
    dataset = _dataset_with_column(scoped_datasets, target_column)
    if target_column is None or dataset is None:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(
                    "Outcome prediction needs an explicit target column present in the "
                    "approved dataset; none was resolvable from the Question Card."
                ),
                next_action=(
                    "Name the target column on the Question Card (e.g. 'target column: churn')."
                ),
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    frame = _scoped_frame(scoped_datasets, dataset)
    try:
        guard_baseline_model_params(frame, target_column=target_column)
        card = run_baseline_model(
            frame,
            dataset_id=dataset.record.dataset_id,
            target_column=target_column,
        )
    except ToolGuardError as exc:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(f"The baseline model rejected its parameters before running: {exc}"),
                next_action="Correct the target column and re-plan.",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    except ValueError as exc:
        # Do not publish a finding when every candidate feature fails leakage checks.
        return [], [
            _method_failed_artifact(
                plan,
                reason=_prediction_failure_reason(str(exc)),
                next_action=(
                    "Provide features that are known before the outcome and are not "
                    "proxies for it, then re-plan the prediction."
                ),
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    model_artifact = create_model_card_artifact(
        card, project_id=project_id, session_id=session_id, parents=parents
    )
    execution_artifacts = [model_artifact]
    reducer = _load_reducer("model_findings")
    if reducer is None:
        return execution_artifacts, [
            _method_unavailable_artifact(
                plan,
                what="baseline-model findings reducer",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    findings = reducer(card, model_artifact.id, role_set=(role_sets or {}).get(dataset.record.name))
    outcome = _finalize_method_finding(
        plan,
        candidate=candidate,
        findings=findings,
        claim_class="predictive",
        extra_limitations=list(card.limitations),
        method_gates=[_leakage_method_gate(card)],
        execution_artifacts=execution_artifacts,
        source_artifact_id=model_artifact.id,
        project_id=project_id,
        session_id=session_id,
        plan_artifact_id=plan_artifact_id,
        record_artifact_id=marker_id,
        llm=llm,
        seeds=seeds,
    )
    return execution_artifacts, outcome


def _run_anomaly_detection(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    source_qcand_id: str,
    marker_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
    role_sets: Mapping[str, ColumnRoleSet] | None = None,
) -> tuple[list[Artifact], list[Artifact]]:
    parents = [plan_artifact_id, source_qcand_id]
    anomaly_tools = _load_anomaly_tools()
    reducer = _load_reducer("anomaly_findings")
    if anomaly_tools is None or reducer is None:
        return [], [
            _method_unavailable_artifact(
                plan,
                what="anomaly screening executor",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    screen_anomalies, create_anomaly_artifact = anomaly_tools
    selection = _select_anomaly_column(candidate, scoped_datasets)
    if selection is None:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(
                    "Anomaly screening needs a numeric column in the approved scope; "
                    "none is available."
                ),
                next_action="Point the Question Card at a numeric measure to screen.",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    dataset, column = selection
    frame = _scoped_frame(scoped_datasets, dataset)
    try:
        result = screen_anomalies(
            frame,
            dataset_name=dataset.record.name,
            column=column,
            method="robust_zscore",
            threshold=3.5,
        )
    except (ToolGuardError, ValueError) as exc:
        return [], [
            _method_failed_artifact(
                plan,
                reason=(
                    "Anomaly screening could not run on the approved column "
                    f"({type(exc).__name__})."
                ),
                next_action="Check the selected numeric column before retrying.",
                project_id=project_id,
                session_id=session_id,
                parent_id=plan_artifact_id,
                marker_id=marker_id,
            )
        ]
    anomaly_artifact = create_anomaly_artifact(
        result, project_id=project_id, session_id=session_id, parents=parents
    )
    execution_artifacts = [anomaly_artifact]
    findings = reducer(
        result, anomaly_artifact.id, role_set=(role_sets or {}).get(dataset.record.name)
    )
    outcome = _finalize_method_finding(
        plan,
        candidate=candidate,
        findings=findings,
        claim_class="observed",
        extra_limitations=[],
        method_gates=[
            InvestigationGate(
                name="method",
                status="passed",
                reason=("Robust anomaly screening completed on an approved numeric column."),
            )
        ],
        execution_artifacts=execution_artifacts,
        source_artifact_id=anomaly_artifact.id,
        project_id=project_id,
        session_id=session_id,
        plan_artifact_id=plan_artifact_id,
        record_artifact_id=marker_id,
        llm=llm,
        seeds=seeds,
    )
    return execution_artifacts, outcome


def _finalize_method_finding(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    findings: list[QuestionFinding],
    claim_class: ClaimClass,
    extra_limitations: Sequence[str],
    method_gates: Sequence[InvestigationGate],
    execution_artifacts: Sequence[Artifact],
    source_artifact_id: str,
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    record_artifact_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
) -> list[Artifact]:
    """Shared method-route validation: same claim/evidence gates as SQL findings."""
    parents = [plan_artifact_id, *[artifact.id for artifact in execution_artifacts]]
    if not findings:
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="no_extractable_finding",
                    reason="The method executed but the deterministic reducer produced no finding.",
                    next_action="Inspect the method result or revise the Question Card.",
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parents,
                artifact_id=record_artifact_id,
            )
        ]
    if any(not finding.evidence for finding in findings):
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="missing_evidence_reference",
                    reason="A reducer statement is missing artifact-linked evidence.",
                    next_action="Revise the method so every finding links to result evidence.",
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parents,
                artifact_id=record_artifact_id,
            )
        ]
    if any(implies_causation(finding.text) for finding in findings):
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="unsupported_causal_claim",
                    reason=(
                        "The reducer wording implies causality without an approved causal design."
                    ),
                    next_action=(
                        "Reframe the Question Card as a descriptive or associative analysis."
                    ),
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parents,
                artifact_id=record_artifact_id,
            )
        ]
    resolvable_ids = {artifact.id for artifact in execution_artifacts}
    interpretation = interpret_findings(
        llm,
        question=plan.question,
        findings=findings,
        method_context=plan.method_recipe,
        limitations=[*candidate.risks, *extra_limitations],
        seeds=seeds,
    )
    finding = _method_validated_finding(
        plan,
        candidate=candidate,
        findings=findings,
        claim_class=claim_class,
        extra_limitations=extra_limitations,
        evidence_support=_evidence_support_findings(findings, resolvable_ids),
        source_artifact_id=source_artifact_id,
        source_artifact_session_id=session_id,
        interpretation=interpretation,
    )
    finding_artifact = _finding_artifact(
        finding, project_id=project_id, session_id=session_id, parent_ids=parents
    )
    record = InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "validated"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="validated",
        reason_code="finding_validated",
        reason="Method execution, evidence, and descriptive-claim gates passed.",
        next_action="Keep this finding available for later synthesis and report review.",
        validation_gates=[
            *plan.validation_gates,
            *method_gates,
            InvestigationGate(
                name="execution",
                status="passed",
                reason="The approved method executor completed successfully.",
            ),
            InvestigationGate(
                name="claim",
                status="passed",
                reason="Findings are evidence-backed and free of unsupported causal language.",
            ),
        ],
        source_artifact_ids=[*plan.source_artifact_ids, source_artifact_id],
        evidence=[reference for item in findings for reference in item.evidence],
        finding_artifact_id=finding_artifact.id,
    )
    return [
        finding_artifact,
        _record_artifact(
            record,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parents, finding_artifact.id],
            artifact_id=record_artifact_id,
        ),
    ]


def _method_validated_finding(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    findings: list[QuestionFinding],
    claim_class: ClaimClass,
    extra_limitations: Sequence[str],
    evidence_support: ReliabilityRating,
    source_artifact_id: str,
    source_artifact_session_id: str,
    interpretation: InterpretationResult,
) -> ValidatedFinding:
    analytical_reliability: ReliabilityRating = (
        "medium" if candidate.origin == "template" else "low"
    )
    ready = candidate.feasibility is not None and candidate.feasibility.status == "ready"
    decision_readiness: ReliabilityRating = "medium" if ready else "low"
    report_eligible = analytical_reliability != "low"
    base_caveats = (
        [_PREDICTIVE_BASELINE_CAVEAT] if claim_class == "predictive" else [_OBSERVED_CAVEAT]
    )
    limitations = _unique(
        [
            *candidate.risks,
            *[context.report_limitation for context in plan.quality_context],
            *extra_limitations,
            *base_caveats,
        ]
    )
    has_report_conditions = bool(plan.quality_context or candidate.risks or list(extra_limitations))
    report_readiness = (
        "not_eligible"
        if not report_eligible
        else "eligible_with_limitations"
        if has_report_conditions
        else "eligible"
    )
    report_readiness_reason = (
        "The available method result is not sufficiently reliable for report use."
        if not report_eligible
        else "Report this finding with its recorded data conditions and limitations."
        if report_readiness == "eligible_with_limitations"
        else "The evidence supports reporting this method finding."
    )
    return ValidatedFinding(
        finding_id="finding_" + stable_hash({"investigation": plan.investigation_id}, length=12),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        question=plan.question,
        value_hypothesis=candidate.value_hypothesis,
        decision_action=candidate.business_decision,
        quality_context=plan.quality_context,
        claim_class=claim_class,
        findings=findings,
        evidence_support=evidence_support,
        analytical_reliability=analytical_reliability,
        decision_readiness=decision_readiness,
        limitations=limitations,
        report_eligible=report_eligible,
        report_readiness=report_readiness,
        report_readiness_reason=report_readiness_reason,
        source_artifact_ids=[*plan.source_artifact_ids, source_artifact_id],
        source_artifact_session_ids={
            **plan.source_artifact_session_ids,
            source_artifact_id: source_artifact_session_id,
        },
        interpretation=interpretation.text,
        interpretation_status=interpretation.status,
    )


def _evidence_support_findings(
    findings: Sequence[QuestionFinding],
    resolvable_ids: set[str],
) -> ReliabilityRating:
    resolved = [
        any(
            reference.artifact_id is not None and reference.artifact_id in resolvable_ids
            for reference in finding.evidence
        )
        for finding in findings
    ]
    if resolved and all(resolved):
        return "high"
    if any(resolved):
        return "medium"
    return "low"


def _select_group_comparison_columns(
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
) -> tuple[LoadedDataset, str, str, int] | None:
    """Deterministically pick a grouping/measure pair mirroring _group_comparison_gate."""
    for dataset in scoped_datasets:
        frame = dataset.frame
        referenced = _referenced_columns(candidate, dataset)
        grouping = [
            column
            for column in map(str, frame.columns)
            if infer_column_semantic_type(_column(frame, column)) == "categorical"
            and 2 <= int(_column(frame, column).nunique(dropna=True)) <= 20
        ]
        numeric = [
            column
            for column in map(str, frame.columns)
            if infer_column_semantic_type(_column(frame, column)) == "numeric"
        ]
        if not grouping or not numeric:
            continue
        value_column = _prefer(referenced, numeric)
        preferred_groups = [column for column in grouping if column in referenced]
        ordered_groups = [*preferred_groups, *[c for c in grouping if c not in preferred_groups]]
        rows = int(len(frame))
        with_enough_rows = [
            column
            for column in ordered_groups
            if rows / max(1, int(_column(frame, column).nunique(dropna=True))) >= 5
        ]
        group_column = (with_enough_rows or ordered_groups)[0]
        group_count = int(_column(frame, group_column).nunique(dropna=True))
        return dataset, group_column, value_column, group_count
    return None


def _select_anomaly_column(
    candidate: QuestionCandidate,
    scoped_datasets: Sequence[LoadedDataset],
) -> tuple[LoadedDataset, str] | None:
    """Pick a numeric column: referenced numeric first, else highest variance."""
    for dataset in scoped_datasets:
        frame = dataset.frame
        numeric = [
            column
            for column in map(str, frame.columns)
            if infer_column_semantic_type(_column(frame, column)) == "numeric"
        ]
        if not numeric:
            continue
        referenced = _referenced_columns(candidate, dataset)
        preferred = [column for column in numeric if column in referenced]
        if preferred:
            return dataset, preferred[0]
        variances: dict[str, float] = {}
        for column in numeric:
            series = cast(pd.Series, pd.to_numeric(_column(frame, column), errors="coerce"))
            variance = cast(float, series.var(ddof=1))
            # NaN (constant/empty column) fails self-equality; treat as zero variance.
            variances[column] = variance if variance == variance else 0.0
        # Deterministic: highest variance, ties broken by original column order.
        best = max(numeric, key=lambda column: (variances[column], -numeric.index(column)))
        return dataset, best
    return None


def _prediction_target_column(candidate: QuestionCandidate) -> str | None:
    for requirement in candidate.data_requirements:
        prefix, separator, value = requirement.partition(":")
        if separator and prefix.strip().lower() == "target column" and value.strip():
            return value.strip()
    return None


def _dataset_with_column(
    scoped_datasets: Sequence[LoadedDataset],
    column: str | None,
) -> LoadedDataset | None:
    if column is None:
        return None
    for dataset in scoped_datasets:
        if column in {str(name) for name in dataset.frame.columns}:
            return dataset
    return None


def _referenced_columns(candidate: QuestionCandidate, dataset: LoadedDataset) -> list[str]:
    available = {str(name) for name in dataset.frame.columns}
    # Fall back to dataset IDs for compatibility with older persisted candidates.
    for key in (dataset.record.name, dataset.record.dataset_id):
        for name, columns in candidate.referenced_columns.items():
            if name == key:
                return [column for column in columns if column in available]
    return []


def _scoped_frame(
    scoped_datasets: Sequence[LoadedDataset],
    dataset: LoadedDataset,
) -> pd.DataFrame:
    """Materialize the analysis frame through the read-only SQL path."""
    catalog = build_catalog(scoped_datasets)
    relation = catalog.relations[dataset.record.name]
    return catalog.engine.execute_select(f"SELECT * FROM {relation}")


def _assumption_warnings(result: StatTestResult) -> list[str]:
    return [check.message for check in result.assumptions if check.status == "warn"]


def _assumption_method_gate(result: StatTestResult) -> InvestigationGate:
    warned = _assumption_warnings(result)
    if warned:
        return InvestigationGate(
            name="method",
            status="warning",
            reason="Statistical assumption checks flagged concerns: " + " ".join(warned),
        )
    return InvestigationGate(
        name="method",
        status="passed",
        reason="Statistical assumption checks did not flag normality or variance problems.",
    )


def _leakage_method_gate(card: ModelCard) -> InvestigationGate:
    excluded = [check for check in card.leakage_checks if check.action == "excluded"]
    if excluded:
        columns = ", ".join(check.column or check.code for check in excluded)
        return InvestigationGate(
            name="method",
            status="warning",
            reason=f"Leakage screening excluded suspected target proxies: {columns}.",
        )
    return InvestigationGate(
        name="method",
        status="passed",
        reason=(
            "Leakage screening found no identical or near-perfect target proxy "
            "among the features used."
        ),
    )


def _prediction_failure_reason(error: str) -> str:
    if "leakage" in error.lower():
        return (
            "The leakage guard removed every candidate feature: each column was either "
            "an identifier or a near-perfect proxy of the target, so a baseline trained "
            "on them would only memorize the answer rather than predict it. No honest "
            "baseline could be produced."
        )
    return (
        "The baseline model could not be trained on the approved data "
        f"({error}). A predictive finding is not available."
    )


def _method_failed_artifact(
    plan: InvestigationPlan,
    *,
    reason: str,
    next_action: str,
    project_id: str,
    session_id: str,
    parent_id: str,
    marker_id: str,
) -> Artifact:
    record = InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "method_failed"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="failed",
        reason_code="method_gate_failed",
        reason=reason,
        next_action=next_action,
        validation_gates=[
            *plan.validation_gates,
            InvestigationGate(name="method", status="failed", reason=reason),
        ],
        source_artifact_ids=plan.source_artifact_ids,
    )
    return _record_artifact(
        record,
        project_id=project_id,
        session_id=session_id,
        parent_ids=[parent_id],
        artifact_id=marker_id,
    )


def _method_unavailable_artifact(
    plan: InvestigationPlan,
    *,
    what: str,
    project_id: str,
    session_id: str,
    parent_id: str,
    marker_id: str,
) -> Artifact:
    reason = (
        f"The {what} is not available in this build, so the approved method could not be completed."
    )
    record = InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "method_unavailable"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="failed",
        reason_code="method_executor_unavailable",
        reason=reason,
        next_action="Enable the method executor and re-run the investigation.",
        validation_gates=[
            *plan.validation_gates,
            InvestigationGate(name="method", status="failed", reason=reason),
        ],
        source_artifact_ids=plan.source_artifact_ids,
    )
    return _record_artifact(
        record,
        project_id=project_id,
        session_id=session_id,
        parent_ids=[parent_id],
        artifact_id=marker_id,
    )


def _load_reducer(name: str) -> Any:
    """Import a DI3-A method reducer lazily; None when not yet on disk."""
    try:
        from eda_platform.tools import method_findings
    except ImportError:
        return None
    return getattr(method_findings, name, None)


def _load_anomaly_tools() -> tuple[Any, Any] | None:
    """Import the DI3-A anomaly tool lazily; None when not yet on disk."""
    try:
        from eda_platform.tools.anomaly import create_anomaly_artifact, screen_anomalies
    except ImportError:
        return None
    return screen_anomalies, create_anomaly_artifact


def _prefer(preferred: Sequence[str], options: Sequence[str]) -> str:
    for column in preferred:
        if column in options:
            return column
    return options[0]


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _build_plan(
    candidate: QuestionCandidate,
    *,
    source_session_id: str,
    session_id: str,
    available_datasets: set[str],
    profiles_by_name: dict[str, DatasetProfile],
    quality_context_by_id: dict[str, QualityContextSet],
    source_qcand_id: str,
    deep: bool = False,
) -> InvestigationPlan:
    gates: list[InvestigationGate] = []
    unknown_datasets = sorted(set(candidate.target_datasets) - available_datasets)
    if unknown_datasets:
        gates.append(
            InvestigationGate(
                name="scope",
                status="failed",
                reason="Target datasets are not available: " + ", ".join(unknown_datasets),
            )
        )
    else:
        gates.append(
            InvestigationGate(
                name="scope",
                status="passed",
                reason="The plan is limited to datasets named by the approved Question Card.",
            )
        )
    if candidate.required_relations:
        gates.append(
            InvestigationGate(
                name="scope",
                status="failed",
                reason=(
                    "This plan needs a relationship reference. The current orchestrator does "
                    "not execute joins without a dedicated reviewed join plan."
                ),
            )
        )

    method_selection = select_investigation_method(candidate, profiles_by_name)
    resolved_feasibility = (
        candidate.feasibility.status
        if candidate.feasibility is not None
        else method_selection.feasibility.status
    )
    if resolved_feasibility in {"needs_data", "unsuitable"}:
        gates.append(
            InvestigationGate(
                name="feasibility",
                status="failed",
                reason="The Question Card identifies missing prerequisites for this analysis.",
            )
        )
    elif resolved_feasibility == "constrained":
        gates.append(
            InvestigationGate(
                name="feasibility",
                status="warning",
                reason="The Question Card has constraints that must remain visible in the result.",
            )
        )
    else:
        gates.append(
            InvestigationGate(
                name="feasibility",
                status="passed",
                reason="Available profile evidence supports an initial investigation.",
            )
        )
    gates.extend(method_selection.validation_gates)

    quality_context = _resolve_quality_context(candidate, quality_context_by_id)
    blocked = any(gate.status == "failed" for gate in gates)
    status_reason = (
        "A scope, feasibility, or method prerequisite failed; do not execute until "
        "the Card is revised."
        if blocked
        else "Awaiting explicit user approval before controlled execution."
    )
    assumptions = _unique(
        [
            *(candidate.feasibility.reasons if candidate.feasibility is not None else []),
            *candidate.risks,
            "Findings are descriptive unless an approved causal design says otherwise.",
        ]
    )
    allowed_tools = list(method_selection.allowed_tools)
    if deep:
        # The deep marker rides on ``allowed_tools`` so the persisted plan (and its
        # fingerprint-bound approval) explicitly authorizes the follow-up probes.
        allowed_tools = _unique([*allowed_tools, _DEEP_MARKER_TOOL])
        assumptions = _unique([*assumptions, _DEEP_ASSUMPTION])
    return InvestigationPlan(
        investigation_id="inv_"
        + stable_hash(
            {
                "session_id": session_id,
                "question_id": candidate.question_id,
                "card_version": candidate.card_version,
            },
            length=12,
        ),
        source_session_id=source_session_id,
        question_id=candidate.question_id,
        card_version=candidate.card_version,
        candidate_fingerprint=_candidate_fingerprint(candidate),
        question=candidate.question_en,
        target_datasets=candidate.target_datasets,
        source_artifact_ids=_unique([source_qcand_id, *candidate.source_artifact_ids]),
        source_artifact_session_ids={
            artifact_id: source_session_id
            for artifact_id in _unique(
                [source_qcand_id, *candidate.source_artifact_ids]
            )
        },
        allowed_relationship_references=candidate.required_relations,
        method_family=method_selection.method_family,
        method_recipe=method_selection.method_recipe,
        allowed_tools=allowed_tools,
        method_requirements=method_selection.method_requirements,
        execution_ready=method_selection.execution_ready and not blocked,
        quality_context=quality_context,
        quality_context_artifact_ids=candidate.quality_context_artifact_ids,
        assumptions=assumptions,
        validation_gates=gates,
        feasibility=resolved_feasibility,
        status="needs_data" if blocked else "planned",
        status_reason=status_reason,
    )


def _validate_execution(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    execution_artifacts: Sequence[Artifact],
    project_id: str,
    session_id: str,
    parent_ids: list[str],
    record_artifact_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
) -> list[Artifact]:
    qexec_artifact = next(
        (
            artifact
            for artifact in execution_artifacts
            if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
        ),
        None,
    )
    if qexec_artifact is None:
        return [
            _record_artifact(
                _failed_execution_record(
                    plan,
                    "Execution produced no QuestionExecutionResult artifact.",
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                artifact_id=record_artifact_id,
            )
        ]
    qexec = QuestionExecutionResult.model_validate(qexec_artifact.payload)
    if qexec.status != "succeeded":
        return [
            _record_artifact(
                _failed_execution_record(plan, qexec.error or "Question execution failed."),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                artifact_id=record_artifact_id,
            )
        ]
    if not qexec.findings:
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="no_extractable_finding",
                    reason="The query completed but did not yield an evidence-backed finding.",
                    next_action="Inspect the result shape or revise the Question Card.",
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                artifact_id=record_artifact_id,
            )
        ]
    if any(not finding.evidence for finding in qexec.findings):
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="missing_evidence_reference",
                    reason="At least one extracted statement has no artifact-linked evidence.",
                    next_action="Revise the method so every finding links to result evidence.",
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                artifact_id=record_artifact_id,
            )
        ]
    if any(contains_causal_phrase(finding.text) for finding in qexec.findings):
        return [
            _record_artifact(
                _inconclusive_record(
                    plan,
                    reason_code="unsupported_causal_claim",
                    reason=(
                        "The extracted wording implies causality without an approved causal design."
                    ),
                    next_action=(
                        "Reframe the Question Card as a descriptive or associative analysis."
                    ),
                ),
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                artifact_id=record_artifact_id,
            )
        ]

    resolvable_ids = {artifact.id for artifact in execution_artifacts}
    interpretation = interpret_findings(
        llm,
        question=plan.question,
        findings=qexec.findings,
        method_context=qexec.plan_summary,
        limitations=candidate.risks,
        seeds=seeds,
    )
    finding = _validated_finding(
        plan,
        candidate=candidate,
        qexec=qexec,
        qexec_artifact=qexec_artifact,
        evidence_support=_evidence_support(qexec, resolvable_ids),
        interpretation=interpretation,
    )
    finding_artifact = _finding_artifact(
        finding,
        project_id=project_id,
        session_id=session_id,
        parent_ids=parent_ids,
    )
    record = InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "validated"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="validated",
        reason_code="finding_validated",
        reason="Execution, evidence, and descriptive-claim gates passed.",
        next_action="Keep this finding available for later synthesis and report review.",
        validation_gates=[
            *plan.validation_gates,
            InvestigationGate(
                name="execution",
                status="passed",
                reason="The approved read-only execution completed successfully.",
            ),
            InvestigationGate(
                name="claim",
                status="passed",
                reason="Findings are evidence-backed and remain descriptive.",
            ),
        ],
        source_artifact_ids=[*plan.source_artifact_ids, qexec_artifact.id],
        evidence=[reference for item in qexec.findings for reference in item.evidence],
        finding_artifact_id=finding_artifact.id,
    )
    return [
        finding_artifact,
        _record_artifact(
            record,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parent_ids, finding_artifact.id],
            artifact_id=record_artifact_id,
        ),
    ]


def _validated_finding(
    plan: InvestigationPlan,
    *,
    candidate: QuestionCandidate,
    qexec: QuestionExecutionResult,
    qexec_artifact: Artifact,
    evidence_support: ReliabilityRating,
    interpretation: InterpretationResult,
) -> ValidatedFinding:
    # Derive reliability, readiness, and report eligibility consistently.
    analytical_reliability: ReliabilityRating = (
        "medium" if candidate.origin == "template" else "low"
    )
    ready = candidate.feasibility is not None and candidate.feasibility.status == "ready"
    decision_readiness: ReliabilityRating = "medium" if ready else "low"
    report_eligible = analytical_reliability != "low"
    has_report_conditions = bool(plan.quality_context or candidate.risks)
    report_readiness = (
        "not_eligible"
        if not report_eligible
        else "eligible_with_limitations"
        if has_report_conditions
        else "eligible"
    )
    report_readiness_reason = (
        "The available execution is not sufficiently reliable for report use."
        if not report_eligible
        else "Report this finding with its recorded data conditions and limitations."
        if report_readiness == "eligible_with_limitations"
        else "The evidence supports reporting this descriptive finding."
    )
    return ValidatedFinding(
        finding_id="finding_" + stable_hash({"investigation": plan.investigation_id}, length=12),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        question=plan.question,
        value_hypothesis=candidate.value_hypothesis,
        decision_action=candidate.business_decision,
        quality_context=plan.quality_context,
        claim_class="observed",
        findings=qexec.findings,
        evidence_support=evidence_support,
        analytical_reliability=analytical_reliability,
        decision_readiness=decision_readiness,
        limitations=_unique(
            [
                *candidate.risks,
                *[context.report_limitation for context in plan.quality_context],
                "This is an observed result in the current data, not a causal claim.",
            ]
        ),
        report_eligible=report_eligible,
        report_readiness=report_readiness,
        report_readiness_reason=report_readiness_reason,
        source_artifact_ids=[*plan.source_artifact_ids, qexec_artifact.id],
        source_artifact_session_ids={
            **plan.source_artifact_session_ids,
            qexec_artifact.id: qexec_artifact.session_id,
        },
        interpretation=interpretation.text,
        interpretation_status=interpretation.status,
    )


def _evidence_support(
    qexec: QuestionExecutionResult,
    resolvable_ids: set[str],
) -> ReliabilityRating:
    resolved = [
        any(
            reference.artifact_id is not None and reference.artifact_id in resolvable_ids
            for reference in finding.evidence
        )
        for finding in qexec.findings
    ]
    if resolved and all(resolved):
        return "high"
    if any(resolved):
        return "medium"
    return "low"


def _plan_requests_deep(plan: InvestigationPlan) -> bool:
    return _DEEP_MARKER_TOOL in plan.allowed_tools


def _initialize_loop_journal(
    store: ArtifactStore,
    *,
    project_id: str,
    session_id: str,
    plan: InvestigationPlan,
) -> JsonlLoopJournal:
    """Create or validate the stable durable loop identity in the run workspace."""
    journal = JsonlLoopJournal(
        store.session_dir(project_id, session_id)
        / "investigations"
        / plan.investigation_id
        / "loop.journal.jsonl"
    )
    journal.initialize(
        investigation_id=plan.investigation_id,
        source_session_id=plan.source_session_id,
        question_id=plan.question_id,
        plan_fingerprint=_plan_fingerprint(plan),
        policy_fingerprint=stable_hash(
            {
                "policy": "bounded-investigation-v1",
                "max_steps": _DEEP_MAX_STEPS,
                "llm_call_cap": _DEEP_LLM_CALL_CAP,
                "planner_task": "di5_bounded_probe",
            },
            length=32,
        ),
        code_fingerprint=stable_hash(
            {
                "component": "investigation_loop",
                "package_version": __version__,
                "journal_integration": 2,
                "loop_contract_revision": "bounded-probe-journal-v2-2026-07-24",
                "prompt_revision": "di5-bounded-probe-v2",
            },
            length=32,
        ),
        max_steps=_DEEP_MAX_STEPS,
        llm_call_cap=_DEEP_LLM_CALL_CAP,
    )
    return journal


def _primary_checkpoint_path(journal: JsonlLoopJournal) -> Path:
    return journal.path.parent / "primary-artifacts.json"


def _primary_started_path(journal: JsonlLoopJournal) -> Path:
    return journal.path.parent / "primary-started.json"


def _mark_primary_execution_started(journal: JsonlLoopJournal) -> None:
    """Fail closed instead of replaying a primary stage with unknown completion."""
    path = _primary_started_path(journal)
    if path.exists():
        raise LoopTransitionError(
            "Primary investigation completion is uncertain; refusing to replay paid work."
        )
    state = journal.rebuild()
    if state is None:
        raise LoopTransitionError("Primary execution requires an initialized journal.")
    _write_json_atomic(
        journal,
        path,
        {
            "schema_version": 1,
            "investigation_id": state.investigation_id,
            "plan_fingerprint": state.plan_fingerprint,
        },
    )


def _load_primary_checkpoint(
    store: ArtifactStore,
    journal: JsonlLoopJournal,
    *,
    project_id: str,
    session_id: str,
) -> tuple[list[Artifact], list[Artifact]] | None:
    """Load the durable primary stage or fail closed on a malformed checkpoint."""
    path = _primary_checkpoint_path(journal)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = journal.rebuild()
        if state is None:
            raise ValueError("journal is empty")
        execution_entries = payload["execution_artifacts"]
        outcome_entries = payload["outcome_artifacts"]
        if (
            payload.get("schema_version") != 2
            or payload.get("project_id") != project_id
            or payload.get("session_id") != session_id
            or payload.get("investigation_id") != state.investigation_id
            or payload.get("plan_fingerprint") != state.plan_fingerprint
            or not isinstance(execution_entries, list)
            or not isinstance(outcome_entries, list)
        ):
            raise ValueError("invalid primary checkpoint envelope")
        execution = [
            _load_checkpoint_artifact(store, item, project_id=project_id, session_id=session_id)
            for item in execution_entries
        ]
        outcomes = [
            _load_checkpoint_artifact(store, item, project_id=project_id, session_id=session_id)
            for item in outcome_entries
        ]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Deep investigation primary checkpoint is invalid: {path}."
        ) from exc
    return execution, outcomes


def _write_primary_checkpoint(
    journal: JsonlLoopJournal,
    execution_artifacts: Sequence[Artifact],
    outcome_artifacts: Sequence[Artifact],
) -> None:
    """Commit primary artifact identities before the resumable loop starts."""
    state = journal.rebuild()
    if state is None:
        raise LoopTransitionError("Primary checkpoint requires an initialized journal.")
    combined = [*execution_artifacts, *outcome_artifacts]
    if not combined:
        raise ValueError("Primary checkpoint cannot be empty.")
    project_ids = {artifact.project_id for artifact in combined}
    session_ids = {artifact.session_id for artifact in combined}
    if len(project_ids) != 1 or len(session_ids) != 1:
        raise ValueError("Primary checkpoint artifacts must belong to one project and run.")
    path = _primary_checkpoint_path(journal)
    _write_json_atomic(
        journal,
        path,
        {
            "schema_version": 2,
            "project_id": combined[0].project_id,
            "session_id": combined[0].session_id,
            "investigation_id": state.investigation_id,
            "plan_fingerprint": state.plan_fingerprint,
            "execution_artifacts": [
                _checkpoint_artifact_entry(item) for item in execution_artifacts
            ],
            "outcome_artifacts": [
                _checkpoint_artifact_entry(item) for item in outcome_artifacts
            ],
        },
    )


def _checkpoint_artifact_entry(artifact: Artifact) -> dict[str, str]:
    return {
        "id": artifact.id,
        "type": artifact.type.value,
        "digest": stable_hash(artifact.model_dump(mode="json"), length=64),
    }


def _load_checkpoint_artifact(
    store: ArtifactStore,
    entry: object,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    if not isinstance(entry, dict) or set(entry) != {"id", "type", "digest"}:
        raise ValueError("invalid primary checkpoint artifact entry")
    artifact_id = entry.get("id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("invalid primary checkpoint artifact id")
    artifact = store.get_artifact(
        artifact_id,
        project_id=project_id,
        session_id=session_id,
    )
    if (
        artifact.project_id != project_id
        or artifact.session_id != session_id
        or artifact.type.value != entry.get("type")
        or stable_hash(artifact.model_dump(mode="json"), length=64) != entry.get("digest")
    ):
        raise ValueError(f"Primary checkpoint artifact {artifact_id!r} failed binding checks.")
    return artifact


def _write_json_atomic(
    journal: JsonlLoopJournal,
    path: Path,
    payload: dict[str, object],
) -> None:
    session_dir = journal.path.parents[2]
    project_dir = session_dir.parent.parent
    workspace = project_dir.parent.parent
    ArtifactStore(workspace, init_db=False).write_session_text(
        project_dir.name,
        session_dir.name,
        path.relative_to(session_dir).as_posix(),
        json.dumps(payload, sort_keys=True),
    )


def _run_deep_investigation(
    plan: InvestigationPlan,
    *,
    scoped_datasets: Sequence[LoadedDataset],
    execution_artifacts: list[Artifact],
    outcome_artifacts: list[Artifact],
    project_id: str,
    session_id: str,
    plan_artifact_id: str,
    marker_id: str,
    llm: LLMClient,
    seeds: SemanticSeeds | None = None,
    trace_sink: Callable[[TraceEvent], None] | None = None,
    journal: JsonlLoopJournal | None = None,
) -> tuple[list[Artifact], list[Artifact]]:
    """Run the L2 bounded loop over a validated primary finding and merge its probes."""
    finding_artifact = next(
        (item for item in outcome_artifacts if item.type is ArtifactType.VALIDATED_FINDING),
        None,
    )
    if finding_artifact is None:
        return execution_artifacts, outcome_artifacts
    primary_finding = ValidatedFinding.model_validate(finding_artifact.payload)

    # The catalog is built ONLY from the plan's already scope-limited datasets, so
    # Its engine cannot reach a table outside the approved scope.
    catalog = build_catalog(scoped_datasets)
    result = run_bounded_loop(
        llm,
        plan=plan,
        primary_findings=list(primary_finding.findings),
        query_engine=catalog.engine,
        catalog=catalog,
        max_steps=_DEEP_MAX_STEPS,
        llm_call_cap=_DEEP_LLM_CALL_CAP,
        trace_sink=trace_sink,
        journal=journal,
    )
    primary_result_id = execution_artifacts[-1].id if execution_artifacts else plan_artifact_id
    deep_artifact = _deep_investigation_artifact(
        result,
        project_id=project_id,
        session_id=session_id,
        parent_ids=[plan_artifact_id, primary_result_id],
    )

    probe_findings = _admissible_probe_findings(result, deep_artifact_id=deep_artifact.id)
    if not probe_findings:
        return [*execution_artifacts, deep_artifact], outcome_artifacts

    merged = primary_finding.model_copy(
        update={
            "findings": [*primary_finding.findings, *probe_findings],
            "source_artifact_ids": _unique(
                [*primary_finding.source_artifact_ids, deep_artifact.id]
            ),
            "source_artifact_session_ids": {
                **primary_finding.source_artifact_session_ids,
                deep_artifact.id: deep_artifact.session_id,
            },
        }
    )
    parents_without_old = [
        parent for parent in finding_artifact.parents if parent != finding_artifact.id
    ]
    merged_finding_artifact = _finding_artifact(
        merged,
        project_id=project_id,
        session_id=session_id,
        parent_ids=_unique([*parents_without_old, deep_artifact.id]),
    )
    new_outcomes = _rebuild_outcomes_with_deep(
        outcome_artifacts,
        old_finding_artifact=finding_artifact,
        new_finding_artifact=merged_finding_artifact,
        merged=merged,
        deep_artifact_id=deep_artifact.id,
        marker_id=marker_id,
        project_id=project_id,
        session_id=session_id,
    )
    return [*execution_artifacts, deep_artifact], new_outcomes


def _probe_finding_numbers_supported(finding: QuestionFinding) -> bool:
    """Every number in a probe finding's text must match its own evidence values."""
    allowed: list[tuple[float, bool]] = []
    for reference in finding.evidence:
        value = reference.value
        if isinstance(value, int | float) and not isinstance(value, bool):
            allowed.append((float(value), reference.unit == "percent"))
    for number, is_percent in extract_numbers(finding.text):
        supported = False
        for target, target_is_percent in allowed:
            if is_percent != target_is_percent:
                continue
            tolerance = max(abs(target) * 0.01, 0.01)
            if abs(number - target) <= tolerance:
                supported = True
                break
        if not supported:
            return False
    return True


def _admissible_probe_findings(
    result: DeepInvestigationResult,
    *,
    deep_artifact_id: str,
) -> list[QuestionFinding]:
    """Select successful, evidence-backed probe findings that pass the claim-language gate."""
    admissible: list[QuestionFinding] = []
    for step in result.steps:
        if step.action != "probe" or step.status != "succeeded":
            continue
        for finding in step.findings:
            if not finding.evidence:
                continue
            if implies_causation(finding.text):
                continue
            # Every LLM-influenced number must match this finding's evidence.
            if not _probe_finding_numbers_supported(finding):
                continue
            rewritten = [
                reference.model_copy(
                    update={
                        "artifact_id": deep_artifact_id,
                        "locator": f"step{step.step_index}.{reference.locator}",
                    }
                )
                for reference in finding.evidence
            ]
            admissible.append(finding.model_copy(update={"evidence": rewritten}))
    return admissible


def _rebuild_outcomes_with_deep(
    outcome_artifacts: Sequence[Artifact],
    *,
    old_finding_artifact: Artifact,
    new_finding_artifact: Artifact,
    merged: ValidatedFinding,
    deep_artifact_id: str,
    marker_id: str,
    project_id: str,
    session_id: str,
) -> list[Artifact]:
    rebuilt: list[Artifact] = []
    for artifact in outcome_artifacts:
        if artifact.id == old_finding_artifact.id:
            rebuilt.append(new_finding_artifact)
            continue
        if artifact.type is ArtifactType.INVESTIGATION_RECORD and artifact.id == marker_id:
            record = InvestigationRecord.model_validate(artifact.payload)
            updated = record.model_copy(
                update={
                    "source_artifact_ids": _unique([*record.source_artifact_ids, deep_artifact_id]),
                    "evidence": [
                        reference for item in merged.findings for reference in item.evidence
                    ],
                    "finding_artifact_id": new_finding_artifact.id,
                }
            )
            parents = [parent for parent in artifact.parents if parent != old_finding_artifact.id]
            rebuilt.append(
                _record_artifact(
                    updated,
                    project_id=project_id,
                    session_id=session_id,
                    parent_ids=_unique([*parents, deep_artifact_id, new_finding_artifact.id]),
                    artifact_id=marker_id,
                )
            )
            continue
        rebuilt.append(artifact)
    return rebuilt


def _deep_investigation_artifact(
    result: DeepInvestigationResult,
    *,
    project_id: str,
    session_id: str,
    parent_ids: list[str],
) -> Artifact:
    payload = result.model_dump(mode="json")
    evidence = [
        reference
        for step in result.steps
        for finding in step.findings
        for reference in finding.evidence
    ]
    return Artifact(
        id=make_artifact_id(
            "deepinv",
            {"session_id": session_id, "investigation_id": result.investigation_id, "deep": payload},
        ),
        type=ArtifactType.DEEP_INVESTIGATION_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=parent_ids,
        payload=payload,
        evidence=evidence,
        plain_language=(
            "A bounded, LLM-planned and deterministically executed follow-up "
            "investigation transcript with a typed exit reason."
        ),
    )


def _scoped_datasets(
    datasets: Sequence[LoadedDataset],
    plan: InvestigationPlan,
) -> list[LoadedDataset]:
    allowed = set(plan.target_datasets)
    return [
        dataset
        for dataset in datasets
        if dataset.record.name in allowed or dataset.record.dataset_id in allowed
    ]


def _resolve_quality_context(
    candidate: QuestionCandidate,
    quality_context_by_id: dict[str, QualityContextSet],
) -> list[QualityContext]:
    contexts: dict[str, QualityContext] = {}
    for artifact_id in candidate.quality_context_artifact_ids:
        context_set = quality_context_by_id.get(artifact_id)
        if context_set is None:
            continue
        for context in context_set.contexts:
            contexts.setdefault(context.context_id, context)
    return list(contexts.values())


def _plan_artifact(plan: InvestigationPlan, *, project_id: str, session_id: str) -> Artifact:
    payload = plan.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("invplan", {"session_id": session_id, "plan": payload}),
        type=ArtifactType.INVESTIGATION_PLAN,
        project_id=project_id,
        session_id=session_id,
        parents=plan.source_artifact_ids,
        payload=payload,
        plain_language=(
            "A user-reviewable, scope-limited investigation plan. It cannot execute until "
            "the user approves it."
        ),
    )


def _approval_artifact(
    approval: InvestigationApproval,
    *,
    project_id: str,
    session_id: str,
    parent_ids: list[str],
) -> Artifact:
    payload = approval.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("invappr", {"session_id": session_id, "approval": payload}),
        type=ArtifactType.INVESTIGATION_APPROVAL,
        project_id=project_id,
        session_id=session_id,
        parents=parent_ids,
        payload=payload,
        plain_language=(
            "A persisted user decision bound to the exact plan content; execution requires a "
            "matching approval."
        ),
    )


def _plan_blocked_record(plan: InvestigationPlan) -> InvestigationRecord:
    failed_gates = [gate for gate in plan.validation_gates if gate.status == "failed"]
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "needs_data"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="needs_data",
        reason_code="plan_gate_failed",
        reason=" ".join(gate.reason for gate in failed_gates),
        next_action="Revise the Question Card or add the missing reviewed prerequisite.",
        validation_gates=plan.validation_gates,
        source_artifact_ids=plan.source_artifact_ids,
    )


def _unknown_question_record(
    *,
    question_id: str,
    project_id: str,
    session_id: str,
    parent_ids: list[str],
) -> Artifact:
    record = InvestigationRecord(
        record_id="irec_" + stable_hash({"session_id": session_id, "question_id": question_id}),
        investigation_id="unknown_" + question_id,
        question_id=question_id,
        status="failed",
        reason_code="question_not_found",
        reason="The selected Question Card does not exist in the source run.",
        next_action="Choose a Question Card from the current run.",
        source_artifact_ids=parent_ids,
    )
    return _record_artifact(record, project_id=project_id, session_id=session_id, parent_ids=parent_ids)


def _executing_marker_record(plan: InvestigationPlan) -> InvestigationRecord:
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "executing"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="failed",
        reason_code="executing",
        reason="A controlled execution is in progress for this investigation.",
        next_action="Wait for the current execution to finalize before re-running.",
        validation_gates=plan.validation_gates,
        source_artifact_ids=plan.source_artifact_ids,
    )


def _stale_candidate_record(plan: InvestigationPlan) -> InvestigationRecord:
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "stale_candidate"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="rejected",
        reason_code="stale_candidate",
        reason="The live Question Card no longer matches the approved plan fingerprint.",
        next_action="Rebuild and re-approve the plan from the current Question Card.",
        validation_gates=plan.validation_gates,
        source_artifact_ids=plan.source_artifact_ids,
    )


def _scope_failed_record(plan: InvestigationPlan, out_of_scope: list[str]) -> InvestigationRecord:
    detail = (
        "The Question Card references datasets outside the approved scope: "
        + ", ".join(out_of_scope)
        if out_of_scope
        else "No approved dataset is available for this plan's scope."
    )
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "scope_failed"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="rejected",
        reason_code="scope_violation",
        reason=detail,
        next_action="Restrict the Question Card to its approved datasets and re-plan.",
        validation_gates=[
            *plan.validation_gates,
            InvestigationGate(name="scope", status="failed", reason=detail),
        ],
        source_artifact_ids=plan.source_artifact_ids,
    )


def _failed_execution_record(plan: InvestigationPlan, reason: str) -> InvestigationRecord:
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "failed"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="failed",
        reason_code="execution_failed",
        reason=reason,
        next_action="Check the plan prerequisites or revise the Question Card before retrying.",
        validation_gates=[
            *plan.validation_gates,
            InvestigationGate(name="execution", status="failed", reason=reason),
        ],
        source_artifact_ids=plan.source_artifact_ids,
    )


def _missing_candidate_record(plan: InvestigationPlan) -> InvestigationRecord:
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, "missing_candidate"),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="failed",
        reason_code="question_card_missing",
        reason="The source Question Card is no longer available for this plan.",
        next_action="Create a new plan from the current Question Card portfolio.",
        validation_gates=plan.validation_gates,
        source_artifact_ids=plan.source_artifact_ids,
    )


def _inconclusive_record(
    plan: InvestigationPlan,
    *,
    reason_code: str,
    reason: str,
    next_action: str,
) -> InvestigationRecord:
    return InvestigationRecord(
        record_id=_record_id(plan.investigation_id, reason_code),
        investigation_id=plan.investigation_id,
        question_id=plan.question_id,
        status="inconclusive",
        reason_code=reason_code,
        reason=reason,
        next_action=next_action,
        validation_gates=plan.validation_gates,
        source_artifact_ids=plan.source_artifact_ids,
    )


def _finding_artifact(
    finding: ValidatedFinding,
    *,
    project_id: str,
    session_id: str,
    parent_ids: list[str],
) -> Artifact:
    payload = finding.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("finding", {"session_id": session_id, "finding": payload}),
        type=ArtifactType.VALIDATED_FINDING,
        project_id=project_id,
        session_id=session_id,
        parents=parent_ids,
        payload=payload,
        evidence=[reference for item in finding.findings for reference in item.evidence],
        plain_language=(
            "An evidence-backed descriptive finding. It has not yet been selected for a report."
        ),
    )


def _record_artifact(
    record: InvestigationRecord,
    *,
    project_id: str,
    session_id: str,
    parent_ids: list[str],
    artifact_id: str | None = None,
) -> Artifact:
    payload = record.model_dump(mode="json")
    return Artifact(
        id=artifact_id or make_artifact_id("irecord", {"session_id": session_id, "record": payload}),
        type=ArtifactType.INVESTIGATION_RECORD,
        project_id=project_id,
        session_id=session_id,
        parents=parent_ids,
        payload=payload,
        evidence=record.evidence,
        plain_language=(
            "A traceable investigation outcome, including blocked and inconclusive work."
        ),
    )


def _persist(store: ArtifactStore, artifact: Artifact) -> Artifact:
    store.save_artifact(artifact)
    return artifact


def _records_by_investigation(
    artifacts: Sequence[Artifact],
) -> dict[str, list[InvestigationRecord]]:
    records: dict[str, list[InvestigationRecord]] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.INVESTIGATION_RECORD:
            continue
        record = InvestigationRecord.model_validate(artifact.payload)
        records.setdefault(record.investigation_id, []).append(record)
    return records


def _approvals_by_investigation(
    artifacts: Sequence[Artifact],
) -> dict[str, list[InvestigationApproval]]:
    approvals: dict[str, list[InvestigationApproval]] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.INVESTIGATION_APPROVAL:
            continue
        approval = InvestigationApproval.model_validate(artifact.payload)
        approvals.setdefault(approval.investigation_id, []).append(approval)
    return approvals


def _plan_execution_blockers(
    plan: InvestigationPlan,
    records_by_investigation: dict[str, list[InvestigationRecord]],
) -> list[str]:
    blockers: list[str] = []
    if plan.status != "planned":
        blockers.append(f"{plan.question_id} is {plan.status}")
    if not plan.execution_ready:
        blockers.append(f"{plan.question_id} is not execution-ready")
    existing_records = records_by_investigation.get(plan.investigation_id, [])
    has_terminal_record = any(record.reason_code != "executing" for record in existing_records)
    # A deep investigation's durable journal fences loop execution, so an
    # executing marker left by a process crash is resumable. Non-deep paths
    # retain the prior fail-closed re-entry behavior.
    if has_terminal_record or (existing_records and not _plan_requests_deep(plan)):
        blockers.append(f"{plan.question_id} already has an outcome")
    return blockers


def _approval_blockers(
    selected: Sequence[tuple[Artifact, InvestigationPlan]],
    approvals_by_investigation: dict[str, list[InvestigationApproval]],
) -> list[str]:
    blockers: list[str] = []
    for _, plan in selected:
        approvals = approvals_by_investigation.get(plan.investigation_id, [])
        approved = [approval for approval in approvals if approval.decision == "approved"]
        if not approved:
            blockers.append(f"{plan.question_id} has no persisted approval")
            continue
        fingerprint = _plan_fingerprint(plan)
        if not any(approval.plan_fingerprint == fingerprint for approval in approved):
            blockers.append(f"{plan.question_id} approval does not match the current plan")
    return blockers


def _has_execution_outcome(records: Sequence[InvestigationRecord]) -> bool:
    return any(record.status != "needs_data" for record in records)


def _source_candidate_set(artifacts: Sequence[Artifact]) -> Artifact:
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET:
            return artifact
    raise ValueError("Source session does not contain a QuestionCandidateSet artifact.")


def _available_dataset_names(artifacts: Sequence[Artifact]) -> set[str]:
    names: set[str] = set()
    for artifact in artifacts:
        if artifact.type is ArtifactType.DATASET_PROFILE:
            name = artifact.payload.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name)
    return names


def _profiles_by_name(artifacts: Sequence[Artifact]) -> dict[str, DatasetProfile]:
    return {
        profile.name: profile
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
        for profile in [DatasetProfile.model_validate(artifact.payload)]
    }


def _role_sets_by_name(artifacts: Sequence[Artifact]) -> dict[str, ColumnRoleSet]:
    """Index the source run's COLUMN_ROLE_SET artifacts by dataset name."""
    return {
        role_set.dataset: role_set
        for artifact in artifacts
        if artifact.type is ArtifactType.COLUMN_ROLE_SET
        for role_set in [ColumnRoleSet.model_validate(artifact.payload)]
    }


def _quality_context_by_id(artifacts: Sequence[Artifact]) -> dict[str, QualityContextSet]:
    return {
        artifact.id: QualityContextSet.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.QUALITY_CONTEXT_SET
    }


def _candidate_fingerprint(candidate: QuestionCandidate) -> str:
    return _canonical_sha256(candidate.model_dump(mode="json"))


def _plan_fingerprint(plan: InvestigationPlan) -> str:
    return _canonical_sha256(plan.model_dump(mode="json"))


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _outcome_artifact_id(investigation_id: str) -> str:
    return "irecord_" + stable_hash({"investigation_id": investigation_id}, length=16)


def _record_id(investigation_id: str, outcome: str) -> str:
    return "irec_" + stable_hash(
        {"investigation_id": investigation_id, "outcome": outcome},
        length=12,
    )


def _generate_plan_session_id(source_session_id: str, question_ids: Sequence[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash(
        {"source_session_id": source_session_id, "question_ids": sorted(set(question_ids))},
        length=8,
    )
    return f"investigation_{stamp}_{suffix}"


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))


# --------------------------------------------------------------------------- #
# L3 macro loop (analysis-loop design 2026-07-23 §2/§5.2/§8)
# --------------------------------------------------------------------------- #

MACRO_LOOP_ROUND_EVENT = "macro_loop_round"
_MACRO_PREAUTH_PREFIX = "macro_loop_"


@dataclass(frozen=True)
class MacroLoopResult:
    """Typed outcome of one macro-loop invocation over an executed plan run."""

    exit_reason: MacroLoopExitReason
    ledger: LoopLedger
    ledger_artifact_id: str
    executed_session_ids: list[str]


def _macro_preauth_investigation_id(plan_session_id: str) -> str:
    return _MACRO_PREAUTH_PREFIX + plan_session_id


def _macro_preauth_fingerprint(plan_session_id: str, *, depth: int, rounds_cap: int) -> str:
    return _canonical_sha256(
        {
            "authorization": "macro_loop_followup_execution",
            "plan_session_id": plan_session_id,
            "depth": depth,
            "rounds_cap": rounds_cap,
        }
    )


def preauthorize_macro_loop(
    *,
    project_id: str,
    plan_session_id: str,
    workspace: Path | str,
    depth: int,
) -> Artifact:
    """Persist the §8.3 pre-authorization for this run's follow-up execution.

    Enabling depth >=2 is the authorizing act; the artifact reuses the existing
    fingerprint-bound approval pattern so run_macro_loop can verify it.
    """
    depth = max(0, min(3, depth))
    profile = DEPTH_PROFILES[depth]
    if profile.rounds == 0:
        raise ValueError("Macro-loop pre-authorization requires depth >= 2.")
    approval = InvestigationApproval(
        approval_id="appr_" + stable_hash({"macro_loop": plan_session_id, "depth": depth}, length=12),
        investigation_id=_macro_preauth_investigation_id(plan_session_id),
        plan_fingerprint=_macro_preauth_fingerprint(
            plan_session_id, depth=depth, rounds_cap=profile.rounds
        ),
        decision="approved",
        reason=(
            "Authorizes follow-up execution within this run, "
            f"depth={depth}, rounds<={profile.rounds}."
        ),
        decided_at=datetime.now(UTC).isoformat(),
    )
    artifact = _approval_artifact(
        approval, project_id=project_id, session_id=plan_session_id, parent_ids=[]
    )
    ArtifactStore(Path(workspace)).save_artifact(artifact)
    return artifact


def _find_macro_preauthorization(
    artifacts: Sequence[Artifact],
    *,
    plan_session_id: str,
    depth: int,
    rounds_cap: int,
) -> Artifact | None:
    expected_id = _macro_preauth_investigation_id(plan_session_id)
    expected_fingerprint = _macro_preauth_fingerprint(
        plan_session_id, depth=depth, rounds_cap=rounds_cap
    )
    for artifact in artifacts:
        if artifact.type is not ArtifactType.INVESTIGATION_APPROVAL:
            continue
        approval = InvestigationApproval.model_validate(artifact.payload)
        if (
            approval.investigation_id == expected_id
            and approval.decision == "approved"
            and approval.plan_fingerprint == expected_fingerprint
        ):
            return artifact
    return None




# Audit findings that mean the probe claim's evidence cannot be resolved (§8.2
# "evidence must resolve"); no_numbers claims are only admissible without them.
_PROBE_EVIDENCE_FINDING_CODES = frozenset({"missing_evidence", "missing_evidence_artifact"})


def _probe_claim_verifiable(
    finding: QuestionFinding,
    *,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> bool:
    """§8.2 numeric gate: run the finding statement through the F0-F2 chain as a probe claim."""
    bundle = ReportBundle(
        project_id="macro_loop_probe",
        session_id="macro_loop_probe",
        sections=[
            ReportSection(
                title="probe",
                claims=[ReportClaim(text=finding.text, evidence=list(finding.evidence))],
            )
        ],
    )
    audit = validate_report_bundle(bundle, evidence_pack, sql_results=sql_results)
    rollup = bundle.sections[0].claims[0].numeric_rollup
    if rollup == "number_verified":
        return True
    if rollup != "no_numbers":
        return False
    # A no-number statement has no numeric gate to fail, so it must at least
    # cite evidence that resolves.
    return not any(item.code in _PROBE_EVIDENCE_FINDING_CODES for item in audit.findings)


def _qexec_fingerprint(
    qexec: QuestionExecutionResult, artifact_types: Mapping[str, str]
) -> str:
    """Content fingerprint (§8.1): evidence value set first, normalized text fallback."""
    values: list[tuple[str, str, float]] = []
    for finding in qexec.findings:
        for reference in finding.evidence:
            if isinstance(reference.value, int | float) and not isinstance(reference.value, bool):
                artifact_type = artifact_types.get(reference.artifact_id or "", reference.kind)
                values.append((artifact_type, reference.locator, float(reference.value)))
    return finding_fingerprint(
        [finding.text for finding in qexec.findings],
        values,
        family_key=qexec.question_id,
    )


@dataclass(frozen=True)
class _BridgeOutcome:
    ledger: LoopLedger
    admitted: list[ParentFinding]
    admitted_count: int
    redundant_count: int
    discarded_count: int


def _bridge_round_results(
    ledger: LoopLedger,
    *,
    batch_artifacts: Sequence[Artifact],
    candidates_by_id: Mapping[str, QuestionCandidate],
) -> _BridgeOutcome:
    """§8.2 validation bridge over one batch of QuestionExecutionResult artifacts."""
    evidence_pack = build_evidence_pack(list(batch_artifacts))
    sql_results = {
        artifact.id: SqlResult.model_validate(artifact.payload)
        for artifact in batch_artifacts
        if artifact.type is ArtifactType.SQL_RESULT
    }
    artifact_types = {artifact.id: artifact.type.value for artifact in batch_artifacts}
    finding_ids_by_question: dict[str, str] = {}
    for artifact in batch_artifacts:
        if artifact.type is ArtifactType.VALIDATED_FINDING:
            question_id = str(artifact.payload.get("question_id", ""))
            finding_id = str(artifact.payload.get("finding_id", ""))
            if question_id and finding_id:
                finding_ids_by_question.setdefault(question_id, finding_id)

    admitted: list[ParentFinding] = []
    admitted_count = 0
    redundant_count = 0
    discarded_count = 0
    for artifact in batch_artifacts:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        qexec = QuestionExecutionResult.model_validate(artifact.payload)
        if qexec.status != "succeeded" or not qexec.findings:
            discarded_count += 1
            continue
        if not all(
            _probe_claim_verifiable(finding, evidence_pack=evidence_pack, sql_results=sql_results)
            for finding in qexec.findings
        ):
            discarded_count += 1
            continue
        fingerprint = _qexec_fingerprint(qexec, artifact_types)
        if is_duplicate_finding(ledger, fingerprint):
            redundant_count += 1
            continue
        finding_id = finding_ids_by_question.get(
            qexec.question_id,
            "finding_" + stable_hash({"question_id": qexec.question_id}, length=12),
        )
        ledger = admit_finding(ledger, finding_id, fingerprint)
        admitted_count += 1
        candidate = candidates_by_id.get(qexec.question_id)
        if candidate is not None:
            admitted.append(
                ParentFinding(
                    finding_id=finding_id,
                    statements=[finding.text for finding in qexec.findings],
                    score=candidate.score,
                    target_datasets=list(candidate.target_datasets),
                    dataset_display_names=dict(candidate.dataset_display_names),
                )
            )
    return _BridgeOutcome(ledger, admitted, admitted_count, redundant_count, discarded_count)


def _copy_artifact_to_session(artifact: Artifact, *, session_id: str) -> Artifact:
    prefix = artifact.id.split("_", 1)[0] or "artifact"
    return Artifact(
        id=make_artifact_id(prefix, {"session_id": session_id, "copy_of": artifact.id}),
        type=artifact.type,
        project_id=artifact.project_id,
        session_id=session_id,
        parents=[artifact.id],
        payload=artifact.payload,
        warnings=list(artifact.warnings),
        evidence=list(artifact.evidence),
        plain_language=artifact.plain_language,
    )


_FOLLOWUP_SOURCE_COPY_TYPES = frozenset(
    {
        ArtifactType.DATASET_PROFILE,
        ArtifactType.COLUMN_ROLE_SET,
        ArtifactType.QUALITY_CONTEXT_SET,
    }
)


def _execute_followup_round(
    store: ArtifactStore,
    *,
    project_id: str,
    plan_session_id: str,
    source_artifacts: Sequence[Artifact],
    candidates: Sequence[QuestionCandidate],
    round_id: int,
    workspace: Path,
    llm: LLMClient,
    session_budget: SessionBudgetState,
    preauth_artifact_id: str,
    preview_rows: int,
    timeout_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[Artifact], int, list[str]]:
    """Run follow-up candidates through the existing plan->approve->execute funnel.

    The candidates plus copies of the source run's profile/role/context artifacts
    form a derived source run, so the unchanged lifecycle functions can operate.
    Both derived runs carry the INTERNAL_SESSION_MARKER so history/coverage skip them.
    """
    raise_if_cancelled(cancel_check, operation="macro loop follow-up")
    followup_source_session_id = f"{plan_session_id}_macro_r{round_id}_source{INTERNAL_SESSION_MARKER}"
    store.start_session(project_id, followup_source_session_id)
    store.write_manifest(
        SessionManifest(
            session_id=followup_source_session_id,
            project_id=project_id,
            input_hashes={plan_session_id: "macro_loop_followups"},
            code_version="investigation-orchestrator-v2",
            model_versions={"orchestrator": "deterministic"},
            source_session_id=plan_session_id,
        )
    )
    candidate_set = QuestionCandidateSet(candidates=list(candidates))
    store.save_artifact(
        Artifact(
            id=make_artifact_id(
                "qcand",
                {
                    "session_id": followup_source_session_id,
                    "questions": [candidate.question_id for candidate in candidates],
                },
            ),
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id=project_id,
            session_id=followup_source_session_id,
            payload=candidate_set.model_dump(mode="json"),
            plain_language="Macro-loop follow-up question candidates for one round.",
        )
    )
    for artifact in source_artifacts:
        raise_if_cancelled(cancel_check, operation="macro loop follow-up")
        if artifact.type in _FOLLOWUP_SOURCE_COPY_TYPES:
            store.save_artifact(_copy_artifact_to_session(artifact, session_id=followup_source_session_id))
    store.mark_session_status(project_id, followup_source_session_id, "completed")

    planned = create_investigation_plans(
        project_id=project_id,
        source_session_id=followup_source_session_id,
        question_ids=[candidate.question_id for candidate in candidates],
        workspace=workspace,
        session_id=f"{plan_session_id}_macro_r{round_id}{INTERNAL_SESSION_MARKER}",
        cancel_check=cancel_check,
    )
    executable_plan_ids = [
        artifact.id
        for artifact in planned.artifacts
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
        for plan in [InvestigationPlan.model_validate(artifact.payload)]
        if plan.status == "planned" and plan.execution_ready
    ]
    session_ids = [followup_source_session_id, planned.session_id]
    if not executable_plan_ids:
        return [], 0, session_ids
    for plan_id in executable_plan_ids:
        raise_if_cancelled(cancel_check, operation="macro loop follow-up")
        approve_plan(
            project_id=project_id,
            plan_session_id=planned.session_id,
            plan_id=plan_id,
            workspace=workspace,
            reason=f"Pre-authorized macro-loop follow-up execution ({preauth_artifact_id}).",
        )
    executed = execute_investigation_plans(
        project_id=project_id,
        plan_session_id=planned.session_id,
        plan_ids=executable_plan_ids,
        workspace=workspace,
        llm=llm,
        preview_rows=preview_rows,
        timeout_seconds=timeout_seconds,
        budget_policy=session_budget.policy,
        restored_session_budget=session_budget,
        llm_session_id=plan_session_id,
        cancel_check=cancel_check,
    )
    return executed.artifacts, len(executable_plan_ids), session_ids


def _budget_exhausted(budget: SessionBudgetState) -> bool:
    try:
        budget.check_wall_time()
    except BudgetExceeded:
        return True
    return any(
        remaining is not None and remaining <= 0
        for remaining in (
            budget.remaining("requests"),
            budget.remaining("input_tokens"),
            budget.remaining("output_tokens"),
            budget.remaining("total_tokens"),
            budget.remaining("cost_usd"),
        )
    )


def _profiles_summary(source_artifacts: Sequence[Artifact]) -> str:
    return "; ".join(
        f"{profile.name}: {profile.rows} rows x {profile.columns} columns"
        for profile in _profiles_by_name(source_artifacts).values()
    )


def run_macro_loop(
    *,
    project_id: str,
    plan_session_id: str,
    workspace: Path | str,
    llm: LLMClient | None = None,
    depth: int = 0,
    budget_policy: SessionBudgetPolicy | None = None,
    restored_session_budget: SessionBudgetState | None = None,
    data_summary: str = "",
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    cancel_check: Callable[[], bool] | None = None,
) -> MacroLoopResult | None:
    """Run the depth >=2 macro loop over an executed plan run (design §2).

    Round 0 bridges the plan run's own execution results into the ledger; each
    numbered round then runs follow-up generation -> deterministic scoring/dedup
    -> pre-authorized funnel execution -> validation bridge, so every executed
    batch is bridged and accounted before the loop judges termination. Depth 0/1
    is a no-op returning None (single-pass status quo). Every exit is typed and
    the ledger persists as a LOOP_LEDGER artifact on the plan run.
    """
    raise_if_cancelled(cancel_check, operation="macro loop")
    depth = max(0, min(3, depth))
    profile = DEPTH_PROFILES[depth]
    if profile.rounds == 0:
        return None

    workspace_path = require_absolute_workspace(workspace)
    store = ArtifactStore(workspace_path)
    plan_artifacts = store.list_artifacts(project_id=project_id, session_id=plan_session_id)
    preauth = _find_macro_preauthorization(
        plan_artifacts, plan_session_id=plan_session_id, depth=depth, rounds_cap=profile.rounds
    )
    if preauth is None:
        raise ValueError(
            "Macro-loop follow-up execution requires a matching pre-authorization "
            "artifact for this run and depth; call preauthorize_macro_loop first."
        )
    plans = [
        InvestigationPlan.model_validate(artifact.payload)
        for artifact in plan_artifacts
        if artifact.type is ArtifactType.INVESTIGATION_PLAN
    ]
    if not plans:
        raise ValueError("Macro loop requires a plan run with InvestigationPlan artifacts.")
    source_session_ids = {plan.source_session_id for plan in plans}
    if len(source_session_ids) != 1:
        raise ValueError("Macro loop requires plans from a single source run.")
    source_session_id = source_session_ids.pop()
    source_artifacts = store.list_artifacts(project_id=project_id, session_id=source_session_id)
    candidates_by_id = {
        candidate.question_id: candidate
        for candidate in QuestionCandidateSet.model_validate(
            _source_candidate_set(source_artifacts).payload
        ).candidates
    }
    summary_text = data_summary or _profiles_summary(source_artifacts)

    effective_budget_policy = (
        restored_session_budget.policy
        if restored_session_budget is not None
        else budget_policy or SessionBudgetPolicy()
    )
    if budget_policy is not None and budget_policy != effective_budget_policy:
        raise ValueError("Restored run budget policy does not match budget_policy.")
    prior_events = store.list_trace_events(project_id=project_id, session_id=plan_session_id)
    session_budget = restored_session_budget or restore_run_budget_state(
        effective_budget_policy,
        prior_events,
    )

    def _sink(event: TraceEvent) -> None:
        store.append_trace(project_id, event)

    # This is idempotent when the Worker already installed the ledger with the
    # same run and restored budget, and is the single seam for direct callers.
    run_llm = meter_llm_client(
        llm or OfflineLLMClient(),
        session_id=plan_session_id,
        emit=_sink,
        budget=session_budget,
        session_dir=store.session_dir(project_id, plan_session_id),
    )

    def _ledger_tokens() -> int:
        """Read the durable accounting truth instead of mutable provider state."""
        return sum(
            int(event.summary.get("total_tokens", 0))
            for event in store.list_trace_events(
                project_id=project_id,
                session_id=plan_session_id,
            )
            if event.event_type == LLM_USAGE_EVENT
        )

    # Cross-round question dedup starts from what this run already asked.
    ledger = LoopLedger(
        depth=depth,
        question_fingerprints=list(
            dict.fromkeys(question_fingerprint(plan.question) for plan in plans)
        ),
    )
    parents: list[ParentFinding] = []
    executed_session_ids: list[str] = []

    def _bridge_counts(record: LoopRoundRecord, bridge: _BridgeOutcome) -> LoopRoundRecord:
        parents.extend(bridge.admitted)
        return record.model_copy(
            update={
                "new_validated_findings": bridge.admitted_count,
                "redundant_findings": bridge.redundant_count,
                "discarded_findings": bridge.discarded_count,
            }
        )

    def _settle_round(
        record: LoopRoundRecord,
        exit_reason: MacroLoopExitReason,
        tokens_before: int,
    ) -> MacroLoopExitReason:
        """Account tokens, fix the exit reason, append the row, emit the trace."""
        nonlocal ledger
        round_tokens = _ledger_tokens() - tokens_before
        if exit_reason != "crash" and _budget_exhausted(session_budget):
            exit_reason = "budget_cap"
        record = record.model_copy(update={"tokens": round_tokens, "exit_reason": exit_reason})
        record = record.model_copy(update={"disposition": keep_or_discard(record)})
        ledger = record_round(ledger, record)
        _sink(
            trace_event(
                session_id=plan_session_id,
                event_type=MACRO_LOOP_ROUND_EVENT,
                name="investigation_orchestrator",
                finished_at=datetime.now(UTC),
                summary={"depth": depth, **record.model_dump(mode="json")},
            )
        )
        return exit_reason

    # Round 0: bridge the plan run's own execution results (the seed batch).
    raise_if_cancelled(cancel_check, operation="macro loop")
    tokens_before = _ledger_tokens()
    seed_record = LoopRoundRecord(round_id=0)
    exit_reason: MacroLoopExitReason = "continue"
    try:
        seed_bridge = _bridge_round_results(
            ledger, batch_artifacts=plan_artifacts, candidates_by_id=candidates_by_id
        )
        ledger = seed_bridge.ledger
        seed_record = _bridge_counts(seed_record, seed_bridge)
        if _budget_exhausted(session_budget):
            exit_reason = "budget_cap"
        elif seed_bridge.admitted_count == 0:
            exit_reason = "no_new_information"
    except BudgetExceeded:
        exit_reason = "budget_cap"
    except Exception:  # noqa: BLE001 - crash rounds are recorded, never raised
        exit_reason = "crash"
    exit_reason = _settle_round(seed_record, exit_reason, tokens_before)

    for round_id in range(1, profile.rounds + 1):
        raise_if_cancelled(cancel_check, operation="macro loop")
        if exit_reason != "continue":
            break
        tokens_before = _ledger_tokens()
        record = LoopRoundRecord(round_id=round_id)
        exit_reason = "continue"
        try:
            if _budget_exhausted(session_budget):
                exit_reason = "budget_cap"
            else:
                generation = generate_followup_proposals(
                    run_llm,
                    ledger=ledger,
                    parent_findings=parents,
                    data_summary=summary_text,
                    round_id=round_id,
                    max_questions=profile.per_round_questions,
                    session_id=plan_session_id,
                    trace_sink=_sink,
                )
                raise_if_cancelled(cancel_check, operation="macro loop")
                if generation.proposal_set.concluded:
                    exit_reason = "concluded"
                else:
                    conversion = followup_question_candidates(
                        generation.proposal_set, ledger=ledger, parent_findings=parents
                    )
                    followups = conversion.candidates[: profile.per_round_questions]
                    if not followups:
                        exit_reason = "no_new_information"
                    elif _budget_exhausted(session_budget):
                        exit_reason = "budget_cap"
                    else:
                        candidates_by_id.update(
                            {candidate.question_id: candidate for candidate in followups}
                        )
                        raise_if_cancelled(cancel_check, operation="macro loop")
                        batch, executed_count, round_session_ids = _execute_followup_round(
                            store,
                            project_id=project_id,
                            plan_session_id=plan_session_id,
                            source_artifacts=source_artifacts,
                            candidates=followups,
                            round_id=round_id,
                            workspace=workspace_path,
                            llm=run_llm,
                            session_budget=session_budget,
                            preauth_artifact_id=preauth.id,
                            preview_rows=preview_rows,
                            timeout_seconds=timeout_seconds,
                            cancel_check=cancel_check,
                        )
                        raise_if_cancelled(cancel_check, operation="macro loop")
                        executed_session_ids.extend(round_session_ids)
                        record = record.model_copy(
                            update={"executed_questions": executed_count}
                        )
                        # Question fingerprints register only after the round
                        # executed, so a crashed half-round leaves no dead
                        # fingerprints that would block a retry.
                        ledger = ledger.model_copy(
                            update={
                                "question_fingerprints": [
                                    *ledger.question_fingerprints,
                                    *[
                                        question_fingerprint(candidate.question_en)
                                        for candidate in followups
                                    ],
                                ]
                            }
                        )
                        # Bridge this round's executed batch before judging
                        # termination: the last round's results reach the
                        # ledger like every other round's.
                        bridge = _bridge_round_results(
                            ledger, batch_artifacts=batch, candidates_by_id=candidates_by_id
                        )
                        raise_if_cancelled(cancel_check, operation="macro loop")
                        ledger = bridge.ledger
                        record = _bridge_counts(record, bridge)
                        if _budget_exhausted(session_budget):
                            exit_reason = "budget_cap"
                        elif bridge.admitted_count == 0:
                            exit_reason = "no_new_information"
                        elif round_id >= profile.rounds:
                            exit_reason = "round_cap"
        except SessionCancelled:
            raise
        except BudgetExceeded:
            exit_reason = "budget_cap"
        except Exception:  # noqa: BLE001 - crash rounds are recorded, never raised
            exit_reason = "crash"
        exit_reason = _settle_round(record, exit_reason, tokens_before)

    raise_if_cancelled(cancel_check, operation="macro loop")
    ledger_artifact = Artifact(
        id=make_artifact_id("loopledger", {"session_id": plan_session_id, "depth": depth}),
        type=ArtifactType.LOOP_LEDGER,
        project_id=project_id,
        session_id=plan_session_id,
        parents=[preauth.id],
        payload=ledger.model_dump(mode="json"),
        plain_language=(
            "Append-only macro-loop ledger: admitted finding fingerprints, question "
            "fingerprints, and per-round keep/discard records."
        ),
    )
    store.save_artifact(ledger_artifact)
    return MacroLoopResult(
        exit_reason=exit_reason,
        ledger=ledger,
        ledger_artifact_id=ledger_artifact.id,
        executed_session_ids=executed_session_ids,
    )

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import duckdb

from eda_platform.agents.interpretation import interpret_findings
from eda_platform.agents.planner import build_plan
from eda_platform.agents.question_runtime import run_question_agent
from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.agents.runtime import AgentRunResult
from eda_platform.core.budget import BudgetExceeded, SessionBudgetPolicy
from eda_platform.core.config import require_absolute_workspace
from eda_platform.core.currency_units import classify_currency_unit, currency_unit_display
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.llm import (
    LLMClient,
    OfflineLLMClient,
    ToolCallingLLM,
    ToolCallingUnsupportedError,
    is_offline_client,
    manifest_model_versions,
)
from eda_platform.core.llm_ledger import meter_llm_client, restore_run_budget_state
from eda_platform.core.query import DuckDBQueryEngine, QueryTimeout, UnsafeQueryError
from eda_platform.core.sandbox import ExecutionBackend
from eda_platform.core.semantic import (
    SemanticSeeds,
    load_join_whitelist,
    record_join_usage,
)
from eda_platform.core.semantic_resources import load_semantic_seeds_safe
from eda_platform.core.session_metrics import persist_run_metrics
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_calling_probe import tool_calling_readiness
from eda_platform.core.tool_guard import ToolGuardError, check_sql_joins_declared
from eda_platform.drivers.cancellation import raise_if_cancelled
from eda_platform.drivers.report_artifacts import build_agentic_report_artifacts
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionExecutionResult,
    QuestionFinding,
)
from eda_platform.schemas.relations import RelationshipCandidateSet
from eda_platform.schemas.sessions import (
    SessionManifest,
    TraceEvent,
    build_run_title,
    clip_run_title,
)
from eda_platform.tools.domain_metrics import (
    DOMAIN_METRIC_REGISTRY,
    MetricDefinition,
    carrier_basis_note,
    metric_definition,
    validate_metric_result,
)
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.relationship_discovery import _relation_name
from eda_platform.tools.report_validator import full_coverage_evidence_refs
from eda_platform.tools.sql_runner import SqlCatalog, build_catalog, run_sql


@dataclass(frozen=True)
class QuestionBatchRunResult:
    project_id: str
    session_id: str
    source_session_id: str
    artifacts: list[Artifact]
    workspace: Path


@dataclass(frozen=True)
class _ResultContractFailure:
    code: str
    reason: str


def select_auto_execution_candidates(
    candidate_set: QuestionCandidateSet,
    *,
    relationship_candidates: RelationshipCandidateSet | None = None,
    limit: int = 3,
) -> list[QuestionCandidate]:
    # Kept for call compatibility. Automatic execution never joins datasets;
    # a user must select a question that explicitly needs a relationship.
    _ = relationship_candidates
    ranked = [
        candidate
        for candidate in sorted(
            candidate_set.candidates,
            key=lambda item: (-item.score.deterministic_score, item.question_id),
        )
        if _eligible_for_auto_execution(candidate)
    ]
    selected: list[QuestionCandidate] = []
    selected_ids: set[str] = set()
    used_templates: set[str] = set()
    used_datasets: set[str] = set()

    for prefer_distinct_dataset in (True, False):
        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate.question_id in selected_ids:
                continue
            template_id = candidate.template_id or ""
            if template_id in used_templates:
                continue
            candidate_datasets = set(candidate.target_datasets)
            if (
                prefer_distinct_dataset
                and used_datasets
                and not candidate_datasets.difference(used_datasets)
            ):
                continue
            _select_auto_candidate(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                used_templates=used_templates,
                used_datasets=used_datasets,
            )
        if len(selected) >= limit:
            break
    return selected[:limit]


def _eligible_for_auto_execution(candidate: QuestionCandidate) -> bool:
    if candidate.origin != "template" or candidate.sql_template is None:
        return False
    if candidate.feasibility is not None and candidate.feasibility.status in {
        "needs_data",
        "unsuitable",
    }:
        return False
    if candidate.score.join_risk > 0.3 or candidate.score.quality_risk > 0.6:
        return False
    if candidate.required_relations:
        return False
    return True


def _select_auto_candidate(
    candidate: QuestionCandidate,
    *,
    selected: list[QuestionCandidate],
    selected_ids: set[str],
    used_templates: set[str],
    used_datasets: set[str],
) -> None:
    selected.append(candidate.model_copy(update={"status": "auto_selected"}))
    selected_ids.add(candidate.question_id)
    used_templates.add(candidate.template_id or "")
    used_datasets.update(candidate.target_datasets)


def _join_whitelist_safe(store: ArtifactStore, project_id: str):
    """The whitelist object itself (labels + disclosure notes); None when unreadable."""
    try:
        return load_join_whitelist(store.project_dir(project_id))
    except (OSError, ValueError):
        return None


def _record_join_usage_safe(store: ArtifactStore, project_id: str, label: str) -> None:
    try:
        record_join_usage(store.project_dir(project_id), label)
    except (OSError, ValueError):
        pass  # usage metering must never fail a completed execution


def follow_up_session_title(
    store: ArtifactStore,
    *,
    project_id: str,
    source_session_id: str,
    question_count: int,
) -> str:
    """Deterministic title for a follow-up batch run."""
    source_title = ""
    try:
        source_manifest = store.read_manifest(project_id, source_session_id)
    except (OSError, ValueError):
        source_manifest = None
    if source_manifest is not None:
        source_title = source_manifest.title or build_run_title(list(source_manifest.input_hashes))
    if source_title:
        return clip_run_title(f"Follow-up: {source_title}")
    noun = "question" if question_count == 1 else "questions"
    return clip_run_title(f"Follow-up ({question_count} {noun})")


def execute_question_candidate(
    candidate: QuestionCandidate,
    *,
    datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
    llm: LLMClient | None = None,
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    seeds: SemanticSeeds | None = None,
    on_guard_rejected: Callable[[ToolGuardError], None] | None = None,
    confirmed_joins: Collection[str] = (),
    on_join_used: Callable[[str], None] | None = None,
) -> list[Artifact]:
    if candidate.origin == "llm":
        return _execute_llm_question(
            candidate,
            datasets=datasets,
            project_id=project_id,
            session_id=session_id,
            parent_ids=parent_ids,
            llm=llm,
            preview_rows=preview_rows,
            timeout_seconds=timeout_seconds,
            seeds=seeds,
            on_guard_rejected=on_guard_rejected,
            confirmed_joins=confirmed_joins,
            on_join_used=on_join_used,
        )
    return _execute_template_question(
        candidate,
        datasets=datasets,
        project_id=project_id,
        session_id=session_id,
        parent_ids=parent_ids,
        llm=llm,
        preview_rows=preview_rows,
        timeout_seconds=timeout_seconds,
        seeds=seeds,
        confirmed_joins=confirmed_joins,
        on_join_used=on_join_used,
    )


def run_question_batch(
    *,
    project_id: str,
    source_session_id: str,
    question_ids: Sequence[str],
    workspace: Path | str,
    llm: LLMClient | None = None,
    session_id: str | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
    business_context: str = "",
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    generate_report: bool = True,
    budget_policy: SessionBudgetPolicy | None = None,
    code_backend: ExecutionBackend | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> QuestionBatchRunResult:
    raise_if_cancelled(cancel_check, operation="question batch")
    actual_session_id = session_id or _generate_batch_session_id(source_session_id, question_ids)
    try:
        return _run_question_batch(
            project_id=project_id,
            source_session_id=source_session_id,
            question_ids=question_ids,
            workspace=workspace,
            llm=llm,
            session_id=actual_session_id,
            payload_policy=payload_policy,
            business_context=business_context,
            preview_rows=preview_rows,
            timeout_seconds=timeout_seconds,
            generate_report=generate_report,
            budget_policy=budget_policy,
            code_backend=code_backend,
            cancel_check=cancel_check,
        )
    except BudgetExceeded as exc:
        store = ArtifactStore(Path(workspace))
        finished_at = datetime.now(UTC)
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=actual_session_id,
                event_type="step_failed",
                name="question_batch",
                finished_at=finished_at,
                summary={
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ),
        )
        store.mark_session_status(project_id, actual_session_id, "failed")
        _persist_batch_metrics_best_effort(store, project_id, actual_session_id)
        raise


def _run_question_batch(
    *,
    project_id: str,
    source_session_id: str,
    question_ids: Sequence[str],
    workspace: Path | str,
    llm: LLMClient | None = None,
    session_id: str | None = None,
    payload_policy: PayloadPolicy = "schema+aggregates",
    business_context: str = "",
    preview_rows: int = 50,
    timeout_seconds: float = 10.0,
    generate_report: bool = True,
    budget_policy: SessionBudgetPolicy | None = None,
    code_backend: ExecutionBackend | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> QuestionBatchRunResult:
    raise_if_cancelled(cancel_check, operation="question batch")
    workspace_path = require_absolute_workspace(workspace)
    store = ArtifactStore(workspace_path)
    source_artifacts = store.list_artifacts(project_id=project_id, session_id=source_session_id)
    source_qcand = _source_question_candidate_set(source_artifacts)
    candidate_set = QuestionCandidateSet.model_validate(source_qcand.payload)
    datasets = _load_source_datasets(
        store,
        project_id=project_id,
        source_session_id=source_session_id,
        source_artifacts=source_artifacts,
    )
    actual_session_id = session_id or _generate_batch_session_id(source_session_id, question_ids)

    def emit_usage(event: TraceEvent) -> None:
        store.append_trace(project_id, event)

    effective_budget_policy = budget_policy or SessionBudgetPolicy()
    session_budget = restore_run_budget_state(
        effective_budget_policy,
        store.list_trace_events(project_id=project_id, session_id=actual_session_id),
    )
    run_llm = meter_llm_client(
        llm or OfflineLLMClient(),
        session_id=actual_session_id,
        emit=emit_usage,
        budget=session_budget,
        session_dir=store.session_dir(project_id, actual_session_id),
    )
    manifest = SessionManifest(
        session_id=actual_session_id,
        project_id=project_id,
        input_hashes={dataset.record.name: dataset.record.content_hash for dataset in datasets},
        code_version=_current_code_version(),
        model_versions=manifest_model_versions(run_llm),
        title=follow_up_session_title(
            store,
            project_id=project_id,
            source_session_id=source_session_id,
            question_count=len(question_ids),
        ),
        source_session_id=source_session_id,
    )
    store.start_session(project_id, actual_session_id)
    store.write_manifest(manifest)
    batch_started_at = datetime.now(UTC)
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=actual_session_id,
            event_type="step_started",
            name="question_batch",
            started_at=batch_started_at,
            summary={"question_count": len(question_ids)},
        ),
    )

    # Settled before the batch spends anything: an unverified model that
    # cannot take a tools payload should cost one probe call, not a full
    # question's worth of planning first.
    readiness = tool_calling_readiness(run_llm)
    if readiness.source in {"probe", "cached"}:
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=actual_session_id,
                event_type="tool_calling_probe",
                name="question_batch",
                started_at=batch_started_at,
                summary={
                    "usable": readiness.usable,
                    "source": readiness.source,
                    "detail": readiness.detail,
                },
            ),
        )
    agent_route = readiness.usable

    seeds = load_semantic_seeds_safe(store, project_id)
    batch_whitelist = _join_whitelist_safe(store, project_id)
    current_dataset_ids = {dataset.record.name: dataset.record.dataset_id for dataset in datasets}
    confirmed_joins = (
        batch_whitelist.confirmed_labels(current_dataset_ids)
        if batch_whitelist is not None
        else set()
    )
    if batch_whitelist is not None:
        freshness_counts = batch_whitelist.validation_freshness_counts(current_dataset_ids)
        if sum(freshness_counts.values()):
            store.append_trace(
                project_id,
                TraceEvent(
                    session_id=actual_session_id,
                    event_type="join_authorization_freshness",
                    name="run_question_batch",
                    summary=freshness_counts,
                ),
            )
    by_id = {candidate.question_id: candidate for candidate in candidate_set.candidates}
    source_parent_ids = [artifact.id for artifact in source_artifacts]
    new_artifacts: list[Artifact] = []
    for question_id in question_ids:
        raise_if_cancelled(cancel_check, operation="question batch")
        session_budget.check_wall_time()
        candidate = by_id.get(question_id)
        if candidate is None:
            artifacts = [
                _failed_qexec_artifact(
                    question_id=question_id,
                    question=f"Unknown question id: {question_id}",
                    origin="template",
                    project_id=project_id,
                    session_id=actual_session_id,
                    parent_ids=source_parent_ids,
                    error="Question id was not found in the source QuestionCandidateSet.",
                )
            ]
        else:
            # Disclose machine-confirmed joins in result risks.
            if candidate.required_relations and batch_whitelist is not None:
                notes = batch_whitelist.disclosure_notes(candidate.required_relations)
                if notes:
                    candidate = candidate.model_copy(update={"risks": [*candidate.risks, *notes]})
            if agent_route:
                try:
                    agent_result = run_question_agent(
                        candidate.question_en,
                        candidate_context=_agent_candidate_context(candidate),
                        datasets=_agent_scoped_datasets(candidate, datasets),
                        project_id=project_id,
                        session_id=actual_session_id,
                        llm=cast(ToolCallingLLM, run_llm),
                        artifacts=source_artifacts,
                        store=store,
                        payload_policy=payload_policy,
                        code_backend=code_backend,
                        timeout_seconds=timeout_seconds,
                    )
                except ToolCallingUnsupportedError as exc:
                    # Only a provider that diagnosed the tools payload lands
                    # here, so this is a capability fact and not a retry:
                    # the rest of the batch takes the deterministic path.
                    agent_route = False
                    store.append_trace(
                        project_id,
                        TraceEvent(
                            session_id=actual_session_id,
                            event_type="agent_route_degraded",
                            name="question_batch",
                            summary={
                                "question_id": question_id,
                                "reason": str(exc)[:500],
                            },
                        ),
                    )
                else:
                    # Reading an existing artifact makes it evidence, not a new
                    # artifact of this derived run. Only locally produced tool
                    # outputs are returned and re-saved with the qexec result.
                    artifacts = [
                        artifact
                        for artifact in cast(list[Artifact], agent_result.artifacts)
                        if artifact.session_id == actual_session_id
                    ]
                    artifacts.append(
                        _agent_qexec_artifact(
                            candidate,
                            agent_result=agent_result,
                            project_id=project_id,
                            session_id=actual_session_id,
                            parent_ids=[*source_parent_ids, source_qcand.id],
                        )
                    )
            if not agent_route:
                # Offline mode, adapters without a tool transport, and a model
                # the provider just refused a tools payload for.
                artifacts = execute_question_candidate(
                    candidate,
                    datasets=datasets,
                    project_id=project_id,
                    session_id=actual_session_id,
                    parent_ids=[*source_parent_ids, source_qcand.id],
                    llm=run_llm,
                    preview_rows=preview_rows,
                    timeout_seconds=timeout_seconds,
                    seeds=seeds,
                    on_guard_rejected=lambda error: _append_tool_guard_rejection(
                        store,
                        project_id,
                        actual_session_id,
                        name="m3_build_plan",
                        error=error,
                    ),
                    confirmed_joins=confirmed_joins,
                    on_join_used=lambda label: _record_join_usage_safe(
                        store, project_id, label
                    ),
                )
        raise_if_cancelled(cancel_check, operation="question batch")
        for artifact in artifacts:
            store.save_artifact(artifact)
        new_artifacts.extend(artifacts)
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=actual_session_id,
                event_type="question_execution_completed",
                name="run_question_batch",
                summary={
                    "question_id": question_id,
                    "artifact_count": len(artifacts),
                    "status": _qexec_status(artifacts),
                    "outcome": _qexec_outcome(artifacts),
                    "abstention_code": _qexec_abstention_code(artifacts),
                },
            ),
        )
        if _qexec_outcome(artifacts) == "abstained":
            store.append_trace(
                project_id,
                TraceEvent(
                    session_id=actual_session_id,
                    event_type="question_result_contract",
                    name="run_question_batch",
                    summary={
                        "question_id": question_id,
                        "verdict": "abstained",
                        "code": _qexec_abstention_code(artifacts),
                    },
                ),
            )

    report_artifacts: list[Artifact] = []
    if generate_report:
        raise_if_cancelled(cancel_check, operation="question batch report")
        session_budget.check_wall_time()
        report_artifacts = _regenerate_report(
            [*source_artifacts, *new_artifacts],
            project_id=project_id,
            session_id=actual_session_id,
            business_context=business_context,
            llm=run_llm,
            payload_policy=payload_policy,
        )
        raise_if_cancelled(cancel_check, operation="question batch report")
        for artifact in report_artifacts:
            store.save_artifact(artifact)
        _write_report_files(store, project_id, actual_session_id, report_artifacts)
    raise_if_cancelled(cancel_check, operation="question batch")
    batch_finished_at = datetime.now(UTC)
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=actual_session_id,
            event_type="step_completed",
            name="question_batch",
            started_at=batch_started_at,
            finished_at=batch_finished_at,
            summary={
                "question_count": len(question_ids),
                "artifact_count": len(new_artifacts) + len(report_artifacts),
            },
        ),
    )
    store.mark_session_status(project_id, actual_session_id, "completed")
    _persist_batch_metrics_best_effort(store, project_id, actual_session_id)
    return QuestionBatchRunResult(
        project_id=project_id,
        session_id=actual_session_id,
        source_session_id=source_session_id,
        artifacts=[*new_artifacts, *report_artifacts],
        workspace=workspace_path,
    )


def _persist_batch_metrics_best_effort(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
) -> None:
    try:
        persist_run_metrics(store, project_id, session_id)
    except Exception as exc:  # noqa: BLE001 - metrics cannot invalidate completed analysis
        store.append_trace(
            project_id,
            TraceEvent(
                session_id=session_id,
                event_type="session_metrics_error",
                name="persist_run_metrics",
                finished_at=datetime.now(UTC),
                summary={"error_type": type(exc).__name__, "error": str(exc)[:500]},
            ),
        )


def _agent_candidate_context(candidate: QuestionCandidate) -> dict[str, Any]:
    """Approval-scoped framing supplied to the agent without prescribing a method."""
    return candidate.model_dump(
        mode="json",
        include={
            "question_id",
            "target_datasets",
            "dataset_display_names",
            "required_relations",
            "business_decision",
            "value_hypothesis",
            "analysis_mode",
            "candidate_methods",
            "data_requirements",
            "success_criterion",
            "risks",
            "data_signal",
            "referenced_columns",
            "source_artifact_ids",
            "quality_context_artifact_ids",
        },
    )


def _agent_scoped_datasets(
    candidate: QuestionCandidate,
    datasets: Sequence[LoadedDataset],
) -> list[LoadedDataset]:
    """Expose only the datasets approved on this question card."""
    approved = set(candidate.target_datasets)
    selected = [
        dataset
        for dataset in datasets
        if dataset.record.name in approved or dataset.record.dataset_id in approved
    ]
    if not selected:
        raise ValueError(
            "None of the question's approved target datasets are available in the source run."
        )
    return selected


def _agent_qexec_artifact(
    candidate: QuestionCandidate,
    *,
    agent_result: AgentRunResult,
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
) -> Artifact:
    evidence_artifacts = _unique_agent_artifacts(agent_result.artifacts)
    evidence_ids = [artifact.id for artifact in evidence_artifacts]
    common = {
        "execution_mode": "agent",
        "tool_calls": agent_result.tool_calls,
        "tool_names": list(agent_result.tool_names),
        "evidence_artifact_ids": evidence_ids,
    }
    if agent_result.status == "answer_unverified":
        # The loop ran and produced evidence, but the answer failed the exit
        # gate twice. Abstaining keeps unverified prose out of the fact layer.
        return _failed_qexec_artifact(
            question_id=candidate.question_id,
            question=candidate.question_en,
            origin=candidate.origin,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parent_ids, *evidence_ids],
            error=agent_result.error or "The agent answer could not be verified.",
            outcome="abstained",
            abstention_code="agent_answer_unverified",
            exploratory=candidate.exploratory,
            **common,
        )
    if agent_result.status != "completed":
        return _failed_qexec_artifact(
            question_id=candidate.question_id,
            question=candidate.question_en,
            origin=candidate.origin,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parent_ids, *evidence_ids],
            error=agent_result.error or f"Agent stopped with status {agent_result.status}.",
            exploratory=candidate.exploratory,
            **common,
        )
    if not evidence_artifacts:
        return _failed_qexec_artifact(
            question_id=candidate.question_id,
            question=candidate.question_en,
            origin=candidate.origin,
            project_id=project_id,
            session_id=session_id,
            parent_ids=parent_ids,
            error="The agent produced an answer without any evidence artifact.",
            outcome="abstained",
            abstention_code="agent_no_evidence",
            exploratory=candidate.exploratory,
            **common,
        )

    sql_artifacts = [
        artifact
        for artifact in evidence_artifacts
        if artifact.type is ArtifactType.SQL_RESULT
    ]
    findings: list[QuestionFinding] = []
    for artifact in sql_artifacts:
        try:
            findings.extend(_findings(candidate, artifact))
        except (ValueError, TypeError):
            # Agent SQL need not have the legacy card template's result shape.
            continue
    if not findings:
        findings = [
            QuestionFinding(
                text=agent_result.answer.strip(),
                evidence=[
                    ref
                    for artifact in evidence_artifacts[:30]
                    for ref in _agent_evidence_refs(artifact)
                ],
                exploratory=candidate.exploratory,
            )
        ]

    last_sql = sql_artifacts[-1] if sql_artifacts else None
    sql_text = (
        SqlResult.model_validate(last_sql.payload).sql
        if last_sql is not None
        else None
    )
    unique_tool_names = list(dict.fromkeys(agent_result.tool_names))
    result = QuestionExecutionResult(
        question_id=candidate.question_id,
        question=candidate.question_en,
        origin=candidate.origin,
        execution_mode="agent",
        tool_calls=agent_result.tool_calls,
        tool_names=unique_tool_names,
        evidence_artifact_ids=evidence_ids,
        plan_summary=(
            f"Autonomous agent completed {agent_result.tool_calls} tool call(s)"
            + (
                f" using {', '.join(unique_tool_names)}."
                if unique_tool_names
                else "."
            )
        ),
        sql=sql_text,
        sql_result_artifact_id=last_sql.id if last_sql is not None else None,
        findings=findings,
        status="succeeded",
        outcome="answered",
        interpretation=agent_result.answer.strip(),
        # Reaching here means the answer passed the agent exit gate: every
        # figure resolved against a persisted tool payload. "fallback" used to
        # be stamped here, but on the legacy path that value means "the
        # validator rejected it", which the UI renders identically.
        interpretation_status="validated",
        exploratory=candidate.exploratory,
        limitations=list(candidate.risks),
    )
    return Artifact(
        id=make_artifact_id(
            "qexec",
            {
                "session_id": session_id,
                "question_id": candidate.question_id,
                "status": "succeeded",
                "execution_mode": "agent",
            },
        ),
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=list(dict.fromkeys([*parent_ids, *evidence_ids])),
        payload=result.model_dump(mode="json"),
    )


def _unique_agent_artifacts(values: Sequence[Any]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Artifact) or value.id in seen:
            continue
        seen.add(value.id)
        artifacts.append(value)
    return artifacts


def _agent_evidence_refs(artifact: Artifact) -> list[EvidenceRef]:
    """Cite an agent artifact with locators the report validator can resolve."""
    # Types the validator resolves no numbers for keep a whole-artifact citation:
    # dropping them would leave the finding with no evidence at all.
    return full_coverage_evidence_refs([artifact]) or [
        EvidenceRef(kind=_agent_evidence_kind(artifact), artifact_id=artifact.id, locator="")
    ]


def _agent_evidence_kind(
    artifact: Artifact,
) -> Literal["sql", "code", "stat", "table", "chart", "profile_field", "artifact"]:
    if artifact.type is ArtifactType.CODE_EXECUTION_RESULT:
        return "code"
    if artifact.type is ArtifactType.CHART_SPEC:
        return "chart"
    return "artifact"


def _execute_template_question(
    candidate: QuestionCandidate,
    *,
    datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
    llm: LLMClient | None,
    preview_rows: int,
    timeout_seconds: float,
    seeds: SemanticSeeds | None = None,
    confirmed_joins: Collection[str] = (),
    on_join_used: Callable[[str], None] | None = None,
) -> list[Artifact]:
    if candidate.sql_template is None:
        return [
            _failed_qexec_artifact(
                question_id=candidate.question_id,
                question=candidate.question_en,
                origin=candidate.origin,
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                error="Template question did not include sql_template.",
                exploratory=candidate.exploratory,
            )
        ]
    # Reject undeclared or unconfirmed joins before execution.
    join_violations = check_sql_joins_declared(
        "sql_template",
        candidate.sql_template,
        required_relations=candidate.required_relations,
        confirmed_joins=confirmed_joins,
    )
    if join_violations:
        guard_error = ToolGuardError("execute_question_candidate", join_violations)
        return [
            _failed_qexec_artifact(
                question_id=candidate.question_id,
                question=candidate.question_en,
                origin=candidate.origin,
                sql=candidate.sql_template,
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                error=guard_error.to_model_feedback()[:1000],
                exploratory=candidate.exploratory,
            )
        ]
    try:
        sql_artifact = run_sql(
            _template_catalog(datasets),
            candidate.sql_template,
            project_id=project_id,
            session_id=session_id,
            preview_rows=preview_rows,
            timeout_seconds=timeout_seconds,
            output_units=candidate.produced_units,
        )
        sql_artifact.parents = list(parent_ids)
        qexec = _successful_qexec_artifact(
            candidate,
            sql_artifact=sql_artifact,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parent_ids, sql_artifact.id],
            plan_summary="Executed deterministic template SQL.",
            llm=llm,
            seeds=seeds,
        )
        # Record successful use of verified join paths.
        if on_join_used is not None:
            for label in candidate.required_relations:
                on_join_used(label)
        return [sql_artifact, qexec]
    except BudgetExceeded:
        raise
    except (UnsafeQueryError, QueryTimeout, RuntimeError, duckdb.Error, ValueError) as exc:
        return [
            _failed_qexec_artifact(
                question_id=candidate.question_id,
                question=candidate.question_en,
                origin=candidate.origin,
                sql=candidate.sql_template,
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
                exploratory=candidate.exploratory,
            )
        ]


def _execute_llm_question(
    candidate: QuestionCandidate,
    *,
    datasets: Sequence[LoadedDataset],
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
    llm: LLMClient | None,
    preview_rows: int,
    timeout_seconds: float,
    seeds: SemanticSeeds | None = None,
    on_guard_rejected: Callable[[ToolGuardError], None] | None = None,
    confirmed_joins: Collection[str] = (),
    on_join_used: Callable[[str], None] | None = None,
) -> list[Artifact]:
    if llm is None or is_offline_client(llm):
        return [
            _failed_qexec_artifact(
                question_id=candidate.question_id,
                question=candidate.question_en,
                origin=candidate.origin,
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                error="LLM client is required for LLM-route question execution.",
                exploratory=candidate.exploratory,
            )
        ]
    catalog = build_catalog(datasets)
    try:
        plan = build_plan(
            candidate.question_en,
            llm=llm,
            catalog_columns=_catalog_columns(datasets, catalog),
            engine=catalog.engine,
            on_guard_rejected=on_guard_rejected,
        )
        if plan.needs_approval:
            return [
                _failed_qexec_artifact(
                    question_id=candidate.question_id,
                    question=candidate.question_en,
                    origin=candidate.origin,
                    sql=plan.sql,
                    project_id=project_id,
                    session_id=session_id,
                    parent_ids=parent_ids,
                    error=(
                        "Automatic execution stopped: the generated plan requires "
                        "explicit user approval. Run it from an approval-aware surface."
                    ),
                    outcome="awaiting_approval",
                    abstention_code="approval_required",
                    exploratory=candidate.exploratory,
                )
            ]
        # Apply the join guard to generated SQL as well as templates.
        join_violations = check_sql_joins_declared(
            "plan.sql",
            plan.sql,
            required_relations=candidate.required_relations,
            confirmed_joins=confirmed_joins,
        )
        if join_violations:
            guard_error = ToolGuardError("execute_question_candidate", join_violations)
            if on_guard_rejected is not None:
                on_guard_rejected(guard_error)
            return [
                _failed_qexec_artifact(
                    question_id=candidate.question_id,
                    question=candidate.question_en,
                    origin=candidate.origin,
                    sql=plan.sql,
                    project_id=project_id,
                    session_id=session_id,
                    parent_ids=parent_ids,
                    error=guard_error.to_model_feedback()[:1000],
                    exploratory=candidate.exploratory,
                )
            ]
        sql_artifact = run_sql(
            catalog,
            plan.sql,
            project_id=project_id,
            session_id=session_id,
            preview_rows=preview_rows,
            timeout_seconds=timeout_seconds,
            output_units=candidate.produced_units,
        )
        sql_artifact.parents = list(parent_ids)
        qexec = _successful_qexec_artifact(
            candidate,
            sql_artifact=sql_artifact,
            project_id=project_id,
            session_id=session_id,
            parent_ids=[*parent_ids, sql_artifact.id],
            plan_summary=f"{plan.method}: {plan.rationale}",
            llm=llm,
            seeds=seeds,
        )
        if on_join_used is not None:
            for label in candidate.required_relations:
                on_join_used(label)
        return [sql_artifact, qexec]
    except BudgetExceeded:
        raise
    except (UnsafeQueryError, QueryTimeout, RuntimeError, duckdb.Error, ValueError) as exc:
        return [
            _failed_qexec_artifact(
                question_id=candidate.question_id,
                question=candidate.question_en,
                origin=candidate.origin,
                project_id=project_id,
                session_id=session_id,
                parent_ids=parent_ids,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
                exploratory=candidate.exploratory,
            )
        ]


def _template_catalog(datasets: Sequence[LoadedDataset]) -> SqlCatalog:
    engine = DuckDBQueryEngine()
    relations: dict[str, str] = {}
    for dataset in datasets:
        relation_name = _relation_name(dataset.record.dataset_id)
        engine.register_frame(relation_name, dataset.frame)
        relations[dataset.record.dataset_id] = relation_name
        relations[dataset.record.name] = relation_name
    return SqlCatalog(engine=engine, relations=relations)


def _catalog_columns(
    datasets: Sequence[LoadedDataset],
    catalog: SqlCatalog,
) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    for dataset in datasets:
        relation_name = catalog.relations[dataset.record.name]
        columns[relation_name] = {str(column) for column in dataset.frame.columns}
    return columns


def _successful_qexec_artifact(
    candidate: QuestionCandidate,
    *,
    sql_artifact: Artifact,
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
    plan_summary: str,
    llm: LLMClient | None = None,
    seeds: SemanticSeeds | None = None,
) -> Artifact:
    contract_failure = _result_contract_failure(candidate, sql_artifact)
    if contract_failure is not None:
        return _failed_qexec_artifact(
            question_id=candidate.question_id,
            question=candidate.question_en,
            origin=candidate.origin,
            sql=SqlResult.model_validate(sql_artifact.payload).sql,
            project_id=project_id,
            session_id=session_id,
            parent_ids=parent_ids,
            error=f"Result contract rejected publication: {contract_failure.reason}",
            outcome="abstained",
            abstention_code=contract_failure.code,
            exploratory=candidate.exploratory,
        )
    findings = _findings(candidate, sql_artifact)
    sql_result = SqlResult.model_validate(sql_artifact.payload)
    basis = _ranking_basis(sql_result.sql, sql_result.rows_preview)
    # Level-1 calibrated interpretation: offline (or no client) returns ``absent``
    # with no model call, so the deterministic auto-exec path is unchanged. Pinned
    # definitions (when present) are injected as fixed context, never a number source.
    interpretation = interpret_findings(
        llm or OfflineLLMClient(),
        question=candidate.question_en,
        findings=findings,
        method_context=plan_summary,
        limitations=candidate.risks,
        seeds=seeds,
        ranking_basis=(
            {"column": basis[0], "direction": basis[1]} if basis is not None else None
        ),
    )
    result = QuestionExecutionResult(
        question_id=candidate.question_id,
        question=candidate.question_en,
        origin=candidate.origin,
        plan_summary=plan_summary,
        sql=sql_result.sql,
        sql_result_artifact_id=sql_artifact.id,
        findings=findings,
        status="succeeded",
        outcome="answered",
        interpretation=interpretation.text,
        interpretation_status=interpretation.status,
        exploratory=candidate.exploratory,
        # Only the whitelist disclosure lines (appended to risks by the callers)
        # are report-facing; ordinary card risks stay on the candidate.
        limitations=[
            risk for risk in candidate.risks if "auto-confirmed (high confidence)" in risk
        ],
    )
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id(
            "qexec",
            {"session_id": session_id, "question_id": candidate.question_id, "status": "succeeded"},
        ),
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=list(parent_ids),
        payload=payload,
    )


def _result_contract_failure(
    candidate: QuestionCandidate, sql_artifact: Artifact
) -> _ResultContractFailure | None:
    """Fail closed when a successful SQL result cannot answer its question."""
    result = SqlResult.model_validate(sql_artifact.payload)
    if not result.rows_preview:
        return _ResultContractFailure("empty_query_result", "query returned no answer rows")
    row = result.rows_preview[0]

    contract_metric_id = (
        candidate.answer_contract.metric_id
        if candidate.answer_contract is not None and candidate.answer_contract.kind == "metric"
        else candidate.metric_id
    )
    definition = metric_definition(contract_metric_id) if contract_metric_id else None
    if definition is None and candidate.template_id == "domain_metric":
        definition = _match_domain_metric_definition(row)
    if definition is not None:
        result_contract = validate_metric_result(definition.metric_id, row)
        if not result_contract.valid:
            return _ResultContractFailure(result_contract.code, result_contract.reason)
        expected_units = (
            candidate.answer_contract.expected_units
            if candidate.answer_contract is not None
            else {}
        )
        if expected_units and expected_units != definition.units:
            return _ResultContractFailure(
                "metric_unit_contract_mismatch",
                "Candidate expected-unit contract does not match the metric registry.",
            )
        for field, expected_unit in expected_units.items():
            produced_unit = result.units.get(field)
            if produced_unit is None:
                return _ResultContractFailure(
                    "metric_unit_metadata_missing",
                    f"Metric field {field!r} has no produced-unit metadata.",
                )
            if not _units_compatible(expected_unit, produced_unit):
                return _ResultContractFailure(
                    "metric_unit_mismatch",
                    f"Metric field {field!r} expected unit {expected_unit!r} "
                    f"but the SQL plan declared {produced_unit!r}.",
                )

    required_tokens = (
        candidate.answer_contract.required_column_tokens
        if candidate.answer_contract is not None and candidate.answer_contract.kind == "threshold"
        else ["threshold"]
        if "threshold" in candidate.question_en.lower()  # legacy artifact compatibility
        else []
    )
    if required_tokens:
        answer_columns = {column.lower() for column in row if column != "row_count"}
        if any(not any(token in column for column in answer_columns) for token in required_tokens):
            return _ResultContractFailure(
                candidate.answer_contract.abstention_code
                if candidate.answer_contract is not None
                else "answer_schema_mismatch",
                "question produced no result column matching its answer contract",
            )
    return None


def _failed_qexec_artifact(
    *,
    question_id: str,
    question: str,
    origin: Literal["template", "llm"],
    project_id: str,
    session_id: str,
    parent_ids: Sequence[str],
    error: str,
    sql: str | None = None,
    outcome: Literal["abstained", "failed", "awaiting_approval"] = "failed",
    abstention_code: str | None = None,
    exploratory: bool = False,
    execution_mode: Literal["pipeline", "agent"] = "pipeline",
    tool_calls: int = 0,
    tool_names: list[str] | None = None,
    evidence_artifact_ids: list[str] | None = None,
) -> Artifact:
    result = QuestionExecutionResult(
        question_id=question_id,
        question=question,
        origin=origin,
        execution_mode=execution_mode,
        tool_calls=tool_calls,
        tool_names=tool_names or [],
        evidence_artifact_ids=evidence_artifact_ids or [],
        sql=sql,
        findings=[],
        status="failed",
        outcome=outcome,
        abstention_code=abstention_code,
        error=error,
        exploratory=exploratory,
    )
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id(
            "qexec",
            {
                "session_id": session_id,
                "question_id": question_id,
                "status": "failed",
                "error": error,
            },
        ),
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=list(parent_ids),
        payload=payload,
    )


def _append_tool_guard_rejection(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    *,
    name: str,
    error: ToolGuardError,
) -> None:
    store.append_trace(
        project_id,
        TraceEvent(
            session_id=session_id,
            event_type="tool_guard_rejected",
            name=name,
            finished_at=datetime.now(UTC),
            summary=error.to_trace_summary(),
        ),
    )


# Question intent determines the answer column and presentation format.
QuestionIntent = Literal["sum", "share", "average", "duration"]

_INTENT_SHARE_WORDS = frozenset({"share", "rate", "percentage", "percent", "proportion"})
_INTENT_AVERAGE_WORDS = frozenset({"average", "avg", "mean"})
_INTENT_SUM_WORDS = frozenset({"total", "sum", "overall"})
_INTENT_DURATION_PHRASES = ("how long", "how many days", "duration")

_SHARE_COLUMN_TOKENS = frozenset(
    {"rate", "share", "percent", "percentage", "pct", "proportion", "ratio"}
)
_AVERAGE_COLUMN_TOKENS = frozenset({"avg", "average", "mean"})
_SUM_COLUMN_TOKENS = frozenset({"total", "sum", "gmv"})
_DURATION_COLUMN_TOKENS = frozenset({"days", "day", "hours", "hour", "minutes", "duration"})
_PERCENT_VALUED_TOKENS = frozenset({"percent", "percentage", "pct"})
# Aggregate-flavored tokens disqualify a column from acting as a share
# numerator ("473 of 97658"): a numerator is a plain count, never a sum/avg.
_NON_COUNT_TOKENS = frozenset(
    {
        "total",
        "sum",
        "avg",
        "average",
        "mean",
        "median",
        "min",
        "max",
        "rate",
        "share",
        "percent",
        "percentage",
        "pct",
        "proportion",
        "ratio",
        "hhi",
    }
)

_INTENT_COLUMN_TOKENS: dict[str, frozenset[str]] = {
    "share": _SHARE_COLUMN_TOKENS,
    "average": _AVERAGE_COLUMN_TOKENS,
    "sum": _SUM_COLUMN_TOKENS,
    "duration": _DURATION_COLUMN_TOKENS,
}


def _column_tokens(column: str) -> set[str]:
    return {token for token in re.split(r"[^0-9a-z]+", column.strip().lower()) if token}


def infer_question_intent(question: str) -> QuestionIntent | None:
    """Deterministically infer what kind of number a question asks for."""
    text = question.lower()
    words = _column_tokens(text)
    if words & _INTENT_SHARE_WORDS:
        return "share"
    if any(phrase in text for phrase in _INTENT_DURATION_PHRASES):
        return "duration"
    if words & _INTENT_AVERAGE_WORDS:
        return "average"
    if words & _INTENT_SUM_WORDS:
        return "sum"
    return None


def intent_metric_column(intent: QuestionIntent, row: dict[str, object]) -> str | None:
    """Pick the result column whose name matches the question intent."""
    tokens = _INTENT_COLUMN_TOKENS[intent]
    for column, value in row.items():
        if column == "row_count":
            continue
        if _column_tokens(column) & tokens and _number(value) is not None:
            return column
    return None


def _format_number(value: float) -> str:
    """Fixed-point text for a cited value; never scientific notation.

    Deliberately NOT the exporter's magnitude-scaled policy: this composes
    `finding.text`, which is persisted as an artifact, and rounding here would
    rewrite stored evidence. The exporter re-renders at display time, so a
    stored "12.4973" still reads "12.5" in the report.
    """
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _is_percent_valued(column: str) -> bool:
    return bool(_column_tokens(column) & _PERCENT_VALUED_TOKENS)


def _share_numerator(row: dict[str, object], share_column: str) -> tuple[str, float, float] | None:
    """Find the (numerator column, numerator, denominator) behind a share."""
    denominator = _number(row.get("row_count"))
    if denominator is None or denominator <= 0:
        return None
    for column, value in row.items():
        if column in {share_column, "row_count"}:
            continue
        numerator = _number(value)
        if numerator is None or numerator != int(numerator):
            continue
        if numerator < 0 or numerator > denominator:
            continue
        if _column_tokens(column) & _NON_COUNT_TOKENS:
            continue
        return column, numerator, denominator
    return None


def _intent_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    row: dict[str, object],
    intent: QuestionIntent,
) -> QuestionFinding | None:
    column = intent_metric_column(intent, row)
    if column is None:
        return None
    value = _number(row.get(column))
    if value is None:
        return None
    if intent == "share":
        return _share_intent_finding(candidate, artifact_id, row, column, value)
    return QuestionFinding(
        text=f"{candidate.question_en} The {column} is {_format_number(value)}.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{column}",
                value=value,
            )
        ],
    )


def _share_intent_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    row: dict[str, object],
    column: str,
    value: float,
) -> QuestionFinding:
    """Build a percentage finding, including its numerator and denominator when available."""
    percent_valued = _is_percent_valued(column)
    rate_text = f"{_format_number(value)}%" if percent_valued else _format_number(value)
    rate_evidence = EvidenceRef(
        kind="sql",
        artifact_id=artifact_id,
        locator=f"rows_preview[0].{column}",
        value=value,
        unit="percent" if percent_valued else "raw",
    )
    pair = _share_numerator(row, column)
    if pair is None:
        return QuestionFinding(
            text=f"{candidate.question_en} The {column} is {rate_text}.",
            evidence=[rate_evidence],
        )
    numerator_column, numerator, denominator = pair
    return QuestionFinding(
        text=(
            f"{candidate.question_en} {_format_number(numerator)} of "
            f"{_format_number(denominator)} rows ({rate_text})."
        ),
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{numerator_column}",
                value=numerator,
            ),
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator="rows_preview[0].row_count",
                value=denominator,
            ),
            rate_evidence,
        ],
    )


_TEMPLATE_PLACEHOLDER = re.compile(r"{(\w+)}")


def _match_domain_metric_definition(
    row: dict[str, object],
) -> MetricDefinition | None:
    """Match a domain-metric SQL result row back to its registry definition."""
    best: MetricDefinition | None = None
    best_specificity = 0
    for definition in DOMAIN_METRIC_REGISTRY:
        names = _TEMPLATE_PLACEHOLDER.findall(definition.interpretation_en)
        if not names:
            continue
        if any(name not in row or _number(row.get(name)) is None for name in names):
            continue
        specificity = sum(1 for name in names if name != "row_count")
        if specificity > best_specificity:
            best = definition
            best_specificity = specificity
    return best


def _domain_metric_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    row: dict[str, object],
    units: dict[str, str],
) -> QuestionFinding | None:
    """Apply a registered metric's interpretation template to SQL results."""
    definition = metric_definition(candidate.metric_id) if candidate.metric_id else None
    if definition is None:
        definition = _match_domain_metric_definition(row)
    if definition is None:
        return None
    template = definition.interpretation_en
    evidence: list[EvidenceRef] = []
    filled = template
    for name in dict.fromkeys(_TEMPLATE_PLACEHOLDER.findall(template)):
        value = _number(row.get(name))
        if value is None:  # pragma: no cover - guarded by the matcher
            return None
        declared_unit = units.get(name)
        if declared_unit is None and f"{{{name}}}%" in template:
            # Historical SqlResult artifacts predate produced-unit metadata.
            declared_unit = "percent"
        rendered = _format_number(value)
        display_unit = currency_unit_display(declared_unit)
        if display_unit is not None:
            rendered = f"{rendered} {display_unit}"
        filled = filled.replace(f"{{{name}}}", rendered)
        unit_classification = classify_currency_unit(declared_unit)
        evidence.append(
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{name}",
                value=value,
                unit=_evidence_unit(declared_unit),
                unit_label=(declared_unit if unit_classification.code is not None else None),
                unit_reference=unit_classification.reference,
            )
        )
    text = f"{candidate.question_en} {filled}"
    # State the measurement basis when using a carrier timestamp.
    note = carrier_basis_note(
        [column for columns in candidate.referenced_columns.values() for column in columns]
    )
    if note is not None and note.strip() not in text:
        text += note
    return QuestionFinding(text=text, evidence=evidence)


def _findings(candidate: QuestionCandidate, sql_artifact: Artifact) -> list[QuestionFinding]:
    findings = _findings_for(candidate, sql_artifact)
    if candidate.exploratory:
        # Preserve exploratory status for downstream validation and disclosure.
        findings = [finding.model_copy(update={"exploratory": True}) for finding in findings]
    return findings


def _findings_for(candidate: QuestionCandidate, sql_artifact: Artifact) -> list[QuestionFinding]:
    result = SqlResult.model_validate(sql_artifact.payload)
    rows = result.rows_preview
    if not rows:
        return []
    # Prefer registered interpretations, then question intent, then generic heuristics.
    if candidate.template_id == "domain_metric":
        finding = _domain_metric_finding(candidate, sql_artifact.id, rows[0], result.units)
        if finding is not None:
            return [finding]
    if candidate.template_id == "trend" and len(rows) >= 2:
        return [_trend_finding(candidate, sql_artifact.id, rows)]
    if candidate.template_id in {"group_difference", "cross_table_aggregation"}:
        return [_ranked_group_finding(candidate, sql_artifact.id, rows, sql=result.sql)]
    if candidate.template_id == "correlation_probe":
        return [_single_value_finding(candidate, sql_artifact.id, rows[0], "pearson")]
    if candidate.template_id == "quality_missing":
        return [_single_value_finding(candidate, sql_artifact.id, rows[0], "missing_percent")]
    # Align single-row answers with the requested metric type.
    intent = infer_question_intent(candidate.question_en)
    if intent is not None and len(rows) == 1:
        finding = _intent_finding(candidate, sql_artifact.id, rows[0], intent)
        if finding is not None:
            return [finding]
    return [_generic_finding(candidate, sql_artifact.id, rows, sql=result.sql)]


def _evidence_unit(unit: str | None) -> Literal["raw", "percent", "currency"]:
    if unit == "percent":
        return "percent"
    if unit in {"currency", "currency_per_order"} or (
        unit is not None
        and (
            re.fullmatch(r"[A-Z]{3}", unit) is not None
            or re.fullmatch(r"[A-Z]{3}/order", unit) is not None
        )
    ):
        return "currency"
    return "raw"


def _units_compatible(expected: str, produced: str) -> bool:
    if produced == expected:
        return True
    if expected == "currency":
        return re.fullmatch(r"[A-Z]{3}", produced) is not None
    if expected == "currency_per_order":
        return re.fullmatch(r"[A-Z]{3}/order", produced) is not None
    return False


def _trend_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    rows: list[dict[str, object]],
) -> QuestionFinding:
    # Prefer the trend metric that matches the question's intent.
    intent = infer_question_intent(candidate.question_en)
    metric_column = (
        (intent_metric_column(intent, rows[0]) if intent is not None else None)
        or _first_prefixed_column(rows[0], "avg_")
        or _first_numeric_column(rows[0])
    )
    if metric_column is None:
        return _generic_finding(candidate, artifact_id, rows)
    start_row = rows[0]
    end_row = rows[-1]
    start = _number(start_row.get(metric_column))
    end = _number(end_row.get(metric_column))
    if start is None or end is None:
        return _generic_finding(candidate, artifact_id, rows)
    direction = "increased" if end > start else "decreased" if end < start else "stayed flat"
    return QuestionFinding(
        text=(
            f"{candidate.question_en} The metric {direction} from {start:g} "
            f"to {end:g} across the returned periods."
        ),
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{metric_column}",
                value=start,
            ),
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[{len(rows) - 1}].{metric_column}",
                value=end,
            ),
        ],
    )


def _ranked_group_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    rows: list[dict[str, object]],
    *,
    sql: str = "",
) -> QuestionFinding:
    top = rows[0]
    group_column = _first_group_column(top)
    intent = infer_question_intent(candidate.question_en)
    metric_column = (
        (intent_metric_column(intent, top) if intent is not None else None)
        or _first_prefixed_column(top, "total_")
        or _first_prefixed_column(top, "avg_")
    )
    if group_column is None or metric_column is None:
        return _generic_finding(candidate, artifact_id, rows, sql=sql)
    return _ranked_finding(
        candidate,
        artifact_id,
        rows,
        label_column=group_column,
        metric_column=metric_column,
        sql=sql,
    )


def _single_value_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    row: dict[str, object],
    column: str,
) -> QuestionFinding:
    value = _number(row.get(column))
    if value is None:
        return _generic_finding(candidate, artifact_id, [row])
    return QuestionFinding(
        text=f"{candidate.question_en} The returned {column} is {value:g}.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{column}",
                value=value,
            )
        ],
    )


def _generic_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    rows: list[dict[str, object]],
    *,
    sql: str = "",
) -> QuestionFinding:
    """Summarize a free-form SQL result by reading the whole ranked table."""
    top = rows[0]
    # Use generic ranking heuristics only when intent does not identify a column.
    intent = infer_question_intent(candidate.question_en)
    metric_column = (
        intent_metric_column(intent, top) if intent is not None else None
    ) or _rank_metric_column(top)
    if metric_column is None:
        return QuestionFinding(text=f"{candidate.question_en} SQL returned result rows.")
    label_column = _label_column(top, metric_column)
    if len(rows) >= 2:
        return _ranked_finding(
            candidate,
            artifact_id,
            rows,
            label_column=label_column,
            metric_column=metric_column,
            sql=sql,
        )
    value = _number(top.get(metric_column))
    if value is None:
        return QuestionFinding(text=f"{candidate.question_en} SQL returned result rows.")
    where = f" for {top.get(label_column)}" if label_column is not None else ""
    return QuestionFinding(
        text=f"{candidate.question_en} The returned {metric_column} is {value:g}{where}.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[0].{metric_column}",
                value=value,
            )
        ],
    )


def _ranked_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    rows: list[dict[str, object]],
    *,
    label_column: str | None,
    metric_column: str,
    sql: str = "",
) -> QuestionFinding:
    """Claim a ranking only on an ordering the executed SQL actually declares.

    2026-07-24 audit: row order used to be narrated as strength ("The strongest
    is <first row>") even with no ORDER BY, and the cited metric was a column
    heuristic that could pick a group key or a sample size.
    """
    if label_column is not None:
        ranked = [
            (row.get(label_column), value)
            for row in rows
            if (value := _number(row.get(metric_column))) is not None
        ]
        if ranked:
            share_finding = _share_finding(
                candidate,
                artifact_id,
                ranked,
                label_column=label_column,
                metric_column=metric_column,
                sql=sql,
            )
            if share_finding is not None:
                return share_finding
    basis = _ranking_basis(sql, rows)
    if basis is None:
        return _unranked_rows_finding(candidate, artifact_id, rows, sql=sql)
    basis_column, direction = basis
    leaders = rows[:3]
    values = [value for row in leaders if (value := _number(row.get(basis_column))) is not None]
    # Keep percentage display and evidence units aligned.
    percent_valued = _is_percent_valued(basis_column)
    suffix = "%" if percent_valued else ""
    evidence = [
        EvidenceRef(
            kind="sql",
            artifact_id=artifact_id,
            locator=f"rows_preview[{index}].{basis_column}",
            value=value,
            unit="percent" if percent_valued else "raw",
        )
        for index, value in enumerate(values)
    ]
    label_columns = [
        column for column in _group_by_columns(sql, rows[0]) if column != basis_column
    ]
    if not label_columns and label_column is not None and label_column != basis_column:
        label_columns = [label_column]
    if label_columns:
        labels = [_row_label(row, label_columns) for row in leaders]
        if len(set(labels)) < len(labels):
            # Indistinguishable labels cannot support a ranked claim.
            return _unranked_rows_finding(candidate, artifact_id, rows, sql=sql)
        for index, row in enumerate(leaders):
            for column in label_columns:
                label_value = _number(row.get(column))
                if label_value is not None:
                    evidence.append(
                        EvidenceRef(
                            kind="sql",
                            artifact_id=artifact_id,
                            locator=f"rows_preview[{index}].{column}",
                            value=label_value,
                        )
                    )
        parts = [
            f"{label} ({basis_column} {value:.4g}{suffix})"
            for label, value in zip(labels, values, strict=False)
        ]
        lead_in = "top is" if direction == "descending" else "smallest is"
        body = f"{lead_in} {parts[0]}"
        if len(parts) > 1:
            body += f", followed by {' and '.join(parts[1:])}"
        all_values = [
            value for row in rows if (value := _number(row.get(basis_column))) is not None
        ]
        cluster = _cluster_note(all_values) if direction == "descending" else ""
        text = f"{candidate.question_en} Ranked by {basis_column} {direction}: {body}{cluster}."
    else:
        noun = "top" if direction == "descending" else "smallest"
        text = (
            f"{candidate.question_en} Ranked by {basis_column} {direction}: the {noun} "
            f"{basis_column} is {values[0]:.4g}{suffix} (first of {len(rows)} ranked results)."
        )
    return QuestionFinding(text=text, evidence=evidence)


_ORDER_BY_KEYWORD = re.compile(r"\border\s+by\b", re.IGNORECASE)
_GROUP_BY_KEYWORD = re.compile(r"\bgroup\s+by\b", re.IGNORECASE)
# A sort/group item this module trusts: a bare, possibly qualified or quoted,
# column name or a 1-based select position. Anything else (expressions, CASE,
# function calls) fails the parse on purpose — degrading beats fabricating.
_PLAIN_IDENTIFIER = r'(?:[A-Za-z_]\w*\.)?(?:"[^"]+"|[A-Za-z_]\w*|\d+)'
_SORT_ITEM = rf"{_PLAIN_IDENTIFIER}(?:\s+(?:asc|desc))?(?:\s+nulls\s+(?:first|last))?"
_ORDER_BY_TAIL = re.compile(
    rf"^\s*(?P<first>{_PLAIN_IDENTIFIER})(?:\s+(?P<direction>asc|desc))?"
    rf"(?:\s+nulls\s+(?:first|last))?"
    rf"(?:\s*,\s*{_SORT_ITEM})*"
    rf"(?:\s+limit\s+\d+)?(?:\s+offset\s+\d+)?\s*;?\s*$",
    re.IGNORECASE,
)
_GROUP_ITEM = re.compile(rf"^{_PLAIN_IDENTIFIER}$")


def _unquote_identifier(token: str) -> str:
    token = re.sub(r"^[A-Za-z_]\w*\.", "", token)
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def _parse_order_by(sql: str) -> tuple[str, str] | None:
    """Primary sort key and direction from the SQL's single top-level ORDER BY.

    Conservative: returns ``None`` for zero or multiple ORDER BY occurrences
    (subqueries, window functions) or any clause shape beyond plain
    column/position keys with optional ASC/DESC, NULLS, LIMIT and OFFSET.
    """
    if not sql:
        return None
    matches = list(_ORDER_BY_KEYWORD.finditer(sql))
    if len(matches) != 1:
        return None
    tail = _ORDER_BY_TAIL.match(sql[matches[0].end() :])
    if tail is None:
        return None
    direction = "descending" if (tail.group("direction") or "").lower() == "desc" else "ascending"
    return _unquote_identifier(tail.group("first")), direction


def _ranking_basis(sql: str, rows: list[dict[str, object]]) -> tuple[str, str] | None:
    """Resolve the ORDER BY key against the result and verify it holds.

    Returns ``None`` when the key is absent from the result columns, any value
    is non-numeric, the preview rows contradict the parsed direction (the parse
    is then presumed wrong), or the key is itself a group-by column — sorting
    by the label proves nothing about magnitudes.
    """
    parsed = _parse_order_by(sql)
    if parsed is None or not rows:
        return None
    name, direction = parsed
    columns = list(rows[0].keys())
    if name.isdigit():
        index = int(name) - 1
        if not 0 <= index < len(columns):
            return None
        column = columns[index]
    else:
        column = next((c for c in columns if c.lower() == name.lower()), None)
        if column is None:
            return None
    values = [_number(row.get(column)) for row in rows]
    numbers = [value for value in values if value is not None]
    if len(numbers) != len(values):
        return None
    pairs = zip(numbers, numbers[1:], strict=False)
    if direction == "descending":
        if any(previous < following for previous, following in pairs):
            return None
    elif any(previous > following for previous, following in pairs):
        return None
    if column in _group_by_columns(sql, rows[0]):
        return None
    return column, direction


def _group_by_columns(sql: str, row: dict[str, object]) -> list[str]:
    """Best-effort group-key list; empty when the clause is not a plain column list."""
    if not sql:
        return []
    matches = list(_GROUP_BY_KEYWORD.finditer(sql))
    if len(matches) != 1:
        return []
    tail = re.split(
        r"\b(?:order\s+by|having|limit|offset|qualify|window)\b",
        sql[matches[0].end() :],
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    columns = list(row.keys())
    resolved: list[str] = []
    for part in tail.split(","):
        token = part.strip().rstrip(";").strip()
        if _GROUP_ITEM.match(token) is None:
            return []
        name = _unquote_identifier(token)
        if name.isdigit():
            index = int(name) - 1
            if not 0 <= index < len(columns):
                return []
            resolved.append(columns[index])
            continue
        column = next((c for c in columns if c.lower() == name.lower()), None)
        if column is not None:
            resolved.append(column)
    return resolved


def _row_label(row: dict[str, object], columns: list[str]) -> str:
    if len(columns) == 1:
        return str(row.get(columns[0]))
    return " / ".join(f"{column}={row.get(column)}" for column in columns)


def _unranked_rows_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    rows: list[dict[str, object]],
    *,
    sql: str = "",
) -> QuestionFinding:
    """Neutral multi-row summary: the SQL proves no ordering, so none is claimed."""
    group_columns = _group_by_columns(sql, rows[0])
    evidence = [
        EvidenceRef(
            kind="sql",
            artifact_id=artifact_id,
            locator="derived: len(rows_preview)",
            value=len(rows),
        )
    ]
    cells: list[str] = []
    for column, value in rows[0].items():
        number = _number(value)
        if number is None:
            cells.append(f"{column}={value}")
        else:
            cells.append(f"{column}={_format_number(number)}")
            evidence.append(
                EvidenceRef(
                    kind="sql",
                    artifact_id=artifact_id,
                    locator=f"rows_preview[0].{column}",
                    value=number,
                )
            )
    grouped = f" grouped by {', '.join(group_columns)}" if group_columns else ""
    text = (
        f"{candidate.question_en} Returned {len(rows)} rows{grouped} "
        f"(no ranking basis in the SQL); example row: {', '.join(cells)}."
    )
    return QuestionFinding(text=text, evidence=evidence)


# A ranked result whose metric is a row count over few labels is a distribution,
# not a magnitude ranking: "the strongest is No (total_employees 1233)" answers
# nothing, while "16.12% (237 of 1,470)" is the answer (2026-07-22 audit).
_MAX_DISTRIBUTION_LABELS = 6


def _is_count_metric(metric_column: str, sql: str) -> bool:
    """Whether the SQL produced this column with COUNT — the query, not the name.

    Column names lie in both directions (``total_employees`` is a count,
    ``total_revenue`` is a sum), so summing them into a share is only safe when
    the aggregate says so.
    """
    if not sql:
        return False
    pattern = rf"count\s*\([^)]*\)\s*(?:as\s+)?{re.escape(metric_column)}\b"
    return re.search(pattern, sql, flags=re.IGNORECASE) is not None


def _share_finding(
    candidate: QuestionCandidate,
    artifact_id: str,
    ranked: list[tuple[object, float]],
    *,
    label_column: str,
    metric_column: str,
    sql: str,
) -> QuestionFinding | None:
    """Phrase a small count distribution as shares, with the shares as evidence."""
    if not _is_count_metric(metric_column, sql):
        return None
    if not 2 <= len(ranked) <= _MAX_DISTRIBUTION_LABELS:
        return None
    total = sum(value for _label, value in ranked)
    if total <= 0 or any(value < 0 for _label, value in ranked):
        return None
    evidence: list[EvidenceRef] = []
    parts: list[str] = []
    for index, (label, value) in enumerate(ranked):
        share = round(value / total * 100, 2)
        parts.append(f"{label} {share:g}% ({value:,.0f})")
        evidence.append(
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=f"rows_preview[{index}].{metric_column}",
                value=value,
            )
        )
        evidence.append(
            EvidenceRef(
                kind="sql",
                artifact_id=artifact_id,
                locator=(
                    f"derived: rows_preview[{index}].{metric_column}"
                    f" / sum(rows_preview[*].{metric_column})"
                ),
                value=share,
                unit="percent",
            )
        )
    evidence.append(
        EvidenceRef(
            kind="sql",
            artifact_id=artifact_id,
            locator=f"derived: sum(rows_preview[*].{metric_column})",
            value=total,
        )
    )
    text = (
        f"{candidate.question_en} Across {total:,.0f} rows the split is "
        f"{', '.join(parts)} by {label_column}."
    )
    return QuestionFinding(text=text, evidence=evidence)


def _cluster_note(values: list[float]) -> str:
    """Flag (in words, no orphan numbers) when the top results bunch together."""
    if len(values) < 3:
        return ""
    top = values[0]
    if top <= 0 or any(value < 0 for value in values[:3]):
        return ""
    within = sum(1 for value in values if value >= 0.8 * top)
    if within >= 3:
        return ", with the leading results closely bunched"
    return ""


def _rank_metric_column(row: dict[str, object]) -> str | None:
    for prefix in ("total_", "avg_"):
        column = _first_prefixed_column(row, prefix)
        if column is not None:
            return column
    for column, value in row.items():
        if column == "row_count":
            continue
        if _number(value) is not None:
            return column
    return None


def _label_column(row: dict[str, object], metric_column: str) -> str | None:
    for column, value in row.items():
        if column == metric_column:
            continue
        if _number(value) is None:
            return column
    return None


def _first_group_column(row: dict[str, object]) -> str | None:
    for column in row:
        if column not in {"row_count"} and not column.startswith(("avg_", "total_")):
            return column
    return None


def _first_prefixed_column(row: dict[str, object], prefix: str) -> str | None:
    for column in row:
        if column.startswith(prefix):
            return column
    return None


def _first_numeric_column(row: dict[str, object]) -> str | None:
    for column, value in row.items():
        if _number(value) is not None:
            return column
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _source_question_candidate_set(source_artifacts: Sequence[Artifact]) -> Artifact:
    for artifact in source_artifacts:
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET:
            return artifact
    raise ValueError("Source session does not contain a QuestionCandidateSet artifact.")


def _load_source_datasets(
    store: ArtifactStore,
    *,
    project_id: str,
    source_session_id: str,
    source_artifacts: Sequence[Artifact],
) -> list[LoadedDataset]:
    del source_session_id
    datasets: list[LoadedDataset] = []
    for artifact in source_artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        profile = DatasetProfile.model_validate(artifact.payload)
        path = store.project_dir(project_id) / "uploads" / profile.dataset_id / "v1" / profile.name
        if not path.exists():
            matches = list(
                (store.project_dir(project_id) / "uploads").glob(
                    f"{profile.dataset_id}/**/{profile.name}"
                )
            )
            if not matches:
                raise FileNotFoundError(f"Could not reload source dataset: {profile.name}")
            path = matches[0]
        datasets.append(load_csv(path, dataset_id=profile.dataset_id))
    return datasets


def _regenerate_report(
    artifacts: list[Artifact],
    *,
    project_id: str,
    session_id: str,
    business_context: str,
    llm: LLMClient,
    payload_policy: PayloadPolicy,
) -> list[Artifact]:
    report = generate_agentic_report(
        artifacts,
        project_id=project_id,
        session_id=session_id,
        business_context=business_context,
        llm=llm,
        payload_policy=payload_policy,
    )
    return build_agentic_report_artifacts(
        report,
        artifacts,
        project_id=project_id,
        session_id=session_id,
        payload_policy=payload_policy,
    )


def _write_report_files(
    store: ArtifactStore,
    project_id: str,
    session_id: str,
    artifacts: Sequence[Artifact],
) -> None:
    report = next(
        (artifact for artifact in artifacts if artifact.type is ArtifactType.MARKDOWN_REPORT),
        None,
    )
    if report is not None:
        store.write_session_text(
            project_id,
            session_id,
            "report/report.md",
            str(report.payload["markdown"]),
        )
    html = next(
        (artifact for artifact in artifacts if artifact.type is ArtifactType.HTML_REPORT),
        None,
    )
    if html is not None:
        store.write_session_text(
            project_id,
            session_id,
            "report/report.html",
            str(html.payload["html"]),
        )


def _qexec_status(artifacts: Sequence[Artifact]) -> str:
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            return str(artifact.payload.get("status", "unknown"))
    return "unknown"


def _qexec_outcome(artifacts: Sequence[Artifact]) -> str:
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            status = str(artifact.payload.get("status", "failed"))
            return str(
                artifact.payload.get("outcome", "answered" if status == "succeeded" else "failed")
            )
    return "unknown"


def _qexec_abstention_code(artifacts: Sequence[Artifact]) -> str | None:
    for artifact in artifacts:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            code = artifact.payload.get("abstention_code")
            return code if isinstance(code, str) and code else None
    return None


def _generate_batch_session_id(source_session_id: str, question_ids: Sequence[str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = stable_hash(
        {"source_session_id": source_session_id, "question_ids": list(question_ids)},
        length=6,
    )
    return f"qsess_{stamp}_{suffix}"


def generate_batch_session_id(source_session_id: str, question_ids: Sequence[str]) -> str:
    """Generate a batch run ID before execution so callers can prepare run-scoped resources."""
    return _generate_batch_session_id(source_session_id, question_ids)


def _current_code_version() -> str:
    """Return the local build marker without spawning a Git subprocess."""
    return "local"

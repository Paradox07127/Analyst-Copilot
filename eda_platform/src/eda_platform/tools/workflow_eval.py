"""Evaluate whole-workflow quality and cost from persisted artifacts."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from statistics import mean

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.questions import QuestionExecutionResult
from eda_platform.schemas.reports import ReportBundle
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.workflow_eval import (
    ExpectedAbstention,
    ExpectedAnswer,
    SemanticEscape,
    WorkflowEvalCase,
    WorkflowEvalComparison,
    WorkflowEvalConversion,
    WorkflowEvalEnvironment,
    WorkflowEvalFailureNode,
    WorkflowEvalHardGateResult,
    WorkflowEvalResult,
    WorkflowEvalScore,
    WorkflowEvalSpec,
    WorkflowEvalStep,
    WorkflowEvalStepDAG,
    WorkflowEvalSuiteResult,
    WorkflowEvalTrialManifest,
    WorkflowEvalUsage,
    WorkflowEvalUsageTotals,
)


def convert_workflow_eval_spec(
    spec: WorkflowEvalSpec,
    *,
    environment: WorkflowEvalEnvironment | None = None,
    case_id: str | None = None,
    trial_id: str | None = None,
    repetition: int = 1,
    dataset_fingerprints: dict[str, str] | None = None,
) -> WorkflowEvalConversion:
    """Losslessly split a legacy spec into case, environment, and trial identity."""
    resolved_case_id = case_id or stable_hash(
        {"name": spec.name, "case_version": spec.case_version}, length=24
    )
    case = WorkflowEvalCase(
        case_id=resolved_case_id,
        case_version=spec.case_version,
        name=spec.name,
        description=spec.description,
        dataset_refs=spec.input_files,
        business_context=spec.business_context,
        probe_questions=spec.probe_questions,
        baseline_policy=spec.baseline_policy,
        expected_dataset_count=spec.expected_dataset_count,
        expected_answers=spec.expected_answers,
        expected_abstentions=spec.expected_abstentions,
        required_executive_summary_patterns=spec.required_executive_summary_patterns,
        forbidden_output_patterns=spec.forbidden_output_patterns,
        min_answer_precision=spec.min_answer_precision,
        min_answer_recall=spec.min_answer_recall,
        min_abstention_precision=spec.min_abstention_precision,
        min_abstention_recall=spec.min_abstention_recall,
        min_report_dataset_coverage=spec.min_report_dataset_coverage,
        min_quality_dataset_coverage=spec.min_quality_dataset_coverage,
        min_executive_summary_recall=spec.min_executive_summary_recall,
        min_stability_rate=spec.min_stability_rate,
        max_semantic_escape_rate=spec.max_semantic_escape_rate,
        max_failures=spec.max_failures,
        max_tokens=spec.max_tokens,
        max_duration_seconds=spec.max_duration_seconds,
        source_spec_schema_version=spec.schema_version,
    )
    resolved_environment = environment or WorkflowEvalEnvironment()
    case_fingerprint = stable_hash(case.model_dump(mode="json"), length=32)
    environment_fingerprint = stable_hash(resolved_environment.model_dump(mode="json"), length=32)
    resolved_trial_id = trial_id or stable_hash(
        {
            "case_fingerprint": case_fingerprint,
            "environment_fingerprint": environment_fingerprint,
            "repetition": repetition,
        },
        length=32,
    )
    manifest = WorkflowEvalTrialManifest(
        trial_id=resolved_trial_id,
        case_id=case.case_id,
        case_version=case.case_version,
        case_fingerprint=case_fingerprint,
        environment_fingerprint=environment_fingerprint,
        repetition=repetition,
        dataset_fingerprints=dataset_fingerprints or {},
    )
    return WorkflowEvalConversion(
        case=case,
        environment=resolved_environment,
        manifest=manifest,
    )


def restore_workflow_eval_spec(case: WorkflowEvalCase) -> WorkflowEvalSpec:
    """Restore the exact legacy spec represented by a converted case."""
    return WorkflowEvalSpec(
        schema_version=case.source_spec_schema_version,
        case_version=case.case_version,
        name=case.name,
        description=case.description,
        input_files=case.dataset_refs,
        business_context=case.business_context,
        probe_questions=case.probe_questions,
        baseline_policy=case.baseline_policy,
        expected_dataset_count=case.expected_dataset_count,
        expected_answers=case.expected_answers,
        expected_abstentions=case.expected_abstentions,
        required_executive_summary_patterns=case.required_executive_summary_patterns,
        forbidden_output_patterns=case.forbidden_output_patterns,
        min_answer_precision=case.min_answer_precision,
        min_answer_recall=case.min_answer_recall,
        min_abstention_precision=case.min_abstention_precision,
        min_abstention_recall=case.min_abstention_recall,
        min_report_dataset_coverage=case.min_report_dataset_coverage,
        min_quality_dataset_coverage=case.min_quality_dataset_coverage,
        min_executive_summary_recall=case.min_executive_summary_recall,
        min_stability_rate=case.min_stability_rate,
        max_semantic_escape_rate=case.max_semantic_escape_rate,
        max_failures=case.max_failures,
        max_tokens=case.max_tokens,
        max_duration_seconds=case.max_duration_seconds,
    )


def grade_workflow_hard_gates(
    *,
    step_dag: WorkflowEvalStepDAG,
    usage: WorkflowEvalUsage,
    available_artifact_refs: Iterable[str],
    available_evidence_refs: Iterable[str],
) -> WorkflowEvalHardGateResult:
    """Run deterministic graph/reference and spend-reconciliation hard gates."""
    graph_result = grade_workflow_step_dag(
        step_dag,
        available_artifact_refs=available_artifact_refs,
        available_evidence_refs=available_evidence_refs,
    )
    usage_score, usage_failures = reconcile_workflow_usage(usage)
    failures = [*graph_result.failure_nodes, *usage_failures]
    return WorkflowEvalHardGateResult(
        passed=not failures,
        scores=[*graph_result.scores, usage_score],
        failure_nodes=failures,
    )


def grade_workflow_step_dag(
    step_dag: WorkflowEvalStepDAG,
    *,
    available_artifact_refs: Iterable[str],
    available_evidence_refs: Iterable[str],
) -> WorkflowEvalHardGateResult:
    """Validate step dependencies and all artifact/evidence references."""
    artifact_refs = set(available_artifact_refs)
    evidence_refs = set(available_evidence_refs)
    nodes_by_id = {step.node_id: step for step in step_dag.steps}
    failures: list[WorkflowEvalFailureNode] = []

    if not step_dag.steps:
        failures.append(
            _eval_failure(
                "__dag__",
                "missing_step_dag",
                "At least one observed workflow step is required.",
            )
        )

    if len(nodes_by_id) != len(step_dag.steps):
        duplicate_ids = sorted(
            node_id
            for node_id, count in Counter(step.node_id for step in step_dag.steps).items()
            if count > 1
        )
        failures.extend(
            _eval_failure(node_id, "duplicate_node_id", "Step node IDs must be unique.")
            for node_id in duplicate_ids
        )

    for step in step_dag.steps:
        if step.status == "succeeded" and not step.artifact_refs:
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_step_artifact",
                    "A succeeded workflow step must reference its output artifact.",
                    depends_on=step.depends_on,
                )
            )
        if (
            step.status == "succeeded"
            and step.kind in {"finding", "report_claim"}
            and not step.evidence_refs
        ):
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_step_evidence",
                    "Succeeded finding/report nodes must reference evidence.",
                    depends_on=step.depends_on,
                )
            )
        missing_dependencies = [
            dependency for dependency in step.depends_on if dependency not in nodes_by_id
        ]
        if missing_dependencies:
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_dependency",
                    "Missing dependency refs: " + ", ".join(sorted(missing_dependencies)),
                    depends_on=step.depends_on,
                )
            )
        missing_artifacts = sorted(set(step.artifact_refs) - artifact_refs)
        if missing_artifacts:
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_artifact_ref",
                    "Missing artifact refs: " + ", ".join(missing_artifacts),
                    depends_on=step.depends_on,
                )
            )
        missing_evidence = sorted(set(step.evidence_refs) - evidence_refs)
        if missing_evidence:
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_evidence_ref",
                    "Missing evidence refs: " + ", ".join(missing_evidence),
                    depends_on=step.depends_on,
                )
            )
        if step.status == "failed":
            failures.append(
                _eval_failure(
                    step.node_id,
                    "step_failed",
                    step.failure_reason or "Step failed without a typed reason.",
                    depends_on=step.depends_on,
                )
            )
        failed_dependencies = [
            dependency
            for dependency in step.depends_on
            if dependency in nodes_by_id and nodes_by_id[dependency].status in {"failed", "skipped"}
        ]
        if step.status == "succeeded" and failed_dependencies:
            failures.append(
                _eval_failure(
                    step.node_id,
                    "succeeded_after_failed_dependency",
                    "Succeeded with failed/skipped dependencies: "
                    + ", ".join(sorted(failed_dependencies)),
                    depends_on=step.depends_on,
                )
            )

    for node_id in _cyclic_node_ids(nodes_by_id):
        failures.append(
            _eval_failure(
                node_id,
                "dependency_cycle",
                "Step dependency graph must be acyclic.",
                depends_on=nodes_by_id[node_id].depends_on,
            )
        )

    depended_on = {dependency for step in step_dag.steps for dependency in step.depends_on}
    terminal_steps = [step for step in step_dag.steps if step.node_id not in depended_on]
    if step_dag.steps and not any(
        (step.status == "succeeded" and step.kind in {"finding", "report_claim", "validation"})
        or (step.status == "abstained" and step.kind == "question")
        for step in terminal_steps
    ):
        failures.append(
            _eval_failure(
                "__dag__",
                "missing_terminal_outcome",
                "The workflow graph must end in a supported outcome or typed abstention.",
            )
        )

    codes = sorted({failure.code for failure in failures})
    score = WorkflowEvalScore(
        name="step_dag_contract",
        value=0.0 if failures else 1.0,
        passed=not failures,
        failure_codes=codes,
        details={"nodes_evaluated": len(step_dag.steps)},
    )
    return WorkflowEvalHardGateResult(
        passed=not failures,
        scores=[score],
        failure_nodes=failures,
    )


def reconcile_workflow_usage(
    usage: WorkflowEvalUsage,
) -> tuple[WorkflowEvalScore, list[WorkflowEvalFailureNode]]:
    """Require ledger, budget, and metrics to report identical usage totals."""
    sources = {
        "ledger": usage.ledger,
        "budget": usage.budget,
        "metrics": usage.metrics,
    }
    failures = [
        _eval_failure(
            "__usage__",
            f"missing_{name}_usage",
            f"{name} usage totals are required for reconciliation.",
        )
        for name, totals in sources.items()
        if totals is None
    ]
    present = {name: totals for name, totals in sources.items() if totals is not None}
    if len(present) == len(sources):
        call_values = {totals.llm_calls for totals in present.values()}
        token_values = {totals.total_tokens for totals in present.values()}
        if len(call_values) != 1:
            failures.append(
                _eval_failure(
                    "__usage__",
                    "llm_calls_mismatch",
                    _usage_values_message(present, "llm_calls"),
                )
            )
        if len(token_values) != 1:
            failures.append(
                _eval_failure(
                    "__usage__",
                    "total_tokens_mismatch",
                    _usage_values_message(present, "total_tokens"),
                )
            )
        costs = [totals.estimated_cost_usd for totals in present.values()]
        llm_calls = next(iter(call_values)) if len(call_values) == 1 else max(call_values)
        some_cost_unknown = any(cost is None for cost in costs)
        all_costs_unknown = all(cost is None for cost in costs)
        if some_cost_unknown and (llm_calls > 0 or not all_costs_unknown):
            failures.append(
                _eval_failure(
                    "__usage__",
                    "estimated_cost_unknown",
                    "Cost known/unknown state must match across all usage sources.",
                )
            )
        elif not some_cost_unknown:
            rounded_costs = {round(cost, 9) for cost in costs if cost is not None}
            if len(rounded_costs) != 1:
                failures.append(
                    _eval_failure(
                        "__usage__",
                        "estimated_cost_mismatch",
                        _usage_values_message(present, "estimated_cost_usd"),
                    )
                )

        expected_calls = next(iter(call_values)) if len(call_values) == 1 else None
        budget_status_fields = {
            "budget_reserved_calls": usage.budget_reserved_calls,
            "budget_settled_calls": usage.budget_settled_calls,
            "budget_rejected_calls": usage.budget_rejected_calls,
            "budget_uncertain_calls": usage.budget_uncertain_calls,
        }
        if any(value is None for value in budget_status_fields.values()):
            failures.append(
                _eval_failure(
                    "__usage__",
                    "missing_budget_status",
                    "SessionMetrics v4 budget lifecycle fields are required for every trial.",
                )
            )
        else:
            assert usage.budget_reserved_calls is not None
            assert usage.budget_settled_calls is not None
            assert usage.budget_rejected_calls is not None
            assert usage.budget_uncertain_calls is not None
            if (
                usage.budget_reserved_calls != expected_calls
                or usage.budget_settled_calls != expected_calls
            ):
                failures.append(
                    _eval_failure(
                        "__usage__",
                        "budget_call_lifecycle_mismatch",
                        "Reserved and settled budget calls must equal ledger calls.",
                    )
                )
            if usage.budget_rejected_calls:
                failures.append(
                    _eval_failure(
                        "__usage__",
                        "budget_call_rejected",
                        "A workflow-eval trial may not contain rejected provider calls.",
                    )
                )
            if usage.budget_uncertain_calls:
                failures.append(
                    _eval_failure(
                        "__usage__",
                        "budget_call_uncertain",
                        "A workflow-eval trial may not contain uncertain provider calls.",
                    )
                )
            accepted_reconciliation = (
                {"verified", "not_applicable"} if expected_calls == 0 else {"verified"}
            )
            if usage.budget_reconciliation not in accepted_reconciliation:
                failures.append(
                    _eval_failure(
                        "__usage__",
                        "budget_reconciliation_unverified",
                        "SessionMetrics budget reconciliation must be verified or "
                        "explicitly not applicable for a zero-call trial.",
                    )
                )

    failure_codes = sorted({failure.code for failure in failures})
    score = WorkflowEvalScore(
        name="usage_reconciliation",
        value=0.0 if failures else 1.0,
        passed=not failures,
        failure_codes=failure_codes,
        details={"sources_present": sorted(present)},
    )
    return score, failures


def evaluate_workflow_run(
    artifacts: Iterable[Artifact],
    spec: WorkflowEvalSpec,
) -> WorkflowEvalResult:
    """Evaluate one persisted run against a deterministic case specification."""
    artifact_list = list(artifacts)
    profiles = [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in artifact_list
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    questions = [
        QuestionExecutionResult.model_validate(artifact.payload)
        for artifact in artifact_list
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    report_artifacts = [
        artifact for artifact in artifact_list if artifact.type is ArtifactType.REPORT_BUNDLE
    ]
    report = (
        ReportBundle.model_validate(max(report_artifacts, key=lambda item: item.created_at).payload)
        if report_artifacts
        else None
    )
    metrics_artifacts = [
        artifact for artifact in artifact_list if artifact.type is ArtifactType.SESSION_METRICS
    ]
    session_id = artifact_list[0].session_id if artifact_list else ""
    metrics = (
        SessionMetrics.model_validate(
            max(metrics_artifacts, key=lambda item: item.created_at).payload
        )
        if metrics_artifacts
        else SessionMetrics(session_id=session_id)
    )

    answered = [question for question in questions if question.outcome == "answered"]
    abstained = [question for question in questions if question.outcome == "abstained"]
    failed = [question for question in questions if question.outcome == "failed"]

    artifact_ids = {artifact.id for artifact in artifact_list}
    answer_match_count = _maximum_match_count(
        [
            [
                index
                for index, expected in enumerate(spec.expected_answers)
                if _matches_expected_answer(question, expected, artifact_ids)
            ]
            for question in answered
        ],
        expected_count=len(spec.expected_answers),
    )
    answer_precision = _precision(answer_match_count, len(answered), spec.expected_answers)
    answer_recall = _recall(answer_match_count, len(spec.expected_answers))

    abstention_match_count = _maximum_match_count(
        [
            [
                index
                for index, expected in enumerate(spec.expected_abstentions)
                if _matches_expected_abstention(question, expected)
            ]
            for question in abstained
        ],
        expected_count=len(spec.expected_abstentions),
    )
    abstention_precision = _precision(
        abstention_match_count, len(abstained), spec.expected_abstentions
    )
    abstention_recall = _recall(abstention_match_count, len(spec.expected_abstentions))

    dataset_names = {profile.name for profile in profiles}
    overview_coverage = _section_dataset_coverage(report, "Dataset Overview", dataset_names)
    file_coverage = _section_dataset_coverage(report, "File-by-File EDA Summary", dataset_names)
    report_dataset_coverage = min(overview_coverage, file_coverage)
    quality_dataset_coverage = _section_dataset_coverage(
        report, "Data Quality Findings", dataset_names
    )

    executive_texts = _section_claim_texts(report, "Executive Summary")
    executive_matches = _matched_pattern_count(
        executive_texts, spec.required_executive_summary_patterns
    )
    executive_recall = _recall(executive_matches, len(spec.required_executive_summary_patterns))

    output_texts = _semantic_output_texts(answered, report)
    escapes = _semantic_escapes(output_texts, spec.forbidden_output_patterns)
    escape_rate = (
        len(escapes) / len(spec.forbidden_output_patterns)
        if spec.forbidden_output_patterns
        else 0.0
    )
    signature = _semantic_signature(questions, report)

    gate_failures: list[str] = []
    if spec.expected_dataset_count != 0 and not profiles and not questions and report is None:
        gate_failures.append("workflow_outputs_missing")
    if not metrics_artifacts:
        gate_failures.append("run_metrics_missing")
    elif len(metrics_artifacts) > 1:
        gate_failures.append(f"run_metrics_count={len(metrics_artifacts)} expected=1")
    target_session_ids = {
        artifact.session_id
        for artifact in artifact_list
        if artifact.type
        in {
            ArtifactType.QUESTION_EXECUTION_RESULT,
            ArtifactType.REPORT_BUNDLE,
            ArtifactType.SESSION_METRICS,
        }
    }
    if len(target_session_ids) > 1:
        gate_failures.append(
            "evaluation_output_run_mismatch=" + ",".join(sorted(target_session_ids))
        )
    profile_session_ids = {
        artifact.session_id
        for artifact in artifact_list
        if artifact.type is ArtifactType.DATASET_PROFILE
    }
    if len(profile_session_ids) > 1:
        gate_failures.append(
            "evaluation_profile_run_mismatch=" + ",".join(sorted(profile_session_ids))
        )
    if metrics_artifacts and metrics.session_id not in target_session_ids:
        gate_failures.append(f"run_metrics_payload_mismatch={metrics.session_id or '<empty>'}")
    if spec.expected_dataset_count is not None and len(profiles) != spec.expected_dataset_count:
        gate_failures.append(
            f"dataset_count={len(profiles)} expected={spec.expected_dataset_count}"
        )
    _append_floor_failure(
        gate_failures,
        "answer_precision",
        answer_precision,
        spec.min_answer_precision,
    )
    _append_floor_failure(gate_failures, "answer_recall", answer_recall, spec.min_answer_recall)
    _append_floor_failure(
        gate_failures,
        "abstention_precision",
        abstention_precision,
        spec.min_abstention_precision,
    )
    _append_floor_failure(
        gate_failures,
        "abstention_recall",
        abstention_recall,
        spec.min_abstention_recall,
    )
    _append_floor_failure(
        gate_failures,
        "report_dataset_coverage",
        report_dataset_coverage,
        spec.min_report_dataset_coverage,
    )
    _append_floor_failure(
        gate_failures,
        "quality_dataset_coverage",
        quality_dataset_coverage,
        spec.min_quality_dataset_coverage,
    )
    _append_floor_failure(
        gate_failures,
        "executive_summary_recall",
        executive_recall,
        spec.min_executive_summary_recall,
    )
    if escape_rate > spec.max_semantic_escape_rate:
        gate_failures.append(
            f"semantic_escape_rate={escape_rate:.4f} max={spec.max_semantic_escape_rate:.4f}"
        )
    if metrics.failures_count > spec.max_failures:
        gate_failures.append(f"failures_count={metrics.failures_count} max={spec.max_failures}")
    if spec.max_tokens is not None and metrics.total_tokens > spec.max_tokens:
        gate_failures.append(f"total_tokens={metrics.total_tokens} max={spec.max_tokens}")
    if (
        spec.max_duration_seconds is not None
        and metrics.duration_seconds > spec.max_duration_seconds
    ):
        gate_failures.append(
            f"duration_seconds={metrics.duration_seconds:.4f} max={spec.max_duration_seconds:.4f}"
        )

    return WorkflowEvalResult(
        case_name=spec.name,
        spec_digest=stable_hash(spec.model_dump(mode="json"), length=32),
        session_id=metrics.session_id or session_id,
        passed=not gate_failures,
        gate_failures=gate_failures,
        dataset_count=len(profiles),
        expected_answer_count=len(spec.expected_answers),
        expected_abstention_count=len(spec.expected_abstentions),
        semantic_rule_count=len(spec.forbidden_output_patterns),
        answered_count=len(answered),
        abstained_count=len(abstained),
        failed_count=len(failed),
        answer_precision=round(answer_precision, 6),
        answer_recall=round(answer_recall, 6),
        abstention_precision=round(abstention_precision, 6),
        abstention_recall=round(abstention_recall, 6),
        report_dataset_coverage=round(report_dataset_coverage, 6),
        quality_dataset_coverage=round(quality_dataset_coverage, 6),
        executive_summary_recall=round(executive_recall, 6),
        semantic_escape_rate=round(escape_rate, 6),
        semantic_escapes=escapes,
        duration_seconds=metrics.duration_seconds,
        llm_calls=metrics.llm_calls,
        total_tokens=metrics.total_tokens,
        failures_count=metrics.failures_count,
        semantic_signature=signature,
    )


def aggregate_workflow_evaluations(
    spec: WorkflowEvalSpec,
    results: Iterable[WorkflowEvalResult],
) -> WorkflowEvalSuiteResult:
    """Aggregate repeated runs and enforce deterministic stability."""
    runs = list(results)
    if not runs:
        raise ValueError("At least one workflow evaluation result is required.")
    signature_counts = Counter(run.semantic_signature for run in runs)
    stability_rate = max(signature_counts.values()) / len(runs)
    gate_failures = [
        f"run[{index}] {failure}"
        for index, run in enumerate(runs, start=1)
        for failure in run.gate_failures
    ]
    expected_digest = stable_hash(spec.model_dump(mode="json"), length=32)
    seen_session_ids: set[str] = set()
    for index, run in enumerate(runs, start=1):
        if run.case_name != spec.name:
            gate_failures.append(f"run[{index}] case_name mismatch")
        if run.spec_digest != expected_digest:
            gate_failures.append(f"run[{index}] spec_digest mismatch")
        if not run.session_id:
            gate_failures.append(f"run[{index}] session_id missing")
        elif run.session_id in seen_session_ids:
            gate_failures.append(f"run[{index}] duplicate session_id={run.session_id}")
        else:
            seen_session_ids.add(run.session_id)
    if stability_rate < spec.min_stability_rate:
        gate_failures.append(
            f"stability_rate={stability_rate:.4f} min={spec.min_stability_rate:.4f}"
        )
    durations = sorted(run.duration_seconds for run in runs)
    p95_index = max(0, math.ceil(0.95 * len(durations)) - 1)
    tokens = [run.total_tokens for run in runs]
    return WorkflowEvalSuiteResult(
        case_name=spec.name,
        spec_digest=expected_digest,
        passed=not gate_failures,
        gate_failures=gate_failures,
        runs=runs,
        stability_rate=round(stability_rate, 6),
        duration_mean_seconds=round(mean(durations), 6),
        duration_p95_seconds=round(durations[p95_index], 6),
        tokens_mean=round(mean(tokens), 6),
        tokens_max=max(tokens),
    )


def compare_workflow_evaluations(
    spec: WorkflowEvalSpec,
    *,
    baseline: WorkflowEvalSuiteResult,
    current: WorkflowEvalSuiteResult,
) -> WorkflowEvalComparison:
    """Reject quality/token regression and bounded latency regression."""
    expected_digest = stable_hash(spec.model_dump(mode="json"), length=32)
    failures: list[str] = []
    if baseline.spec_digest != expected_digest:
        failures.append("baseline spec_digest does not match the current case spec")
    if current.spec_digest != expected_digest:
        failures.append("current spec_digest does not match the current case spec")
    if baseline.case_name != spec.name or current.case_name != spec.name:
        failures.append("case_name mismatch between spec, baseline, and current suite")
    if not current.passed:
        failures.append("current suite does not pass its absolute gates")

    baseline_metrics = _suite_quality_metrics(baseline)
    current_metrics = _suite_quality_metrics(current)
    deltas = {
        name: round(current_metrics[name] - baseline_metrics[name], 6) for name in baseline_metrics
    }
    policy = spec.baseline_policy
    for name in (
        "answer_precision",
        "answer_recall",
        "abstention_precision",
        "abstention_recall",
        "report_dataset_coverage",
        "quality_dataset_coverage",
        "executive_summary_recall",
    ):
        if deltas[name] < policy.min_quality_metric_delta:
            failures.append(
                f"{name}_delta={deltas[name]:.4f} min={policy.min_quality_metric_delta:.4f}"
            )
    if deltas["semantic_escape_rate"] > policy.max_semantic_escape_delta:
        failures.append(
            f"semantic_escape_delta={deltas['semantic_escape_rate']:.4f} "
            f"max={policy.max_semantic_escape_delta:.4f}"
        )
    if deltas["failures_count"] > policy.max_failures_delta:
        failures.append(
            f"failures_delta={deltas['failures_count']:.0f} max={policy.max_failures_delta}"
        )
    stability_delta = current.stability_rate - baseline.stability_rate
    deltas["stability_rate"] = round(stability_delta, 6)
    if stability_delta < policy.min_stability_delta:
        failures.append(
            f"stability_delta={stability_delta:.4f} min={policy.min_stability_delta:.4f}"
        )
    _compare_duration(
        failures,
        deltas,
        name="duration_mean",
        baseline_value=baseline.duration_mean_seconds,
        current_value=current.duration_mean_seconds,
        maximum_ratio=policy.max_duration_mean_regression_ratio,
        maximum_seconds=policy.max_duration_mean_regression_seconds,
    )
    _compare_duration(
        failures,
        deltas,
        name="duration_p95",
        baseline_value=baseline.duration_p95_seconds,
        current_value=current.duration_p95_seconds,
        maximum_ratio=policy.max_duration_p95_regression_ratio,
        maximum_seconds=policy.max_duration_p95_regression_seconds,
    )
    tokens_mean_delta = current.tokens_mean - baseline.tokens_mean
    tokens_max_delta = current.tokens_max - baseline.tokens_max
    deltas["tokens_mean"] = round(tokens_mean_delta, 6)
    deltas["tokens_max"] = float(tokens_max_delta)
    if tokens_mean_delta > policy.max_tokens_mean_increase:
        failures.append(
            f"tokens_mean_delta={tokens_mean_delta:.2f} max={policy.max_tokens_mean_increase:.2f}"
        )
    if tokens_max_delta > policy.max_tokens_max_increase:
        failures.append(f"tokens_max_delta={tokens_max_delta} max={policy.max_tokens_max_increase}")
    return WorkflowEvalComparison(
        case_name=spec.name,
        spec_digest=expected_digest,
        passed=not failures,
        gate_failures=failures,
        baseline_passed=baseline.passed,
        current_passed=current.passed,
        metric_deltas=deltas,
    )


def _matched_pattern_count(texts: list[str], patterns: list[str]) -> int:
    return sum(
        any(re.search(pattern, text, re.IGNORECASE) for text in texts) for pattern in patterns
    )


def _matches_expected_answer(
    question: QuestionExecutionResult,
    expected: ExpectedAnswer,
    available_artifact_ids: set[str],
) -> bool:
    if not re.search(expected.question_pattern, question.question, re.IGNORECASE):
        return False
    supported_findings = [
        finding
        for finding in question.findings
        if any(
            evidence.artifact_id in available_artifact_ids
            for evidence in finding.evidence
            if evidence.artifact_id
        )
    ]
    if not supported_findings:
        return False
    output_texts = [finding.text for finding in supported_findings]
    return all(
        any(re.search(pattern, text, re.IGNORECASE) for text in output_texts)
        for pattern in expected.required_output_patterns
    )


def _matches_expected_abstention(
    question: QuestionExecutionResult, expected: ExpectedAbstention
) -> bool:
    if not re.search(expected.question_pattern, question.question, re.IGNORECASE):
        return False
    return not expected.allowed_codes or question.abstention_code in expected.allowed_codes


def _precision(correct: int, predicted: int, expected: object) -> float:
    if predicted:
        return correct / predicted
    return 1.0 if not expected else 0.0


def _maximum_match_count(
    candidate_expected_indexes: list[list[int]], *, expected_count: int
) -> int:
    """Maximum one-to-one prediction/ground-truth matches."""
    matched_prediction_by_expected = [-1] * expected_count

    def assign(prediction_index: int, visited: set[int]) -> bool:
        for expected_index in candidate_expected_indexes[prediction_index]:
            if expected_index in visited:
                continue
            visited.add(expected_index)
            previous = matched_prediction_by_expected[expected_index]
            if previous == -1 or assign(previous, visited):
                matched_prediction_by_expected[expected_index] = prediction_index
                return True
        return False

    return sum(
        assign(prediction_index, set())
        for prediction_index in range(len(candidate_expected_indexes))
    )


def _recall(matched: int, expected: int) -> float:
    return matched / expected if expected else 1.0


def _section_claim_texts(report: ReportBundle | None, title: str) -> list[str]:
    if report is None:
        return []
    section = next((section for section in report.sections if section.title == title), None)
    return [claim.text for claim in section.claims] if section is not None else []


def _section_dataset_coverage(
    report: ReportBundle | None,
    title: str,
    dataset_names: set[str],
) -> float:
    if not dataset_names:
        return 1.0
    if report is None:
        return 0.0
    section = next((section for section in report.sections if section.title == title), None)
    if section is None:
        return 0.0
    covered = {
        name
        for claim in section.claims
        for name in claim.referenced_datasets
        if name in dataset_names
    }
    return len(covered) / len(dataset_names)


def _semantic_output_texts(
    answered: list[QuestionExecutionResult], report: ReportBundle | None
) -> list[str]:
    texts = [question.question for question in answered]
    texts.extend(finding.text for question in answered for finding in question.findings)
    if report is not None:
        texts.extend(claim.text for section in report.sections for claim in section.claims)
    return texts


def _semantic_escapes(texts: list[str], patterns: list[str]) -> list[SemanticEscape]:
    escapes: list[SemanticEscape] = []
    for pattern in patterns:
        matched = [text for text in texts if re.search(pattern, text, re.IGNORECASE)]
        if matched:
            escapes.append(SemanticEscape(pattern=pattern, matched_texts=matched[:5]))
    return escapes


def _artifact_id_aliases(questions: list[QuestionExecutionResult]) -> dict[str, str]:
    """Alias artifact ids by first appearance in a run-independent traversal.

    Several artifact ids are minted from the run id (`sql_<hash(session_id, sql)>`),
    so hashing them verbatim made the signature differ on every repeat and sank
    stability_rate to 1/repeat. Aliases keep evidence sharing observable without
    the run-scoped hash.
    """
    aliases: dict[str, str] = {}
    for question in sorted(questions, key=lambda item: (item.question, item.question_id)):
        for finding in sorted(question.findings, key=lambda item: item.text):
            for evidence in sorted(finding.evidence, key=lambda item: (item.kind, item.locator)):
                artifact_id = evidence.artifact_id
                if artifact_id and artifact_id not in aliases:
                    aliases[artifact_id] = f"artifact#{len(aliases)}"
    return aliases


def _semantic_signature(
    questions: list[QuestionExecutionResult], report: ReportBundle | None
) -> str:
    aliases = _artifact_id_aliases(questions)
    question_payload = sorted(
        (
            question.question,
            question.outcome,
            question.abstention_code or "",
            tuple(
                sorted(
                    (
                        finding.text,
                        tuple(
                            sorted(
                                (
                                    evidence.kind,
                                    aliases.get(evidence.artifact_id or "", ""),
                                    evidence.locator,
                                )
                                for evidence in finding.evidence
                            )
                        ),
                    )
                    for finding in question.findings
                )
            ),
        )
        for question in questions
    )
    report_payload = (
        sorted(
            (section.title, tuple(sorted(claim.text for claim in section.claims)))
            for section in report.sections
        )
        if report is not None
        else []
    )
    return stable_hash({"questions": question_payload, "report": report_payload}, length=32)


def _append_floor_failure(failures: list[str], name: str, actual: float, minimum: float) -> None:
    if actual < minimum:
        failures.append(f"{name}={actual:.4f} min={minimum:.4f}")


def _suite_quality_metrics(suite: WorkflowEvalSuiteResult) -> dict[str, float]:
    return {
        "answer_precision": min(run.answer_precision for run in suite.runs),
        "answer_recall": min(run.answer_recall for run in suite.runs),
        "abstention_precision": min(run.abstention_precision for run in suite.runs),
        "abstention_recall": min(run.abstention_recall for run in suite.runs),
        "report_dataset_coverage": min(run.report_dataset_coverage for run in suite.runs),
        "quality_dataset_coverage": min(run.quality_dataset_coverage for run in suite.runs),
        "executive_summary_recall": min(run.executive_summary_recall for run in suite.runs),
        "semantic_escape_rate": max(run.semantic_escape_rate for run in suite.runs),
        "failures_count": float(max(run.failures_count for run in suite.runs)),
    }


def _compare_duration(
    failures: list[str],
    deltas: dict[str, float],
    *,
    name: str,
    baseline_value: float,
    current_value: float,
    maximum_ratio: float,
    maximum_seconds: float,
) -> None:
    absolute_delta = current_value - baseline_value
    deltas[f"{name}_seconds"] = round(absolute_delta, 6)
    if baseline_value <= 0:
        return
    ratio = absolute_delta / baseline_value
    deltas[f"{name}_ratio"] = round(ratio, 6)
    # Micro-benchmarks amplify millisecond scheduler noise into large ratios.
    if ratio > maximum_ratio and absolute_delta > maximum_seconds:
        failures.append(
            f"{name}_ratio={ratio:.4f} max={maximum_ratio:.4f}; "
            f"{name}_seconds={absolute_delta:.4f} max={maximum_seconds:.4f}"
        )


def _eval_failure(
    node_id: str,
    code: str,
    message: str,
    *,
    depends_on: list[str] | None = None,
) -> WorkflowEvalFailureNode:
    return WorkflowEvalFailureNode(
        node_id=node_id,
        code=code,
        message=message,
        depends_on=depends_on or [],
    )


def _cyclic_node_ids(nodes_by_id: dict[str, WorkflowEvalStep]) -> list[str]:
    """Return nodes participating in dependency cycles, ignoring missing refs."""
    visiting: set[str] = set()
    visited: set[str] = set()
    cyclic: set[str] = set()

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in visiting:
            cycle_start = path.index(node_id)
            cyclic.update(path[cycle_start:])
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        path.append(node_id)
        for dependency in nodes_by_id[node_id].depends_on:
            if dependency in nodes_by_id:
                visit(dependency, path)
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes_by_id:
        visit(node_id, [])
    return sorted(cyclic)


def _usage_values_message(
    sources: dict[str, WorkflowEvalUsageTotals],
    field_name: str,
) -> str:
    values = ", ".join(
        f"{name}={getattr(totals, field_name)}" for name, totals in sorted(sources.items())
    )
    return f"Usage reconciliation mismatch: {values}."

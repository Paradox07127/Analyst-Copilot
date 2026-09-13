"""Evaluate whole-workflow quality and cost from persisted artifacts."""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from statistics import mean

from eda_platform.agents.data_tools import data_tool_registry_digest
from eda_platform.agents.question_runtime import (
    QUESTION_AGENT_POLICY_VERSION,
    QUESTION_AGENT_PROMPT_DIGEST,
)
from eda_platform.core.exploration_profiles import (
    EXPLORATION_PROFILE_VERSION,
    EXPLORATION_STATISTICAL_POLICY_VERSION,
)
from eda_platform.core.ids import stable_hash
from eda_platform.core.llm import LLMClient, build_generation_controls
from eda_platform.core.llm_ledger import BUDGET_SETTLED_EVENT
from eda_platform.core.publication_fingerprint import DECISION_REPORT_POLICY_VERSION
from eda_platform.core.sandbox_docker import default_policy_digest
from eda_platform.core.session_metrics import spend_events
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, EvidenceRef
from eda_platform.schemas.questions import QuestionExecutionResult
from eda_platform.schemas.reports import ReportAudit, ReportBundle, ReportStatus
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.schemas.workflow_eval import (
    CompiledWorkflowEvalCase,
    EvalActionSpan,
    EvalMilestone,
    EvalMinefield,
    ExpectedAbstention,
    ExpectedAnswer,
    SemanticEscape,
    WorkflowEvalCase,
    WorkflowEvalComparison,
    WorkflowEvalEnvironment,
    WorkflowEvalFailureNode,
    WorkflowEvalGraderCertificate,
    WorkflowEvalHardGateResult,
    WorkflowEvalMutationResult,
    WorkflowEvalScore,
    WorkflowEvalSpec,
    WorkflowEvalStep,
    WorkflowEvalStepDAG,
    WorkflowEvalSuiteResult,
    WorkflowEvalTrial,
    WorkflowEvalTrialManifest,
    WorkflowEvalUsage,
    WorkflowEvalUsageTotals,
    WorkflowQualityResult,
)


@lru_cache(maxsize=1)
def _code_revision() -> str:
    """Git commit of the running checkout, or the installed package version."""
    repo_dir = Path(__file__).resolve().parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return "pkg:" + metadata.version("eda-agent-platform")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_workflow_eval_environment(llm: LLMClient | None = None) -> WorkflowEvalEnvironment:
    """Describe the real execution environment, so its fingerprint discriminates
    between code revisions, prompts, policies, tools, and provider settings."""
    settings = getattr(llm, "settings", None)
    provider = str(getattr(settings, "provider", "") or "offline")
    model = str(getattr(settings, "model", "") or "deterministic")
    model_settings = (
        dict(build_generation_controls(settings)) if settings is not None else {}
    )
    return WorkflowEvalEnvironment(
        environment_id=f"{provider}:{model}",
        provider=provider,
        model=model,
        model_settings=model_settings,
        prompt_versions={"question_agent_system": QUESTION_AGENT_PROMPT_DIGEST},
        code_revision=_code_revision(),
        policy_versions={
            "question_agent": QUESTION_AGENT_POLICY_VERSION,
            "exploration_profile": EXPLORATION_PROFILE_VERSION,
            "exploration_statistical": EXPLORATION_STATISTICAL_POLICY_VERSION,
            "decision_report": DECISION_REPORT_POLICY_VERSION,
        },
        tool_registry_digest=data_tool_registry_digest(),
        sandbox_policy_digest=default_policy_digest(),
    )


def compile_workflow_eval_case(
    spec: WorkflowEvalSpec,
    *,
    environment: WorkflowEvalEnvironment | None = None,
    case_id: str | None = None,
    trial_id: str | None = None,
    repetition: int = 1,
    dataset_fingerprints: dict[str, str] | None = None,
) -> CompiledWorkflowEvalCase:
    """Compile a case, environment, dataset identity, and repetition."""
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
        required_milestones=spec.required_milestones,
        minefields=spec.minefields,
    )
    resolved_environment = environment or build_workflow_eval_environment()
    case_fingerprint = stable_hash(case.model_dump(mode="json"), length=32)
    environment_fingerprint = stable_hash(resolved_environment.model_dump(mode="json"), length=32)
    resolved_dataset_fingerprints = dataset_fingerprints or {}
    resolved_trial_id = trial_id or stable_hash(
        {
            "case_fingerprint": case_fingerprint,
            "environment_fingerprint": environment_fingerprint,
            "dataset_fingerprints": resolved_dataset_fingerprints,
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
        dataset_fingerprints=resolved_dataset_fingerprints,
    )
    return CompiledWorkflowEvalCase(
        case=case,
        environment=resolved_environment,
        manifest=manifest,
    )


def build_workflow_eval_trial(
    spec: WorkflowEvalSpec,
    artifacts: list[Artifact],
    *,
    events: Iterable[TraceEvent] = (),
    environment: WorkflowEvalEnvironment | None = None,
    repetition: int = 1,
    trace_ref: str | None = None,
) -> WorkflowEvalTrial:
    """Project one real run into a reproducible, trajectory-graded eval trial."""
    event_list = list(events)
    metrics_artifact = _latest_artifact(artifacts, ArtifactType.SESSION_METRICS)
    metrics = (
        SessionMetrics.model_validate(metrics_artifact.payload)
        if metrics_artifact is not None
        else None
    )
    session_id = (
        metrics.session_id
        if metrics is not None
        else artifacts[0].session_id
        if artifacts
        else "missing"
    )
    dataset_fingerprints = {
        str(artifact.payload.get("name") or artifact.id): str(
            artifact.payload.get("content_hash") or stable_hash(artifact.payload, length=32)
        )
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    }
    compiled_case = compile_workflow_eval_case(
        spec,
        environment=environment,
        repetition=repetition,
        dataset_fingerprints=dataset_fingerprints,
    )
    dag, evidence_refs = _workflow_step_dag(artifacts)
    action_spans = _action_spans(event_list)
    usage = _workflow_usage(metrics, event_list)
    hard_gates = grade_workflow_hard_gates(
        step_dag=dag,
        usage=usage,
        available_artifact_refs={artifact.id for artifact in artifacts},
        available_evidence_refs=evidence_refs,
    )
    quality = grade_workflow_quality(artifacts, spec)
    quality_score = WorkflowEvalScore(
        name="workflow_quality",
        value=1.0 if quality.passed else 0.0,
        passed=quality.passed,
        failure_codes=list(quality.gate_failures),
        details={
            "answer_precision": quality.answer_precision,
            "answer_recall": quality.answer_recall,
            "abstention_precision": quality.abstention_precision,
            "abstention_recall": quality.abstention_recall,
            "semantic_escape_rate": quality.semantic_escape_rate,
        },
    )
    quality_failures = [_eval_failure("__quality__", code, code) for code in quality.gate_failures]
    release_score, release_failures = _grade_release_readiness(artifacts, metrics=metrics)
    trajectory_score, trajectory_failures = _grade_action_trajectory(
        action_spans,
        milestones=spec.required_milestones,
        minefields=spec.minefields,
    )
    failures = [
        *hard_gates.failure_nodes,
        *quality_failures,
        *release_failures,
        *trajectory_failures,
    ]
    scores = [*hard_gates.scores, quality_score, release_score, trajectory_score]
    passed = not failures and all(score.passed for score in scores)
    status = (
        "passed"
        if passed and not (metrics and metrics.degraded)
        else "degraded"
        if passed
        else "failed"
    )
    return WorkflowEvalTrial(
        manifest=compiled_case.manifest,
        session_id=session_id,
        status=status,
        trace_ref=trace_ref,
        artifact_refs=sorted(artifact.id for artifact in artifacts),
        evidence_refs=sorted(evidence_refs),
        artifact_digests={
            artifact.id: _artifact_eval_digest(artifact)
            for artifact in artifacts
            if artifact.type is not ArtifactType.WORKFLOW_EVAL_TRIAL
        },
        action_spans=action_spans,
        usage=usage,
        scores=scores,
        failure_nodes=failures,
    )


def verify_workflow_eval_trial_sources(
    trial: WorkflowEvalTrial, artifacts: Iterable[Artifact]
) -> list[WorkflowEvalFailureNode]:
    """Detect source deletion or same-ID payload replacement after grading."""
    current = {
        artifact.id: _artifact_eval_digest(artifact)
        for artifact in artifacts
        if artifact.type is not ArtifactType.WORKFLOW_EVAL_TRIAL
    }
    failures: list[WorkflowEvalFailureNode] = []
    for artifact_id, expected in trial.artifact_digests.items():
        if artifact_id not in current:
            failures.append(
                _eval_failure(artifact_id, "trial_source_missing", "A graded source is missing.")
            )
        elif current[artifact_id] != expected:
            failures.append(
                _eval_failure(
                    artifact_id,
                    "trial_source_digest_mismatch",
                    "A graded source changed while retaining its artifact ID.",
                )
            )
    for artifact_id in sorted(set(current) - set(trial.artifact_digests)):
        failures.append(
            _eval_failure(
                artifact_id,
                "trial_source_ungraded",
                "A source artifact was added after the trial was graded.",
            )
        )
    return failures


def certify_workflow_eval_grader(
    spec: WorkflowEvalSpec,
    clean_artifacts: list[Artifact],
    *,
    events: Iterable[TraceEvent] = (),
) -> WorkflowEvalGraderCertificate:
    """Run the grader against a clean oracle and fixed adversarial mutations."""
    event_list = list(events)
    clean_trial = build_workflow_eval_trial(spec, clean_artifacts, events=event_list)
    results: list[WorkflowEvalMutationResult] = []

    def record(mutation_id: str, trial: WorkflowEvalTrial) -> None:
        codes = sorted({failure.code for failure in trial.failure_nodes})
        results.append(
            WorkflowEvalMutationResult(
                mutation_id=mutation_id,
                detected=trial.status != "passed" and bool(codes),
                failure_codes=codes,
            )
        )

    evidence_mutant = deepcopy(clean_artifacts)
    for artifact in evidence_mutant:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        result = QuestionExecutionResult.model_validate(artifact.payload)
        if result.findings and result.findings[0].evidence:
            result.findings[0].evidence[0].locator = "meta_eval.missing_locator"
            artifact.payload = result.model_dump(mode="json")
            break
    record(
        "broken_evidence_locator",
        build_workflow_eval_trial(spec, evidence_mutant, events=event_list),
    )

    approval_mutant = deepcopy(clean_artifacts)
    for artifact in approval_mutant:
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT:
            result = QuestionExecutionResult.model_validate(artifact.payload)
            result.outcome = "awaiting_approval"
            result.status = "failed"
            result.findings = []
            artifact.payload = result.model_dump(mode="json")
            break
    record(
        "unresolved_approval",
        build_workflow_eval_trial(spec, approval_mutant, events=event_list),
    )

    usage_mutant = deepcopy(clean_artifacts)
    for artifact in usage_mutant:
        if artifact.type is ArtifactType.SESSION_METRICS:
            artifact.payload["total_tokens"] = int(artifact.payload.get("total_tokens") or 0) + 1
            break
    record(
        "usage_mismatch",
        build_workflow_eval_trial(spec, usage_mutant, events=event_list),
    )

    replacement_mutant = deepcopy(clean_artifacts)
    if replacement_mutant:
        replacement_mutant[0].payload["__meta_eval_tamper__"] = True
    replacement_failures = verify_workflow_eval_trial_sources(clean_trial, replacement_mutant)
    results.append(
        WorkflowEvalMutationResult(
            mutation_id="same_id_payload_replacement",
            detected=bool(replacement_failures),
            failure_codes=sorted({failure.code for failure in replacement_failures}),
        )
    )

    detected = sum(result.detected for result in results)
    recall = detected / len(results) if results else 0.0
    clean_passed = clean_trial.status == "passed"
    return WorkflowEvalGraderCertificate(
        protocol_digest=stable_hash(
            {
                "grader": "workflow-eval-hard-gates",
                "mutations": [result.mutation_id for result in results],
                "spec": clean_trial.manifest.case_fingerprint,
            },
            length=32,
        ),
        clean_oracle_passed=clean_passed,
        mutation_recall=round(recall, 6),
        release_eligible=clean_passed and recall == 1.0,
        mutations=results,
    )


def _artifact_eval_digest(artifact: Artifact) -> str:
    return stable_hash(
        {
            "type": artifact.type.value,
            "session_id": artifact.session_id,
            "parents": artifact.parents,
            "evidence": [item.model_dump(mode="json") for item in artifact.evidence],
            "payload": artifact.payload,
        },
        length=64,
    )


def _action_spans(events: list[TraceEvent]) -> list[EvalActionSpan]:
    spans: list[EvalActionSpan] = []
    for index, event in enumerate(events):
        summary = event.summary
        error_type = str(summary.get("error_type") or summary.get("error") or "") or None
        raw_status = str(summary.get("status") or "").lower()
        status = (
            "failed"
            if error_type or raw_status in {"failed", "error", "blocked", "rejected"}
            else "succeeded"
            if event.finished_at is not None or raw_status in {"success", "succeeded", "passed"}
            else "pending"
        )
        arguments = summary.get("canonical_arguments", summary.get("arguments"))
        result = summary.get("result", summary.get("output"))
        artifact_refs_raw = summary.get("artifact_refs", summary.get("artifact_ids", []))
        artifact_refs = (
            [str(item) for item in artifact_refs_raw]
            if isinstance(artifact_refs_raw, list | tuple)
            else [str(artifact_refs_raw)]
            if artifact_refs_raw
            else []
        )
        spans.append(
            EvalActionSpan(
                span_id=event.span_id
                or stable_hash(
                    {
                        "event_type": event.event_type,
                        "name": event.name,
                        "started_at": event.started_at.isoformat(),
                        "index": index,
                    },
                    length=24,
                ),
                parent_span_id=event.parent_span_id,
                operation=event.name,
                event_type=event.event_type,
                status=status,
                tool_name=(str(summary.get("tool_name")) if summary.get("tool_name") else None),
                canonical_arguments_digest=(
                    stable_hash(arguments, length=32) if arguments is not None else None
                ),
                result_digest=stable_hash(result, length=32) if result is not None else None,
                error_type=error_type,
                approval_decision=(
                    str(summary.get("approval_decision"))
                    if summary.get("approval_decision") is not None
                    else None
                ),
                retry_attempt=event.attempt_id,
                handoff_target=(
                    str(summary.get("handoff_target")) if summary.get("handoff_target") else None
                ),
                state_before_digest=(
                    str(summary.get("state_before_digest"))
                    if summary.get("state_before_digest")
                    else None
                ),
                state_after_digest=(
                    str(summary.get("state_after_digest"))
                    if summary.get("state_after_digest")
                    else None
                ),
                artifact_refs=artifact_refs,
            )
        )
    return spans


def _grade_action_trajectory(
    spans: list[EvalActionSpan],
    *,
    milestones: list[EvalMilestone],
    minefields: list[EvalMinefield],
) -> tuple[WorkflowEvalScore, list[WorkflowEvalFailureNode]]:
    failures: list[WorkflowEvalFailureNode] = []
    for milestone in milestones:
        if not any(_span_matches(span, milestone) for span in spans):
            failures.append(
                _eval_failure(
                    milestone.milestone_id,
                    "milestone_missing",
                    "Required action milestone was not observed in the durable trace.",
                )
            )
    for minefield in minefields:
        if any(_span_matches(span, minefield) for span in spans):
            failures.append(
                _eval_failure(
                    minefield.minefield_id,
                    "minefield_triggered",
                    "A forbidden action was observed in the durable trace.",
                )
            )
    return (
        WorkflowEvalScore(
            name="action_trajectory",
            value=0.0 if failures else 1.0,
            passed=not failures,
            failure_codes=sorted({failure.code for failure in failures}),
            details={
                "spans": len(spans),
                "milestones": len(milestones),
                "minefields": len(minefields),
            },
        ),
        failures,
    )


def _span_matches(span: EvalActionSpan, rule: EvalMilestone | EvalMinefield) -> bool:
    if rule.event_type is not None and span.event_type != rule.event_type:
        return False
    if rule.operation_pattern is not None:
        try:
            if re.search(rule.operation_pattern, span.operation, flags=re.IGNORECASE) is None:
                return False
        except re.error:
            return False
    return rule.event_type is not None or rule.operation_pattern is not None


def _grade_release_readiness(
    artifacts: list[Artifact], *, metrics: SessionMetrics | None
) -> tuple[WorkflowEvalScore, list[WorkflowEvalFailureNode]]:
    failures: list[WorkflowEvalFailureNode] = []
    questions = [
        QuestionExecutionResult.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    if any(question.outcome == "awaiting_approval" for question in questions):
        failures.append(
            _eval_failure(
                "__release__",
                "approval_unresolved",
                "A release trial cannot contain unresolved approvals.",
            )
        )
    if metrics is not None:
        if metrics.degraded:
            failures.append(
                _eval_failure("__release__", "run_degraded", "The run is marked degraded.")
            )
        if metrics.trace_status != "verified":
            failures.append(
                _eval_failure(
                    "__release__", "trace_unverified", "Trace accounting is unverifiable."
                )
            )
        if metrics.publication_blocked:
            failures.append(
                _eval_failure("__release__", "publication_blocked", "Publication is blocked.")
            )

    report_artifact = _latest_artifact(artifacts, ArtifactType.REPORT_BUNDLE)
    if report_artifact is not None:
        report = ReportBundle.model_validate(report_artifact.payload)
        if report.status is not ReportStatus.VALIDATED:
            failures.append(
                _eval_failure(
                    report_artifact.id,
                    "report_not_validated",
                    f"Report status is {report.status.value}, not validated.",
                )
            )
        audit = report.audit
        audit_artifact = _latest_artifact(artifacts, ArtifactType.REPORT_AUDIT)
        if audit_artifact is not None:
            audit = ReportAudit.model_validate(audit_artifact.payload)
            if report_artifact.id not in audit_artifact.parents:
                # Recency pairing is not a binding: a stale passing audit must
                # not clear a rewritten, never-audited report (H3).
                failures.append(
                    _eval_failure(
                        audit_artifact.id,
                        "report_audit_stale",
                        "The latest report audit does not audit the latest "
                        "report bundle.",
                    )
                )
        if audit is None:
            failures.append(
                _eval_failure(
                    report_artifact.id,
                    "report_audit_missing",
                    "A published report requires a typed audit.",
                )
            )
        elif (
            audit.status is not ReportStatus.VALIDATED
            or audit.gate_verdict != "pass"
            or audit.has_critical_findings
        ):
            failures.append(
                _eval_failure(
                    report_artifact.id,
                    "report_audit_failed",
                    "The latest report audit is not release-eligible.",
                )
            )
    return (
        WorkflowEvalScore(
            name="release_readiness",
            value=0.0 if failures else 1.0,
            passed=not failures,
            failure_codes=sorted({failure.code for failure in failures}),
        ),
        failures,
    )


def _latest_artifact(artifacts: list[Artifact], artifact_type: ArtifactType) -> Artifact | None:
    candidates = [artifact for artifact in artifacts if artifact.type is artifact_type]
    return max(candidates, key=lambda artifact: artifact.created_at) if candidates else None


def _workflow_step_dag(
    artifacts: list[Artifact],
) -> tuple[WorkflowEvalStepDAG, set[str]]:
    steps: list[WorkflowEvalStep] = []
    evidence_refs: set[str] = set()
    artifacts_by_id = {artifact.id: artifact for artifact in artifacts}
    ancestor_ids = {
        artifact.id: _artifact_ancestor_ids(artifact.id, artifacts_by_id) for artifact in artifacts
    }
    dataset_nodes: list[str] = []
    dataset_node_by_artifact: dict[str, str] = {}
    finding_nodes: list[str] = []
    question_nodes: list[str] = []
    question_node_by_artifact: dict[str, str] = {}
    finding_nodes_by_question: dict[str, list[str]] = {}

    for artifact in artifacts:
        if artifact.type is not ArtifactType.DATASET_PROFILE:
            continue
        node_id = f"dataset:{artifact.id}"
        dataset_nodes.append(node_id)
        dataset_node_by_artifact[artifact.id] = node_id
        steps.append(
            WorkflowEvalStep(
                node_id=node_id,
                kind="dataset",
                status="succeeded",
                artifact_refs=[artifact.id],
            )
        )

    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        result = QuestionExecutionResult.model_validate(artifact.payload)
        question_node = f"question:{artifact.id}"
        question_nodes.append(question_node)
        question_node_by_artifact[artifact.id] = question_node
        question_dataset_dependencies = sorted(
            dataset_node_by_artifact[ancestor]
            for ancestor in ancestor_ids[artifact.id]
            if ancestor in dataset_node_by_artifact
        )
        question_status = (
            "succeeded"
            if result.outcome == "answered"
            else "abstained"
            if result.outcome == "abstained"
            else "awaiting_approval"
            if result.outcome == "awaiting_approval"
            else "failed"
        )
        steps.append(
            WorkflowEvalStep(
                node_id=question_node,
                kind="question",
                status=question_status,
                depends_on=question_dataset_dependencies,
                artifact_refs=[artifact.id],
                failure_reason=result.failure_reason or result.error,
            )
        )
        for index, finding in enumerate(result.findings):
            refs = _evidence_keys(finding.evidence)
            evidence_refs.update(_resolvable_evidence_keys(finding.evidence, artifacts_by_id))
            node_id = f"finding:{artifact.id}:{index}"
            finding_nodes.append(node_id)
            finding_nodes_by_question.setdefault(artifact.id, []).append(node_id)
            steps.append(
                WorkflowEvalStep(
                    node_id=node_id,
                    kind="finding",
                    status="succeeded",
                    depends_on=[question_node],
                    artifact_refs=[artifact.id],
                    evidence_refs=refs,
                )
            )

    report_nodes: list[str] = []
    report_nodes_by_artifact: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.REPORT_BUNDLE:
            continue
        bundle = ReportBundle.model_validate(artifact.payload)
        report_question_ids = sorted(
            ancestor
            for ancestor in ancestor_ids[artifact.id]
            if ancestor in question_node_by_artifact
        )
        report_dependencies = [
            node
            for question_id in report_question_ids
            for node in (
                finding_nodes_by_question.get(question_id)
                or [question_node_by_artifact[question_id]]
            )
        ]
        if not report_dependencies:
            report_dependencies = sorted(
                dataset_node_by_artifact[ancestor]
                for ancestor in ancestor_ids[artifact.id]
                if ancestor in dataset_node_by_artifact
            )
        for section_index, section in enumerate(bundle.sections):
            for claim_index, claim in enumerate(section.claims):
                refs = _evidence_keys(claim.evidence)
                evidence_refs.update(_resolvable_evidence_keys(claim.evidence, artifacts_by_id))
                node_id = f"report:{artifact.id}:{section_index}:{claim_index}"
                report_nodes.append(node_id)
                report_nodes_by_artifact.setdefault(artifact.id, []).append(node_id)
                steps.append(
                    WorkflowEvalStep(
                        node_id=node_id,
                        kind="report_claim",
                        status="succeeded",
                        depends_on=list(report_dependencies),
                        artifact_refs=[artifact.id],
                        evidence_refs=refs,
                    )
                )

    for artifact in artifacts:
        if artifact.type is ArtifactType.REPORT_AUDIT:
            validation_dependencies = [
                node
                for ancestor in ancestor_ids[artifact.id]
                for node in report_nodes_by_artifact.get(ancestor, [])
            ]
            steps.append(
                WorkflowEvalStep(
                    node_id=f"validation:{artifact.id}",
                    kind="validation",
                    status="succeeded",
                    depends_on=validation_dependencies,
                    artifact_refs=[artifact.id],
                )
            )
    for artifact in artifacts:
        evidence_refs.update(_resolvable_evidence_keys(artifact.evidence, artifacts_by_id))
    return WorkflowEvalStepDAG(steps=steps), evidence_refs


def _artifact_ancestor_ids(artifact_id: str, artifacts_by_id: dict[str, Artifact]) -> set[str]:
    """Return only ancestors proven through persisted ``parents`` edges."""
    ancestors: set[str] = set()
    pending = list(artifacts_by_id[artifact_id].parents) if artifact_id in artifacts_by_id else []
    while pending:
        parent_id = pending.pop()
        if parent_id in ancestors:
            continue
        ancestors.add(parent_id)
        parent = artifacts_by_id.get(parent_id)
        if parent is not None:
            pending.extend(parent.parents)
    return ancestors


def _evidence_keys(values: Iterable[object]) -> list[str]:
    keys: list[str] = []
    for value in values:
        artifact_id = getattr(value, "artifact_id", None)
        locator = getattr(value, "locator", "")
        if artifact_id:
            keys.append(f"{artifact_id}:{locator}")
    return keys


def _resolvable_evidence_keys(
    values: Iterable[EvidenceRef], artifacts_by_id: dict[str, Artifact]
) -> list[str]:
    """Admit only refs whose artifact and locator resolve to persisted content."""
    keys: list[str] = []
    for value in values:
        if value.artifact_id is None:
            continue
        artifact = artifacts_by_id.get(value.artifact_id)
        if artifact is not None and _evidence_ref_resolves(artifact, value):
            keys.append(f"{value.artifact_id}:{value.locator}")
    return keys


def _evidence_ref_resolves(artifact: Artifact, evidence: EvidenceRef) -> bool:
    locator = evidence.locator.strip()
    if not locator:
        return bool(artifact.payload)

    payload: object = artifact.payload
    # Report evidence uses ``rows`` as the stable public locator for SQL while
    # SqlResult persists the bounded materialization under ``rows_preview``.
    if artifact.type is ArtifactType.SQL_RESULT:
        rows = artifact.payload.get("rows_preview")
        if locator == "rows":
            return isinstance(rows, list) and bool(rows)
        if isinstance(rows, list) and locator in artifact.payload.get("columns", []):
            return any(isinstance(row, dict) and locator in row for row in rows)
        if locator.startswith("rows["):
            payload = {"rows": rows}
        if locator in {"derived: row_count", "derived: len(rows_preview)"}:
            expected = (
                artifact.payload.get("row_count")
                if locator == "derived: row_count"
                else len(rows)
                if isinstance(rows, list)
                else None
            )
            return expected is not None and evidence.value == expected
    # ``summary`` is a documented aggregate locator synthesized from the
    # typed DatasetProfile rather than a literal payload field.
    if artifact.type is ArtifactType.DATASET_PROFILE:
        if locator == "summary":
            return all(key in artifact.payload for key in ("rows", "columns"))
        if locator == "dataset.row_count":
            return evidence.value == artifact.payload.get("rows")
    if artifact.type is ArtifactType.CHART_SPEC and locator == "chart":
        return "title" in artifact.payload and "encoding" in artifact.payload
    if artifact.type is ArtifactType.TABLE and locator in {"table", "rows"}:
        return isinstance(artifact.payload.get("rows"), list)
    if artifact.type is ArtifactType.QUALITY_ISSUE_SET:
        issues = artifact.payload.get("issues")
        if locator == "issues":
            return isinstance(issues, list)
        if locator.startswith("quality_issue:") and isinstance(issues, list):
            code, separator, column = locator.removeprefix("quality_issue:").partition(":")
            return bool(separator) and any(
                isinstance(issue, dict)
                and issue.get("code") == code
                and (issue.get("column") or "") == column
                for issue in issues
            )

    parts = _locator_parts(locator)
    if not parts:
        return False
    return _locator_value_resolves(payload, parts)


_LOCATOR_PART = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+|\*)\])?$")


def _locator_parts(locator: str) -> list[tuple[str, int | None, bool]]:
    parts: list[tuple[str, int | None, bool]] = []
    for raw in locator.split("."):
        match = _LOCATOR_PART.fullmatch(raw)
        if match is None:
            return []
        token = match.group(2)
        parts.append(
            (
                match.group(1),
                int(token) if token is not None and token != "*" else None,
                token == "*",
            )
        )
    return parts


def _locator_value_resolves(current: object, parts: list[tuple[str, int | None, bool]]) -> bool:
    if not parts:
        return current is not None and current not in ("", [], {})
    name, index, wildcard = parts[0]
    if not isinstance(current, dict) or name not in current:
        return False
    selected = current[name]
    if wildcard:
        values = (
            list(selected.values())
            if isinstance(selected, dict)
            else selected
            if isinstance(selected, list)
            else []
        )
        return any(_locator_value_resolves(value, parts[1:]) for value in values)
    if index is not None:
        if not isinstance(selected, list) or not 0 <= index < len(selected):
            return False
        selected = selected[index]
    return _locator_value_resolves(selected, parts[1:])


def _workflow_usage(metrics: SessionMetrics | None, events: list[TraceEvent]) -> WorkflowEvalUsage:
    billed = spend_events(events)
    settled = [event for event in events if event.event_type == BUDGET_SETTLED_EVENT]

    def totals(source: list[TraceEvent]) -> WorkflowEvalUsageTotals:
        costs = [
            float(event.summary["estimated_cost_usd"])
            for event in source
            if event.summary.get("estimated_cost_usd") is not None
        ]
        return WorkflowEvalUsageTotals(
            llm_calls=len(source),
            total_tokens=sum(int(event.summary.get("total_tokens") or 0) for event in source),
            estimated_cost_usd=round(sum(costs), 9) if costs else None,
        )

    metrics_totals = (
        WorkflowEvalUsageTotals(
            llm_calls=metrics.llm_calls,
            total_tokens=metrics.total_tokens,
            estimated_cost_usd=metrics.est_cost_usd,
        )
        if metrics is not None
        else None
    )
    return WorkflowEvalUsage(
        ledger=totals(billed),
        budget=totals(settled),
        metrics=metrics_totals,
        budget_reserved_calls=metrics.budget_reserved_calls if metrics else None,
        budget_settled_calls=metrics.budget_settled_calls if metrics else None,
        budget_rejected_calls=metrics.budget_rejected_calls if metrics else None,
        budget_uncertain_calls=metrics.budget_uncertain_calls if metrics else None,
        budget_reconciliation=metrics.budget_reconciliation if metrics else None,
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
    has_dataset_root = any(step.kind == "dataset" for step in step_dag.steps)

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
            has_dataset_root
            and step.kind in {"question", "report_claim", "validation"}
            and step.status in {"succeeded", "abstained"}
            and not step.depends_on
        ):
            failures.append(
                _eval_failure(
                    step.node_id,
                    "missing_lineage_dependency",
                    "Observed output has no persisted lineage path to its upstream step.",
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
        if step.status == "awaiting_approval":
            failures.append(
                _eval_failure(
                    step.node_id,
                    "approval_unresolved",
                    "A release trial cannot finish with unresolved approval.",
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


def grade_workflow_quality(
    artifacts: Iterable[Artifact],
    spec: WorkflowEvalSpec,
) -> WorkflowQualityResult:
    """Grade final-output quality as one component of a canonical trial."""
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

    return WorkflowQualityResult(
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


def _aggregate_quality_results(
    spec: WorkflowEvalSpec,
    results: Iterable[WorkflowQualityResult],
) -> WorkflowEvalSuiteResult:
    """Aggregate component scores; caller must still apply canonical trial gates."""
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
        quality_results=runs,
        stability_rate=round(stability_rate, 6),
        duration_mean_seconds=round(mean(durations), 6),
        duration_p95_seconds=round(durations[p95_index], 6),
        tokens_mean=round(mean(tokens), 6),
        tokens_max=max(tokens),
    )


def aggregate_workflow_eval_trials(
    spec: WorkflowEvalSpec,
    *,
    results: Iterable[WorkflowQualityResult],
    trials: Iterable[WorkflowEvalTrial],
) -> WorkflowEvalSuiteResult:
    """Canonical suite rollup: component quality may not override hard gates."""
    result_list = list(results)
    trial_list = list(trials)
    suite = _aggregate_quality_results(spec, result_list)
    failures = list(suite.gate_failures)
    if len(trial_list) != len(result_list):
        failures.append(f"trial_count={len(trial_list)} expected={len(result_list)}")
    seen_trials: set[str] = set()
    environment_fingerprints: set[str] = set()
    dataset_identities: set[str] = set()
    # Trials must belong to THIS case: without the fingerprint check, passing
    # trials from a simpler case could carry the target case's quality results
    # through the hard gates (H2, 2026-08-12 review).
    expected_case_fingerprint = compile_workflow_eval_case(spec).manifest.case_fingerprint
    for index, trial in enumerate(trial_list, start=1):
        if trial.status != "passed":
            failures.append(f"trial[{index}] status={trial.status}")
        if trial.failure_nodes or any(not score.passed for score in trial.scores):
            failures.append(f"trial[{index}] hard_gate_failed")
        if trial.manifest.trial_id in seen_trials:
            failures.append(f"trial[{index}] duplicate_trial_id")
        if trial.manifest.case_fingerprint != expected_case_fingerprint:
            failures.append(f"trial[{index}] case_fingerprint_mismatch")
        seen_trials.add(trial.manifest.trial_id)
        environment_fingerprints.add(trial.manifest.environment_fingerprint)
        dataset_identities.add(
            stable_hash(trial.manifest.dataset_fingerprints, length=32)
        )
    if len(environment_fingerprints) > 1:
        failures.append("trial_environment_mismatch")
    if len(dataset_identities) > 1:
        # Dataset fingerprints were previously identity bookkeeping only (H1).
        failures.append("trial_dataset_mismatch")
    protocol_digest = stable_hash(
        {
            "schema": 3,
            "case": suite.spec_digest,
            "environment": sorted(environment_fingerprints),
            "datasets": sorted(dataset_identities),
        },
        length=32,
    )
    return suite.model_copy(
        update={
            "passed": not failures,
            "gate_failures": failures,
            "trials": trial_list,
            "protocol_digest": protocol_digest,
        }
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
    if baseline.protocol_digest != current.protocol_digest:
        # Covers dataset identity: a baseline produced on one dataset (or with
        # no recorded dataset identity at all) cannot certify another (H1).
        failures.append(
            "protocol_digest mismatch: baseline and current differ in "
            "case/environment/dataset identity; regenerate the baseline"
        )
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
        "answer_precision": min(run.answer_precision for run in suite.quality_results),
        "answer_recall": min(run.answer_recall for run in suite.quality_results),
        "abstention_precision": min(run.abstention_precision for run in suite.quality_results),
        "abstention_recall": min(run.abstention_recall for run in suite.quality_results),
        "report_dataset_coverage": min(
            run.report_dataset_coverage for run in suite.quality_results
        ),
        "quality_dataset_coverage": min(
            run.quality_dataset_coverage for run in suite.quality_results
        ),
        "executive_summary_recall": min(
            run.executive_summary_recall for run in suite.quality_results
        ),
        "semantic_escape_rate": max(run.semantic_escape_rate for run in suite.quality_results),
        "failures_count": float(max(run.failures_count for run in suite.quality_results)),
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

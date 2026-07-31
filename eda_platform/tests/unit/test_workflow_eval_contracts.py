from __future__ import annotations

import pytest
from pydantic import ValidationError

from eda_platform.schemas.workflow_eval import (
    ExpectedAbstention,
    ExpectedAnswer,
    WorkflowEvalEnvironment,
    WorkflowEvalScore,
    WorkflowEvalSpec,
    WorkflowEvalStep,
    WorkflowEvalStepDAG,
    WorkflowEvalTrial,
    WorkflowEvalUsage,
    WorkflowEvalUsageTotals,
)
from eda_platform.tools.workflow_eval import (
    convert_workflow_eval_spec,
    grade_workflow_hard_gates,
    grade_workflow_step_dag,
    reconcile_workflow_usage,
    restore_workflow_eval_spec,
)


def _matching_usage() -> WorkflowEvalUsage:
    totals = WorkflowEvalUsageTotals(
        llm_calls=2,
        total_tokens=120,
        estimated_cost_usd=0.012,
    )
    return WorkflowEvalUsage(
        ledger=totals,
        budget=totals,
        metrics=totals,
        budget_reserved_calls=2,
        budget_settled_calls=2,
        budget_rejected_calls=0,
        budget_uncertain_calls=0,
        budget_reconciliation="verified",
    )


def _valid_step_dag() -> WorkflowEvalStepDAG:
    return WorkflowEvalStepDAG(
        steps=[
            WorkflowEvalStep(
                node_id="question",
                kind="question",
                status="succeeded",
                artifact_refs=["question.json"],
            ),
            WorkflowEvalStep(
                node_id="finding",
                kind="finding",
                status="succeeded",
                depends_on=["question"],
                artifact_refs=["finding.json"],
                evidence_refs=["evidence:1"],
            ),
            WorkflowEvalStep(
                node_id="report",
                kind="report_claim",
                status="succeeded",
                depends_on=["finding"],
                artifact_refs=["report.json"],
                evidence_refs=["evidence:1"],
            ),
        ]
    )


def test_legacy_spec_conversion_is_lossless_and_environment_is_separate() -> None:
    spec = WorkflowEvalSpec(
        schema_version=1,
        case_version="7",
        name="revenue-contract",
        description="Revenue must remain supported.",
        input_files=["sales.csv"],
        business_context="Finance review",
        expected_dataset_count=1,
        expected_answers=[
            ExpectedAnswer(
                question_pattern="revenue",
                required_output_patterns=["100"],
            )
        ],
        expected_abstentions=[
            ExpectedAbstention(
                question_pattern="forecast",
                allowed_codes=["unsupported"],
            )
        ],
        forbidden_output_patterns=["invented"],
        min_answer_recall=0.75,
        max_tokens=500,
        max_duration_seconds=10.0,
    )
    environment = WorkflowEvalEnvironment(
        environment_id="provider-a",
        provider="example",
        model="model-v2",
        prompt_versions={"report": "3"},
        code_revision="abc123",
    )

    conversion = convert_workflow_eval_spec(
        spec,
        environment=environment,
        repetition=2,
        dataset_fingerprints={"sales.csv": "sha256:dataset"},
    )

    assert restore_workflow_eval_spec(conversion.case) == spec
    assert conversion.case.dataset_refs == ["sales.csv"]
    assert conversion.environment == environment
    assert conversion.manifest.repetition == 2
    assert conversion.manifest.dataset_fingerprints == {
        "sales.csv": "sha256:dataset"
    }
    assert conversion.manifest.case_fingerprint
    assert conversion.manifest.environment_fingerprint
    assert conversion.model_validate_json(conversion.model_dump_json()) == conversion


def test_conversion_identity_is_stable_and_repetition_specific() -> None:
    spec = WorkflowEvalSpec(name="stable")

    first = convert_workflow_eval_spec(spec, repetition=1)
    repeated = convert_workflow_eval_spec(spec, repetition=1)
    second_trial = convert_workflow_eval_spec(spec, repetition=2)

    assert first.manifest == repeated.manifest
    assert first.manifest.trial_id != second_trial.manifest.trial_id
    assert first.manifest.case_fingerprint == second_trial.manifest.case_fingerprint


def test_step_dag_grader_accepts_complete_question_finding_report_chain() -> None:
    result = grade_workflow_step_dag(
        _valid_step_dag(),
        available_artifact_refs={
            "question.json",
            "finding.json",
            "report.json",
        },
        available_evidence_refs={"evidence:1"},
    )

    assert result.passed
    assert result.failure_nodes == []
    assert result.scores[0].name == "step_dag_contract"
    assert result.scores[0].value == 1.0


def test_step_dag_grader_fails_closed_for_missing_refs_and_dependency() -> None:
    dag = WorkflowEvalStepDAG(
        steps=[
            WorkflowEvalStep(
                node_id="report",
                kind="report_claim",
                status="succeeded",
                depends_on=["missing_finding"],
                artifact_refs=["missing-report"],
                evidence_refs=["missing-evidence"],
            )
        ]
    )

    result = grade_workflow_step_dag(
        dag,
        available_artifact_refs=set(),
        available_evidence_refs=set(),
    )

    assert not result.passed
    assert {failure.code for failure in result.failure_nodes} == {
        "missing_dependency",
        "missing_artifact_ref",
        "missing_evidence_ref",
    }


def test_empty_step_dag_fails_closed() -> None:
    result = grade_workflow_step_dag(
        WorkflowEvalStepDAG(),
        available_artifact_refs=set(),
        available_evidence_refs=set(),
    )

    assert not result.passed
    assert {failure.code for failure in result.failure_nodes} == {
        "missing_step_dag"
    }


def test_step_dag_rejects_pipeline_without_terminal_outcome() -> None:
    result = grade_workflow_step_dag(
        WorkflowEvalStepDAG(
            steps=[
                WorkflowEvalStep(
                    node_id="dataset",
                    kind="dataset",
                    status="succeeded",
                    artifact_refs=["dataset.json"],
                )
            ]
        ),
        available_artifact_refs={"dataset.json"},
        available_evidence_refs=set(),
    )

    assert not result.passed
    assert "missing_terminal_outcome" in {
        failure.code for failure in result.failure_nodes
    }


def test_step_dag_grader_reports_cycles_and_failed_dependency_propagation() -> None:
    dag = WorkflowEvalStepDAG(
        steps=[
            WorkflowEvalStep(
                node_id="question",
                kind="question",
                status="failed",
                depends_on=["report"],
                failure_reason="question contract failed",
            ),
            WorkflowEvalStep(
                node_id="report",
                kind="report_claim",
                status="succeeded",
                depends_on=["question"],
            ),
        ]
    )

    result = grade_workflow_step_dag(
        dag,
        available_artifact_refs=set(),
        available_evidence_refs=set(),
    )

    codes = {failure.code for failure in result.failure_nodes}
    assert not result.passed
    assert "step_failed" in codes
    assert "succeeded_after_failed_dependency" in codes
    assert "dependency_cycle" in codes


def test_usage_reconciliation_matches_ledger_budget_and_metrics() -> None:
    score, failures = reconcile_workflow_usage(_matching_usage())

    assert score.passed
    assert score.value == 1.0
    assert failures == []


def test_usage_reconciliation_fails_closed_for_missing_and_mismatched_sources() -> None:
    missing_score, missing_failures = reconcile_workflow_usage(
        WorkflowEvalUsage(
            ledger=WorkflowEvalUsageTotals(llm_calls=1, total_tokens=5),
            metrics=WorkflowEvalUsageTotals(llm_calls=1, total_tokens=5),
        )
    )
    mismatch_score, mismatch_failures = reconcile_workflow_usage(
        WorkflowEvalUsage(
            ledger=WorkflowEvalUsageTotals(
                llm_calls=1,
                total_tokens=5,
                estimated_cost_usd=0.1,
            ),
            budget=WorkflowEvalUsageTotals(
                llm_calls=1,
                total_tokens=6,
                estimated_cost_usd=0.1,
            ),
            metrics=WorkflowEvalUsageTotals(
                llm_calls=1,
                total_tokens=5,
                estimated_cost_usd=0.1,
            ),
        )
    )

    assert not missing_score.passed
    assert {failure.code for failure in missing_failures} == {
        "missing_budget_usage"
    }
    assert not mismatch_score.passed
    assert {failure.code for failure in mismatch_failures} == {
        "total_tokens_mismatch",
        "missing_budget_status",
    }


def test_usage_reconciliation_rejects_unverifiable_or_uncertain_budget() -> None:
    usage = _matching_usage().model_copy(
        update={
            "budget_uncertain_calls": 1,
            "budget_reconciliation": "unverifiable",
        }
    )

    score, failures = reconcile_workflow_usage(usage)

    assert not score.passed
    assert {failure.code for failure in failures} == {
        "budget_call_uncertain",
        "budget_reconciliation_unverified",
    }


def test_zero_call_usage_still_requires_clean_budget_lifecycle() -> None:
    totals = WorkflowEvalUsageTotals(
        llm_calls=0,
        total_tokens=0,
        estimated_cost_usd=None,
    )
    score, failures = reconcile_workflow_usage(
        WorkflowEvalUsage(
            ledger=totals,
            budget=totals,
            metrics=totals,
            budget_reserved_calls=0,
            budget_settled_calls=0,
            budget_rejected_calls=1,
            budget_uncertain_calls=0,
            budget_reconciliation="unverifiable",
        )
    )

    assert not score.passed
    assert {failure.code for failure in failures} == {
        "budget_call_rejected",
        "budget_reconciliation_unverified",
    }


def test_passed_trial_cannot_contain_failed_scores() -> None:
    conversion = convert_workflow_eval_spec(WorkflowEvalSpec(name="trial"))

    with pytest.raises(ValidationError, match="cannot contain failures"):
        WorkflowEvalTrial(
            manifest=conversion.manifest,
            session_id="run-1",
            status="passed",
            scores=[WorkflowEvalScore(name="contract", value=0.0, passed=False)],
        )


def test_hard_gate_combines_dag_and_usage_without_an_llm_grader() -> None:
    result = grade_workflow_hard_gates(
        step_dag=_valid_step_dag(),
        usage=_matching_usage(),
        available_artifact_refs={
            "question.json",
            "finding.json",
            "report.json",
        },
        available_evidence_refs={"evidence:1"},
    )
    conversion = convert_workflow_eval_spec(WorkflowEvalSpec(name="trial"))
    trial = WorkflowEvalTrial(
        manifest=conversion.manifest,
        session_id="run-1",
        status="passed" if result.passed else "failed",
        usage=_matching_usage(),
        scores=result.scores,
        failure_nodes=result.failure_nodes,
    )

    assert result.passed
    assert {score.name for score in result.scores} == {
        "step_dag_contract",
        "usage_reconciliation",
    }
    assert trial.status == "passed"

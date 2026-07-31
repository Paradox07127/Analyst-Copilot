from __future__ import annotations

from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, EvidenceRef
from eda_platform.schemas.questions import QuestionExecutionResult, QuestionFinding
from eda_platform.schemas.reports import ReportBundle, ReportClaim
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.workflow_eval import (
    ExpectedAbstention,
    ExpectedAnswer,
    WorkflowEvalBaselinePolicy,
    WorkflowEvalSpec,
)
from eda_platform.tools.workflow_eval import (
    aggregate_workflow_evaluations,
    compare_workflow_evaluations,
    evaluate_workflow_run,
)


def _artifact(artifact_id: str, artifact_type: ArtifactType, payload: dict) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id="eval_project",
        session_id="eval_run",
        payload=payload,
    )


def _profile(dataset_id: str, name: str) -> Artifact:
    profile = DatasetProfile(
        dataset_id=dataset_id,
        name=name,
        rows=10,
        columns=2,
        column_names=["segment", "amount"],
        dtypes={"segment": "str", "amount": "float64"},
        missing_values={"segment": 0, "amount": 0},
        missing_percent={"segment": 0.0, "amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=["segment"],
    )
    return _artifact(f"profile_{dataset_id}", ArtifactType.DATASET_PROFILE, profile.model_dump())


def _question(
    question_id: str,
    question: str,
    *,
    outcome: str,
    code: str | None = None,
) -> Artifact:
    result = QuestionExecutionResult(
        question_id=question_id,
        question=question,
        origin="template",
        status="succeeded" if outcome == "answered" else "failed",
        outcome=outcome,  # type: ignore[arg-type]
        abstention_code=code,
        findings=(
            [
                QuestionFinding(
                    text=f"Finding: {question}",
                    evidence=[
                        EvidenceRef(
                            kind="artifact",
                            artifact_id="profile_ds_sales",
                            locator="dataset",
                        )
                    ],
                )
            ]
            if outcome == "answered"
            else []
        ),
    )
    return _artifact(
        f"qexec_{question_id}",
        ArtifactType.QUESTION_EXECUTION_RESULT,
        result.model_dump(mode="json"),
    )


def _report(dataset_names: list[str]) -> Artifact:
    bundle = ReportBundle.empty(project_id="eval_project", session_id="eval_run")
    for title in ("Dataset Overview", "File-by-File EDA Summary", "Data Quality Findings"):
        section = next(section for section in bundle.sections if section.title == title)
        section.claims.extend(
            ReportClaim(
                id=f"{title}_{index}",
                text=f"{name} has 10 rows.",
                referenced_datasets=[name],
            )
            for index, name in enumerate(dataset_names)
        )
    executive = next(
        section for section in bundle.sections if section.title == "Executive Summary"
    )
    executive.claims.append(
        ReportClaim(id="summary", text="Revenue reached 100 across the observed period.")
    )
    return _artifact(
        "bundle_eval",
        ArtifactType.REPORT_BUNDLE,
        bundle.model_dump(mode="json"),
    )


def _metrics() -> Artifact:
    metrics = SessionMetrics(
        session_id="eval_run",
        duration_seconds=1.25,
        total_tokens=0,
        failures_count=0,
    )
    return _artifact(
        "metrics_eval",
        ArtifactType.SESSION_METRICS,
        metrics.model_dump(mode="json"),
    )


def test_workflow_evaluator_combines_quality_coverage_and_cost_gates() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _profile("ds_returns", "returns.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _question(
            "threshold",
            "How many rows exceed the configured threshold?",
            outcome="abstained",
            code="answer_schema_mismatch",
        ),
        _report(["sales.csv", "returns.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="unit",
        expected_dataset_count=2,
        expected_answers=[
            ExpectedAnswer(
                question_pattern="total revenue",
                required_output_patterns=["Finding: What is total revenue"],
            )
        ],
        expected_abstentions=[
            ExpectedAbstention(
                question_pattern="configured threshold",
                allowed_codes=["answer_schema_mismatch"],
            )
        ],
        required_executive_summary_patterns=["Revenue reached 100"],
        forbidden_output_patterns=["p-value 0", "postal code average"],
        max_duration_seconds=2.0,
    )

    result = evaluate_workflow_run(artifacts, spec)

    assert result.passed
    assert result.answer_precision == result.answer_recall == 1.0
    assert result.abstention_precision == result.abstention_recall == 1.0
    assert result.report_dataset_coverage == 1.0
    assert result.quality_dataset_coverage == 1.0
    assert result.executive_summary_recall == 1.0
    assert result.semantic_escape_rate == 0.0
    assert result.total_tokens == 0


def test_workflow_evaluator_fails_on_unlabelled_answer_and_semantic_escape() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("bad", "Postal code average is 1002", outcome="answered"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="guardrail",
        expected_answers=[],
        forbidden_output_patterns=["postal code average"],
    )

    result = evaluate_workflow_run(artifacts, spec)

    assert not result.passed
    assert result.answer_precision == 0.0
    assert result.semantic_escape_rate == 1.0
    assert len(result.semantic_escapes) == 1
    assert any("answer_precision" in failure for failure in result.gate_failures)
    assert any("semantic_escape_rate" in failure for failure in result.gate_failures)


def test_workflow_evaluator_fails_closed_when_run_metrics_are_missing() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
    ]

    result = evaluate_workflow_run(artifacts, WorkflowEvalSpec(name="missing_metrics"))

    assert not result.passed
    assert "run_metrics_missing" in result.gate_failures


def test_answer_precision_requires_expected_numeric_output() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="numeric_ground_truth",
        expected_answers=[
            ExpectedAnswer(
                question_pattern="total revenue",
                required_output_patterns=["Revenue is 999"],
            )
        ],
    )

    result = evaluate_workflow_run(artifacts, spec)

    assert not result.passed
    assert result.answer_precision == 0.0
    assert result.answer_recall == 0.0


def test_answer_match_requires_evidence_resolving_to_available_artifact() -> None:
    question = _question(
        "revenue",
        "What is total revenue?",
        outcome="answered",
    )
    question.payload["findings"][0]["evidence"][0]["artifact_id"] = "missing-artifact"
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        question,
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="evidence_ground_truth",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
    )

    result = evaluate_workflow_run(artifacts, spec)

    assert not result.passed
    assert result.answer_precision == 0.0
    assert result.answer_recall == 0.0


def test_metrics_only_run_cannot_pass_without_explicit_empty_output_policy() -> None:
    result = evaluate_workflow_run(
        [_metrics()],
        WorkflowEvalSpec(name="empty"),
    )

    assert not result.passed
    assert "workflow_outputs_missing" in result.gate_failures


def test_one_output_cannot_satisfy_two_overlapping_ground_truth_rules() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="one_to_one_ground_truth",
        expected_answers=[
            ExpectedAnswer(question_pattern="total revenue"),
            ExpectedAnswer(question_pattern="revenue"),
        ],
    )

    result = evaluate_workflow_run(artifacts, spec)

    assert not result.passed
    assert result.answer_precision == 1.0
    assert result.answer_recall == 0.5


def test_repeated_eval_requires_stable_semantic_signature() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(name="stability", min_stability_rate=1.0)
    first = evaluate_workflow_run(artifacts, spec)
    second = first.model_copy(update={"semantic_signature": "different"})

    suite = aggregate_workflow_evaluations(spec, [first, second])

    assert not suite.passed
    assert suite.stability_rate == 0.5
    assert any("stability_rate" in failure for failure in suite.gate_failures)


def test_semantic_signature_ignores_run_scoped_artifact_ids() -> None:
    def _run(sql_artifact_id: str) -> Artifact:
        result = QuestionExecutionResult(
            question_id="gmv",
            question="What seeded-currency GMV is supported?",
            origin="template",
            status="succeeded",
            outcome="answered",
            findings=[
                QuestionFinding(
                    text="Total GMV over 10 rows is 100 BRL.",
                    evidence=[
                        EvidenceRef(
                            kind="sql",
                            artifact_id=sql_artifact_id,
                            locator="rows_preview[0].gmv_total",
                        )
                    ],
                )
            ],
        )
        return _artifact(
            "qexec_gmv", ArtifactType.QUESTION_EXECUTION_RESULT, result.model_dump(mode="json")
        )

    spec = WorkflowEvalSpec(name="stability", min_stability_rate=1.0)
    first = evaluate_workflow_run(
        [_profile("ds_sales", "sales.csv"), _run("sql_76cd07a7dfe9"), _metrics()], spec
    )
    second = evaluate_workflow_run(
        [_profile("ds_sales", "sales.csv"), _run("sql_b7641833ab1e"), _metrics()], spec
    )

    assert first.semantic_signature == second.semantic_signature


def test_semantic_signature_tracks_evidence_binding_changes() -> None:
    def _run(evidence: list[EvidenceRef]) -> Artifact:
        result = QuestionExecutionResult(
            question_id="gmv",
            question="What seeded-currency GMV is supported?",
            origin="template",
            status="succeeded",
            outcome="answered",
            findings=[
                QuestionFinding(text="Total GMV over 10 rows is 100 BRL.", evidence=evidence)
            ],
        )
        return _artifact(
            "qexec_gmv", ArtifactType.QUESTION_EXECUTION_RESULT, result.model_dump(mode="json")
        )

    spec = WorkflowEvalSpec(name="stability", min_stability_rate=1.0)
    shared = evaluate_workflow_run(
        [
            _profile("ds_sales", "sales.csv"),
            _run(
                [
                    EvidenceRef(kind="sql", artifact_id="sql_a", locator="rows_preview[0].gmv"),
                    EvidenceRef(kind="sql", artifact_id="sql_a", locator="rows_preview[0].rows"),
                ]
            ),
            _metrics(),
        ],
        spec,
    )
    split = evaluate_workflow_run(
        [
            _profile("ds_sales", "sales.csv"),
            _run(
                [
                    EvidenceRef(kind="sql", artifact_id="sql_a", locator="rows_preview[0].gmv"),
                    EvidenceRef(kind="sql", artifact_id="sql_b", locator="rows_preview[0].rows"),
                ]
            ),
            _metrics(),
        ],
        spec,
    )
    dropped = evaluate_workflow_run(
        [
            _profile("ds_sales", "sales.csv"),
            _run([EvidenceRef(kind="sql", artifact_id="sql_a", locator="rows_preview[0].gmv")]),
            _metrics(),
        ],
        spec,
    )

    assert shared.semantic_signature != split.semantic_signature
    assert shared.semantic_signature != dropped.semantic_signature


def test_repeated_eval_rejects_duplicate_run_and_foreign_result_identity() -> None:
    spec = WorkflowEvalSpec(name="identity")
    run = evaluate_workflow_run(
        [_profile("ds_sales", "sales.csv"), _report(["sales.csv"]), _metrics()],
        spec,
    )
    foreign = run.model_copy(
        update={"case_name": "other", "spec_digest": "foreign"}
    )

    suite = aggregate_workflow_evaluations(spec, [run, foreign])

    assert not suite.passed
    assert any("duplicate session_id" in failure for failure in suite.gate_failures)
    assert any("case_name mismatch" in failure for failure in suite.gate_failures)
    assert any("spec_digest mismatch" in failure for failure in suite.gate_failures)


def test_baseline_comparison_accepts_quality_gain_with_bounded_latency() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(name="delta")
    run = evaluate_workflow_run(artifacts, spec)
    current = aggregate_workflow_evaluations(spec, [run]).model_copy(
        update={"duration_mean_seconds": 1.1, "duration_p95_seconds": 1.1}
    )
    baseline_run = run.model_copy(
        update={
            "report_dataset_coverage": 0.5,
            "quality_dataset_coverage": 0.5,
            "semantic_escape_rate": 0.5,
        }
    )
    baseline = aggregate_workflow_evaluations(spec, [baseline_run]).model_copy(
        update={
            "passed": False,
            "duration_mean_seconds": 1.0,
            "duration_p95_seconds": 1.0,
        }
    )

    comparison = compare_workflow_evaluations(
        spec, baseline=baseline, current=current
    )

    assert comparison.passed
    assert comparison.metric_deltas["report_dataset_coverage"] == 0.5
    assert comparison.metric_deltas["semantic_escape_rate"] == -0.5
    assert comparison.metric_deltas["duration_mean_ratio"] == 0.1


def test_baseline_comparison_rejects_quality_latency_and_token_regression() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(name="delta_fail")
    baseline_run = evaluate_workflow_run(artifacts, spec)
    baseline = aggregate_workflow_evaluations(spec, [baseline_run]).model_copy(
        update={"duration_mean_seconds": 1.0, "duration_p95_seconds": 1.0}
    )
    current_run = baseline_run.model_copy(
        update={"answer_precision": 0.5, "total_tokens": 1}
    )
    current = aggregate_workflow_evaluations(spec, [current_run]).model_copy(
        update={
            "duration_mean_seconds": 1.3,
            "duration_p95_seconds": 1.3,
            "tokens_mean": 1.0,
            "tokens_max": 1,
        }
    )

    comparison = compare_workflow_evaluations(
        spec, baseline=baseline, current=current
    )

    assert not comparison.passed
    assert any("answer_precision_delta" in failure for failure in comparison.gate_failures)
    assert any("duration_mean_ratio" in failure for failure in comparison.gate_failures)
    assert any("tokens_mean_delta" in failure for failure in comparison.gate_failures)


def test_baseline_comparison_allows_microbenchmark_absolute_noise_budget() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="micro_delta",
        baseline_policy=WorkflowEvalBaselinePolicy(
            max_duration_mean_regression_ratio=0.15,
            max_duration_p95_regression_ratio=0.15,
            max_duration_mean_regression_seconds=0.01,
            max_duration_p95_regression_seconds=0.01,
        ),
    )
    run = evaluate_workflow_run(artifacts, spec)
    baseline = aggregate_workflow_evaluations(spec, [run]).model_copy(
        update={"duration_mean_seconds": 0.01, "duration_p95_seconds": 0.01}
    )
    current = aggregate_workflow_evaluations(spec, [run]).model_copy(
        update={"duration_mean_seconds": 0.012, "duration_p95_seconds": 0.012}
    )

    comparison = compare_workflow_evaluations(
        spec, baseline=baseline, current=current
    )

    assert comparison.passed
    assert comparison.metric_deltas["duration_mean_ratio"] == 0.2
    assert comparison.metric_deltas["duration_mean_seconds"] == 0.002

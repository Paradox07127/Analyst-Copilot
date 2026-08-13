from __future__ import annotations

from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile, EvidenceRef
from eda_platform.schemas.questions import QuestionExecutionResult, QuestionFinding
from eda_platform.schemas.reports import ReportAudit, ReportBundle, ReportClaim, ReportStatus
from eda_platform.schemas.session_metrics import SessionMetrics
from eda_platform.schemas.workflow_eval import (
    ExpectedAbstention,
    ExpectedAnswer,
    WorkflowEvalBaselinePolicy,
    WorkflowEvalSpec,
    WorkflowEvalSuiteResult,
    WorkflowEvalTrial,
    WorkflowQualityResult,
)
from eda_platform.tools.workflow_eval import (
    aggregate_workflow_eval_trials,
    build_workflow_eval_trial,
    certify_workflow_eval_grader,
    compare_workflow_evaluations,
    compile_workflow_eval_case,
    grade_workflow_quality,
    verify_workflow_eval_trial_sources,
)


def _canonical_suite(
    spec: WorkflowEvalSpec, results: list[WorkflowQualityResult]
) -> WorkflowEvalSuiteResult:
    trials = [
        WorkflowEvalTrial(
            manifest=compile_workflow_eval_case(spec, repetition=index).manifest,
            session_id=result.session_id,
            status="passed",
        )
        for index, result in enumerate(results, start=1)
    ]
    return aggregate_workflow_eval_trials(spec, results=results, trials=trials)


def _artifact(
    artifact_id: str,
    artifact_type: ArtifactType,
    payload: dict,
    *,
    parents: list[str] | None = None,
) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id="eval_project",
        session_id="eval_run",
        parents=parents or [],
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
                            locator="rows",
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
        parents=["profile_ds_sales"],
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
    executive = next(section for section in bundle.sections if section.title == "Executive Summary")
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

    result = grade_workflow_quality(artifacts, spec)

    assert result.passed
    assert result.answer_precision == result.answer_recall == 1.0
    assert result.abstention_precision == result.abstention_recall == 1.0
    assert result.report_dataset_coverage == 1.0
    assert result.quality_dataset_coverage == 1.0
    assert result.executive_summary_recall == 1.0
    assert result.semantic_escape_rate == 0.0
    assert result.total_tokens == 0


def test_real_artifacts_project_to_reproducible_trajectory_trial() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="trajectory",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    first = build_workflow_eval_trial(spec, artifacts, repetition=1)
    repeated = build_workflow_eval_trial(spec, artifacts, repetition=1)

    assert first.status == "passed"
    assert first.manifest == repeated.manifest
    assert {score.name for score in first.scores} == {
        "step_dag_contract",
        "usage_reconciliation",
        "workflow_quality",
        "release_readiness",
        "action_trajectory",
    }
    assert any(ref.startswith("profile_ds_sales:") for ref in first.evidence_refs)


def test_trajectory_trial_rejects_evidence_pointing_to_missing_artifact() -> None:
    question = _question("revenue", "What is total revenue?", outcome="answered")
    question.payload["findings"][0]["evidence"][0]["artifact_id"] = "missing"
    artifacts = [_profile("ds_sales", "sales.csv"), question, _metrics()]
    spec = WorkflowEvalSpec(
        name="trajectory_missing_evidence",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    trial = build_workflow_eval_trial(spec, artifacts)

    assert trial.status == "failed"
    assert any(node.code == "missing_evidence_ref" for node in trial.failure_nodes)


def test_trajectory_trial_rejects_unresolvable_evidence_locator() -> None:
    question = _question("revenue", "What is total revenue?", outcome="answered")
    question.payload["findings"][0]["evidence"][0]["locator"] = "not.a.real.field"
    artifacts = [_profile("ds_sales", "sales.csv"), question, _metrics()]
    spec = WorkflowEvalSpec(
        name="trajectory_bad_locator",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    trial = build_workflow_eval_trial(spec, artifacts)

    assert trial.status == "failed"
    assert any(node.code == "missing_evidence_ref" for node in trial.failure_nodes)


def test_trajectory_trial_rejects_synthetic_lineage_not_backed_by_parents() -> None:
    question = _question("revenue", "What is total revenue?", outcome="answered")
    question.parents = []
    artifacts = [_profile("ds_sales", "sales.csv"), question, _metrics()]
    spec = WorkflowEvalSpec(
        name="trajectory_missing_lineage",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    trial = build_workflow_eval_trial(spec, artifacts)

    assert trial.status == "failed"
    assert any(node.code == "missing_lineage_dependency" for node in trial.failure_nodes)


def test_trajectory_trial_rejects_unresolved_approval() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="awaiting_approval"),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="approval_pending",
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    trial = build_workflow_eval_trial(spec, artifacts)

    assert trial.status == "failed"
    assert any(node.code == "approval_unresolved" for node in trial.failure_nodes)


def test_trial_source_digest_detects_same_id_payload_replacement() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="immutable_sources",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )
    trial = build_workflow_eval_trial(spec, artifacts)
    artifacts[0].payload["rows"] = 999

    failures = verify_workflow_eval_trial_sources(trial, artifacts)

    assert [failure.code for failure in failures] == ["trial_source_digest_mismatch"]


def test_workflow_grader_certificate_detects_fixed_mutation_suite() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _question("revenue", "What is total revenue?", outcome="answered"),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(
        name="grader_meta_eval",
        expected_answers=[ExpectedAnswer(question_pattern="total revenue")],
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    certificate = certify_workflow_eval_grader(spec, artifacts)

    assert certificate.clean_oracle_passed
    assert certificate.mutation_recall == 1.0
    assert certificate.release_eligible
    assert {item.mutation_id for item in certificate.mutations} == {
        "broken_evidence_locator",
        "unresolved_approval",
        "usage_mismatch",
        "same_id_payload_replacement",
    }


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

    result = grade_workflow_quality(artifacts, spec)

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

    result = grade_workflow_quality(artifacts, WorkflowEvalSpec(name="missing_metrics"))

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

    result = grade_workflow_quality(artifacts, spec)

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

    result = grade_workflow_quality(artifacts, spec)

    assert not result.passed
    assert result.answer_precision == 0.0
    assert result.answer_recall == 0.0


def test_metrics_only_run_cannot_pass_without_explicit_empty_output_policy() -> None:
    result = grade_workflow_quality(
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

    result = grade_workflow_quality(artifacts, spec)

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
    first = grade_workflow_quality(artifacts, spec)
    second = first.model_copy(update={"semantic_signature": "different"})

    suite = _canonical_suite(spec, [first, second])

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
    first = grade_workflow_quality(
        [_profile("ds_sales", "sales.csv"), _run("sql_76cd07a7dfe9"), _metrics()], spec
    )
    second = grade_workflow_quality(
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
    shared = grade_workflow_quality(
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
    split = grade_workflow_quality(
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
    dropped = grade_workflow_quality(
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
    run = grade_workflow_quality(
        [_profile("ds_sales", "sales.csv"), _report(["sales.csv"]), _metrics()],
        spec,
    )
    foreign = run.model_copy(update={"case_name": "other", "spec_digest": "foreign"})

    suite = _canonical_suite(spec, [run, foreign])

    assert not suite.passed
    assert any("duplicate session_id" in failure for failure in suite.gate_failures)
    assert any("case_name mismatch" in failure for failure in suite.gate_failures)
    assert any("spec_digest mismatch" in failure for failure in suite.gate_failures)


def test_aggregate_rejects_trials_compiled_from_a_foreign_case() -> None:
    """H2 (2026-08-12 review): passing trials from a simpler case could be
    paired with the target case's quality results and certify it."""
    target = WorkflowEvalSpec(name="target_case")
    foreign = WorkflowEvalSpec(
        name="easy_case", expected_answers=[ExpectedAnswer(question_pattern="anything")]
    )
    run = grade_workflow_quality(
        [_profile("ds_sales", "sales.csv"), _report(["sales.csv"]), _metrics()], target
    )
    foreign_trial = WorkflowEvalTrial(
        manifest=compile_workflow_eval_case(foreign).manifest,
        session_id=run.session_id,
        status="passed",
    )

    suite = aggregate_workflow_eval_trials(target, results=[run], trials=[foreign_trial])

    assert not suite.passed
    assert any("case_fingerprint_mismatch" in failure for failure in suite.gate_failures)


def test_aggregate_rejects_mixed_dataset_identities() -> None:
    """H1 (2026-08-12 review): dataset fingerprints were recorded in the trial
    id but never compared anywhere."""
    spec = WorkflowEvalSpec(name="dataset_identity")
    run = grade_workflow_quality(
        [_profile("ds_sales", "sales.csv"), _report(["sales.csv"]), _metrics()], spec
    )
    trial_a = WorkflowEvalTrial(
        manifest=compile_workflow_eval_case(
            spec, repetition=1, dataset_fingerprints={"sales.csv": "aaa111"}
        ).manifest,
        session_id=run.session_id,
        status="passed",
    )
    trial_b = WorkflowEvalTrial(
        manifest=compile_workflow_eval_case(
            spec, repetition=2, dataset_fingerprints={"sales.csv": "bbb222"}
        ).manifest,
        session_id=run.session_id,
        status="passed",
    )

    suite = aggregate_workflow_eval_trials(spec, results=[run, run], trials=[trial_a, trial_b])

    assert not suite.passed
    assert "trial_dataset_mismatch" in suite.gate_failures


def test_baseline_comparison_rejects_a_different_dataset_identity() -> None:
    """H1: a baseline produced on one dataset must not certify another."""
    spec = WorkflowEvalSpec(name="cross_dataset")
    run = grade_workflow_quality(
        [_profile("ds_sales", "sales.csv"), _report(["sales.csv"]), _metrics()], spec
    )

    def _suite_with(fingerprint: str) -> WorkflowEvalSuiteResult:
        trial = WorkflowEvalTrial(
            manifest=compile_workflow_eval_case(
                spec, dataset_fingerprints={"sales.csv": fingerprint}
            ).manifest,
            session_id=run.session_id,
            status="passed",
        )
        return aggregate_workflow_eval_trials(spec, results=[run], trials=[trial])

    comparison = compare_workflow_evaluations(
        spec, baseline=_suite_with("aaa111"), current=_suite_with("bbb222")
    )

    assert not comparison.passed
    assert any("identity" in failure for failure in comparison.gate_failures)


def test_release_readiness_rejects_an_audit_for_a_different_bundle() -> None:
    """H3 (2026-08-12 review): the latest audit and the latest report were
    paired by recency only, so a stale passing audit could clear a rewritten,
    never-audited report."""
    bundle = ReportBundle.empty(project_id="eval_project", session_id="eval_run")
    bundle.status = ReportStatus.VALIDATED
    audit = ReportAudit(status=ReportStatus.VALIDATED, gate_verdict="pass")
    spec = WorkflowEvalSpec(
        name="stale_audit",
        min_report_dataset_coverage=0.0,
        min_quality_dataset_coverage=0.0,
    )

    def _trial(audit_parents: list[str]):
        artifacts = [
            _profile("ds_sales", "sales.csv"),
            _artifact(
                "bundle_eval", ArtifactType.REPORT_BUNDLE, bundle.model_dump(mode="json")
            ),
            _artifact(
                "audit_eval",
                ArtifactType.REPORT_AUDIT,
                audit.model_dump(mode="json"),
                parents=audit_parents,
            ),
            _metrics(),
        ]
        return build_workflow_eval_trial(spec, artifacts)

    stale = _trial(audit_parents=["bundle_someone_else"])
    assert stale.status == "failed"
    assert any(node.code == "report_audit_stale" for node in stale.failure_nodes)

    bound = _trial(audit_parents=["bundle_eval"])
    assert not any(node.code == "report_audit_stale" for node in bound.failure_nodes)


def test_baseline_comparison_accepts_quality_gain_with_bounded_latency() -> None:
    artifacts = [
        _profile("ds_sales", "sales.csv"),
        _report(["sales.csv"]),
        _metrics(),
    ]
    spec = WorkflowEvalSpec(name="delta")
    run = grade_workflow_quality(artifacts, spec)
    current = _canonical_suite(spec, [run]).model_copy(
        update={"duration_mean_seconds": 1.1, "duration_p95_seconds": 1.1}
    )
    baseline_run = run.model_copy(
        update={
            "report_dataset_coverage": 0.5,
            "quality_dataset_coverage": 0.5,
            "semantic_escape_rate": 0.5,
        }
    )
    baseline = _canonical_suite(spec, [baseline_run]).model_copy(
        update={
            "passed": False,
            "duration_mean_seconds": 1.0,
            "duration_p95_seconds": 1.0,
        }
    )

    comparison = compare_workflow_evaluations(spec, baseline=baseline, current=current)

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
    baseline_run = grade_workflow_quality(artifacts, spec)
    baseline = _canonical_suite(spec, [baseline_run]).model_copy(
        update={"duration_mean_seconds": 1.0, "duration_p95_seconds": 1.0}
    )
    current_run = baseline_run.model_copy(update={"answer_precision": 0.5, "total_tokens": 1})
    current = _canonical_suite(spec, [current_run]).model_copy(
        update={
            "duration_mean_seconds": 1.3,
            "duration_p95_seconds": 1.3,
            "tokens_mean": 1.0,
            "tokens_max": 1,
        }
    )

    comparison = compare_workflow_evaluations(spec, baseline=baseline, current=current)

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
    run = grade_workflow_quality(artifacts, spec)
    baseline = _canonical_suite(spec, [run]).model_copy(
        update={"duration_mean_seconds": 0.01, "duration_p95_seconds": 0.01}
    )
    current = _canonical_suite(spec, [run]).model_copy(
        update={"duration_mean_seconds": 0.012, "duration_p95_seconds": 0.012}
    )

    comparison = compare_workflow_evaluations(spec, baseline=baseline, current=current)

    assert comparison.passed
    assert comparison.metric_deltas["duration_mean_ratio"] == 0.2
    assert comparison.metric_deltas["duration_mean_seconds"] == 0.002

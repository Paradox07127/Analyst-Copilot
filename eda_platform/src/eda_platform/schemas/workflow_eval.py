"""Typed contracts for deterministic end-to-end workflow evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from eda_platform.schemas.questions import QuestionAnswerContract


class ExpectedAbstention(BaseModel):
    """One question that should fail closed, optionally with allowed reason codes."""

    question_pattern: str
    allowed_codes: list[str] = Field(default_factory=list)


class ExpectedAnswer(BaseModel):
    """One answerable question and the output facts its finding must contain."""

    question_pattern: str
    required_output_patterns: list[str] = Field(default_factory=list)


class WorkflowEvalProbe(BaseModel):
    """Evaluator-only deterministic question executed through the real batch path."""

    question_id: str
    question: str
    dataset_file: str
    sql_template: str
    answer_contract: QuestionAnswerContract
    produced_units: dict[str, str] = Field(default_factory=dict)


class WorkflowEvalBaselinePolicy(BaseModel):
    """Allowed regression when comparing the same case spec across versions."""

    min_quality_metric_delta: float = Field(default=0.0, le=0.0)
    min_stability_delta: float = Field(default=0.0, le=0.0)
    max_duration_mean_regression_ratio: float = Field(default=0.15, ge=0.0)
    max_duration_p95_regression_ratio: float = Field(default=0.15, ge=0.0)
    max_duration_mean_regression_seconds: float = Field(default=0.0, ge=0.0)
    max_duration_p95_regression_seconds: float = Field(default=0.0, ge=0.0)
    max_tokens_mean_increase: float = Field(default=0.0, ge=0.0)
    max_tokens_max_increase: int = Field(default=0, ge=0)
    max_semantic_escape_delta: float = Field(default=0.0, ge=0.0)
    max_failures_delta: int = Field(default=0, ge=0)


class WorkflowEvalSpec(BaseModel):
    """Ground truth and gates for one offline workflow case."""

    schema_version: int = 1
    case_version: str = "1"
    name: str
    description: str = ""
    input_files: list[str] = Field(default_factory=list)
    business_context: str = ""
    probe_questions: list[WorkflowEvalProbe] = Field(default_factory=list)
    baseline_policy: WorkflowEvalBaselinePolicy = Field(
        default_factory=WorkflowEvalBaselinePolicy
    )
    expected_dataset_count: int | None = Field(default=None, ge=0)
    expected_answers: list[ExpectedAnswer] = Field(default_factory=list)
    expected_abstentions: list[ExpectedAbstention] = Field(default_factory=list)
    required_executive_summary_patterns: list[str] = Field(default_factory=list)
    forbidden_output_patterns: list[str] = Field(default_factory=list)
    min_answer_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    min_answer_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_report_dataset_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_quality_dataset_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_executive_summary_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_stability_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_semantic_escape_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_failures: int = Field(default=0, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_duration_seconds: float | None = Field(default=None, gt=0.0)


class SemanticEscape(BaseModel):
    """A forbidden semantic pattern that escaped into an answered/report output."""

    pattern: str
    matched_texts: list[str] = Field(default_factory=list)


class WorkflowEvalResult(BaseModel):
    """Quality/cost result for one completed workflow run."""

    schema_version: int = 1
    case_name: str
    spec_digest: str
    session_id: str
    passed: bool
    gate_failures: list[str] = Field(default_factory=list)
    dataset_count: int = 0
    expected_answer_count: int = 0
    expected_abstention_count: int = 0
    semantic_rule_count: int = 0
    answered_count: int = 0
    abstained_count: int = 0
    failed_count: int = 0
    answer_precision: float = 0.0
    answer_recall: float = 0.0
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    report_dataset_coverage: float = 0.0
    quality_dataset_coverage: float = 0.0
    executive_summary_recall: float = 0.0
    semantic_escape_rate: float = 0.0
    semantic_escapes: list[SemanticEscape] = Field(default_factory=list)
    duration_seconds: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0
    failures_count: int = 0
    semantic_signature: str


class WorkflowEvalSuiteResult(BaseModel):
    """Repeated-run rollup used to balance quality, stability, latency and cost."""

    schema_version: int = 1
    case_name: str
    spec_digest: str
    passed: bool
    gate_failures: list[str] = Field(default_factory=list)
    runs: list[WorkflowEvalResult]
    stability_rate: float
    duration_mean_seconds: float
    duration_p95_seconds: float
    tokens_mean: float
    tokens_max: int


class WorkflowEvalComparison(BaseModel):
    """Regression comparison between two suites using the identical case spec."""

    schema_version: int = 1
    case_name: str
    spec_digest: str
    passed: bool
    gate_failures: list[str] = Field(default_factory=list)
    baseline_passed: bool
    current_passed: bool
    metric_deltas: dict[str, float] = Field(default_factory=dict)


class WorkflowEvalCase(BaseModel):
    """Versioned benchmark truth, independent from its execution environment."""

    schema_version: int = 1
    case_id: str
    case_version: str = "1"
    name: str
    description: str = ""
    dataset_refs: list[str] = Field(default_factory=list)
    business_context: str = ""
    probe_questions: list[WorkflowEvalProbe] = Field(default_factory=list)
    baseline_policy: WorkflowEvalBaselinePolicy = Field(
        default_factory=WorkflowEvalBaselinePolicy
    )
    expected_dataset_count: int | None = Field(default=None, ge=0)
    expected_answers: list[ExpectedAnswer] = Field(default_factory=list)
    expected_abstentions: list[ExpectedAbstention] = Field(default_factory=list)
    required_executive_summary_patterns: list[str] = Field(default_factory=list)
    forbidden_output_patterns: list[str] = Field(default_factory=list)
    min_answer_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    min_answer_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    min_abstention_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_report_dataset_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_quality_dataset_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_executive_summary_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    min_stability_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_semantic_escape_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_failures: int = Field(default=0, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_duration_seconds: float | None = Field(default=None, gt=0.0)
    source_spec_schema_version: int = 1


class WorkflowEvalEnvironment(BaseModel):
    """Provider and build configuration used to execute a case."""

    schema_version: int = 1
    environment_id: str = "offline"
    provider: str = "offline"
    model: str = "deterministic"
    model_settings: dict[str, Any] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    code_revision: str = "unknown"
    policy_versions: dict[str, str] = Field(default_factory=dict)
    pricing_catalog_version: str | None = None


class WorkflowEvalTrialManifest(BaseModel):
    """Reproducibility identity for one case/environment repetition."""

    schema_version: int = 1
    trial_id: str
    case_id: str
    case_version: str
    case_fingerprint: str
    environment_fingerprint: str
    repetition: int = Field(default=1, ge=1)
    dataset_fingerprints: dict[str, str] = Field(default_factory=dict)


class WorkflowEvalUsageTotals(BaseModel):
    """Comparable accounting totals emitted by one authoritative source."""

    llm_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class WorkflowEvalUsage(BaseModel):
    """Ledger, budget, and metrics inputs for exact reconciliation."""

    schema_version: int = 2
    ledger: WorkflowEvalUsageTotals | None = None
    budget: WorkflowEvalUsageTotals | None = None
    metrics: WorkflowEvalUsageTotals | None = None
    budget_reserved_calls: int | None = Field(default=None, ge=0)
    budget_settled_calls: int | None = Field(default=None, ge=0)
    budget_rejected_calls: int | None = Field(default=None, ge=0)
    budget_uncertain_calls: int | None = Field(default=None, ge=0)
    budget_reconciliation: Literal[
        "verified", "unverifiable", "not_applicable"
    ] | None = None


class WorkflowEvalScore(BaseModel):
    """One deterministic or externally supplied score."""

    schema_version: int = 1
    name: str
    value: float
    passed: bool
    failure_codes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvalStep(BaseModel):
    """One node in the observed workflow dependency graph."""

    schema_version: int = 1
    node_id: str
    kind: Literal[
        "dataset",
        "question",
        "probe",
        "finding",
        "report_claim",
        "validation",
    ]
    status: Literal["succeeded", "failed", "abstained", "skipped"]
    depends_on: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class WorkflowEvalStepDAG(BaseModel):
    """Versioned step graph supplied to deterministic trajectory graders."""

    schema_version: int = 1
    steps: list[WorkflowEvalStep] = Field(default_factory=list)


class WorkflowEvalFailureNode(BaseModel):
    """A failed node or contract violation with dependency context."""

    schema_version: int = 1
    node_id: str
    code: str
    message: str
    depends_on: list[str] = Field(default_factory=list)


class WorkflowEvalHardGateResult(BaseModel):
    """Deterministic Layer-A verdict; no model grader is involved."""

    schema_version: int = 1
    passed: bool
    scores: list[WorkflowEvalScore] = Field(default_factory=list)
    failure_nodes: list[WorkflowEvalFailureNode] = Field(default_factory=list)


class WorkflowEvalTrial(BaseModel):
    """Outcome of one case/environment repetition."""

    schema_version: int = 1
    manifest: WorkflowEvalTrialManifest
    session_id: str
    status: Literal["passed", "failed", "degraded", "inconclusive"]
    trace_ref: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    usage: WorkflowEvalUsage = Field(default_factory=WorkflowEvalUsage)
    scores: list[WorkflowEvalScore] = Field(default_factory=list)
    failure_nodes: list[WorkflowEvalFailureNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _passed_trial_has_no_failed_contracts(self) -> WorkflowEvalTrial:
        if self.status == "passed" and (
            self.failure_nodes or any(not score.passed for score in self.scores)
        ):
            raise ValueError("A passed workflow-eval trial cannot contain failures.")
        return self


class WorkflowEvalConversion(BaseModel):
    """Lossless legacy-spec split into benchmark, environment, and manifest."""

    schema_version: int = 1
    case: WorkflowEvalCase
    environment: WorkflowEvalEnvironment
    manifest: WorkflowEvalTrialManifest

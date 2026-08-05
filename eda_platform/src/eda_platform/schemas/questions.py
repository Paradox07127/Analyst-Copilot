from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from eda_platform.schemas.artifacts import EvidenceRef

QuestionOrigin = Literal["template", "llm"]
QuestionStatus = Literal[
    "proposed",
    "auto_selected",
    "approved",
    "executed",
    "failed",
    "rejected",
]

# Typed opportunity-card vocabulary.
AnalysisMode = Literal[
    "descriptive",
    "diagnostic",
    "forecast",
    "prediction",
    "segmentation",
    "anomaly",
    "causal_experiment",
]
FeasibilityStatus = Literal["ready", "constrained", "needs_data", "unsuitable"]
ProposedAction = Literal[
    "run_analysis",
    "collect_data",
    "confirm_relationship",
    "design_experiment",
]
# Broad value paths do not imply quantified monetary impact.
ValueCategory = Literal[
    "financial_performance",
    "cost_efficiency",
    "risk_or_service",
    "customer_or_entity",
    "decision_quality",
]


class OpportunityFeasibility(BaseModel):
    """Deterministic feasibility verdict for one card (never LLM-authored).

    ``reasons`` are teaching-style sentences a business reader can act on;
    ``missing`` lists concrete prerequisites (columns, labels, history).
    """

    status: FeasibilityStatus
    method_id: str | None = Field(
        default=None,
        description="Selected method-registry id when a supported method fits",
    )
    reasons: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class QuestionScore(BaseModel):
    """Deterministic scores gate execution; LLM scores only affect display order
    (v2-plan §4.10)."""

    data_availability: float = Field(ge=0.0, le=1.0)
    statistical_signal: float = Field(ge=0.0, le=1.0)
    quality_risk: float = Field(ge=0.0, le=1.0, description="higher = riskier")
    join_risk: float = Field(ge=0.0, le=1.0, description="higher = riskier")
    deterministic_score: float = Field(ge=0.0, le=1.0)
    llm_business_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    llm_actionability: float | None = Field(default=None, ge=0.0, le=1.0)


class QuestionAnswerContract(BaseModel):
    """Typed minimum output shape required to call a question answered."""

    kind: Literal["metric", "threshold"]
    metric_id: str | None = None
    required_column_tokens: list[str] = Field(default_factory=list)
    expected_units: dict[str, str] = Field(
        default_factory=dict,
        description="Case-sensitive output units required before metric publication",
    )
    abstention_code: str = "answer_schema_mismatch"

    @model_validator(mode="after")
    def _kind_has_binding(self) -> QuestionAnswerContract:
        if self.kind == "metric" and not self.metric_id:
            raise ValueError("metric answer contracts require metric_id")
        if self.kind == "threshold" and not self.required_column_tokens:
            raise ValueError("threshold answer contracts require column tokens")
        return self


class QuestionCandidate(BaseModel):
    question_id: str
    question_en: str
    origin: QuestionOrigin
    template_id: str | None = None
    metric_id: str | None = Field(
        default=None,
        description="Stable registry id for domain-metric candidates; absent on legacy artifacts",
    )
    answer_contract: QuestionAnswerContract | None = None
    produced_units: dict[str, str] = Field(
        default_factory=dict,
        description="Case-sensitive units declared by the deterministic SQL plan",
    )
    target_datasets: list[str] = Field(default_factory=list)
    dataset_display_names: dict[str, str] = Field(
        default_factory=dict,
        description="Business-facing display name for each target dataset",
    )
    required_relations: list[str] = Field(
        default_factory=list, description="RelationshipColumnPair labels"
    )
    sql_template: str | None = Field(
        default=None,
        description="Deterministic SQL for template-route questions; None for llm route",
    )
    score: QuestionScore
    status: QuestionStatus = "proposed"

    # Optional framing context; feasibility remains deterministic.
    business_decision: str = Field(
        default="",
        description="The decision this analysis can inform, in business language",
    )
    value_hypothesis: str = Field(
        default="",
        description="Why the answer matters (retention, revenue, cost, risk, ...)",
    )
    analysis_mode: AnalysisMode | None = Field(
        default=None,
        description="Descriptive/diagnostic/forecast/prediction/segmentation/anomaly/causal",
    )
    candidate_methods: list[str] = Field(
        default_factory=list,
        description="Method-registry ids, recommended first",
    )
    data_requirements: list[str] = Field(
        default_factory=list,
        description="Human-readable prerequisites: datasets, columns, labels, history",
    )
    feasibility: OpportunityFeasibility | None = Field(
        default=None,
        description="Deterministic gate verdict; None on legacy candidates",
    )
    success_criterion: str = Field(
        default="",
        description="What output quality would make the result decision-useful",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Missingness, leakage, sample size, join risk, non-causal limits",
    )
    proposed_action: ProposedAction | None = Field(
        default=None,
        description="Run analysis, collect data, confirm relationship, or design experiment",
    )

    # Optional workflow fields preserve compatibility with older artifacts.
    value_category: ValueCategory | None = Field(
        default=None,
        description="Broad value path this card serves (never a monetary claim)",
    )
    data_signal: str = Field(
        default="",
        description="The observed data pattern that motivated this card",
    )
    priority_rationale: str = Field(
        default="",
        description="Explainable ranking prose; display-only, never a gate",
    )
    card_version: int = Field(
        default=1,
        ge=1,
        description="Incremented on every material edit; investigation plans "
        "bind to a specific version",
    )
    referenced_columns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Columns this card references, keyed by dataset name",
    )
    source_artifact_ids: list[str] = Field(
        default_factory=list,
        description="EDA artifacts whose evidence motivated this card",
    )
    quality_context_artifact_ids: list[str] = Field(
        default_factory=list,
        description="QualityContextSet artifacts relevant to this card; "
        "payloads are fetched on demand, not embedded",
    )

    # Exploratory metadata propagates to report validation and disclosure.
    exploratory: bool = Field(
        default=False,
        description="True for LLM free-form questions; findings inherit it "
        "so the report side can apply statistical-validation gating and "
        "multiple-comparison disclosure",
    )

    @field_validator("question_id", "question_en")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @field_validator("target_datasets")
    @classmethod
    def _target_datasets_are_non_empty(cls, value: list[str]) -> list[str]:
        if value and all(dataset.strip() for dataset in value):
            return value
        raise ValueError("target_datasets must contain at least one dataset.")

    @model_validator(mode="after")
    def _display_names_match_target_datasets(self) -> QuestionCandidate:
        unknown_datasets = set(self.dataset_display_names) - set(self.target_datasets)
        if unknown_datasets:
            raise ValueError("dataset_display_names must only name target datasets.")
        if any(not display_name.strip() for display_name in self.dataset_display_names.values()):
            raise ValueError("dataset display names must be non-empty.")
        return self


class QuestionCandidateSet(BaseModel):
    """All ranked candidates for one run; artifact prefix `qcand`."""

    candidates: list[QuestionCandidate] = Field(default_factory=list)
    dedup_dropped: int = 0
    trivial_dropped: int = 0
    value_map_artifact_id: str | None = Field(
        default=None,
        description="The Role-1 ValueMap this candidate set was framed from",
    )
    # Meter template questions injected as a coverage backstop.
    template_backstop_used: int = 0
    template_backstop_categories: list[str] = Field(default_factory=list)


class FindingScore(BaseModel):
    """Insight ranking triple (DI8-D, QuickInsights-style score = impact x significance).

    - ``impact``       — business weight from the column-role layer
      (``ColumnRoleSet.impact_weight``): identifier/sequence columns weigh 0.0,
      so a statistically significant finding about a row counter sinks to the
      bottom instead of being deleted.
    - ``significance`` — the existing deterministic statistical signal
      (p-value / metric / flagged share), unchanged by roles.
    - ``interestingness`` — H9-B optional fourth component (deviation x
      coverage x nontriviality, ``tools.interestingness``); ``None`` on
      legacy scores and whenever the producer lacked sufficient inputs.
    - ``final``        — ``impact * significance``, multiplied by
      ``interestingness`` when it is present; the ranking key.

    All components are persisted so ranking stays observable. A ``None``
    interestingness is omitted from serialized payloads so pre-H9 artifacts
    and scores round-trip byte-identically (backward compatibility).
    """

    impact: float = Field(ge=0.0, le=1.0, default=1.0)
    significance: float = Field(ge=0.0, le=1.0, default=0.0)
    interestingness: float | None = Field(default=None, ge=0.0, le=1.0)
    final: float = Field(ge=0.0, le=1.0, default=0.0)

    @model_serializer(mode="wrap")
    def _omit_absent_interestingness(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if isinstance(data, dict) and data.get("interestingness") is None:
            data.pop("interestingness", None)
        return data


class QuestionFinding(BaseModel):
    """One evidence-backed statement produced by executing a question."""

    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    # DI8-D: optional so artifacts persisted by earlier runs stay valid.
    score: FindingScore | None = None
    # Inherited exploratory status for report-side validation.
    exploratory: bool = False
    # Cluster role retains duplicate evidence without repeating claims.
    dedup_role: Literal["representative", "supporting"] | None = None
    dedup_cluster_key: str | None = None


_ABSTENTION_REASONS: dict[str, str] = {
    "approval_required": (
        "This question is waiting for your approval before it can run."
    ),
    "agent_no_evidence": (
        "The agent produced an answer but no evidence artifact behind it, so the "
        "answer was not published."
    ),
    "agent_answer_unverified": (
        "The agent's answer could not be checked against the evidence it cited."
    ),
    "answer_schema_mismatch": (
        "The answer did not have the shape this question requires, so it could "
        "not be read as a result."
    ),
    "metric_contract_failed": (
        "The computed metric failed its sanity check, for example a share "
        "outside 0-100% or a duration that cannot occur."
    ),
}
_GUARD_PROBLEM_HEADING = "What was wrong:"
_MAX_FAILURE_REASON_CHARS = 300
# Database and planner errors name functions, argument types and candidate
# overloads: written for whoever debugs the planner, and three of them were
# pasted into a business report (2026-08-05 run). The detail stays in `error`
# and on the Trace page; the reader gets what it means for the question.
_TECHNICAL_ERROR_REASONS: tuple[tuple[str, str], ...] = (
    (
        "binding failed",
        "The generated query did not fit this data's column types, so it could "
        "not run.",
    ),
    (
        "produced invalid sql",
        "The generated query could not be run against this data, and the retry "
        "did not fix it.",
    ),
)


def _guard_problems(error: str) -> str:
    """The 'What was wrong' bullets of a tool-guard message, without the repair
    protocol that follows them -- that half is addressed to the model."""
    start = error.find(_GUARD_PROBLEM_HEADING)
    if start == -1:
        return ""
    problems: list[str] = []
    for line in error[start + len(_GUARD_PROBLEM_HEADING) :].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            break
        # "- `field` got 'rule': explanation." keeps only the explanation.
        _, separator, explanation = stripped.partition("': ")
        problems.append((explanation if separator else stripped[2:]).strip())
    return " ".join(problems)


def question_failure_reason(*, error: str | None, abstention_code: str | None) -> str:
    """One reader-facing sentence for a question that produced no answer."""
    mapped = _ABSTENTION_REASONS.get(abstention_code or "")
    if mapped:
        return mapped
    text = (error or "").strip()
    if not text:
        return "The run recorded no reason for this failure."
    problems = _guard_problems(text)
    if not problems:
        lowered = text.lower()
        for marker, sentence in _TECHNICAL_ERROR_REASONS:
            if marker in lowered:
                return sentence
    # An unmapped error is still more specific than a generic apology, but only
    # its first sentence: the rest is a stack trace or a retry protocol.
    reason = problems or text.splitlines()[0]
    reason = reason.replace("`", "")
    if len(reason) > _MAX_FAILURE_REASON_CHARS:
        reason = reason[: _MAX_FAILURE_REASON_CHARS - 1].rstrip() + "…"
    return reason


class QuestionExecutionResult(BaseModel):
    """Outcome of executing one question; artifact prefix `qexec`."""

    question_id: str
    question: str
    origin: QuestionOrigin
    execution_mode: Literal["pipeline", "agent"] = "pipeline"
    """The legacy fixed pipeline or the autonomous tool-using agent."""
    tool_calls: int = 0
    tool_names: list[str] = Field(default_factory=list)
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    plan_summary: str = ""
    sql: str | None = None
    sql_result_artifact_id: str | None = None
    chart_artifact_id: str | None = None
    findings: list[QuestionFinding] = Field(default_factory=list)
    status: Literal["succeeded", "failed"]
    # Normalized outcome; status remains for compatibility.
    outcome: Literal["answered", "abstained", "failed", "awaiting_approval"] | None = None
    abstention_code: str | None = None
    error: str | None = None
    # `error` is addressed to the model that must retry ("Return corrected
    # parameters..."); this is the one sentence a reader gets instead. Derived,
    # never authored, so no failure path can forget to explain itself.
    failure_reason: str = ""
    # Validator-gated interpretation; evidence remains the number source.
    interpretation: str = ""
    interpretation_status: Literal["validated", "fallback", "absent"] = "absent"
    # Mirrors the originating candidate's exploratory status.
    exploratory: bool = False
    # Registry metric this answered, when it came from the domain-metric lane.
    # The report routes a background metric's answer by it.
    metric_id: str | None = None
    # Required report disclosures for this execution.
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_legacy_outcome(self) -> QuestionExecutionResult:
        if self.outcome is None:
            self.outcome = "answered" if self.status == "succeeded" else "failed"
        if not self.failure_reason and self.outcome != "answered":
            self.failure_reason = question_failure_reason(
                error=self.error, abstention_code=self.abstention_code
            )
        return self

    @field_validator("question_id", "question")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

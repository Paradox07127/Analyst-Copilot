from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.core.methods import (
    METHOD_REGISTRY,
    MethodGateContext,
    MethodGateResult,
    evaluate_feasibility,
)
from eda_platform.drivers.question_exec import select_auto_execution_candidates
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.schemas.questions import (
    OpportunityFeasibility,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import discover_question_candidates

T = TypeVar("T", bound=BaseModel)


class _FakeQuestionLLM:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate(self.result)

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def test_registry_contains_each_sprint_one_method_family() -> None:
    assert set(METHOD_REGISTRY) == {
        "group_comparison",
        "outcome_prediction",
        "anomaly_detection",
        "descriptive_sql",
        "forecast",
        "segmentation",
        "causal_experiment",
        # H9-C: registered domain business metrics (GMV/AOV/...); rides the
        # descriptive mode but never steals descriptive_sql's first slot.
        "domain_metric_pack",
    }


def test_supported_method_gates_cover_ready_constrained_and_needs_data() -> None:
    profile = _profile(
        rows=100,
        columns=[
            _column("region", "categorical", unique_count=4),
            _column("revenue", "numeric", unique_count=90),
            _column("churned", "boolean", unique_count=2),
        ],
    )

    group = _evaluate(profile, mode="diagnostic")
    prediction = _evaluate(profile, mode="prediction", target_column="churned")
    missing_target = _evaluate(profile, mode="prediction", target_column="unknown")
    anomaly = _evaluate(profile, mode="anomaly")

    assert group.status == "ready" and group.method_id == "group_comparison"
    assert prediction.status == "constrained"
    assert prediction.method_id == "outcome_prediction"
    assert any("leakage risk" in reason for reason in prediction.reasons)
    assert missing_target.status == "needs_data"
    assert missing_target.missing == ["target column: unknown"]
    assert anomaly.status == "ready" and anomaly.method_id == "anomaly_detection"


def test_structural_and_sample_size_failures_are_distinguished() -> None:
    text_only = _profile(
        rows=100,
        columns=[_column("comment", "text", unique_count=80)],
    )
    short_numeric = _profile(
        rows=20,
        columns=[_column("amount", "numeric", unique_count=20)],
    )
    sparse_groups = _profile(
        rows=12,
        columns=[
            _column("region", "categorical", unique_count=4),
            _column("amount", "numeric", unique_count=12),
        ],
    )

    assert _evaluate(text_only, mode="anomaly").status == "unsuitable"
    assert _evaluate(short_numeric, mode="anomaly").status == "needs_data"
    group = _evaluate(sparse_groups, mode="diagnostic")
    assert group.status == "needs_data"
    assert group.missing == ["at least 5 expected rows per group"]


def test_gate_missing_kinds_label_each_missing_item() -> None:
    """DI7-B item ②: every gate labels its missing items with a structural/data/scale
    kind, parallel to ``missing``, and ``evaluate_feasibility`` derives the verdict from
    those kinds rather than matching missing-item strings."""
    text_only = _profile(rows=100, columns=[_column("comment", "text", unique_count=80)])
    short_numeric = _profile(rows=20, columns=[_column("amount", "numeric", unique_count=20)])
    sparse_groups = _profile(
        rows=12,
        columns=[
            _column("region", "categorical", unique_count=4),
            _column("amount", "numeric", unique_count=12),
        ],
    )

    # structural (wrong-shaped data) -> unsuitable
    anomaly_structural = _gate("anomaly_detection", text_only)
    assert anomaly_structural.missing == ["numeric column"]
    assert anomaly_structural.missing_kinds == ["structural"]

    # scale (not enough rows) -> needs_data
    anomaly_scale = _gate("anomaly_detection", short_numeric)
    assert anomaly_scale.missing == ["at least 30 rows"]
    assert anomaly_scale.missing_kinds == ["scale"]

    # scale (too few expected rows per group) -> needs_data
    group_scale = _gate("group_comparison", sparse_groups)
    assert group_scale.missing == ["at least 5 expected rows per group"]
    assert group_scale.missing_kinds == ["scale"]

    # data (absent prerequisite: named target column) -> needs_data
    prediction_data = _gate("outcome_prediction", text_only, target_column="unknown")
    assert prediction_data.missing == ["target column: unknown"]
    assert prediction_data.missing_kinds == ["data"]

    # every result keeps missing and missing_kinds the same length
    for result in (anomaly_structural, anomaly_scale, group_scale, prediction_data):
        assert len(result.missing) == len(result.missing_kinds)


def test_unsupported_method_families_explain_their_constraints() -> None:
    with_time = _profile(
        rows=100,
        columns=[
            _column("occurred_at", "datetime", unique_count=100),
            _column("amount", "numeric", unique_count=90),
        ],
    )
    without_time = _profile(
        rows=100,
        columns=[_column("amount", "numeric", unique_count=90)],
    )

    forecast = _evaluate(with_time, mode="forecast")
    no_time = _evaluate(without_time, mode="forecast")
    segmentation = _evaluate(with_time, mode="segmentation")
    causal = _evaluate(with_time, mode="causal_experiment")

    assert forecast.status == "constrained"
    assert forecast.reasons == [
        "forecast method family not implemented yet; start with a descriptive trend"
    ]
    assert no_time.status == "needs_data" and no_time.missing == ["time column"]
    assert segmentation.status == "constrained"
    assert causal.status == "constrained"


def test_none_mode_uses_deterministic_descriptive_fallback_without_llm_text() -> None:
    profile = _profile(rows=1, columns=[])
    result = evaluate_feasibility(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode=None,
            target_column="LLM-authored target",
        )
    )

    assert result.status == "ready"
    assert result.method_id == "descriptive_sql"
    assert "LLM-authored" not in " ".join([*result.reasons, *result.missing])


def test_template_candidates_receive_conservative_registry_defaults(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "region,revenue,note\n"
        + "\n".join(
            f"Region {index // 5},{index + 1},{'' if index < 5 else 'ok'}"
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project", session_id="run")

    candidate_set = discover_question_candidates([loaded], profile_artifacts=[profile])

    assert candidate_set.candidates
    for candidate in candidate_set.candidates:
        assert candidate.business_decision == ""
        assert candidate.value_hypothesis == ""
        assert candidate.feasibility is not None
        if candidate.template_id == "group_difference":
            assert candidate.analysis_mode == "diagnostic"
            assert candidate.candidate_methods == ["group_comparison"]
        else:
            assert candidate.analysis_mode == "descriptive"
            assert candidate.candidate_methods == ["descriptive_sql"]


def test_auto_selection_excludes_failed_feasibility_but_keeps_legacy_candidates() -> None:
    candidate_set = QuestionCandidateSet(
        candidates=[
            _question("needs_data", 0.99, status="needs_data"),
            _question("unsuitable", 0.98, status="unsuitable"),
            _question("legacy", 0.90),
            _question("ready", 0.80, status="ready"),
        ]
    )

    selected = select_auto_execution_candidates(candidate_set, limit=4)

    assert [candidate.question_id for candidate in selected] == ["legacy", "ready"]


def test_llm_card_mapping_and_invalid_mode_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "customer_id,churned,spend\n"
        + "\n".join(f"{index},{index % 2},{index * 3}" for index in range(60))
        + "\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project", session_id="run")
    llm = _FakeQuestionLLM(
        {
            "questions": [
                {
                    "question_en": "Which customers should receive retention outreach?",
                    "target_datasets": ["customers.csv"],
                    "llm_business_relevance": 0.9,
                    "llm_actionability": 0.8,
                    "business_decision": "Prioritize retention outreach",
                    "value_hypothesis": "Reduce avoidable churn",
                    "analysis_mode": "prediction",
                    "success_criterion": "Outperform a simple baseline",
                    "risks": ["Outcome leakage"],
                    "data_requirements": ["Observed churn label"],
                    "target_column": "churned",
                },
                {
                    "question_en": "What is the customer mix?",
                    "target_datasets": ["customers.csv"],
                    "llm_business_relevance": 0.5,
                    "llm_actionability": 0.4,
                    "analysis_mode": "invented_mode",
                },
                {
                    "question_en": "What experiment would test a retention offer?",
                    "target_datasets": ["customers.csv"],
                    "llm_business_relevance": 0.8,
                    "llm_actionability": 0.7,
                    "analysis_mode": "causal_experiment",
                },
            ]
        }
    )

    result = propose_llm_question_candidates([profile], llm=cast(Any, llm), max_questions=3)

    assert result.error is None
    prediction, invalid, causal = result.candidates
    assert prediction.business_decision == "Prioritize retention outreach"
    assert prediction.value_hypothesis == "Reduce avoidable churn"
    assert prediction.analysis_mode == "prediction"
    assert prediction.success_criterion == "Outperform a simple baseline"
    assert prediction.risks == ["Outcome leakage"]
    assert prediction.data_requirements == ["Observed churn label"]
    assert prediction.candidate_methods == ["outcome_prediction"]
    assert prediction.feasibility is not None
    assert prediction.feasibility.status == "constrained"
    assert invalid.analysis_mode is None
    assert invalid.candidate_methods == ["descriptive_sql"]
    assert invalid.feasibility is not None and invalid.feasibility.status == "ready"
    assert causal.proposed_action == "design_experiment"


def _gate(
    method_id: str,
    profile: DatasetProfile,
    *,
    target_column: str | None = None,
) -> MethodGateResult:
    return METHOD_REGISTRY[method_id].gate(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode=None,
            target_column=target_column,
        )
    )


def _evaluate(
    profile: DatasetProfile,
    *,
    mode: Any,
    target_column: str | None = None,
) -> OpportunityFeasibility:
    return evaluate_feasibility(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode=cast(Any, mode),
            target_column=target_column,
        )
    )


def _profile(*, rows: int, columns: list[ColumnProfile]) -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=rows,
        columns=len(columns),
        column_names=[column.name for column in columns],
        dtypes={column.name: column.dtype for column in columns},
        missing_values={column.name: column.missing_count for column in columns},
        missing_percent={column.name: column.missing_percent for column in columns},
        numeric_columns=[column.name for column in columns if column.semantic_type == "numeric"],
        categorical_columns=[
            column.name
            for column in columns
            if column.semantic_type in {"categorical", "boolean"}
        ],
        columns_detail=columns,
    )


def _column(
    name: str,
    semantic_type: Any,
    *,
    unique_count: int,
    missing_percent: float = 0.0,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object" if semantic_type != "numeric" else "float64",
        semantic_type=cast(Any, semantic_type),
        missing_count=0,
        missing_percent=missing_percent,
        unique_count=unique_count,
        unique_percent=0.0,
    )


def _question(
    question_id: str,
    deterministic_score: float,
    *,
    status: Any | None = None,
) -> QuestionCandidate:
    feasibility = (
        OpportunityFeasibility(status=cast(Any, status), method_id="descriptive_sql")
        if status is not None
        else None
    )
    return QuestionCandidate(
        question_id=question_id,
        question_en=f"Question {question_id}?",
        origin="template",
        template_id=question_id,
        target_datasets=["orders.csv"],
        sql_template="select 1",
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=deterministic_score,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=deterministic_score,
        ),
        feasibility=feasibility,
    )

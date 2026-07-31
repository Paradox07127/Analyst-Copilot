from typing import Literal

from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.schemas.questions import AnalysisMode, QuestionCandidate, QuestionScore
from eda_platform.tools.investigation_methods import select_investigation_method


def test_adapter_verdict_equals_registry_at_anomaly_threshold() -> None:
    for rows in (20, 30):
        profile = _profile(rows=rows, columns=[_column("amount", "numeric", rows)])
        candidate = _candidate(analysis_mode="anomaly")

        selection = select_investigation_method(candidate, {profile.name: profile})
        canonical = evaluate_feasibility(
            MethodGateContext(
                profiles=[profile],
                target_datasets=candidate.target_datasets,
                analysis_mode=candidate.analysis_mode,
                target_column=None,
            )
        )

        assert selection.feasibility == canonical
        assert selection.execution_ready is (canonical.status in {"ready", "constrained"})
        assert any("30" in gate.reason for gate in selection.validation_gates)


def test_needs_data_and_unsuitable_are_never_execution_ready() -> None:
    short_numeric = _profile(rows=20, columns=[_column("amount", "numeric", 20)])
    text_only = _profile(rows=100, columns=[_column("note", "text", 80)])

    needs_data = select_investigation_method(
        _candidate(analysis_mode="anomaly"),
        {short_numeric.name: short_numeric},
    )
    unsuitable = select_investigation_method(
        _candidate(analysis_mode="anomaly"),
        {text_only.name: text_only},
    )

    assert needs_data.feasibility.status == "needs_data"
    assert unsuitable.feasibility.status == "unsuitable"
    assert not needs_data.execution_ready
    assert not unsuitable.execution_ready


def test_template_route_with_sql_accepts_ready_or_constrained_registry_verdict() -> None:
    profile = _profile(
        rows=40,
        columns=[
            _column("occurred_at", "datetime", 40),
            _column("amount", "numeric", 40),
        ],
    )
    candidate = _candidate(
        analysis_mode="forecast",
        template_id="trend",
        sql_template='SELECT * FROM "orders.csv"',
    )

    selection = select_investigation_method(candidate, {profile.name: profile})

    assert selection.feasibility.status == "constrained"
    assert selection.execution_ready
    assert selection.allowed_tools == ["read_only_sql"]


def test_none_mode_dispatches_to_descriptive_registry_fallback() -> None:
    profile = _profile(rows=1, columns=[])
    selection = select_investigation_method(
        _candidate(analysis_mode=None),
        {profile.name: profile},
    )

    assert selection.feasibility == evaluate_feasibility(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode="descriptive",
            target_column=None,
        )
    )
    assert selection.method_family == "descriptive_analysis"


_SemanticType = Literal[
    "numeric", "categorical", "datetime", "id", "boolean", "text", "unknown"
]


def _candidate(
    *,
    analysis_mode: AnalysisMode | None,
    template_id: str | None = None,
    sql_template: str | None = None,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_1",
        question_en="What should be investigated?",
        origin="template" if template_id else "llm",
        template_id=template_id,
        target_datasets=["orders.csv"],
        sql_template=sql_template,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
        analysis_mode=analysis_mode,
    )


def _profile(*, rows: int, columns: list[ColumnProfile]) -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=rows,
        columns=len(columns),
        column_names=[column.name for column in columns],
        dtypes={column.name: column.dtype for column in columns},
        missing_values={column.name: 0 for column in columns},
        missing_percent={column.name: 0.0 for column in columns},
        numeric_columns=[column.name for column in columns if column.semantic_type == "numeric"],
        categorical_columns=[],
        columns_detail=columns,
    )


def _column(name: str, semantic_type: _SemanticType, unique_count: int) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        semantic_type=semantic_type,
        missing_count=0,
        missing_percent=0.0,
        unique_count=unique_count,
        unique_percent=100.0,
    )

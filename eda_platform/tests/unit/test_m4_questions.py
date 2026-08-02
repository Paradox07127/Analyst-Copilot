from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import pytest
from pydantic import BaseModel

import eda_platform.drivers.question_exec as question_exec_driver
from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.core.budget import BudgetExceeded, SessionBudgetPolicy
from eda_platform.core.column_roles import ColumnRoleName, infer_column_roles
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.question_exec import (
    run_question_batch,
    select_auto_execution_candidates,
)
from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    DatasetProfile,
)
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipColumnPair,
    RelationshipSignals,
)
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.question_discovery import discover_question_candidates
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    validate_relationships,
)

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"

T = TypeVar("T", bound=BaseModel)


class FakeQuestionLLM:
    def __init__(self, result: BaseModel | dict[str, Any] | list[dict[str, Any]]) -> None:
        self.results = result if isinstance(result, list) else [result]
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "schema": schema.__name__, "payload": payload})
        index = min(self.call_count, len(self.results) - 1)
        self.call_count += 1
        result = self.results[index]
        if isinstance(result, BaseModel):
            return cast(T, result)
        return schema.model_validate(result)

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def test_llm_question_scores_on_ten_point_scale_are_retried_with_guard_feedback(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    llm = FakeQuestionLLM(
        [
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 8,
                        "llm_actionability": 9,
                    }
                ]
            },
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 0.8,
                        "llm_actionability": 0.9,
                    }
                ]
            },
        ]
    )

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    assert result.error is None
    assert len(result.candidates) == 1
    assert len(llm.calls) == 2
    assert "previous_error" not in llm.calls[0]["payload"]
    assert "llm_business_relevance" in llm.calls[1]["payload"]["previous_error"]
    assert "0.8 not 8" in llm.calls[1]["payload"]["previous_error"]
    score = result.candidates[0].score
    assert score.llm_business_relevance == 0.8
    assert score.llm_actionability == 0.9
    instructions = llm.calls[0]["payload"]["instructions"]
    assert "[0.0, 1.0]" in instructions
    assert "Example" in instructions and "0.8" in instructions


def test_llm_questions_receive_business_context_and_keep_friendly_dataset_labels(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_orders")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    llm = FakeQuestionLLM(
        {
            "questions": [
                {
                    "question_en": "Which regions are driving sales growth?",
                    "target_datasets": ["orders.csv"],
                    "dataset_display_names": {"orders.csv": "Sales orders"},
                    "llm_business_relevance": 0.9,
                    "llm_actionability": 0.8,
                }
            ]
        }
    )

    result = propose_llm_question_candidates(
        [profile],
        llm=llm,
        business_context="Regional retail sales data used to improve growth planning.",
        max_questions=1,
    )

    assert result.error is None
    assert result.candidates[0].dataset_display_names == {"orders.csv": "Sales orders"}
    assert llm.calls[0]["payload"]["business_context"] == (
        "Regional retail sales data used to improve growth planning."
    )
    assert "never raw dataset file names" in llm.calls[0]["payload"]["instructions"]


def test_llm_question_validation_error_retries_with_previous_error(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("region,revenue\nEast,10\nWest,20\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    llm = FakeQuestionLLM(
        [
            {
                "questions": [
                    {
                        "question_en": "Malformed question",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": "high",
                        "llm_actionability": 7,
                    }
                ]
            },
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 0.7,
                        "llm_actionability": 0.6,
                    }
                ]
            },
        ]
    )

    result = propose_llm_question_candidates([profile], llm=llm, max_questions=1)

    assert result.error is None
    assert len(result.candidates) == 1
    assert len(llm.calls) == 2
    assert "previous_error" not in llm.calls[0]["payload"]
    assert "llm_business_relevance" in llm.calls[1]["payload"]["previous_error"]
    assert result.candidates[0].score.llm_business_relevance == 0.7


def test_trivial_correlation_pairs_are_flagged_and_skipped(tmp_path: Path) -> None:
    csv_path = tmp_path / "football.csv"
    csv_path.write_text(
        "home_possession,away_possession,shots\n60,40,10\n55,45,12\n40,60,8\n48,52,9\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_football")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    analysis_artifacts = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )
    correlation_table = next(
        AnalysisTable.model_validate(artifact.payload)
        for artifact in analysis_artifacts
        if artifact.payload["kind"] == "correlation"
    )
    possession_row = next(
        row
        for row in correlation_table.rows
        if {row["column_a"], row["column_b"]} == {"home_possession", "away_possession"}
    )

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
        quality_artifacts=[scan_quality(profile, project_id="project_demo", session_id="run_demo")],
        analysis_artifacts=analysis_artifacts,
    )

    assert possession_row["is_trivial_pair"] is True
    assert candidates.trivial_dropped == 1
    assert all(
        {"home_possession", "away_possession"} != set(candidate.question_en.split())
        for candidate in candidates.candidates
    )
    assert all(
        not (
            candidate.template_id == "correlation_probe"
            and {"home_possession", "away_possession"}.issubset(set(candidate.sql_template or ""))
        )
        for candidate in candidates.candidates
    )


def test_perfect_correlation_and_mid_token_rescale_names_are_trivial(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "football.csv"
    csv_path.write_text(
        "shots,passes,assists_against,assists_per90_against\n"
        "1,2,10,1\n"
        "2,4,20,2\n"
        "3,6,30,3\n"
        "4,8,40,4\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_football")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    analysis_artifacts = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )
    correlation_table = next(
        AnalysisTable.model_validate(artifact.payload)
        for artifact in analysis_artifacts
        if artifact.payload["kind"] == "correlation"
    )
    perfect_different_name = next(
        row
        for row in correlation_table.rows
        if {row["column_a"], row["column_b"]} == {"shots", "passes"}
    )
    per90_mid_token = next(
        row
        for row in correlation_table.rows
        if {row["column_a"], row["column_b"]} == {"assists_against", "assists_per90_against"}
    )

    assert perfect_different_name["pearson"] == 1.0
    assert perfect_different_name["is_trivial_pair"] is True
    assert per90_mid_token["pearson"] == 1.0
    assert per90_mid_token["is_trivial_pair"] is True


def test_near_constant_columns_are_excluded_from_question_templates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "near_constant.csv"
    rows = ["segment,near_metric,other_metric"]
    near_values = [100.0, 100.1, 99.9, 100.05]
    for index in range(40):
        rows.append(f"S{index % 4},{near_values[index % len(near_values)]},{index + 1}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_near_constant")
    artifacts = _base_artifacts([loaded])

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=artifacts["profiles"],
        quality_artifacts=artifacts["quality"],
        analysis_artifacts=artifacts["analysis"],
    )

    assert candidates.trivial_dropped > 0
    assert all(
        "near_metric" not in (candidate.sql_template or "")
        for candidate in candidates.candidates
        if candidate.template_id in {"correlation_probe", "group_difference"}
    )


def test_template_instantiation_keeps_relationships_as_references() -> None:
    sales = load_csv(GOLDEN_DATA / "time_series_sales.csv", dataset_id="ds_sales")
    sales_artifacts = _base_artifacts([sales])

    sales_candidates = discover_question_candidates(
        [sales],
        profile_artifacts=sales_artifacts["profiles"],
        quality_artifacts=sales_artifacts["quality"],
        analysis_artifacts=sales_artifacts["analysis"],
    )

    trend = _candidate_by_template(sales_candidates, "trend")
    assert trend.target_datasets == ["time_series_sales.csv"]
    assert trend.sql_template is not None
    assert "order_date" in trend.sql_template
    assert "limit" in trend.sql_template.lower()

    ecommerce = _load_ecommerce()
    ecommerce_artifacts = _base_artifacts(ecommerce)
    engine = _engine(ecommerce)
    relationships = discover_relationship_candidates(ecommerce, engine)
    validations = validate_relationships(relationships, engine)

    ecommerce_candidates = discover_question_candidates(
        ecommerce,
        profile_artifacts=ecommerce_artifacts["profiles"],
        quality_artifacts=ecommerce_artifacts["quality"],
        analysis_artifacts=ecommerce_artifacts["analysis"],
        relationship_candidates=relationships,
        relationship_validations=validations,
    )
    assert all(not candidate.required_relations for candidate in ecommerce_candidates.candidates)


def test_llm_candidates_replace_template_candidates_when_requested() -> None:
    sales = load_csv(GOLDEN_DATA / "time_series_sales.csv", dataset_id="ds_sales")
    artifacts = _base_artifacts([sales])
    llm_candidate = QuestionCandidate(
        question_id="q_llm_only",
        question_en="Which sales periods deserve management attention?",
        origin="llm",
        target_datasets=["time_series_sales.csv"],
        dataset_display_names={"time_series_sales.csv": "Sales transactions"},
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
            llm_business_relevance=0.9,
            llm_actionability=0.8,
        ),
    )

    candidates = discover_question_candidates(
        [sales],
        profile_artifacts=artifacts["profiles"],
        quality_artifacts=artifacts["quality"],
        analysis_artifacts=artifacts["analysis"],
        llm_candidates=[llm_candidate],
        include_template_candidates=False,
    )

    assert [candidate.question_id for candidate in candidates.candidates] == ["q_llm_only"]


def test_deterministic_scoring_and_ordering_are_byte_stable() -> None:
    datasets = _load_ecommerce()
    artifacts = _base_artifacts(datasets)
    engine = _engine(datasets)
    relationships = discover_relationship_candidates(datasets, engine)
    validations = validate_relationships(relationships, engine)

    first = discover_question_candidates(
        datasets,
        profile_artifacts=artifacts["profiles"],
        quality_artifacts=artifacts["quality"],
        analysis_artifacts=artifacts["analysis"],
        relationship_candidates=relationships,
        relationship_validations=validations,
    )
    second = discover_question_candidates(
        datasets,
        profile_artifacts=artifacts["profiles"],
        quality_artifacts=artifacts["quality"],
        analysis_artifacts=artifacts["analysis"],
        relationship_candidates=relationships,
        relationship_validations=validations,
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert [
        (candidate.score.deterministic_score, candidate.question_id)
        for candidate in first.candidates
    ] == sorted(
        [
            (candidate.score.deterministic_score, candidate.question_id)
            for candidate in first.candidates
        ],
        key=lambda item: (-item[0], item[1]),
    )


def test_auto_execution_gates_reject_medium_join_and_cap_top_three() -> None:
    medium_relation = _relationship_candidate(
        confidence="medium",
        auto_adopted=False,
        label_seed="medium",
    )
    candidates = QuestionCandidateSet(
        candidates=[
            _question("eligible_1", 0.95, template_id="trend"),
            _question("eligible_2", 0.90, template_id="group_difference"),
            _question("eligible_3", 0.85, template_id="quality_missing"),
            _question("eligible_4", 0.80, template_id="correlation_probe"),
            _question(
                "medium_join",
                0.99,
                required_relations=[medium_relation.pair.label()],
                join_risk=0.75,
            ),
        ]
    )

    selected = select_auto_execution_candidates(
        candidates,
        relationship_candidates=RelationshipCandidateSet(candidates=[medium_relation]),
        limit=3,
    )

    assert [candidate.question_id for candidate in selected] == [
        "eligible_1",
        "eligible_2",
        "eligible_3",
    ]


def test_auto_execution_never_selects_a_question_that_requires_a_join() -> None:
    relation = _relationship_candidate(
        confidence="high",
        auto_adopted=True,
        label_seed="cross",
    )
    candidates = QuestionCandidateSet(
        candidates=[
            _question(
                "corr_1",
                0.99,
                template_id="correlation_probe",
                target_datasets=["teams.csv"],
            ),
            _question(
                "corr_2",
                0.98,
                template_id="correlation_probe",
                target_datasets=["players.csv"],
            ),
            _question(
                "corr_3",
                0.97,
                template_id="correlation_probe",
                target_datasets=["matches.csv"],
            ),
            _question(
                "group_1",
                0.96,
                template_id="group_difference",
                target_datasets=["players.csv"],
            ),
            _question(
                "cross_1",
                0.70,
                template_id="cross_table_aggregation",
                target_datasets=["orders.csv", "products.csv"],
                required_relations=[relation.pair.label()],
            ),
        ]
    )

    selected = select_auto_execution_candidates(
        candidates,
        relationship_candidates=RelationshipCandidateSet(candidates=[relation]),
        limit=3,
    )

    assert all(not candidate.required_relations for candidate in selected)
    assert len({candidate.template_id for candidate in selected}) >= 2
    assert sum(candidate.template_id == "correlation_probe" for candidate in selected) == 1


def test_auto_eda_executes_top_questions_without_automatic_joins(tmp_path: Path) -> None:
    result = run_auto_eda(
        [
            GOLDEN_DATA / "ecommerce_orders.csv",
            GOLDEN_DATA / "ecommerce_customers.csv",
            GOLDEN_DATA / "ecommerce_products.csv",
            GOLDEN_DATA / "ecommerce_marketing.csv",
        ],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="ecommerce_questions",
    )
    qexec_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    assert qexec_artifacts
    candidate_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidates_by_id = {
        candidate.question_id: candidate
        for candidate in QuestionCandidateSet.model_validate(candidate_artifact.payload).candidates
    }
    executed_candidates = [
        candidates_by_id[artifact.payload["question_id"]] for artifact in qexec_artifacts
    ]

    # Auto EDA may answer safe single-table questions, but joins must originate
    # from a user-selected question rather than relationship discovery.
    assert all(not candidate.required_relations for candidate in executed_candidates)
    assert all(
        " join " not in (artifact.payload.get("sql") or "").lower() for artifact in qexec_artifacts
    )
    assert any(
        artifact.payload["status"] == "succeeded" and artifact.payload["findings"]
        for artifact in qexec_artifacts
    )


def test_second_offset_column_named_time_never_becomes_a_trend_axis(tmp_path: Path) -> None:
    """A "Time" column holding 0-29 offsets is numeric, and stays out of trends.

    The profiler used to type it datetime on the name alone and rely on the role
    layer to veto the trend question; since 2026-07-22 values beat naming, so the
    veto is now defence in depth rather than the only guard.
    """
    path = tmp_path / "offsets.csv"
    path.write_text(
        "Time,V1\n" + "\n".join(f"{index}.0,{index / 10:.1f}" for index in range(30)),
        encoding="utf-8",
    )
    loaded = load_csv(path, dataset_id="ds_offsets")
    profile_artifact = profile_dataset(loaded, project_id="project", session_id="run")
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    time_profile = next(column for column in profile.columns_detail if column.name == "Time")
    assert time_profile.semantic_type == "numeric"
    role_set = infer_column_roles(profile, frame=loaded.frame)
    time_role = role_set.role_of("Time")
    assert time_role is not None
    assert time_role.role is ColumnRoleName.MEASURE

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile_artifact],
        column_role_sets={profile.name: role_set},
    )

    assert not any(
        candidate.template_id == "trend"
        and "Time" in candidate.referenced_columns.get(profile.name, [])
        for candidate in candidates.candidates
    )


def test_verified_code_role_never_fills_group_metric_slot(tmp_path: Path) -> None:
    path = tmp_path / "customers.csv"
    rows = ["customer_state,customer_zip_code_prefix,order_value"]
    states = ("SP", "RJ", "MG", "BA")
    for index in range(40):
        rows.append(f"{states[index % 4]},{10000 + index},{20 + index * 1.5}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(path, dataset_id="ds_customers")
    profile_artifact = profile_dataset(loaded, project_id="project", session_id="run")
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    role_set = infer_column_roles(profile, frame=loaded.frame)
    zip_role = role_set.role_of("customer_zip_code_prefix")
    assert zip_role is not None and zip_role.role is ColumnRoleName.CODE

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile_artifact],
        column_role_sets={profile.name: role_set},
    ).candidates
    group_candidates = [
        candidate for candidate in candidates if candidate.template_id == "group_difference"
    ]

    assert group_candidates
    assert any("average order_value" in candidate.question_en for candidate in group_candidates)
    assert not any(
        "average customer_zip_code_prefix" in candidate.question_en
        for candidate in group_candidates
    )


def test_deferred_discovery_preserves_default_executed_question_outputs(
    tmp_path: Path,
) -> None:
    paths = [
        GOLDEN_DATA / "ecommerce_orders.csv",
        GOLDEN_DATA / "ecommerce_customers.csv",
        GOLDEN_DATA / "ecommerce_products.csv",
        GOLDEN_DATA / "ecommerce_marketing.csv",
    ]
    deferred = run_auto_eda(
        paths,
        workspace=tmp_path / "deferred",
        project_id="deferred",
        session_id="run",
    )
    eager = run_auto_eda(
        paths,
        workspace=tmp_path / "eager",
        project_id="eager",
        session_id="run",
        relationship_discovery="eager",
    )

    def executed(result: AutoEDAResult) -> set[tuple[str, str, str]]:
        return {
            (
                str(artifact.payload["question_id"]),
                str(artifact.payload.get("outcome")),
                str(artifact.payload.get("sql")),
            )
            for artifact in result.artifacts
            if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
        }

    assert executed(deferred) == executed(eager)


def test_run_question_batch_executes_template_and_fails_llm_route_without_llm(
    tmp_path: Path,
) -> None:
    source = run_auto_eda(
        [
            GOLDEN_DATA / "ecommerce_orders.csv",
            GOLDEN_DATA / "ecommerce_customers.csv",
            GOLDEN_DATA / "ecommerce_products.csv",
            GOLDEN_DATA / "ecommerce_marketing.csv",
        ],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="source_questions",
    )
    source_qcand = next(
        artifact
        for artifact in source.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate_set = QuestionCandidateSet.model_validate(source_qcand.payload)
    template_question = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.origin == "template" and candidate.sql_template is not None
    )

    first_batch = run_question_batch(
        project_id="project_demo",
        source_session_id=source.session_id,
        question_ids=[template_question.question_id],
        workspace=tmp_path / "workspace",
        session_id="batch_template",
    )

    assert first_batch.session_id == "batch_template"
    assert any(
        artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
        and artifact.payload["status"] == "succeeded"
        for artifact in first_batch.artifacts
    )
    batch_metrics = summarize_session(
        ArtifactStore(tmp_path / "workspace"), "project_demo", first_batch.session_id
    )
    assert batch_metrics.duration_seconds > 0
    assert [step.step_name for step in batch_metrics.steps] == ["question_batch"]
    assert batch_metrics.steps[0].duration_seconds > 0

    llm_question = _question("llm_question", 0.7, origin="llm", sql_template=None)
    candidate_set.candidates.append(llm_question)
    source_qcand.payload = candidate_set.model_dump(mode="json")
    ArtifactStore(tmp_path / "workspace").save_artifact(source_qcand)

    second_batch = run_question_batch(
        project_id="project_demo",
        source_session_id=source.session_id,
        question_ids=[llm_question.question_id],
        workspace=tmp_path / "workspace",
        session_id="batch_llm_without_client",
    )
    qexec = next(
        artifact
        for artifact in second_batch.artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )

    assert qexec.payload["status"] == "failed"
    assert "LLM client is required" in qexec.payload["error"]


def test_question_batch_budget_failure_marks_run_failed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = run_auto_eda(
        [GOLDEN_DATA / "time_series_sales.csv"],
        workspace=workspace,
        project_id="project_budget",
        session_id="source_budget",
    )
    candidate_artifact = next(
        artifact
        for artifact in source.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate = QuestionCandidateSet.model_validate(candidate_artifact.payload).candidates[0]

    with pytest.raises(BudgetExceeded):
        run_question_batch(
            project_id="project_budget",
            source_session_id=source.session_id,
            question_ids=[candidate.question_id],
            workspace=workspace,
            session_id="batch_budget_exhausted",
            budget_policy=SessionBudgetPolicy(max_wall_seconds=0),
        )

    store = ArtifactStore(workspace)
    assert store.get_session_status("batch_budget_exhausted") == "failed"
    events = store.list_trace_events(
        project_id="project_budget",
        session_id="batch_budget_exhausted",
    )
    assert any(
        event.event_type == "step_failed" and event.name == "question_batch" for event in events
    )


def test_question_batch_cancellation_stops_before_next_question_and_terminal_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source = run_auto_eda(
        [GOLDEN_DATA / "ecommerce_orders.csv"],
        workspace=workspace,
        project_id="project_cancel",
        session_id="source_cancel",
    )
    candidate_artifact = next(
        artifact
        for artifact in source.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidates = QuestionCandidateSet.model_validate(candidate_artifact.payload).candidates
    question_ids = [candidate.question_id for candidate in candidates[:2]]
    assert len(question_ids) == 2
    entered: list[str] = []

    def fake_execute(candidate: QuestionCandidate, **kwargs: object) -> list[Artifact]:
        entered.append(candidate.question_id)
        return []

    monkeypatch.setattr(
        question_exec_driver,
        "execute_question_candidate",
        fake_execute,
    )

    with pytest.raises(SessionCancelled):
        run_question_batch(
            project_id="project_cancel",
            source_session_id=source.session_id,
            question_ids=question_ids,
            workspace=workspace,
            session_id="batch_cancelled",
            generate_report=False,
            cancel_check=lambda: bool(entered),
        )

    assert entered == [question_ids[0]]
    store = ArtifactStore(workspace)
    assert store.get_session_status("batch_cancelled") == "running"
    assert not any(
        event.event_type == "step_completed"
        for event in store.list_trace_events(
            project_id="project_cancel",
            session_id="batch_cancelled",
        )
    )


def test_dedup_drops_duplicate_questions() -> None:
    loaded = load_csv(GOLDEN_DATA / "time_series_sales.csv", dataset_id="ds_sales")
    artifacts = _base_artifacts([loaded])
    duplicate = QuestionCandidate(
        question_id="llm_duplicate",
        question_en="How is amount trending over order_date in time_series_sales.csv?",
        origin="llm",
        target_datasets=["time_series_sales.csv"],
        sql_template=None,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=1.0,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=1.0,
            llm_business_relevance=0.9,
            llm_actionability=0.8,
        ),
    )

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=artifacts["profiles"],
        quality_artifacts=artifacts["quality"],
        analysis_artifacts=artifacts["analysis"],
        llm_candidates=[duplicate],
    )

    assert candidates.dedup_dropped == 1
    assert (
        sum(
            1
            for candidate in candidates.candidates
            if candidate.question_en
            == "How is amount trending over order_date in time_series_sales.csv?"
        )
        == 1
    )


def _load_ecommerce() -> list[LoadedDataset]:
    return [
        load_csv(GOLDEN_DATA / "ecommerce_orders.csv", dataset_id="ds_orders"),
        load_csv(GOLDEN_DATA / "ecommerce_customers.csv", dataset_id="ds_customers"),
        load_csv(GOLDEN_DATA / "ecommerce_products.csv", dataset_id="ds_products"),
        load_csv(GOLDEN_DATA / "ecommerce_marketing.csv", dataset_id="ds_marketing"),
    ]


def _base_artifacts(datasets: list[LoadedDataset]) -> dict[str, list[Artifact]]:
    profiles = [
        profile_dataset(dataset, project_id="project_demo", session_id="run_demo")
        for dataset in datasets
    ]
    quality = [
        scan_quality(profile, project_id="project_demo", session_id="run_demo")
        for profile in profiles
    ]
    analysis = [
        artifact
        for dataset, profile in zip(datasets, profiles, strict=True)
        for artifact in create_analysis_tables(
            dataset,
            profile,
            project_id="project_demo",
            session_id="run_demo",
        )
    ]
    return {"profiles": profiles, "quality": quality, "analysis": analysis}


def _engine(datasets: list[LoadedDataset]) -> DuckDBQueryEngine:
    engine = DuckDBQueryEngine()
    for loaded in datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    return engine


def _candidate_by_template(
    candidate_set: QuestionCandidateSet,
    template_id: str,
) -> QuestionCandidate:
    for candidate in candidate_set.candidates:
        if candidate.template_id == template_id:
            return candidate
    raise AssertionError(f"missing template question: {template_id}")


def _question(
    question_id: str,
    score: float,
    *,
    origin: str = "template",
    template_id: str | None = None,
    target_datasets: list[str] | None = None,
    required_relations: list[str] | None = None,
    sql_template: str | None = "select 1 as value",
    join_risk: float = 0.0,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        question_en=f"Question {question_id}?",
        origin=cast(Any, origin),
        template_id=template_id or ("unit_test" if origin == "template" else None),
        target_datasets=target_datasets or ["unit.csv"],
        required_relations=required_relations or [],
        sql_template=sql_template,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=score,
            quality_risk=0.0,
            join_risk=join_risk,
            deterministic_score=score,
        ),
    )


def _relationship_candidate(
    *,
    confidence: str,
    auto_adopted: bool,
    label_seed: str,
) -> RelationshipCandidate:
    return RelationshipCandidate(
        pair=RelationshipColumnPair(
            left_dataset_id=f"ds_left_{label_seed}",
            left_dataset_name=f"left_{label_seed}.csv",
            left_columns=["left_id"],
            right_dataset_id=f"ds_right_{label_seed}",
            right_dataset_name=f"right_{label_seed}.csv",
            right_columns=["right_id"],
        ),
        signals=RelationshipSignals(
            name_similarity=1.0,
            type_compatible=True,
            overlap_left_in_right=0.8,
            overlap_right_in_left=0.8,
            right_unique_rate=0.95,
            left_null_rate=0.0,
            right_null_rate=0.0,
        ),
        ensemble_score=0.8,
        confidence=cast(Any, confidence),
        auto_adopted=auto_adopted,
    )


def test_template_sql_survives_physically_varchar_numeric_columns(tmp_path: Path) -> None:
    """Football-style columns profile as numeric but stay VARCHAR in DuckDB
    (e.g. age "27-158"); template SQL must try_cast instead of crashing."""
    csv_path = tmp_path / "squad.csv"
    csv_path.write_text(
        "position,rating\n"
        "GK,12\n"
        "GK,15\n"
        "GK,14\n"
        "DF,18\n"
        "DF,21\n"
        "DF,19\n"
        "MF,24\n"
        "MF,26\n"
        "MF,23\n"
        "FW,30\n"
        "FW,28\n"
        "FW,n/a\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_varchar_metric")
    loaded.frame["rating"] = loaded.frame["rating"].astype(str)
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
    )
    group_candidates = [
        candidate
        for candidate in candidates.candidates
        if candidate.template_id == "group_difference" and candidate.sql_template
    ]
    assert group_candidates, "expected a group-difference template question"
    sql = group_candidates[0].sql_template
    assert sql is not None and "try_cast" in sql

    engine = DuckDBQueryEngine()
    engine.register_frame("ds_varchar_metric", loaded.frame)
    frame = engine.execute_select(sql)
    assert not frame.empty
    avg_column = next(column for column in frame.columns if column.startswith("avg_"))
    assert bool(cast(pd.Series, frame[avg_column]).notna().any())


def _ranked_sql_artifact(
    columns: list[str],
    rows: list[dict[str, Any]],
    sql: str = "select ... order by separation desc",
) -> Artifact:
    from eda_platform.schemas.artifacts import SqlResult

    payload = SqlResult(
        sql=sql,
        columns=columns,
        dtypes={column: "float64" for column in columns},
        rows_preview=rows,
        row_count=len(rows),
    ).model_dump(mode="json")
    return Artifact(
        id="sql_ranktest",
        type=ArtifactType.SQL_RESULT,
        project_id="p",
        session_id="r",
        payload=payload,
    )


def test_generic_finding_reads_the_whole_ranked_table() -> None:
    """A free-form ranked result names the winner + runners-up, not one number.

    Regression for the Level-0 depth fix: previously ``_generic_finding`` read
    only ``rows[0]`` and one numeric column, collapsing a per-component fraud
    separation into "The returned separation is 7.05" (winner identity lost).
    """
    from eda_platform.drivers.question_exec import _findings
    from eda_platform.tools.report_validator import _numbers_from_text

    rows = [
        {"component": "V3", "separation": 7.0455},
        {"component": "V14", "separation": 6.9838},
        {"component": "V17", "separation": 6.6774},
        {"component": "V12", "separation": 6.2702},
    ]
    sql_artifact = _ranked_sql_artifact(["component", "separation"], rows)
    candidate = _question("q_fraud", 0.5, origin="llm", template_id=None)

    findings = _findings(candidate, sql_artifact)

    assert len(findings) == 1
    text = findings[0].text
    # Names the leader and both runners-up (labels, not just a scalar).
    assert "V3" in text and "V14" in text and "V17" in text
    assert "The returned separation is" not in text  # old collapsed phrasing gone
    # Every number in the text is a cell of the metric column -> validator-safe.
    text_numbers = {round(value, 3) for value, _ in _numbers_from_text(text)}
    assert text_numbers == {7.045, 6.984, 6.677}
    # "V3"/"V14"/"V17" must not leak their digits as orphan numbers.
    assert 3.0 not in text_numbers and 14.0 not in text_numbers
    # Precise, per-leader cell evidence.
    locators = [ref.locator for ref in findings[0].evidence]
    assert locators == [
        "rows_preview[0].separation",
        "rows_preview[1].separation",
        "rows_preview[2].separation",
    ]


def test_ranked_finding_without_label_column_reports_leading_row() -> None:
    from eda_platform.drivers.question_exec import _findings

    rows = [{"score": 0.91}, {"score": 0.72}, {"score": 0.55}]
    sql_artifact = _ranked_sql_artifact(["score"], rows, sql="select ... order by score desc")
    candidate = _question("q_scores", 0.5, origin="llm", template_id=None)

    findings = _findings(candidate, sql_artifact)

    assert len(findings) == 1
    assert "0.91" in findings[0].text
    assert "ranked results" in findings[0].text

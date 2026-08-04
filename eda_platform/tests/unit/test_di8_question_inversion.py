"""DI8-E question-generation master/servant inversion.

LLM free questioning is primary; the template families degrade to a coverage
backstop checklist (metered, never silent). The DI8-B role layer gates the
template candidate pools (identifier/sequence columns never trend/correlate)
and multiplies statistical signal by business impact (row counters weigh 0).
LLM free-form questions carry an ``exploratory`` flag through execution into
findings for the report side's multiple-comparison defense. All LLM calls are
mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.question_agent import (
    _join_instruction,
    build_data_summary,
    propose_llm_question_candidates,
)
from eda_platform.core.column_roles import (
    ColumnRoleName,
    infer_column_roles,
)
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers import question_exec as question_exec_driver
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.plans import AnalysisPlan
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import (
    discover_question_candidates,
    question_coverage,
    rank_and_deduplicate_questions,
    score_question,
    select_backstop_candidates,
)

T = TypeVar("T", bound=BaseModel)


class FakeQuestionLLM:
    """Same mock contract as tests/unit/test_m4_questions.py."""

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


# --------------------------------------------------------------------------- #
# Fixtures: an Olist-shaped order-items table whose ``order_item_id`` is a
# per-order 1..n counter (the exact column the DI8 trigger run averaged).
# --------------------------------------------------------------------------- #
def _order_items_csv(tmp_path: Path) -> Path:
    rows = ["order_id,order_item_id,order_date,price,category"]
    categories = ["books", "toys", "garden", "sports"]
    day = 1
    for order_index in range(6):
        for item in range(1, 6):  # order_item_id strictly 1..5 per order
            price = 10.0 + order_index * 3.7 + item * 1.3
            rows.append(
                f"O{order_index:03d},{item},2026-01-{day:02d},"
                f"{price:.2f},{categories[(order_index + item) % len(categories)]}"
            )
            day = day % 28 + 1
    path = tmp_path / "order_items.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _loaded_with_profile(path: Path) -> tuple[LoadedDataset, Artifact]:
    loaded = load_csv(path, dataset_id=f"ds_{path.stem}")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    return loaded, profile


def _llm_question(question_en: str, **overrides: Any) -> QuestionCandidate:
    values: dict[str, Any] = {
        "question_id": f"q_llm_{abs(hash(question_en)) % 10_000}",
        "question_en": question_en,
        "origin": "llm",
        "target_datasets": ["order_items.csv"],
        "sql_template": None,
        "exploratory": True,
        "score": QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    }
    values.update(overrides)
    return QuestionCandidate.model_validate(values)


TREND_Q = "How is revenue trending over time?"
GROUP_Q = "Which segment groups have the highest revenue?"
QUALITY_Q = "How much data is missing from price?"


# --------------------------------------------------------------------------- #
# 1. Coverage classification + backstop selection (pure functions)
# --------------------------------------------------------------------------- #
def test_question_coverage_classifies_required_categories() -> None:
    assert "trend" in question_coverage(_llm_question(TREND_Q))
    assert "group_difference" in question_coverage(_llm_question(GROUP_Q))
    assert "quality" in question_coverage(_llm_question(QUALITY_Q))
    assert question_coverage(
        _llm_question("What explains customer churn?")
    ) == set()
    # analysis_mode also counts as coverage evidence.
    assert "trend" in question_coverage(
        _llm_question("Project next period revenue.", analysis_mode="forecast")
    )


def test_threshold_question_receives_typed_answer_contract() -> None:
    ranked = rank_and_deduplicate_questions(
        [_llm_question("Which decision threshold minimizes false-positive cost?")]
    )

    contract = ranked.candidates[0].answer_contract
    assert contract is not None
    assert contract.kind == "threshold"
    assert contract.required_column_tokens == ["threshold"]


def test_llm_route_covering_everything_needs_no_template_backstop(tmp_path: Path) -> None:
    loaded, profile = _loaded_with_profile(_order_items_csv(tmp_path))
    llm_candidates = [
        _llm_question(TREND_Q),
        _llm_question(GROUP_Q),
        _llm_question(QUALITY_Q),
    ]

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
        llm_candidates=llm_candidates,
        include_template_candidates=True,
        template_backstop_only=True,
    )

    assert candidates.template_backstop_used == 0
    assert candidates.template_backstop_categories == []
    # LLM free questioning is primary: no template question rode along.
    assert all(candidate.origin == "llm" for candidate in candidates.candidates)


def test_missing_trend_category_is_backstopped_and_metered(tmp_path: Path) -> None:
    loaded, profile = _loaded_with_profile(_order_items_csv(tmp_path))
    llm_candidates = [_llm_question(GROUP_Q), _llm_question(QUALITY_Q)]

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
        llm_candidates=llm_candidates,
        include_template_candidates=True,
        template_backstop_only=True,
    )

    template_candidates = [
        candidate for candidate in candidates.candidates if candidate.origin == "template"
    ]
    assert template_candidates, "expected trend backstop questions"
    assert {candidate.template_id for candidate in template_candidates} == {"trend"}
    assert candidates.template_backstop_used == len(template_candidates)
    assert candidates.template_backstop_categories == ["trend"]
    # Backstop questions are deterministic templates, never exploratory.
    assert all(not candidate.exploratory for candidate in template_candidates)


def test_empty_llm_route_falls_back_to_the_full_template_pool(tmp_path: Path) -> None:
    loaded, profile = _loaded_with_profile(_order_items_csv(tmp_path))

    candidates = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
        llm_candidates=[],
        include_template_candidates=True,
        template_backstop_only=True,
    )

    # Reproducibility floor: template code is not deleted, only demoted.
    families = {candidate.template_id for candidate in candidates.candidates}
    assert "trend" in families and "group_difference" in families
    assert candidates.template_backstop_used == 0


def test_select_backstop_candidates_caps_per_missing_category(tmp_path: Path) -> None:
    loaded, profile = _loaded_with_profile(_order_items_csv(tmp_path))
    full = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile],
    )
    template_pool = [c for c in full.candidates if c.origin == "template"]

    kept, missing = select_backstop_candidates(
        [_llm_question(QUALITY_Q)], template_pool, per_category=1
    )

    assert missing == ["trend", "group_difference"]
    assert len(kept) <= 2
    assert {candidate.template_id for candidate in kept} <= {"trend", "group_difference"}


# --------------------------------------------------------------------------- #
# 2. Role layer gates template candidate pools + impact-weighted scoring
# --------------------------------------------------------------------------- #
def test_sequence_column_is_verified_and_leaves_stat_template_pools(
    tmp_path: Path,
) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    role_set = infer_column_roles(profile, frame=loaded.frame)
    item_role = role_set.role_of("order_item_id")
    assert item_role is not None and item_role.role is ColumnRoleName.SEQUENCE
    assert "order_item_id" in role_set.excluded_from_stats()

    with_roles = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile_artifact],
        column_role_sets={profile.name: role_set},
    )
    for candidate in with_roles.candidates:
        if candidate.template_id not in {"trend", "group_difference", "correlation_probe"}:
            continue
        referenced = [
            column
            for columns in candidate.referenced_columns.values()
            for column in columns
        ]
        assert "order_item_id" not in referenced
        assert "order_item_id" not in (candidate.sql_template or "")

    # Sanity: without the role layer the same counter DID enter the pools
    # (this is exactly the pre-DI8 "average of order_item_id" failure).
    without_roles = discover_question_candidates(
        [loaded],
        profile_artifacts=[profile_artifact],
    )
    assert any(
        "order_item_id" in (candidate.sql_template or "")
        for candidate in without_roles.candidates
        if candidate.template_id in {"trend", "group_difference"}
    )


def test_impact_weight_zero_zeroes_the_statistical_signal(tmp_path: Path) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    role_set = infer_column_roles(profile, frame=loaded.frame)
    assert role_set.impact_weight("order_item_id") == 0.0
    assert role_set.impact_weight("price") == 1.0

    weighted = score_question(
        profile_artifacts=[profile_artifact],
        target_datasets=[profile.name],
        referenced_columns={profile.name: ["order_item_id"]},
        statistical_signal=0.9,
        column_role_sets={profile.name: role_set},
    )
    unweighted = score_question(
        profile_artifacts=[profile_artifact],
        target_datasets=[profile.name],
        referenced_columns={profile.name: ["order_item_id"]},
        statistical_signal=0.9,
    )

    assert weighted.statistical_signal == 0.0
    assert unweighted.statistical_signal == 0.9
    assert weighted.deterministic_score < unweighted.deterministic_score


# --------------------------------------------------------------------------- #
# 3. LIDA-style data summary rides in the LLM payload
# --------------------------------------------------------------------------- #
def test_build_data_summary_compresses_roles_samples_and_cardinality(
    tmp_path: Path,
) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    role_set = infer_column_roles(profile, frame=loaded.frame)

    summary = build_data_summary(
        [profile],
        role_sets={profile.name: role_set},
        confirmed_joins=["order_items.csv.order_id -> orders.csv.order_id"],
    )

    assert "order_items.csv" in summary
    assert "30 rows x 5 columns" in summary
    assert "role=sequence" in summary  # verified role visible to the model
    assert "role=measure" in summary
    assert "e.g." in summary  # sample values included
    assert "Relationships:" in summary
    assert "order_items.csv.order_id -> orders.csv.order_id [confirmed join]" in summary
    # Deterministic: identical inputs, byte-identical summary.
    assert summary == build_data_summary(
        [profile],
        role_sets={profile.name: role_set},
        confirmed_joins=["order_items.csv.order_id -> orders.csv.order_id"],
    )


def test_multi_table_summary_states_that_no_join_is_confirmed(tmp_path: Path) -> None:
    # A silent Relationships section read as permission: the model proposed
    # cross-table questions that the executor rejected against an empty
    # whitelist, burning two of seven question slots on the FIFA run.
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    second = profile.model_copy(update={"name": "orders.csv", "dataset_id": "ds_orders"})

    summary = build_data_summary([profile, second], confirmed_joins=[])

    assert "Relationships:" in summary
    assert "No join between these tables is confirmed" in summary
    assert "required_relations empty" in summary


def test_the_join_instruction_forbids_joins_when_none_are_confirmed() -> None:
    forbidding = _join_instruction([])
    assert "NO joins are confirmed" in forbidding
    assert "rejected before it runs" in forbidding

    permitting = _join_instruction(["a.csv.id -> b.csv.id"])
    assert "confirmed_join_whitelist" in permitting
    assert "NO joins are confirmed" not in permitting


def test_llm_payload_carries_data_summary_and_join_whitelist(tmp_path: Path) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    role_set = infer_column_roles(profile, frame=loaded.frame)
    llm = FakeQuestionLLM(
        {
            "questions": [
                {
                    "question_en": "Which category drives the highest prices?",
                    "target_datasets": ["order_items.csv"],
                    "llm_business_relevance": 0.8,
                    "llm_actionability": 0.7,
                }
            ]
        }
    )

    result = propose_llm_question_candidates(
        [profile_artifact],
        llm=llm,
        max_questions=3,
        role_sets={profile.name: role_set},
        confirmed_joins={"a.csv.x -> b.csv.x"},
    )

    assert result.error is None
    payload = llm.calls[0]["payload"]
    assert "role=sequence" in payload["data_summary"]
    assert payload["confirmed_join_whitelist"] == ["a.csv.x -> b.csv.x"]
    assert "required_relations" in payload["instructions"]


# --------------------------------------------------------------------------- #
# 4. Exploratory flag: candidate -> execution -> findings (D-side contract)
# --------------------------------------------------------------------------- #
def test_llm_candidates_are_exploratory_and_flag_rides_into_findings(
    tmp_path: Path,
) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    llm = FakeQuestionLLM(
        {
            "questions": [
                {
                    "question_en": "Which category drives the highest prices?",
                    "target_datasets": ["order_items.csv"],
                    "llm_business_relevance": 0.8,
                    "llm_actionability": 0.7,
                }
            ]
        }
    )
    result = propose_llm_question_candidates([profile_artifact], llm=llm, max_questions=1)
    assert result.candidates[0].exploratory is True

    # Execute an exploratory candidate through the template path (deterministic
    # SQL, no live LLM needed) and watch the flag land on result + findings.
    candidate = result.candidates[0].model_copy(
        update={
            "origin": "template",
            "template_id": "group_difference",
            "sql_template": (
                'select category, count(*) as row_count, '
                'avg(price) as avg_price, sum(price) as total_price '
                'from "ds_order_items" group by 1 order by avg_price desc'
            ),
        }
    )
    artifacts = execute_question_candidate(
        candidate,
        datasets=[loaded],
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
    )
    qexec = next(
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert qexec.payload["status"] == "succeeded"
    assert qexec.payload["exploratory"] is True
    assert qexec.payload["findings"]
    assert all(finding["exploratory"] is True for finding in qexec.payload["findings"])


def test_automatic_llm_execution_stops_when_plan_needs_approval(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    loaded, _ = _loaded_with_profile(_order_items_csv(tmp_path))
    candidate = _llm_question("How many rows are present?")
    plan = AnalysisPlan(
        question=candidate.question_en,
        dataset_names=["order_items.csv"],
        columns=["order_id"],
        filters=[],
        sql='select count(*) as row_count from "ds_order_items"',
        method="count rows",
        rationale="A full-table scan requires confirmation.",
        needs_approval=True,
        estimated_scan="large",
    )
    monkeypatch.setattr(question_exec_driver, "build_plan", lambda *args, **kwargs: plan)
    llm = FakeQuestionLLM({})

    artifacts = execute_question_candidate(
        candidate,
        datasets=[loaded],
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
        llm=llm,  # type: ignore[arg-type] -- protocol-compatible test double
    )

    assert len(artifacts) == 1
    assert artifacts[0].type is ArtifactType.QUESTION_EXECUTION_RESULT
    assert artifacts[0].payload["status"] == "failed"
    assert artifacts[0].payload["outcome"] == "awaiting_approval"
    assert artifacts[0].payload["abstention_code"] == "approval_required"
    assert "requires explicit user approval" in artifacts[0].payload["error"]
    assert artifacts[0].payload["sql"] is not None


def test_template_route_findings_stay_non_exploratory(tmp_path: Path) -> None:
    loaded, profile_artifact = _loaded_with_profile(_order_items_csv(tmp_path))
    candidates = discover_question_candidates(
        [loaded], profile_artifacts=[profile_artifact]
    )
    template = next(
        candidate
        for candidate in candidates.candidates
        if candidate.template_id == "group_difference"
    )
    assert template.exploratory is False

    artifacts = execute_question_candidate(
        template,
        datasets=[loaded],
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
    )
    qexec = next(
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert qexec.payload["status"] == "succeeded"
    assert qexec.payload["exploratory"] is False
    assert all(
        finding["exploratory"] is False for finding in qexec.payload["findings"]
    )


# --------------------------------------------------------------------------- #
# 5. Wiring: auto EDA persists role caches + meters degradation / backstop
# --------------------------------------------------------------------------- #
def test_auto_eda_offline_persists_role_sets_and_meters_degradation(
    tmp_path: Path,
) -> None:
    csv_path = _order_items_csv(tmp_path)
    result = run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="run_roles",
    )

    role_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.COLUMN_ROLE_SET
    ]
    assert len(role_artifacts) == 1
    payload = role_artifacts[0].payload
    assert payload["dataset"] == "order_items.csv"
    roles_by_column = {role["column"]: role for role in payload["roles"]}
    assert roles_by_column["order_item_id"]["role"] == "sequence"
    assert roles_by_column["order_item_id"]["provenance"] == "inferred"

    metrics = summarize_session(
        ArtifactStore(tmp_path / "workspace"), "project_demo", "run_roles"
    )
    # Offline run: the semantic bootstrap explicitly degraded to the
    # deterministic pass — metered, never silent.
    assert metrics.semantic_bootstrap_degraded is True

    # The excluded counter must not be executed as a stat question either.
    qexec_payloads = [
        artifact.payload
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    assert all(
        "order_item_id" not in (payload.get("sql") or "")
        for payload in qexec_payloads
    )


def test_run_metrics_aggregate_semantic_and_backstop_events(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="semantic_bootstrap",
            name="di8_semantic_bootstrap",
            summary={"dataset": "a.csv", "degraded": True, "unverified_count": 2},
        ),
    )
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="semantic_bootstrap",
            name="di8_semantic_bootstrap",
            summary={"dataset": "b.csv", "degraded": False, "unverified_count": 1},
        ),
    )
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="template_backstop",
            name="discover_questions",
            summary={"backstop_count": 2, "missing_categories": ["trend"]},
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.semantic_bootstrap_degraded is True
    assert metrics.column_roles_unverified == 3
    assert metrics.template_backstop_used == 2

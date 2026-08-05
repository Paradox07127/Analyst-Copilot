"""DI10-W2: widened auto-execution funnel + LLM accounting completion.

Sprint-10 live review evidence: three real runs generated 20/15/19 candidates
but auto-executed only 1/1/2 — all templates, zero exploratory LLM questions,
and the deterministic domain-metric SQL (GMV/AOV/...) never ran. Separately,
SessionMetrics under-counted LLM spend (llm_calls=4 vs >=13 real) because the
semantic bootstrap and m4 questioning calls emitted no ``llm_call`` events.

Covered here:
1. every ``domain_metric`` candidate enters the auto-execution set;
2. the exploratory floor picks >=1 LLM question, highest composite score;
3. the total is capped and the floor survives the cap;
4. the selection composition rides on the trace;
5. bootstrap/m4 accounting events carry token usage (mocked LLM);
6. SessionMetrics aggregates the new events.

All LLM calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.semantic_bootstrap import bootstrap_semantics
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import ArtifactType, DatasetProfile
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.domain_metrics import background_section_for
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import (
    _AUTO_EXEC_MAX_DOMAIN_METRIC,
    auto_execution_composition,
    select_auto_execution_set,
)

T = TypeVar("T", bound=BaseModel)

_USAGE = LLMResultMetadata(
    provider="fake",
    model="fake-model-1",
    usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    estimated_cost_usd=0.00123,
)


class UsageFakeLLM:
    """Schema-dispatching mock that always reports token usage.

    Bootstrap and question-discovery tasks succeed; every other task raises
    RuntimeError, degrading exactly like the offline client (reporting and
    plan generation catch it).
    """

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.tasks.append(task)
        fields = schema.model_fields
        if "entity" in fields and "columns" in fields:  # semantic bootstrap
            return schema.model_validate({"entity": "order item", "columns": []})
        if "questions" in fields:  # m4 question discovery
            return schema.model_validate(
                {
                    "questions": [
                        {
                            "question_en": (
                                "Which category drives the highest total price?"
                            ),
                            "target_datasets": ["order_items.csv"],
                            "llm_business_relevance": 0.8,
                            "llm_actionability": 0.7,
                        }
                    ]
                }
            )
        raise RuntimeError(f"LLM task {task!r} is not mocked.")

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> LLMResultMetadata:
        return _USAGE


def _score(
    *,
    deterministic: float = 0.6,
    signal: float = 0.5,
    quality_risk: float = 0.0,
    join_risk: float = 0.0,
    relevance: float | None = None,
    actionability: float | None = None,
) -> QuestionScore:
    return QuestionScore(
        data_availability=1.0,
        statistical_signal=signal,
        quality_risk=quality_risk,
        join_risk=join_risk,
        deterministic_score=deterministic,
        llm_business_relevance=relevance,
        llm_actionability=actionability,
    )


def _candidate(question_id: str, **overrides: Any) -> QuestionCandidate:
    values: dict[str, Any] = {
        "question_id": question_id,
        "question_en": f"Question {question_id}?",
        "origin": "template",
        "template_id": "trend",
        "target_datasets": ["orders.csv"],
        "sql_template": "select 1",
        "exploratory": False,
        "score": _score(),
    }
    values.update(overrides)
    return QuestionCandidate.model_validate(values)


def _domain_metric(question_id: str, **overrides: Any) -> QuestionCandidate:
    values: dict[str, Any] = {
        "template_id": "domain_metric",
        "score": _score(deterministic=0.65, signal=0.65),
    }
    values.update(overrides)
    return _candidate(question_id, **values)


def _exploratory(question_id: str, **overrides: Any) -> QuestionCandidate:
    values: dict[str, Any] = {
        "origin": "llm",
        "template_id": None,
        "sql_template": None,
        "exploratory": True,
        "score": _score(deterministic=0.6, relevance=0.8, actionability=0.7),
    }
    values.update(overrides)
    return _candidate(question_id, **values)


def _set(*candidates: QuestionCandidate) -> QuestionCandidateSet:
    return QuestionCandidateSet(candidates=list(candidates))


# --------------------------------------------------------------------------- #
# 1. domain_metric lane: capped, best-scored first
# --------------------------------------------------------------------------- #
def test_domain_metrics_are_capped_and_taken_best_first() -> None:
    metrics = [
        _domain_metric(f"q_dm_{index}", score=_score(deterministic=0.9 - index / 100))
        for index in range(5)
    ]
    others = [
        _candidate("q_trend", template_id="trend"),
        _candidate("q_group", template_id="group_difference"),
        _exploratory("q_llm_1"),
    ]
    selected = select_auto_execution_set(_set(*metrics, *others))

    metric_ids = [c.question_id for c in selected if c.template_id == "domain_metric"]
    assert metric_ids == ["q_dm_0", "q_dm_1"]
    assert all(candidate.status == "auto_selected" for candidate in selected)


def test_domain_metrics_cannot_crowd_out_the_llm_questions() -> None:
    # The World Cup run's shape: three domain-agnostic metrics (time coverage,
    # HHI, missing hotspots) outscored every free-form question, took an
    # uncapped floor, and left 3 of 10 slots unused while 5 LLM questions went
    # unrun. The metric lane is now the capped one.
    metrics = [
        _domain_metric(f"q_dm_{index}", score=_score(deterministic=0.84 - index / 100))
        for index in range(3)
    ]
    pool = [
        _exploratory(f"q_llm_{index}", score=_score(relevance=0.9 - index / 100))
        for index in range(9)
    ]
    selected = select_auto_execution_set(_set(*metrics, *pool))

    assert len(selected) == 10  # the budget is spent, not abandoned
    metric_ids = [c.question_id for c in selected if c.template_id == "domain_metric"]
    exploratory_ids = [c.question_id for c in selected if c.exploratory]
    assert len(metric_ids) == 2
    assert len(exploratory_ids) == 8


def test_domain_metric_with_confirmed_relations_still_selected() -> None:
    # The old funnel rejected any required_relations; cross-table metrics
    # (e.g. repeat purchase rate over a confirmed join) must now run — the
    # execution layer still re-checks the join whitelist unconditionally.
    metric = _domain_metric(
        "q_dm_join",
        required_relations=["orders.csv.id -> items.csv.order_id"],
    )
    selected = select_auto_execution_set(_set(metric))
    assert [candidate.question_id for candidate in selected] == ["q_dm_join"]


# --------------------------------------------------------------------------- #
# 2. exploratory floor: >=1, highest composite score
# --------------------------------------------------------------------------- #
def test_exploratory_floor_picks_highest_composite_score() -> None:
    lower = _exploratory(
        "q_llm_low",
        score=_score(deterministic=0.7, relevance=0.2, actionability=0.1),
    )
    higher = _exploratory(
        "q_llm_high",
        score=_score(deterministic=0.6, relevance=0.9, actionability=0.8),
    )
    selected = select_auto_execution_set(
        _set(lower, higher, _domain_metric("q_dm_0"))
    )

    exploratory_ids = [c.question_id for c in selected if c.exploratory]
    # Highest composite is reserved by the floor; the rest fills leftover budget.
    assert exploratory_ids[0] == "q_llm_high"
    assert set(exploratory_ids) == {"q_llm_high", "q_llm_low"}


def test_leftover_budget_is_filled_with_ranked_exploratory_questions() -> None:
    """A thin domain-metric catalogue must not mean a two-question run."""
    pool = [
        _exploratory(f"q_llm_{index}", score=_score(relevance=0.9 - index / 100))
        for index in range(8)
    ]
    selected = select_auto_execution_set(_set(_domain_metric("q_dm_0"), *pool))

    exploratory_ids = [c.question_id for c in selected if c.exploratory]
    assert len(exploratory_ids) == 8  # spends the whole leftover budget
    assert exploratory_ids == sorted(exploratory_ids)  # best-scored first


def test_fill_never_exceeds_the_total_budget() -> None:
    metrics = [_domain_metric(f"q_dm_{index}") for index in range(10)]
    pool = [_exploratory(f"q_llm_{index}") for index in range(15)]
    selected = select_auto_execution_set(_set(*metrics, *pool))

    assert len(selected) == 10


def test_exploratory_infeasible_candidates_are_skipped() -> None:
    infeasible = _exploratory(
        "q_llm_bad",
        feasibility={"status": "needs_data", "reasons": [], "missing": []},
    )
    feasible = _exploratory("q_llm_ok")
    selected = select_auto_execution_set(_set(infeasible, feasible))
    exploratory_ids = [c.question_id for c in selected if c.exploratory]
    assert exploratory_ids == ["q_llm_ok"]


# --------------------------------------------------------------------------- #
# 3. cap: total bounded, floors survive, one template per family
# --------------------------------------------------------------------------- #
def test_total_is_capped_and_exploratory_floor_survives() -> None:
    metrics = [_domain_metric(f"q_dm_{index}") for index in range(3)]
    templates = [
        _candidate(f"q_t_{index}", template_id=f"family_{index}")
        for index in range(6)
    ]
    exploratory = _exploratory("q_llm_1")

    selected = select_auto_execution_set(
        _set(*metrics, *templates, exploratory), max_total=4
    )

    assert len(selected) == 4
    assert any(candidate.exploratory for candidate in selected)  # floor kept
    assert sum(
        1 for candidate in selected if candidate.template_id == "domain_metric"
    ) == 2


def test_default_cap_and_template_family_dedup() -> None:
    metrics = [_domain_metric(f"q_dm_{index}") for index in range(12)]
    trends = [
        _candidate(f"q_trend_{index}", template_id="trend") for index in range(3)
    ]
    group = _candidate("q_group", template_id="group_difference")
    selected = select_auto_execution_set(_set(*metrics, *trends, group))

    assert len(selected) <= 10  # _AUTO_EXEC_MAX_TOTAL
    non_metric_templates = [
        candidate
        for candidate in selected
        if candidate.template_id not in (None, "domain_metric")
    ]
    families = [candidate.template_id for candidate in non_metric_templates]
    assert len(families) == len(set(families))  # one per family
    assert len(non_metric_templates) <= 2  # _AUTO_EXEC_TEMPLATE_TOP_N


def test_old_template_eligibility_rules_still_apply_to_non_metrics() -> None:
    risky = _candidate("q_risky", score=_score(join_risk=0.5))
    joined = _candidate(
        "q_joined", required_relations=["a.csv.x -> b.csv.x"]
    )
    llm_origin_non_exploratory = _candidate(
        "q_llm_plain", origin="llm", template_id=None, sql_template=None
    )
    clean = _candidate("q_clean", template_id="quality")
    selected = select_auto_execution_set(
        _set(risky, joined, llm_origin_non_exploratory, clean)
    )
    assert [candidate.question_id for candidate in selected] == ["q_clean"]


# --------------------------------------------------------------------------- #
# 4. composition telemetry helper
# --------------------------------------------------------------------------- #
def test_auto_execution_composition_counts() -> None:
    selected = select_auto_execution_set(
        _set(
            _domain_metric("q_dm_0"),
            _domain_metric("q_dm_1"),
            _exploratory("q_llm_1"),
            _candidate("q_trend", template_id="trend"),
        )
    )
    composition = auto_execution_composition(selected)
    assert composition["n_domain_metric"] == 2
    assert composition["n_exploratory"] == 1
    assert composition["n_template"] == 1
    assert composition["selected_count"] == len(selected) == 4


# --------------------------------------------------------------------------- #
# Fixtures for the wiring tests (same Olist-shaped table as test_di8/di9)
# --------------------------------------------------------------------------- #
def _order_items_csv(tmp_path: Path) -> Path:
    rows = ["order_id,order_item_id,order_date,price,category"]
    categories = ["books", "toys", "garden", "sports"]
    day = 1
    for order_index in range(6):
        for item in range(1, 6):
            price = 10.0 + order_index * 3.7 + item * 1.3
            rows.append(
                f"O{order_index:03d},{item},2026-01-{day:02d},"
                f"{price:.2f},{categories[(order_index + item) % len(categories)]}"
            )
            day = day % 28 + 1
    path = tmp_path / "order_items.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _events(workspace: Path, session_id: str) -> list[TraceEvent]:
    return ArtifactStore(workspace).list_trace_events(
        project_id="project_demo", session_id=session_id
    )


# --------------------------------------------------------------------------- #
# 5. Wiring: offline run executes the capped metric lane + emits composition
# --------------------------------------------------------------------------- #
def test_auto_eda_offline_executes_capped_domain_metrics_and_meters_selection(
    tmp_path: Path,
) -> None:
    csv_path = _order_items_csv(tmp_path)
    workspace = tmp_path / "workspace"
    result = run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="project_demo",
        session_id="run_funnel",
    )

    qcand = next(
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate_set = QuestionCandidateSet.model_validate(qcand.payload)
    domain_metric_ids = {
        candidate.question_id
        for candidate in candidate_set.candidates
        if candidate.template_id == "domain_metric"
    }
    # The fixture resolves more metrics than the lane admits; that is the point.
    assert len(domain_metric_ids) > _AUTO_EXEC_MAX_DOMAIN_METRIC

    events = _events(workspace, "run_funnel")
    selection_events = [
        event
        for event in events
        if event.event_type == "question_auto_execution_selected"
    ]
    assert len(selection_events) == 1
    summary = selection_events[0].summary
    assert summary["n_domain_metric"] == _AUTO_EXEC_MAX_DOMAIN_METRIC
    assert (
        summary["n_background"]
        + summary["n_domain_metric"]
        + summary["n_exploratory"]
        + summary["n_template"]
        == summary["selected_count"]
    )
    # Background metrics run outside the analysis budget, so the lane cap only
    # binds the business metrics.
    business_metric_ids = {
        candidate.question_id
        for candidate in candidate_set.candidates
        if candidate.template_id == "domain_metric"
        and background_section_for(candidate.metric_id) is None
    }
    selected_metric_ids = business_metric_ids & set(summary["question_ids"])
    assert len(selected_metric_ids) == _AUTO_EXEC_MAX_DOMAIN_METRIC

    executed_ids = {
        event.summary["question_id"]
        for event in events
        if event.event_type == "question_auto_execution"
    }
    assert selected_metric_ids <= executed_ids
    # Per-question events now disclose their funnel lane.
    per_question = [
        event for event in events if event.event_type == "question_auto_execution"
    ]
    assert all(
        "template_id" in event.summary and "exploratory" in event.summary
        for event in per_question
    )


# --------------------------------------------------------------------------- #
# 6. Accounting: bootstrap + m4 llm_call events carry tokens; metrics roll up
# --------------------------------------------------------------------------- #
def test_bootstrap_result_carries_llm_usage(tmp_path: Path) -> None:
    loaded = load_csv(_order_items_csv(tmp_path), dataset_id="ds_order_items")
    profile_artifact = profile_dataset(
        loaded, project_id="project_demo", session_id="run_demo"
    )
    profile = DatasetProfile.model_validate(profile_artifact.payload)

    result = bootstrap_semantics(profile, llm=cast(Any, UsageFakeLLM()), frame=loaded.frame)

    assert result.llm_usage is not None
    assert result.llm_usage.usage.total_tokens == 150
    # Offline degrade keeps the field empty — no phantom accounting.
    offline = bootstrap_semantics(profile, llm=None, frame=loaded.frame)
    assert offline.llm_usage is None


def test_auto_eda_emits_bootstrap_and_m4_accounting_events(tmp_path: Path) -> None:
    csv_path = _order_items_csv(tmp_path)
    workspace = tmp_path / "workspace"
    run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id="project_demo",
        session_id="run_tokens",
        llm=cast(Any, UsageFakeLLM()),
    )

    events = _events(workspace, "run_tokens")
    bootstrap_calls = [
        event
        for event in events
        if event.event_type == "llm_call" and event.name == "semantic_bootstrap"
    ]
    assert len(bootstrap_calls) == 1  # one dataset -> one bootstrap call
    summary = bootstrap_calls[0].summary
    assert summary["provider"] == "fake"
    assert summary["model"] == "fake-model-1"
    assert summary["prompt_tokens"] == 100
    assert summary["completion_tokens"] == 50
    assert summary["total_tokens"] == 150
    assert summary["estimated_cost_usd"] == 0.00123

    m4_calls = [
        event
        for event in events
        if event.event_type == "llm_call" and event.name == "m4_question_discovery"
    ]
    assert len(m4_calls) == 1  # enriched in place, never duplicated
    m4_summary = m4_calls[0].summary
    assert m4_summary["candidate_count"] >= 1
    assert m4_summary["total_tokens"] == 150
    assert m4_summary["estimated_cost_usd"] == 0.00123

    # The exploratory floor put the LLM question into the auto-exec set (its
    # execution fails offline-style and is recorded without blocking the run).
    selection = next(
        event
        for event in events
        if event.event_type == "question_auto_execution_selected"
    )
    assert selection.summary["n_exploratory"] >= 1

    metrics = summarize_session(ArtifactStore(workspace), "project_demo", "run_tokens")
    assert metrics.llm_calls >= 2  # bootstrap + m4 at minimum
    assert metrics.total_tokens >= 300
    assert metrics.est_cost_usd is not None and metrics.est_cost_usd > 0


def test_run_metrics_aggregate_new_accounting_events(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="llm_call",
            name="semantic_bootstrap",
            summary={
                "dataset": "a.csv",
                "provider": "fake",
                "model": "fake-model-1",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.001,
            },
        ),
    )
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="llm_call",
            name="m4_question_discovery",
            summary={
                "candidate_count": 3,
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
                "estimated_cost_usd": 0.002,
            },
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.llm_calls == 2
    assert metrics.total_tokens == 350
    assert metrics.est_cost_usd == 0.003

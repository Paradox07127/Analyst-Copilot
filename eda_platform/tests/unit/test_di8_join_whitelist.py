"""DI8-C cross-table join whitelist: propose -> confirm -> template consumption
-> JOIN SQL execution, plus the red line that SQL never joins over an
undeclared/unconfirmed relation. All LLM calls are mocked."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.question_agent import propose_llm_question_candidates
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.semantic import (
    JoinWhitelist,
    confirm_join,
    load_join_whitelist,
    record_join_usage,
    save_join_whitelist,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import QuestionCandidateSet
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import discover_question_candidates
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    propose_join_candidates,
    validate_relationships,
)

T = TypeVar("T", bound=BaseModel)

JOIN_LABEL = "orders.csv.customer_id -> customers.csv.customer_id"


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
# Fixtures: a clean FK pair — orders.customer_id (many) -> customers (unique).
# --------------------------------------------------------------------------- #
def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    regions = ["East", "West", "North", "South"]
    customers = ["customer_id,region,tier"]
    for index in range(8):
        customers.append(f"C{index:02d},{regions[index % 4]},T{index % 5}")
    customers_path = tmp_path / "customers.csv"
    customers_path.write_text("\n".join(customers) + "\n", encoding="utf-8")

    orders = ["order_id,customer_id,amount"]
    for index in range(32):
        orders.append(f"O{index:03d},C{index % 8:02d},{25.0 + index * 3.17:.2f}")
    orders_path = tmp_path / "orders.csv"
    orders_path.write_text("\n".join(orders) + "\n", encoding="utf-8")
    return orders_path, customers_path


def _load_pair(tmp_path: Path) -> list[LoadedDataset]:
    orders_path, customers_path = _write_pair(tmp_path)
    return [
        load_csv(orders_path, dataset_id="ds_orders"),
        load_csv(customers_path, dataset_id="ds_customers"),
    ]


def _profiles(datasets: list[LoadedDataset]) -> list[Artifact]:
    return [
        profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
        for loaded in datasets
    ]


def _engine(datasets: list[LoadedDataset]) -> DuckDBQueryEngine:
    engine = DuckDBQueryEngine()
    for loaded in datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    return engine


def _proposed_whitelist(datasets: list[LoadedDataset]) -> JoinWhitelist:
    engine = _engine(datasets)
    candidates = discover_relationship_candidates(datasets, engine)
    validations = validate_relationships(candidates, engine)
    whitelist = JoinWhitelist()
    whitelist.merge_proposals(propose_join_candidates(candidates, validations))
    return whitelist


# --------------------------------------------------------------------------- #
# 1. Proposal: M2 output -> structured JoinWhitelistEntry with cardinality
# --------------------------------------------------------------------------- #
def test_propose_join_candidates_carries_cardinality_and_confidence_source(
    tmp_path: Path,
) -> None:
    datasets = _load_pair(tmp_path)
    engine = _engine(datasets)
    candidates = discover_relationship_candidates(datasets, engine)
    validations = validate_relationships(candidates, engine)

    proposals = propose_join_candidates(candidates, validations)

    fk_entry = next(entry for entry in proposals if entry.label() == JOIN_LABEL)
    # DI10-W5: high confidence + id naming on both sides + non-m:n cardinality
    # auto-confirms at proposal time (machine decision, revocable, disclosed).
    assert fk_entry.status == "auto_confirmed"
    assert fk_entry.confirmed_by == "auto"
    assert fk_entry.cardinality == "many_to_one"  # row-explosion signal recorded
    assert fk_entry.join_row_multiplier is not None
    assert abs(fk_entry.join_row_multiplier - 1.0) < 0.001
    assert "relationship_discovery" in fk_entry.confidence_source
    # Low-confidence pairs never even reach "proposed".
    proposed_labels = {entry.label() for entry in proposals}
    low_labels = {
        candidate.pair.label()
        for candidate in candidates.candidates
        if candidate.confidence == "low"
    }
    assert not (proposed_labels & low_labels)


# --------------------------------------------------------------------------- #
# 2. State machine: proposed -> confirmed (programmatic path) -> reusable
# --------------------------------------------------------------------------- #
def test_whitelist_state_machine_and_programmatic_confirmation(tmp_path: Path) -> None:
    datasets = _load_pair(tmp_path)
    whitelist = _proposed_whitelist(datasets)
    project_dir = tmp_path / "project"
    save_join_whitelist(project_dir, whitelist)

    # DI10-W5: the clean FK pair auto-confirms at proposal time, so it is
    # already usable (confirmed_labels is the converged usable-labels alias).
    stored = load_join_whitelist(project_dir)
    auto_entry = stored.entry(JOIN_LABEL)
    assert auto_entry is not None and auto_entry.status == "auto_confirmed"
    assert stored.confirmed_labels() == {JOIN_LABEL}

    # confirm_join "promotes" the machine decision to a human one.
    confirmed = confirm_join(project_dir, JOIN_LABEL, confirmed_by="pytest")
    entry = confirmed.entry(JOIN_LABEL)
    assert entry is not None
    assert entry.status == "confirmed"
    assert entry.confirmed_by == "pytest"
    assert load_join_whitelist(project_dir).confirmed_labels() == {JOIN_LABEL}

    # Re-proposing must not downgrade the confirmed entry (idempotent merge).
    reloaded = load_join_whitelist(project_dir)
    added = reloaded.merge_proposals(_proposed_whitelist(datasets).entries)
    assert added == 0
    entry = reloaded.entry(JOIN_LABEL)
    assert entry is not None and entry.status == "confirmed"

    # Vanna-style reuse memory: usage counting on the stored object.
    record_join_usage(project_dir, JOIN_LABEL)
    record_join_usage(project_dir, JOIN_LABEL)
    entry = load_join_whitelist(project_dir).entry(JOIN_LABEL)
    assert entry is not None and entry.usage_count == 2


# --------------------------------------------------------------------------- #
# 3. Template family consumes ONLY confirmed joins
# --------------------------------------------------------------------------- #
def test_cross_table_templates_require_a_confirmed_join(tmp_path: Path) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    whitelist = _proposed_whitelist(datasets)
    # DI10-W5 auto-confirms the clean FK pair at proposal time; pin every
    # entry back to "proposed" (the revoked shape) so this test still proves
    # that proposed-only whitelists produce no cross-table templates.
    for entry in whitelist.entries:
        entry.status = "proposed"

    proposed_only = discover_question_candidates(
        datasets,
        profile_artifacts=profiles,
        join_whitelist=whitelist,
    )
    assert all(
        candidate.template_id != "cross_table_aggregation"
        for candidate in proposed_only.candidates
    )

    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    entry.status = "confirmed"
    confirmed = discover_question_candidates(
        datasets,
        profile_artifacts=profiles,
        join_whitelist=whitelist,
    )
    cross = [
        candidate
        for candidate in confirmed.candidates
        if candidate.template_id == "cross_table_aggregation"
    ]
    assert cross, "expected a cross-table template question over the confirmed join"
    for candidate in cross:
        assert candidate.required_relations == [JOIN_LABEL]
        assert candidate.sql_template is not None
        assert re.search(r"\bjoin\b", candidate.sql_template, re.IGNORECASE)
        assert set(candidate.target_datasets) == {"orders.csv", "customers.csv"}
        # Deterministic templates are not exploratory.
        assert candidate.exploratory is False


def test_many_to_many_confirmed_joins_are_never_templated(tmp_path: Path) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    whitelist = _proposed_whitelist(datasets)
    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    entry.status = "confirmed"
    entry.cardinality = "many_to_many"  # row-explosion defense

    candidates = discover_question_candidates(
        datasets,
        profile_artifacts=profiles,
        join_whitelist=whitelist,
    )

    assert all(
        candidate.template_id != "cross_table_aggregation"
        for candidate in candidates.candidates
    )


# --------------------------------------------------------------------------- #
# 4. Execution: JOIN SQL runs over a confirmed join, usage is counted
# --------------------------------------------------------------------------- #
def test_cross_table_execution_generates_join_sql_and_counts_usage(
    tmp_path: Path,
) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    whitelist = _proposed_whitelist(datasets)
    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    entry.status = "confirmed"
    candidate = next(
        item
        for item in discover_question_candidates(
            datasets, profile_artifacts=profiles, join_whitelist=whitelist
        ).candidates
        if item.template_id == "cross_table_aggregation"
    )
    used: list[str] = []

    artifacts = execute_question_candidate(
        candidate,
        datasets=datasets,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
        confirmed_joins={JOIN_LABEL},
        on_join_used=used.append,
    )

    qexec = next(
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert qexec.payload["status"] == "succeeded"
    assert re.search(r"\bjoin\b", qexec.payload["sql"], re.IGNORECASE)
    assert qexec.payload["findings"], "expected a deterministic cross-table finding"
    assert used == [JOIN_LABEL]


# --------------------------------------------------------------------------- #
# 5. Red line: SQL never joins over an undeclared / unconfirmed relation
# --------------------------------------------------------------------------- #
def test_join_sql_without_declared_relations_is_refused(tmp_path: Path) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    whitelist = _proposed_whitelist(datasets)
    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    entry.status = "confirmed"
    candidate = next(
        item
        for item in discover_question_candidates(
            datasets, profile_artifacts=profiles, join_whitelist=whitelist
        ).candidates
        if item.template_id == "cross_table_aggregation"
    )
    undeclared = candidate.model_copy(update={"required_relations": []})

    artifacts = execute_question_candidate(
        undeclared,
        datasets=datasets,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
        confirmed_joins={JOIN_LABEL},
    )

    qexec = next(
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert qexec.payload["status"] == "failed"
    assert "required_relations" in qexec.payload["error"]


def test_unconfirmed_join_is_refused_with_the_whitelist_attached(
    tmp_path: Path,
) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    whitelist = _proposed_whitelist(datasets)
    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    entry.status = "confirmed"
    candidate = next(
        item
        for item in discover_question_candidates(
            datasets, profile_artifacts=profiles, join_whitelist=whitelist
        ).candidates
        if item.template_id == "cross_table_aggregation"
    )

    # Same candidate, but the runtime whitelist no longer confirms the join.
    artifacts = execute_question_candidate(
        candidate,
        datasets=datasets,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[],
        confirmed_joins=set(),
    )

    qexec = next(
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    assert qexec.payload["status"] == "failed"
    assert "not a confirmed join" in qexec.payload["error"]
    assert "no confirmed joins in the whitelist" in qexec.payload["error"]


# --------------------------------------------------------------------------- #
# 6. LLM cross-table proposals: repair-retry with the whitelist attached
# --------------------------------------------------------------------------- #
def test_llm_cross_table_question_outside_whitelist_gets_repair_feedback(
    tmp_path: Path,
) -> None:
    datasets = _load_pair(tmp_path)
    profiles = _profiles(datasets)
    base_question = {
        "question_en": "Which region generates the highest order amounts?",
        "target_datasets": ["orders.csv", "customers.csv"],
        "llm_business_relevance": 0.9,
        "llm_actionability": 0.8,
    }
    llm = FakeQuestionLLM(
        [
            {
                "questions": [
                    {
                        **base_question,
                        "required_relations": [
                            "orders.csv.order_id -> customers.csv.customer_id"
                        ],
                    }
                ]
            },
            {"questions": [{**base_question, "required_relations": [JOIN_LABEL]}]},
        ]
    )

    result = propose_llm_question_candidates(
        profiles,
        llm=llm,
        max_questions=2,
        confirmed_joins={JOIN_LABEL},
    )

    assert result.error is None
    assert len(llm.calls) == 2  # repair retry, not a wholesale drop
    retry_payload = llm.calls[1]["payload"]
    assert "previous_error" in retry_payload
    # The rejection feedback carries the confirmed whitelist verbatim.
    assert JOIN_LABEL in retry_payload["previous_error"]
    assert "not a confirmed whitelist join" in retry_payload["previous_error"]
    assert result.candidates[0].required_relations == [JOIN_LABEL]
    assert result.candidates[0].exploratory is True


# --------------------------------------------------------------------------- #
# 7. End-to-end wiring: auto EDA proposes; confirmation unlocks the template
# --------------------------------------------------------------------------- #
def test_auto_eda_proposes_joins_then_confirmation_unlocks_cross_table(
    tmp_path: Path,
) -> None:
    from eda_platform.drivers.auto_eda import run_auto_eda

    orders_path, customers_path = _write_pair(tmp_path)
    workspace = tmp_path / "workspace"

    first = run_auto_eda(
        [orders_path, customers_path],
        workspace=workspace,
        project_id="project_demo",
        session_id="run_propose",
        relationship_discovery="eager",
    )
    project_dir = ArtifactStore(workspace).project_dir("project_demo")
    whitelist = load_join_whitelist(project_dir)
    labels = {entry.label() for entry in whitelist.entries}
    assert JOIN_LABEL in labels  # persisted by the run
    # DI10-W5: the clean FK pair auto-confirms during the unattended run, so
    # cross-table templates unlock without a human click — and the whitelist
    # discloses the machine decision for report limitations.
    entry = whitelist.entry(JOIN_LABEL)
    assert entry is not None
    assert entry.status == "auto_confirmed"
    assert entry.confirmed_by == "auto"
    assert JOIN_LABEL in whitelist.usable_labels()
    assert whitelist.disclosure_notes([JOIN_LABEL])
    metrics = summarize_session(ArtifactStore(workspace), "project_demo", "run_propose")
    assert metrics.join_candidates_proposed >= 1
    first_qcand = next(
        artifact
        for artifact in first.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    first_set = QuestionCandidateSet.model_validate(first_qcand.payload)
    first_cross = [
        candidate
        for candidate in first_set.candidates
        if candidate.template_id == "cross_table_aggregation"
    ]
    assert first_cross, "auto-confirmed join should unlock cross-table templates"
    assert first_cross[0].required_relations == [JOIN_LABEL]

    # Promotion to a human confirmation keeps everything working.
    confirm_join(project_dir, JOIN_LABEL, confirmed_by="pytest")

    second = run_auto_eda(
        [orders_path, customers_path],
        workspace=workspace,
        project_id="project_demo",
        session_id="run_confirmed",
    )
    second_qcand = next(
        artifact
        for artifact in second.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    second_set = QuestionCandidateSet.model_validate(second_qcand.payload)
    cross = [
        candidate
        for candidate in second_set.candidates
        if candidate.template_id == "cross_table_aggregation"
    ]
    assert cross
    assert cross[0].required_relations == [JOIN_LABEL]


def test_changed_dataset_content_invalidates_prior_join_authorization(
    tmp_path: Path,
) -> None:
    from eda_platform.drivers.auto_eda import (
        discover_relationships_on_demand,
        run_auto_eda,
    )

    orders_path, customers_path = _write_pair(tmp_path)
    workspace = tmp_path / "workspace_freshness"
    first = run_auto_eda(
        [orders_path, customers_path],
        workspace=workspace,
        project_id="project_freshness",
        session_id="run_original",
        relationship_discovery="eager",
    )
    project_dir = ArtifactStore(workspace).project_dir("project_freshness")
    whitelist = load_join_whitelist(project_dir)
    assert JOIN_LABEL in whitelist.confirmed_labels(
        {
            loaded.record.name: loaded.record.dataset_id
            for loaded in first.loaded_datasets
        }
    )

    customers_path.write_text(
        customers_path.read_text(encoding="utf-8") + "C999,West,T0\n",
        encoding="utf-8",
    )
    changed = run_auto_eda(
        [orders_path, customers_path],
        workspace=workspace,
        project_id="project_freshness",
        session_id="run_changed",
    )
    current_ids = {
        loaded.record.name: loaded.record.dataset_id for loaded in changed.loaded_datasets
    }
    persisted = load_join_whitelist(project_dir)

    assert persisted.entry(JOIN_LABEL) is not None
    assert persisted.confirmed_labels(current_ids) == set()
    freshness_events = [
        event
        for event in ArtifactStore(workspace).list_trace_events(
            project_id="project_freshness", session_id="run_changed"
        )
        if event.event_type == "join_authorization_freshness"
    ]
    assert freshness_events
    assert freshness_events[-1].summary["stale"] >= 1
    candidate_artifact = next(
        artifact
        for artifact in changed.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate_set = QuestionCandidateSet.model_validate(candidate_artifact.payload)
    assert all(
        JOIN_LABEL not in candidate.required_relations
        for candidate in candidate_set.candidates
    )

    discover_relationships_on_demand(changed)
    refreshed = load_join_whitelist(project_dir)
    refreshed_entry = refreshed.entry(JOIN_LABEL)
    assert refreshed_entry is not None
    assert refreshed_entry.validation_freshness(current_ids) == "fresh"
    assert refreshed_entry.status != "confirmed"
    assert refreshed.confirmed_labels(current_ids) == (
        {JOIN_LABEL} if refreshed_entry.status == "auto_confirmed" else set()
    )


def test_run_metrics_aggregate_join_proposal_events(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    store.append_trace(
        "project_demo",
        TraceEvent(
            session_id="run_demo",
            event_type="join_candidates_proposed",
            name="discover_questions",
            summary={"proposed_count": 3, "labels": ["a", "b", "c"]},
        ),
    )

    metrics = summarize_session(store, "project_demo", "run_demo")

    assert metrics.join_candidates_proposed == 3

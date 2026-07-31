from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import eda_platform.drivers.auto_eda as auto_eda_driver
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import (
    AutoEDAResult,
    validate_relationship_candidate_on_demand,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.relations import (
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipValidation,
    RelationshipValidationSet,
)
from eda_platform.tools.er_diagram import build_er_diagram
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    eager_validation_candidates,
    validate_relationships,
)

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _load_ecommerce() -> list[LoadedDataset]:
    return [
        load_csv(GOLDEN_DATA / "ecommerce_orders.csv", dataset_id="ds_orders"),
        load_csv(GOLDEN_DATA / "ecommerce_customers.csv", dataset_id="ds_customers"),
        load_csv(GOLDEN_DATA / "ecommerce_products.csv", dataset_id="ds_products"),
        load_csv(GOLDEN_DATA / "ecommerce_marketing.csv", dataset_id="ds_marketing"),
    ]


def _engine(datasets: list[LoadedDataset]) -> DuckDBQueryEngine:
    engine = DuckDBQueryEngine()
    for loaded in datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    return engine


def _candidate_key(candidate: RelationshipCandidate) -> tuple[str, str, str, str]:
    pair = candidate.pair
    return (
        pair.left_dataset_name,
        pair.left_columns[0],
        pair.right_dataset_name,
        pair.right_columns[0],
    )


def _candidate(
    candidates: RelationshipCandidateSet,
    left_dataset: str,
    left_column: str,
    right_dataset: str,
    right_column: str,
) -> RelationshipCandidate:
    expected = (left_dataset, left_column, right_dataset, right_column)
    for candidate in candidates.candidates:
        if _candidate_key(candidate) == expected:
            return candidate
    raise AssertionError(f"missing relationship candidate: {expected}")


def _validation(
    validations: RelationshipValidationSet,
    candidate: RelationshipCandidate,
) -> RelationshipValidation:
    label = candidate.pair.label()
    for validation in validations.validations:
        if validation.pair.label() == label:
            return validation
    raise AssertionError(f"missing relationship validation: {label}")


def test_ecommerce_customer_relationship_flags_duplicate_and_orphan_traps() -> None:
    datasets = _load_ecommerce()
    engine = _engine(datasets)

    candidates = discover_relationship_candidates(datasets, engine)
    customer_candidate = _candidate(
        candidates,
        "ecommerce_orders.csv",
        "customer_id",
        "ecommerce_customers.csv",
        "customer_id",
    )
    validations = validate_relationships(candidates, engine)
    customer_validation = _validation(validations, customer_candidate)

    assert customer_candidate.confidence == "medium"
    assert customer_candidate.auto_adopted is False
    assert customer_candidate.signals.right_unique_rate < 1.0
    assert customer_validation.join_row_multiplier > 1.0
    assert customer_validation.orphan_rate_left > 0.0
    assert customer_validation.cardinality == "many_to_many"
    assert "left join" in customer_validation.verification_sql.lower()
    assert customer_validation.warnings


def test_deferred_relationship_validation_is_persisted_and_idempotent(
    tmp_path: Path,
) -> None:
    datasets = _load_ecommerce()
    candidates = discover_relationship_candidates(datasets, _engine(datasets))
    candidate = _candidate(
        candidates,
        "ecommerce_orders.csv",
        "customer_id",
        "ecommerce_customers.csv",
        "customer_id",
    )
    candidate_artifact = Artifact(
        id="relcand_test",
        type=ArtifactType.RELATIONSHIP_CANDIDATE_SET,
        project_id="project",
        session_id="run",
        payload=candidates.model_dump(mode="json"),
    )
    store = ArtifactStore(tmp_path)
    store.ensure_project("project", "project")
    store.start_session("project", "run")
    store.save_artifact(candidate_artifact)
    result = AutoEDAResult(
        project_id="project",
        session_id="run",
        business_context="",
        artifacts=[candidate_artifact],
        report_markdown="",
        workspace=tmp_path,
        loaded_datasets=datasets,
    )

    first = validate_relationship_candidate_on_demand(result, candidate)
    second = validate_relationship_candidate_on_demand(result, candidate)

    assert first == second
    assert first.verified is True
    persisted = store.list_artifacts(project_id="project", session_id="run")
    validation_artifacts = [
        artifact
        for artifact in persisted
        if artifact.type is ArtifactType.RELATIONSHIP_VALIDATION_SET
    ]
    assert len(validation_artifacts) == 1
    events = store.list_trace_events(project_id="project", session_id="run")
    assert sum(e.event_type == "relationship_validation_on_demand" for e in events) == 1
    assert summarize_session(store, "project", "run").relationship_full_validations == 1


def test_deferred_validation_records_nonfatal_metrics_refresh_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _load_ecommerce()
    candidates = discover_relationship_candidates(datasets, _engine(datasets))
    candidate = _candidate(
        candidates,
        "ecommerce_orders.csv",
        "customer_id",
        "ecommerce_customers.csv",
        "customer_id",
    )
    candidate_artifact = Artifact(
        id="relcand_metrics_failure",
        type=ArtifactType.RELATIONSHIP_CANDIDATE_SET,
        project_id="project",
        session_id="run",
        payload=candidates.model_dump(mode="json"),
    )
    store = ArtifactStore(tmp_path)
    store.ensure_project("project", "project")
    store.start_session("project", "run")
    store.save_artifact(candidate_artifact)
    result = AutoEDAResult(
        project_id="project",
        session_id="run",
        business_context="",
        artifacts=[candidate_artifact],
        report_markdown="",
        workspace=tmp_path,
        loaded_datasets=datasets,
    )

    def fail_metrics(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("rollup unavailable")

    monkeypatch.setattr(auto_eda_driver, "persist_run_metrics", fail_metrics)

    validation = validate_relationship_candidate_on_demand(result, candidate)

    assert validation.verified is True
    events = store.list_trace_events(project_id="project", session_id="run")
    error = next(event for event in events if event.event_type == "session_metrics_error")
    assert error.summary == {
        "error_type": "RuntimeError",
        "error": "rollup unavailable",
    }


def test_unrelated_numeric_columns_do_not_reach_high_confidence(tmp_path) -> None:
    left = tmp_path / "left.csv"
    left.write_text("row_id,amount\n1,10\n2,20\n3,30\n", encoding="utf-8")
    right = tmp_path / "right.csv"
    right.write_text("code,price\nA,10\nB,20\nC,30\n", encoding="utf-8")
    datasets = [
        load_csv(left, dataset_id="ds_left"),
        load_csv(right, dataset_id="ds_right"),
    ]
    engine = _engine(datasets)

    candidates = discover_relationship_candidates(datasets, engine)

    assert all(
        candidate.confidence != "high"
        for candidate in candidates.candidates
        if {
            candidate.pair.left_columns[0],
            candidate.pair.right_columns[0],
        }
        == {"amount", "price"}
    )


def test_identical_low_uniqueness_int_metrics_do_not_reach_medium(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    left.write_text(
        "interceptions\n"
        "1\n"
        "1\n"
        "2\n"
        "2\n"
        "3\n"
        "3\n",
        encoding="utf-8",
    )
    right = tmp_path / "right.csv"
    right.write_text(
        "interceptions\n"
        "1\n"
        "1\n"
        "2\n"
        "2\n"
        "3\n"
        "3\n",
        encoding="utf-8",
    )
    datasets = [
        load_csv(left, dataset_id="ds_left"),
        load_csv(right, dataset_id="ds_right"),
    ]

    candidates = discover_relationship_candidates(datasets, _engine(datasets))

    metric_candidates = [
        candidate
        for candidate in candidates.candidates
        if candidate.pair.left_columns == ["interceptions"]
        and candidate.pair.right_columns == ["interceptions"]
    ]
    assert all(candidate.confidence == "low" for candidate in metric_candidates)
    assert all(not candidate.auto_adopted for candidate in metric_candidates)


def test_competing_targets_for_same_left_column_are_demoted(tmp_path: Path) -> None:
    left = tmp_path / "matches.csv"
    left.write_text(
        "home_team\n"
        "Mexico\n"
        "Canada\n"
        "United States\n",
        encoding="utf-8",
    )
    right = tmp_path / "teams.csv"
    right.write_text(
        "team,team_country\n"
        "Mexico,Mexico\n"
        "Canada,Canada\n"
        "United States,United States\n",
        encoding="utf-8",
    )
    datasets = [
        load_csv(left, dataset_id="ds_matches"),
        load_csv(right, dataset_id="ds_teams"),
    ]

    candidates = discover_relationship_candidates(datasets, _engine(datasets))

    home_team_candidates = [
        candidate
        for candidate in candidates.candidates
        if candidate.pair.left_dataset_name == "matches.csv"
        and candidate.pair.left_columns == ["home_team"]
        and candidate.pair.right_dataset_name == "teams.csv"
        and candidate.pair.right_columns[0] in {"team", "team_country"}
    ]
    retained = [
        candidate for candidate in home_team_candidates if candidate.confidence != "low"
    ]
    demoted = [
        candidate for candidate in home_team_candidates if candidate.confidence == "low"
    ]

    assert len(retained) == 1
    assert retained[0].pair.right_columns == ["team"]
    assert retained[0].auto_adopted is False
    assert {cast(str, candidate.pair.right_columns[0]) for candidate in demoted} == {
        "team_country"
    }
    assert all(not candidate.auto_adopted for candidate in demoted)


def test_ecommerce_relationship_precision_and_recall_harness() -> None:
    datasets = _load_ecommerce()
    engine = _engine(datasets)
    expected_true_fk = {
        ("ecommerce_orders.csv", "customer_id", "ecommerce_customers.csv", "customer_id"),
        ("ecommerce_orders.csv", "product_id", "ecommerce_products.csv", "product_id"),
    }

    candidates = discover_relationship_candidates(datasets, engine)
    predicted = {
        _candidate_key(candidate)
        for candidate in candidates.candidates
        if candidate.confidence in {"high", "medium"}
    }
    true_positive = predicted & expected_true_fk
    precision = len(true_positive) / len(predicted) if predicted else 0.0
    recall = len(true_positive) / len(expected_true_fk)

    assert precision >= 0.9, predicted
    assert recall >= 0.8, predicted


def test_er_diagram_contains_expected_nodes_edges_and_dashed_medium_edge() -> None:
    datasets = _load_ecommerce()
    engine = _engine(datasets)
    candidates = discover_relationship_candidates(datasets, engine)
    validations = validate_relationships(candidates, engine)

    diagram = build_er_diagram(candidates, validations)

    assert "digraph er" in diagram.dot_source
    assert "ecommerce_orders.csv" in diagram.dot_source
    assert "ecommerce_customers.csv" in diagram.dot_source
    assert "customer_id" in diagram.dot_source
    # Tables are HTML-like labels: name on top, one key per stacked port row, so
    # they render vertically under rankdir=LR (fixes the horizontal record that
    # made edges cross).
    assert "<table" in diagram.dot_source
    assert 'port="f' in diagram.dot_source
    # Edge attaches to a specific key-column port (not the whole box): the tail
    # exits ds_orders on a port, the head enters ds_customers on a port.
    assert '"ds_orders":' in diagram.dot_source
    assert '-> "ds_customers":' in diagram.dot_source
    assert 'label="many_to_many / medium"' in diagram.dot_source
    assert 'style="dashed"' in diagram.dot_source
    assert any(row.confidence == "low" for row in diagram.relations)


def test_relationship_discovery_is_byte_deterministic() -> None:
    datasets = _load_ecommerce()

    first = discover_relationship_candidates(datasets, _engine(datasets))
    second = discover_relationship_candidates(datasets, _engine(datasets))

    assert first.model_dump_json() == second.model_dump_json()


def test_overlap_queries_are_bidirectional_and_bounded(monkeypatch) -> None:
    datasets = _load_ecommerce()
    engine = _engine(datasets)
    original_execute = engine.execute_select
    query_count = 0

    def counted_execute(sql: str):  # noqa: ANN202 - mirrors engine dynamically
        nonlocal query_count
        query_count += 1
        return original_execute(sql)

    monkeypatch.setattr(engine, "execute_select", counted_execute)
    candidates = discover_relationship_candidates(
        datasets,
        engine,
        max_overlap_checks_per_dataset_pair=2,
    )

    dataset_pair_count = len(datasets) * (len(datasets) - 1) // 2
    assert candidates.overlap_pairs_evaluated <= dataset_pair_count * 2
    assert candidates.overlap_pairs_prefiltered > 0
    assert candidates.coverage_status == "limited"
    # Each overlap query yields both A→B and B→A hypotheses.
    assert query_count == candidates.overlap_pairs_evaluated
    assert len(candidates.candidates) <= candidates.overlap_pairs_evaluated * 2


def test_eager_validation_only_targets_high_confidence_id_edges() -> None:
    datasets = _load_ecommerce()
    candidates = discover_relationship_candidates(datasets, _engine(datasets))

    targets = eager_validation_candidates(candidates)

    assert all(candidate.confidence == "high" for candidate in targets)
    assert all(
        candidate.pair.left_columns[0].endswith("_id")
        and candidate.pair.right_columns[0].endswith("_id")
        for candidate in targets
    )
    medium_customer = _candidate(
        candidates,
        "ecommerce_orders.csv",
        "customer_id",
        "ecommerce_customers.csv",
        "customer_id",
    )
    assert medium_customer.confidence == "medium"
    assert medium_customer not in targets

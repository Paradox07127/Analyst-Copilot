from __future__ import annotations

from pathlib import Path

from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    eager_validation_candidates,
    validate_relationships,
)

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"
EXPECTED_EDGES = {
    "ecommerce_orders.csv.customer_id -> ecommerce_customers.csv.customer_id",
    "ecommerce_orders.csv.product_id -> ecommerce_products.csv.product_id",
}


def _datasets() -> list[LoadedDataset]:
    names = [
        "ecommerce_orders.csv",
        "ecommerce_customers.csv",
        "ecommerce_products.csv",
        "ecommerce_marketing.csv",
    ]
    return [
        load_csv(GOLDEN_DATA / name, dataset_id=f"golden_{index}")
        for index, name in enumerate(names)
    ]


def _engine(datasets: list[LoadedDataset]) -> DuckDBQueryEngine:
    engine = DuckDBQueryEngine()
    for dataset in datasets:
        engine.register_frame(dataset.record.dataset_id, dataset.frame)
    return engine


def test_bounded_discovery_preserves_expected_join_recall_and_overlap_floor() -> None:
    datasets = _datasets()
    candidates = discover_relationship_candidates(datasets, _engine(datasets))
    useful = {
        candidate.pair.label()
        for candidate in candidates.candidates
        if candidate.confidence in {"high", "medium"}
    }

    assert EXPECTED_EDGES <= useful
    assert all(
        candidate.signals.overlap_left_in_right >= 0.60
        for candidate in candidates.candidates
        if candidate.confidence in {"high", "medium"}
    )
    assert candidates.overlap_pairs_evaluated <= 4 * 6  # 4 datasets => 6 pairs
    assert candidates.overlap_pairs_prefiltered > 0
    assert candidates.coverage_status == "limited"


def test_eager_validation_is_safe_subset_and_deferred_edge_remains_validatable() -> None:
    datasets = _datasets()
    engine = _engine(datasets)
    candidates = discover_relationship_candidates(datasets, engine)
    eager = validate_relationships(eager_validation_candidates(candidates), engine)
    eager_labels = {validation.pair.label() for validation in eager.validations}

    product_edge = "ecommerce_orders.csv.product_id -> ecommerce_products.csv.product_id"
    customer_edge = "ecommerce_orders.csv.customer_id -> ecommerce_customers.csv.customer_id"
    assert eager_labels == {product_edge}

    customer_candidate = next(
        candidate
        for candidate in candidates.candidates
        if candidate.pair.label() == customer_edge
    )
    deferred = validate_relationships([customer_candidate], engine)
    assert deferred.validations[0].verified is True
    assert deferred.validations[0].cardinality == "many_to_many"


def test_uniqueness_without_overlap_never_reaches_medium_confidence() -> None:
    datasets = _datasets()
    candidates = discover_relationship_candidates(datasets, _engine(datasets))

    zero_overlap = [
        candidate
        for candidate in candidates.candidates
        if candidate.signals.overlap_left_in_right == 0
    ]
    assert zero_overlap
    assert all(candidate.confidence == "low" for candidate in zero_overlap)

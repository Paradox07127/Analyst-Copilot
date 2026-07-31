"""Join-correctness eval: relationship engine vs human-annotated ground truth.

Runs the deterministic relationship-discovery engine over the e-commerce
multi-table golden set and scores it against
``ecommerce_relations_ground_truth.json`` (annotation basis documented in the
file's ``_meta`` block):

- precision >= 0.9 and recall >= 0.8 over FK-direction predictions at
  confidence >= medium (M4 plan §8 DoD line 1);
- every annotated trap reading (join multiplier, orphan rates, cardinality,
  warnings) matches the DuckDB validation numbers exactly;
- every annotated negative stays below medium confidence.

Measured on 2026-07-05: precision 2/2 = 1.0, recall 2/2 = 1.0 (thresholds in
the assertions are the M4 acceptance line, not the measured ceiling).

Deterministic — runs in regular pytest, no LLM involved. Third multi-table set
(football, user-local) is not in the repo; recorded as deferred in
``docs/archive/2026-07/base/eda-agent-platform-m4-eval-report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.schemas.relations import (
    RelationshipCandidateSet,
    RelationshipValidationSet,
)
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.relationship_discovery import (
    discover_relationship_candidates,
    validate_relationships,
)

GROUND_TRUTH_PATH = Path(__file__).parent / "ecommerce_relations_ground_truth.json"
GOLDEN_DATA = Path(__file__).parents[2] / "golden" / "data"

PRECISION_FLOOR = 0.9  # M4 DoD line 1
RECALL_FLOOR = 0.8  # M4 DoD line 1

DirectedRelation = tuple[str, str, str, str]

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def _relation_key(entry: dict) -> DirectedRelation:
    return (
        entry["left_dataset"],
        entry["left_column"],
        entry["right_dataset"],
        entry["right_column"],
    )


def _load_datasets() -> list[LoadedDataset]:
    return [
        load_csv(GOLDEN_DATA / "ecommerce_orders.csv", dataset_id="ds_orders"),
        load_csv(GOLDEN_DATA / "ecommerce_customers.csv", dataset_id="ds_customers"),
        load_csv(GOLDEN_DATA / "ecommerce_products.csv", dataset_id="ds_products"),
        load_csv(GOLDEN_DATA / "ecommerce_marketing.csv", dataset_id="ds_marketing"),
    ]


@pytest.fixture(scope="module")
def discovery() -> tuple[RelationshipCandidateSet, RelationshipValidationSet]:
    datasets = _load_datasets()
    engine = DuckDBQueryEngine()
    for loaded in datasets:
        engine.register_frame(loaded.record.dataset_id, loaded.frame)
    candidates = discover_relationship_candidates(datasets, engine)
    validations = validate_relationships(candidates, engine)
    return candidates, validations


def _predicted_at_least_medium(
    candidates: RelationshipCandidateSet,
) -> dict[DirectedRelation, str]:
    predicted: dict[DirectedRelation, str] = {}
    for candidate in candidates.candidates:
        if candidate.confidence not in {"medium", "high"}:
            continue
        pair = candidate.pair
        key = (
            pair.left_dataset_name,
            pair.left_columns[0],
            pair.right_dataset_name,
            pair.right_columns[0],
        )
        predicted[key] = candidate.confidence
    return predicted


def test_ground_truth_file_is_annotated() -> None:
    truth = _ground_truth()
    meta = truth["_meta"]
    assert meta["annotation_basis"], "ground truth must document its annotation basis"
    assert meta["direction_semantics"].startswith("left = referencing")
    assert len(truth["true_relations"]) == 2
    assert len(truth["negative_relations"]) >= 5
    keys = [_relation_key(entry) for entry in truth["true_relations"]]
    assert len(set(keys)) == len(keys)


def test_fk_precision_and_recall_meet_m4_acceptance_line(
    discovery: tuple[RelationshipCandidateSet, RelationshipValidationSet],
) -> None:
    candidates, _ = discovery
    truth = _ground_truth()
    expected = {_relation_key(entry) for entry in truth["true_relations"]}
    predicted = set(_predicted_at_least_medium(candidates))

    true_positives = predicted & expected
    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(expected)

    detail = (
        f"predicted(>=medium)={sorted(predicted)} expected={sorted(expected)} "
        f"precision={precision:.3f} recall={recall:.3f}"
    )
    assert precision >= PRECISION_FLOOR, detail
    assert recall >= RECALL_FLOOR, detail


def test_every_true_relation_meets_annotated_confidence_and_adoption(
    discovery: tuple[RelationshipCandidateSet, RelationshipValidationSet],
) -> None:
    candidates, _ = discovery
    by_key = {
        (
            candidate.pair.left_dataset_name,
            candidate.pair.left_columns[0],
            candidate.pair.right_dataset_name,
            candidate.pair.right_columns[0],
        ): candidate
        for candidate in candidates.candidates
    }
    for entry in _ground_truth()["true_relations"]:
        key = _relation_key(entry)
        assert key in by_key, f"true relation not even proposed: {key}"
        candidate = by_key[key]
        floor = entry["expected_confidence_at_least"]
        assert (
            _CONFIDENCE_RANK[candidate.confidence] >= _CONFIDENCE_RANK[floor]
        ), f"{key}: confidence {candidate.confidence} below annotated floor {floor}"
        assert candidate.auto_adopted is entry["expected_auto_adopted"], (
            f"{key}: auto_adopted={candidate.auto_adopted}, "
            f"annotation says {entry['expected_auto_adopted']} "
            "(relationships are reference-only since PR #5: none auto-adopt; a "
            "join is a deliberate choice inside an approved analysis plan)"
        )


def test_validation_numbers_match_hand_computed_trap_readings(
    discovery: tuple[RelationshipCandidateSet, RelationshipValidationSet],
) -> None:
    _, validations = discovery
    by_key = {
        (
            validation.pair.left_dataset_name,
            validation.pair.left_columns[0],
            validation.pair.right_dataset_name,
            validation.pair.right_columns[0],
        ): validation
        for validation in validations.validations
    }
    for entry in _ground_truth()["true_relations"]:
        key = _relation_key(entry)
        expected = entry["expected_validation"]
        assert key in by_key, f"true relation was not validated: {key}"
        validation = by_key[key]
        assert validation.verified is True
        assert validation.join_row_multiplier == pytest.approx(
            expected["join_row_multiplier"], abs=1e-6
        ), key
        assert validation.orphan_rate_left == pytest.approx(
            expected["orphan_rate_left"], abs=1e-6
        ), key
        assert validation.orphan_rate_right == pytest.approx(
            expected["orphan_rate_right"], abs=1e-6
        ), key
        assert validation.cardinality == expected["cardinality"], key
        if expected["warnings_required"]:
            assert validation.warnings, f"{key}: trap must carry warnings"
        else:
            assert not validation.warnings, f"{key}: clean FK must not warn"
        assert "left join" in validation.verification_sql.lower(), key


def test_annotated_negatives_never_reach_medium_confidence(
    discovery: tuple[RelationshipCandidateSet, RelationshipValidationSet],
) -> None:
    candidates, _ = discovery
    predicted = _predicted_at_least_medium(candidates)
    offenders = [
        (key, predicted[key])
        for entry in _ground_truth()["negative_relations"]
        if (key := _relation_key(entry)) in predicted
    ]
    assert not offenders, f"annotated negatives crossed the medium line: {offenders}"

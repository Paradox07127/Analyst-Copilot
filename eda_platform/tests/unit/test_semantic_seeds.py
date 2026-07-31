"""M6.1 T5 — SemanticSeeds backward compatibility and new FR-15 knowledge classes.

The load-compat tests are the DoD gate: an old ``seeds.json`` written before the
M6 knowledge classes existed must still load, with the new lists defaulting to
empty rather than raising.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from semantic_test_helpers import load_seeds, save_seeds

from eda_platform.core.semantic import (
    FieldMeaning,
    MetricDefinition,
    SemanticSeeds,
    VerifiedAnswer,
    VerifiedRelation,
)

# --- backward compatibility: legacy files load into the extended model ------


def test_legacy_seeds_file_loads_with_empty_new_classes(tmp_path: Path) -> None:
    """A pre-M6 file (only version + relations + entity_notes) loads unchanged."""
    legacy = {
        "version": 1,
        "verified_relations": [
            {
                "left": "players.team_id",
                "right": "teams.team_id",
                "cardinality": "many_to_one",
                "confirmed_by": "user",
                "confirmed_at": "2026-07-03T00:00:00+00:00",
                "source_session_id": "run_a",
            }
        ],
        "entity_notes": [{"name": "team", "note": "A club roster."}],
    }
    path = tmp_path / "semantic" / "seeds.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy), encoding="utf-8")

    seeds = load_seeds(tmp_path)

    assert seeds.version == 1
    assert seeds.verified_relations[0].right == "teams.team_id"
    assert seeds.entity_notes[0].name == "team"
    # New FR-15 classes are absent in the file -> default to empty.
    assert seeds.field_meanings == []
    assert seeds.metric_definitions == []
    assert seeds.verified_answers == []


def test_empty_object_loads_as_all_empty() -> None:
    """The most minimal legacy file — just ``{}`` — is still valid."""
    seeds = SemanticSeeds.model_validate({})

    assert seeds.version == 1
    assert seeds.verified_relations == []
    assert seeds.entity_notes == []
    assert seeds.field_meanings == []
    assert seeds.metric_definitions == []
    assert seeds.verified_answers == []


def test_missing_file_returns_empty_seeds_with_new_classes(tmp_path: Path) -> None:
    seeds = load_seeds(tmp_path)

    assert seeds.field_meanings == []
    assert seeds.metric_definitions == []
    assert seeds.verified_answers == []
    assert not (tmp_path / "semantic" / "seeds.json").exists()


# --- new knowledge classes: defaults, optionals, round-trip -----------------


def test_field_meaning_optionals_default_empty() -> None:
    field = FieldMeaning(dataset="orders.csv", column="gmv", meaning="Gross value.")

    assert field.unit is None
    assert field.aliases == []


def test_metric_and_answer_optionals_default_none() -> None:
    metric = MetricDefinition(name="Active user", definition="≥1 session / 28d.")
    answer = VerifiedAnswer(question="Q3 revenue?", answer="$4.2M")

    assert metric.formula is None
    assert metric.caveats is None
    assert answer.evidence_note is None
    assert isinstance(answer.verified_at, datetime)


def test_full_seeds_roundtrip_persists_all_five_classes(tmp_path: Path) -> None:
    seeds = SemanticSeeds(
        verified_relations=[
            VerifiedRelation(
                left="players.team_id",
                right="teams.team_id",
                cardinality="many_to_one",
            )
        ],
        entity_notes=[],
        field_meanings=[
            FieldMeaning(
                dataset="orders.csv",
                column="gmv",
                meaning="Gross merchandise value.",
                unit="USD",
                aliases=["gross_sales", "gmv_usd"],
            )
        ],
        metric_definitions=[
            MetricDefinition(
                name="Active user",
                definition="A user with >=1 session in 28 days.",
                formula="count(distinct user_id)",
                caveats="Excludes test accounts.",
            )
        ],
        verified_answers=[
            VerifiedAnswer(
                question="What was Q3 revenue?",
                answer="$4.2M, up 12% QoQ.",
                evidence_note="Audited close.",
                verified_at=datetime(2026, 7, 4, tzinfo=UTC),
            )
        ],
    )

    save_seeds(tmp_path, seeds)
    loaded = load_seeds(tmp_path)

    assert loaded.field_meanings[0].unit == "USD"
    assert loaded.field_meanings[0].aliases == ["gross_sales", "gmv_usd"]
    assert loaded.metric_definitions[0].formula == "count(distinct user_id)"
    assert loaded.metric_definitions[0].caveats == "Excludes test accounts."
    assert loaded.verified_answers[0].answer == "$4.2M, up 12% QoQ."
    assert loaded.verified_answers[0].evidence_note == "Audited close."
    assert loaded.verified_answers[0].verified_at == datetime(2026, 7, 4, tzinfo=UTC)
    # Existing classes untouched by the extension.
    assert loaded.verified_relations[0].right == "teams.team_id"


def test_partial_new_classes_load_and_keep_defaults(tmp_path: Path) -> None:
    """A file with only field_meanings present leaves the others empty."""
    partial = {
        "version": 1,
        "field_meanings": [
            {"dataset": "orders.csv", "column": "gmv", "meaning": "Gross value."}
        ],
    }
    path = tmp_path / "semantic" / "seeds.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(partial), encoding="utf-8")

    seeds = load_seeds(tmp_path)

    assert len(seeds.field_meanings) == 1
    assert seeds.field_meanings[0].unit is None
    assert seeds.metric_definitions == []
    assert seeds.verified_answers == []
    assert seeds.verified_relations == []

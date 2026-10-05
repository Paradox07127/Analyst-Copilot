"""Model-bound question context obeys policy; lexical dedup keeps data scope."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from eda_platform.agents.question_agent import build_data_summary, propose_llm_question_candidates
from eda_platform.schemas.artifacts import Artifact, DatasetProfile
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import (
    normalized_question_key,
    rank_and_deduplicate_questions,
)

MARKER = "SYNTHETIC_INTERNAL_VALUE"


def _profile() -> Artifact:
    loaded = LoadedDataset(
        record=DatasetRecord(
            dataset_id="synthetic", name="synthetic.csv", path=Path("/unused.csv"),
            content_hash="synthetic",
        ),
        frame=pd.DataFrame({"category": [MARKER, "beta", "gamma"]}),
    )
    return profile_dataset(loaded, project_id="p", session_id="s")


class _RecordingModel:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.payloads.append(payload)
        return schema.model_validate({"questions": []})

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


@pytest.mark.parametrize("policy", ["schema_only", "schema+aggregates", "schema+aggregates+sample"])
def test_question_model_payload_discloses_samples_only_with_sample_policy(
    policy: PayloadPolicy,
) -> None:
    model = _RecordingModel()
    propose_llm_question_candidates([_profile()], llm=model, payload_policy=policy)
    assert model.payloads
    for payload in model.payloads:
        assert (MARKER in json.dumps(payload)) == (policy == "schema+aggregates+sample")
        assert "category" in payload["data_summary"]
        assert "synthetic.csv" in payload["data_summary"]


def test_summary_default_and_negative_sample_limit_cannot_disclose_values() -> None:
    profile = DatasetProfile.model_validate(_profile().payload)
    assert MARKER not in build_data_summary([profile])
    assert MARKER not in build_data_summary(
        [profile], payload_policy="schema+aggregates+sample", max_sample_values=-1
    )


def _question(key: str, text: str, **changes: Any) -> QuestionCandidate:
    return QuestionCandidate.model_validate({
        "question_id": key, "question_en": text, "origin": "llm",
        "target_datasets": ["sales.csv"],
        "score": QuestionScore(
            data_availability=1, statistical_signal=0.5, quality_risk=0,
            join_risk=0, deterministic_score=0.8,
        ),
        **changes,
    })


@pytest.mark.parametrize(("left", "right"), [
    ("哪些客户更容易流失？", "哪些产品的利润率最高？"),
    ("ما المنتجات الأكثر ربحًا؟", "ما العملاء الأكثر نشاطًا؟"),
    ("Revenue > 10", "Revenue < 10"),
    ("Margin 50%", "Margin 50"),
    ("???", "!!!"),
])
def test_distinct_questions_do_not_collapse_after_normalization(left: str, right: str) -> None:
    assert normalized_question_key(left)
    assert normalized_question_key(left) != normalized_question_key(right)
    result = rank_and_deduplicate_questions([_question("a", left), _question("b", right)])
    assert len(result.candidates) == 2


@pytest.mark.parametrize(("left", "right"), [
    ("哪些产品利润高？", "哪些产品利润高!"),
    ("CAFÉ revenue?", "cafe\u0301 revenue!"),
    ("किस उत्पाद का लाभ अधिक है?", "किस उत्पाद का लाभ अधिक है！"),
    ("How is REVENUE trending?", "how is revenue trending!"),
])
def test_equivalent_unicode_spelling_and_punctuation_deduplicate(left: str, right: str) -> None:
    assert normalized_question_key(left) == normalized_question_key(right)
    result = rank_and_deduplicate_questions([_question("a", left), _question("b", right)])
    assert len(result.candidates) == 1
    assert result.dedup_dropped == 1


@pytest.mark.parametrize("changes", [
    {"target_datasets": ["other.csv"]},
    {"target_datasets": ["Sales.csv"]},
    {"referenced_columns": {"sales.csv": ["revenue"]}},
    {"required_relations": ["sales.csv.customer -> customers.csv.id"]},
    {"analysis_mode": "prediction"},
    {"metric_id": "net_revenue"},
    {"produced_units": {"value": "USD"}},
])
def test_same_question_with_different_data_or_analysis_scope_is_retained(changes: dict) -> None:
    result = rank_and_deduplicate_questions([
        _question("a", "Which segment performs best?"),
        _question("b", "Which segment performs best?", **changes),
    ])
    assert len(result.candidates) == 2


def test_scope_order_and_duplicate_identifiers_do_not_prevent_deduplication() -> None:
    first = _question(
        "a", "Which group performs best?", target_datasets=["a.csv", "b.csv"],
        referenced_columns={"a.csv": ["x", "y"], "b.csv": ["z"]},
        required_relations=["r1", "r2"],
    )
    second = _question(
        "b", "Which group performs best!", target_datasets=["b.csv", "a.csv", "a.csv"],
        referenced_columns={"b.csv": ["z"], "a.csv": ["y", "x", "x"]},
        required_relations=["r2", "r1", "r1"],
    )
    assert rank_and_deduplicate_questions([first, second]).dedup_dropped == 1

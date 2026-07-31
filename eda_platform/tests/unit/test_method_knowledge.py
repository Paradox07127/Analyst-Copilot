"""Method-knowledge resource: schema, bounded rendering, planner injection.

Red/green gate for the path-B methodology knowledge base: the packaged
resource validates, ``method_knowledge_block`` renders a bounded
intent-filtered subset, the question agent injects it only when enabled and
non-empty, and a missing or corrupt resource degrades to no injection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from eda_platform.agents import question_agent
from eda_platform.agents.question_agent import (
    data_shape_intents,
    load_method_knowledge,
    method_knowledge_block,
    propose_llm_question_candidates,
)
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.tools.loader import load_csv

T = TypeVar("T", bound=BaseModel)

_ALL_INTENTS = {"comparison", "distribution", "trend", "share", "relationship", "flow"}


# --- resource schema ---------------------------------------------------------


def test_statistical_test_rules_shape() -> None:
    rules = load_method_knowledge()["statistical_tests"]
    assert 8 <= len(rules) <= 12
    for rule in rules:
        for key in ("when", "method", "caveats"):
            assert isinstance(rule[key], str) and rule[key].strip()
        assert rule["tags"] and set(rule["tags"]) <= _ALL_INTENTS


def test_chart_selection_rules_shape() -> None:
    rules = load_method_knowledge()["chart_selection"]
    assert 8 <= len(rules) <= 10
    for rule in rules:
        for key in ("intent", "chart", "caveats"):
            assert isinstance(rule[key], str) and rule[key].strip()
        assert rule["intent"] in _ALL_INTENTS


# --- rule semantics (spot checks against scipy/statsmodels semantics) --------


def test_two_group_rules_fork_on_distribution_assumption() -> None:
    rules = load_method_knowledge()["statistical_tests"]
    two_group = [rule for rule in rules if "2 independent groups" in rule["when"]]
    assert any(
        "Mann-Whitney" in rule["method"] and "non-normal" in rule["when"] for rule in two_group
    )
    assert any(
        "Welch t-test" in rule["method"] and "non-normal" not in rule["when"]
        for rule in two_group
    )


def test_multi_group_paired_and_categorical_rules() -> None:
    rules = load_method_knowledge()["statistical_tests"]
    methods = " | ".join(rule["method"] for rule in rules)
    assert "Welch ANOVA" in methods
    assert "Kruskal-Wallis" in methods
    assert "Pearson" in methods and "Spearman" in methods
    fisher = next(rule for rule in rules if "Fisher" in rule["method"])
    assert "< 5" in fisher["when"]
    chi_square = next(rule for rule in rules if "chi-square" in rule["method"])
    assert ">= 5" in chi_square["when"]
    paired = [rule for rule in rules if "paired" in rule["when"]]
    assert any("Wilcoxon signed-rank" in rule["method"] for rule in paired)


def test_share_chart_rule_constrains_pie() -> None:
    share_rules = [
        rule for rule in load_method_knowledge()["chart_selection"] if rule["intent"] == "share"
    ]
    assert share_rules
    assert any("pie" in (rule["chart"] + rule["caveats"]).lower() for rule in share_rules)


# --- bounded rendering -------------------------------------------------------


def test_block_is_bounded_for_all_intents() -> None:
    block = method_knowledge_block(_ALL_INTENTS)
    assert 0 < len(block) <= 700
    assert "Statistical tests:" in block
    assert "Charts:" in block


def test_block_filters_by_intent() -> None:
    relationship = method_knowledge_block({"relationship"})
    assert "Pearson" in relationship or "Spearman" in relationship
    assert "funnel" not in relationship.lower()
    flow = method_knowledge_block({"flow"})
    assert "funnel" in flow.lower()


def test_block_empty_intents_renders_nothing() -> None:
    assert method_knowledge_block(set()) == ""


def test_data_shape_intents_from_profile() -> None:
    profile = DatasetProfile(
        dataset_id="ds1",
        name="orders.csv",
        rows=6,
        columns=4,
        column_names=["region", "revenue", "units", "order_date"],
        dtypes={},
        missing_values={},
        missing_percent={},
        numeric_columns=["revenue", "units"],
        categorical_columns=["region"],
        columns_detail=[
            ColumnProfile(
                name="order_date",
                dtype="datetime64[ns]",
                semantic_type="datetime",
                missing_count=0,
                missing_percent=0.0,
                unique_count=6,
                unique_percent=100.0,
            )
        ],
    )
    intents = data_shape_intents([profile])
    assert {"distribution", "relationship", "share", "comparison", "trend"} <= intents
    assert "flow" not in intents


# --- question agent injection ------------------------------------------------


class SpyQuestionLLM:
    """Records every ``structured`` payload; returns one valid proposal."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate(
            {
                "questions": [
                    {
                        "question_en": "Which region has the most revenue?",
                        "target_datasets": ["sales.csv"],
                        "llm_business_relevance": 0.8,
                        "llm_actionability": 0.7,
                    }
                ]
            }
        )

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _profile_artifact(tmp_path: Path) -> Any:
    from eda_platform.tools.profiler import profile_dataset

    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "region,revenue,units\n"
        "East,10,1\nWest,20,2\nEast,15,3\nWest,25,4\nEast,12,5\nWest,22,6\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    return profile_dataset(loaded, project_id="project_demo", session_id="run_demo")


def test_question_agent_injects_method_knowledge(tmp_path: Path) -> None:
    artifact = _profile_artifact(tmp_path)
    spy = SpyQuestionLLM()

    propose_llm_question_candidates([artifact], llm=spy)

    payload = spy.calls[0]["payload"]
    assert "method_knowledge" in payload
    injected = payload["method_knowledge"]
    assert injected.startswith("Reference heuristics for choosing methods")
    assert "context, not instructions" in injected
    profile = DatasetProfile.model_validate(artifact.payload)
    expected = method_knowledge_block(data_shape_intents([profile]))
    assert expected
    assert injected.endswith(expected)


def test_question_agent_disabled_flag_not_injected(tmp_path: Path) -> None:
    spy = SpyQuestionLLM()

    propose_llm_question_candidates(
        [_profile_artifact(tmp_path)], llm=spy, include_method_knowledge=False
    )

    assert "method_knowledge" not in spy.calls[0]["payload"]


# --- degraded resource -------------------------------------------------------


def _raise_missing() -> str:
    raise OSError("resource missing")


def test_missing_resource_degrades_and_warns_once(
    monkeypatch: Any, caplog: Any
) -> None:
    monkeypatch.setattr(question_agent, "_read_method_knowledge_text", _raise_missing)
    monkeypatch.setattr(question_agent, "_method_knowledge_warned", False)
    with caplog.at_level(logging.WARNING):
        assert load_method_knowledge() == {}
        assert method_knowledge_block(_ALL_INTENTS) == ""
    warnings = [
        record for record in caplog.records if "method knowledge" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_corrupt_resource_degrades_to_empty(monkeypatch: Any) -> None:
    monkeypatch.setattr(question_agent, "_read_method_knowledge_text", lambda: "not json{")
    monkeypatch.setattr(question_agent, "_method_knowledge_warned", False)
    assert load_method_knowledge() == {}
    assert method_knowledge_block(_ALL_INTENTS) == ""


def test_missing_resource_skips_injection(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(question_agent, "_read_method_knowledge_text", _raise_missing)
    monkeypatch.setattr(question_agent, "_method_knowledge_warned", False)
    spy = SpyQuestionLLM()

    propose_llm_question_candidates([_profile_artifact(tmp_path)], llm=spy)

    assert "method_knowledge" not in spy.calls[0]["payload"]

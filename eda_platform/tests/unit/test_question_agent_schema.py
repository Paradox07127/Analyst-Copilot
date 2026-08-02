"""OpenAI strict structured-output contract for the question agent's raw schema.

Bare `Any` fields make Pydantic emit properties with no `type`, which OpenAI's
json_schema strict mode rejects with HTTP 400 (observed 2026-08-01 on
gpt-5.6-luna). The Raw model must stay lenient at runtime while still
advertising a legal schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.agents.chat_router import Intent
from eda_platform.agents.code_agent import CodeDraft
from eda_platform.agents.followup_agent import _RawFollowUpResponse
from eda_platform.agents.interpretation import _InterpretationDraft
from eda_platform.agents.investigation_loop import _ProbeDecision
from eda_platform.agents.planner import AnalysisPlan
from eda_platform.agents.question_agent import (
    RawLLMQuestionProposal,
    RawLLMQuestionProposalSet,
    _validate_proposals,
)
from eda_platform.agents.reporting import ReportPlanDraft
from eda_platform.agents.semantic_bootstrap import RawSemanticHypotheses
from eda_platform.core.llm import to_strict_json_schema
from eda_platform.drivers.decision_report import _SCQAInterleavedRewrite, _SCQARewrite

STRUCTURED_OUTPUT_SCHEMAS: tuple[type[BaseModel], ...] = (
    RawLLMQuestionProposalSet,
    AnalysisPlan,
    CodeDraft,
    Intent,
    _InterpretationDraft,
    RawSemanticHypotheses,
    _ProbeDecision,
    ReportPlanDraft,
    _RawFollowUpResponse,
    _SCQARewrite,
    _SCQAInterleavedRewrite,
)

_TYPE_KEYS = ("type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const")


def _untyped_properties(node: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            found.extend(
                f"{path}.{name}"
                for name, sub in properties.items()
                if not any(key in sub for key in _TYPE_KEYS)
            )
        for key, sub in node.items():
            found.extend(_untyped_properties(sub, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            found.extend(_untyped_properties(sub, f"{path}[{index}]"))
    return found


def _strict_mode_violations(node: Any, path: str = "$") -> list[str]:
    """Objects must close additionalProperties and require every property."""
    problems: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append(
                    f"{path}: additionalProperties={node.get('additionalProperties')!r}"
                )
            properties = node.get("properties")
            if isinstance(properties, dict):
                missing = sorted(set(properties) - set(node.get("required") or []))
                if missing:
                    problems.append(f"{path}: not required {missing}")
        for key, sub in node.items():
            problems.extend(_strict_mode_violations(sub, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, sub in enumerate(node):
            problems.extend(_strict_mode_violations(sub, f"{path}[{index}]"))
    return problems


def test_raw_question_schema_has_no_untyped_property() -> None:
    untyped = _untyped_properties(RawLLMQuestionProposalSet.model_json_schema())
    assert untyped == [], f"OpenAI rejects properties without a type: {untyped}"


def test_raw_question_schema_satisfies_openai_strict_mode() -> None:
    strict = to_strict_json_schema(RawLLMQuestionProposalSet.model_json_schema())
    assert _untyped_properties(strict) == []
    assert _strict_mode_violations(strict) == []


@pytest.mark.parametrize("model", STRUCTURED_OUTPUT_SCHEMAS, ids=lambda m: m.__name__)
def test_structured_output_schemas_have_typed_properties(model: type[BaseModel]) -> None:
    untyped = _untyped_properties(to_strict_json_schema(model.model_json_schema()))
    assert untyped == [], f"{model.__name__} has untyped properties: {untyped}"


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis_mode": 123},
        {"risks": "a string instead of a list"},
        {"target_datasets": {"orders.csv": 1}},
        {"llm_business_relevance": "high"},
        {"dataset_display_names": ["orders.csv"]},
        {"question_en": None},
        {},
    ],
    ids=[
        "int_analysis_mode",
        "str_risks",
        "dict_target_datasets",
        "str_relevance",
        "list_display_names",
        "null_question",
        "all_fields_missing",
    ],
)
def test_raw_proposal_still_accepts_malformed_model_output(payload: dict[str, Any]) -> None:
    proposal = RawLLMQuestionProposal.model_validate(payload)
    for field, value in payload.items():
        assert getattr(proposal, field) == value


def _one_proposal(**overrides: Any) -> RawLLMQuestionProposalSet:
    question = {
        "question_en": "Which region drives the most revenue?",
        "target_datasets": ["orders.csv"],
        "llm_business_relevance": 0.8,
        "llm_actionability": 0.7,
        **overrides,
    }
    return RawLLMQuestionProposalSet.model_validate({"questions": [question]})


@pytest.mark.parametrize(
    ("display_names", "expected"),
    [
        ([{"dataset": "orders.csv", "display_name": "Orders"}], {"orders.csv": "Orders"}),
        ({"orders.csv": "Orders"}, {"orders.csv": "Orders"}),
        ([], {}),
    ],
    ids=["strict_mode_pairs", "json_object_mapping", "empty_list"],
)
def test_display_names_accepted_as_pairs_or_mapping(
    display_names: Any, expected: dict[str, str]
) -> None:
    outcome = _validate_proposals(
        _one_proposal(dataset_display_names=display_names), known_datasets={"orders.csv"}
    )
    assert outcome.error is None
    assert outcome.accepted[0].dataset_display_names == expected


def test_downstream_normalization_survives_malformed_proposals() -> None:
    raw = RawLLMQuestionProposalSet.model_validate(
        {
            "questions": [
                {
                    "question_en": "Which region drives the most revenue?",
                    "target_datasets": "orders.csv",
                    "dataset_display_names": {"orders.csv": "Orders"},
                    "llm_business_relevance": 0.8,
                    "llm_actionability": 0.7,
                    "risks": "Seasonality may skew the comparison.",
                    "analysis_mode": 123,
                },
                {"question_en": None, "target_datasets": {"bad": 1}},
            ]
        }
    )
    outcome = _validate_proposals(raw, known_datasets={"orders.csv"})
    assert len(outcome.accepted) == 1
    accepted = outcome.accepted[0]
    assert accepted.target_datasets == ["orders.csv"]
    assert accepted.risks == ["Seasonality may skew the comparison."]
    assert accepted.analysis_mode is None
    assert outcome.dropped_count == 1

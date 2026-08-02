"""Chat routing schemas must survive OpenAI json_schema strict mode.

Strict mode rejects dynamic-key objects (`additionalProperties: true` with no
`properties`) with HTTP 400, which took down every OpenAI model on the chat path.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.agents import chat_router
from eda_platform.core.llm import to_strict_json_schema


def _walk(node: Any, path: str) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        found.append((path, node))
        for key in ("$defs", "definitions", "properties"):
            for name, sub in (node.get(key) or {}).items():
                found.extend(_walk(sub, f"{path}.{key}.{name}"))
        for key in ("anyOf", "oneOf", "allOf"):
            for index, sub in enumerate(node.get(key) or []):
                found.extend(_walk(sub, f"{path}.{key}[{index}]"))
        if "items" in node:
            found.extend(_walk(node["items"], f"{path}.items"))
    return found


def _is_object(node: dict[str, Any]) -> bool:
    return node.get("type") == "object" or "properties" in node


def assert_strict_legal(schema: type[BaseModel]) -> None:
    strict = to_strict_json_schema(schema.model_json_schema())
    for path, node in _walk(strict, schema.__name__):
        if _is_object(node):
            properties = node.get("properties")
            assert properties, f"{path}: dynamic-key object is rejected by strict mode: {node}"
            assert node.get("additionalProperties") is False, f"{path}: additionalProperties"
            assert sorted(node.get("required") or []) == sorted(properties), f"{path}: required"
            for name, sub in properties.items():
                assert isinstance(sub, dict) and (
                    {"type", "anyOf", "oneOf", "allOf", "$ref", "enum"} & set(sub)
                ), f"{path}.{name}: property has no declared type"


@pytest.mark.parametrize("schema", chat_router.STRUCTURED_OUTPUT_SCHEMAS, ids=lambda s: s.__name__)
def test_structured_output_schemas_are_strict_legal(schema: type[BaseModel]) -> None:
    assert_strict_legal(schema)


def _intent(params: Any) -> dict[str, Any]:
    return {
        "kind": "new_analysis",
        "params": params,
        "confidence": 0.9,
        "raw_message": "sales by region",
    }


def test_pairs_and_mapping_parse_to_the_same_params() -> None:
    from_pairs = chat_router.RoutedIntent.model_validate(
        _intent([{"name": "dataset", "value": "orders.csv"}, {"name": "column", "value": "region"}])
    )
    from_mapping = chat_router.RoutedIntent.model_validate(
        _intent({"dataset": "orders.csv", "column": "region"})
    )
    assert from_pairs.params == {"dataset": "orders.csv", "column": "region"}
    assert from_pairs.params == from_mapping.params


@pytest.mark.parametrize(
    "params, expected",
    [
        ([], {}),
        ({}, {}),
        (
            {"unexpected": 3, "listed": [1, 2], "empty": None},
            {"unexpected": 3, "listed": [1, 2], "empty": None},
        ),
        (
            [{"name": "unexpected", "value": 3}, {"name": "listed", "value": [1, 2]}],
            {"unexpected": 3, "listed": [1, 2]},
        ),
        ([{"name": "kept", "value": "x"}, {"no_name": 1}, "junk"], {"kept": "x"}),
    ],
)
def test_unexpected_params_still_parse(params: Any, expected: dict[str, Any]) -> None:
    assert chat_router.RoutedIntent.model_validate(_intent(params)).params == expected

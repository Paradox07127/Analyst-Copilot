"""`orders.amount` is how you name a column in a join, and the guard refused it.

2026-08-06 NL2SQL eval: two of twenty golden questions died on
"Unknown column: ecommerce_orders.amount". `guard_plan_references` compared
`plan.columns` against a flat set of bare names, so qualifying a column -- which
a join *requires* in the SQL, and which the model naturally mirrors in the
`columns` field -- read as a hallucination. Whether a cross-table question
survived came down to whether the model happened to write the prefix.
"""

from __future__ import annotations

import pytest

from eda_platform.agents.planner import guard_plan_references
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.schemas.plans import AnalysisPlan

_CATALOG = {
    "ecommerce_orders": {"order_id", "amount", "product_id"},
    "ecommerce_products": {"product_id", "category"},
}


def _plan(columns: list[str], *, datasets: list[str] | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        question="Total order amount by product category?",
        dataset_names=datasets or ["ecommerce_orders", "ecommerce_products"],
        columns=columns,
        filters=[],
        sql="select p.category, sum(o.amount) from ecommerce_orders o join ecommerce_products p"
        " on o.product_id = p.product_id group by p.category",
        method="join and aggregate",
        rationale="joins orders to products",
        needs_approval=False,
        estimated_scan="small",
    )


def test_a_qualified_column_is_accepted() -> None:
    guard_plan_references(
        _plan(["ecommerce_orders.amount", "ecommerce_products.category"]), _CATALOG
    )


def test_bare_and_qualified_names_may_be_mixed() -> None:
    """The model has no reason to be consistent, and consistency is not the point."""
    guard_plan_references(_plan(["amount", "ecommerce_products.category"]), _CATALOG)


def test_a_qualifier_naming_a_dataset_outside_the_plan_is_still_refused() -> None:
    """Accepting the prefix must not become accepting anything with a dot in it."""
    with pytest.raises(ToolGuardError, match="ecommerce_marketing.spend"):
        guard_plan_references(_plan(["ecommerce_marketing.spend"]), _CATALOG)


def test_a_qualified_column_the_named_dataset_lacks_is_still_refused() -> None:
    with pytest.raises(ToolGuardError, match="ecommerce_products.amount"):
        guard_plan_references(_plan(["ecommerce_products.amount"]), _CATALOG)


def test_a_hallucinated_bare_column_is_still_refused() -> None:
    with pytest.raises(ToolGuardError, match="revenue"):
        guard_plan_references(_plan(["revenue"]), _CATALOG)


def test_the_plan_schema_says_what_columns_means() -> None:
    """The field the guard checks was never described to the model that fills it.

    2026-08-06 eval, the last failing case: `columns` came back as
    ['month', 'total_amount'] -- the aliases the query produces, not the columns
    it reads -- for "aggregate order amount by month". Both readings are
    reasonable from the name alone, and only one passes the guard.

    Asserted through `to_strict_json_schema` because that is what actually
    ships: a converter that strips unknown keys would drop the description on
    the way out and leave this documented only to us.
    """
    from eda_platform.core.llm import to_strict_json_schema

    schema = to_strict_json_schema(AnalysisPlan.model_json_schema())
    description = str(schema["properties"]["columns"].get("description", ""))

    assert description, "the model cannot honour a convention it is never told"
    assert "alias" in description.lower(), description

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.core.query import QueryTimeout, UnsafeQueryError
from eda_platform.schemas.artifacts import ArtifactType, SqlResult
from eda_platform.tools.loader import load_csv
from eda_platform.tools.sql_runner import build_catalog, rewrite_relation_names, run_sql


def test_build_catalog_registers_loaded_datasets_and_runs_join(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    customers = tmp_path / "customers.csv"
    orders.write_text(
        "order_id,customer_id,amount\n"
        "1,CU001,10\n"
        "2,CU002,20\n",
        encoding="utf-8",
    )
    customers.write_text(
        "customer_id,region\n"
        "CU001,East\n"
        "CU002,West\n",
        encoding="utf-8",
    )

    catalog = build_catalog(
        [
            load_csv(orders, dataset_id="ds_orders"),
            load_csv(customers, dataset_id="ds_customers"),
        ]
    )
    artifact = run_sql(
        catalog,
        "select c.region, sum(o.amount) as total_amount "
        "from orders o join customers c on o.customer_id = c.customer_id "
        "group by c.region order by c.region",
        project_id="project_demo",
        session_id="run_demo",
        preview_rows=10,
    )

    result = SqlResult.model_validate(artifact.payload)

    assert artifact.type is ArtifactType.SQL_RESULT
    assert catalog.relations["orders.csv"] == "orders"
    assert result.columns == ["region", "total_amount"]
    assert result.row_count == 2
    assert result.truncated is False
    assert result.rows_preview == [
        {"region": "East", "total_amount": 10.0},
        {"region": "West", "total_amount": 20.0},
    ]


def test_run_sql_marks_truncated_when_preview_is_smaller_than_result(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "order_id,amount\n"
        "1,10\n"
        "2,20\n"
        "3,30\n",
        encoding="utf-8",
    )
    catalog = build_catalog([load_csv(orders, dataset_id="ds_orders")])

    artifact = run_sql(
        catalog,
        "select * from orders order by order_id",
        project_id="project_demo",
        session_id="run_demo",
        preview_rows=2,
    )
    result = SqlResult.model_validate(artifact.payload)

    assert result.row_count == 3
    assert result.truncated is True
    assert len(result.rows_preview) == 2


def test_run_sql_rejects_unsafe_queries(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("order_id,amount\n1,10\n", encoding="utf-8")
    catalog = build_catalog([load_csv(orders, dataset_id="ds_orders")])

    with pytest.raises(UnsafeQueryError):
        run_sql(
            catalog,
            "select * from read_csv('/etc/passwd')",
            project_id="project_demo",
            session_id="run_demo",
        )


def test_run_sql_zero_timeout_raises_timeout(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("order_id,amount\n1,10\n", encoding="utf-8")
    catalog = build_catalog([load_csv(orders, dataset_id="ds_orders")])

    with pytest.raises(QueryTimeout):
        run_sql(
            catalog,
            "select * from orders",
            project_id="project_demo",
            session_id="run_demo",
            timeout_seconds=0,
        )


def test_run_sql_serialises_list_aggregates(tmp_path: Path) -> None:
    """A ``list()`` aggregate yields array cells; ``pd.isna`` on those returns a
    mask, which used to raise ValueError and fail the whole job (review J3)."""
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\nEast,10\nEast,20\nWest,30\n",
        encoding="utf-8",
    )
    catalog = build_catalog([load_csv(orders, dataset_id="ds_orders")])

    artifact = run_sql(
        catalog,
        "select region, list(amount) as amounts from orders group by 1 order by 1",
        project_id="project_demo",
        session_id="run_demo",
    )

    result = SqlResult.model_validate(artifact.payload)
    assert [row["amounts"] for row in result.rows_preview] == [[10, 20], [30]]
    # Must survive the JSON round trip the API does on every artifact read.
    assert json.loads(json.dumps(result.rows_preview))[0]["amounts"] == [10, 20]


def test_run_sql_nulls_inside_list_aggregate_become_none(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("region,amount\nEast,10\nEast,\n", encoding="utf-8")
    catalog = build_catalog([load_csv(orders, dataset_id="ds_orders")])

    artifact = run_sql(
        catalog,
        "select region, list(amount) as amounts from orders group by 1",
        project_id="project_demo",
        session_id="run_demo",
    )

    assert SqlResult.model_validate(artifact.payload).rows_preview[0]["amounts"] == [10.0, None]


def test_rewrite_relation_names_leaves_string_literals_alone() -> None:
    """The whole-word regex the replay driver uses also rewrote quoted text, so
    a skill labelling its rows with a literal returned wrong data (review J1)."""
    sql = (
        "SELECT 'orders' AS source_label, region FROM orders "
        "WHERE channel <> 'orders' AND note = 'from orders table'"
    )
    rewritten = rewrite_relation_names(sql, {"orders": "sales"})
    assert rewritten == (
        "SELECT 'orders' AS source_label, region FROM sales "
        "WHERE channel <> 'orders' AND note = 'from orders table'"
    )


def test_rewrite_relation_names_handles_escapes_comments_and_quoted_identifiers() -> None:
    sql = (
        "-- keep orders in the comment\n"
        "SELECT 'it''s orders' AS label /* orders */ FROM \"orders\" o "
        "JOIN orders_lines l ON l.id = o.id"
    )
    rewritten = rewrite_relation_names(sql, {"orders": "sales"})
    assert rewritten == (
        "-- keep orders in the comment\n"
        "SELECT 'it''s orders' AS label /* orders */ FROM \"sales\" o "
        "JOIN orders_lines l ON l.id = o.id"
    )


def test_rewrite_relation_names_is_identity_when_names_already_match() -> None:
    """What makes the replay driver's second rewrite a no-op after the service
    already rebound the skill."""
    sql = "SELECT * FROM sales WHERE label = 'orders'"
    assert rewrite_relation_names(sql, {"sales": "sales"}) == sql
    assert rewrite_relation_names(sql, {}) == sql

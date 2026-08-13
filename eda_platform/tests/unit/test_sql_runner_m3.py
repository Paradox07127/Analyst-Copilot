from __future__ import annotations

import json
from pathlib import Path

import pytest

from eda_platform.core.query import QueryTimeout, UnsafeQueryError
from eda_platform.schemas.artifacts import ArtifactType, SqlResult
from eda_platform.tools.loader import DatasetFramePool, defer_csv, load_csv
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


def test_lazy_catalog_materializes_full_csvs_to_private_duckdb(
    tmp_path: Path,
) -> None:
    orders = tmp_path / "orders.csv"
    customers = tmp_path / "customers.csv"
    orders.write_text(
        "order_id,customer_id,amount\n1,C1,10\n2,C2,20\n3,C1,30\n",
        encoding="utf-8",
    )
    customers.write_text(
        "customer_id,region\nC1,East\nC2,West\n",
        encoding="utf-8",
    )
    pool = DatasetFramePool()
    sources = [
        defer_csv(orders, dataset_id="ds_orders", frame_pool=pool),
        defer_csv(customers, dataset_id="ds_customers", frame_pool=pool),
    ]

    catalog = build_catalog(sources)
    result = SqlResult.model_validate(
        run_sql(
            catalog,
            "select c.region, sum(o.amount) total "
            "from orders o join customers c using (customer_id) "
            "group by 1 order by 1",
            project_id="p",
            session_id="s",
        ).payload
    )

    assert result.rows_preview == [
        {"region": "East", "total": 40.0},
        {"region": "West", "total": 20.0},
    ]
    assert catalog.storage is not None
    assert (Path(catalog.storage.name) / "catalog.duckdb").is_file()
    with pytest.raises(RuntimeError, match="not available"):
        catalog.engine.register_trusted_csv("late", orders)


def test_lazy_catalog_preserves_identifier_strings_like_the_loader(tmp_path: Path) -> None:
    """F3 (2026-08-12 review): the pandas loader forces identifier-named
    columns to string; the DuckDB trusted-CSV path must apply the same rule or
    SQL answers and profiles silently disagree about the same column."""
    stores = tmp_path / "stores.csv"
    stores.write_text("store_code,amount\n12,10\n34,20\n", encoding="utf-8")

    # Contract precondition: the pandas path keeps store_code textual.
    pandas_values = list(load_csv(stores, dataset_id="ds_pd").frame["store_code"])
    assert pandas_values == ["12", "34"]

    pool = DatasetFramePool()
    catalog = build_catalog([defer_csv(stores, dataset_id="ds_stores", frame_pool=pool)])
    result = SqlResult.model_validate(
        run_sql(
            catalog,
            "select store_code from stores order by store_code",
            project_id="p",
            session_id="s",
        ).payload
    )

    assert [row["store_code"] for row in result.rows_preview] == ["12", "34"]


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


def test_lazy_catalog_survives_non_utf8_source_files(tmp_path: Path) -> None:
    """Codex pre-commit review (2026-08-13): DuckDB's from_csv_auto only reads
    UTF-8 without the network-installed encodings extension, but the loader
    accepts gb18030/utf-16/latin-1. Non-UTF-8 sources must fall back to the
    pandas frame instead of failing catalog construction."""
    sales = tmp_path / "sales_gbk.csv"
    sales.write_bytes("地区,销售额\n华东,120000\n华北,58000\n".encode("gb18030"))

    pool = DatasetFramePool()
    catalog = build_catalog([defer_csv(sales, dataset_id="ds_gbk", frame_pool=pool)])
    result = SqlResult.model_validate(
        run_sql(
            catalog,
            'select "地区" from sales_gbk order by "销售额" desc',
            project_id="p",
            session_id="s",
        ).payload
    )

    assert [row["地区"] for row in result.rows_preview] == ["华东", "华北"]

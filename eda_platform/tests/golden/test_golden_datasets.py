"""R17: golden datasets that later milestones (M3 SQL, M4 relationships) reuse.

These assert the datasets are wired the way downstream tests expect: a multi-table
e-commerce set with a deliberate join trap, plus a GBK/Chinese dataset.
"""

from __future__ import annotations

from pathlib import Path

from eda_platform.schemas.artifacts import DatasetProfile
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset

GOLDEN_DATA = Path(__file__).parent / "data"


def _profile(name: str) -> DatasetProfile:
    loaded = load_csv(GOLDEN_DATA / name, dataset_id=name)
    artifact = profile_dataset(loaded, project_id="p", session_id="r")
    return DatasetProfile.model_validate(artifact.payload)


def test_ecommerce_tables_have_expected_shapes() -> None:
    orders = _profile("ecommerce_orders.csv")
    customers = _profile("ecommerce_customers.csv")
    products = _profile("ecommerce_products.csv")
    marketing = _profile("ecommerce_marketing.csv")

    assert orders.rows == 8
    assert customers.rows == 6
    assert products.rows == 4
    assert marketing.rows == 6
    assert "customer_id" in orders.column_names
    assert "customer_id" in customers.column_names


def test_ecommerce_customers_has_join_trap() -> None:
    """customer_id is NOT a clean key in customers (duplicate CU004), so an
    orders x customers join would inflate rows — the trap M4 must detect."""
    customers = _profile("ecommerce_customers.csv")
    column = {c.name: c for c in customers.columns_detail}["customer_id"]

    # 6 rows but only 5 distinct customer_ids -> duplicate key.
    assert column.unique_count == 5
    assert customers.rows == 6
    assert "customer_id" not in customers.primary_key_candidates


def test_ecommerce_products_customer_id_is_clean_key() -> None:
    products = _profile("ecommerce_products.csv")
    orders = _profile("ecommerce_orders.csv")

    assert "product_id" in products.primary_key_candidates
    # orders references a customer (CU999) absent from customers -> orphan FK.
    assert "order_id" in orders.primary_key_candidates


def test_chinese_gbk_dataset_profiles_without_error() -> None:
    profile = _profile("chinese_sales_gbk.csv")

    assert "地区" in profile.column_names
    assert profile.rows == 5

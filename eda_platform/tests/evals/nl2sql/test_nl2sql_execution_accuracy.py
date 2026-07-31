"""Offline self-checks proving the NL2SQL execution-accuracy ruler is sound.

Three properties are asserted (no LLM involved):

1. *Soundness*: every golden SQL, evaluated against itself, scores 100%.
2. *Invariance*: semantics-preserving rewrites (row order, column order,
   aliases, equivalent date predicates) still count as equivalent.
3. *Discriminative power*: known-wrong SQL (wrong aggregate, dropped filter,
   naive join over the duplicate-key trap, fan-trap join) must NOT pass.

Live LLM scoring lives in ``test_nl2sql_live_execution_accuracy.py`` and is
gated behind ``EDA_LIVE_LLM_TEST=1``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from eda_platform.core.query import UnsafeQueryError, validate_select_statement
from eda_platform.tools.loader import LoadedDataset

from .nl2sql_eval_harness import (
    DEFAULT_DATA_DIR,
    DEFAULT_GOLDEN_PATH,
    GoldenNL2SQLCase,
    build_case_catalog,
    execute_readonly,
    golden_sql_provider,
    load_golden_cases,
    results_equivalent,
    run_execution_accuracy,
)

CASES = load_golden_cases()
CASES_BY_ID = {case.case_id: case for case in CASES}


# --- golden-file shape --------------------------------------------------------


def test_golden_file_has_20_cases() -> None:
    assert len(CASES) == 20
    assert len({case.case_id for case in CASES}) == 20


def test_golden_cases_reference_existing_datasets() -> None:
    for case in CASES:
        assert case.datasets, case.case_id
        for name in case.datasets:
            path = DEFAULT_DATA_DIR / f"{name}.csv"
            assert path.exists(), f"{case.case_id} references missing dataset {path}"


def test_golden_sql_is_read_only_select() -> None:
    for case in CASES:
        validate_select_statement(case.golden_sql)


def test_golden_file_json_declares_version_and_dataset_dir() -> None:
    raw = json.loads(DEFAULT_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["dataset_dir"] == "eda_platform/tests/golden/data"


# --- soundness: golden vs golden = 100% ----------------------------------------


def test_execution_accuracy_self_check_is_100_percent() -> None:
    result = run_execution_accuracy(CASES, golden_sql_provider)
    assert result.total == 20
    assert result.accuracy == 1.0, "self-check failed:\n" + result.summary_table()


@pytest.mark.parametrize("case_id", sorted(CASES_BY_ID))
def test_each_golden_sql_is_equivalent_to_itself(case_id: str) -> None:
    case = CASES_BY_ID[case_id]
    catalog, _ = build_case_catalog(case.datasets)
    frame = execute_readonly(catalog, case.golden_sql)
    report = results_equivalent(frame, frame.copy())
    assert report.equivalent, report.reason


# --- canary values: pin the golden data itself ---------------------------------


@pytest.mark.parametrize(
    ("case_id", "expected_cell"),
    [
        ("nl2sql_001", 2481.0),  # total order amount
        ("nl2sql_004", 6),  # distinct customers in orders
        ("nl2sql_005", 4),  # orders in Feb 2026
        ("nl2sql_007", 1),  # orphan orders (CU999)
        ("nl2sql_008", 310.125),  # average order amount
        ("nl2sql_011", 8),  # order count
        ("nl2sql_017", 3),  # distinct dirty_sales customers
        ("nl2sql_020", "华东"),  # top region by sales
    ],
)
def test_single_cell_canaries(case_id: str, expected_cell: object) -> None:
    """Guard against silent golden-data drift: these queries return exactly one
    cell with a hand-computed value."""
    case = CASES_BY_ID[case_id]
    catalog, _ = build_case_catalog(case.datasets)
    frame = execute_readonly(catalog, case.golden_sql)
    assert frame.shape == (1, 1), f"{case_id} no longer returns a single cell"
    actual = frame.iloc[0, 0]
    if isinstance(expected_cell, float):
        assert float(actual) == pytest.approx(expected_cell)
    else:
        assert actual == expected_cell


def test_join_trap_case_006_regions_are_deduplicated() -> None:
    case = CASES_BY_ID["nl2sql_006"]
    catalog, _ = build_case_catalog(case.datasets)
    frame = execute_readonly(catalog, case.golden_sql)
    by_region = {row["region"]: row["total_amount"] for row in frame.to_dict("records")}
    assert by_region == pytest.approx(
        {"East": 1174.5, "West": 1119.0, "North": 55.5, "South": 12.0}
    )


# --- invariance: harmless rewrites still pass -----------------------------------


@pytest.mark.parametrize(
    ("case_id", "variant_sql"),
    [
        (  # column order + aliases changed, ORDER BY removed
            "nl2sql_002",
            "select sum(o.amount) as sales_total, p.category as product_category "
            "from ecommerce_orders as o join ecommerce_products as p "
            "on o.product_id = p.product_id group by p.category",
        ),
        (  # different but equivalent date predicate (LIKE on month prefix)
            "nl2sql_005",
            "select count(*) as feb_orders from ecommerce_orders "
            "where order_date like '2026-02%'",
        ),
        (  # row order changed
            "nl2sql_012",
            "select customer_id, sum(amount) as total_amount from ecommerce_orders "
            "group by customer_id order by customer_id asc",
        ),
        (  # anti-join written as LEFT JOIN ... IS NULL
            "nl2sql_007",
            "select count(*) as missing_customers from ecommerce_orders as o "
            "left join (select distinct customer_id from ecommerce_customers) as c "
            "on o.customer_id = c.customer_id where c.customer_id is null",
        ),
        (  # Chinese identifiers with different aliases and ordering
            "nl2sql_015",
            'select sum("销售额") as total_sales, "地区" as region_name '
            'from chinese_sales_gbk group by "地区" order by total_sales',
        ),
    ],
)
def test_equivalence_accepts_semantics_preserving_rewrites(
    case_id: str, variant_sql: str
) -> None:
    case = CASES_BY_ID[case_id]
    catalog, _ = build_case_catalog(case.datasets)
    golden = execute_readonly(catalog, case.golden_sql)
    candidate = execute_readonly(catalog, variant_sql)
    report = results_equivalent(golden, candidate)
    assert report.equivalent, f"{case_id}: {report.reason}"


# --- discriminative power: wrong SQL must fail -----------------------------------


@pytest.mark.parametrize(
    ("case_id", "mutant_sql", "mutation"),
    [
        (
            "nl2sql_001",
            "select avg(amount) as total_amount from ecommerce_orders",
            "aggregate swapped sum->avg",
        ),
        (
            "nl2sql_005",
            "select count(*) as order_count from ecommerce_orders",
            "filter dropped",
        ),
        (
            "nl2sql_006",
            "select c.region, sum(o.amount) as total_amount from ecommerce_orders as o "
            "join ecommerce_customers as c on o.customer_id = c.customer_id "
            "group by c.region order by c.region",
            "naive join over duplicated CU004 double-counts North",
        ),
        (
            "nl2sql_009",
            "select region, sum(spend) as total_spend from ecommerce_marketing "
            "group by region",
            "grouped by the wrong column",
        ),
        (
            "nl2sql_018",
            "select m.region, sum(m.spend) as total_spend, "
            "count(distinct c.customer_id) as customer_count "
            "from ecommerce_marketing as m left join ecommerce_customers as c "
            "on m.region = c.region group by m.region order by m.region",
            "fan trap: spend inflated by per-customer row duplication",
        ),
        (
            "nl2sql_016",
            'select "渠道", avg(coalesce("订单数", 0)) as "平均订单数" '
            "from chinese_sales_gbk group by \"渠道\"",
            "NULL treated as zero changes the average",
        ),
    ],
)
def test_equivalence_rejects_known_wrong_sql(
    case_id: str, mutant_sql: str, mutation: str
) -> None:
    case = CASES_BY_ID[case_id]
    catalog, _ = build_case_catalog(case.datasets)
    golden = execute_readonly(catalog, case.golden_sql)
    candidate = execute_readonly(catalog, mutant_sql)
    report = results_equivalent(golden, candidate)
    assert not report.equivalent, (
        f"{case_id}: ruler failed to reject mutant ({mutation}); "
        f"verdict was: {report.reason}"
    )


def test_run_execution_accuracy_scores_mutants_below_100() -> None:
    """End-to-end: a provider that corrupts two cases scores exactly 18/20."""
    broken = {
        "nl2sql_001": "select avg(amount) as total_amount from ecommerce_orders",
        "nl2sql_011": "select count(distinct order_id) as order_count "
        "from ecommerce_orders where amount > 100",
    }

    def provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
        del datasets
        return broken.get(case.case_id, case.golden_sql)

    result = run_execution_accuracy(CASES, provider)
    assert result.passed_count == 18
    assert {outcome.case_id for outcome in result.failed} == set(broken)


# --- NULL semantics -------------------------------------------------------------


def test_null_rows_are_compared_consistently() -> None:
    catalog, _ = build_case_catalog(("chinese_sales_gbk",))
    with_null = execute_readonly(catalog, 'select "订单数" from chinese_sales_gbk')
    also_with_null = execute_readonly(
        catalog, 'select "订单数" as orders from chinese_sales_gbk'
    )
    without_null = execute_readonly(
        catalog, 'select "订单数" from chinese_sales_gbk where "订单数" is not null'
    )
    assert results_equivalent(with_null, also_with_null).equivalent
    assert not results_equivalent(with_null, without_null).equivalent


# --- float tolerance -------------------------------------------------------------


def test_float_tolerance_boundary() -> None:
    catalog, _ = build_case_catalog(("ecommerce_orders",))
    golden = execute_readonly(catalog, "select 310.125 as v")
    within = execute_readonly(catalog, "select 310.12500000004 as v")
    beyond = execute_readonly(catalog, "select 310.126 as v")
    assert results_equivalent(golden, within, float_tol=1e-6).equivalent
    assert not results_equivalent(golden, beyond, float_tol=1e-6).equivalent


# --- safety: unsafe candidate SQL fails the case, not the run ---------------------


def test_unsafe_candidate_sql_is_recorded_as_failure() -> None:
    def provider(case: GoldenNL2SQLCase, datasets: Sequence[LoadedDataset]) -> str:
        del datasets
        if case.case_id == "nl2sql_001":
            return "drop table ecommerce_orders"
        return case.golden_sql

    result = run_execution_accuracy(CASES[:2], provider)
    first = result.outcomes[0]
    assert not first.passed
    assert first.error is not None and UnsafeQueryError.__name__ in first.error
    assert result.outcomes[1].passed

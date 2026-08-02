"""The profile_slice WHERE guard must reject anything DuckDB does not parse as a filter.

A substring blacklist cannot see FROM-first set operations, GROUP BY, LIMIT or
USING SAMPLE: they leave ``rows_in_slice`` honest while the profile is computed
over a different row set, so the resulting EvidenceReceipt passes its digest
check while reporting statistics that contradict its own row-count fact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eda_platform.agents.data_tools import (
    DataToolContext,
    ProfileSliceArguments,
    build_data_tools,
)
from eda_platform.core.query import UnsafeQueryError
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.slice_profile import compute_slice_profile, validate_where_clause
from eda_platform.tools.sql_runner import build_catalog
from pathlib import Path

# Ten rows, amount constant at 1.0: any mean other than 1.0 is injected data.
_ROWS = 10


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"amount": [1.0] * _ROWS, "region": ["北区"] * _ROWS})


def _context() -> DataToolContext:
    dataset = LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_guard",
            name="guard.csv",
            path=Path("/data/guard.csv"),
            content_hash="hash_guard",
        ),
        frame=_frame(),
    )
    return DataToolContext(
        datasets=[dataset],
        catalog=build_catalog([dataset]),
        project_id="project_guard",
        session_id="run_guard",
        store=None,
        payload_policy="schema+aggregates",
        artifacts=[],
    )


# The single-column projection is what makes a 1-column UNION bind; multi-column
# projections are stopped by the binder, not by the guard.
_UNION_INJECTION = "1=1) union all (from (values (1e12),(2e12),(3e12)) v(x)"

_INJECTIONS = [
    pytest.param(_UNION_INJECTION, id="union_all_values"),
    pytest.param("1=1) union all (from generate_series(1,5)", id="union_generate_series"),
    pytest.param("1=1) UNION ALL (from (values (1e12)) v(x)", id="union_upper"),
    pytest.param("1=1) uNiOn aLl (from (values (1e12)) v(x)", id="union_mixed_case"),
    pytest.param("1=1) union (from (values (1e12)) v(x)", id="union_distinct"),
    pytest.param("1=1) except (from (values (1.0)) v(x)", id="except"),
    pytest.param("1=1) intersect (from (values (1.0)) v(x)", id="intersect"),
    pytest.param("1=1) group by amount having (count(*) > 0", id="group_by_having"),
    pytest.param("1=1) using sample reservoir(3 rows", id="using_sample"),
    pytest.param("1=1) order by 1 desc limit 1 offset (0", id="order_by_limit"),
    pytest.param("1=1) qualify (row_number() over () = 1", id="qualify"),
    pytest.param("amount in (from (values (1.0)))", id="from_first_subquery"),
    pytest.param("1=1) union all (from slice_src", id="union_self"),
]

_LEGITIMATE = [
    "amount > 100",
    "amount > 0",
    "region = 'North'",
    "region = '北区'",
    "amount between 1 and 5",
    "region in ('N','S')",
    "amount is not null",
    "lower(region) = 'n'",
    "amount > 0 and region = '北区'",
    "amount > 5 or (region = '北区' and amount is not null)",
    "not (amount > 5)",
    "((amount >= 1) and (amount <= 1))",
    "region like '%北%'",
    "cast(amount as integer) = 1",
]


@pytest.mark.parametrize("where_sql", _INJECTIONS)
def test_injected_where_bodies_are_rejected(where_sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_where_clause(where_sql)


@pytest.mark.parametrize("where_sql", _LEGITIMATE)
def test_legitimate_conditions_are_not_rejected(where_sql: str) -> None:
    assert validate_where_clause(where_sql) == where_sql.strip()


@pytest.mark.parametrize("where_sql", _LEGITIMATE)
def test_legitimate_conditions_still_profile(where_sql: str) -> None:
    profile = compute_slice_profile(
        _frame(),
        dataset_id="ds_guard",
        dataset_name="guard.csv",
        where_sql=where_sql,
        columns=["amount"],
    )
    assert profile.rows_total == _ROWS
    if profile.table is not None:
        assert profile.table.rows[0]["mean"] == pytest.approx(1.0)


def test_union_injection_cannot_poison_the_profile() -> None:
    with pytest.raises(UnsafeQueryError):
        compute_slice_profile(
            _frame(),
            dataset_id="ds_guard",
            dataset_name="guard.csv",
            where_sql=_UNION_INJECTION,
            columns=["amount"],
        )


def test_injection_leaves_no_receipt_whose_stats_contradict_its_row_count() -> None:
    context = _context()
    tool = next(t for t in build_data_tools(context) if t.name == "profile_slice")
    with pytest.raises(ValueError):
        tool.execute(
            ProfileSliceArguments(
                dataset_id="ds_guard",
                where_sql=_UNION_INJECTION,
                columns=["amount"],
            )
        )
    assert not [a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT]

    # Control: an honest slice does produce a verifiable receipt whose mean
    # matches the rows the row-count fact claims were profiled.
    tool.execute(
        ProfileSliceArguments(dataset_id="ds_guard", where_sql="amount > 0", columns=["amount"])
    )
    artifact = [a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT][-1]
    receipt = EvidenceReceipt.model_validate(artifact.payload)
    assert verify_receipt_digest(receipt)
    assert next(f for f in receipt.facts if f.fact_id == "rows_in_slice").value == _ROWS
    assert next(f for f in receipt.facts if f.fact_id == "amount.mean").value == pytest.approx(1.0)

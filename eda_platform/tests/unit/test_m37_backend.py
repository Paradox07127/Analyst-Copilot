from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import pytest
from pydantic import BaseModel

from eda_platform.agents.planner import build_plan
from eda_platform.core import query as query_module
from eda_platform.core.query import DuckDBQueryEngine
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    ColumnProfile,
    DatasetProfile,
    PiiColumn,
    PiiReport,
)
from eda_platform.schemas.plans import AnalysisPlan, Intent
from eda_platform.tools import pii as pii_tools
from eda_platform.tools.loader import load_csv
from eda_platform.tools.nl2sql_eval import NL2SQLEvalCase, run_nl2sql_eval_case
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.sql_runner import build_catalog
from eda_platform.tools.value_profile import top_n_values

T = TypeVar("T", bound=BaseModel)


class ScriptedStructuredLLM:
    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "schema": schema, "payload": payload})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return cast(T, response)


def test_dry_run_rejects_hallucinated_column_before_execution() -> None:
    engine = DuckDBQueryEngine()
    engine.register_frame("orders", pd.DataFrame({"amount": [10, 20]}))

    with pytest.raises(query_module.SqlBindingError, match="missing_amount"):
        engine.dry_run("select missing_amount from orders")


def test_build_plan_retries_once_with_previous_error_and_uses_corrected_plan() -> None:
    engine = DuckDBQueryEngine()
    engine.register_frame("orders", pd.DataFrame({"amount": [10, 20]}))
    llm = ScriptedStructuredLLM(
        [
            _analysis_plan(
                sql="select missing_amount from orders",
                columns=["amount"],
            ),
            _analysis_plan(sql="select sum(amount) as total_amount from orders"),
        ]
    )

    plan = build_plan(
        "total amount",
        llm=llm,
        catalog_columns={"orders": {"amount"}},
        engine=engine,
    )

    assert plan.sql == "select sum(amount) as total_amount from orders"
    assert len(llm.calls) == 2
    assert "previous_error" not in llm.calls[0]["payload"]
    assert "missing_amount" in llm.calls[1]["payload"]["previous_error"]


def test_build_plan_raises_value_error_after_retry_also_fails() -> None:
    engine = DuckDBQueryEngine()
    engine.register_frame("orders", pd.DataFrame({"amount": [10, 20]}))
    llm = ScriptedStructuredLLM(
        [
            _analysis_plan(sql="select missing_amount from orders", columns=["amount"]),
            _analysis_plan(sql="select still_missing from orders", columns=["amount"]),
        ]
    )

    with pytest.raises(ValueError, match="invalid SQL after retry"):
        build_plan(
            "total amount",
            llm=llm,
            catalog_columns={"orders": {"amount"}},
            engine=engine,
        )

    assert len(llm.calls) == 2
    assert "missing_amount" in llm.calls[1]["payload"]["previous_error"]


def test_relations_keep_dataset_id_entries_when_filenames_collide(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_path = left_dir / "orders.csv"
    right_path = right_dir / "orders.csv"
    left_path.write_text("amount\n10\n", encoding="utf-8")
    right_path.write_text("amount\n20\n", encoding="utf-8")

    catalog = build_catalog(
        [
            load_csv(left_path, dataset_id="ds_left"),
            load_csv(right_path, dataset_id="ds_right"),
        ]
    )

    assert catalog.relations["ds_left"] == "orders"
    assert catalog.relations["ds_right"] == "orders_2"
    assert catalog.relations["orders.csv"] == "orders"


def test_pii_labels_match_mask_value_behavior() -> None:
    pii_artifact = Artifact(
        id="pii",
        type=ArtifactType.PII_REPORT,
        project_id="project_demo",
        session_id="run_demo",
        payload=PiiReport(
            dataset_id="ds_customers",
            columns=[
                PiiColumn(column="email", label="email", reason="column-name"),
                PiiColumn(column="customer_name", label="name", reason="column-name"),
            ],
        ).model_dump(mode="json"),
    )

    labels = pii_tools.pii_labels(pii_artifact)

    assert labels == {"email": "email", "customer_name": "name"}
    assert pii_tools.mask_value("email", "a@example.com", pii_artifact) == "[PII:email]"
    assert pii_tools.mask_value("customer_name", "Alice", pii_artifact) == "[PII:name]"
    assert pii_tools.mask_value("region", "East", pii_artifact) == "East"


def test_top_n_values_validates_pii_report_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "email,region\n"
        "a@example.com,East\n"
        "b@example.com,East\n"
        "c@example.com,West\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_customers")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    pii = pii_tools.tag_pii_columns(profile, project_id="project_demo", session_id="run_demo")
    original_model_validate = PiiReport.model_validate
    validation_calls = 0

    def counting_model_validate(payload: object) -> PiiReport:
        nonlocal validation_calls
        validation_calls += 1
        return original_model_validate(payload)

    monkeypatch.setattr(PiiReport, "model_validate", staticmethod(counting_model_validate))

    value_profile = top_n_values(loaded, profile, pii, top_n=2)

    assert validation_calls == 1
    assert value_profile.values["email"] == [{"value": "[PII:email]", "count": 3}]
    assert value_profile.values["region"] == [
        {"value": "East", "count": 2},
        {"value": "West", "count": 1},
    ]


def test_phone_detection_requires_plus_or_separator() -> None:
    artifact = pii_tools.tag_pii_columns(
        _profile_artifact(
            {
                "numeric_identifier": ["10012345", "10012346", "10012347"],
                "short_digit_code": ["80824", "80825", "80826"],
                "intl_contact": [
                    "+86 138-0013-8000",
                    "+1 415-555-0134",
                    "+44 20 7946 0958",
                ],
                "paren_contact": [
                    "(415) 555-0134",
                    "(212) 555-0199",
                    "(650) 555-0177",
                ],
                "dashed_contact": [
                    "138-0013-8000",
                    "415-555-0134",
                    "212-555-0199",
                ],
            }
        ),
        project_id="project_demo",
        session_id="run_demo",
    )

    labels = _labels_by_column(artifact)
    assert "numeric_identifier" not in labels
    assert "short_digit_code" not in labels
    assert labels["intl_contact"] == "phone"
    assert labels["paren_contact"] == "phone"
    assert labels["dashed_contact"] == "phone"


def test_person_name_detection_uses_tokens_not_substrings() -> None:
    artifact = pii_tools.tag_pii_columns(
        _profile_artifact(
            {
                "product_name": ["Widget"],
                "filename": ["orders.csv"],
                "dataset_name": ["orders"],
                "column_name": ["amount"],
                "customer_name": ["Alice"],
                "user_name": ["Bob"],
                "name": ["Carol"],
            }
        ),
        project_id="project_demo",
        session_id="run_demo",
    )

    labels = _labels_by_column(artifact)
    assert "product_name" not in labels
    assert "filename" not in labels
    assert "dataset_name" not in labels
    assert "column_name" not in labels
    assert labels["customer_name"] == "name"
    assert labels["user_name"] == "name"
    assert labels["name"] == "name"


def test_nl2sql_eval_compares_shuffled_rows_as_pass(tmp_path: Path) -> None:
    outcome = _run_sales_eval(
        tmp_path,
        expected_rows_preview=[
            {"region": "West", "total_amount": 20.0},
            {"region": "East", "total_amount": 15},
        ],
    )

    assert outcome.passed is True
    assert outcome.actual_rows_preview == [
        {"region": "East", "total_amount": 15.0},
        {"region": "West", "total_amount": 20.0},
    ]


def test_nl2sql_eval_fails_when_rows_are_different(tmp_path: Path) -> None:
    outcome = _run_sales_eval(
        tmp_path,
        expected_rows_preview=[
            {"region": "West", "total_amount": 20.0},
            {"region": "East", "total_amount": 999.0},
        ],
    )

    assert outcome.passed is False


def _analysis_plan(
    *,
    sql: str,
    columns: list[str] | None = None,
) -> AnalysisPlan:
    return AnalysisPlan(
        question="total amount",
        dataset_names=["orders"],
        columns=columns or ["amount"],
        filters=[],
        sql=sql,
        method="aggregate",
        rationale="Aggregate amount.",
        needs_approval=False,
        estimated_scan="small",
    )


def _profile_artifact(samples_by_column: dict[str, list[str]]) -> Artifact:
    rows = max(len(samples) for samples in samples_by_column.values())
    profile = DatasetProfile(
        dataset_id="ds_customers",
        name="customers.csv",
        rows=rows,
        columns=len(samples_by_column),
        column_names=list(samples_by_column),
        dtypes={column: "object" for column in samples_by_column},
        missing_values={column: 0 for column in samples_by_column},
        missing_percent={column: 0.0 for column in samples_by_column},
        numeric_columns=[],
        categorical_columns=list(samples_by_column),
        columns_detail=[
            ColumnProfile(
                name=column,
                dtype="object",
                semantic_type="categorical",
                missing_count=0,
                missing_percent=0.0,
                unique_count=len(set(samples)),
                unique_percent=100.0,
                sample_values=samples,
            )
            for column, samples in samples_by_column.items()
        ],
    )
    return Artifact(
        id="profile",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload=profile.model_dump(mode="json"),
    )


def _labels_by_column(pii_artifact: Artifact) -> dict[str, str]:
    report = PiiReport.model_validate(pii_artifact.payload)
    return {column.column: column.label for column in report.columns}


def _run_sales_eval(
    tmp_path: Path,
    *,
    expected_rows_preview: list[dict[str, Any]],
):
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\n"
        "East,10\n"
        "West,20\n"
        "East,5\n",
        encoding="utf-8",
    )
    llm = ScriptedStructuredLLM(
        [
            Intent(
                kind="new_analysis",
                params={},
                confidence=0.92,
                raw_message="total sales by region",
            ),
            AnalysisPlan(
                question="total sales by region",
                dataset_names=["orders"],
                columns=["region", "amount"],
                filters=[],
                sql=(
                    "select region, sum(amount) as total_amount "
                    "from orders group by region order by region"
                ),
                method="grouped_aggregate",
                rationale="Aggregate sales by region.",
                needs_approval=False,
                estimated_scan="small",
            ),
        ]
    )
    return run_nl2sql_eval_case(
        NL2SQLEvalCase(
            name="sales_by_region",
            question="total sales by region",
            expected_rows_preview=expected_rows_preview,
        ),
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        llm=llm,
    )

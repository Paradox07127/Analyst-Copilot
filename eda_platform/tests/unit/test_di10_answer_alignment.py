"""DI10-W1: answer-question alignment + metric basis + domain-pack gate.

Live-review evidence (2026-07-18) this sprint fixes:

1. The finding builder blindly grabbed the first ``avg_``/numeric column, so
   "What is the total GMV (sum of Amount)?" was answered with
   ``avg_line_value`` and "What share of rows were late?" with the raw
   ``late_rows`` count. Answers must now follow the question's intent, and
   ``domain_metric`` questions consume their MetricDefinition interpretation
   template.
2. The late-delivery rate compared the CARRIER handoff against the promise
   (Olist: 0.48%) instead of the customer delivery date (true 8.11%).
3. creditcard.csv (a single "Amount" column) was dressed up with e-commerce
   "GMV" wording; the pack now needs >= 2 independent e-commerce signals.
"""

from __future__ import annotations

from pathlib import Path

from eda_platform.core.column_roles import ColumnRoleSet, infer_column_roles
from eda_platform.core.ids import make_artifact_id
from eda_platform.drivers.question_exec import (
    _findings_for,
    _successful_qexec_artifact,
    infer_question_intent,
    intent_metric_column,
)
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    SqlResult,
)
from eda_platform.schemas.questions import (
    QuestionAnswerContract,
    QuestionCandidate,
    QuestionScore,
)
from eda_platform.tools.domain_metrics import (
    applicable_metrics,
    ecommerce_signals,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sql_artifact(
    rows: list[dict[str, object]], *, units: dict[str, str] | None = None
) -> Artifact:
    columns = list(rows[0]) if rows else []
    payload = SqlResult(
        sql="select 1",
        columns=columns,
        dtypes=dict.fromkeys(columns, "DOUBLE"),
        units=units or {},
        rows_preview=rows,
        row_count=len(rows),
    ).model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("sql", payload),
        type=ArtifactType.SQL_RESULT,
        project_id="project_demo",
        session_id="run_demo",
        payload=payload,
    )


def _candidate(
    question_en: str,
    *,
    template_id: str | None = "domain_metric",
    referenced_columns: dict[str, list[str]] | None = None,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_demo",
        question_en=question_en,
        origin="template" if template_id is not None else "llm",
        template_id=template_id,
        target_datasets=["orders.csv"],
        referenced_columns=referenced_columns or {},
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    )


def _profile_with_roles(path: Path) -> tuple[DatasetProfile, ColumnRoleSet]:
    loaded = load_csv(path, dataset_id=f"ds_{path.stem}")
    artifact = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    profile = DatasetProfile.model_validate(artifact.payload)
    return profile, infer_column_roles(profile, frame=loaded.frame)


def _orders_csv(tmp_path: Path, *, with_customer_date: bool) -> Path:
    columns = [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date",
    ]
    if with_customer_date:
        columns.insert(4, "order_delivered_customer_date")
    rows = [",".join(columns)]
    for index in range(30):
        day = index % 27 + 1
        cells = [
            f"O{index:03d}",
            f"C{index:03d}",
            f"2026-01-{day:02d} 10:00:00",
            f"2026-01-{min(day + 1, 28):02d} 09:00:00",
            f"2026-01-{min(day + 5, 28):02d} 23:59:59",
        ]
        if with_customer_date:
            cells.insert(4, f"2026-01-{min(day + 3 + index % 4, 28):02d} 15:00:00")
        rows.append(",".join(cells))
    path = tmp_path / "orders.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _creditcard_csv(tmp_path: Path) -> Path:
    """The creditcard.csv shape: anonymized components + a lone Amount column."""
    rows = ["Time,V1,V2,V3,Amount,Class"]
    for index in range(40):
        rows.append(
            f"{index * 37},{0.1 * index:.3f},{-0.05 * index:.3f},"
            f"{0.02 * index:.3f},{5.0 + index * 2.7:.2f},{index % 2}"
        )
    path = tmp_path / "creditcard.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _order_items_csv(tmp_path: Path) -> Path:
    rows = ["order_id,order_item_id,price,freight_value,category"]
    categories = ["books", "toys", "garden", "sports"]
    for order_index in range(10):
        for item in range(1, 4):
            price = 20.0 + order_index * 3.7 + item * 1.3
            freight = 4.0 + order_index * 0.53 + item * 0.21
            rows.append(
                f"O{order_index:03d},{item},{price:.2f},{freight:.2f},"
                f"{categories[(order_index + item) % len(categories)]}"
            )
    path = tmp_path / "order_items.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 1. Intent inference: a pure, deterministic function of the question text
# --------------------------------------------------------------------------- #
def test_intent_inference_branches() -> None:
    assert infer_question_intent("What is the total GMV (sum of Amount)?") == "sum"
    assert infer_question_intent("What share of rows were late?") == "share"
    assert infer_question_intent("What is the late-delivery rate?") == "share"
    assert infer_question_intent("What is the average order value?") == "average"
    assert infer_question_intent("How long does fulfillment take?") == "duration"
    assert infer_question_intent("Which category sells best?") is None
    # Share outranks sum: a share-of-total question is a share question.
    assert infer_question_intent("What share of total value is freight?") == "share"


def test_intent_column_selection_is_deterministic() -> None:
    row: dict[str, object] = {
        "row_count": 284807,
        "gmv_total": 25162590.01,
        "avg_line_value": 88.3496,
    }
    assert intent_metric_column("sum", row) == "gmv_total"
    assert intent_metric_column("average", row) == "avg_line_value"
    assert intent_metric_column("share", row) is None
    late_row: dict[str, object] = {
        "row_count": 97658,
        "late_rows": 473,
        "late_delivery_rate_percent": 0.4843,
    }
    assert intent_metric_column("share", late_row) == "late_delivery_rate_percent"
    duration_row: dict[str, object] = {
        "row_count": 96476,
        "avg_fulfillment_days": 12.09,
        "median_fulfillment_days": 10.0,
    }
    assert intent_metric_column("duration", duration_row) == "avg_fulfillment_days"


# --------------------------------------------------------------------------- #
# 2. Sum questions answer the sum column (the live GMV regression)
# --------------------------------------------------------------------------- #
def test_sum_question_answers_sum_column_not_average() -> None:
    candidate = _candidate(
        "What is the total GMV (sum of Amount) in Creditcard?", template_id=None
    )
    artifact = _sql_artifact(
        [{"row_count": 284807, "gmv_total": 25162590.01, "avg_line_value": 88.3496}]
    )

    findings = _findings_for(candidate, artifact)

    assert len(findings) == 1
    text = findings[0].text
    assert "25162590.01" in text
    assert "88.3496" not in text
    assert "avg_line_value" not in text
    locators = {evidence.locator for evidence in findings[0].evidence}
    assert "rows_preview[0].gmv_total" in locators


# --------------------------------------------------------------------------- #
# 3. Share questions answer in percent with numerator/denominator
# --------------------------------------------------------------------------- #
def test_share_question_answers_percentage_with_fraction() -> None:
    candidate = _candidate(
        "What share of rows in Orders were late?", template_id=None
    )
    artifact = _sql_artifact(
        [{"row_count": 97658, "late_rows": 473, "late_delivery_rate_percent": 0.4843}]
    )

    findings = _findings_for(candidate, artifact)

    assert len(findings) == 1
    text = findings[0].text
    assert "0.4843%" in text
    assert "473 of 97658" in text
    # The raw count alone (the live regression) is no longer the answer.
    assert not text.rstrip(".").endswith("late_rows is 473")
    units = {
        evidence.locator: evidence.unit for evidence in findings[0].evidence
    }
    assert units["rows_preview[0].late_delivery_rate_percent"] == "percent"
    assert units["rows_preview[0].late_rows"] == "raw"
    assert units["rows_preview[0].row_count"] == "raw"


# --------------------------------------------------------------------------- #
# 4. domain_metric questions consume the MetricDefinition template
# --------------------------------------------------------------------------- #
def test_domain_metric_consumes_interpretation_template() -> None:
    candidate = _candidate("What is the total GMV (sum of Amount) in Creditcard?")
    artifact = _sql_artifact(
        [{"row_count": 284807, "gmv_total": 25162590.01, "avg_line_value": 88.3496}],
        units={"row_count": "count", "gmv_total": "currency"},
    )

    findings = _findings_for(candidate, artifact)

    assert len(findings) == 1
    assert "Total GMV over 284807 rows is 25162590.01." in findings[0].text
    assert "88.3496" not in findings[0].text
    # Every filled placeholder is a cited SQL cell.
    locators = {evidence.locator for evidence in findings[0].evidence}
    assert locators == {"rows_preview[0].row_count", "rows_preview[0].gmv_total"}
    units = {evidence.locator: evidence.unit for evidence in findings[0].evidence}
    assert units["rows_preview[0].row_count"] == "raw"
    assert units["rows_preview[0].gmv_total"] == "currency"


def test_specific_currency_unit_is_visible_and_keeps_versioned_provenance() -> None:
    candidate = _candidate("What is the total GMV in Orders?")
    artifact = _sql_artifact(
        [{"row_count": 10, "gmv_total": 100.0}],
        units={"row_count": "count", "gmv_total": "BRL"},
    )

    finding = _findings_for(candidate, artifact)[0]
    evidence = next(
        ref for ref in finding.evidence if ref.locator.endswith(".gmv_total")
    )

    assert "Total GMV over 10 rows is 100 BRL." in finding.text
    assert evidence.unit == "currency"
    assert evidence.unit_label == "BRL"
    assert evidence.unit_reference == "ISO 4217 List One@2026-01-01"


def test_historical_and_custom_currency_labels_remain_user_visible() -> None:
    candidate = _candidate("What is the average order value in Orders?")
    historical = _sql_artifact(
        [{"order_count": 4, "avg_order_value": 25.0}],
        units={"order_count": "count", "avg_order_value": "BGN/order"},
    )
    custom = _sql_artifact(
        [{"order_count": 4, "avg_order_value": 0.002}],
        units={"order_count": "count", "avg_order_value": "BTC/order"},
    )

    historical_finding = _findings_for(candidate, historical)[0]
    custom_finding = _findings_for(candidate, custom)[0]
    historical_evidence = next(
        ref
        for ref in historical_finding.evidence
        if ref.locator.endswith(".avg_order_value")
    )
    custom_evidence = next(
        ref
        for ref in custom_finding.evidence
        if ref.locator.endswith(".avg_order_value")
    )

    assert "25 BGN per order" in historical_finding.text
    assert historical_evidence.unit_reference == "ISO 4217 List Three@2026-01-01"
    assert "0.002 BTC per order" in custom_finding.text
    assert custom_evidence.unit_reference == (
        "semantic seed (unlisted in ISO 4217 snapshot@2026-01-01)"
    )


def test_domain_metric_late_rate_template_answers_in_percent() -> None:
    candidate = _candidate("What share of rows in Orders were late?")
    artifact = _sql_artifact(
        [{"row_count": 97658, "late_rows": 473, "late_delivery_rate_percent": 0.4843}]
    )

    findings = _findings_for(candidate, artifact)

    assert "473 of 97658 rows were late (0.4843%)." in findings[0].text
    percent_units = [
        evidence
        for evidence in findings[0].evidence
        if evidence.unit == "percent"
    ]
    assert [e.locator for e in percent_units] == [
        "rows_preview[0].late_delivery_rate_percent"
    ]


def test_invalid_hhi_result_fails_closed_before_interpretation() -> None:
    candidate = _candidate("How concentrated is V1 across Class groups (HHI)?")
    sql_artifact = _sql_artifact(
        [{"row_count": 2, "hhi": 2.0178324e26, "top_share": 1.0044482e13}]
    )

    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[sql_artifact.id],
        plan_summary="deterministic HHI",
    )

    assert qexec.payload["status"] == "failed"
    assert qexec.payload["outcome"] == "abstained"
    assert qexec.payload["abstention_code"] == "hhi_out_of_range"
    assert "Metric field 'hhi' is above 1" in qexec.payload["error"]
    assert qexec.payload["findings"] == []


def test_valid_metric_values_with_wrong_produced_unit_fail_closed() -> None:
    candidate = _candidate(
        "How concentrated is payment value across payment type groups (HHI)?"
    ).model_copy(
        update={
            "metric_id": "concentration_hhi",
            "answer_contract": QuestionAnswerContract(
                kind="metric",
                metric_id="concentration_hhi",
                expected_units={
                    "row_count": "count",
                    "hhi": "fraction",
                    "top_share": "fraction",
                },
            ),
            "produced_units": {
                "row_count": "count",
                "hhi": "percent",
                "top_share": "percent",
            },
        }
    )
    sql_artifact = _sql_artifact(
        [{"row_count": 4, "hhi": 0.6467, "top_share": 0.7834}],
        units=candidate.produced_units,
    )

    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[sql_artifact.id],
        plan_summary="deterministic HHI",
    )

    assert qexec.payload["outcome"] == "abstained"
    assert qexec.payload["abstention_code"] == "metric_unit_mismatch"
    assert "expected unit 'fraction'" in qexec.payload["error"]
    assert qexec.payload["findings"] == []


def test_new_unit_contract_without_produced_metadata_fails_closed() -> None:
    candidate = _candidate("What is the total GMV?").model_copy(
        update={
            "metric_id": "gmv",
            "answer_contract": QuestionAnswerContract(
                kind="metric",
                metric_id="gmv",
                expected_units={"row_count": "count", "gmv_total": "currency"},
            ),
        }
    )
    sql_artifact = _sql_artifact([{"row_count": 10, "gmv_total": 100.0}])

    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[sql_artifact.id],
        plan_summary="deterministic GMV",
    )

    assert qexec.payload["outcome"] == "abstained"
    assert qexec.payload["abstention_code"] == "metric_unit_metadata_missing"
    assert "no produced-unit metadata" in qexec.payload["error"]


def test_candidate_cannot_override_registry_expected_units() -> None:
    candidate = _candidate(
        "How concentrated is payment value across payment type groups (HHI)?"
    ).model_copy(
        update={
            "metric_id": "concentration_hhi",
            "answer_contract": QuestionAnswerContract(
                kind="metric",
                metric_id="concentration_hhi",
                expected_units={
                    "row_count": "count",
                    "hhi": "percent",
                    "top_share": "percent",
                },
            ),
            "produced_units": {
                "row_count": "count",
                "hhi": "percent",
                "top_share": "percent",
            },
        }
    )
    sql_artifact = _sql_artifact(
        [{"row_count": 4, "hhi": 0.6467, "top_share": 0.7834}],
        units=candidate.produced_units,
    )

    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[sql_artifact.id],
        plan_summary="tampered HHI contract",
    )

    assert qexec.payload["outcome"] == "abstained"
    assert qexec.payload["abstention_code"] == "metric_unit_contract_mismatch"
    assert "does not match the metric registry" in qexec.payload["error"]


def test_threshold_question_without_threshold_answer_fails_closed() -> None:
    candidate = _candidate(
        "Which threshold minimizes false-positive and false-negative cost?",
        template_id=None,
    )
    sql_artifact = _sql_artifact([{"row_count": 284807}])

    qexec = _successful_qexec_artifact(
        candidate,
        sql_artifact=sql_artifact,
        project_id="project_demo",
        session_id="run_demo",
        parent_ids=[sql_artifact.id],
        plan_summary="model-generated query",
    )

    assert qexec.payload["status"] == "failed"
    assert qexec.payload["outcome"] == "abstained"
    assert qexec.payload["abstention_code"] == "answer_schema_mismatch"
    assert "no result column matching its answer contract" in qexec.payload["error"]


def test_domain_metric_template_prefers_most_specific_match() -> None:
    # An AOV result also fully fills the GMV template (column subset); the
    # more specific AOV template must win.
    candidate = _candidate("What is the average order value in Orders?")
    artifact = _sql_artifact(
        [
            {
                "row_count": 112650,
                "order_count": 98666,
                "gmv_total": 13591643.7,
                "avg_order_value": 137.75,
            }
        ]
    )

    findings = _findings_for(candidate, artifact)

    assert "average order value is 137.75" in findings[0].text


def test_domain_metric_template_falls_back_to_intent_alignment() -> None:
    # A result row matching no registry template still gets an intent-aligned
    # answer instead of the blind avg_ grab.
    candidate = _candidate("What is the total revenue in Orders?")
    artifact = _sql_artifact(
        [{"row_count": 100, "revenue_total": 5000.5, "avg_revenue": 50.005}]
    )

    findings = _findings_for(candidate, artifact)

    assert "5000.5" in findings[0].text
    assert "50.005" not in findings[0].text


def test_domain_metric_carrier_fallback_disclosed_in_finding(tmp_path: Path) -> None:
    profile, role_set = _profile_with_roles(
        _orders_csv(tmp_path, with_customer_date=False)
    )
    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )
    late = next(m for m in resolution.resolved if m.metric_id == "late_delivery_rate")

    candidate = _candidate(
        late.question_en, referenced_columns=late.referenced_columns
    )
    artifact = _sql_artifact(
        [{"row_count": 30, "late_rows": 2, "late_delivery_rate_percent": 6.6667}]
    )

    findings = _findings_for(candidate, artifact)

    assert "late to carrier handoff, customer delivery date unavailable" in (
        findings[0].text
    )


# --------------------------------------------------------------------------- #
# 5. Late-delivery basis: customer delivery beats carrier handoff
# --------------------------------------------------------------------------- #
def test_late_rate_prefers_customer_delivery_column(tmp_path: Path) -> None:
    profile, role_set = _profile_with_roles(
        _orders_csv(tmp_path, with_customer_date=True)
    )

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )

    late = next(m for m in resolution.resolved if m.metric_id == "late_delivery_rate")
    assert late.referenced_columns == {
        "orders.csv": [
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
        ]
    }
    assert "order_delivered_customer_date" in late.sql
    fulfillment = next(
        m for m in resolution.resolved if m.metric_id == "fulfillment_time"
    )
    assert "order_delivered_customer_date" in fulfillment.sql
    assert "order_delivered_carrier_date" not in fulfillment.sql
    # Clean basis -> no degradation caveat anywhere.
    assert "carrier handoff" not in late.interpretation_en
    assert "carrier handoff" not in fulfillment.interpretation_en


def test_late_rate_degrades_to_carrier_with_disclosure(tmp_path: Path) -> None:
    profile, role_set = _profile_with_roles(
        _orders_csv(tmp_path, with_customer_date=False)
    )

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )

    late = next(m for m in resolution.resolved if m.metric_id == "late_delivery_rate")
    assert late.referenced_columns["orders.csv"][1] == "order_delivered_carrier_date"
    assert (
        "late to carrier handoff, customer delivery date unavailable"
        in late.interpretation_en
    )
    assert "customer delivery date unavailable" in late.question_en
    fulfillment = next(
        m for m in resolution.resolved if m.metric_id == "fulfillment_time"
    )
    assert "order_delivered_carrier_date" in fulfillment.sql
    assert "customer delivery date unavailable" in fulfillment.interpretation_en


# --------------------------------------------------------------------------- #
# 6. E-commerce pack gate: >= 2 independent signals required
# --------------------------------------------------------------------------- #
def test_creditcard_shape_rejected_by_ecommerce_gate(tmp_path: Path) -> None:
    profile, role_set = _profile_with_roles(_creditcard_csv(tmp_path))

    # A lone Amount column is at most one signal.
    assert len(ecommerce_signals(profile, role_set)) < 2

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )

    resolved_domains = {metric.domain for metric in resolution.resolved}
    assert "ecommerce" not in resolved_domains
    skipped = {skip.metric_id: skip for skip in resolution.skipped}
    for metric_id in (
        "gmv",
        "aov",
        "repeat_purchase_rate",
        "late_delivery_rate",
        "fulfillment_time",
        "freight_ratio",
    ):
        assert skipped[metric_id].reason == "domain_signals_insufficient", metric_id
    # No e-commerce vocabulary can reach the report copy.
    for metric in resolution.resolved:
        for text in (metric.question_en, metric.interpretation_en):
            assert "GMV" not in text
            assert "order value" not in text
    # Generic HHI may still apply, but never to signed anonymized PCA components
    # or elapsed Time; the positive additive Amount column is the safe binding.
    hhi = next(metric for metric in resolution.resolved if metric.metric_id == "concentration_hhi")
    hhi_columns = next(iter(hhi.referenced_columns.values()))
    assert hhi_columns[-1] == "Amount"


def test_olist_shape_passes_ecommerce_gate(tmp_path: Path) -> None:
    items_profile, items_roles = _profile_with_roles(_order_items_csv(tmp_path))
    orders_profile, orders_roles = _profile_with_roles(
        _orders_csv(tmp_path, with_customer_date=True)
    )

    # price+freight measures, order key, and the table name all signal.
    assert len(ecommerce_signals(items_profile, items_roles)) >= 2
    assert len(ecommerce_signals(orders_profile, orders_roles)) >= 2

    resolution = applicable_metrics(
        role_sets={
            items_profile.name: items_roles,
            orders_profile.name: orders_roles,
        },
        join_whitelist=None,
        profiles=[items_profile, orders_profile],
    )

    resolved = {metric.metric_id for metric in resolution.resolved}
    assert {"gmv", "aov", "late_delivery_rate", "fulfillment_time", "freight_ratio"} <= (
        resolved
    )

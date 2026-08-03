"""Independent durable-result contracts for release-grade data-tool receipts.

The receipt is never accepted as its own evidence source.  These contracts
derive its facts, statistical fields, output digest and artifact references
from the separately persisted primary artifact and provider-facing result
content.  Production adapters invoke the same contract before returning, and
the E4a issuer invokes it again over the durable executor body.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eda_platform.agents.runtime import AgentToolResult, canonical_json_sha256
from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.receipts import EvidenceReceipt, ReceiptFact
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.missingness import MissingnessDiagnosticResult

_MAX_FACT_COLUMNS = 8
_MAX_FACT_PAIRS = 5
_MAX_MANIFEST_ENTRIES = 512


def verify_data_tool_result_contract(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    canonical_arguments: Mapping[str, object],
) -> None:
    """Fail closed unless durable non-receipt output reconstructs the receipt."""
    primary = _primary_artifact(receipt, result)
    if receipt.tool_name == "run_stat_test":
        _verify_stat_test(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "diagnose_missingness":
        _verify_missingness(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "analyze_time_series":
        _verify_time_series(receipt, result, primary, canonical_arguments)
    else:
        raise ValueError(
            f"release evidence has no durable result contract for {receipt.tool_name!r}"
        )


def _primary_artifact(receipt: EvidenceReceipt, result: AgentToolResult) -> Artifact:
    if len(result.artifacts) != 1 or not isinstance(result.artifacts[0], Artifact):
        raise ValueError("release-grade tool result requires one primary artifact")
    primary = result.artifacts[0]
    prefix = "stat" if primary.type is ArtifactType.STAT_TEST_RESULT else "table"
    if (
        primary.id != make_artifact_id(prefix, primary.payload)
        or receipt.artifact_ids != (primary.id,)
        or result.receipt_artifact is None
        or result.receipt_artifact.id
        != make_artifact_id("receipt", receipt.model_dump(mode="json"))
        or result.receipt_artifact.parents != [primary.id]
    ):
        raise ValueError("receipt artifact binding is not content addressed")
    content = _content(result)
    if content.get("artifact_id") != primary.id or content.get("receipt_id") != receipt.receipt_id:
        raise ValueError("durable result content does not select its receipt and artifact")
    return primary


def _verify_stat_test(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    if primary.type is not ArtifactType.STAT_TEST_RESULT:
        raise ValueError("run_stat_test primary artifact has the wrong type")
    stat = StatTestResult.model_validate(primary.payload)
    facts = tuple(
        fact
        for fact in (
            _fact("p_value", stat.p_value, "number"),
            _fact("statistic", stat.statistic, "number"),
            _fact("effect_size", stat.effect_size, "number")
            if stat.effect_size is not None
            else None,
            _fact("sample_size", stat.sample_size, "count"),
        )
        if fact is not None
    )
    statistics = receipt.statistics
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        "test_type": stat.test_type,
        "requested_test_type": arguments.get("test_type"),
        "p_value": stat.p_value,
        "adjusted_p_value": stat.adjusted_p_value,
        "effect_size": stat.effect_size,
        "effect_ci": [stat.effect_ci_low, stat.effect_ci_high],
        "sample_size": stat.sample_size,
        "assumptions": [f"{item.name}={item.status}" for item in stat.assumptions],
        "warnings": list(receipt.method.warnings),
    }
    if (
        receipt.facts != facts
        or receipt.output_digest != canonical_json_sha256(stat.model_dump(mode="json"))
        or receipt.result_count != 1
        or statistics is None
        or statistics.test_name != stat.test_type
        or statistics.test_statistic != stat.statistic
        or statistics.p_value != stat.p_value
        or statistics.adjusted_p_value != stat.adjusted_p_value
        or statistics.effect_size != stat.effect_size
        or statistics.ci_low != stat.effect_ci_low
        or statistics.ci_high != stat.effect_ci_high
        or statistics.sample_size != stat.sample_size
        or receipt.method.family != stat.test_type
        or receipt.method.assumptions != tuple(expected_content["assumptions"])
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns != _stat_columns(arguments)
        or _content(result) != expected_content
    ):
        raise ValueError("run_stat_test receipt is not reconstructed by its durable result")


def _verify_time_series(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    if primary.type is not ArtifactType.TABLE:
        raise ValueError("analyze_time_series primary artifact has the wrong type")
    metrics = _metric_rows(primary)

    def numeric_fact(name: str) -> ReceiptFact:
        value = metrics[name]
        return _fact(name, value, "null" if value is None else "number")

    facts = (
        _fact("n_periods", metrics["n_periods"], "count"),
        _fact("gap_count", metrics["gap_count"], "count"),
        _fact("regular_frequency", metrics["regular_frequency"], "string"),
        _fact("trend_direction", metrics["trend_direction"], "string"),
        _fact("spike_detected", metrics["spike_detected"], "bool"),
        _fact("spike_period", metrics["spike_period"], "string"),
        numeric_fact("spike_value"),
        numeric_fact("spike_score"),
        numeric_fact("seasonal_strength"),
        numeric_fact("ljung_box_p"),
        numeric_fact("adf_p"),
        numeric_fact("kpss_p"),
        _fact("stationarity_verdict", metrics["stationarity_verdict"], "string"),
    )
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        **{
            key: metrics[key]
            for key in (
                "n_periods",
                "gap_count",
                "regular_frequency",
                "trend_direction",
                "spike_detected",
                "spike_period",
                "spike_value",
                "spike_score",
                "seasonal_strength",
                "ljung_box_p",
                "adf_p",
                "kpss_p",
                "stationarity_verdict",
            )
        },
        "warnings": list(primary.warnings),
    }
    statistics = receipt.statistics
    expected_parameters = {
        "agg": arguments.get("agg"),
        "freq": metrics["regular_frequency"],
        "period": metrics["period"],
        "decomposition_performed": metrics["decomposition_performed"],
        "log_transformed": metrics["log_transformed"],
        "ljung_box_lag": metrics["ljung_box_lag"],
    }
    if (
        receipt.facts != facts
        or receipt.output_digest != canonical_json_sha256(primary.payload)
        or receipt.result_count != metrics["n_periods"]
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns
        != (arguments.get("time_column"), arguments.get("value_column"))
        or receipt.scope.time_range != metrics["time_range"]
        or receipt.method.family != "time_series_diagnostics"
        or dict(receipt.method.parameters) != expected_parameters
        or receipt.method.warnings != tuple(primary.warnings)
        or statistics is None
        or statistics.test_name != "ljung_box"
        or statistics.p_value != metrics["ljung_box_p"]
        or statistics.sample_size != metrics["n_periods"]
        or _content(result) != expected_content
    ):
        raise ValueError("time-series receipt is not reconstructed by its durable result")


def _verify_missingness(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    if primary.type is not ArtifactType.TABLE:
        raise ValueError("diagnose_missingness primary artifact has the wrong type")
    payload = primary.payload
    rows = _rows(payload)
    missing_percent = {str(row["column"]): row["missing_percent"] for row in rows}
    raw_result = {
        "dataset_id": payload.get("dataset_id"),
        "rows_total": payload.get("rows_total"),
        "columns_analyzed": payload.get("columns_analyzed"),
        "columns_with_missing": payload.get("columns_with_missing"),
        "missing_percent": missing_percent,
        "group_columns": payload.get("group_columns"),
        "target_column": payload.get("target_column"),
        "indicator_correlations": payload.get("indicator_correlations"),
        "group_rate_ranges": payload.get("group_rate_ranges"),
        "target_associations": payload.get("target_associations"),
        "mnar_ruled_out": payload.get("mnar_ruled_out"),
        "limitations": payload.get("limitations"),
    }
    table = {
        key: payload[key] for key in ("dataset_id", "title", "description", "kind", "rows")
    }
    diagnostic = MissingnessDiagnosticResult.model_validate({**raw_result, "table": table})
    facts: list[ReceiptFact] = [
        _fact("rows_total", diagnostic.rows_total, "count"),
        _fact("columns_analyzed", diagnostic.columns_analyzed, "count"),
        _fact("columns_with_missing", diagnostic.columns_with_missing, "count"),
        _fact("mnar_ruled_out", diagnostic.mnar_ruled_out, "bool"),
        _fact("group_columns_analyzed", len(diagnostic.group_columns), "count"),
        _fact("target_associations_tested", len(diagnostic.target_associations), "count"),
    ]
    for index, row in enumerate(rows[:_MAX_FACT_COLUMNS]):
        facts.extend(
            (
                _fact(f"column{index}.name", str(row["column"]), "string"),
                _fact(f"column{index}.missing_count", row["missing_count"], "count"),
                _fact(
                    f"column{index}.missing_percent",
                    row["missing_percent"],
                    "percent",
                    unit="percent",
                ),
            )
        )
    for index, item in enumerate(diagnostic.indicator_correlations[:_MAX_FACT_PAIRS]):
        facts.extend(
            (
                _fact(
                    f"indicator_pair{index}.columns",
                    f"{item.column_a}~{item.column_b}",
                    "string",
                ),
                _fact(f"indicator_pair{index}.phi", item.phi, "number"),
            )
        )
    for index, item in enumerate(diagnostic.group_rate_ranges[:_MAX_FACT_PAIRS]):
        facts.extend(
            (
                _fact(
                    f"group_range{index}.columns",
                    f"{item.missing_column}~{item.group_column}",
                    "string",
                ),
                _fact(
                    f"group_range{index}.percentage_points",
                    item.range_percentage_points,
                    "percent",
                    unit="percentage_points",
                ),
            )
        )
    for index, item in enumerate(diagnostic.target_associations[:_MAX_FACT_PAIRS]):
        facts.extend(
            (
                _fact(
                    f"target_association{index}.missing_column",
                    item.missing_column,
                    "string",
                ),
                _fact(
                    f"target_association{index}.adjusted_p",
                    item.adjusted_p_value,
                    "number",
                ),
                _fact(
                    f"target_association{index}.effect_size", item.effect_size, "number"
                ),
            )
        )
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        "rows_total": diagnostic.rows_total,
        "columns_analyzed": diagnostic.columns_analyzed,
        "columns_with_missing": diagnostic.columns_with_missing,
        "missing_percent": diagnostic.missing_percent,
        "group_columns": diagnostic.group_columns,
        "indicator_correlations": [
            item.model_dump(mode="json") for item in diagnostic.indicator_correlations
        ],
        "group_rate_ranges": [
            item.model_dump(mode="json") for item in diagnostic.group_rate_ranges
        ],
        "target_associations": [
            item.model_dump(mode="json") for item in diagnostic.target_associations
        ],
        "mnar_ruled_out": diagnostic.mnar_ruled_out,
        "limitations": diagnostic.limitations,
    }
    expected_parameters = {
        "target_column": arguments.get("target_column"),
        "group_column_count": len(diagnostic.group_columns),
        "target_correction": "holm",
        "top_k": arguments.get("top_k"),
        "mnar_ruled_out": False,
    }
    expected_raw = diagnostic.model_dump(mode="json", exclude={"table"})
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest != canonical_json_sha256(expected_raw)
        or receipt.result_count != len(rows)
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns != tuple(str(row["column"]) for row in rows)
        or receipt.scope.scope_resolution != "resolved"
        or receipt.method.family != "missingness_diagnostic"
        or dict(receipt.method.parameters) != expected_parameters
        or receipt.method.assumptions
        != ("MNAR is not identifiable from observed data alone.",)
        or receipt.method.warnings != tuple(diagnostic.limitations)
        or _content(result) != expected_content
    ):
        raise ValueError("missingness receipt is not reconstructed by its durable result")
    _verify_fact_manifest(receipt, rows)


def _verify_fact_manifest(receipt: EvidenceReceipt, rows: list[dict[str, Any]]) -> None:
    manifest = receipt.fact_manifest
    listed = rows[:_MAX_MANIFEST_ENTRIES]
    if (
        manifest is None
        or manifest.total_rows != len(rows)
        or manifest.unlisted_rows != len(rows) - len(listed)
    ):
        raise ValueError("missingness receipt fact manifest is incomplete")
    if len(manifest.entries) != len(listed):
        raise ValueError("missingness receipt fact manifest does not cover artifact rows")
    for index, (entry, row) in enumerate(zip(manifest.entries, listed, strict=True)):
        expected_status = "evaluated" if index < _MAX_FACT_COLUMNS else "unevaluated"
        if (
            entry.fact_id != f"column{index}"
            or entry.row_index != index
            or entry.status != expected_status
            or entry.row_digest != canonical_json_sha256(row)
        ):
            raise ValueError("missingness receipt manifest diverges from artifact rows")


def _content(result: AgentToolResult) -> dict[str, Any]:
    if not isinstance(result.content, dict):
        raise ValueError("release-grade durable result content must be structured")
    return result.content


def _metric_rows(primary: Artifact) -> dict[str, object]:
    return {
        str(row["metric"]): row.get("value")
        for row in _rows(primary.payload)
        if isinstance(row.get("metric"), str)
    }


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("primary artifact rows are malformed")
    return rows


def _stat_columns(arguments: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        arguments[key]
        for key in ("group_column", "value_column", "category_column", "pair_column")
        if arguments.get(key) is not None
    )


def _fact(
    fact_id: str,
    value: object,
    value_type: str,
    *,
    unit: str | None = None,
) -> ReceiptFact:
    return ReceiptFact.model_validate(
        {
            "fact_id": fact_id,
            "name": fact_id,
            "value": value,
            "value_type": value_type,
            "unit": unit,
        }
    )

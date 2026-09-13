"""Independent durable-result contracts for release-grade data-tool receipts.

The receipt is never accepted as its own evidence source.  These contracts
derive its facts, statistical fields, output digest and artifact references
from the separately persisted primary artifact and provider-facing result
content.  Production adapters invoke the same contract before returning, and
the E4a issuer invokes it again over the durable executor body.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from eda_platform.agents.runtime import AgentToolResult, canonical_json_sha256
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.stat_registry import derive_family_id
from eda_platform.schemas.anomaly import AnomalyScreenResult
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.model_card import ModelCard
from eda_platform.schemas.receipts import EvidenceReceipt, ReceiptFact
from eda_platform.schemas.segmentation import SegmentationResult
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.causal_experiment import (
    BALANCE_SMD_THRESHOLD as CAUSAL_BALANCE_SMD_THRESHOLD,
)
from eda_platform.tools.causal_experiment import (
    OBSERVATIONAL_CONTRAST_DISCLAIMER,
    OBSERVATIONAL_DESIGN_FAMILY,
    RANDOMIZED_DESIGN_ASSUMPTION,
    RANDOMIZED_EXPERIMENT_FAMILY,
    observational_ci_method,
    randomized_design_credential_id,
)
from eda_platform.tools.forecast import FORECAST_LIMITATION
from eda_platform.tools.missingness import MissingnessDiagnosticResult
from eda_platform.tools.segmentation import (
    ARI_STABLE_FROM,
    ARI_UNSTABLE_BELOW,
    SEGMENTATION_LIMITATION,
)

# Each tool mints its primary artifact id under its own prefix; the contract
# must recompute the same one or a legitimate receipt reads as unaddressed.
_ARTIFACT_ID_PREFIX = {
    ArtifactType.STAT_TEST_RESULT: "stat",
    ArtifactType.ANOMALY_SCREEN_RESULT: "anomaly",
    ArtifactType.MODEL_CARD: "model",
    ArtifactType.SEGMENTATION_RESULT: "segmentation",
}

_MAX_FACT_COLUMNS = 8
_MAX_FACT_PAIRS = 5
_MAX_MANIFEST_ENTRIES = 512


def verify_data_tool_result_contract(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    canonical_arguments: Mapping[str, object],
) -> None:
    """Fail closed unless durable non-receipt output reconstructs the receipt."""
    if receipt.tool_name == "run_causal_experiment":
        # Two durable artifacts (stat result + balance table) on one receipt,
        # so it carries its own binding check instead of _primary_artifact.
        _verify_causal(receipt, result, canonical_arguments)
        return
    primary = _primary_artifact(receipt, result)
    if receipt.tool_name == "run_stat_test":
        _verify_stat_test(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "diagnose_missingness":
        _verify_missingness(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "analyze_time_series":
        _verify_time_series(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "run_forecast":
        _verify_forecast(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "run_segmentation":
        _verify_segmentation(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "correlate_columns":
        _verify_correlate_columns(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "screen_anomalies":
        _verify_screen_anomalies(receipt, result, primary, canonical_arguments)
    elif receipt.tool_name == "run_baseline_model":
        _verify_baseline_model(receipt, result, primary, canonical_arguments)
    else:
        raise ValueError(
            f"release evidence has no durable result contract for {receipt.tool_name!r}"
        )


def _primary_artifact(receipt: EvidenceReceipt, result: AgentToolResult) -> Artifact:
    if len(result.artifacts) != 1 or not isinstance(result.artifacts[0], Artifact):
        raise ValueError("release-grade tool result requires one primary artifact")
    primary = result.artifacts[0]
    prefix = _ARTIFACT_ID_PREFIX.get(primary.type, "table")
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


def _verify_forecast(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild backtest and projection facts from the published table rows."""
    if primary.type is not ArtifactType.TABLE:
        raise ValueError("run_forecast primary artifact has the wrong type")
    rows = _rows(primary.payload)
    baseline_rows = [row for row in rows if "baseline" in row]
    projection_rows = [row for row in rows if "period" in row]
    if not baseline_rows or not projection_rows:
        raise ValueError("run_forecast table must publish baseline and projection rows")
    content = _content(result)
    chosen = next(
        (row for row in baseline_rows if row["baseline"] == content.get("chosen_baseline")),
        None,
    )
    if chosen is None:
        raise ValueError("run_forecast chosen baseline is not among the published rows")
    horizon = len(projection_rows)
    mase = chosen["mase"]
    facts: list[ReceiptFact] = [
        _fact("chosen_baseline", chosen["baseline"], "string"),
        _fact("horizon", horizon, "count"),
        _fact("n_periods", content.get("n_periods"), "count"),
        _fact("gap_count", content.get("gap_count"), "count"),
        _fact("backtest_origins", chosen["origins"], "count"),
        _fact("backtest_mae", chosen["backtest_mae"], "number"),
        _fact("backtest_rmse", chosen["backtest_rmse"], "number"),
        _fact("mase", mase, "null" if mase is None else "number"),
        _fact("projection_start", projection_rows[0]["period"], "string"),
        _fact("projection_end", projection_rows[-1]["period"], "string"),
    ]
    facts.extend(
        _fact(f"projection{index}.value", row["projected_value"], "number")
        for index, row in enumerate(projection_rows[:_MAX_FACT_COLUMNS])
    )
    parameters = dict(receipt.method.parameters)
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        "chosen_baseline": chosen["baseline"],
        "horizon": horizon,
        "n_periods": content.get("n_periods"),
        "gap_count": content.get("gap_count"),
        "backtest_origins": chosen["origins"],
        "backtest_mae": chosen["backtest_mae"],
        "backtest_rmse": chosen["backtest_rmse"],
        "mase": mase,
        "projection_start": projection_rows[0]["period"],
        "projection_end": projection_rows[-1]["period"],
        "baselines": baseline_rows,
        "projections": projection_rows,
        "limitation": FORECAST_LIMITATION,
        "warnings": list(primary.warnings),
    }
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest != canonical_json_sha256(primary.payload)
        or receipt.result_count != horizon
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns
        != (arguments.get("time_column"), arguments.get("value_column"))
        or receipt.method.family != "forecast_baseline"
        # freq/period are grid metadata with no durable duplicate to check
        # against; every number a claim can cite is verified above.
        or set(parameters)
        != {"agg", "freq", "period", "horizon", "chosen_baseline", "backtest_scheme"}
        or parameters["agg"] != arguments.get("agg")
        or parameters["horizon"] != horizon
        or parameters["chosen_baseline"] != chosen["baseline"]
        or parameters["backtest_scheme"] != "rolling_origin"
        or receipt.method.assumptions != (FORECAST_LIMITATION,)
        or receipt.method.warnings != tuple(primary.warnings)
        or _clipped(content) != _clipped(expected_content)
    ):
        raise ValueError("run_forecast receipt is not reconstructed by its durable result")
    if receipt.statistics is not None and receipt.statistics.p_value is not None:
        # A descriptive projection runs no hypothesis test; a p-value here
        # would be an unregistered test.
        raise ValueError("run_forecast receipt must not carry a test p-value")
    manifest = receipt.fact_manifest
    if (
        manifest is None
        or manifest.total_rows != len(projection_rows)
        or manifest.unlisted_rows
        != len(projection_rows) - len(projection_rows[:_MAX_MANIFEST_ENTRIES])
        or len(manifest.entries) != len(projection_rows[:_MAX_MANIFEST_ENTRIES])
    ):
        raise ValueError("run_forecast receipt fact manifest is incomplete")
    for index, (entry, row) in enumerate(
        zip(manifest.entries, projection_rows[:_MAX_MANIFEST_ENTRIES], strict=True)
    ):
        expected_status = "evaluated" if index < _MAX_FACT_COLUMNS else "unevaluated"
        if (
            entry.fact_id != f"projection{index}"
            or entry.row_index != index
            or entry.status != expected_status
            or entry.row_digest != canonical_json_sha256(row)
        ):
            raise ValueError("run_forecast receipt manifest diverges from projection rows")


def _verify_segmentation(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild segment facts, stability grading and the evidence verdict."""
    if primary.type is not ArtifactType.SEGMENTATION_RESULT:
        raise ValueError("run_segmentation primary artifact has the wrong type")
    segmentation = SegmentationResult.model_validate(primary.payload)
    facts: list[ReceiptFact] = [
        _fact("k", segmentation.k, "count"),
        _fact("silhouette", segmentation.silhouette, "number"),
        _fact("ari_mean", segmentation.ari_mean, "number"),
        _fact("ari_min", segmentation.ari_min, "number"),
        _fact("stability", segmentation.stability, "string"),
        _fact("sample_rows", segmentation.sample_rows, "count"),
    ]
    for index, cluster in enumerate(segmentation.clusters[:_MAX_FACT_COLUMNS]):
        facts.extend(
            (
                _fact(f"cluster{index}.size", cluster.size, "count"),
                _fact(f"cluster{index}.share", cluster.share, "number"),
            )
        )
    expected_parameters: dict[str, object] = {
        "k": segmentation.k,
        "k_selection": segmentation.k_selection,
        "n_init": 10,
        "resamples": segmentation.resamples,
        "ari_unstable_below": ARI_UNSTABLE_BELOW,
        "ari_stable_from": ARI_STABLE_FROM,
        "standardization": "zscore",
        "random_state": arguments.get("random_state"),
    }
    if segmentation.stability == "unstable":
        expected_parameters["hypothesis_evidence_valid"] = False
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        **segmentation.model_dump(mode="json"),
        "limitation": SEGMENTATION_LIMITATION,
    }
    # The stability label must be the one these numbers imply, and an unstable
    # partition must carry the producer-owned invalid-evidence verdict.
    implied_stability = (
        "unstable"
        if segmentation.ari_mean < ARI_UNSTABLE_BELOW
        else "moderate"
        if segmentation.ari_mean < ARI_STABLE_FROM
        else "stable"
    )
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest != canonical_json_sha256(primary.payload)
        or receipt.result_count != segmentation.k
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or list(receipt.scope.columns) != list(segmentation.feature_columns)
        or receipt.method.family != "segmentation_kmeans"
        or dict(receipt.method.parameters) != expected_parameters
        or receipt.method.assumptions != (SEGMENTATION_LIMITATION,)
        or receipt.method.warnings != tuple(segmentation.notes)
        or segmentation.stability != implied_stability
        or _clipped(_content(result)) != _clipped(expected_content)
    ):
        raise ValueError(
            "run_segmentation receipt is not reconstructed by its durable result"
        )
    if receipt.statistics is not None and receipt.statistics.p_value is not None:
        # Silhouette and ARI are not hypothesis tests; a p-value here would be
        # an unregistered test.
        raise ValueError("run_segmentation receipt must not carry a test p-value")
    cluster_rows = _rows_from(primary.payload, "clusters")
    manifest = receipt.fact_manifest
    if (
        manifest is None
        or manifest.total_rows != len(cluster_rows)
        or manifest.unlisted_rows != 0
        or len(manifest.entries) != len(cluster_rows)
    ):
        raise ValueError("run_segmentation receipt fact manifest is incomplete")
    for index, (entry, row) in enumerate(
        zip(manifest.entries, cluster_rows, strict=True)
    ):
        expected_status = "evaluated" if index < _MAX_FACT_COLUMNS else "unevaluated"
        if (
            entry.fact_id != f"cluster{index}"
            or entry.row_index != index
            or entry.status != expected_status
            or entry.row_digest != canonical_json_sha256(row)
        ):
            raise ValueError(
                "run_segmentation receipt manifest diverges from cluster rows"
            )


def _verify_causal(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild both causal tiers from the stat and balance artifacts, and bind
    the randomized tier to a re-derived design-confirmation credential id."""
    if len(result.artifacts) != 2 or not all(
        isinstance(artifact, Artifact) for artifact in result.artifacts
    ):
        raise ValueError(
            "run_causal_experiment requires a stat artifact and a balance table"
        )
    primary, balance = result.artifacts
    if primary.type is not ArtifactType.STAT_TEST_RESULT:
        raise ValueError("run_causal_experiment primary artifact has the wrong type")
    if balance.type is not ArtifactType.TABLE:
        raise ValueError("run_causal_experiment balance artifact has the wrong type")
    if (
        primary.id != make_artifact_id("stat", primary.payload)
        or balance.id != make_artifact_id("table", balance.payload)
        or receipt.artifact_ids != (primary.id, balance.id)
        or result.receipt_artifact is None
        or result.receipt_artifact.id
        != make_artifact_id("receipt", receipt.model_dump(mode="json"))
        or result.receipt_artifact.parents != [primary.id, balance.id]
    ):
        raise ValueError("receipt artifact binding is not content addressed")
    content = _content(result)
    if (
        content.get("artifact_id") != primary.id
        or content.get("balance_artifact_id") != balance.id
        or content.get("receipt_id") != receipt.receipt_id
    ):
        raise ValueError(
            "durable result content does not select its receipt and artifacts"
        )
    stat = StatTestResult.model_validate(primary.payload)
    balance_rows = _rows(balance.payload)
    parameters = dict(receipt.method.parameters)
    design = parameters.get("experiment_design")
    randomized = design == "randomized"
    if not randomized and design != "observational":
        raise ValueError("run_causal_experiment receipt has an unknown design tier")
    if randomized != (stat.test_type == "two_sample_ate"):
        raise ValueError(
            "run_causal_experiment test type does not match its design tier"
        )
    expected_family = (
        RANDOMIZED_EXPERIMENT_FAMILY if randomized else OBSERVATIONAL_DESIGN_FAMILY
    )
    if randomized:
        # The whitelist anchor: the credential id must re-derive from this
        # receipt's dataset and treatment column, never be taken on faith.
        expected_credential = randomized_design_credential_id(
            dataset_id=str(arguments.get("dataset_id")),
            treatment_column=str(arguments.get("treatment_column")),
        )
        if parameters.get("design_confirmation_id") != expected_credential:
            raise ValueError(
                "randomized-tier receipt carries no matching design credential"
            )
    elif "design_confirmation_id" in parameters:
        raise ValueError(
            "observational-tier receipt must not carry a design credential"
        )

    facts: list[ReceiptFact] = [
        _fact("p_value", stat.p_value, "number"),
        _fact("statistic", stat.statistic, "number"),
    ]
    if stat.effect_size is not None:
        facts.append(_fact("effect_size", stat.effect_size, "number"))
    imbalanced = [
        str(row["covariate"]) for row in balance_rows if row.get("imbalanced")
    ]
    facts.extend(
        [
            _fact("sample_size", stat.sample_size, "count"),
            _fact("design", design, "string"),
            _fact("outcome_kind", parameters.get("outcome_kind"), "string"),
            _fact("imbalanced_covariate_count", len(imbalanced), "count"),
        ]
    )
    for index, row in enumerate(balance_rows[:_MAX_FACT_COLUMNS]):
        facts.extend(
            (
                _fact(f"covariate{index}.name", str(row["covariate"]), "string"),
                _fact(
                    f"covariate{index}.smd",
                    row["smd"],
                    "null" if row["smd"] is None else "number",
                ),
            )
        )
    for row in balance_rows:
        if row.get("smd") is not None and bool(row.get("imbalanced")) != (
            abs(float(row["smd"])) > CAUSAL_BALANCE_SMD_THRESHOLD
        ):
            raise ValueError(
                "balance row imbalance flag disagrees with its SMD value"
            )
    expected_disclaimer = (
        RANDOMIZED_DESIGN_ASSUMPTION if randomized else OBSERVATIONAL_CONTRAST_DISCLAIMER
    )
    expected_ci_method = (
        "welch_tconfint_unequal" if randomized else observational_ci_method(stat.test_type)
    )
    expected_keys = {
        "requested_design",
        "experiment_design",
        "treatment_column",
        "outcome_column",
        "outcome_kind",
        "balance_threshold",
        "ci_method",
        "requested_test_type",
        "test_family_id",
        "comparison_count",
        "session_attempt_count",
        "stat_attempt_id",
    }
    if randomized:
        expected_keys.add("design_confirmation_id")
    if "ate_definition" in parameters:
        expected_keys.add("ate_definition")
    statistics = receipt.statistics
    expected_content = {
        "artifact_id": primary.id,
        "balance_artifact_id": balance.id,
        "receipt_id": receipt.receipt_id,
        "design": design,
        "requested_design": arguments.get("design"),
        "test_type": stat.test_type,
        "outcome_kind": parameters.get("outcome_kind"),
        "p_value": stat.p_value,
        "adjusted_p_value": stat.adjusted_p_value,
        "effect_size": stat.effect_size,
        "effect_ci": [stat.effect_ci_low, stat.effect_ci_high],
        "sample_size": stat.sample_size,
        "groups": stat.groups,
        "ate_definition": parameters.get("ate_definition"),
        "ci_method": expected_ci_method,
        "imbalanced_covariates": imbalanced,
        "balance": balance_rows,
        "assumptions": [f"{item.name}={item.status}" for item in stat.assumptions],
        "disclaimer": expected_disclaimer,
        "warnings": [warning.message for warning in stat.warnings],
    }
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest
        != canonical_json_sha256(
            {"stat": stat.model_dump(mode="json"), "balance": balance.payload}
        )
        or receipt.result_count != 1
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns
        != (
            arguments.get("treatment_column"),
            arguments.get("outcome_column"),
            *(str(row["covariate"]) for row in balance_rows),
        )
        or receipt.method.family != expected_family
        or set(parameters) != expected_keys
        or parameters["requested_design"] != arguments.get("design")
        or parameters["treatment_column"] != arguments.get("treatment_column")
        or parameters["outcome_column"] != arguments.get("outcome_column")
        or parameters["balance_threshold"] != CAUSAL_BALANCE_SMD_THRESHOLD
        or parameters["ci_method"] != expected_ci_method
        or parameters["test_family_id"]
        != derive_family_id(
            dataset_id=str(arguments.get("dataset_id")),
            columns=(
                str(arguments.get("treatment_column")),
                str(arguments.get("outcome_column")),
            ),
        )
        or receipt.method.assumptions != (expected_disclaimer,)
        or receipt.method.warnings
        != tuple(warning.message for warning in stat.warnings)
        or statistics is None
        or statistics.test_name != stat.test_type
        or statistics.test_statistic != stat.statistic
        or statistics.p_value != stat.p_value
        or statistics.adjusted_p_value != stat.adjusted_p_value
        or statistics.effect_size != stat.effect_size
        or statistics.ci_low != stat.effect_ci_low
        or statistics.ci_high != stat.effect_ci_high
        or statistics.sample_size != stat.sample_size
        or statistics.sequence_index != parameters["comparison_count"]
        or _clipped(content) != _clipped(expected_content)
    ):
        raise ValueError(
            "run_causal_experiment receipt is not reconstructed by its durable result"
        )
    manifest = receipt.fact_manifest
    if not balance_rows:
        if manifest is not None:
            raise ValueError(
                "run_causal_experiment manifest must be empty without balance rows"
            )
        return
    if (
        manifest is None
        or manifest.total_rows != len(balance_rows)
        or manifest.unlisted_rows
        != len(balance_rows) - len(balance_rows[:_MAX_MANIFEST_ENTRIES])
        or len(manifest.entries) != len(balance_rows[:_MAX_MANIFEST_ENTRIES])
    ):
        raise ValueError("run_causal_experiment receipt fact manifest is incomplete")
    for index, (entry, row) in enumerate(
        zip(manifest.entries, balance_rows[:_MAX_MANIFEST_ENTRIES], strict=True)
    ):
        expected_status = "evaluated" if index < _MAX_FACT_COLUMNS else "unevaluated"
        if (
            entry.fact_id != f"covariate{index}"
            or entry.row_index != index
            or entry.status != expected_status
            or entry.row_digest != canonical_json_sha256(row)
        ):
            raise ValueError(
                "run_causal_experiment receipt manifest diverges from balance rows"
            )


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
    expected_parameters: dict[str, object] = {
        "target_column": arguments.get("target_column"),
        "group_column_count": len(diagnostic.group_columns),
        "target_correction": "holm",
        "top_k": arguments.get("top_k"),
        "mnar_ruled_out": False,
    }
    target_column = arguments.get("target_column")
    if isinstance(target_column, str) and target_column:
        # A target was tested, so the receipt must carry the registered
        # statistics of the top (smallest adjusted p) association.
        family_id = derive_family_id(
            dataset_id=str(arguments.get("dataset_id")), columns=(target_column,)
        )
        expected_parameters["requested_test_type"] = "missingness_target_association"
        expected_parameters["test_family_id"] = family_id
        top = (
            diagnostic.target_associations[0]
            if diagnostic.target_associations
            else None
        )
        statistics = receipt.statistics
        if (
            statistics is None
            or statistics.statistical_family_id != family_id
            or statistics.test_name
            != (top.test_name if top else "missingness_target_association")
            or statistics.p_value != (top.p_value if top else None)
            or statistics.adjusted_p_value != (top.adjusted_p_value if top else None)
            or statistics.effect_size != (top.effect_size if top else None)
            or statistics.sample_size != (top.sample_size if top else None)
        ):
            raise ValueError(
                "missingness receipt statistics do not match the top target association"
            )
    elif receipt.statistics is not None and receipt.statistics.p_value is not None:
        # Hypothesis adjudication may overlay a p-less statistics block; a
        # p-value without a target means an unregistered test.
        raise ValueError("missingness receipt without a target must carry no test p-value")
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


def _verify_correlate_columns(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild the screen from the published pair table, never from the receipt."""
    if primary.type is not ArtifactType.TABLE:
        raise ValueError("correlate_columns primary artifact has the wrong type")
    payload = primary.payload
    rows = _rows(payload)
    tested_rows = [row for row in rows if not row.get("insufficient_n")]
    significant = sum(
        1
        for row in tested_rows
        if row["adjusted_p"] is not None and float(row["adjusted_p"]) < 0.05
    )
    facts: list[ReceiptFact] = [
        _fact("pairs_tested", payload.get("pairs_tested"), "count"),
        _fact("pairs_insufficient_n", payload.get("pairs_insufficient_n"), "count"),
        _fact("correlation_method", payload.get("correlation_method"), "string"),
        _fact("correction_method", payload.get("correction_method"), "string"),
        _fact("min_pairwise_n", payload.get("min_pairwise_n"), "count"),
        _fact("significant_adjusted_pairs", significant, "count"),
    ]
    # Published rows are ordered tested-first, so an evaluated entry can never
    # land on an insufficient_n row.
    evaluated_count = min(_MAX_FACT_PAIRS, len(tested_rows))
    for index, row in enumerate(rows[:evaluated_count]):
        facts.extend(
            (
                _fact(f"pair{index}.coefficient", row["coefficient"], "number"),
                _fact(f"pair{index}.adjusted_p", row["adjusted_p"], "number"),
                _fact(
                    f"pair{index}.columns",
                    f"{row['column_a']}~{row['column_b']}",
                    "string",
                ),
            )
        )
    trivial = sum(1 for row in rows if row.get("is_trivial_pair"))
    expected_parameters = {
        "correlation_method": payload.get("correlation_method"),
        "correction_method": payload.get("correction_method"),
        "min_pairwise_n": payload.get("min_pairwise_n"),
        "pairs_tested": payload.get("pairs_tested"),
        "pairs_insufficient_n": payload.get("pairs_insufficient_n"),
        "pairs_degenerate": payload.get("pairs_degenerate"),
    }
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        "pairs_tested": payload.get("pairs_tested"),
        "pairs_insufficient_n": payload.get("pairs_insufficient_n"),
        "correlation_method": payload.get("correlation_method"),
        "correction_method": payload.get("correction_method"),
        "significant_adjusted_pairs": significant,
        "top_pairs": rows[:10],
    }
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest != canonical_json_sha256(payload)
        or receipt.result_count != payload.get("pairs_tested")
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.method.family
        != f"{payload.get('correlation_method')}_correlation_screen"
        or dict(receipt.method.parameters) != expected_parameters
        or receipt.method.warnings
        != (
            (f"{trivial} published pair(s) look trivially coupled (is_trivial_pair).",)
            if trivial
            else ()
        )
        or _clipped(_content(result)) != _clipped(expected_content)
    ):
        raise ValueError("correlate_columns receipt is not reconstructed by its durable result")
    _verify_pair_manifest(receipt, rows, evaluated_count=evaluated_count)


def _verify_screen_anomalies(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild the screen from the published anomaly result."""
    if primary.type is not ArtifactType.ANOMALY_SCREEN_RESULT:
        raise ValueError("screen_anomalies primary artifact has the wrong type")
    screen = AnomalyScreenResult.model_validate(primary.payload)
    facts = (
        _fact("outlier_count", screen.outlier_count, "count"),
        _fact("outlier_percent", round(screen.outlier_percent, 6), "percent", unit="percent"),
        _fact("median", screen.median, "number"),
        _fact("mad", screen.mad, "number"),
        _fact("q1", screen.q1, "number"),
        _fact("q3", screen.q3, "number"),
    )
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        "facts": {fact.fact_id: fact.value for fact in facts},
        "method": screen.method,
        "notes": screen.notes,
    }
    if (
        receipt.facts != facts
        or receipt.output_digest != canonical_json_sha256(screen.model_dump(mode="json"))
        or receipt.result_count != screen.outlier_count
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.columns != (arguments.get("column"),)
        # The screen records the method it actually ran, which falls back to
        # iqr when the MAD collapses — never merely what was requested.
        or receipt.method.family != screen.method
        or dict(receipt.method.parameters)
        != {
            "requested_method": arguments.get("method"),
            "threshold": screen.threshold,
        }
        or receipt.method.warnings != tuple(screen.notes)
        or _content(result) != expected_content
    ):
        raise ValueError("screen_anomalies receipt is not reconstructed by its durable result")


def _verify_baseline_model(
    receipt: EvidenceReceipt,
    result: AgentToolResult,
    primary: Artifact,
    arguments: Mapping[str, object],
) -> None:
    """Rebuild the model card's facts from the published card."""
    if primary.type is not ArtifactType.MODEL_CARD:
        raise ValueError("run_baseline_model primary artifact has the wrong type")
    card = ModelCard.model_validate(primary.payload)
    facts: list[ReceiptFact] = [
        _fact("task_type", card.task_type, "string"),
        _fact("target_column", card.target_column, "string"),
        _fact("split_strategy", card.split_strategy, "string"),
        _fact("model_type", card.model_type, "string"),
        _fact("train_rows", card.train_rows, "count"),
        _fact("test_rows", card.test_rows, "count"),
        _fact("feature_count", len(card.feature_columns), "count"),
        _fact("excluded_feature_count", len(card.excluded_features), "count"),
    ]
    if card.baseline_accuracy is not None:
        facts.append(_fact("baseline_accuracy", card.baseline_accuracy, "number"))
    for metric, value in sorted(card.metrics.items()):
        facts.append(_fact(f"metric.{metric}", value, "number"))
    for index, item in enumerate(card.feature_importance[:10]):
        facts.extend(
            (
                _fact(f"feature{index}.name", item.feature, "string"),
                _fact(f"feature{index}.importance", item.importance, "number"),
            )
        )
        if item.signed_importance is not None:
            facts.append(
                _fact(f"feature{index}.signed_importance", item.signed_importance, "number")
            )
        if item.importance_std is not None:
            facts.append(
                _fact(f"feature{index}.importance_std", item.importance_std, "number")
            )
    leakage_warnings = tuple(
        check.message
        for check in card.leakage_checks
        if check.severity in {"warn", "critical"}
    )
    expected_parameters = {
        "target_column": arguments.get("target_column"),
        "time_column": arguments.get("time_column"),
        "group_column": arguments.get("group_column"),
        "split_policy": arguments.get("split_policy"),
        "actual_split_strategy": card.split_strategy,
        "cv_folds": arguments.get("cv_folds"),
        "random_state": arguments.get("random_state"),
    }
    expected_content = {
        "artifact_id": primary.id,
        "receipt_id": receipt.receipt_id,
        **card.model_dump(mode="json"),
    }
    if (
        receipt.facts != tuple(facts)
        or receipt.output_digest != canonical_json_sha256(card.model_dump(mode="json"))
        or receipt.result_count != 1
        or receipt.scope.dataset_ids != (arguments.get("dataset_id"),)
        or receipt.scope.scope_resolution != "resolved"
        or receipt.method.family != "ml_baseline"
        or dict(receipt.method.parameters) != expected_parameters
        or receipt.method.assumptions
        != (
            f"split_strategy={card.split_strategy}",
            f"cross_validation_folds={arguments.get('cv_folds')}",
            "performance is predictive association, not causal evidence",
        )
        or receipt.method.warnings != tuple([*card.limitations, *leakage_warnings])
        or _clipped(_content(result)) != _clipped(expected_content)
    ):
        raise ValueError(
            "run_baseline_model receipt is not reconstructed by its durable result"
        )


def _verify_pair_manifest(
    receipt: EvidenceReceipt,
    rows: list[dict[str, Any]],
    *,
    evaluated_count: int,
) -> None:
    manifest = receipt.fact_manifest
    listed = rows[:_MAX_MANIFEST_ENTRIES]
    if manifest is None or manifest.unlisted_rows != len(rows) - len(listed):
        raise ValueError("correlate_columns receipt fact manifest is incomplete")
    if len(manifest.entries) != len(listed):
        raise ValueError(
            "correlate_columns receipt fact manifest does not cover artifact rows"
        )
    for index, (entry, row) in enumerate(zip(manifest.entries, listed, strict=True)):
        expected_id = (
            f"pair{index}.insufficient_n" if row.get("insufficient_n") else f"pair{index}"
        )
        expected_status = "evaluated" if index < evaluated_count else "unevaluated"
        if (
            entry.fact_id != expected_id
            or entry.row_index != index
            or entry.status != expected_status
        ):
            raise ValueError(
                "correlate_columns receipt manifest diverges from artifact rows"
            )


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


def _clipped(content: dict[str, Any]) -> dict[str, Any]:
    """Compare provider-facing content after the same clipping the tool applies."""
    from eda_platform.agents.data_tools import _clip_json

    return cast(dict[str, Any], _clip_json(content))


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
    return _rows_from(payload, "rows")


def _rows_from(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"primary artifact {key} are malformed")
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

"""Agent-facing deterministic analysis tools and their EvidenceReceipts.

Every tool must: validate arguments through a closed Pydantic model, stay
read-only, persist an EvidenceReceipt artifact whose digest verifies, and take
the robust alternative path (not a warning) when its preconditions fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    AssessJoinKeysArguments,
    CorrelateColumnsArguments,
    DataToolContext,
    RecommendCleaningArguments,
    RunDomainMetricsArguments,
    RunStatTestArguments,
    ScreenAnomaliesArguments,
    build_data_tools,
)
from eda_platform.core.column_roles import ColumnRole, ColumnRoleName, ColumnRoleSet
from eda_platform.core.ids import make_artifact_id
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    QualityIssue,
    QualityIssueSet,
)
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.receipts import EvidenceReceipt, verify_receipt_digest
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.ml_baseline import run_baseline_model
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.sql_runner import build_catalog
from eda_platform.tools.stat_tests import run_stat_test


def _dataset(name: str, frame: pd.DataFrame, dataset_id: str) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id=dataset_id,
            name=name,
            path=Path(f"/data/{name}"),
            content_hash="hash_" + dataset_id,
        ),
        frame=frame,
    )


def _context(
    datasets: list[LoadedDataset],
    *,
    artifacts: list[Artifact] | None = None,
    store: ArtifactStore | None = None,
) -> DataToolContext:
    return DataToolContext(
        datasets=datasets,
        catalog=build_catalog(datasets),
        project_id="project_t",
        session_id="run_t",
        store=store,
        payload_policy="schema+aggregates",
        artifacts=list(artifacts or []),
    )


def _tool(context: DataToolContext, name: str) -> Any:
    return next(tool for tool in build_data_tools(context) if tool.name == name)


def _receipts(context: DataToolContext) -> list[Artifact]:
    return [a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT]


def _last_receipt(context: DataToolContext) -> tuple[Artifact, EvidenceReceipt]:
    artifact = _receipts(context)[-1]
    return artifact, EvidenceReceipt.model_validate(artifact.payload)


def _fact(receipt: EvidenceReceipt, fact_id: str) -> Any:
    return next(fact for fact in receipt.facts if fact.fact_id == fact_id)


# ---------------------------------------------------------------------------
# assess_join_keys
# ---------------------------------------------------------------------------


def _join_datasets() -> list[LoadedDataset]:
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i:03d}" for i in range(20)],
            "region": ["east", "west"] * 10,
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i:03d}" for i in range(40)],
            "customer_id": [f"c{i % 20:03d}" for i in range(40)],
            "amount": [float(10 + i) for i in range(40)],
        }
    )
    return [
        _dataset("orders.csv", orders, "ds_orders"),
        _dataset("customers.csv", customers, "ds_customers"),
    ]


def test_assess_join_keys_produces_a_verified_receipt() -> None:
    context = _context(_join_datasets())
    tool = _tool(context, "assess_join_keys")
    result = tool.execute(
        AssessJoinKeysArguments(
            left_dataset_id="ds_orders",
            right_dataset_id="ds_customers",
            left_columns=["customer_id"],
            right_columns=["customer_id"],
        )
    )

    primary = next(
        a for a in context.artifacts if a.type is ArtifactType.RELATIONSHIP_VALIDATION_SET
    )
    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.tool_name == "assess_join_keys"
    assert receipt.artifact_ids == (primary.id,)
    assert receipt_artifact.parents == [primary.id]
    assert _fact(receipt, "containment_left_in_right").value == 1.0
    assert _fact(receipt, "right_unique_rate").value == 1.0
    assert _fact(receipt, "join_row_multiplier").value == 1.0
    assert _fact(receipt, "cardinality").value == "many_to_one"
    assert _fact(receipt, "orphan_rate_left").value == 0.0
    assert isinstance(result.content, dict)
    assert result.content["receipt_id"] == receipt.receipt_id


def test_assess_join_keys_rejects_malformed_arguments() -> None:
    with pytest.raises(ValidationError):
        AssessJoinKeysArguments(
            left_dataset_id="ds_orders",
            right_dataset_id="ds_customers",
            left_columns=[],
            right_columns=["customer_id"],
        )
    with pytest.raises(ValidationError):
        AssessJoinKeysArguments(
            left_dataset_id="ds_orders",
            right_dataset_id="ds_customers",
            left_columns=["a", "b"],
            right_columns=["a"],
        )
    with pytest.raises(ValidationError):
        AssessJoinKeysArguments(
            left_dataset_id="ds_orders",
            right_dataset_id="ds_customers",
            left_columns=["customer_id"],
            right_columns=["customer_id"],
            extra_field=1,  # type: ignore[call-arg]
        )


def test_assess_join_keys_missing_column_is_rejected() -> None:
    context = _context(_join_datasets())
    tool = _tool(context, "assess_join_keys")
    with pytest.raises(ValueError, match="not_a_column"):
        tool.execute(
            AssessJoinKeysArguments(
                left_dataset_id="ds_orders",
                right_dataset_id="ds_customers",
                left_columns=["not_a_column"],
                right_columns=["customer_id"],
            )
        )


def test_assess_join_keys_validates_even_a_pair_discovery_prefilters() -> None:
    """A measure-vs-measure pair is structurally implausible; the tool must
    still run the SQL validation and disclose that overlap signals are absent."""
    left = pd.DataFrame({"amount": [1.5, 2.5, 3.5, 4.5], "k": list("abcd")})
    right = pd.DataFrame({"amount": [1.5, 2.5, 9.9, 8.8], "k": list("wxyz")})
    context = _context(
        [_dataset("l.csv", left, "ds_l"), _dataset("r.csv", right, "ds_r")]
    )
    tool = _tool(context, "assess_join_keys")
    tool.execute(
        AssessJoinKeysArguments(
            left_dataset_id="ds_l",
            right_dataset_id="ds_r",
            left_columns=["amount"],
            right_columns=["amount"],
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert any("signal" in warning.lower() for warning in receipt.method.warnings)
    assert _fact(receipt, "join_row_multiplier").value is not None
    assert not any(fact.fact_id == "containment_left_in_right" for fact in receipt.facts)


# ---------------------------------------------------------------------------
# screen_anomalies
# ---------------------------------------------------------------------------


def test_screen_anomalies_produces_receipt_and_persists_it(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_t", name="T")
    store.start_session("project_t", "run_t")
    values = [10.0] * 40 + [11.0] * 40 + [500.0]
    frame = pd.DataFrame({"amount": values})
    context = _context([_dataset("a.csv", frame, "ds_a")], store=store)
    tool = _tool(context, "screen_anomalies")
    tool.execute(ScreenAnomaliesArguments(dataset_id="ds_a", column="amount"))

    primary = next(
        a for a in context.artifacts if a.type is ArtifactType.ANOMALY_SCREEN_RESULT
    )
    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.artifact_ids == (primary.id,)
    assert receipt_artifact.parents == [primary.id]
    assert _fact(receipt, "outlier_count").value == 1
    assert receipt.result_count == 1
    assert receipt.method.family == "robust_zscore"
    # persist=True must round-trip through the store like _run_sql artifacts do.
    stored = store.get_artifact(
        receipt_artifact.id, project_id="project_t", session_id="run_t"
    )
    assert stored.type is ArtifactType.EVIDENCE_RECEIPT


def test_screen_anomalies_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        ScreenAnomaliesArguments(dataset_id="ds_a", column="amount", method="zscore")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ScreenAnomaliesArguments(dataset_id="ds_a", column="amount", threshold=-1.0)
    with pytest.raises(ValidationError):
        ScreenAnomaliesArguments(dataset_id="ds_a", column="amount", nope=1)  # type: ignore[call-arg]


def test_screen_anomalies_zero_mad_takes_the_iqr_fallback_path() -> None:
    # 60 identical values force MAD to zero while the IQR stays positive, so the
    # deterministic fallback (not a warning) must be recorded as the method.
    values = [5.0] * 60 + [float(v) for v in range(6, 46)]
    frame = pd.DataFrame({"amount": values})
    context = _context([_dataset("a.csv", frame, "ds_a")])
    tool = _tool(context, "screen_anomalies")
    tool.execute(
        ScreenAnomaliesArguments(dataset_id="ds_a", column="amount", method="robust_zscore")
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.method.family == "iqr"
    assert receipt.method.parameters["requested_method"] == "robust_zscore"


# ---------------------------------------------------------------------------
# run_domain_metrics
# ---------------------------------------------------------------------------


def _events_dataset() -> LoadedDataset:
    frame = pd.DataFrame(
        {
            # 2024-01-01 + 59 days ends on 2024-02-29: exactly two covered months.
            "created_at": pd.date_range("2024-01-01", periods=60, freq="D").astype(str),
            "value": [float(i % 7) for i in range(60)],
        }
    )
    return _dataset("events.csv", frame, "ds_events")


def test_run_domain_metrics_computes_resolved_metrics_with_receipt() -> None:
    events = _events_dataset()
    profile_artifact = profile_dataset(events, project_id="project_t", session_id="run_t")
    role_set = ColumnRoleSet(
        dataset="events.csv",
        roles=[
            ColumnRole(
                column="created_at",
                role=ColumnRoleName.TIMESTAMP,
                confidence=0.9,
                provenance="seeded",
            )
        ],
    )
    role_artifact = Artifact(
        id=make_artifact_id("roles", role_set.model_dump(mode="json")),
        type=ArtifactType.COLUMN_ROLE_SET,
        project_id="project_t",
        session_id="run_t",
        payload=role_set.model_dump(mode="json"),
    )
    context = _context([events], artifacts=[profile_artifact, role_artifact])
    tool = _tool(context, "run_domain_metrics")
    result = tool.execute(RunDomainMetricsArguments())

    sql_artifacts = [a for a in context.artifacts if a.type is ArtifactType.SQL_RESULT]
    assert sql_artifacts, "each computed metric must produce a SQL_RESULT artifact"
    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.result_count >= 1
    assert set(receipt.artifact_ids) == {a.id for a in sql_artifacts}
    assert set(receipt_artifact.parents) == {a.id for a in sql_artifacts}
    coverage_facts = [f for f in receipt.facts if f.fact_id.startswith("time_coverage.")]
    assert any(f.fact_id == "time_coverage.covered_months" and f.value == 2 for f in coverage_facts)
    assert _fact(receipt, "time_coverage.contract_valid").value is True
    assert isinstance(result.content, dict)


def test_run_domain_metrics_rejects_unknown_arguments() -> None:
    with pytest.raises(ValidationError):
        RunDomainMetricsArguments(metric="gmv")  # type: ignore[call-arg]


def test_run_domain_metrics_without_profiles_is_a_clear_error() -> None:
    context = _context([_events_dataset()])
    tool = _tool(context, "run_domain_metrics")
    with pytest.raises(ValueError, match="profile"):
        tool.execute(RunDomainMetricsArguments())


def test_run_domain_metrics_with_no_applicable_metric_emits_absence_fact() -> None:
    events = _events_dataset()
    profile_artifact = profile_dataset(events, project_id="project_t", session_id="run_t")
    # No role sets at all: every registered metric must skip deterministically.
    context = _context([events], artifacts=[profile_artifact])
    tool = _tool(context, "run_domain_metrics")
    tool.execute(RunDomainMetricsArguments())
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.result_count == 0
    absence = _fact(receipt, "no_applicable_metrics")
    assert absence.support_type == "absence"
    assert receipt.method.warnings, "skip reasons must be disclosed"


# ---------------------------------------------------------------------------
# recommend_cleaning
# ---------------------------------------------------------------------------


def _cleaning_artifacts(events: LoadedDataset, issues: list[QualityIssue]) -> list[Artifact]:
    profile_artifact = profile_dataset(events, project_id="project_t", session_id="run_t")
    issue_set = QualityIssueSet(dataset_id=events.record.dataset_id, issues=issues)
    quality_artifact = Artifact(
        id=make_artifact_id("quality", issue_set.model_dump(mode="json")),
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_t",
        session_id="run_t",
        payload=issue_set.model_dump(mode="json"),
    )
    return [profile_artifact, quality_artifact]


def test_recommend_cleaning_proposes_without_executing() -> None:
    events = _events_dataset()
    issues = [
        QualityIssue(
            severity="warn",
            code="surrounding_whitespace",
            column="created_at",
            message="3 values carry surrounding whitespace.",
            recommendation="Trim whitespace.",
            affected_count=3,
        )
    ]
    artifacts = _cleaning_artifacts(events, issues)
    context = _context([events], artifacts=artifacts)
    tool = _tool(context, "recommend_cleaning")
    result = tool.execute(RecommendCleaningArguments(dataset_id="ds_events"))

    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.result_count == 1
    assert _fact(receipt, "op0.operation").value == "trim_whitespace"
    assert _fact(receipt, "op0.column").value == "created_at"
    assert _fact(receipt, "op0.severity").value == "warn"
    assert _fact(receipt, "op0.lossy").value is False
    # Proposals only: the dataset frame must be untouched and no recipe created.
    assert not any(a.type is ArtifactType.CLEANING_RECIPE for a in context.artifacts)
    assert set(receipt_artifact.parents) == {artifacts[0].id, artifacts[1].id}
    assert isinstance(result.content, dict)


def test_recommend_cleaning_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        RecommendCleaningArguments(dataset_id="")
    with pytest.raises(ValidationError):
        RecommendCleaningArguments(dataset_id="ds", apply=True)  # type: ignore[call-arg]


def test_recommend_cleaning_without_quality_artifacts_is_a_clear_error() -> None:
    events = _events_dataset()
    profile_artifact = profile_dataset(events, project_id="project_t", session_id="run_t")
    context = _context([events], artifacts=[profile_artifact])
    tool = _tool(context, "recommend_cleaning")
    with pytest.raises(ValueError, match="quality"):
        tool.execute(RecommendCleaningArguments(dataset_id="ds_events"))


def test_recommend_cleaning_with_no_issues_emits_absence_fact() -> None:
    events = _events_dataset()
    context = _context([events], artifacts=_cleaning_artifacts(events, []))
    tool = _tool(context, "recommend_cleaning")
    tool.execute(RecommendCleaningArguments(dataset_id="ds_events"))
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.result_count == 0
    assert _fact(receipt, "no_recommended_operations").support_type == "absence"


# ---------------------------------------------------------------------------
# run_stat_test (agent tool) + precondition gates in tools/stat_tests.py
# ---------------------------------------------------------------------------


def _t_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["A"] * 40 + ["B"] * 40,
            "revenue": [10.0 + (i % 5) * 0.5 for i in range(40)]
            + [20.0 + (i % 7) * 0.4 for i in range(40)],
        }
    )


def test_run_stat_test_receipt_records_family_id_and_effect_ci() -> None:
    context = _context([_dataset("sales.csv", _t_test_frame(), "ds_sales")])
    tool = _tool(context, "run_stat_test")
    tool.execute(
        RunStatTestArguments(
            dataset_id="ds_sales",
            test_type="independent_t_test",
            group_column="segment",
            value_column="revenue",
            test_family_id="fam_revenue_by_segment",
        )
    )
    primary = next(a for a in context.artifacts if a.type is ArtifactType.STAT_TEST_RESULT)
    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.artifact_ids == (primary.id,)
    assert receipt_artifact.parents == [primary.id]
    assert receipt.statistics is not None
    assert receipt.statistics.hypothesis_id == "fam_revenue_by_segment"
    assert receipt.statistics.test_name == "independent_t_test"
    assert receipt.statistics.p_value is not None
    assert receipt.statistics.ci_low is not None
    assert receipt.statistics.ci_high is not None
    assert (
        receipt.statistics.ci_low
        <= receipt.statistics.effect_size
        <= receipt.statistics.ci_high
    )
    assert receipt.method.assumptions, "assumption checks must be recorded"


def test_run_stat_test_requires_test_family_id() -> None:
    with pytest.raises(ValidationError):
        RunStatTestArguments(
            dataset_id="ds_sales",
            test_type="independent_t_test",
            group_column="segment",
            value_column="revenue",
        )  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        RunStatTestArguments(
            dataset_id="ds_sales",
            test_type="no_such_test",  # type: ignore[arg-type]
            test_family_id="fam",
        )


def test_run_stat_test_heteroscedastic_anova_switches_to_welch() -> None:
    frame = pd.DataFrame(
        {
            "grp": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
            "val": [10.0 + (i % 3) * 0.01 for i in range(30)]
            + [12.0 + ((-1) ** i) * 8.0 + (i % 5) for i in range(30)]
            + [11.0 + (i % 2) * 0.02 for i in range(30)],
        }
    )
    context = _context([_dataset("g.csv", frame, "ds_g")])
    tool = _tool(context, "run_stat_test")
    tool.execute(
        RunStatTestArguments(
            dataset_id="ds_g",
            test_type="one_way_anova",
            group_column="grp",
            value_column="val",
            test_family_id="fam_var",
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.method.family == "welch_anova"
    assert receipt.method.parameters["requested_test_type"] == "one_way_anova"
    assert any("welch" in w.lower() for w in receipt.method.warnings)


def test_run_stat_test_sparse_2x2_receipt_is_fisher_not_chi2() -> None:
    frame = pd.DataFrame(
        {
            "grp": ["x"] * 10 + ["y"] * 10,
            "cat": ["yes"] * 2 + ["no"] * 8 + ["yes"] * 1 + ["no"] * 9,
        }
    )
    context = _context([_dataset("s.csv", frame, "ds_s")])
    tool = _tool(context, "run_stat_test")
    tool.execute(
        RunStatTestArguments(
            dataset_id="ds_s",
            test_type="chi_square_independence",
            group_column="grp",
            category_column="cat",
            test_family_id="fam_sparse",
        )
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.method.family == "fisher_exact"
    assert receipt.method.family != "chi_square_independence"
    assert receipt.statistics is not None
    assert receipt.statistics.test_name == "fisher_exact"


def test_chi_square_sparse_2x2_switches_to_fisher_in_stat_tests() -> None:
    frame = pd.DataFrame(
        {
            "grp": ["x"] * 10 + ["y"] * 10,
            "cat": ["yes"] * 2 + ["no"] * 8 + ["yes"] * 1 + ["no"] * 9,
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="ds_s",
        test_type="chi_square_independence",
        group_column="grp",
        category_column="cat",
    )
    assert result.test_type == "fisher_exact"
    assert result.p_value is not None
    assert any(w.code == "fisher_exact_fallback" for w in result.warnings)


def test_chi_square_sparse_rxc_is_rejected_not_waved_through() -> None:
    frame = pd.DataFrame(
        {
            "grp": ["x"] * 4 + ["y"] * 4 + ["z"] * 4,
            "cat": ["a", "a", "b", "b"] * 3,
        }
    )
    with pytest.raises(ValueError, match="[Ff]isher|expected"):
        run_stat_test(
            frame,
            dataset_id="ds_s",
            test_type="chi_square_independence",
            group_column="grp",
            category_column="cat",
        )


def test_effect_ci_is_opt_in_and_brackets_the_effect() -> None:
    frame = _t_test_frame()
    plain = run_stat_test(
        frame,
        dataset_id="ds_sales",
        test_type="independent_t_test",
        group_column="segment",
        value_column="revenue",
    )
    assert plain.effect_ci_low is None and plain.effect_ci_high is None

    with_ci = run_stat_test(
        frame,
        dataset_id="ds_sales",
        test_type="independent_t_test",
        group_column="segment",
        value_column="revenue",
        effect_ci=True,
    )
    assert with_ci.effect_ci_low is not None and with_ci.effect_ci_high is not None
    assert with_ci.effect_size is not None
    assert with_ci.effect_ci_low <= with_ci.effect_size <= with_ci.effect_ci_high

    again = run_stat_test(
        frame,
        dataset_id="ds_sales",
        test_type="independent_t_test",
        group_column="segment",
        value_column="revenue",
        effect_ci=True,
    )
    assert again.effect_ci_low == with_ci.effect_ci_low, "bootstrap must be deterministic"


def test_stat_test_result_payloads_without_ci_fields_still_load() -> None:
    legacy = {
        "dataset_id": "ds_old",
        "test_type": "independent_t_test",
        "group_column": "g",
        "value_column": "v",
        "statistic": 2.0,
        "p_value": 0.04,
        "sample_size": 20,
    }
    restored = StatTestResult.model_validate(legacy)
    assert restored.effect_ci_low is None
    assert restored.effect_ci_high is None


# ---------------------------------------------------------------------------
# correlate_columns
# ---------------------------------------------------------------------------


def _corr_frame() -> pd.DataFrame:
    x = [float(i) for i in range(60)]
    return pd.DataFrame(
        {
            "x": x,
            "y": [2.0 * v + ((i * 13) % 7) * 0.6 for i, v in enumerate(x)],
            "z": [float((i * 7) % 13) for i in range(60)],
        }
    )


def test_correlate_columns_reports_adjusted_p_for_every_pair() -> None:
    context = _context([_dataset("c.csv", _corr_frame(), "ds_c")])
    tool = _tool(context, "correlate_columns")
    result = tool.execute(CorrelateColumnsArguments(dataset_id="ds_c"))

    table_artifact = next(a for a in context.artifacts if a.type is ArtifactType.TABLE)
    assert table_artifact.payload["kind"] == "correlation"
    rows = table_artifact.payload["rows"]
    assert rows and all("adjusted_p" in row and row["adjusted_p"] is not None for row in rows)
    assert all(row["correction_method"] == "fdr_bh" for row in rows)
    assert all(row["pairs_tested"] == 3 for row in rows)
    assert all("pairwise_complete_n" in row for row in rows)
    assert all("is_trivial_pair" in row for row in rows)

    receipt_artifact, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.artifact_ids == (table_artifact.id,)
    assert receipt_artifact.parents == [table_artifact.id]
    assert _fact(receipt, "pairs_tested").value == 3
    assert _fact(receipt, "correction_method").value == "fdr_bh"
    assert isinstance(result.content, dict)


def test_correlate_columns_rejects_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        CorrelateColumnsArguments(dataset_id="ds_c", correction_method="bonferroni")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CorrelateColumnsArguments(dataset_id="ds_c", columns=["only_one"])
    with pytest.raises(ValidationError):
        CorrelateColumnsArguments(dataset_id="ds_c", nope=1)  # type: ignore[call-arg]


def test_correlate_columns_needs_two_numeric_columns() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    context = _context([_dataset("c.csv", frame, "ds_c")])
    tool = _tool(context, "correlate_columns")
    with pytest.raises(ValueError, match="numeric"):
        tool.execute(CorrelateColumnsArguments(dataset_id="ds_c"))


def test_correlate_columns_supports_holm_correction() -> None:
    context = _context([_dataset("c.csv", _corr_frame(), "ds_c")])
    tool = _tool(context, "correlate_columns")
    tool.execute(
        CorrelateColumnsArguments(dataset_id="ds_c", correction_method="holm")
    )
    _, receipt = _last_receipt(context)
    assert verify_receipt_digest(receipt)
    assert _fact(receipt, "correction_method").value == "holm"


# ---------------------------------------------------------------------------
# run_baseline_model: permutation importance replaces MDI
# ---------------------------------------------------------------------------


def _baseline_frame() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 240
    segment = ["a" if i % 2 == 0 else "b" for i in range(n)]
    labels = [
        seg if rng.rand() > 0.05 else ("b" if seg == "a" else "a") for seg in segment
    ]
    return pd.DataFrame(
        {
            "seg": segment,
            "noise": rng.rand(n),  # continuous high-cardinality, unrelated to the label
            "label": labels,
        }
    )


def test_baseline_importance_is_computed_by_permutation_on_the_test_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.inspection as inspection

    calls: dict[str, Any] = {}
    real = inspection.permutation_importance

    def spy(model: Any, features: Any, target: Any, **kwargs: Any) -> Any:
        calls["count"] = calls.get("count", 0) + 1
        calls["rows"] = len(features)
        calls["n_repeats"] = kwargs.get("n_repeats")
        return real(model, features, target, **kwargs)

    monkeypatch.setattr(inspection, "permutation_importance", spy)
    card = run_baseline_model(_baseline_frame(), dataset_id="ds_b", target_column="label")
    assert calls["count"] == 1
    assert calls["n_repeats"] == 5
    assert calls["rows"] == card.test_rows, "importance must be computed on the test split"


def test_baseline_importance_no_longer_crowns_a_high_cardinality_noise_column() -> None:
    card = run_baseline_model(_baseline_frame(), dataset_id="ds_b", target_column="label")
    assert card.feature_importance, "importance rows must be present"
    top = card.feature_importance[0]
    assert not top.feature.startswith("noise"), (
        f"high-cardinality noise column still tops importance: {top}"
    )
    noise_importance = sum(
        row.importance for row in card.feature_importance if row.feature.startswith("noise")
    )
    seg_importance = sum(
        row.importance for row in card.feature_importance if row.feature.startswith("seg")
    )
    assert seg_importance > noise_importance
    # Discriminates permutation from MDI: on this exact data the impurity-based
    # (MDI) importance of the continuous noise column is ~0.247, while its
    # test-set permutation importance collapses to ~0.008.
    assert noise_importance < 0.05


def test_baseline_importance_falls_back_to_mdi_with_disclosed_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.inspection as inspection

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("permutation backend unavailable")

    monkeypatch.setattr(inspection, "permutation_importance", broken)
    card = run_baseline_model(_baseline_frame(), dataset_id="ds_b", target_column="label")
    assert card.feature_importance, "MDI fallback must still deliver importances"
    assert any(
        "impurity" in limitation.lower() or "mdi" in limitation.lower()
        for limitation in card.limitations
    ), "the MDI bias must be disclosed in limitations"


def test_list_saved_skills_exposes_provenance_and_usage_guidance(tmp_path: Path) -> None:
    """The agent decides whether to call a skill from this listing alone, so the
    trust tier and when-to-use guidance must survive the passthrough."""
    from eda_platform.core.skills_store import add_skill
    from eda_platform.drivers.analysis_skill import skill_from_plan
    from eda_platform.schemas.plans import AnalysisPlan

    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_t", name="T")
    plan = AnalysisPlan(
        question="Which region has the highest total?",
        dataset_names=["orders"],
        columns=["region", "amount"],
        sql="SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
        method="descriptive_sql",
        rationale="test",
    )
    skill = skill_from_plan(plan, "regional totals", "", source_session_id="run_x")
    skill = skill.model_copy(
        update={
            "origin": "user_template",
            "when_to_use": "When comparing totals across one dimension.",
            "when_not_to_use": "When the metric needs deduplication first.",
        }
    )
    add_skill(store.project_dir("project_t"), skill)

    dataset = _dataset(
        "orders", pd.DataFrame({"region": ["E", "W"], "amount": [1, 2]}), "ds_orders"
    )
    tool = _tool(_context([dataset], store=store), "list_saved_skills")
    listing = tool.execute(tool.args_schema())
    row = next(
        item
        for item in listing.content["skills"]
        if item["name"] == "regional totals"
    )

    assert row["origin"] == "user_template"
    assert row["when_to_use"].startswith("When comparing")
    assert row["when_not_to_use"]

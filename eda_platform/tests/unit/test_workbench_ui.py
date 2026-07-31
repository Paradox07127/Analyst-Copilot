from eda_platform.application.workbench import (
    SMALL_SAMPLE_THRESHOLD,
    checkbox_disabled_for_feasibility,
    dataset_display_rows,
    dataset_names_by_id,
    effect_size_magnitude_badge,
    format_p_value,
    group_quality_issues,
    leakage_verdict_badge,
    min_sample_size,
    run_cost_summary,
    semantic_type_counts,
    split_trivial_correlation_rows,
    summarize_session,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.model_card import LeakageCheck


def test_workbench_helpers_summarize_artifacts_and_quality() -> None:
    profile = Artifact(
        id="prof_1",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload={
            "dataset_id": "ds_sales",
            "name": "sales.csv",
            "rows": 2,
            "columns": 2,
            "column_names": ["order_id", "amount"],
            "dtypes": {"order_id": "int64", "amount": "float64"},
            "missing_values": {"order_id": 0, "amount": 1},
            "missing_percent": {"order_id": 0.0, "amount": 50.0},
            "numeric_columns": ["amount"],
            "categorical_columns": ["order_id"],
            "duplicate_rows": 0,
            "columns_detail": [
                {
                    "name": "order_id",
                    "dtype": "int64",
                    "semantic_type": "id",
                    "missing_count": 0,
                    "missing_percent": 0.0,
                    "unique_count": 2,
                    "unique_percent": 100.0,
                    "sample_values": ["1", "2"],
                },
                {
                    "name": "amount",
                    "dtype": "float64",
                    "semantic_type": "numeric",
                    "missing_count": 1,
                    "missing_percent": 50.0,
                    "unique_count": 1,
                    "unique_percent": 50.0,
                    "sample_values": ["10.0"],
                },
            ],
            "semantic_type_counts": {"id": 1, "numeric": 1},
        },
    )
    quality = Artifact(
        id="quality_1",
        type=ArtifactType.QUALITY_ISSUE_SET,
        project_id="project_demo",
        session_id="run_demo",
        payload={
            "dataset_id": "ds_sales",
            "issues": [
                {
                    "severity": "warn",
                    "code": "high_missing",
                    "column": "amount",
                    "message": "Column amount has 50.00% missing values.",
                    "recommendation": "Review missingness.",
                }
            ],
        },
    )

    artifacts = [profile, quality]

    assert summarize_session(artifacts) == {
        "artifacts": 2,
        "datasets": 1,
        "critical": 0,
        "warn": 1,
        "info": 0,
    }
    assert dataset_display_rows(profile)[0]["semantic_type"] == "id"
    assert semantic_type_counts(profile) == {"id": 1, "numeric": 1}
    assert dataset_names_by_id(artifacts) == {"ds_sales": "sales.csv"}
    quality_row = group_quality_issues(artifacts)["warn"][0]
    assert quality_row["code"] == "high_missing"
    assert quality_row["dataset_name"] == "sales.csv"
    assert "dataset_id" not in quality_row
    assert "artifact_id" not in quality_row
    assert list(quality_row.keys())[:2] == ["message", "column"]


def test_workbench_helpers_handle_legacy_profile_payload_without_m1_fields() -> None:
    legacy_profile = Artifact(
        id="prof_legacy",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project_demo",
        session_id="run_demo",
        payload={
            "dataset_id": "ds_legacy",
            "name": "legacy.csv",
            "rows": 2,
            "columns": 2,
            "column_names": ["order_id", "amount"],
            "dtypes": {"order_id": "int64", "amount": "float64"},
            "missing_values": {"order_id": 0, "amount": 1},
            "missing_percent": {"order_id": 0.0, "amount": 50.0},
            "numeric_columns": ["order_id", "amount"],
            "categorical_columns": [],
            "sample_rows": [{"order_id": 1, "amount": 10}, {"order_id": 2, "amount": None}],
        },
    )

    assert semantic_type_counts(legacy_profile) == {"numeric": 2}
    assert dataset_display_rows(legacy_profile)[0]["semantic_type"] == "numeric"


def test_split_trivial_correlation_rows_partitions_without_mutation() -> None:
    rows = [
        {"column_a": "x", "column_b": "y", "pearson": 0.4, "is_trivial_pair": False},
        {"column_a": "a", "column_b": "b", "pearson": 1.0, "is_trivial_pair": True},
    ]

    substantive, trivial = split_trivial_correlation_rows(rows)

    assert [row["column_a"] for row in substantive] == ["x"]
    assert [row["column_a"] for row in trivial] == ["a"]
    substantive[0]["pearson"] = 0.99
    assert rows[0]["pearson"] == 0.4


def test_min_sample_size_and_format_p_value() -> None:
    assert min_sample_size([{"sample_size": 40}, {"sample_size": 20}]) == 20
    assert min_sample_size([{"count": 5}]) is None
    assert min_sample_size([]) is None
    assert SMALL_SAMPLE_THRESHOLD == 30
    assert format_p_value(0.0) == "<0.001"
    assert format_p_value(1e-50) == "<0.001"
    assert format_p_value(0.0012) == "0.0012"
    assert format_p_value(0.05) == "0.05"
    assert format_p_value(0.123456) == "0.123"
    assert format_p_value(None) == ""


def test_effect_size_and_leakage_labels() -> None:
    assert effect_size_magnitude_badge("independent_t_test", 0.1) == "negligible"
    assert effect_size_magnitude_badge("independent_t_test", 0.3) == "small"
    assert effect_size_magnitude_badge("independent_t_test", 0.6) == "medium"
    assert effect_size_magnitude_badge("independent_t_test", 0.9) == "large"
    assert effect_size_magnitude_badge("independent_t_test", None) == ""
    assert leakage_verdict_badge([]) == "unchecked"
    assert (
        leakage_verdict_badge(
            [
                LeakageCheck(
                    code="id_like_feature",
                    severity="info",
                    column="id",
                    action="excluded",
                    message="excluded",
                )
            ]
        )
        == "mitigated"
    )


def test_checkbox_disabled_for_feasibility() -> None:
    assert checkbox_disabled_for_feasibility("needs_data") is True
    assert checkbox_disabled_for_feasibility("unsuitable") is True
    assert checkbox_disabled_for_feasibility("ready") is False
    assert checkbox_disabled_for_feasibility(None) is False


def test_run_cost_summary_prefers_session_metrics() -> None:
    metrics = Artifact(
        id="metrics_1",
        type=ArtifactType.SESSION_METRICS,
        project_id="project_demo",
        session_id="run_demo",
        payload={
            "llm_calls": 4,
            "total_tokens": 1000,
            "prompt_tokens": 800,
            "cached_tokens": 100,
            "est_cost_usd": 0.01,
        },
    )
    summary = Artifact(
        id="summary_1",
        type=ArtifactType.SESSION_SUMMARY,
        project_id="project_demo",
        session_id="run_demo",
        payload={"model": "offline", "llm_call_count": 1, "total_tokens": 10},
    )
    cost = run_cost_summary([metrics, summary])
    assert cost is not None
    assert cost["scope"] == "session"
    assert cost["llm_call_count"] == 4
    assert cost["model"] == "offline"

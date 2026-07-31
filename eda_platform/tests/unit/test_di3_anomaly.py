from __future__ import annotations

import pandas as pd
import pytest

from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.tools.anomaly import create_anomaly_artifact, screen_anomalies


def test_robust_zscore_sorts_and_bounds_top_outliers() -> None:
    frame = pd.DataFrame({"amount": list(range(20)) + [1000, -900]})

    result = screen_anomalies(frame, dataset_name="orders", column="amount")

    assert result.method == "robust_zscore"
    assert result.threshold == 3.5
    assert result.outlier_count == 2
    assert len(result.top_outliers) == 2
    assert [item.row_index for item in result.top_outliers] == [20, 21]
    assert abs(result.top_outliers[0].score) >= abs(result.top_outliers[1].score)


def test_constant_column_reports_zero_outliers_when_mad_and_iqr_are_zero() -> None:
    result = screen_anomalies(
        pd.DataFrame({"amount": [7.0] * 12}),
        dataset_name="orders",
        column="amount",
    )

    assert result.mad == 0
    assert result.outlier_count == 0
    assert result.top_outliers == []
    assert any("no individual outliers" in note for note in result.notes)


def test_zero_mad_uses_deterministic_iqr_fallback_when_available() -> None:
    result = screen_anomalies(
        pd.DataFrame({"amount": [0, 0, 0, 0, 1, 2, 100]}),
        dataset_name="orders",
        column="amount",
    )

    assert result.mad == 0
    assert result.method == "iqr"
    assert result.threshold == 1.5
    assert "IQR fallback" in result.notes[0]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"amount": [None, None]}), "only null values"),
        (pd.DataFrame({"amount": ["10", "20"]}), "not numeric"),
    ],
)
def test_invalid_columns_raise_teaching_value_errors(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        screen_anomalies(frame, dataset_name="orders", column="amount")


def test_more_than_twenty_percent_outliers_warns_about_distribution_shift() -> None:
    result = screen_anomalies(
        pd.DataFrame({"amount": list(range(10))}),
        dataset_name="orders",
        column="amount",
        method="iqr",
        threshold=0.1,
    )

    assert result.outlier_percent > 20
    assert any(
        "distribution shift rather than individual anomalies" in note
        for note in result.notes
    )


def test_anomaly_artifact_has_bounded_payload_and_computation_provenance() -> None:
    result = screen_anomalies(
        pd.DataFrame({"amount": [1, 2, 3, 100]}),
        dataset_name="orders",
        column="amount",
        method="iqr",
    )

    artifact = create_anomaly_artifact(
        result,
        project_id="project",
        session_id="run",
        parents=["dataset"],
    )

    assert artifact.type == ArtifactType.ANOMALY_SCREEN_RESULT
    assert artifact.parents == ["dataset"]
    assert artifact.code_ref is not None and "screen_anomalies" in artifact.code_ref
    assert artifact.plain_language
    assert len(artifact.payload["top_outliers"]) <= 10

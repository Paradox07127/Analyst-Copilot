from __future__ import annotations

import pandas as pd
import pytest

from eda_platform.schemas.artifacts import AnalysisTable, ArtifactType
from eda_platform.tools.missingness import (
    create_missingness_artifact,
    diagnose_missingness,
)


def _structured_frame() -> pd.DataFrame:
    channels = ["phone"] * 40 + ["web"] * 40
    satisfaction: list[float | None] = [None] * 30 + [float(i) for i in range(10)]
    satisfaction += [None] * 2 + [float(i) for i in range(38)]
    return pd.DataFrame(
        {
            "channel": channels,
            "satisfaction": satisfaction,
            "spend": [10.0 if value is None else 100.0 + value for value in satisfaction],
            "complete": list(range(80)),
        }
    )


def test_diagnose_missingness_finds_group_and_target_structure() -> None:
    result = diagnose_missingness(
        _structured_frame(),
        dataset_id="ds_survey",
        dataset_name="survey.csv",
        target_column="spend",
        group_columns=["channel"],
    )

    assert result.mnar_ruled_out is False
    assert result.columns_with_missing == 1
    assert result.missing_percent["satisfaction"] == 40.0
    group = result.group_rate_ranges[0]
    assert group.missing_column == "satisfaction"
    assert group.group_column == "channel"
    assert group.range_percentage_points == 70.0
    association = result.target_associations[0]
    assert association.missing_column == "satisfaction"
    assert association.test_name == "point_biserial"
    assert association.adjusted_p_value < 0.05
    assert any("cannot rule out MNAR" in item for item in result.limitations)


def test_missing_indicator_correlations_are_ranked_by_absolute_phi() -> None:
    frame = pd.DataFrame(
        {
            "a": [None, None, 1.0, 1.0, None, 1.0],
            "b": [None, None, 2.0, 2.0, None, 2.0],
            "c": [3.0, None, 3.0, None, 3.0, None],
        }
    )
    result = diagnose_missingness(
        frame,
        dataset_id="ds",
        dataset_name="x.csv",
        group_columns=[],
    )

    assert result.indicator_correlations[0].column_a == "a"
    assert result.indicator_correlations[0].column_b == "b"
    assert result.indicator_correlations[0].phi == 1.0


def test_missingness_auto_groups_only_low_cardinality_columns() -> None:
    frame = _structured_frame().assign(
        row_id=[f"id-{index}" for index in range(80)],
    )
    result = diagnose_missingness(
        frame,
        dataset_id="ds",
        dataset_name="x.csv",
    )

    assert "channel" in result.group_columns
    assert "row_id" not in result.group_columns


def test_missingness_rejects_explicit_high_cardinality_group() -> None:
    frame = _structured_frame().assign(
        email=[f"person-{index}@example.com" for index in range(80)],
    )
    with pytest.raises(ValueError, match="high-cardinality"):
        diagnose_missingness(
            frame,
            dataset_id="ds",
            dataset_name="x.csv",
            group_columns=["email"],
        )


def test_missingness_skips_high_cardinality_categorical_target() -> None:
    frame = _structured_frame().assign(
        case_id=[f"case-{index}" for index in range(80)],
    )
    result = diagnose_missingness(
        frame,
        dataset_id="ds",
        dataset_name="x.csv",
        target_column="case_id",
        group_columns=[],
    )

    assert result.target_associations == []
    assert any("categorical target" in item for item in result.limitations)


def test_missingness_complete_data_still_publishes_zero_rates() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "group": ["a", "a", "b"]})
    result = diagnose_missingness(
        frame,
        dataset_id="ds",
        dataset_name="complete.csv",
        group_columns=["group"],
    )

    assert result.columns_with_missing == 0
    assert result.missing_percent == {"x": 0.0, "group": 0.0}
    assert result.indicator_correlations == []
    assert result.group_rate_ranges == []
    assert result.target_associations == []


def test_missingness_artifact_is_typed_and_self_describing() -> None:
    result = diagnose_missingness(
        _structured_frame(),
        dataset_id="ds",
        dataset_name="survey.csv",
        group_columns=["channel"],
    )
    artifact = create_missingness_artifact(
        result,
        project_id="project",
        session_id="session",
    )

    assert artifact.type is ArtifactType.TABLE
    table = AnalysisTable.model_validate(artifact.payload)
    assert table.kind == "missingness_diagnostic"
    assert artifact.payload["mnar_ruled_out"] is False
    assert artifact.code_ref == "tools.missingness.diagnose_missingness"

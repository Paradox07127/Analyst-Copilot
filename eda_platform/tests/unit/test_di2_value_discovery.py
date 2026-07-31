from typing import Literal

from eda_platform.core.methods import MethodGateContext, evaluate_feasibility
from eda_platform.schemas.artifacts import Artifact, ArtifactType, ColumnProfile, DatasetProfile
from eda_platform.schemas.questions import AnalysisMode
from eda_platform.tools.value_discovery import build_value_map

_VALID_MODES: set[AnalysisMode] = {
    "descriptive",
    "diagnostic",
    "forecast",
    "prediction",
    "segmentation",
    "anomaly",
    "causal_experiment",
}


def test_value_opportunities_use_enum_modes_and_registry_feasibility() -> None:
    profile = _profile(
        rows=40,
        columns=[
            _column("occurred_at", "datetime", 40),
            _column("customer_region", "categorical", 4),
            _column("cost_amount", "numeric", 38),
            _column("status", "categorical", 3),
        ],
    )
    value_map = build_value_map([_profile_artifact(profile)])

    assert value_map.opportunities
    assert {item.analysis_mode for item in value_map.opportunities} <= _VALID_MODES
    for opportunity in value_map.opportunities:
        canonical = evaluate_feasibility(
            MethodGateContext(
                profiles=[profile],
                target_datasets=opportunity.target_datasets,
                analysis_mode=opportunity.analysis_mode,
                target_column=None,
            )
        )
        assert opportunity.feasibility == canonical.status
        assert opportunity.feasibility_reasons == canonical.reasons


def test_value_map_mode_selection_is_deterministic_from_profile_capabilities() -> None:
    profile = _profile(
        rows=40,
        columns=[
            _column("event_time", "datetime", 40),
            _column("region", "categorical", 4),
            _column("revenue", "numeric", 40),
        ],
    )
    artifact = _profile_artifact(profile)

    first = build_value_map([artifact])
    second = build_value_map([artifact])

    assert first == second
    assert {item.analysis_mode for item in first.opportunities} == {"forecast", "diagnostic"}


def _profile(*, rows: int, columns: list[ColumnProfile]) -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=rows,
        columns=len(columns),
        column_names=[column.name for column in columns],
        dtypes={column.name: column.dtype for column in columns},
        missing_values={column.name: column.missing_count for column in columns},
        missing_percent={column.name: column.missing_percent for column in columns},
        numeric_columns=[column.name for column in columns if column.semantic_type == "numeric"],
        categorical_columns=[
            column.name for column in columns if column.semantic_type == "categorical"
        ],
        columns_detail=columns,
    )


_SemanticType = Literal[
    "numeric", "categorical", "datetime", "id", "boolean", "text", "unknown"
]


def _column(name: str, semantic_type: _SemanticType, unique_count: int) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        semantic_type=semantic_type,
        missing_count=0,
        missing_percent=0.0,
        unique_count=unique_count,
        unique_percent=100.0,
    )


def _profile_artifact(profile: DatasetProfile) -> Artifact:
    return Artifact(
        id="profile_1",
        type=ArtifactType.DATASET_PROFILE,
        project_id="project",
        session_id="run",
        payload=profile.model_dump(mode="json"),
    )

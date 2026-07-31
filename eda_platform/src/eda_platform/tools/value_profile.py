from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eda_platform.schemas.artifacts import Artifact, DatasetProfile
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.pii import pii_labels


class ColumnValueProfile(BaseModel):
    dataset_id: str
    values: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


def top_n_values(
    loaded: LoadedDataset,
    profile_artifact: Artifact,
    pii_artifact: Artifact,
    *,
    top_n: int = 5,
) -> ColumnValueProfile:
    profile = DatasetProfile.model_validate(profile_artifact.payload)
    labels = pii_labels(pii_artifact)
    values: dict[str, list[dict[str, Any]]] = {}
    for column in profile.column_names:
        if column not in loaded.frame.columns:
            continue
        series = loaded.frame[column].dropna()
        if series.empty:
            values[column] = []
            continue
        masked = series.map(
            lambda value, column_name=column: _mask_with_labels(column_name, value, labels)
        )
        counts = masked.value_counts(dropna=True).head(top_n)
        values[column] = [
            {"value": _json_value(value), "count": int(count)}
            for value, count in counts.items()
        ]
    return ColumnValueProfile(dataset_id=profile.dataset_id, values=values)


def _mask_with_labels(column: str, value: object, labels: dict[str, str]) -> object:
    label = labels.get(column)
    if label is None:
        return value
    return f"[PII:{label}]"


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value

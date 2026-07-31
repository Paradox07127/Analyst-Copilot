from __future__ import annotations

import math
from typing import Literal, cast

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import code_ref
from eda_platform.schemas.anomaly import AnomalyOutlier, AnomalyScreenResult
from eda_platform.schemas.artifacts import Artifact, ArtifactType


def screen_anomalies(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    column: str,
    method: Literal["robust_zscore", "iqr"] = "robust_zscore",
    threshold: float | None = None,
) -> AnomalyScreenResult:
    """Deterministically screen one numeric column for robust outliers."""
    if column not in frame.columns:
        raise ValueError(
            f"Column `{column}` was not found. Choose a numeric column from the dataset."
        )
    series = cast(pd.Series, frame[column])
    if not bool(series.notna().any()):
        raise ValueError(
            f"Column `{column}` contains only null values. Choose a column with observed numbers."
        )
    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(
            f"Column `{column}` is not numeric. Choose a numeric measure for anomaly screening."
        )
    if method not in {"robust_zscore", "iqr"}:
        raise ValueError("Method must be `robust_zscore` or `iqr`.")

    working = pd.DataFrame(
        {"value": series.astype(float), "row_index": range(len(frame))}
    ).dropna(subset=["value"])
    values = cast(pd.Series, working["value"]).reset_index(drop=True)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Column `{column}` contains non-finite values. Replace infinities before screening."
        )

    selected_threshold = (
        threshold
        if threshold is not None
        else (3.5 if method == "robust_zscore" else 1.5)
    )
    if not isinstance(selected_threshold, (int, float)) or isinstance(selected_threshold, bool):
        raise ValueError("Threshold must be a positive finite number.")
    selected_threshold = float(selected_threshold)
    if not math.isfinite(selected_threshold) or selected_threshold <= 0:
        raise ValueError("Threshold must be a positive finite number.")

    median = float(values.median())
    deviations = cast(pd.Series, (values - median).abs())
    mad = float(deviations.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    notes: list[str] = []
    result_method = method

    if method == "robust_zscore" and mad > 0:
        scores = cast(pd.Series, 0.6745 * (values - median) / mad)
    elif method == "robust_zscore" and iqr > 0:
        result_method = "iqr"
        notes.append(
            "Median absolute deviation is zero; the deterministic IQR fallback was used."
        )
        scores = _iqr_scores(values, q1=q1, q3=q3, iqr=iqr)
        if threshold is None:
            selected_threshold = 1.5
    elif method == "robust_zscore":
        if values.nunique() <= 1:
            notes.append(
                "Median absolute deviation and IQR are zero because the column is constant; "
                "no individual outliers can be scored."
            )
            scores = pd.Series(0.0, index=values.index, dtype=float)
        else:
            notes.append(
                "Median absolute deviation and IQR are zero for the dominant value; "
                "values different from the median are flagged as sparse spikes."
            )
            direction = cast(
                pd.Series,
                (values - median).map(lambda value: 1.0 if value > 0 else -1.0),
            )
            scores = cast(
                pd.Series,
                direction.where(values != median, 0.0) * (selected_threshold + 1.0),
            )
    elif iqr > 0:
        scores = _iqr_scores(values, q1=q1, q3=q3, iqr=iqr)
    else:
        if values.nunique() <= 1:
            notes.append("IQR is zero because the column is constant.")
            scores = pd.Series(0.0, index=values.index, dtype=float)
        else:
            notes.append(
                "IQR is zero for the dominant value; values different from the median "
                "are flagged as sparse spikes."
            )
            direction = cast(
                pd.Series,
                (values - median).map(lambda value: 1.0 if value > 0 else -1.0),
            )
            scores = cast(
                pd.Series,
                direction.where(values != median, 0.0) * (selected_threshold + 1.0),
            )

    flagged = cast(pd.Series, scores.abs() > selected_threshold)
    outliers = [
        AnomalyOutlier(
            row_index=int(working.iloc[index]["row_index"]),
            value=float(values.iloc[index]),
            score=float(scores.iloc[index]),
        )
        for index in range(len(values))
        if bool(flagged.iloc[index])
    ]
    outliers.sort(key=lambda item: (-abs(item.score), item.row_index))
    outlier_count = int(flagged.sum())
    outlier_percent = 100.0 * outlier_count / len(values)
    if outlier_percent > 20.0:
        notes.append(
            "The flagged share may indicate distribution shift rather than individual anomalies."
        )

    return AnomalyScreenResult(
        dataset_name=dataset_name,
        column=column,
        method=result_method,
        threshold=selected_threshold,
        total_rows=len(frame),
        non_null_rows=len(values),
        outlier_count=outlier_count,
        outlier_percent=outlier_percent,
        median=median,
        mad=mad,
        q1=q1,
        q3=q3,
        top_outliers=outliers[:10],
        notes=notes,
    )


def create_anomaly_artifact(
    result: AnomalyScreenResult,
    *,
    project_id: str,
    session_id: str,
    parents: list[str] | None = None,
) -> Artifact:
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("anomaly", payload),
        type=ArtifactType.ANOMALY_SCREEN_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=parents or [],
        payload=payload,
        code_ref=code_ref(screen_anomalies),
        plain_language=(
            f"{result.method.replace('_', ' ')} screening of {result.column} found "
            f"{result.outlier_count} outliers among {result.non_null_rows} observed rows."
        ),
    )


def _iqr_scores(values: pd.Series, *, q1: float, q3: float, iqr: float) -> pd.Series:
    lower_scores = cast(pd.Series, (values - q1) / iqr)
    upper_scores = cast(pd.Series, (values - q3) / iqr)
    return cast(pd.Series, lower_scores.where(values < q1, upper_scores.where(values > q3, 0.0)))

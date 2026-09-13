"""KMeans segmentation with resampling stability (von Luxburg 2010).

Features are explicitly chosen by the caller, z-score standardized, and
clustered with k-means++; k is picked by mean silhouette over 2..8 when not
given. Stability is measured by re-clustering 20 deterministic 80% subsamples
and comparing each refit partition to the reference with ARI; a mean ARI
below 0.6 marks the segmentation unstable and strips its evidence standing.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_semantic_type,
    raise_for_violations,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.segmentation import (
    ClusterSegment,
    SegmentationResult,
    SegmentFeatureSummary,
    SegmentStability,
)
from eda_platform.tools.ml_baseline import _is_id_like

SEGMENTATION_LIMITATION = (
    "Segments are a descriptive partition of the chosen features, not proof "
    "that distinct real-world groups exist."
)
SEGMENTATION_UNSTABLE_LIMITATION = (
    "The segmentation is unstable under resampling (mean ARI below 0.6): the "
    "partition changes when the data is subsampled, so it must not be used as "
    "hypothesis evidence or as an operational segmentation."
)

_MIN_VALID_ROWS = 30
_MAX_CLUSTER_ROWS = 20_000
_SILHOUETTE_MAX_ROWS = 5_000
_K_MIN, _K_MAX = 2, 8
_N_INIT = 10
STABILITY_RESAMPLES = 20
_SUBSAMPLE_FRACTION = 0.8
# Engineering thresholds, not literature constants: resampling ARI bands for
# calling a partition stable/moderate/unstable (declared in the tool contract).
ARI_UNSTABLE_BELOW = 0.6
ARI_STABLE_FROM = 0.8


def run_segmentation(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    feature_columns: list[str],
    k: int | None = None,
    random_state: int = 42,
) -> SegmentationResult:
    """Cluster the explicitly chosen numeric features and grade the stability."""
    guard_segmentation_params(frame, feature_columns=feature_columns)
    notes: list[str] = []
    valid = frame.loc[:, feature_columns].dropna()
    total_rows = int(len(frame))
    if len(valid) > _MAX_CLUSTER_ROWS:
        keep = np.sort(
            np.random.default_rng(random_state).choice(
                len(valid), size=_MAX_CLUSTER_ROWS, replace=False
            )
        )
        valid = valid.iloc[keep]
        notes.append(
            f"Clustering ran on a deterministic subsample of {_MAX_CLUSTER_ROWS} of "
            f"the {total_rows} rows."
        )
    values = valid.to_numpy(dtype="float64")
    n = int(values.shape[0])

    # z-score with population std (StandardScaler semantics) so no single
    # feature's scale dominates the euclidean distances.
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    def fit(k_value: int, seed: int, data: np.ndarray) -> KMeans:
        return KMeans(
            n_clusters=k_value, init="k-means++", n_init=_N_INIT, random_state=seed
        ).fit(data)

    def score(data: np.ndarray, labels: np.ndarray) -> float:
        sample_size = _SILHOUETTE_MAX_ROWS if data.shape[0] > _SILHOUETTE_MAX_ROWS else None
        return float(
            silhouette_score(
                data, labels, sample_size=sample_size, random_state=random_state
            )
        )

    k_selection: Literal["silhouette", "explicit"]
    if k is not None:
        if k > n - 1:
            raise ValueError(
                f"k={k} needs at least {k + 1} rows with every feature present; "
                f"only {n} are available."
            )
        k_selection = "explicit"
        model = fit(k, random_state, standardized)
        silhouette = score(standardized, model.labels_)
    else:
        k_selection = "silhouette"
        best: tuple[float, int, KMeans] | None = None
        for candidate_k in range(_K_MIN, min(_K_MAX, n - 1) + 1):
            candidate = fit(candidate_k, random_state, standardized)
            candidate_score = score(standardized, candidate.labels_)
            # Strict > keeps ties on the smaller k deterministically.
            if best is None or candidate_score > best[0]:
                best = (candidate_score, candidate_k, candidate)
        assert best is not None  # _MIN_VALID_ROWS > _K_MIN guarantees a candidate
        silhouette, k, model = best[0], best[1], best[2]

    reference_labels = model.labels_
    subsample_size = max(int(round(n * _SUBSAMPLE_FRACTION)), k + 1)
    rng = np.random.default_rng(random_state)
    ari_scores: list[float] = []
    for resample in range(STABILITY_RESAMPLES):
        indices = rng.choice(n, size=subsample_size, replace=False)
        refit = fit(k, random_state + resample + 1, standardized[indices])
        # The refit centroids label every row, so both partitions cover the
        # same points and ARI compares like with like.
        ari_scores.append(
            float(adjusted_rand_score(reference_labels, refit.predict(standardized)))
        )
    ari_mean = float(np.mean(ari_scores))
    ari_min = float(np.min(ari_scores))
    stability: SegmentStability = (
        "unstable"
        if ari_mean < ARI_UNSTABLE_BELOW
        else "moderate"
        if ari_mean < ARI_STABLE_FROM
        else "stable"
    )
    if stability == "unstable":
        notes.append(SEGMENTATION_UNSTABLE_LIMITATION)

    overall_means = values.mean(axis=0)
    clusters: list[ClusterSegment] = []
    for label in range(k):
        member = values[reference_labels == label]
        clusters.append(
            ClusterSegment(
                label=label,
                size=int(member.shape[0]),
                share=round(member.shape[0] / n, 6),
                feature_summary=[
                    SegmentFeatureSummary(
                        feature=feature,
                        mean=round(float(member[:, index].mean()), 6),
                        median=round(float(np.median(member[:, index])), 6),
                        delta_vs_overall_mean=round(
                            float(member[:, index].mean() - overall_means[index]), 6
                        ),
                    )
                    for index, feature in enumerate(feature_columns)
                ],
            )
        )

    return SegmentationResult(
        dataset_name=dataset_name,
        feature_columns=list(feature_columns),
        k=k,
        k_selection=k_selection,
        silhouette=round(silhouette, 6),
        ari_mean=round(ari_mean, 6),
        ari_min=round(ari_min, 6),
        resamples=STABILITY_RESAMPLES,
        stability=stability,
        total_rows=total_rows,
        sample_rows=n,
        clusters=clusters,
        notes=notes,
    )


def guard_segmentation_params(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> None:
    violations: list[GuardViolation | None] = []
    if len(feature_columns) != len(set(feature_columns)):
        violations.append(
            GuardViolation(
                field="feature_columns",
                got=feature_columns,
                allowed="2-10 distinct numeric column names",
                fix_hint="Remove the duplicated column names.",
                problem="feature_columns contains duplicates.",
            )
        )
    for column in feature_columns:
        violation = check_column_semantic_type(
            "feature_columns",
            column,
            frame,
            allowed_semantic_types=["numeric"],
            fix_hint="Choose numeric measure columns for clustering.",
        )
        if violation is not None:
            violations.append(violation)
            continue
        series = cast(pd.Series, frame[column])
        if _is_id_like(series, column):
            violations.append(
                GuardViolation(
                    field="feature_columns",
                    got=column,
                    allowed="numeric measure columns that are not identifiers",
                    fix_hint=f"Drop `{column}`; identifiers separate rows, not segments.",
                    problem="column looks like a row identifier.",
                )
            )
        elif int(series.dropna().nunique()) <= 1:
            violations.append(
                GuardViolation(
                    field="feature_columns",
                    got=column,
                    allowed="numeric columns with at least two distinct values",
                    fix_hint=f"Drop `{column}`; a constant contributes no separation.",
                    problem="column is constant.",
                )
            )
    if not violations:
        valid_rows = int(len(frame.loc[:, feature_columns].dropna()))
        if valid_rows < _MIN_VALID_ROWS:
            violations.append(
                GuardViolation(
                    field="feature_columns",
                    got=f"{valid_rows} rows with every feature present",
                    allowed=f"at least {_MIN_VALID_ROWS} rows with every feature present",
                    fix_hint=(
                        "Choose features with fewer missing values; below "
                        f"{_MIN_VALID_ROWS} rows the stability check is meaningless."
                    ),
                    problem="too few complete rows for a stability-checked segmentation.",
                )
            )
    raise_for_violations("run_segmentation", violations)


def create_segmentation_artifact(
    result: SegmentationResult,
    *,
    project_id: str,
    session_id: str,
) -> Artifact:
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("segmentation", payload),
        type=ArtifactType.SEGMENTATION_RESULT,
        project_id=project_id,
        session_id=session_id,
        payload=payload,
        warnings=list(result.notes),
        plain_language=describe_segmentation(result),
    )


def describe_segmentation(result: SegmentationResult) -> str:
    sizes = ", ".join(
        f"segment {cluster.label}: {cluster.size} rows ({cluster.share:.0%})"
        for cluster in result.clusters
    )
    stability_clause = (
        "The segmentation is unstable under resampling and must not be used as "
        "evidence."
        if result.stability == "unstable"
        else f"Resampling stability is {result.stability} "
        f"(mean ARI {result.ari_mean:.2f} over {result.resamples} subsamples)."
    )
    return (
        f"KMeans split {result.sample_rows} rows into {result.k} segments on "
        f"{', '.join(result.feature_columns)} (mean silhouette "
        f"{result.silhouette:.2f}). {sizes}. {stability_clause} "
        f"{SEGMENTATION_LIMITATION}"
    )

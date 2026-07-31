from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    QualityIssueSet,
)

_PRIMARY_TYPES = {
    ArtifactType.DATASET_PROFILE,
    ArtifactType.QUALITY_ISSUE_SET,
    ArtifactType.QUALITY_CONTEXT_SET,
    ArtifactType.TABLE,
    ArtifactType.STAT_TEST_RESULT,
    ArtifactType.MODEL_CARD,
    ArtifactType.RELATIONSHIP_CANDIDATE_SET,
    ArtifactType.RELATIONSHIP_VALIDATION_SET,
    ArtifactType.ER_DIAGRAM,
    ArtifactType.CLEANING_RECIPE,
    ArtifactType.PII_REPORT,
    ArtifactType.RAW_DATASET_PROFILE,
    ArtifactType.RAW_DATA_PREVIEW,
}
_SINGLETON_KEYS = {
    ArtifactType.DATASET_PROFILE: "profile",
    ArtifactType.QUALITY_ISSUE_SET: "quality",
    ArtifactType.QUALITY_CONTEXT_SET: "quality_context",
    ArtifactType.PII_REPORT: "pii",
    ArtifactType.CLEANING_RECIPE: "cleaning_recipe",
}
_COLLECTION_KEYS = {
    ArtifactType.TABLE: "tables",
    ArtifactType.STAT_TEST_RESULT: "stat_tests",
    ArtifactType.MODEL_CARD: "models",
    ArtifactType.RELATIONSHIP_CANDIDATE_SET: "relationships",
    ArtifactType.RELATIONSHIP_VALIDATION_SET: "relationship_validations",
    ArtifactType.ER_DIAGRAM: "er_diagrams",
}
# Pre-clean artifacts carry the raw dataset's id; when lineage maps them to an
# analysed dataset they land in that dataset's bucket under raw-scoped keys.
_RAW_SINGLETON_KEYS = {
    ArtifactType.RAW_DATASET_PROFILE: "raw_profile",
    ArtifactType.RAW_DATA_PREVIEW: "raw_preview",
    ArtifactType.CLEANING_RECIPE: "cleaning_recipe",
    ArtifactType.PII_REPORT: "raw_pii",
}
_CHART_CATEGORY_ORDER = ("quality", "time", "relationship", "comparison", "distribution")


def create_eda_handoff_artifact(
    artifacts: Sequence[Artifact],
    *,
    project_id: str,
    session_id: str,
    raw_dataset_lineage: Mapping[str, str] | None = None,
) -> Artifact:
    """Build one compact, machine-readable index for downstream analysis agents."""
    raw_to_clean = dict(raw_dataset_lineage or {})
    clean_to_raw = {clean: raw for raw, clean in raw_to_clean.items()}
    profiles = [
        DatasetProfile.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    quality_by_dataset = {
        issue_set.dataset_id: issue_set
        for artifact in artifacts
        if artifact.type is ArtifactType.QUALITY_ISSUE_SET
        for issue_set in [QualityIssueSet.model_validate(artifact.payload)]
    }
    datasets: list[dict[str, Any]] = []
    for profile in profiles:
        issue_set = quality_by_dataset.get(
            profile.dataset_id,
            QualityIssueSet(dataset_id=profile.dataset_id),
        )
        severity_counts = Counter(issue.severity for issue in issue_set.issues)
        material_codes = sorted(
            {
                issue.code
                for issue in issue_set.issues
                if issue.severity in {"critical", "warn"}
            }
        )
        datasets.append(
            {
                "dataset_id": profile.dataset_id,
                "raw_dataset_id": clean_to_raw.get(profile.dataset_id),
                "name": profile.name,
                "content_hash": profile.content_hash,
                "rows": profile.rows,
                "columns": profile.columns,
                "semantic_type_counts": profile.semantic_type_counts,
                "grain": profile.grain,
                "primary_key_candidates": profile.primary_key_candidates,
                "composite_key_candidates": profile.composite_key_candidates,
                "pii_columns": profile.pii_columns,
                "quality": {
                    "critical": severity_counts["critical"],
                    "warn": severity_counts["warn"],
                    "info": severity_counts["info"],
                    "material_codes": material_codes,
                },
                "analysis_ready": profile.rows > 0 and severity_counts["critical"] == 0,
            }
        )
    by_dataset: dict[str, dict[str, Any]] = {
        profile.dataset_id: {} for profile in profiles
    }
    global_artifacts: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.type not in _PRIMARY_TYPES:
            continue
        dataset_id = artifact.payload.get("dataset_id")
        summary = {
            "id": artifact.id,
            "type": artifact.type.value,
            "title": artifact.payload.get("title"),
            "kind": artifact.payload.get("kind"),
        }
        if isinstance(dataset_id, str) and dataset_id in raw_to_clean:
            raw_key = _RAW_SINGLETON_KEYS.get(artifact.type)
            if raw_key:
                by_dataset[raw_to_clean[dataset_id]][raw_key] = artifact.id
                continue
            global_artifacts.append(summary)
            continue
        if not isinstance(dataset_id, str) or dataset_id not in by_dataset:
            global_artifacts.append(summary)
            continue
        bucket = by_dataset[dataset_id]
        singleton_key = _SINGLETON_KEYS.get(artifact.type)
        collection_key = _COLLECTION_KEYS.get(artifact.type)
        if singleton_key:
            bucket[singleton_key] = artifact.id
        elif collection_key:
            bucket.setdefault(collection_key, []).append(summary)

    all_charts = [
        {
            "id": artifact.id,
            "dataset_id": artifact.payload.get("dataset_id"),
            "title": artifact.payload.get("title"),
            "category": artifact.payload.get("category", "distribution"),
        }
        for artifact in artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    ]
    chart_category_counts: dict[str, dict[str, int]] = {}
    recommended_charts: list[dict[str, Any]] = []
    for profile in profiles:
        dataset_charts = [
            chart for chart in all_charts if chart["dataset_id"] == profile.dataset_id
        ]
        chart_category_counts[profile.dataset_id] = dict(
            Counter(str(chart["category"]) for chart in dataset_charts)
        )
        # One chart per analytical purpose is enough for first-pass agent
        # context. The complete ChartSpec inventory remains queryable on demand.
        for category in _CHART_CATEGORY_ORDER:
            chart = next(
                (
                    candidate
                    for candidate in dataset_charts
                    if candidate["category"] == category
                ),
                None,
            )
            if chart is not None:
                recommended_charts.append(chart)
    relationship_materialized = any(
        artifact.type is ArtifactType.RELATIONSHIP_CANDIDATE_SET
        for artifact in artifacts
    )
    payload = {
        "schema_version": 2,
        "datasets": datasets,
        "cross_dataset_relationships": {
            "status": (
                "not_applicable"
                if len(profiles) < 2
                else "materialized"
                if relationship_materialized
                else "deferred"
            ),
            "action": (
                None
                if len(profiles) < 2 or relationship_materialized
                else "Run bounded relationship discovery before cross-table analysis."
            ),
        },
        "artifact_index": {
            "by_dataset": by_dataset,
            "global": global_artifacts,
            "recommended_charts": recommended_charts,
            "chart_category_counts": chart_category_counts,
        },
        "artifact_counts": dict(Counter(artifact.type.value for artifact in artifacts)),
        "usage": (
            "Start with this artifact. Load referenced primary artifacts only when their "
            "dataset, method, or quality condition is relevant. Recommended charts are a "
            "small purpose-balanced subset; query the complete ChartSpec inventory on demand."
        ),
    }
    return Artifact(
        id=make_artifact_id("eda_handoff", payload),
        type=ArtifactType.EDA_HANDOFF,
        project_id=project_id,
        session_id=session_id,
        parents=[artifact.id for artifact in artifacts if artifact.type in _PRIMARY_TYPES],
        payload=payload,
        plain_language=(
            "Compact EDA handoff with dataset readiness, material quality conditions, "
            "and a curated artifact index."
        ),
    )

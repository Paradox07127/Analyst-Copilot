"""Typed segmentation result (method family: segmentation_kmeans).

Cluster profiles stay bounded (k <= 8, features <= 10) per the artifact
granularity policy; every number a claim cites must resolve to this payload.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SegmentStability = Literal["stable", "moderate", "unstable"]


class SegmentFeatureSummary(BaseModel):
    """Per-feature profile of one segment, in original (unstandardized) units."""

    feature: str
    mean: float
    median: float
    delta_vs_overall_mean: float


class ClusterSegment(BaseModel):
    label: int = Field(ge=0)
    size: int = Field(ge=1)
    share: float = Field(gt=0.0, le=1.0)
    feature_summary: list[SegmentFeatureSummary] = Field(min_length=1, max_length=10)


class SegmentationResult(BaseModel):
    """Outcome of standardized KMeans clustering with resampling stability."""

    schema_version: int = 1
    dataset_name: str
    feature_columns: list[str] = Field(min_length=2, max_length=10)
    k: int = Field(ge=2, le=8)
    k_selection: Literal["silhouette", "explicit"]
    silhouette: float = Field(ge=-1.0, le=1.0)
    ari_mean: float = Field(ge=-1.0, le=1.0)
    ari_min: float = Field(ge=-1.0, le=1.0)
    resamples: int = Field(ge=1)
    stability: SegmentStability
    total_rows: int = Field(ge=0)
    sample_rows: int = Field(ge=1)
    clusters: list[ClusterSegment] = Field(min_length=2, max_length=8)
    notes: list[str] = Field(default_factory=list)

    @field_validator("dataset_name")
    @classmethod
    def _dataset_name_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("dataset_name must be non-empty.")

    @model_validator(mode="after")
    def _clusters_partition_the_sample(self) -> SegmentationResult:
        if len(self.clusters) != self.k:
            raise ValueError("clusters must list exactly k segments.")
        if sum(cluster.size for cluster in self.clusters) != self.sample_rows:
            raise ValueError("cluster sizes must sum to sample_rows.")
        return self

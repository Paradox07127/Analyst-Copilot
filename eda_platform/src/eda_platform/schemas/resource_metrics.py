"""Typed resource-policy, preflight, and Auto-EDA runtime telemetry models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ResourceLimitAction = Literal["limited", "reject"]
ResourcePreflightStatus = Literal["accepted", "limited", "rejected"]
ResourceComputeMode = Literal["exact_in_memory", "metadata_only"]
ResourceMeasurement = Literal["exact", "estimated", "unavailable"]
DatasetRole = Literal["analysis", "raw_lineage"]
PeakRssMethod = Literal[
    "getrusage_ru_maxrss",
    "get_process_memory_info_peak_working_set",
    "unavailable",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EdaResourcePolicy(_StrictModel):
    """Server-resolved limits persisted with a job for reproducible execution."""

    max_dataset_count: int = Field(default=20, ge=1, le=10_000)
    max_single_input_bytes: int = Field(default=1 << 30, ge=1)
    max_input_bytes_total: int = Field(default=2 << 30, ge=1)
    max_columns_per_dataset: int = Field(default=512, ge=1)
    max_rows_per_dataset: int = Field(default=10_000_000, ge=1)
    max_working_set_bytes: int = Field(default=2 << 30, ge=1)
    sample_rows: int = Field(default=10_000, ge=1, le=1_000_000)
    max_dataset_workers: int = Field(default=2, ge=1, le=2)
    on_exceed: ResourceLimitAction = "limited"
    frame_estimate_safety_factor: float = Field(default=1.25, ge=1.0, le=10.0)
    held_frame_multiplier: float = Field(default=2.0, ge=0.0, le=20.0)
    active_frame_multiplier: float = Field(default=5.0, ge=0.0, le=50.0)

    @model_validator(mode="after")
    def _total_limit_covers_one_input(self) -> EdaResourcePolicy:
        if self.max_input_bytes_total < self.max_single_input_bytes:
            raise ValueError(
                "max_input_bytes_total must be >= max_single_input_bytes"
            )
        return self


class EdaDatasetEstimate(_StrictModel):
    """Bounded-sample estimate for one source without retaining sample data."""

    role: DatasetRole = "analysis"
    name: str
    file_bytes: int = Field(ge=0)
    columns: int = Field(ge=0)
    sample_rows: int = Field(ge=0)
    sample_complete: bool = False
    sample_frame_deep_bytes: int = Field(ge=0)
    sample_serialized_bytes: int = Field(ge=0)
    frame_expansion_ratio: float = Field(ge=0.0)
    estimated_rows: int = Field(ge=0)
    estimated_frame_deep_bytes: int = Field(ge=0)
    exact_rows: int | None = Field(default=None, ge=0)
    exact_frame_deep_bytes: int | None = Field(default=None, ge=0)

    @property
    def best_rows(self) -> int:
        return self.exact_rows if self.exact_rows is not None else self.estimated_rows

    @property
    def best_frame_deep_bytes(self) -> int:
        return (
            self.exact_frame_deep_bytes
            if self.exact_frame_deep_bytes is not None
            else self.estimated_frame_deep_bytes
        )


class EdaResourcePreflight(_StrictModel):
    """Persistable decision made before expensive full-frame computation."""

    schema_version: int = 1
    status: ResourcePreflightStatus
    phase: Literal["estimated", "verified"] = "estimated"
    compute_mode: ResourceComputeMode
    reason_codes: list[str] = Field(default_factory=list)
    requested_dataset_workers: int = Field(ge=1, le=2)
    effective_dataset_workers: int = Field(ge=0, le=2)
    worker_adjustment_reason: str | None = None
    precleaning_enabled: bool = False
    policy: EdaResourcePolicy
    input_dataset_count: int = Field(ge=0)
    input_bytes_total: int = Field(ge=0)
    input_rows_estimated: int = Field(ge=0)
    input_columns_total: int = Field(ge=0)
    estimated_frame_deep_bytes_total: int = Field(ge=0)
    baseline_peak_rss_bytes: int = Field(ge=0)
    estimated_working_set_bytes: int = Field(ge=0)
    verified_working_set_bytes: int | None = Field(default=None, ge=0)
    datasets: list[EdaDatasetEstimate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _decision_is_consistent(self) -> EdaResourcePreflight:
        if self.effective_dataset_workers > self.requested_dataset_workers:
            raise ValueError("effective_dataset_workers cannot exceed requested workers")
        if self.effective_dataset_workers > self.policy.max_dataset_workers:
            raise ValueError("effective_dataset_workers exceeds policy maximum")
        if self.input_dataset_count != len(self.datasets):
            raise ValueError("input_dataset_count must equal len(datasets)")
        if self.status == "accepted" and self.compute_mode != "exact_in_memory":
            raise ValueError("accepted preflight must use exact_in_memory mode")
        if self.status != "accepted" and self.compute_mode != "metadata_only":
            raise ValueError("limited/rejected preflight must use metadata_only mode")
        return self


class EdaDataFootprint(_StrictModel):
    """Aggregate shape and memory footprint for analysis or raw-lineage frames."""

    dataset_count: int = Field(default=0, ge=0)
    file_bytes: int = Field(default=0, ge=0)
    rows: int = Field(default=0, ge=0)
    columns: int = Field(default=0, ge=0)
    max_rows: int = Field(default=0, ge=0)
    max_columns: int = Field(default=0, ge=0)
    frame_deep_bytes: int = Field(default=0, ge=0)
    measurement: ResourceMeasurement = "unavailable"


class EdaInputMetrics(_StrictModel):
    analysis: EdaDataFootprint = Field(default_factory=EdaDataFootprint)
    raw_lineage: EdaDataFootprint = Field(default_factory=EdaDataFootprint)
    unique_file_bytes: int = Field(default=0, ge=0)


class EdaMemoryMetrics(_StrictModel):
    baseline_peak_rss_bytes: int | None = Field(default=None, ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    peak_rss_delta_bytes: int | None = Field(default=None, ge=0)
    peak_rss_method: PeakRssMethod = "unavailable"
    working_set_budget_bytes: int = Field(default=0, ge=0)
    estimated_working_set_bytes: int = Field(default=0, ge=0)
    verified_working_set_bytes: int | None = Field(default=None, ge=0)


class EdaArtifactMetrics(_StrictModel):
    artifact_count: int = Field(default=0, ge=0)
    storage_bytes_excluding_session_metrics: int = Field(default=0, ge=0)
    canonical_json_bytes_excluding_session_metrics: int = Field(default=0, ge=0)
    agent_handoff_payload_bytes: int = Field(default=0, ge=0)
    default_context_bytes: int = Field(default=0, ge=0)
    default_context_estimated_tokens: int = Field(default=0, ge=0)


class AutoEdaResourceUsage(_StrictModel):
    """Nested SessionMetrics v6 payload; defaults keep legacy runs readable."""

    measurement_status: Literal["verified", "partial", "unavailable"] = "unavailable"
    wall_duration_seconds: float = Field(default=0.0, ge=0.0)
    preprocessing_duration_seconds: float = Field(default=0.0, ge=0.0)
    ingest_duration_seconds: float = Field(default=0.0, ge=0.0)
    processing_mode: Literal[
        "exact_in_memory", "metadata_only", "unknown"
    ] = "unknown"
    preflight_status: Literal[
        "accepted", "limited", "rejected", "unavailable"
    ] = "unavailable"
    requested_dataset_workers: int = Field(default=1, ge=1, le=2)
    effective_dataset_workers: int = Field(default=0, ge=0, le=2)
    worker_adjustment_reason: str | None = None
    inputs: EdaInputMetrics = Field(default_factory=EdaInputMetrics)
    memory: EdaMemoryMetrics = Field(default_factory=EdaMemoryMetrics)
    artifacts: EdaArtifactMetrics = Field(default_factory=EdaArtifactMetrics)

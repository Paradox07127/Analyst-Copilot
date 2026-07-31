from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FreshnessStatus = Literal["fresh", "stale", "unverifiable"]


class FindingFreshness(BaseModel):
    """Read-only verdict on whether a persisted finding is still reusable."""

    schema_version: int = 1
    finding_artifact_id: str
    status: FreshnessStatus
    reasons: list[str] = Field(default_factory=list)
    checked_dataset_names: list[str] = Field(default_factory=list)


class DecisionReportFreshness(BaseModel):
    """Aggregate freshness of every content-addressed finding in a report."""

    schema_version: int = 1
    report_artifact_id: str
    status: FreshnessStatus
    finding_statuses: dict[str, FreshnessStatus] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)

"""Shadow-only exploration projections written outside the product fact store."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eda_platform.schemas.insights import InsightRecord

_EXPLORATION_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_EXPLORATION_ID_LENGTH = 128


def validate_exploration_id(value: str) -> str:
    """Return a path-safe exploration id for every shadow entry point."""
    if not isinstance(value, str):
        raise ValueError("exploration_id must be a string.")
    if not value or len(value) > _MAX_EXPLORATION_ID_LENGTH:
        raise ValueError(
            f"exploration_id must contain 1-{_MAX_EXPLORATION_ID_LENGTH} characters."
        )
    if (
        value in {".", ".."}
        or ".." in value
        or "/" in value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError(
            "exploration_id cannot be absolute or contain '..' or path separators."
        )
    if _EXPLORATION_ID.fullmatch(value) is None:
        raise ValueError("exploration_id contains unsupported path characters.")
    return value


class ShadowExplorationProjection(BaseModel):
    """A terminal, rebuildable evaluation projection; the journal is authoritative.

    ``insight_records`` accepts only reducer-produced records whose ClaimBundles
    have already passed the deterministic gates. Intermediate hypotheses, raw
    model payloads, and product artifact references do not belong in this sink.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration_id: str = Field(min_length=1, max_length=_MAX_EXPLORATION_ID_LENGTH)
    last_seq: int = Field(ge=0)
    status: Literal["running", "pause_requested", "paused", "stopped"]
    stop_reason: str | None = None
    policy_fingerprint: str = Field(min_length=1)
    data_state_witness: str = Field(min_length=1)
    insight_records: tuple[InsightRecord, ...] = ()
    coverage_completed: tuple[str, ...] = ()
    coverage_unexplored: tuple[str, ...] = ()
    user_visible: Literal[False] = False
    production_artifact_ids: tuple[()] = ()
    projected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("exploration_id")
    @classmethod
    def _path_safe_exploration_id(cls, value: str) -> str:
        return validate_exploration_id(value)

    @model_validator(mode="after")
    def _validate_terminal_projection(self) -> ShadowExplorationProjection:
        if (self.stop_reason is not None) != (self.status == "stopped"):
            raise ValueError("stop_reason must be set exactly for a stopped projection.")
        if self.insight_records and self.status != "stopped":
            raise ValueError(
                "insight_records may be projected only after the exploration has stopped."
            )
        insight_ids = [record.insight_id for record in self.insight_records]
        if len(insight_ids) != len(set(insight_ids)):
            raise ValueError("insight_records must contain unique insight ids.")
        for record in self.insight_records:
            _validate_proof_projection(record)
        return self


def _validate_proof_projection(record: InsightRecord) -> None:
    supporting = set(record.supporting_receipt_ids)
    contradicting = set(record.contradicting_receipt_ids)
    proved: set[str] = set()
    proof_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for proof in record.proof:
        if proof.comparison == "supports" and proof.receipt_id not in supporting:
            raise ValueError(
                f"insight {record.insight_id!r} has support proof for an uncited receipt."
            )
        if proof.comparison == "contradicts" and proof.receipt_id not in contradicting:
            raise ValueError(
                f"insight {record.insight_id!r} has contradiction proof for an uncited receipt."
            )
        key = (proof.receipt_id, proof.comparison, proof.fact_ids)
        if key in proof_keys:
            raise ValueError(f"insight {record.insight_id!r} contains duplicate proof edges.")
        proof_keys.add(key)
        proved.add(proof.receipt_id)
    cited = supporting | contradicting
    if proved != cited:
        raise ValueError(
            f"insight {record.insight_id!r} proof receipts must exactly match cited receipts."
        )

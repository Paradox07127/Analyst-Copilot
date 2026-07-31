from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    dataset_id: str
    name: str
    path: Path
    content_hash: str
    version: int = 1
    parent_dataset_id: str | None = None
    parent_version: int | None = None
    lineage_recipe_id: str | None = None
    encoding: str = "utf-8"
    delimiter: str = ","
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

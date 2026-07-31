from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Cardinality = Literal["one_to_one", "many_to_one", "one_to_many", "many_to_many"]


class RelationshipColumnPair(BaseModel):
    """One directed join hypothesis: left (FK side) -> right (PK side)."""

    left_dataset_id: str
    left_dataset_name: str
    left_columns: list[str]
    right_dataset_id: str
    right_dataset_name: str
    right_columns: list[str]

    def label(self) -> str:
        left = f"{self.left_dataset_name}.{'+'.join(self.left_columns)}"
        right = f"{self.right_dataset_name}.{'+'.join(self.right_columns)}"
        return f"{left} -> {right}"


class RelationshipSignals(BaseModel):
    """Deterministic candidate signals (v2-plan §4.9); no LLM involvement."""

    name_similarity: float = Field(ge=0.0, le=1.0)
    type_compatible: bool
    overlap_left_in_right: float = Field(ge=0.0, le=1.0, description="|A∩B| / |A|")
    overlap_right_in_left: float = Field(ge=0.0, le=1.0, description="|A∩B| / |B|")
    right_unique_rate: float = Field(ge=0.0, le=1.0)
    left_null_rate: float = Field(ge=0.0, le=1.0)
    right_null_rate: float = Field(ge=0.0, le=1.0)
    format_fingerprint_match: bool = True
    sampled: bool = False


class RelationshipCandidate(BaseModel):
    pair: RelationshipColumnPair
    signals: RelationshipSignals
    ensemble_score: float = Field(ge=0.0, le=1.0)
    confidence: Confidence
    auto_adopted: bool = False


class RelationshipCandidateSet(BaseModel):
    """All scored candidates for one run; artifact prefix `relcand`."""

    dataset_ids: list[str] = Field(default_factory=list)
    candidates: list[RelationshipCandidate] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    truncated_pairs: int = 0
    overlap_pairs_evaluated: int = 0
    overlap_pairs_prefiltered: int = 0
    coverage_status: Literal["complete", "limited"] = "complete"
    coverage_reason: str = ""


class RelationshipValidation(BaseModel):
    """DuckDB-verified join facts; numbers never come from an LLM."""

    pair: RelationshipColumnPair
    join_row_multiplier: float
    orphan_rate_left: float = Field(ge=0.0, le=1.0)
    orphan_rate_right: float = Field(ge=0.0, le=1.0)
    cardinality: Cardinality
    verified: bool
    verification_sql: str
    sampled: bool = False
    warnings: list[str] = Field(default_factory=list)


class RelationshipValidationSet(BaseModel):
    """Validations for medium/high candidates; artifact prefix `relval`."""

    validations: list[RelationshipValidation] = Field(default_factory=list)


class ErRelationRow(BaseModel):
    """Degraded tabular rendering of one edge (for HTML export)."""

    left: str
    right: str
    cardinality: str
    confidence: str
    note: str = ""


class ErDiagram(BaseModel):
    """Graphviz DOT source plus tabular fallback; artifact prefix `erd`."""

    dot_source: str
    relations: list[ErRelationRow] = Field(default_factory=list)

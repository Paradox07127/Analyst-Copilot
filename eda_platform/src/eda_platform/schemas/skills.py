"""AnalysisSkill — a named, replayable snapshot of a validated ``AnalysisPlan``.

M6.2 P2 (ADR D8). Once the user blesses an analysis plan, we freeze it into a
named *skill*: the exact plan that was validated (its ``sql``/``method``/
``rationale``), plus the two facts replay needs to retarget it — the columns it
reads (``param_columns``) and the datasets it referenced (``expected_datasets``).
Replaying a skill re-runs that frozen plan against new datasets through the same
deterministic DuckDB permission gate, no LLM required. See
docs/archive/2026-07/base/eda-agent-platform-m6.2-plan.md §3.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from eda_platform.schemas.plans import AnalysisPlan


class AnalysisSkill(BaseModel):
    """A validated ``AnalysisPlan`` promoted to a named, reusable skill."""

    skill_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    plan: AnalysisPlan
    # Columns the frozen plan reads; every one must exist in a target dataset
    # before replay is allowed (checked with the shared tool_guard column gate).
    param_columns: list[str] = Field(default_factory=list)
    # Relation/dataset names the plan referenced; mapped onto the replay targets.
    expected_datasets: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_session_id: str | None = None

    @field_validator("name")
    @classmethod
    def _name_is_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("name must be non-empty.")


# Seed-library templates: parameterized analysis patterns that become concrete
# AnalysisSkills only after the user binds each placeholder to a real column
# (the replay gate requires literal column names, so seeds cannot replay as-is).

SeedParamRole = Literal["measure", "dimension", "timestamp", "identifier", "any"]

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SeedParam(BaseModel):
    """One placeholder of a seed template and the column role it expects."""

    name: str
    role: SeedParamRole
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_placeholder_token(cls, value: str) -> str:
        if _PARAM_NAME_RE.match(value):
            return value
        raise ValueError("param name must be a lowercase identifier (e.g. 'group_col').")


class SeedSkillTemplate(BaseModel):
    """A parameterized SQL analysis pattern shipped with the platform."""

    seed_id: str
    name: str
    question: str
    sql: str
    method: str
    rationale: str
    params: list[SeedParam] = Field(min_length=1)
    source_url: str = ""

    @field_validator("seed_id", "name", "question", "sql", "method", "rationale")
    @classmethod
    def _required_strings_are_non_empty(cls, value: str) -> str:
        if value.strip():
            return value
        raise ValueError("field must be non-empty.")

    @model_validator(mode="after")
    def _placeholders_match_params(self) -> SeedSkillTemplate:
        param_names = [param.name for param in self.params]
        if len(param_names) != len(set(param_names)):
            raise ValueError("param names must be unique.")
        if "dataset" in param_names:
            # {dataset} is bound to the target relation at instantiation; a
            # param of that name would silently collide with the substitution.
            raise ValueError("'dataset' is a reserved placeholder name; rename the param.")
        allowed = set(param_names) | {"dataset"}
        sql_placeholders = set(_PLACEHOLDER_RE.findall(self.sql))
        if "dataset" not in sql_placeholders:
            raise ValueError("sql must reference the {dataset} placeholder.")
        if sql_placeholders != allowed:
            raise ValueError(
                "sql placeholders must exactly match the declared params plus {dataset}; "
                f"got {sorted(sql_placeholders)}, declared {sorted(allowed)}."
            )
        question_placeholders = set(_PLACEHOLDER_RE.findall(self.question))
        if not question_placeholders <= allowed:
            raise ValueError(
                "question placeholders must be declared params or {dataset}; "
                f"got {sorted(question_placeholders - allowed)}."
            )
        return self

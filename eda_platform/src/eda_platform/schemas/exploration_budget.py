"""Budget schema contract for exploration runs (plan §4.2, R3.2).

Frozen shape shared by the journal (policy fingerprint) and the budget
runtime. Composes the existing core.budget.SessionBudgetPolicy for the LLM
dimensions instead of forking it; amendments may only raise caps.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from eda_platform.core.budget import SessionBudgetPolicy

_LIMIT_DIMENSIONS = ("requests", "input_tokens", "output_tokens", "total_tokens")


class SessionBudgetPolicyModel(BaseModel):
    """Pydantic mirror of core.budget.SessionBudgetPolicy (field-for-field)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_requests: int | None = Field(default=None, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: Decimal | None = Field(default=None, gt=0)
    max_wall_seconds: float | None = Field(default=None, gt=0)
    protected_requests: int = Field(default=0, ge=0)
    protected_input_tokens: int = Field(default=0, ge=0)
    protected_output_tokens: int = Field(default=0, ge=0)
    protected_total_tokens: int = Field(default=0, ge=0)
    protected_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    unknown_usage_policy: Literal["consume_reservation", "reject"] = "consume_reservation"

    @model_validator(mode="after")
    def _protected_within_caps(self) -> SessionBudgetPolicyModel:
        pairs = [
            (name, getattr(self, f"protected_{name}"), getattr(self, f"max_{name}"))
            for name in _LIMIT_DIMENSIONS
        ]
        pairs.append(("cost_usd", self.protected_cost_usd, self.max_cost_usd))
        for dimension, protected, maximum in pairs:
            if protected and maximum is None:
                raise ValueError(f"protected_{dimension} requires a max_{dimension} limit.")
            if maximum is not None and protected > maximum:
                raise ValueError(f"protected_{dimension} cannot exceed max_{dimension}.")
        return self

    def to_policy(self) -> "SessionBudgetPolicy":
        from eda_platform.core.budget import SessionBudgetPolicy

        return SessionBudgetPolicy(**self.model_dump())


class ExplorationBudgetPolicy(BaseModel):
    """Multi-dimensional hard limits for one exploration run (plan §4.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm: SessionBudgetPolicyModel
    max_successful_tool_calls: int = Field(ge=1)
    max_tool_calls_by_kind: dict[str, int] = Field(min_length=1)
    max_rows_scanned: int | None = Field(default=None, ge=1)
    max_result_cells: int | None = Field(default=None, ge=1)
    idle_timeout_seconds: float = Field(gt=0)
    max_rounds: int = Field(ge=1)

    @model_validator(mode="after")
    def _per_kind_caps_positive(self) -> ExplorationBudgetPolicy:
        for kind, cap in self.max_tool_calls_by_kind.items():
            if not kind or cap < 1:
                raise ValueError(
                    "max_tool_calls_by_kind requires non-empty kinds with caps >= 1."
                )
        return self


class BudgetCapIncrease(BaseModel):
    """Strictly additive deltas; a zero increase across the board is invalid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_requests: int = Field(default=0, ge=0)
    max_input_tokens: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=0, ge=0)
    max_total_tokens: int = Field(default=0, ge=0)
    max_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    max_wall_seconds: float = Field(default=0.0, ge=0)
    max_successful_tool_calls: int = Field(default=0, ge=0)
    max_tool_calls_by_kind: dict[str, int] = Field(default_factory=dict)
    max_rows_scanned: int = Field(default=0, ge=0)
    max_result_cells: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _at_least_one_increase(self) -> BudgetCapIncrease:
        for kind, delta in self.max_tool_calls_by_kind.items():
            if not kind or delta < 1:
                raise ValueError(
                    "per-kind increases require non-empty kinds with deltas >= 1."
                )
        numeric = (
            self.max_requests,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_total_tokens,
            self.max_cost_usd,
            Decimal(str(self.max_wall_seconds)),
            self.max_successful_tool_calls,
            self.max_rows_scanned,
            self.max_result_cells,
            self.max_rounds,
        )
        if all(value == 0 for value in numeric) and not self.max_tool_calls_by_kind:
            raise ValueError("an amendment must raise at least one cap.")
        return self


class BudgetAmendment(BaseModel):
    """One monotonic link in the amendment chain (R3.2: caps only go up)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_id: str = Field(min_length=1)
    previous_effective_fingerprint: str = Field(min_length=1)
    increase: BudgetCapIncrease
    reason: str = Field(min_length=1, max_length=2000)
    approved_by: str = Field(min_length=1)
    created_at: str = Field(min_length=1)

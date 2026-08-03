"""Exploration budget runtime: success-counted tool ledger, monotonic amendment
chain, and finalization-reserve view (plan §4.2/§5.4, R3.2/R3.4). Wall clock and
idle timeouts belong to the journal/supervisor; LLM dimensions stay in
core.budget.SessionBudgetState."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from threading import RLock
from typing import Any, Literal

from pydantic import ValidationError

from eda_platform.core.budget import BudgetDimension, BudgetExceeded, SessionBudgetState
from eda_platform.core.ids import stable_hash
from eda_platform.schemas.exploration import ExplorationLoopState
from eda_platform.schemas.exploration_budget import (
    BudgetAmendment,
    BudgetCapIncrease,
    ExplorationBudgetPolicy,
    SessionBudgetPolicyModel,
)

type ToolBudgetDimension = Literal[
    "successful_tool_calls",
    "tool_calls_by_kind",
    "rows_scanned",
    "result_cells",
]

_SNAPSHOT_KEYS = frozenset(
    {"successful_tool_calls", "calls_by_kind", "rows_scanned", "result_cells"}
)
_RESERVABLE_LLM_DIMENSIONS: tuple[BudgetDimension, ...] = (
    "requests",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
)


class ToolBudgetExceeded(BudgetExceeded):
    """Raised when a batch pre-check or a settlement crosses an exploration tool cap."""

    def __init__(
        self,
        dimension: ToolBudgetDimension,
        *,
        limit: int,
        attempted: int,
        kind: str | None = None,
        stage: Literal["check", "settlement"] = "check",
        latched: bool = False,
    ) -> None:
        self.dimension = dimension
        self.limit = limit
        self.attempted = attempted
        self.kind = kind
        self.stage = stage
        self.latched = latched
        kind_suffix = f" for kind {kind!r}" if kind is not None else ""
        super().__init__(
            f"Exploration {dimension} budget exhausted{kind_suffix}: {attempted} > {limit}."
        )


class BudgetAmendmentError(ValueError):
    """Raised when an amendment chain is broken, replayed, forged, or lowers a cap."""


def _validate_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


@dataclass(frozen=True, slots=True)
class ToolCallProjection:
    """Worst-case projection of one tool call for the batch pre-check."""

    kind: str
    rows_scanned: int = 0
    result_cells: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a non-empty string.")
        _validate_count("rows_scanned", self.rows_scanned)
        _validate_count("result_cells", self.result_cells)


class ToolCallLedger:
    """Success-counted tool budget: failed calls consume nothing, and a dimension
    that reaches its cap latches until an amendment raises the cap."""

    def __init__(self, policy: ExplorationBudgetPolicy) -> None:
        self._policy = policy
        self._successful_calls = 0
        self._calls_by_kind: dict[str, int] = {}
        self._rows_scanned = 0
        self._result_cells = 0
        self._lock = RLock()

    @property
    def policy(self) -> ExplorationBudgetPolicy:
        return self._policy

    def check_batch(self, projected: Sequence[ToolCallProjection]) -> None:
        """Projection-only pre-check: rejects the whole batch, commits nothing."""
        with self._lock:
            calls = 0
            rows = 0
            cells = 0
            by_kind: dict[str, int] = {}
            for item in projected:
                if not isinstance(item, ToolCallProjection):
                    raise TypeError("check_batch expects ToolCallProjection items.")
                calls += 1
                rows += item.rows_scanned
                cells += item.result_cells
                by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
            self._enforce(
                "successful_tool_calls",
                used=self._successful_calls,
                increment=calls,
                limit=self._policy.max_successful_tool_calls,
            )
            for kind, count in by_kind.items():
                # A kind absent from the policy dict has cap 0: fail closed.
                self._enforce(
                    "tool_calls_by_kind",
                    used=self._calls_by_kind.get(kind, 0),
                    increment=count,
                    limit=self._policy.max_tool_calls_by_kind.get(kind, 0),
                    kind=kind,
                )
            self._enforce(
                "rows_scanned",
                used=self._rows_scanned,
                increment=rows,
                limit=self._policy.max_rows_scanned,
            )
            self._enforce(
                "result_cells",
                used=self._result_cells,
                increment=cells,
                limit=self._policy.max_result_cells,
            )

    def record_success(self, kind: str, *, rows_scanned: int = 0, result_cells: int = 0) -> None:
        """Commit actual usage of one successful call, then enforce (settle semantics)."""
        actual = ToolCallProjection(kind=kind, rows_scanned=rows_scanned, result_cells=result_cells)
        with self._lock:
            self._successful_calls += 1
            self._calls_by_kind[actual.kind] = self._calls_by_kind.get(actual.kind, 0) + 1
            self._rows_scanned += actual.rows_scanned
            self._result_cells += actual.result_cells
            self._enforce_used(kind=actual.kind)

    def record_failure_usage(
        self,
        kind: str,
        *,
        rows_scanned: int = 0,
        result_cells: int = 0,
    ) -> None:
        """Settle measurable resource use without charging the success counters.

        A tool that scans data and then fails must not be free to repeat the
        same expensive failure until the run ends. Callers use their projected
        usage when an executed tool cannot report a more precise measurement.
        """
        actual = ToolCallProjection(
            kind=kind,
            rows_scanned=rows_scanned,
            result_cells=result_cells,
        )
        with self._lock:
            self._rows_scanned += actual.rows_scanned
            self._result_cells += actual.result_cells
            self._enforce_resource_usage()

    def exhausted_dimensions(self) -> tuple[str, ...]:
        """Latched dimensions, derived from counts vs the current policy."""
        with self._lock:
            latched: list[str] = []
            if self._successful_calls >= self._policy.max_successful_tool_calls:
                latched.append("successful_tool_calls")
            kinds = set(self._policy.max_tool_calls_by_kind) | set(self._calls_by_kind)
            for kind in sorted(kinds):
                cap = self._policy.max_tool_calls_by_kind.get(kind, 0)
                if self._calls_by_kind.get(kind, 0) >= cap:
                    latched.append(f"tool_calls_by_kind:{kind}")
            rows_cap = self._policy.max_rows_scanned
            if rows_cap is not None and self._rows_scanned >= rows_cap:
                latched.append("rows_scanned")
            cells_cap = self._policy.max_result_cells
            if cells_cap is not None and self._result_cells >= cells_cap:
                latched.append("result_cells")
            return tuple(latched)

    def remaining(self) -> dict[str, Any]:
        """Remaining tool budget for the ORIENT countdown; None means unlimited."""
        with self._lock:
            rows_cap = self._policy.max_rows_scanned
            cells_cap = self._policy.max_result_cells
            return {
                "successful_tool_calls": max(
                    0, self._policy.max_successful_tool_calls - self._successful_calls
                ),
                "tool_calls_by_kind": {
                    kind: max(0, cap - self._calls_by_kind.get(kind, 0))
                    for kind, cap in sorted(self._policy.max_tool_calls_by_kind.items())
                },
                "rows_scanned": (
                    None if rows_cap is None else max(0, rows_cap - self._rows_scanned)
                ),
                "result_cells": (
                    None if cells_cap is None else max(0, cells_cap - self._result_cells)
                ),
            }

    def update_policy(self, new_policy: ExplorationBudgetPolicy) -> None:
        """Adopt an amended policy; anything but a monotonic cap raise fails closed."""
        with self._lock:
            _verify_monotonic(self._policy, new_policy)
            self._policy = new_policy

    def snapshot(self) -> dict[str, Any]:
        """Counts-only snapshot for the journal reducer; latches are derived state."""
        with self._lock:
            return {
                "successful_tool_calls": self._successful_calls,
                "calls_by_kind": dict(sorted(self._calls_by_kind.items())),
                "rows_scanned": self._rows_scanned,
                "result_cells": self._result_cells,
            }

    @classmethod
    def restore(
        cls, policy: ExplorationBudgetPolicy, snapshot: Mapping[str, Any]
    ) -> ToolCallLedger:
        """Rebuild from a journal-held snapshot; corrupt snapshots fail closed."""
        if set(snapshot) != _SNAPSHOT_KEYS:
            raise ValueError(f"snapshot keys must be exactly {sorted(_SNAPSHOT_KEYS)}.")
        for name in ("successful_tool_calls", "rows_scanned", "result_cells"):
            _validate_count(name, snapshot[name])
        calls_by_kind = snapshot["calls_by_kind"]
        if not isinstance(calls_by_kind, Mapping):
            raise ValueError("calls_by_kind must be a mapping.")
        validated: dict[str, int] = {}
        for kind, count in calls_by_kind.items():
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError("calls_by_kind keys must be non-empty strings.")
            _validate_count(f"calls_by_kind[{kind}]", count)
            validated[kind] = count
        if sum(validated.values()) != snapshot["successful_tool_calls"]:
            raise ValueError("calls_by_kind must sum to successful_tool_calls.")
        ledger = cls(policy)
        ledger._successful_calls = snapshot["successful_tool_calls"]
        ledger._calls_by_kind = validated
        ledger._rows_scanned = snapshot["rows_scanned"]
        ledger._result_cells = snapshot["result_cells"]
        return ledger

    @classmethod
    def restore_from_journal_state(
        cls,
        policy: ExplorationBudgetPolicy,
        state: ExplorationLoopState,
    ) -> ToolCallLedger:
        """Rebuild every durable tool dimension from the journal projection."""
        expected_caps = (
            policy.max_successful_tool_calls,
            policy.max_tool_calls_by_kind,
            policy.max_rows_scanned,
            policy.max_result_cells,
        )
        journal_caps = (
            state.max_successful_tool_calls,
            state.max_tool_calls_by_kind,
            state.max_rows_scanned,
            state.max_result_cells,
        )
        if expected_caps != journal_caps:
            raise ValueError(
                "effective tool policy does not match the journal amendment chain."
            )
        return cls.restore(
            policy,
            {
                "successful_tool_calls": state.tool_calls_committed,
                "calls_by_kind": state.tool_calls_by_kind,
                "rows_scanned": state.rows_scanned,
                "result_cells": state.result_cells,
            },
        )

    def restore_durable_state(self, state: ExplorationLoopState) -> None:
        """Refresh an already-wired ledger after journal crash recovery settles work."""
        restored = self.restore_from_journal_state(self._policy, state)
        with self._lock:
            self._successful_calls = restored._successful_calls
            self._calls_by_kind = dict(restored._calls_by_kind)
            self._rows_scanned = restored._rows_scanned
            self._result_cells = restored._result_cells

    def _enforce(
        self,
        dimension: ToolBudgetDimension,
        *,
        used: int,
        increment: int,
        limit: int | None,
        kind: str | None = None,
    ) -> None:
        if limit is None or increment == 0:
            return
        if used >= limit:
            raise ToolBudgetExceeded(
                dimension, limit=limit, attempted=used + increment, kind=kind, latched=True
            )
        if used + increment > limit:
            raise ToolBudgetExceeded(dimension, limit=limit, attempted=used + increment, kind=kind)

    def _enforce_used(self, *, kind: str) -> None:
        checks: tuple[tuple[ToolBudgetDimension, int, int | None, str | None], ...] = (
            (
                "successful_tool_calls",
                self._successful_calls,
                self._policy.max_successful_tool_calls,
                None,
            ),
            (
                "tool_calls_by_kind",
                self._calls_by_kind.get(kind, 0),
                self._policy.max_tool_calls_by_kind.get(kind, 0),
                kind,
            ),
            ("rows_scanned", self._rows_scanned, self._policy.max_rows_scanned, None),
            ("result_cells", self._result_cells, self._policy.max_result_cells, None),
        )
        for dimension, used, limit, used_kind in checks:
            if limit is not None and used > limit:
                raise ToolBudgetExceeded(
                    dimension,
                    limit=limit,
                    attempted=used,
                    kind=used_kind,
                    stage="settlement",
                    latched=True,
                )

    def _enforce_resource_usage(self) -> None:
        checks: tuple[tuple[ToolBudgetDimension, int, int | None], ...] = (
            ("rows_scanned", self._rows_scanned, self._policy.max_rows_scanned),
            ("result_cells", self._result_cells, self._policy.max_result_cells),
        )
        for dimension, used, limit in checks:
            if limit is not None and used > limit:
                raise ToolBudgetExceeded(
                    dimension,
                    limit=limit,
                    attempted=used,
                    stage="settlement",
                    latched=True,
                )


def policy_fingerprint(policy: ExplorationBudgetPolicy) -> str:
    """Deterministic digest over every budget field (R3.2: approvals/resume bind to it)."""
    return stable_hash(policy.model_dump(mode="json"))


def amendment_chain_digest(chain: Sequence[BudgetAmendment]) -> str:
    """Order-sensitive digest of the full amendment chain."""
    return stable_hash([amendment.model_dump(mode="json") for amendment in chain])


def effective_policy(
    base: ExplorationBudgetPolicy, chain: Sequence[BudgetAmendment]
) -> ExplorationBudgetPolicy:
    """Apply a monotonic amendment chain; broken/replayed/lowering links fail closed."""
    current = base
    fingerprint = policy_fingerprint(base)
    seen: set[str] = set()
    for link in chain:
        if not isinstance(link, BudgetAmendment):
            raise BudgetAmendmentError("chain entries must be BudgetAmendment instances.")
        try:
            # Re-validate: model_construct-forged links must not bypass schema rules.
            amendment = BudgetAmendment.model_validate(link.model_dump())
        except ValidationError as exc:
            raise BudgetAmendmentError(f"amendment failed schema validation: {exc}") from exc
        if amendment.amendment_id in seen:
            raise BudgetAmendmentError(
                f"amendment_id {amendment.amendment_id!r} replayed in chain."
            )
        seen.add(amendment.amendment_id)
        if amendment.previous_effective_fingerprint != fingerprint:
            raise BudgetAmendmentError(
                f"amendment {amendment.amendment_id!r} expects fingerprint "
                f"{amendment.previous_effective_fingerprint!r} but the chain is at "
                f"{fingerprint!r}."
            )
        amended = _apply_increase(current, amendment.increase)
        _verify_monotonic(current, amended)
        current = amended
        fingerprint = policy_fingerprint(current)
    return current


def apply_budget_increase(
    policy: ExplorationBudgetPolicy,
    increase: BudgetCapIncrease,
) -> ExplorationBudgetPolicy:
    """Apply one schema-validated, monotonic increase to an effective policy.

    Exploration journal events use the run's policy-fingerprint chain rather
    than the standalone budget-chain fingerprint.  Resume composition therefore
    replays their already-validated ``BudgetCapIncrease`` values through this
    narrow public helper instead of reaching into the private implementation.
    """
    validated = BudgetCapIncrease.model_validate(increase.model_dump())
    amended = _apply_increase(policy, validated)
    _verify_monotonic(policy, amended)
    return amended


def finalization_view(
    policy: ExplorationBudgetPolicy, state: SessionBudgetState | None = None
) -> dict[str, dict[str, int | float | Decimal | None]]:
    """Per LLM dimension: the configured finalization reserve, and — given the live
    state — how much protected (finalization) work can still spend (R3.4)."""
    view: dict[str, dict[str, int | float | Decimal | None]] = {}
    llm = policy.llm
    for dimension in _RESERVABLE_LLM_DIMENSIONS:
        attr = "protected_cost_usd" if dimension == "cost_usd" else f"protected_{dimension}"
        remaining = state.remaining(dimension, protected=True) if state is not None else None
        view[dimension] = {"reserved": getattr(llm, attr), "remaining": remaining}
    return view


def _raised[T: (int, float, Decimal)](cap: T | None, delta: T) -> T | None:
    """None means unlimited; raising an unlimited cap keeps it unlimited."""
    if cap is None:
        return None
    return cap + delta


def _apply_increase(
    policy: ExplorationBudgetPolicy, increase: BudgetCapIncrease
) -> ExplorationBudgetPolicy:
    llm = policy.llm
    amended_llm = SessionBudgetPolicyModel(
        max_requests=_raised(llm.max_requests, increase.max_requests),
        max_input_tokens=_raised(llm.max_input_tokens, increase.max_input_tokens),
        max_output_tokens=_raised(llm.max_output_tokens, increase.max_output_tokens),
        max_total_tokens=_raised(llm.max_total_tokens, increase.max_total_tokens),
        max_cost_usd=_raised(llm.max_cost_usd, increase.max_cost_usd),
        max_wall_seconds=_raised(llm.max_wall_seconds, increase.max_wall_seconds),
        protected_requests=llm.protected_requests,
        protected_input_tokens=llm.protected_input_tokens,
        protected_output_tokens=llm.protected_output_tokens,
        protected_total_tokens=llm.protected_total_tokens,
        protected_cost_usd=llm.protected_cost_usd,
        unknown_usage_policy=llm.unknown_usage_policy,
    )
    merged_kinds = dict(policy.max_tool_calls_by_kind)
    for kind, delta in increase.max_tool_calls_by_kind.items():
        merged_kinds[kind] = merged_kinds.get(kind, 0) + delta
    return ExplorationBudgetPolicy(
        llm=amended_llm,
        max_successful_tool_calls=(
            policy.max_successful_tool_calls + increase.max_successful_tool_calls
        ),
        max_tool_calls_by_kind=merged_kinds,
        max_rows_scanned=_raised(policy.max_rows_scanned, increase.max_rows_scanned),
        max_result_cells=_raised(policy.max_result_cells, increase.max_result_cells),
        idle_timeout_seconds=policy.idle_timeout_seconds,
        max_rounds=policy.max_rounds + increase.max_rounds,
    )


def _not_lowered(
    old: int | float | Decimal | None, new: int | float | Decimal | None
) -> bool:
    if old is None:
        return new is None
    if new is None:
        return True
    return new >= old


def _verify_monotonic(old: ExplorationBudgetPolicy, new: ExplorationBudgetPolicy) -> None:
    """R3.2: caps may only rise (None = unlimited); non-cap fields may not change."""
    cap_pairs = (
        ("llm.max_requests", old.llm.max_requests, new.llm.max_requests),
        ("llm.max_input_tokens", old.llm.max_input_tokens, new.llm.max_input_tokens),
        ("llm.max_output_tokens", old.llm.max_output_tokens, new.llm.max_output_tokens),
        ("llm.max_total_tokens", old.llm.max_total_tokens, new.llm.max_total_tokens),
        ("llm.max_cost_usd", old.llm.max_cost_usd, new.llm.max_cost_usd),
        ("llm.max_wall_seconds", old.llm.max_wall_seconds, new.llm.max_wall_seconds),
        (
            "max_successful_tool_calls",
            old.max_successful_tool_calls,
            new.max_successful_tool_calls,
        ),
        ("max_rows_scanned", old.max_rows_scanned, new.max_rows_scanned),
        ("max_result_cells", old.max_result_cells, new.max_result_cells),
        ("max_rounds", old.max_rounds, new.max_rounds),
    )
    for name, old_cap, new_cap in cap_pairs:
        if not _not_lowered(old_cap, new_cap):
            raise BudgetAmendmentError(f"{name} may not be lowered ({old_cap} -> {new_cap}).")
    for kind, old_cap in old.max_tool_calls_by_kind.items():
        new_cap = new.max_tool_calls_by_kind.get(kind, 0)
        if new_cap < old_cap:
            raise BudgetAmendmentError(
                f"max_tool_calls_by_kind[{kind!r}] may not be lowered ({old_cap} -> {new_cap})."
            )
    frozen_pairs = (
        ("idle_timeout_seconds", old.idle_timeout_seconds, new.idle_timeout_seconds),
        ("llm.protected_requests", old.llm.protected_requests, new.llm.protected_requests),
        (
            "llm.protected_input_tokens",
            old.llm.protected_input_tokens,
            new.llm.protected_input_tokens,
        ),
        (
            "llm.protected_output_tokens",
            old.llm.protected_output_tokens,
            new.llm.protected_output_tokens,
        ),
        (
            "llm.protected_total_tokens",
            old.llm.protected_total_tokens,
            new.llm.protected_total_tokens,
        ),
        ("llm.protected_cost_usd", old.llm.protected_cost_usd, new.llm.protected_cost_usd),
        ("llm.unknown_usage_policy", old.llm.unknown_usage_policy, new.llm.unknown_usage_policy),
    )
    for name, old_value, new_value in frozen_pairs:
        if old_value != new_value:
            raise BudgetAmendmentError(
                f"{name} may not change via amendment ({old_value} -> {new_value})."
            )

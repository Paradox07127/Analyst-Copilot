from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite
from threading import RLock
from time import monotonic
from typing import Literal, cast


class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its configured time or token budget."""


@dataclass
class Budget:
    max_seconds: float | None = None
    max_tokens: int | None = None
    started_at: float = field(default_factory=monotonic)
    tokens_used: int = 0

    def check(self) -> None:
        if self.max_seconds is None:
            return
        if self.max_seconds <= 0:
            raise BudgetExceeded("Session time budget exhausted.")
        if monotonic() - self.started_at > self.max_seconds:
            raise BudgetExceeded("Session time budget exhausted.")

    def add_tokens(self, tokens: int) -> None:
        """Record token usage and enforce the token ceiling if configured."""
        if tokens < 0:
            raise ValueError("Token usage cannot be negative.")
        self.tokens_used += tokens
        if self.max_tokens is not None and self.tokens_used > self.max_tokens:
            raise BudgetExceeded(
                f"Session token budget exhausted: {self.tokens_used} > {self.max_tokens}."
            )

    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)


type BudgetDimension = Literal[
    "requests",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "wall_seconds",
]
type ReservationStatus = Literal["reserved", "settled", "released", "uncertain"]
type UnknownUsagePolicy = Literal["consume_reservation", "reject"]
type Money = Decimal | int | float


class SessionBudgetExceeded(BudgetExceeded):
    """Raised before or after a call when a typed run-budget limit is exceeded."""

    def __init__(
        self,
        dimension: BudgetDimension,
        *,
        limit: int | float | Decimal,
        attempted: int | float | Decimal,
        call_id: str | None = None,
        stage: Literal["reservation", "settlement"] | None = None,
    ) -> None:
        self.dimension = dimension
        self.limit = limit
        self.attempted = attempted
        self.call_id = call_id
        self.stage = stage
        call_suffix = f" for call {call_id!r}" if call_id is not None else ""
        super().__init__(
            f"Session {dimension} budget exhausted{call_suffix}: {attempted} > {limit}."
        )


class BudgetReservationConflict(ValueError):
    """Raised when a call id is reused for a run-budget reservation."""


class BudgetUsageUncertain(BudgetExceeded):
    """Raised when a hard budget cannot safely account for unknown usage."""

    def __init__(
        self,
        call_id: str,
        *,
        stage: Literal["reservation", "settlement", "restore"],
        missing: tuple[str, ...],
    ) -> None:
        self.call_id = call_id
        self.stage = stage
        self.missing = missing
        super().__init__(
            f"Cannot safely account for call {call_id!r} during {stage}; "
            f"missing {', '.join(missing)}."
        )


def _validate_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _validate_seconds(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_wall_seconds must be a number or None.")
    if not isfinite(value):
        raise ValueError("max_wall_seconds must be finite.")
    if value < 0:
        raise ValueError("max_wall_seconds cannot be negative.")


def _money(value: Money, *, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float)):
        raise TypeError(f"{name} must be a Decimal, int, or float.")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    converted = Decimal(str(value))
    if not converted.is_finite():
        raise ValueError(f"{name} must be finite.")
    if converted < 0:
        raise ValueError(f"{name} cannot be negative.")
    return converted


@dataclass(frozen=True, slots=True)
class SessionBudgetPolicy:
    """Hard run limits plus capacity reserved for protected, terminal work."""

    max_requests: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: Money | None = None
    max_wall_seconds: float | None = None
    protected_requests: int = 0
    protected_input_tokens: int = 0
    protected_output_tokens: int = 0
    protected_total_tokens: int = 0
    protected_cost_usd: Money = Decimal("0")
    unknown_usage_policy: UnknownUsagePolicy = "consume_reservation"

    def __post_init__(self) -> None:
        for name in (
            "max_requests",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            _validate_limit(name, getattr(self, name))
        for name in (
            "protected_requests",
            "protected_input_tokens",
            "protected_output_tokens",
            "protected_total_tokens",
        ):
            _validate_limit(name, getattr(self, name))
        _validate_seconds(self.max_wall_seconds)

        max_cost = (
            None if self.max_cost_usd is None else _money(self.max_cost_usd, name="max_cost_usd")
        )
        protected_cost = _money(self.protected_cost_usd, name="protected_cost_usd")
        object.__setattr__(self, "max_cost_usd", max_cost)
        object.__setattr__(self, "protected_cost_usd", protected_cost)
        if self.unknown_usage_policy not in ("consume_reservation", "reject"):
            raise ValueError("unknown_usage_policy must be 'consume_reservation' or 'reject'.")

        pairs = (
            ("requests", self.protected_requests, self.max_requests),
            ("input_tokens", self.protected_input_tokens, self.max_input_tokens),
            ("output_tokens", self.protected_output_tokens, self.max_output_tokens),
            ("total_tokens", self.protected_total_tokens, self.max_total_tokens),
            ("cost_usd", protected_cost, max_cost),
        )
        for dimension, protected, maximum in pairs:
            if protected and maximum is None:
                raise ValueError(f"protected_{dimension} requires a max_{dimension} limit.")
            if maximum is not None and protected > maximum:
                raise ValueError(f"protected_{dimension} cannot exceed max_{dimension}.")


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """Immutable record of one call's budget lifecycle."""

    call_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Decimal
    protected: bool
    status: ReservationStatus = "reserved"
    usage_known: bool | None = None


@dataclass(slots=True)
class SessionBudgetState:
    """Thread-safe reservation and settlement state for one agent run."""

    policy: SessionBudgetPolicy
    started_at: float = field(default_factory=monotonic)
    requests_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    total_tokens_used: int = 0
    cost_usd_used: Decimal = field(default_factory=lambda: Decimal("0"))
    _active: dict[str, BudgetReservation] = field(default_factory=dict, init=False)
    _completed: dict[str, BudgetReservation] = field(default_factory=dict, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.started_at, bool) or not isinstance(self.started_at, (int, float)):
            raise TypeError("started_at must be a number.")
        if not isfinite(self.started_at):
            raise ValueError("started_at must be finite.")
        for name in (
            "requests_used",
            "input_tokens_used",
            "output_tokens_used",
            "total_tokens_used",
        ):
            _validate_limit(name, getattr(self, name))
        self.cost_usd_used = _money(self.cost_usd_used, name="cost_usd_used")

    def check_wall_time(self, *, now: float | None = None) -> None:
        """Reject work after the configured wall-clock deadline."""
        current = monotonic() if now is None else now
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise TypeError("now must be a number.")
        if not isfinite(current):
            raise ValueError("now must be finite.")
        limit = self.policy.max_wall_seconds
        elapsed = max(0.0, current - self.started_at)
        if limit is not None and elapsed > limit:
            raise SessionBudgetExceeded(
                "wall_seconds",
                limit=limit,
                attempted=elapsed,
                stage="reservation",
            )

    def reserve(
        self,
        call_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cost_usd: Money | None = None,
        protected: bool = False,
        now: float | None = None,
    ) -> BudgetReservation:
        """Atomically reserve worst-case capacity before a provider call."""
        if not isinstance(call_id, str):
            raise TypeError("call_id must be a string.")
        if not call_id.strip():
            raise ValueError("call_id cannot be empty.")
        if not isinstance(protected, bool):
            raise TypeError("protected must be a boolean.")
        _validate_limit("input_tokens", input_tokens)
        _validate_limit("output_tokens", output_tokens)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        _validate_limit("total_tokens", total_tokens)
        if total_tokens < input_tokens + output_tokens:
            raise ValueError("total_tokens cannot be less than input_tokens + output_tokens.")
        if cost_usd is None:
            if self.policy.max_cost_usd is not None:
                raise BudgetUsageUncertain(
                    call_id,
                    stage="reservation",
                    missing=("cost_usd",),
                )
            cost = Decimal("0")
        else:
            cost = _money(cost_usd, name="cost_usd")

        with self._lock:
            self.check_wall_time(now=now)
            if call_id in self._active or call_id in self._completed:
                raise BudgetReservationConflict(f"call_id {call_id!r} is already reserved.")
            reservation = BudgetReservation(
                call_id=call_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                protected=protected,
            )
            projected = self._projected(reservation)
            self._enforce(
                projected,
                nonprotected_projected=self._nonprotected_projected(reservation),
                protected=protected,
                call_id=call_id,
            )
            self._active[call_id] = reservation
            return reservation

    def settle(
        self,
        call_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: Money | None = None,
    ) -> BudgetReservation:
        """Settle actual usage; unknown fields conservatively retain their reservation."""
        with self._lock:
            reserved = self._get_active(call_id)
            missing = tuple(
                name
                for name, value in (
                    ("input_tokens", input_tokens),
                    ("output_tokens", output_tokens),
                    ("total_tokens", total_tokens),
                    ("cost_usd", cost_usd),
                )
                if value is None
            )
            if missing and self.policy.unknown_usage_policy == "reject":
                raise BudgetUsageUncertain(
                    call_id,
                    stage="settlement",
                    missing=missing,
                )
            actual_input = self._usage_or_reserved(
                "input_tokens", input_tokens, reserved.input_tokens
            )
            actual_output = self._usage_or_reserved(
                "output_tokens", output_tokens, reserved.output_tokens
            )
            actual_total = self._usage_or_reserved(
                "total_tokens", total_tokens, reserved.total_tokens
            )
            if actual_total < actual_input + actual_output:
                raise ValueError("total_tokens cannot be less than input_tokens + output_tokens.")
            actual_cost = (
                reserved.cost_usd if cost_usd is None else _money(cost_usd, name="cost_usd")
            )
            usage_known = all(
                value is not None for value in (input_tokens, output_tokens, total_tokens, cost_usd)
            )
            settled = BudgetReservation(
                call_id=call_id,
                input_tokens=actual_input,
                output_tokens=actual_output,
                total_tokens=actual_total,
                cost_usd=actual_cost,
                protected=reserved.protected,
                status="settled",
                usage_known=usage_known,
            )
            self._consume(settled)
            del self._active[call_id]
            self._completed[call_id] = settled
            self._enforce_used(call_id=call_id)
            return settled

    def release(self, call_id: str) -> BudgetReservation:
        """Release a call that is known not to have reached the provider."""
        with self._lock:
            reserved = self._get_active(call_id)
            released = BudgetReservation(
                call_id=reserved.call_id,
                input_tokens=reserved.input_tokens,
                output_tokens=reserved.output_tokens,
                total_tokens=reserved.total_tokens,
                cost_usd=reserved.cost_usd,
                protected=reserved.protected,
                status="released",
                usage_known=False,
            )
            del self._active[call_id]
            self._completed[call_id] = released
            return released

    def mark_uncertain(self, call_id: str) -> BudgetReservation:
        """Consume the full reservation when provider execution is uncertain."""
        with self._lock:
            reserved = self._get_active(call_id)
            uncertain = BudgetReservation(
                call_id=reserved.call_id,
                input_tokens=reserved.input_tokens,
                output_tokens=reserved.output_tokens,
                total_tokens=reserved.total_tokens,
                cost_usd=reserved.cost_usd,
                protected=reserved.protected,
                status="uncertain",
                usage_known=False,
            )
            self._consume(uncertain)
            del self._active[call_id]
            self._completed[call_id] = uncertain
            return uncertain

    def reservation(self, call_id: str) -> BudgetReservation | None:
        """Return the latest immutable reservation record for a call."""
        with self._lock:
            return self._active.get(call_id) or self._completed.get(call_id)

    @property
    def active_reservations(self) -> tuple[BudgetReservation, ...]:
        with self._lock:
            return tuple(self._active.values())

    def remaining(
        self, dimension: BudgetDimension, *, protected: bool = False
    ) -> int | Decimal | float | None:
        """Return currently reservable capacity for a budget dimension."""
        with self._lock:
            if dimension == "wall_seconds":
                limit = self.policy.max_wall_seconds
                if limit is None:
                    return None
                return max(0.0, limit - max(0.0, monotonic() - self.started_at))
            used, active, maximum, protected_amount = self._dimension_values(dimension)
            if maximum is None:
                return None
            total_remaining = maximum - used - active
            if protected:
                return max(type(maximum)(0), total_remaining)
            nonprotected_used, nonprotected_active = self._nonprotected_values(dimension)
            nonprotected_remaining = (
                maximum - protected_amount - nonprotected_used - nonprotected_active
            )
            return max(
                type(maximum)(0),
                min(total_remaining, nonprotected_remaining),
            )

    def _get_active(self, call_id: str) -> BudgetReservation:
        if not isinstance(call_id, str):
            raise TypeError("call_id must be a string.")
        try:
            return self._active[call_id]
        except KeyError as exc:
            if call_id in self._completed:
                status = self._completed[call_id].status
                raise BudgetReservationConflict(
                    f"call_id {call_id!r} is already {status}."
                ) from exc
            raise KeyError(f"No active reservation for call_id {call_id!r}.") from exc

    @staticmethod
    def _usage_or_reserved(name: str, value: int | None, reserved: int) -> int:
        if value is None:
            return reserved
        _validate_limit(name, value)
        return value

    def _projected(self, reservation: BudgetReservation) -> dict[BudgetDimension, int | Decimal]:
        active_requests = len(self._active)
        return {
            "requests": self.requests_used + active_requests + 1,
            "input_tokens": self.input_tokens_used
            + sum(item.input_tokens for item in self._active.values())
            + reservation.input_tokens,
            "output_tokens": self.output_tokens_used
            + sum(item.output_tokens for item in self._active.values())
            + reservation.output_tokens,
            "total_tokens": self.total_tokens_used
            + sum(item.total_tokens for item in self._active.values())
            + reservation.total_tokens,
            "cost_usd": self.cost_usd_used
            + sum((item.cost_usd for item in self._active.values()), Decimal("0"))
            + reservation.cost_usd,
        }

    def _enforce(
        self,
        projected: dict[BudgetDimension, int | Decimal],
        *,
        nonprotected_projected: dict[BudgetDimension, int | Decimal],
        protected: bool,
        call_id: str,
    ) -> None:
        for dimension, attempted in projected.items():
            _, _, maximum, protected_amount = self._dimension_values(dimension)
            if maximum is None:
                continue
            if attempted > maximum:
                raise SessionBudgetExceeded(
                    dimension,
                    limit=maximum,
                    attempted=attempted,
                    call_id=call_id,
                    stage="reservation",
                )
            if not protected:
                nonprotected_attempted = nonprotected_projected[dimension]
                nonprotected_limit = maximum - protected_amount
                if nonprotected_attempted > nonprotected_limit:
                    raise SessionBudgetExceeded(
                        dimension,
                        limit=nonprotected_limit,
                        attempted=nonprotected_attempted,
                        call_id=call_id,
                        stage="reservation",
                    )

    def _enforce_used(self, *, call_id: str) -> None:
        for dimension in (
            "requests",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
        ):
            used, _, maximum, protected_amount = self._dimension_values(dimension)
            if maximum is None:
                continue
            if used > maximum:
                raise SessionBudgetExceeded(
                    dimension,
                    limit=maximum,
                    attempted=used,
                    call_id=call_id,
                    stage="settlement",
                )
            # Symmetric with _enforce: without this branch an ordinary call that
            # settled above its reservation ate the protected reserve silently,
            # and finalization was then locked out by its own quota.
            nonprotected_used, nonprotected_active = self._nonprotected_values(dimension)
            nonprotected_attempted = nonprotected_used + nonprotected_active
            nonprotected_limit = maximum - protected_amount
            if nonprotected_attempted > nonprotected_limit:
                raise SessionBudgetExceeded(
                    dimension,
                    limit=nonprotected_limit,
                    attempted=nonprotected_attempted,
                    call_id=call_id,
                    stage="settlement",
                )

    def _consume(self, reservation: BudgetReservation) -> None:
        self.requests_used += 1
        self.input_tokens_used += reservation.input_tokens
        self.output_tokens_used += reservation.output_tokens
        self.total_tokens_used += reservation.total_tokens
        self.cost_usd_used += reservation.cost_usd

    def _nonprotected_projected(
        self, reservation: BudgetReservation
    ) -> dict[BudgetDimension, int | Decimal]:
        projected: dict[BudgetDimension, int | Decimal] = {}
        for dimension in (
            "requests",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
        ):
            used, active = self._nonprotected_values(dimension)
            increment = (
                self._reservation_value(reservation, dimension) if not reservation.protected else 0
            )
            projected[dimension] = used + active + increment
        return projected

    def _nonprotected_values(
        self, dimension: BudgetDimension
    ) -> tuple[int | Decimal, int | Decimal]:
        used, _, _, _ = self._dimension_values(dimension)
        protected_used = sum(
            (
                self._reservation_value(item, dimension)
                for item in self._completed.values()
                if item.protected and item.status in ("settled", "uncertain")
            ),
            Decimal("0") if dimension == "cost_usd" else 0,
        )
        active = sum(
            (
                self._reservation_value(item, dimension)
                for item in self._active.values()
                if not item.protected
            ),
            Decimal("0") if dimension == "cost_usd" else 0,
        )
        return used - protected_used, active

    @staticmethod
    def _reservation_value(
        reservation: BudgetReservation, dimension: BudgetDimension
    ) -> int | Decimal:
        if dimension == "requests":
            return 1
        if dimension == "input_tokens":
            return reservation.input_tokens
        if dimension == "output_tokens":
            return reservation.output_tokens
        if dimension == "total_tokens":
            return reservation.total_tokens
        if dimension == "cost_usd":
            return reservation.cost_usd
        raise ValueError("wall_seconds does not have a reservation value.")

    def _dimension_values(
        self, dimension: BudgetDimension
    ) -> tuple[int | Decimal, int | Decimal, int | Decimal | None, int | Decimal]:
        if dimension == "requests":
            return (
                self.requests_used,
                len(self._active),
                self.policy.max_requests,
                self.policy.protected_requests,
            )
        if dimension == "input_tokens":
            return (
                self.input_tokens_used,
                sum(item.input_tokens for item in self._active.values()),
                self.policy.max_input_tokens,
                self.policy.protected_input_tokens,
            )
        if dimension == "output_tokens":
            return (
                self.output_tokens_used,
                sum(item.output_tokens for item in self._active.values()),
                self.policy.max_output_tokens,
                self.policy.protected_output_tokens,
            )
        if dimension == "total_tokens":
            return (
                self.total_tokens_used,
                sum(item.total_tokens for item in self._active.values()),
                self.policy.max_total_tokens,
                self.policy.protected_total_tokens,
            )
        if dimension == "cost_usd":
            return (
                self.cost_usd_used,
                sum((item.cost_usd for item in self._active.values()), Decimal("0")),
                cast(Decimal | None, self.policy.max_cost_usd),
                cast(Decimal, self.policy.protected_cost_usd),
            )
        raise ValueError("wall_seconds does not have a reservable counter.")

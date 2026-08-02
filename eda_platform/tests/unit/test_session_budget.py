from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from eda_platform.core.budget import (
    Budget,
    BudgetExceeded,
    BudgetReservationConflict,
    BudgetUsageUncertain,
    SessionBudgetExceeded,
    SessionBudgetPolicy,
    SessionBudgetState,
)


def test_legacy_budget_api_remains_compatible() -> None:
    budget = Budget(max_tokens=5)

    budget.check()
    budget.add_tokens(3)

    assert budget.remaining_tokens() == 2
    with pytest.raises(BudgetExceeded):
        budget.add_tokens(3)


def test_policy_validation_errors_are_strict() -> None:
    with pytest.raises(ValueError, match="max_requests"):
        SessionBudgetPolicy(max_requests=-1)
    with pytest.raises(ValueError, match="protected_cost"):
        SessionBudgetPolicy(max_cost_usd=1, protected_cost_usd=-0.1)
    with pytest.raises(ValueError, match="protected_total_tokens"):
        SessionBudgetPolicy(max_total_tokens=10, protected_total_tokens=11)
    with pytest.raises(TypeError, match="max_requests"):
        SessionBudgetPolicy(max_requests=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_cost_usd"):
        SessionBudgetPolicy(max_cost_usd=float("inf"))
    with pytest.raises(ValueError, match="requires"):
        SessionBudgetPolicy(protected_requests=1)


@pytest.mark.parametrize(
    ("policy", "kwargs", "dimension"),
    [
        (SessionBudgetPolicy(max_requests=0), {}, "requests"),
        (SessionBudgetPolicy(max_input_tokens=4), {"input_tokens": 5}, "input_tokens"),
        (SessionBudgetPolicy(max_output_tokens=4), {"output_tokens": 5}, "output_tokens"),
        (SessionBudgetPolicy(max_total_tokens=4), {"total_tokens": 5}, "total_tokens"),
        (SessionBudgetPolicy(max_cost_usd=0.1), {"cost_usd": 0.11}, "cost_usd"),
    ],
)
def test_reserve_rejects_each_hard_limit_before_call(
    policy: SessionBudgetPolicy,
    kwargs: dict[str, object],
    dimension: str,
) -> None:
    state = SessionBudgetState(policy)

    with pytest.raises(SessionBudgetExceeded) as caught:
        state.reserve("call-1", **kwargs)  # type: ignore[arg-type]

    assert caught.value.dimension == dimension
    assert state.active_reservations == ()
    assert state.requests_used == 0


def test_protected_reserve_is_unavailable_to_normal_work() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(
            max_requests=2,
            max_total_tokens=100,
            protected_requests=1,
            protected_total_tokens=30,
        )
    )
    state.reserve("normal", total_tokens=70)

    with pytest.raises(SessionBudgetExceeded):
        state.reserve("normal-2")

    protected = state.reserve("report", total_tokens=30, protected=True)
    assert protected.protected is True


def test_protected_reservation_does_not_double_reduce_normal_capacity() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(
            max_requests=2,
            max_total_tokens=100,
            protected_requests=1,
            protected_total_tokens=30,
        )
    )

    state.reserve("report", total_tokens=30, protected=True)
    normal = state.reserve("normal", total_tokens=70)

    assert normal.protected is False
    assert state.remaining("total_tokens") == 0


def test_settlement_replaces_reservation_with_actual_usage() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(max_requests=2, max_total_tokens=100, max_cost_usd=1)
    )
    state.reserve(
        "call-1",
        input_tokens=40,
        output_tokens=20,
        total_tokens=60,
        cost_usd=0.6,
    )

    settled = state.settle(
        "call-1",
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        cost_usd=Decimal("0.25"),
    )

    assert settled.status == "settled"
    assert settled.usage_known is True
    assert state.requests_used == 1
    assert state.input_tokens_used == 20
    assert state.output_tokens_used == 10
    assert state.total_tokens_used == 30
    assert state.cost_usd_used == Decimal("0.25")
    assert state.remaining("total_tokens") == 70


def test_unknown_usage_conservatively_consumes_reserved_amounts() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_total_tokens=100, max_cost_usd=1))
    state.reserve(
        "call-1",
        input_tokens=30,
        output_tokens=20,
        total_tokens=50,
        cost_usd=0.4,
    )

    settled = state.settle("call-1")

    assert settled.usage_known is False
    assert state.total_tokens_used == 50
    assert state.cost_usd_used == Decimal("0.4")


def test_unknown_price_rejected_when_cost_limit_is_configured() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_cost_usd=1))

    with pytest.raises(BudgetUsageUncertain) as caught:
        state.reserve("unpriced")

    assert caught.value.stage == "reservation"
    assert caught.value.missing == ("cost_usd",)


def test_strict_unknown_usage_policy_rejects_incomplete_settlement() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(max_total_tokens=100, unknown_usage_policy="reject")
    )
    state.reserve("call-1", total_tokens=50)

    with pytest.raises(BudgetUsageUncertain) as caught:
        state.settle("call-1", total_tokens=30)

    assert caught.value.stage == "settlement"
    assert "input_tokens" in caught.value.missing
    assert state.reservation("call-1").status == "reserved"  # type: ignore[union-attr]


def test_release_and_uncertain_have_distinct_accounting() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_requests=2, max_total_tokens=20))
    state.reserve("not-sent", total_tokens=10)
    state.reserve("maybe-sent", total_tokens=10)

    released = state.release("not-sent")
    uncertain = state.mark_uncertain("maybe-sent")

    assert released.status == "released"
    assert uncertain.status == "uncertain"
    assert state.requests_used == 1
    assert state.total_tokens_used == 10


def test_settlement_records_overage_before_raising() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_total_tokens=10))
    state.reserve("call-1", total_tokens=5)

    with pytest.raises(SessionBudgetExceeded) as caught:
        state.settle("call-1", total_tokens=12)

    assert caught.value.dimension == "total_tokens"
    assert state.total_tokens_used == 12
    assert state.reservation("call-1").status == "settled"  # type: ignore[union-attr]


def test_wall_limit_rejects_preflight() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(max_wall_seconds=2),
        started_at=100,
    )

    state.reserve("on-time", now=102)
    state.release("on-time")

    with pytest.raises(SessionBudgetExceeded) as caught:
        state.reserve("late", now=102.01)

    assert caught.value.dimension == "wall_seconds"


def test_reused_call_id_is_always_a_conflict() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_requests=2))
    state.reserve("same")

    with pytest.raises(BudgetReservationConflict):
        state.reserve("same")

    state.settle("same")
    with pytest.raises(BudgetReservationConflict):
        state.reserve("same")
    with pytest.raises(BudgetReservationConflict):
        state.settle("same")


def test_concurrent_reservations_cannot_cross_request_limit() -> None:
    state = SessionBudgetState(SessionBudgetPolicy(max_requests=1))

    def attempt(call_id: str) -> str:
        try:
            state.reserve(call_id)
        except SessionBudgetExceeded:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(attempt, (f"call-{index}" for index in range(8))))

    assert outcomes.count("reserved") == 1
    assert outcomes.count("rejected") == 7
    assert len(state.active_reservations) == 1


def test_settlement_enforces_the_nonprotected_sub_cap() -> None:
    # R3-F5: reserve() honours max - protected, settle() only compared against
    # max, so an ordinary call whose actuals exceeded its reservation ate the
    # finalization reserve without raising.
    state = SessionBudgetState(
        SessionBudgetPolicy(
            max_requests=10,
            max_total_tokens=1000,
            max_input_tokens=1000,
            max_output_tokens=1000,
            max_cost_usd=Decimal("1.00"),
            protected_requests=2,
            protected_total_tokens=400,
            protected_input_tokens=400,
            protected_output_tokens=400,
            protected_cost_usd=Decimal("0.40"),
        )
    )
    state.reserve("call-1", input_tokens=300, output_tokens=300, cost_usd=Decimal("0.30"))

    with pytest.raises(SessionBudgetExceeded) as caught:
        state.settle(
            "call-1",
            input_tokens=500,
            output_tokens=500,
            total_tokens=1000,
            cost_usd=Decimal("0.95"),
        )

    assert caught.value.stage == "settlement"
    assert caught.value.limit == 600


def test_settlement_within_the_nonprotected_sub_cap_is_untouched() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(
            max_requests=10,
            max_total_tokens=1000,
            protected_requests=2,
            protected_total_tokens=400,
        )
    )
    state.reserve("call-1", total_tokens=500)
    settled = state.settle("call-1", input_tokens=300, output_tokens=300, total_tokens=600)

    assert settled.status == "settled"
    assert state.remaining("total_tokens", protected=True) == 400


def test_a_protected_call_may_still_settle_into_its_own_reserve() -> None:
    state = SessionBudgetState(
        SessionBudgetPolicy(
            max_requests=10,
            max_total_tokens=1000,
            protected_requests=2,
            protected_total_tokens=400,
        )
    )
    state.reserve("normal", total_tokens=600)
    state.settle("normal", input_tokens=300, output_tokens=300, total_tokens=600)
    state.reserve("finalize", total_tokens=300, protected=True)
    settled = state.settle(
        "finalize", input_tokens=200, output_tokens=200, total_tokens=400
    )

    assert settled.status == "settled"
    assert state.total_tokens_used == 1000

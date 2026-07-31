"""DI8-B — semantic bootstrap: hypothesise (LLM, batched) then verify (code).

Covers label remapping, the full mocked-LLM loop, priors-never-verify, the
LLM-failure degrade path, retry behaviour, structural-role precedence, and
seed override priority.
"""

from __future__ import annotations

from typing import TypeVar, cast

import pandas as pd
from pydantic import BaseModel

from eda_platform.agents.semantic_bootstrap import (
    RawColumnRoleHypothesis,
    RawSemanticHypotheses,
    bootstrap_semantics,
    remap_label,
)
from eda_platform.core.column_roles import ColumnRoleName
from eda_platform.core.llm import LLMResultMetadata, OfflineLLMClient
from eda_platform.core.semantic import ColumnRoleSeed, SemanticSeeds
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile

# --- fake LLM ---------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class FakeLLM:
    """Scripted stand-in for LLMClient: returns/raises each outcome in order."""

    def __init__(self, outcomes: list[object], model: str = "fake-model-v1") -> None:
        self._outcomes = list(outcomes)
        self._model = model
        self.calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        assert task == "di8_semantic_bootstrap"
        assert schema is RawSemanticHypotheses
        assert "role_definitions" in payload and "table" in payload
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return cast(T, outcome)

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        return LLMResultMetadata(provider="test", model=self._model)


# --- fixtures ---------------------------------------------------------------


def _frame() -> pd.DataFrame:
    order_ids = [f"g{index:02d}" for index in range(10) for _ in range(3)]
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "order_item_id": [1, 2, 3] * 10,
            "price": [round(10.0 + i * 1.5, 2) for i in range(30)],
            "seller_zip_code_prefix": [13023, 1037, 20031] * 10,
            "order_status": ["delivered", "shipped", "delivered"] * 10,
            "shipped_at": [f"2017-10-{2 + i % 27:02d} 10:56:33" for i in range(30)],
            "notes": [
                f"Customer left a fairly long free-form comment number {i}." for i in range(30)
            ],
        }
    )


def _detail(frame: pd.DataFrame, name: str, semantic_type: str) -> ColumnProfile:
    series = cast(pd.Series, frame[name])
    rows = len(frame)
    unique = int(series.nunique())
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype) if str(series.dtype) != "object" else "str",
        semantic_type=semantic_type,  # type: ignore[arg-type]
        missing_count=0,
        missing_percent=0.0,
        unique_count=unique,
        unique_percent=round(unique / rows * 100, 2),
        sample_values=[str(value) for value in series.head(5)],
    )


def _profile(frame: pd.DataFrame) -> DatasetProfile:
    semantic_types = {
        "order_id": "id",
        "order_item_id": "numeric",
        "price": "numeric",
        "seller_zip_code_prefix": "numeric",
        "order_status": "categorical",
        "shipped_at": "datetime",
        "notes": "text",
    }
    details = [_detail(frame, name, semantic_types[name]) for name in frame.columns]
    return DatasetProfile(
        dataset_id="ds_items",
        name="olist_order_items.csv",
        rows=len(frame),
        columns=len(details),
        column_names=[detail.name for detail in details],
        dtypes={detail.name: detail.dtype for detail in details},
        missing_values={detail.name: 0 for detail in details},
        missing_percent={detail.name: 0.0 for detail in details},
        numeric_columns=[d.name for d in details if d.semantic_type == "numeric"],
        categorical_columns=[d.name for d in details if d.semantic_type == "categorical"],
        columns_detail=details,
    )


def _hypotheses() -> RawSemanticHypotheses:
    return RawSemanticHypotheses(
        entity="Order line item",
        columns=[
            RawColumnRoleHypothesis(column="order_id", role="flurble", rationale="?"),
            RawColumnRoleHypothesis(
                column="order_item_id", role="line number", rationale="Counts items per order."
            ),
            RawColumnRoleHypothesis(column="price", role="metric", rationale="Item price."),
            RawColumnRoleHypothesis(
                column="seller_zip_code_prefix", role="postal code", rationale="Zip prefix."
            ),
            RawColumnRoleHypothesis(column="order_status", role="category"),
            RawColumnRoleHypothesis(column="shipped_at", role="date"),
            RawColumnRoleHypothesis(column="notes", role="free text"),
            RawColumnRoleHypothesis(column="ghost", role="measure"),  # hallucinated column
        ],
    )


# --- label remapping --------------------------------------------------------


def test_remap_label_exact_synonym_and_miss() -> None:
    assert remap_label("measure") is ColumnRoleName.MEASURE
    assert remap_label("Primary Key") is ColumnRoleName.IDENTIFIER
    assert remap_label("line number") is ColumnRoleName.SEQUENCE
    assert remap_label("date-time") is ColumnRoleName.TIMESTAMP
    assert remap_label("postal code") is ColumnRoleName.CODE
    assert remap_label("total gibberish") is None
    assert remap_label("") is None


# --- full bootstrap flow (mocked LLM) ---------------------------------------


def test_bootstrap_full_flow_verifies_hypotheses_deterministically() -> None:
    frame = _frame()
    result = bootstrap_semantics(_profile(frame), llm=FakeLLM([_hypotheses()]), frame=frame)

    assert not result.degraded
    assert result.degraded_reason == ""
    role_set = result.role_set
    assert role_set.model_version == "fake-model-v1"
    assert role_set.entity == "Order line item"

    seq = role_set.role_of("order_item_id")
    assert seq is not None
    assert seq.role is ColumnRoleName.SEQUENCE
    assert seq.provenance == "inferred"
    assert seq.verified_by == ["sequence_strict_1n_within_group:order_id"]
    assert seq.rationale == "Counts items per order."

    assert role_set.role_of("price").role is ColumnRoleName.MEASURE  # type: ignore[union-attr]
    assert role_set.role_of("seller_zip_code_prefix").role is ColumnRoleName.CODE  # type: ignore[union-attr]
    assert role_set.role_of("order_status").role is ColumnRoleName.DIMENSION  # type: ignore[union-attr]
    assert role_set.role_of("shipped_at").role is ColumnRoleName.TIMESTAMP  # type: ignore[union-attr]
    assert role_set.role_of("notes").role is ColumnRoleName.TEXT  # type: ignore[union-attr]

    # Label that maps to nothing is reported, not coerced into a role.
    assert result.unmapped_labels == {"order_id": "flurble"}
    # The hallucinated column is ignored entirely.
    assert role_set.role_of("ghost") is None
    assert result.hypothesis_count == 7  # ghost excluded
    assert result.verified_count == 6
    assert result.unverified_count == 0

    # Consumption API: verified sequence gates; everything else keeps weight.
    assert role_set.excluded_from_stats() == {"order_item_id"}
    assert role_set.impact_weight("order_item_id") == 0.0
    assert role_set.impact_weight("price") == 1.0


def test_open_book_prior_fails_verification_and_stays_unverified() -> None:
    # Priors accelerate hypothesis generation but never count as verification:
    # claiming the repeated order_id is an "identifier" (an open-book answer
    # about Olist) fails the uniqueness check and must stay wording-only.
    frame = _frame()
    hypotheses = RawSemanticHypotheses(
        columns=[RawColumnRoleHypothesis(column="order_id", role="primary_key")]
    )
    result = bootstrap_semantics(_profile(frame), llm=FakeLLM([hypotheses]), frame=frame)

    role = result.role_set.role_of("order_id")
    assert role is not None
    assert role.role is ColumnRoleName.IDENTIFIER
    assert role.provenance == "unverified"
    assert role.verified_by == []
    # Unverified roles never gate: order_id stays in every candidate pool.
    assert "order_id" not in result.role_set.excluded_from_stats()
    assert result.role_set.impact_weight("order_id") == 1.0
    assert result.unverified_count >= 1


def test_verified_structural_baseline_beats_a_conflicting_verified_prior() -> None:
    # The zip prefix verifies as code (structural). A "dimension" hypothesis
    # also passes the low-cardinality check, but the structural role wins.
    frame = _frame()
    hypotheses = RawSemanticHypotheses(
        columns=[RawColumnRoleHypothesis(column="seller_zip_code_prefix", role="dimension")]
    )
    result = bootstrap_semantics(_profile(frame), llm=FakeLLM([hypotheses]), frame=frame)

    role = result.role_set.role_of("seller_zip_code_prefix")
    assert role is not None
    assert role.role is ColumnRoleName.CODE
    assert role.provenance == "inferred"


# --- degrade paths ----------------------------------------------------------


def test_bootstrap_without_llm_degrades_to_deterministic_roles() -> None:
    frame = _frame()
    for llm in (None, OfflineLLMClient()):
        result = bootstrap_semantics(_profile(frame), llm=llm, frame=frame)

        assert result.degraded
        assert result.degraded_reason == "llm_unavailable"
        role_set = result.role_set
        assert role_set.model_version == "deterministic"
        # Structural roles need no LLM: sequence/timestamp/code still verified.
        assert role_set.role_of("order_item_id").role is ColumnRoleName.SEQUENCE  # type: ignore[union-attr]
        assert role_set.role_of("shipped_at").role is ColumnRoleName.TIMESTAMP  # type: ignore[union-attr]
        assert role_set.role_of("seller_zip_code_prefix").role is ColumnRoleName.CODE  # type: ignore[union-attr]
        assert role_set.excluded_from_stats() == {"order_item_id"}


def test_bootstrap_llm_failure_degrades_with_reason_after_retries() -> None:
    frame = _frame()
    llm = FakeLLM([RuntimeError("boom-1"), RuntimeError("boom-2")])
    result = bootstrap_semantics(_profile(frame), llm=llm, frame=frame)

    assert llm.calls == 2  # bounded retry, no infinite loop
    assert result.degraded
    assert result.degraded_reason.startswith("llm_error: RuntimeError")
    # Deterministic baseline survives the failure.
    assert result.role_set.role_of("order_item_id").role is ColumnRoleName.SEQUENCE  # type: ignore[union-attr]


def test_bootstrap_retries_once_then_succeeds() -> None:
    frame = _frame()
    llm = FakeLLM([RuntimeError("transient"), _hypotheses()])
    result = bootstrap_semantics(_profile(frame), llm=llm, frame=frame)

    assert llm.calls == 2
    assert not result.degraded
    assert result.role_set.role_of("price").role is ColumnRoleName.MEASURE  # type: ignore[union-attr]


# --- seed priority ----------------------------------------------------------


def test_seed_overrides_llm_verified_role() -> None:
    frame = _frame()
    seeds = SemanticSeeds(
        column_role_seeds=[
            ColumnRoleSeed(
                dataset="olist_order_items.csv",
                column="price",
                role="dimension",
                note="Treat price as a tier for this project.",
            )
        ]
    )
    result = bootstrap_semantics(
        _profile(frame), llm=FakeLLM([_hypotheses()]), frame=frame, seeds=seeds
    )

    role = result.role_set.role_of("price")
    assert role is not None
    assert role.provenance == "seeded"
    assert role.role is ColumnRoleName.DIMENSION
    assert role.confidence == 1.0


def test_seed_applies_on_degraded_path_too() -> None:
    frame = _frame()
    seeds = SemanticSeeds(
        column_role_seeds=[
            ColumnRoleSeed(dataset="olist_order_items.csv", column="order_id", role="identifier")
        ]
    )
    result = bootstrap_semantics(_profile(frame), llm=None, frame=frame, seeds=seeds)

    role = result.role_set.role_of("order_id")
    assert role is not None
    assert role.provenance == "seeded"
    # A seeded identifier gates (human authority), unlike an unverified one.
    assert "order_id" in result.role_set.excluded_from_stats()

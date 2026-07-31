"""Resolve registered domain metrics and generate deterministic SQL."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eda_platform.core.column_roles import ColumnRoleName, ColumnRoleSet
from eda_platform.core.semantic import JoinWhitelist, JoinWhitelistEntry, SemanticSeeds
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.tools.relationship_discovery import _quote_identifier, _relation_name
from eda_platform.tools.sql_names import safe_alias

MetricDomain = Literal["ecommerce", "generic"]

# Column-name token vocabularies (deterministic refinement on top of roles).
_PRICE_TOKENS = frozenset(
    {"price", "payment", "amount", "revenue", "sales", "gmv", "gross", "turnover"}
)
_FREIGHT_TOKENS = frozenset({"freight", "shipping", "shipment", "postage"})
_ORDER_TOKENS = frozenset({"order", "purchase", "transaction", "invoice"})
_CUSTOMER_TOKENS = frozenset({"customer", "user", "buyer", "client", "member"})
_PROMISED_TS_TOKENS = frozenset({"estimated", "promised", "expected", "due"})
_ACTUAL_TS_TOKENS = frozenset({"delivered", "delivery", "arrival", "arrived", "actual", "received"})
# Prefer customer delivery timestamps over carrier handoff timestamps.
_CARRIER_TS_TOKENS = frozenset(
    {"carrier", "courier", "shipper", "logistics", "handoff", "hub", "warehouse"}
)
_CARRIER_BASIS_NOTE_EN = " Basis: late to carrier handoff, customer delivery date unavailable."
_CARRIER_FULFILLMENT_NOTE_EN = (
    " Basis: measured to carrier handoff, customer delivery date unavailable."
)
_START_TS_TOKENS = frozenset(
    {"purchase", "purchased", "order", "ordered", "created", "placed", "approved"}
)
# Key-column shape: the last name token must look like a key ...
_KEY_SUFFIX_TOKENS = frozenset({"id", "key", "uuid", "guid"})
# ... and per-group line counters must never pass as an order/customer key
# (order_item_id carries the "order" token but is a sequence, not a key).
_KEY_EXCLUDE_TOKENS = frozenset(
    {"item", "line", "detail", "seq", "sequence", "sequential", "ordinal"}
)
# Roles that disqualify a column from acting as an order/customer key.
_KEY_EXCLUDED_ROLES = frozenset(
    {
        ColumnRoleName.MEASURE,
        ColumnRoleName.TIMESTAMP,
        ColumnRoleName.TEXT,
        ColumnRoleName.SEQUENCE,
    }
)
# DI10-W1 e-commerce pack applicability gate.
_ECOMMERCE_MIN_SIGNALS = 2
_ECOMMERCE_NAME_TOKENS = frozenset(
    {
        "order",
        "orders",
        "customer",
        "customers",
        "product",
        "products",
        "sale",
        "sales",
        "ecommerce",
        "commerce",
        "retail",
        "shop",
        "store",
        "cart",
        "checkout",
        "purchase",
        "purchases",
        "seller",
        "sellers",
        "item",
        "items",
    }
)
_ECOMMERCE_ENTITY_KEY_TOKENS = frozenset({"product", "sku", "seller", "merchant", "shop", "store"})
# A repeat-purchase key must repeat within the table.
_SAME_TABLE_CUSTOMER_MAX_UNIQUE_PERCENT = 99.0
_MISSING_HOTSPOT_THRESHOLD_PERCENT = 5.0
_MISSING_HOTSPOT_MAX_COLUMNS = 3
_HHI_MIN_BUCKETS = 2
_HHI_MAX_BUCKETS = 50


class MetricRequirement(BaseModel):
    """Declarative requirement statement for one registered metric."""

    roles: list[str] = Field(default_factory=list)
    column_patterns: dict[str, list[str]] = Field(default_factory=dict)
    cross_table: bool = False
    join_requirement: str = ""
    notes: str = ""


class MetricInputConstraint(BaseModel):
    """Declarative input-domain requirement used by metric resolvers."""

    kind: Literal["nonnegative_measure", "timestamp_role", "additive_measure"]
    field_role: str = ""


class MetricPostcondition(BaseModel):
    """One deterministic condition checked before a metric result is published."""

    field: str
    kind: Literal["required", "finite", "range", "minimum"]
    minimum: float | None = None
    maximum: float | None = None
    code: str


class MetricContractResult(BaseModel):
    valid: bool
    code: str = ""
    reason: str = ""


class MetricDefinition(BaseModel):
    """One registered domain metric with requirements and result interpretation."""

    metric_id: str
    name_en: str
    domain: MetricDomain
    requirement: MetricRequirement
    # Deterministic result-interpretation skeletons; numeric placeholders are
    # filled downstream from SQL execution results only.
    interpretation_en: str
    input_constraints: list[MetricInputConstraint] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    postconditions: list[MetricPostcondition] = Field(default_factory=list)
    abstention_codes: list[str] = Field(default_factory=list)


class ResolvedMetric(BaseModel):
    """A metric with concrete column/table bindings and generated SQL."""

    metric_id: str
    name_en: str
    domain: MetricDomain
    target_datasets: list[str]
    referenced_columns: dict[str, list[str]]
    required_relations: list[str] = Field(default_factory=list)
    sql: str
    question_en: str
    interpretation_en: str
    output_units: dict[str, str] = Field(default_factory=dict)


class MetricSkip(BaseModel):
    """Describe a metric that did not apply and its machine-readable reason."""

    metric_id: str
    name_en: str
    domain: MetricDomain
    reason: str


class DomainMetricResolution(BaseModel):
    """Full outcome of the deterministic applicability pass."""

    resolved: list[ResolvedMetric] = Field(default_factory=list)
    skipped: list[MetricSkip] = Field(default_factory=list)


# Deterministic helpers (pure; no LLM anywhere in this module).
def _name_tokens(name: str) -> list[str]:
    normalized = name.strip().lower().replace(" ", "_")
    return [token for token in re.split(r"[^0-9a-z]+", normalized) if token]


def _display_name(dataset_name: str) -> str:
    stem = Path(dataset_name).stem
    words = [word for word in re.split(r"[_\-]+", stem) if word]
    return " ".join(word.capitalize() for word in words) or "Dataset"


def _num(column: str) -> str:
    return f"try_cast({_quote_identifier(column)} as double)"


def _ts(column: str) -> str:
    return f"try_cast({_quote_identifier(column)} as timestamp)"


def _relation(profile: DatasetProfile) -> str:
    return _quote_identifier(_relation_name(profile.dataset_id))


def _column_detail(profile: DatasetProfile, column: str) -> ColumnProfile | None:
    for detail in profile.columns_detail:
        if detail.name == column:
            return detail
    return None


def _verified_role_columns(
    profile: DatasetProfile, role_set: ColumnRoleSet, role: ColumnRoleName
) -> list[str]:
    """Columns holding a *verified* (inferred/seeded) role, in profile order."""
    names = {
        item.column
        for item in role_set.roles
        if item.role is role and item.provenance in ("inferred", "seeded")
    }
    return [name for name in profile.column_names if name in names]


def _pattern_columns(columns: Sequence[str], tokens: frozenset[str]) -> list[str]:
    return [column for column in columns if set(_name_tokens(column)) & tokens]


def _key_columns(
    profile: DatasetProfile, role_set: ColumnRoleSet, tokens: frozenset[str]
) -> list[str]:
    """Order/customer key candidates: entity tokens + key-shaped suffix."""
    out: list[str] = []
    for column in profile.column_names:
        parts = _name_tokens(column)
        token_set = set(parts)
        if not (token_set & tokens):
            continue
        if not parts or parts[-1] not in _KEY_SUFFIX_TOKENS:
            continue
        if token_set & _KEY_EXCLUDE_TOKENS:
            continue
        role = role_set.role_of(column)
        if (
            role is not None
            and role.provenance in ("inferred", "seeded")
            and role.role in _KEY_EXCLUDED_ROLES
        ):
            continue
        out.append(column)
    return out


def ecommerce_signals(profile: DatasetProfile, role_set: ColumnRoleSet) -> frozenset[str]:
    """Independent e-commerce evidence signals present on one dataset."""
    signals: set[str] = set()
    measures = _verified_role_columns(profile, role_set, ColumnRoleName.MEASURE)
    if _pattern_columns(measures, _PRICE_TOKENS | _FREIGHT_TOKENS):
        signals.add("price_freight_measure")
    entity_tokens = _ORDER_TOKENS | _CUSTOMER_TOKENS | _ECOMMERCE_ENTITY_KEY_TOKENS
    if _key_columns(profile, role_set, entity_tokens):
        signals.add("entity_key")
    if set(_name_tokens(Path(profile.name).stem)) & _ECOMMERCE_NAME_TOKENS:
        signals.add("table_name")
    return frozenset(signals)


def _ecommerce_enabled(profile: DatasetProfile, role_set: ColumnRoleSet) -> bool:
    return len(ecommerce_signals(profile, role_set)) >= _ECOMMERCE_MIN_SIGNALS


def carrier_basis_note(columns: Sequence[str]) -> str | None:
    """English basis disclosure when a metric's timestamp is carrier-tier."""
    for column in columns:
        if set(_name_tokens(column)) & _CARRIER_TS_TOKENS:
            return _CARRIER_BASIS_NOTE_EN
    return None


@dataclass(frozen=True)
class _ResolutionContext:
    """Everything a resolver may consult; all of it deterministic artifacts."""

    profiles: tuple[DatasetProfile, ...]  # sorted by dataset name
    role_sets: Mapping[str, ColumnRoleSet]
    whitelist: JoinWhitelist | None
    semantic_seeds: SemanticSeeds | None = None

    def role_set(self, profile: DatasetProfile) -> ColumnRoleSet | None:
        return self.role_sets.get(profile.name)

    def profile_by_name(self, name: str) -> DatasetProfile | None:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return None

    def field_units(self, dataset: str, column: str) -> set[str]:
        if self.semantic_seeds is None:
            return set()
        return {
            field.unit.strip()
            for field in self.semantic_seeds.field_meanings
            if field.dataset == dataset
            and field.column == column
            and field.unit is not None
            and field.unit.strip()
        }


class _SkipTracker:
    """Keeps the most *specific* skip reason seen across datasets."""

    def __init__(self) -> None:
        self._stage = -1
        self._reason = "no_dataset_profiles"

    def note(self, stage: int, reason: str) -> None:
        if stage > self._stage:
            self._stage = stage
            self._reason = reason

    def skip(self, definition: MetricDefinition) -> MetricSkip:
        return MetricSkip(
            metric_id=definition.metric_id,
            name_en=definition.name_en,
            domain=definition.domain,
            reason=self._reason,
        )


def _resolved(
    definition: MetricDefinition,
    *,
    target_datasets: list[str],
    referenced_columns: dict[str, list[str]],
    sql: str,
    question_en: str,
    required_relations: list[str] | None = None,
    interpretation_note_en: str = "",
    output_units: Mapping[str, str] | None = None,
) -> ResolvedMetric:
    return ResolvedMetric(
        metric_id=definition.metric_id,
        name_en=definition.name_en,
        domain=definition.domain,
        target_datasets=target_datasets,
        referenced_columns=referenced_columns,
        required_relations=list(required_relations or []),
        sql=sql,
        question_en=question_en,
        # DI10-W1: a degraded binding (e.g. carrier-handoff late basis) appends
        # its basis disclosure to the deterministic interpretation skeleton.
        interpretation_en=definition.interpretation_en + interpretation_note_en,
        output_units=dict(output_units or definition.units),
    )


def _single_field_unit(
    ctx: _ResolutionContext,
    tracker: _SkipTracker,
    *,
    dataset: str,
    column: str,
) -> tuple[str | None, bool]:
    units = ctx.field_units(dataset, column)
    if len(units) > 1:
        tracker.note(4, f"conflicting_unit_seed:{dataset}.{column}")
        return None, False
    return (next(iter(units)) if units else None), True


# SQL generators — pure functions from resolved bindings to deterministic SQL.
def _gmv_sql(profile: DatasetProfile, price_column: str) -> str:
    price = _num(price_column)
    return f"""
select
    count(*) as row_count,
    sum({price}) as gmv_total,
    avg({price}) as avg_line_value
from {_relation(profile)}
where {price} is not null
limit 1
""".strip()


def _aov_sql(profile: DatasetProfile, price_column: str, order_column: str) -> str:
    price = _num(price_column)
    order_key = _quote_identifier(order_column)
    return f"""
select
    count(*) as row_count,
    count(distinct {order_key}) as order_count,
    sum({price}) as gmv_total,
    sum({price}) / nullif(count(distinct {order_key}), 0) as avg_order_value
from {_relation(profile)}
where {price} is not null and {order_key} is not null
limit 1
""".strip()


def _repeat_rate_same_table_sql(
    profile: DatasetProfile, customer_column: str, order_column: str
) -> str:
    customer = _quote_identifier(customer_column)
    order_key = _quote_identifier(order_column)
    return f"""
with per_customer as (
    select {customer} as customer_key, count(distinct {order_key}) as order_count
    from {_relation(profile)}
    where {customer} is not null and {order_key} is not null
    group by 1
)
select
    count(*) as row_count,
    sum(case when order_count >= 2 then 1 else 0 end) as repeat_customers,
    100.0 * sum(case when order_count >= 2 then 1 else 0 end)
        / nullif(count(*), 0) as repeat_purchase_rate_percent
from per_customer
limit 1
""".strip()


def _repeat_rate_cross_table_sql(
    order_profile: DatasetProfile,
    customer_profile: DatasetProfile,
    *,
    order_column: str,
    customer_column: str,
    order_join_columns: Sequence[str],
    customer_join_columns: Sequence[str],
) -> str:
    join_condition = " and ".join(
        f"o.{_quote_identifier(left)} = c.{_quote_identifier(right)}"
        for left, right in zip(order_join_columns, customer_join_columns, strict=True)
    )
    customer = f"c.{_quote_identifier(customer_column)}"
    order_key = f"o.{_quote_identifier(order_column)}"
    return f"""
with per_customer as (
    select {customer} as customer_key, count(distinct {order_key}) as order_count
    from {_relation(order_profile)} as o
    join {_relation(customer_profile)} as c
        on {join_condition}
    where {customer} is not null and {order_key} is not null
    group by 1
)
select
    count(*) as row_count,
    sum(case when order_count >= 2 then 1 else 0 end) as repeat_customers,
    100.0 * sum(case when order_count >= 2 then 1 else 0 end)
        / nullif(count(*), 0) as repeat_purchase_rate_percent
from per_customer
limit 1
""".strip()


def _late_rate_sql(profile: DatasetProfile, promised_column: str, actual_column: str) -> str:
    promised = _ts(promised_column)
    actual = _ts(actual_column)
    return f"""
select
    count(*) as row_count,
    sum(case when {actual} > {promised} then 1 else 0 end) as late_rows,
    100.0 * sum(case when {actual} > {promised} then 1 else 0 end)
        / nullif(count(*), 0) as late_delivery_rate_percent
from {_relation(profile)}
where {actual} is not null and {promised} is not null
limit 1
""".strip()


def _fulfillment_sql(profile: DatasetProfile, start_column: str, end_column: str) -> str:
    start = _ts(start_column)
    end = _ts(end_column)
    return f"""
select
    count(*) as row_count,
    avg(date_diff('day', {start}, {end})) as avg_fulfillment_days,
    median(date_diff('day', {start}, {end})) as median_fulfillment_days,
    max(date_diff('day', {start}, {end})) as max_fulfillment_days
from {_relation(profile)}
where {start} is not null and {end} is not null and {end} >= {start}
limit 1
""".strip()


def _freight_ratio_sql(profile: DatasetProfile, price_column: str, freight_column: str) -> str:
    price = _num(price_column)
    freight = _num(freight_column)
    return f"""
select
    count(*) as row_count,
    sum({freight}) as freight_total,
    sum({price}) as merchandise_total,
    100.0 * sum({freight}) / nullif(sum({price}) + sum({freight}), 0)
        as freight_share_percent
from {_relation(profile)}
where {price} is not null and {freight} is not null
limit 1
""".strip()


def _hhi_sql(profile: DatasetProfile, dimension_column: str, measure_column: str) -> str:
    dimension = _quote_identifier(dimension_column)
    measure = _num(measure_column)
    return f"""
with grouped as (
    select cast({dimension} as varchar) as bucket, sum({measure}) as bucket_total
    from {_relation(profile)}
    where {dimension} is not null and {measure} is not null
    group by 1
),
stats as (
    select min({measure}) as min_measure, sum({measure}) as total_measure
    from {_relation(profile)}
    where {dimension} is not null and {measure} is not null
),
shares as (
    select bucket, bucket_total,
        bucket_total / nullif(sum(bucket_total) over (), 0) as share
    from grouped
)
select
    count(*) as row_count,
    sum(share * share) as hhi,
    max(share) as top_share
from shares
where (select min_measure from stats) >= 0
  and (select total_measure from stats) > 0
limit 1
""".strip()


def _missing_hotspot_sql(profile: DatasetProfile, columns: Sequence[str]) -> str:
    parts: list[str] = []
    for column in columns:
        quoted = _quote_identifier(column)
        alias = safe_alias(column)
        parts.append(f"sum(case when {quoted} is null then 1 else 0 end) as missing_{alias}")
        parts.append(
            f"100.0 * sum(case when {quoted} is null then 1 else 0 end)"
            f" / nullif(count(*), 0) as missing_{alias}_percent"
        )
    joined = ",\n    ".join(parts)
    return f"""
select
    count(*) as row_count,
    {joined}
from {_relation(profile)}
limit 1
""".strip()


def _time_coverage_sql(profile: DatasetProfile, timestamp_column: str) -> str:
    stamp = _ts(timestamp_column)
    return f"""
select
    count(*) as row_count,
    min({stamp}) as first_timestamp,
    max({stamp}) as last_timestamp,
    date_diff('day', min({stamp}), max({stamp})) as span_days,
    count(distinct date_trunc('month', {stamp})) as covered_months
from {_relation(profile)}
where {stamp} is not null
limit 1
""".strip()


# Per-metric resolvers: (definition, context) -> ResolvedMetric | MetricSkip.
def _price_measure(
    profile: DatasetProfile, role_set: ColumnRoleSet, tracker: _SkipTracker
) -> str | None:
    measures = _verified_role_columns(profile, role_set, ColumnRoleName.MEASURE)
    if not measures:
        tracker.note(1, "missing_role:measure")
        return None
    price_columns = _pattern_columns(measures, _PRICE_TOKENS)
    if not price_columns:
        tracker.note(2, "missing_column_pattern:price")
        return None
    return price_columns[0]


def _resolve_gmv(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        price = _price_measure(profile, role_set, tracker)
        if price is None:
            continue
        source_unit, unit_ok = _single_field_unit(
            ctx, tracker, dataset=profile.name, column=price
        )
        if not unit_ok:
            continue
        output_units = dict(definition.units)
        if source_unit is not None:
            output_units["gmv_total"] = source_unit
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [price]},
            sql=_gmv_sql(profile, price),
            question_en=f"What is the total GMV (sum of {price}) in {display}?",
            output_units=output_units,
        )
    return tracker.skip(definition)


def _resolve_aov(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        price = _price_measure(profile, role_set, tracker)
        if price is None:
            continue
        order_columns = _key_columns(profile, role_set, _ORDER_TOKENS)
        if not order_columns:
            tracker.note(3, "missing_column_pattern:order_key")
            continue
        order_column = order_columns[0]
        source_unit, unit_ok = _single_field_unit(
            ctx, tracker, dataset=profile.name, column=price
        )
        if not unit_ok:
            continue
        output_units = dict(definition.units)
        if source_unit is not None:
            output_units["avg_order_value"] = f"{source_unit}/order"
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [price, order_column]},
            sql=_aov_sql(profile, price, order_column),
            question_en=(
                f"What is the average order value ({price} per distinct "
                f"{order_column}) in {display}?"
            ),
            output_units=output_units,
        )
    return tracker.skip(definition)


def _repeat_rate_pair(
    entry: JoinWhitelistEntry, ctx: _ResolutionContext
) -> tuple[DatasetProfile, DatasetProfile, str, str, list[str], list[str]] | None:
    """Try both orientations of a whitelist entry as (order side, customer side)."""
    if not entry.left_columns or len(entry.left_columns) != len(entry.right_columns):
        return None
    orientations = (
        (entry.left_dataset, entry.left_columns, entry.right_dataset, entry.right_columns),
        (entry.right_dataset, entry.right_columns, entry.left_dataset, entry.left_columns),
    )
    for order_name, order_join, customer_name, customer_join in orientations:
        order_profile = ctx.profile_by_name(order_name)
        customer_profile = ctx.profile_by_name(customer_name)
        if order_profile is None or customer_profile is None:
            continue
        order_roles = ctx.role_set(order_profile)
        customer_roles = ctx.role_set(customer_profile)
        if order_roles is None or customer_roles is None:
            continue
        # DI10-W1 gate: both sides of the join must independently look like
        # e-commerce data before a repeat-purchase rate may bind across them.
        if not _ecommerce_enabled(order_profile, order_roles) or not (
            _ecommerce_enabled(customer_profile, customer_roles)
        ):
            continue
        order_columns = _key_columns(order_profile, order_roles, _ORDER_TOKENS)
        customer_columns = [
            column
            for column in _key_columns(customer_profile, customer_roles, _CUSTOMER_TOKENS)
            if column not in set(customer_join)
        ]
        if order_columns and customer_columns:
            return (
                order_profile,
                customer_profile,
                order_columns[0],
                customer_columns[0],
                list(order_join),
                list(customer_join),
            )
    return None


def _resolve_repeat_purchase(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    # Same-table route: a customer key that actually repeats + an order key.
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        order_columns = _key_columns(profile, role_set, _ORDER_TOKENS)
        customer_columns = [
            column
            for column in _key_columns(profile, role_set, _CUSTOMER_TOKENS)
            if column not in set(order_columns)
        ]
        if not customer_columns:
            tracker.note(1, "missing_column_pattern:customer")
            continue
        if not order_columns:
            tracker.note(1, "missing_column_pattern:order_key")
            continue
        customer_column = customer_columns[0]
        detail = _column_detail(profile, customer_column)
        if (
            detail is None
            or detail.unique_count < 2
            or detail.unique_percent > _SAME_TABLE_CUSTOMER_MAX_UNIQUE_PERCENT
        ):
            # A per-row-unique customer key (Olist orders.customer_id) makes the
            # same-table rate trivially 0 — fall through to the join route.
            tracker.note(2, "customer_key_unique_per_row")
            continue
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [customer_column, order_columns[0]]},
            sql=_repeat_rate_same_table_sql(profile, customer_column, order_columns[0]),
            question_en=(
                f"What share of customers ({customer_column}) placed two or more "
                f"orders ({order_columns[0]}) in {display}?"
            ),
        )
    # Cross-table route: requires a CONFIRMED whitelist join (H9-C red line —
    # execution re-checks the guard, but applicability already refuses here).
    if ctx.whitelist is not None:
        dataset_ids_by_name = {profile.name: profile.dataset_id for profile in ctx.profiles}
        scoped = ctx.whitelist.entries_for({profile.name for profile in ctx.profiles})
        entries = sorted(scoped, key=lambda entry: entry.label())
        for entry in entries:
            if not entry.is_usable(dataset_ids_by_name):
                continue
            pair = _repeat_rate_pair(entry, ctx)
            if pair is None:
                continue
            order_profile, customer_profile, order_column, customer_column, oj, cj = pair
            label = entry.label()
            order_display = _display_name(order_profile.name)
            customer_display = _display_name(customer_profile.name)
            return _resolved(
                definition,
                target_datasets=[order_profile.name, customer_profile.name],
                referenced_columns={
                    order_profile.name: [order_column, *oj],
                    customer_profile.name: [customer_column, *cj],
                },
                required_relations=[label],
                sql=_repeat_rate_cross_table_sql(
                    order_profile,
                    customer_profile,
                    order_column=order_column,
                    customer_column=customer_column,
                    order_join_columns=oj,
                    customer_join_columns=cj,
                ),
                question_en=(
                    f"What share of customers ({customer_column}) placed two or "
                    f"more orders across {order_display} and {customer_display}?"
                ),
            )
        # No usable entry fit; if an unconfirmed one WOULD fit, say so.
        for entry in entries:
            if entry.is_usable(dataset_ids_by_name) or entry.cardinality == "many_to_many":
                continue
            if _repeat_rate_pair(entry, ctx) is not None:
                tracker.note(3, "join_not_confirmed")
                break
    return tracker.skip(definition)


def _actual_delivery_columns(stamps: Sequence[str]) -> tuple[list[str], bool]:
    """Actual-delivery endpoint candidates, customer tier before carrier tier."""
    all_actual = [
        column
        for column in _pattern_columns(stamps, _ACTUAL_TS_TOKENS)
        if not (set(_name_tokens(column)) & _PROMISED_TS_TOKENS)
    ]
    customer_tier = [
        column for column in all_actual if not (set(_name_tokens(column)) & _CARRIER_TS_TOKENS)
    ]
    if customer_tier:
        return customer_tier, False
    return all_actual, bool(all_actual)


def _resolve_late_delivery(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        stamps = _verified_role_columns(profile, role_set, ColumnRoleName.TIMESTAMP)
        if not stamps:
            tracker.note(1, "missing_role:timestamp")
            continue
        promised_columns = _pattern_columns(stamps, _PROMISED_TS_TOKENS)
        actual_columns, carrier_only = _actual_delivery_columns(stamps)
        if not promised_columns:
            tracker.note(2, "missing_column_pattern:promised_timestamp")
            continue
        if not actual_columns:
            tracker.note(2, "missing_column_pattern:actual_timestamp")
            continue
        promised, actual = promised_columns[0], actual_columns[0]
        display = _display_name(profile.name)
        basis_en = "; customer delivery date unavailable" if carrier_only else ""
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [promised, actual]},
            sql=_late_rate_sql(profile, promised, actual),
            question_en=(
                f"What share of rows in {display} were late ({actual} after {promised}{basis_en})?"
            ),
            interpretation_note_en=_CARRIER_BASIS_NOTE_EN if carrier_only else "",
        )
    return tracker.skip(definition)


def _resolve_fulfillment_time(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        stamps = _verified_role_columns(profile, role_set, ColumnRoleName.TIMESTAMP)
        if not stamps:
            tracker.note(1, "missing_role:timestamp")
            continue
        # DI10-W1 basis fix: fulfillment ends at customer delivery; carrier
        # handoff is only a disclosed fallback (see _actual_delivery_columns).
        end_columns, carrier_only = _actual_delivery_columns(stamps)
        start_columns = [
            column
            for column in _pattern_columns(stamps, _START_TS_TOKENS)
            if not (set(_name_tokens(column)) & (_ACTUAL_TS_TOKENS | _PROMISED_TS_TOKENS))
        ]
        if not start_columns:
            tracker.note(2, "missing_column_pattern:start_timestamp")
            continue
        if not end_columns:
            tracker.note(2, "missing_column_pattern:actual_timestamp")
            continue
        start, end = start_columns[0], end_columns[0]
        display = _display_name(profile.name)
        basis_en = "; customer delivery date unavailable" if carrier_only else ""
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [start, end]},
            sql=_fulfillment_sql(profile, start, end),
            question_en=(
                f"How long does fulfillment take in {display} (from {start} to {end}{basis_en})?"
            ),
            interpretation_note_en=(_CARRIER_FULFILLMENT_NOTE_EN if carrier_only else ""),
        )
    return tracker.skip(definition)


def _resolve_freight_ratio(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        if not _ecommerce_enabled(profile, role_set):
            tracker.note(1, "domain_signals_insufficient")
            continue
        measures = _verified_role_columns(profile, role_set, ColumnRoleName.MEASURE)
        if not measures:
            tracker.note(1, "missing_role:measure")
            continue
        price_columns = _pattern_columns(measures, _PRICE_TOKENS)
        freight_columns = _pattern_columns(measures, _FREIGHT_TOKENS)
        if not price_columns:
            tracker.note(2, "missing_column_pattern:price")
            continue
        if not freight_columns:
            tracker.note(2, "missing_column_pattern:freight")
            continue
        price, freight = price_columns[0], freight_columns[0]
        price_unit, price_ok = _single_field_unit(
            ctx, tracker, dataset=profile.name, column=price
        )
        freight_unit, freight_ok = _single_field_unit(
            ctx, tracker, dataset=profile.name, column=freight
        )
        if not price_ok or not freight_ok:
            continue
        if (
            price_unit is not None
            and freight_unit is not None
            and price_unit != freight_unit
        ):
            tracker.note(4, f"input_unit_mismatch:{price_unit}!={freight_unit}")
            continue
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [price, freight]},
            sql=_freight_ratio_sql(profile, price, freight),
            question_en=(
                f"What share of total value is freight ({freight} vs {price}) in {display}?"
            ),
        )
    return tracker.skip(definition)


def _resolve_concentration_hhi(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        measures = [
            column
            for column in _verified_role_columns(profile, role_set, ColumnRoleName.MEASURE)
            if _is_additive_nonnegative_measure(profile, column)
        ]
        if not measures:
            tracker.note(1, "missing_input_domain:additive_nonnegative_measure")
            continue
        dimensions: list[str] = []
        for column in _verified_role_columns(profile, role_set, ColumnRoleName.DIMENSION):
            parts = _name_tokens(column)
            if parts and parts[-1] in _KEY_SUFFIX_TOKENS:
                continue  # an id-suffixed bucket is a key, not a business dimension
            detail = _column_detail(profile, column)
            if detail is None or not (_HHI_MIN_BUCKETS <= detail.unique_count <= _HHI_MAX_BUCKETS):
                continue
            dimensions.append(column)
        if not dimensions:
            tracker.note(2, "missing_role:dimension")
            continue
        dimension, measure = dimensions[0], measures[0]
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [dimension, measure]},
            sql=_hhi_sql(profile, dimension, measure),
            question_en=(
                f"How concentrated is {measure} across {dimension} groups in {display} (HHI)?"
            ),
        )
    return tracker.skip(definition)


_NON_ADDITIVE_MEASURE_TOKENS = frozenset(
    {"avg", "average", "mean", "percent", "percentage", "ratio", "rate", "score", "time"}
)
_ADDITIVE_MASS_TOKENS = frozenset(
    {
        "amount",
        "balance",
        "capacity",
        "cost",
        "count",
        "freight",
        "gmv",
        "price",
        "quantity",
        "qty",
        "revenue",
        "sales",
        "spend",
        "unit",
        "units",
        "usage",
        "users",
        "value",
        "volume",
    }
)
_ANONYMIZED_COMPONENT_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def _is_additive_nonnegative_measure(profile: DatasetProfile, column: str) -> bool:
    """Conservative HHI input-domain gate."""
    tokens = set(_name_tokens(column))
    if tokens & _NON_ADDITIVE_MEASURE_TOKENS:
        return False
    # Require an explicit money, quantity, or volume signal.
    if not tokens & _ADDITIVE_MASS_TOKENS:
        return False
    if _ANONYMIZED_COMPONENT_RE.fullmatch(column.strip()):
        return False
    detail = _column_detail(profile, column)
    if detail is None:
        return False
    parsed_samples: list[float] = []
    for value in detail.sample_values:
        try:
            parsed_samples.append(float(value))
        except (TypeError, ValueError):
            return False
    return bool(parsed_samples) and all(
        math.isfinite(value) and value >= 0 for value in parsed_samples
    )


def _resolve_missing_hotspots(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        if ctx.role_set(profile) is None:
            tracker.note(0, "missing_role_set")
            continue
        hotspots = sorted(
            (
                detail
                for detail in profile.columns_detail
                if detail.missing_percent >= _MISSING_HOTSPOT_THRESHOLD_PERCENT
            ),
            key=lambda detail: (-detail.missing_percent, detail.name),
        )[:_MISSING_HOTSPOT_MAX_COLUMNS]
        if not hotspots:
            tracker.note(1, "no_high_missing_column")
            continue
        columns = [detail.name for detail in hotspots]
        display = _display_name(profile.name)
        joined = ", ".join(columns)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: columns},
            sql=_missing_hotspot_sql(profile, columns),
            question_en=(f"How severe are the missing-data hotspots ({joined}) in {display}?"),
        )
    return tracker.skip(definition)


def _resolve_time_coverage(
    definition: MetricDefinition, ctx: _ResolutionContext
) -> ResolvedMetric | MetricSkip:
    tracker = _SkipTracker()
    for profile in ctx.profiles:
        role_set = ctx.role_set(profile)
        if role_set is None:
            tracker.note(0, "missing_role_set")
            continue
        stamps = _verified_role_columns(profile, role_set, ColumnRoleName.TIMESTAMP)
        if not stamps:
            tracker.note(1, "missing_role:timestamp")
            continue
        stamp = stamps[0]
        display = _display_name(profile.name)
        return _resolved(
            definition,
            target_datasets=[profile.name],
            referenced_columns={profile.name: [stamp]},
            sql=_time_coverage_sql(profile, stamp),
            question_en=(
                f"What time span does {stamp} cover in {display} "
                f"(first/last timestamp, covered months)?"
            ),
        )
    return tracker.skip(definition)


# The registry: declarative definitions + their deterministic resolvers.
DOMAIN_METRIC_REGISTRY: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        metric_id="gmv",
        name_en="GMV (gross merchandise value)",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["measure"],
            column_patterns={"value": sorted(_PRICE_TOKENS)},
        ),
        interpretation_en="Total GMV over {row_count} rows is {gmv_total}.",
        output_fields=["row_count", "gmv_total"],
        units={"row_count": "count", "gmv_total": "currency"},
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(field="gmv_total", kind="finite", code="non_finite_metric_output"),
        ],
        abstention_codes=["empty_metric_input", "non_finite_metric_output"],
    ),
    MetricDefinition(
        metric_id="aov",
        name_en="Average order value",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["measure"],
            column_patterns={
                "value": sorted(_PRICE_TOKENS),
                "order_key": sorted(_ORDER_TOKENS),
            },
        ),
        interpretation_en=(
            "Across {order_count} orders the average order value is {avg_order_value}."
        ),
        output_fields=["order_count", "avg_order_value"],
        units={"order_count": "count", "avg_order_value": "currency_per_order"},
        postconditions=[
            MetricPostcondition(
                field="order_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="avg_order_value", kind="finite", code="non_finite_metric_output"
            ),
        ],
        abstention_codes=["empty_metric_input", "non_finite_metric_output"],
    ),
    MetricDefinition(
        metric_id="repeat_purchase_rate",
        name_en="Repeat-purchase rate",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["identifier"],
            column_patterns={
                "order_key": sorted(_ORDER_TOKENS),
                "customer_key": sorted(_CUSTOMER_TOKENS),
            },
            cross_table=True,
            join_requirement=(
                "A same-table customer key with actual repeats, or a CONFIRMED "
                "join-whitelist entry linking the order table to the customer table."
            ),
        ),
        interpretation_en=(
            "{repeat_customers} of {row_count} customers purchased again "
            "({repeat_purchase_rate_percent}%)."
        ),
        output_fields=["row_count", "repeat_customers", "repeat_purchase_rate_percent"],
        units={
            "row_count": "count",
            "repeat_customers": "count",
            "repeat_purchase_rate_percent": "percent",
        },
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="repeat_purchase_rate_percent",
                kind="range",
                minimum=0,
                maximum=100,
                code="percentage_out_of_range",
            ),
        ],
        abstention_codes=["empty_metric_input", "percentage_out_of_range"],
    ),
    MetricDefinition(
        metric_id="late_delivery_rate",
        name_en="Late-delivery rate",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["timestamp", "timestamp"],
            column_patterns={
                "promised": sorted(_PROMISED_TS_TOKENS),
                "actual": sorted(_ACTUAL_TS_TOKENS),
            },
        ),
        interpretation_en=(
            "{late_rows} of {row_count} rows were late ({late_delivery_rate_percent}%)."
        ),
        output_fields=["row_count", "late_rows", "late_delivery_rate_percent"],
        units={
            "row_count": "count",
            "late_rows": "count",
            "late_delivery_rate_percent": "percent",
        },
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="late_delivery_rate_percent",
                kind="range",
                minimum=0,
                maximum=100,
                code="percentage_out_of_range",
            ),
        ],
        abstention_codes=["empty_metric_input", "percentage_out_of_range"],
    ),
    MetricDefinition(
        metric_id="fulfillment_time",
        name_en="Fulfillment time",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["timestamp", "timestamp"],
            column_patterns={
                "start": sorted(_START_TS_TOKENS),
                "actual": sorted(_ACTUAL_TS_TOKENS),
            },
        ),
        interpretation_en=(
            "Average fulfillment takes {avg_fulfillment_days} days "
            "(median {median_fulfillment_days})."
        ),
        output_fields=["row_count", "avg_fulfillment_days", "median_fulfillment_days"],
        units={
            "row_count": "count",
            "avg_fulfillment_days": "days",
            "median_fulfillment_days": "days",
        },
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="avg_fulfillment_days",
                kind="minimum",
                minimum=0,
                code="duration_out_of_range",
            ),
            MetricPostcondition(
                field="median_fulfillment_days",
                kind="minimum",
                minimum=0,
                code="duration_out_of_range",
            ),
        ],
        abstention_codes=["empty_metric_input", "duration_out_of_range"],
    ),
    MetricDefinition(
        metric_id="freight_ratio",
        name_en="Freight share of value",
        domain="ecommerce",
        requirement=MetricRequirement(
            roles=["measure", "measure"],
            column_patterns={
                "value": sorted(_PRICE_TOKENS),
                "freight": sorted(_FREIGHT_TOKENS),
            },
        ),
        interpretation_en="Freight is {freight_share_percent}% of total value.",
        output_fields=["row_count", "freight_share_percent"],
        units={"row_count": "count", "freight_share_percent": "percent"},
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="freight_share_percent",
                kind="range",
                minimum=0,
                maximum=100,
                code="percentage_out_of_range",
            ),
        ],
        abstention_codes=["empty_metric_input", "percentage_out_of_range"],
    ),
    MetricDefinition(
        metric_id="concentration_hhi",
        name_en="Concentration (HHI)",
        domain="generic",
        requirement=MetricRequirement(roles=["dimension", "measure"]),
        interpretation_en=("Across {row_count} groups the HHI is {hhi} (top share {top_share})."),
        input_constraints=[
            MetricInputConstraint(kind="nonnegative_measure", field_role="measure"),
            MetricInputConstraint(kind="additive_measure", field_role="measure"),
        ],
        output_fields=["row_count", "hhi", "top_share"],
        units={"row_count": "count", "hhi": "fraction", "top_share": "fraction"},
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="hhi",
                kind="range",
                minimum=0,
                maximum=1,
                code="hhi_out_of_range",
            ),
            MetricPostcondition(
                field="top_share",
                kind="range",
                minimum=0,
                maximum=1,
                code="share_out_of_range",
            ),
        ],
        abstention_codes=[
            "negative_measure_input",
            "empty_metric_input",
            "hhi_out_of_range",
            "share_out_of_range",
        ],
    ),
    MetricDefinition(
        metric_id="missing_hotspots",
        name_en="Missing-data hotspots",
        domain="generic",
        requirement=MetricRequirement(
            notes=(
                "Applies when at least one column has "
                f">= {_MISSING_HOTSPOT_THRESHOLD_PERCENT}% missing values."
            ),
        ),
        interpretation_en="The highest-missing columns and their null shares.",
    ),
    MetricDefinition(
        metric_id="time_coverage",
        name_en="Time-span coverage",
        domain="generic",
        requirement=MetricRequirement(roles=["timestamp"]),
        interpretation_en=(
            "Data spans {first_timestamp} to {last_timestamp} "
            "({span_days} days, {covered_months} months)."
        ),
        input_constraints=[MetricInputConstraint(kind="timestamp_role", field_role="timestamp")],
        output_fields=[
            "row_count",
            "first_timestamp",
            "last_timestamp",
            "span_days",
            "covered_months",
        ],
        units={
            "row_count": "count",
            "first_timestamp": "timestamp",
            "last_timestamp": "timestamp",
            "span_days": "days",
            "covered_months": "months",
        },
        postconditions=[
            MetricPostcondition(
                field="row_count", kind="minimum", minimum=1, code="empty_metric_input"
            ),
            MetricPostcondition(
                field="first_timestamp", kind="required", code="timestamp_endpoint_missing"
            ),
            MetricPostcondition(
                field="last_timestamp", kind="required", code="timestamp_endpoint_missing"
            ),
            MetricPostcondition(
                field="span_days", kind="minimum", minimum=0, code="duration_out_of_range"
            ),
            MetricPostcondition(
                field="covered_months",
                kind="minimum",
                minimum=1,
                code="time_coverage_empty",
            ),
        ],
        abstention_codes=[
            "empty_metric_input",
            "timestamp_endpoint_missing",
            "duration_out_of_range",
            "time_coverage_empty",
        ],
    ),
)


def metric_definition(metric_id: str) -> MetricDefinition | None:
    """Return a registry definition by stable id (legacy-safe)."""
    return next(
        (definition for definition in DOMAIN_METRIC_REGISTRY if definition.metric_id == metric_id),
        None,
    )


def validate_metric_result(metric_id: str, row: Mapping[str, object]) -> MetricContractResult:
    """Evaluate a registered metric's deterministic publication contract."""
    definition = metric_definition(metric_id)
    if definition is None:
        return MetricContractResult(
            valid=False,
            code="unknown_metric_contract",
            reason=f"No result contract is registered for metric {metric_id!r}.",
        )
    for field in definition.output_fields:
        if field not in row:
            return MetricContractResult(
                valid=False,
                code="missing_metric_output",
                reason=f"Required metric output {field!r} is missing.",
            )
    for condition in definition.postconditions:
        value = row.get(condition.field)
        if condition.kind == "required":
            if value is None or (isinstance(value, str) and not value.strip()):
                return _contract_failure(condition, "is missing")
            continue
        number = _finite_number(value)
        if number is None:
            return _contract_failure(condition, "is not a finite number")
        if condition.kind == "minimum" and condition.minimum is not None:
            if number < condition.minimum:
                return _contract_failure(condition, f"is below {condition.minimum:g}")
        elif condition.kind == "range":
            if condition.minimum is not None and number < condition.minimum:
                return _contract_failure(condition, f"is below {condition.minimum:g}")
            if condition.maximum is not None and number > condition.maximum:
                return _contract_failure(condition, f"is above {condition.maximum:g}")
    return MetricContractResult(valid=True)


def _contract_failure(condition: MetricPostcondition, detail: str) -> MetricContractResult:
    return MetricContractResult(
        valid=False,
        code=condition.code,
        reason=f"Metric field {condition.field!r} {detail}.",
    )


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_RESOLVERS: dict[
    str, Callable[[MetricDefinition, _ResolutionContext], ResolvedMetric | MetricSkip]
] = {
    "gmv": _resolve_gmv,
    "aov": _resolve_aov,
    "repeat_purchase_rate": _resolve_repeat_purchase,
    "late_delivery_rate": _resolve_late_delivery,
    "fulfillment_time": _resolve_fulfillment_time,
    "freight_ratio": _resolve_freight_ratio,
    "concentration_hhi": _resolve_concentration_hhi,
    "missing_hotspots": _resolve_missing_hotspots,
    "time_coverage": _resolve_time_coverage,
}


def applicable_metrics(
    *,
    role_sets: Mapping[str, ColumnRoleSet],
    join_whitelist: JoinWhitelist | None,
    profiles: Sequence[DatasetProfile],
    semantic_seeds: SemanticSeeds | None = None,
) -> DomainMetricResolution:
    """Purely deterministic applicability pass over the metric registry."""
    ordered = tuple(sorted(profiles, key=lambda profile: profile.name))
    ctx = _ResolutionContext(
        profiles=ordered,
        role_sets=role_sets,
        whitelist=join_whitelist,
        semantic_seeds=semantic_seeds,
    )
    resolved: list[ResolvedMetric] = []
    skipped: list[MetricSkip] = []
    for definition in DOMAIN_METRIC_REGISTRY:
        outcome = _RESOLVERS[definition.metric_id](definition, ctx)
        if isinstance(outcome, ResolvedMetric):
            resolved.append(outcome)
        else:
            skipped.append(outcome)
    return DomainMetricResolution(resolved=resolved, skipped=skipped)

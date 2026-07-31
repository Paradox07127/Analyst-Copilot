"""H9-C registered domain business-metric pack.

Applicability is purely deterministic: the DI8-B role layer supplies candidate
columns, column-name patterns refine them, and cross-table metrics require a
CONFIRMED DI8-C whitelist join. Metrics that do not apply are skipped with a
structured reason and counted on the trace. Every generated aggregate carries
a ``count(*)`` column (time-boundary convention). No LLM anywhere.
"""

from __future__ import annotations

from pathlib import Path

from semantic_test_helpers import save_seeds

from eda_platform.core.column_roles import ColumnRoleSet, infer_column_roles
from eda_platform.core.methods import METHOD_REGISTRY, MethodGateContext
from eda_platform.core.semantic import (
    FieldMeaning,
    JoinWhitelist,
    JoinWhitelistEntry,
    SemanticSeeds,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.schemas.artifacts import Artifact, DatasetProfile
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.tools.domain_metrics import (
    DOMAIN_METRIC_REGISTRY,
    applicable_metrics,
    validate_metric_result,
)
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.question_discovery import discover_question_candidates

JOIN_LABEL = "orders.csv.customer_id -> customers.csv.customer_id"


# --------------------------------------------------------------------------- #
# Fixtures: an Olist-shaped mini corpus.
#
# order_items.csv — order_id repeats per line item (so it is NOT a verified
# identifier; the key-column rule must still find it), price + freight_value
# measures, order_item_id is a per-order 1..n sequence.
# orders.csv      — purchase / delivered / estimated timestamps (late rate and
# fulfillment time), customer_id unique per order (forces the cross-table
# repeat-purchase route, exactly like Olist).
# customers.csv   — customer_unique_id repeats for one customer (the actual
# repeat-purchase entity), joined from orders over the whitelist.
# --------------------------------------------------------------------------- #
def _order_items_csv(tmp_path: Path) -> Path:
    rows = ["order_id,order_item_id,price,freight_value,category"]
    categories = ["books", "toys", "garden", "sports"]
    for order_index in range(10):
        for item in range(1, 4):
            price = 20.0 + order_index * 3.7 + item * 1.3
            freight = 4.0 + order_index * 0.53 + item * 0.21
            rows.append(
                f"O{order_index:03d},{item},{price:.2f},{freight:.2f},"
                f"{categories[(order_index + item) % len(categories)]}"
            )
    path = tmp_path / "order_items.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _orders_csv(tmp_path: Path) -> Path:
    rows = [
        "order_id,customer_id,order_status,order_purchase_timestamp,"
        "order_delivered_customer_date,order_estimated_delivery_date"
    ]
    for index in range(30):
        purchase_day = index % 28 + 1
        delivered_day = min(purchase_day + 3 + index % 4, 28)
        estimated_day = min(purchase_day + 5, 28)
        delivered = f"2026-01-{delivered_day:02d} 15:00:00"
        if index % 10 == 9:
            delivered = ""  # undelivered rows -> a missing hotspot
        rows.append(
            f"O{index:03d},C{index:03d},delivered,"
            f"2026-01-{purchase_day:02d} 10:00:00,{delivered},"
            f"2026-01-{estimated_day:02d} 23:59:59"
        )
    path = tmp_path / "orders.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _customers_csv(tmp_path: Path) -> Path:
    rows = ["customer_id,customer_unique_id,customer_state"]
    states = ["SP", "RJ", "MG"]
    for index in range(30):
        # One person (U000) owns two customer_id rows -> a genuine repeat buyer;
        # uniqueness stays >= 95% so the identifier role still verifies.
        unique = "U000" if index == 29 else f"U{index:03d}"
        rows.append(f"C{index:03d},{unique},{states[index % 3]}")
    path = tmp_path / "customers.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _load_with_roles(
    path: Path,
) -> tuple[LoadedDataset, Artifact, DatasetProfile, ColumnRoleSet]:
    loaded = load_csv(path, dataset_id=f"ds_{path.stem}")
    artifact = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    profile = DatasetProfile.model_validate(artifact.payload)
    role_set = infer_column_roles(profile, frame=loaded.frame)
    return loaded, artifact, profile, role_set


def _olist_corpus(
    tmp_path: Path,
) -> tuple[list[LoadedDataset], list[Artifact], list[DatasetProfile], dict[str, ColumnRoleSet]]:
    datasets: list[LoadedDataset] = []
    artifacts: list[Artifact] = []
    profiles: list[DatasetProfile] = []
    role_sets: dict[str, ColumnRoleSet] = {}
    for path in (
        _order_items_csv(tmp_path),
        _orders_csv(tmp_path),
        _customers_csv(tmp_path),
    ):
        loaded, artifact, profile, role_set = _load_with_roles(path)
        datasets.append(loaded)
        artifacts.append(artifact)
        profiles.append(profile)
        role_sets[profile.name] = role_set
    return datasets, artifacts, profiles, role_sets


def _whitelist(status: str = "confirmed") -> JoinWhitelist:
    return JoinWhitelist(
        entries=[
            JoinWhitelistEntry(
                left_dataset="orders.csv",
                left_dataset_id="ds_orders",
                left_columns=["customer_id"],
                right_dataset="customers.csv",
                right_dataset_id="ds_customers",
                right_columns=["customer_id"],
                cardinality="one_to_one",
                validation_verified=True,
                status=status,  # type: ignore[arg-type]
            )
        ]
    )


def _resolved_by_id(resolution) -> dict:  # noqa: ANN001 - helper for readability
    return {metric.metric_id: metric for metric in resolution.resolved}


def _skipped_by_id(resolution) -> dict:  # noqa: ANN001
    return {skip.metric_id: skip for skip in resolution.skipped}


# --------------------------------------------------------------------------- #
# 1. E-commerce pack: applicability positives on the Olist shape
# --------------------------------------------------------------------------- #
def test_ecommerce_pack_resolves_on_olist_shape(tmp_path: Path) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)

    resolution = applicable_metrics(
        role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles
    )
    resolved = _resolved_by_id(resolution)

    assert {
        "gmv",
        "aov",
        "repeat_purchase_rate",
        "late_delivery_rate",
        "fulfillment_time",
        "freight_ratio",
    } <= set(resolved)

    gmv = resolved["gmv"]
    assert gmv.target_datasets == ["order_items.csv"]
    assert gmv.referenced_columns == {"order_items.csv": ["price"]}
    assert "sum(" in gmv.sql

    aov = resolved["aov"]
    assert "count(distinct" in aov.sql
    assert "order_id" in aov.sql
    # order_item_id carries the "order" token but is a sequence, never a key.
    assert "order_item_id" not in aov.sql

    late = resolved["late_delivery_rate"]
    assert late.referenced_columns == {
        "orders.csv": ["order_estimated_delivery_date", "order_delivered_customer_date"]
    }

    fulfillment = resolved["fulfillment_time"]
    assert "order_purchase_timestamp" in fulfillment.sql
    assert "order_delivered_customer_date" in fulfillment.sql

    freight = resolved["freight_ratio"]
    assert set(freight.referenced_columns["order_items.csv"]) == {
        "price",
        "freight_value",
    }


def test_seeded_currency_specializes_money_outputs_and_candidates(
    tmp_path: Path,
) -> None:
    datasets, artifacts, profiles, role_sets = _olist_corpus(tmp_path)
    seeds = SemanticSeeds(
        field_meanings=[
            FieldMeaning(
                dataset="order_items.csv",
                column="price",
                meaning="Item price in Brazilian reais.",
                unit="BRL",
            )
        ]
    )

    resolution = applicable_metrics(
        role_sets=role_sets,
        join_whitelist=_whitelist(),
        profiles=profiles,
        semantic_seeds=seeds,
    )
    resolved = _resolved_by_id(resolution)
    assert resolved["gmv"].output_units["gmv_total"] == "BRL"
    assert resolved["aov"].output_units["avg_order_value"] == "BRL/order"

    candidates = discover_question_candidates(
        datasets,
        profile_artifacts=artifacts,
        column_role_sets=role_sets,
        join_whitelist=_whitelist(),
        semantic_seeds=seeds,
    )
    by_metric = {
        candidate.metric_id: candidate
        for candidate in candidates.candidates
        if candidate.template_id == "domain_metric"
    }
    assert by_metric["gmv"].produced_units["gmv_total"] == "BRL"
    assert by_metric["gmv"].answer_contract is not None
    assert by_metric["gmv"].answer_contract.expected_units["gmv_total"] == "currency"


def test_conflicting_or_mixed_seeded_units_fail_metric_resolution(
    tmp_path: Path,
) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)
    conflicting = SemanticSeeds(
        field_meanings=[
            FieldMeaning(
                dataset="order_items.csv", column="price", meaning="Price.", unit="BRL"
            ),
            FieldMeaning(
                dataset="order_items.csv", column="price", meaning="Price.", unit="USD"
            ),
        ]
    )
    conflict_result = applicable_metrics(
        role_sets=role_sets,
        join_whitelist=_whitelist(),
        profiles=profiles,
        semantic_seeds=conflicting,
    )
    conflict_skips = _skipped_by_id(conflict_result)
    assert conflict_skips["gmv"].reason == (
        "conflicting_unit_seed:order_items.csv.price"
    )
    assert conflict_skips["aov"].reason == (
        "conflicting_unit_seed:order_items.csv.price"
    )

    mixed = SemanticSeeds(
        field_meanings=[
            FieldMeaning(
                dataset="order_items.csv", column="price", meaning="Price.", unit="BRL"
            ),
            FieldMeaning(
                dataset="order_items.csv",
                column="freight_value",
                meaning="Freight charge.",
                unit="USD",
            ),
        ]
    )
    mixed_result = applicable_metrics(
        role_sets=role_sets,
        join_whitelist=_whitelist(),
        profiles=profiles,
        semantic_seeds=mixed,
    )
    mixed_skips = _skipped_by_id(mixed_result)
    assert mixed_skips["freight_ratio"].reason == "input_unit_mismatch:BRL!=USD"


def test_repeat_purchase_requires_confirmed_join(tmp_path: Path) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)

    # Proposed-but-unconfirmed join: skipped, with the structured reason.
    unconfirmed = applicable_metrics(
        role_sets=role_sets, join_whitelist=_whitelist(status="proposed"), profiles=profiles
    )
    skip = _skipped_by_id(unconfirmed)["repeat_purchase_rate"]
    assert skip.reason == "join_not_confirmed"

    # No whitelist at all: also skipped (never resolved).
    without = applicable_metrics(role_sets=role_sets, join_whitelist=None, profiles=profiles)
    assert "repeat_purchase_rate" in _skipped_by_id(without)

    # Confirmed join: resolves cross-table over the whitelist label.
    confirmed = applicable_metrics(
        role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles
    )
    repeat = _resolved_by_id(confirmed)["repeat_purchase_rate"]
    assert repeat.required_relations == [JOIN_LABEL]
    assert "join" in repeat.sql.lower()
    assert "customer_unique_id" in repeat.sql
    assert set(repeat.target_datasets) == {"orders.csv", "customers.csv"}


def test_late_rate_skipped_without_timestamps(tmp_path: Path) -> None:
    _, _, profile, role_set = _load_with_roles(_order_items_csv(tmp_path))

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )
    skipped = _skipped_by_id(resolution)

    assert skipped["late_delivery_rate"].reason == "missing_role:timestamp"
    assert skipped["fulfillment_time"].reason == "missing_role:timestamp"
    assert skipped["time_coverage"].reason == "missing_role:timestamp"
    # But the same shape still supports the value metrics.
    resolved = _resolved_by_id(resolution)
    assert {"gmv", "aov", "freight_ratio"} <= set(resolved)


def test_every_generated_sql_carries_a_count_column(tmp_path: Path) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)

    resolution = applicable_metrics(
        role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles
    )

    assert resolution.resolved, "expected at least one resolved metric"
    for metric in resolution.resolved:
        # Time-boundary convention: aggregates always expose count(*).
        assert "count(*) as row_count" in metric.sql, metric.metric_id


def test_resolution_is_deterministic(tmp_path: Path) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)

    first = applicable_metrics(role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles)
    second = applicable_metrics(role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles)

    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------- #
# 2. Generic pack: HHI / missing hotspots / time-span coverage
# --------------------------------------------------------------------------- #
def test_generic_pack_resolves_hhi_hotspots_and_time_coverage(tmp_path: Path) -> None:
    _, _, profiles, role_sets = _olist_corpus(tmp_path)

    resolution = applicable_metrics(
        role_sets=role_sets, join_whitelist=_whitelist(), profiles=profiles
    )
    resolved = _resolved_by_id(resolution)

    hhi = resolved["concentration_hhi"]
    assert "hhi" in hhi.sql
    # The HHI bucket is a business dimension, never an id-suffixed key.
    dimension = hhi.referenced_columns[hhi.target_datasets[0]][0]
    assert not dimension.endswith("_id")
    assert hhi.metric_id == "concentration_hhi"

    hotspots = resolved["missing_hotspots"]
    assert hotspots.target_datasets == ["orders.csv"]
    assert "order_delivered_customer_date" in hotspots.referenced_columns["orders.csv"]
    assert "missing_" in hotspots.sql

    coverage = resolved["time_coverage"]
    assert coverage.target_datasets == ["orders.csv"]
    assert "min(" in coverage.sql and "max(" in coverage.sql
    assert "covered_months" in coverage.sql


def test_hhi_prefers_additive_value_over_installment_setting(tmp_path: Path) -> None:
    path = tmp_path / "payments.csv"
    rows = ["payment_type,payment_installments,payment_value"]
    payment_types = ("credit_card", "debit_card", "voucher", "boleto")
    for index in range(40):
        rows.append(f"{payment_types[index % 4]},{1 + index % 6},{25 + index * 2.75}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _, _, profile, role_set = _load_with_roles(path)

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )
    hhi = next(metric for metric in resolution.resolved if metric.metric_id == "concentration_hhi")

    assert hhi.referenced_columns[profile.name][-1] == "payment_value"
    assert "payment_installments" not in hhi.referenced_columns[profile.name]


def test_registered_metric_contracts_reject_invalid_result_domains() -> None:
    hhi = validate_metric_result(
        "concentration_hhi", {"row_count": 2, "hhi": 1.2, "top_share": 0.8}
    )
    assert hhi.valid is False
    assert hhi.code == "hhi_out_of_range"

    percentage = validate_metric_result(
        "freight_ratio", {"row_count": 10, "freight_share_percent": 101}
    )
    assert percentage.valid is False
    assert percentage.code == "percentage_out_of_range"

    coverage = validate_metric_result(
        "time_coverage",
        {
            "row_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "span_days": None,
            "covered_months": 0,
        },
    )
    assert coverage.valid is False
    assert coverage.code == "empty_metric_input"


def test_missing_hotspots_skip_on_a_complete_dataset(tmp_path: Path) -> None:
    _, _, profile, role_set = _load_with_roles(_order_items_csv(tmp_path))

    resolution = applicable_metrics(
        role_sets={profile.name: role_set}, join_whitelist=None, profiles=[profile]
    )

    assert _skipped_by_id(resolution)["missing_hotspots"].reason == ("no_high_missing_column")


# --------------------------------------------------------------------------- #
# 3. Question discovery: domain_metric family merged, capped, never suppressed
# --------------------------------------------------------------------------- #
def _llm_question(question_en: str) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=f"q_llm_{abs(hash(question_en)) % 10_000}",
        question_en=question_en,
        origin="llm",
        target_datasets=["orders.csv"],
        exploratory=True,
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.6,
        ),
    )


def test_domain_metric_candidates_merge_and_are_capped(tmp_path: Path) -> None:
    datasets, artifacts, _, role_sets = _olist_corpus(tmp_path)

    candidates = discover_question_candidates(
        datasets,
        profile_artifacts=artifacts,
        column_role_sets=role_sets,
        join_whitelist=_whitelist(),
    )

    domain = [
        candidate for candidate in candidates.candidates if candidate.template_id == "domain_metric"
    ]
    assert domain, "expected registered domain-metric candidates"
    assert len(domain) <= 6
    for candidate in domain:
        assert candidate.origin == "template"
        assert candidate.exploratory is False
        assert candidate.candidate_methods == ["domain_metric_pack"]
        assert candidate.metric_id is not None
        assert candidate.answer_contract is not None
        assert candidate.answer_contract.kind == "metric"
        assert candidate.answer_contract.metric_id == candidate.metric_id
        definition = next(
            item for item in DOMAIN_METRIC_REGISTRY if item.metric_id == candidate.metric_id
        )
        assert candidate.answer_contract.expected_units == definition.units
        assert candidate.produced_units == definition.units
        assert candidate.sql_template is not None
        assert "count(*) as row_count" in candidate.sql_template
        assert candidate.feasibility is not None
        assert candidate.proposed_action == "run_analysis"
    # The cross-table repeat rate declares its whitelist join.
    repeat = [c for c in domain if "two or more" in c.question_en]
    assert repeat and repeat[0].required_relations == [JOIN_LABEL]


def test_domain_metrics_are_not_backstop_suppressed(tmp_path: Path) -> None:
    datasets, artifacts, _, role_sets = _olist_corpus(tmp_path)
    llm_candidates = [
        _llm_question("How is revenue trending over time?"),
        _llm_question("Which segment groups have the highest revenue?"),
        _llm_question("How much data is missing from price?"),
    ]

    candidates = discover_question_candidates(
        datasets,
        profile_artifacts=artifacts,
        llm_candidates=llm_candidates,
        include_template_candidates=True,
        template_backstop_only=True,
        column_role_sets=role_sets,
        join_whitelist=_whitelist(),
    )

    # LLM coverage is complete -> zero backstop questions ...
    assert candidates.template_backstop_used == 0
    # ... yet the registered metrics still ride along: guaranteed supply,
    # not a fallback (H9-C semantic distinction from the DI8-E backstop).
    domain = [
        candidate for candidate in candidates.candidates if candidate.template_id == "domain_metric"
    ]
    assert domain
    assert len(domain) <= 6


def test_without_role_sets_no_domain_metric_is_generated(tmp_path: Path) -> None:
    datasets, artifacts, _, _ = _olist_corpus(tmp_path)

    candidates = discover_question_candidates(datasets, profile_artifacts=artifacts)

    assert all(candidate.template_id != "domain_metric" for candidate in candidates.candidates)


# --------------------------------------------------------------------------- #
# 4. Method registry: domain_metric_pack visible, descriptive_sql still first
# --------------------------------------------------------------------------- #
def test_domain_metric_pack_is_registered_without_stealing_descriptive(
    tmp_path: Path,
) -> None:
    assert "domain_metric_pack" in METHOD_REGISTRY
    method = METHOD_REGISTRY["domain_metric_pack"]
    assert method.mode == "descriptive"
    assert method.supported is True

    _, _, profile, _ = _load_with_roles(_order_items_csv(tmp_path))
    gate = method.gate(
        MethodGateContext(
            profiles=[profile],
            target_datasets=[profile.name],
            analysis_mode="descriptive",
            target_column=None,
        )
    )
    assert gate.ok is True

    empty = method.gate(
        MethodGateContext(
            profiles=[], target_datasets=[], analysis_mode="descriptive", target_column=None
        )
    )
    assert empty.ok is False and empty.missing_kinds == ["data"]

    # evaluate_feasibility must keep resolving descriptive to descriptive_sql
    # (registration order contract).
    descriptive = [method for method in METHOD_REGISTRY.values() if method.mode == "descriptive"]
    assert descriptive[0].method_id == "descriptive_sql"


# --------------------------------------------------------------------------- #
# 5. Wiring: auto EDA meters skipped metrics on the trace
# --------------------------------------------------------------------------- #
def test_auto_eda_meters_skipped_domain_metrics(tmp_path: Path) -> None:
    csv_path = _order_items_csv(tmp_path)
    run_auto_eda(
        [csv_path],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="session_metrics",
    )

    events = ArtifactStore(tmp_path / "workspace").list_trace_events(
        project_id="project_demo", session_id="session_metrics"
    )
    skipped_events = [event for event in events if event.event_type == "domain_metrics_skipped"]
    assert len(skipped_events) == 1
    summary = skipped_events[0].summary
    assert summary["skipped_count"] >= 1
    assert summary["resolved_count"] >= 1
    # Structured per-metric reasons, not prose.
    assert summary["skipped"]["late_delivery_rate"] == "missing_role:timestamp"
    assert "repeat_purchase_rate" in summary["skipped"]


def test_auto_eda_traces_conflicting_seeded_unit_skip(tmp_path: Path) -> None:
    csv_path = _order_items_csv(tmp_path)
    workspace = tmp_path / "workspace"
    project_id = "project_seed_conflict"
    store = ArtifactStore(workspace)
    save_seeds(
        store.project_dir(project_id),
        SemanticSeeds(
            field_meanings=[
                FieldMeaning(
                    dataset="order_items.csv",
                    column="price",
                    meaning="Price.",
                    unit="BRL",
                ),
                FieldMeaning(
                    dataset="order_items.csv",
                    column="price",
                    meaning="Price.",
                    unit="USD",
                ),
            ]
        ),
    )

    run_auto_eda(
        [csv_path],
        workspace=workspace,
        project_id=project_id,
        session_id="run_seed_conflict",
    )

    events = store.list_trace_events(
        project_id=project_id, session_id="run_seed_conflict"
    )
    domain_event = next(
        event for event in events if event.event_type == "domain_metrics_skipped"
    )
    assert domain_event.summary["skipped"]["gmv"] == (
        "conflicting_unit_seed:order_items.csv.price"
    )


# --------------------------------------------------------------------------- #
# 6. Registry hygiene
# --------------------------------------------------------------------------- #
def test_registry_covers_both_domains_with_unique_ids() -> None:
    ids = [definition.metric_id for definition in DOMAIN_METRIC_REGISTRY]
    assert len(ids) == len(set(ids))
    domains = {definition.domain for definition in DOMAIN_METRIC_REGISTRY}
    assert domains == {"ecommerce", "generic"}
    ecommerce = [d for d in DOMAIN_METRIC_REGISTRY if d.domain == "ecommerce"]
    generic = [d for d in DOMAIN_METRIC_REGISTRY if d.domain == "generic"]
    assert len(ecommerce) == 6
    assert len(generic) == 3
    # Cross-table metrics must declare their join requirement.
    repeat = next(d for d in DOMAIN_METRIC_REGISTRY if d.metric_id == "repeat_purchase_rate")
    assert repeat.requirement.cross_table is True
    assert "CONFIRMED" in repeat.requirement.join_requirement

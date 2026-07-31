"""DI10-W5 semantic hygiene: join proposal quality gate, high-confidence
auto-confirmation (revocable + disclosed), dataset-scoped whitelist access,
the numeric-offset timestamp veto, and the <2-period timeline defense.

Everything here is deterministic — no LLM is involved anywhere in W5.
"""

from __future__ import annotations

from pathlib import Path

from eda_platform.core.column_roles import (
    ColumnFacts,
    ColumnRole,
    ColumnRoleName,
    ColumnRoleSet,
    TableFacts,
    check_timestamp,
    infer_column_roles,
    timestamp_numeric_offset_rationale,
)
from eda_platform.core.semantic import (
    JoinWhitelist,
    JoinWhitelistEntry,
    confirm_join,
    load_join_whitelist,
    revoke_auto_confirmation,
    save_join_whitelist,
)
from eda_platform.schemas.artifacts import ColumnProfile, DatasetProfile
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.relations import (
    Cardinality,
    Confidence,
    RelationshipCandidate,
    RelationshipCandidateSet,
    RelationshipColumnPair,
    RelationshipSignals,
    RelationshipValidation,
    RelationshipValidationSet,
)
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.relationship_discovery import propose_join_candidates

# --- fixtures ---------------------------------------------------------------


def _candidate(
    left_dataset: str,
    left_column: str,
    right_dataset: str,
    right_column: str,
    *,
    confidence: Confidence = "high",
    right_unique_rate: float = 1.0,
) -> RelationshipCandidate:
    pair = RelationshipColumnPair(
        left_dataset_id=f"ds_{left_dataset}",
        left_dataset_name=left_dataset,
        left_columns=[left_column],
        right_dataset_id=f"ds_{right_dataset}",
        right_dataset_name=right_dataset,
        right_columns=[right_column],
    )
    signals = RelationshipSignals(
        name_similarity=0.9,
        type_compatible=True,
        overlap_left_in_right=1.0,
        overlap_right_in_left=0.9,
        right_unique_rate=right_unique_rate,
        left_null_rate=0.0,
        right_null_rate=0.0,
    )
    return RelationshipCandidate(
        pair=pair,
        signals=signals,
        ensemble_score=0.9,
        confidence=confidence,
    )


def _candidate_set(*candidates: RelationshipCandidate) -> RelationshipCandidateSet:
    return RelationshipCandidateSet(
        dataset_ids=sorted(
            {candidate.pair.left_dataset_id for candidate in candidates}
            | {candidate.pair.right_dataset_id for candidate in candidates}
        ),
        candidates=list(candidates),
    )


def _validation(
    candidate: RelationshipCandidate, cardinality: Cardinality
) -> RelationshipValidation:
    return RelationshipValidation(
        pair=candidate.pair,
        join_row_multiplier=1.0,
        orphan_rate_left=0.0,
        orphan_rate_right=0.0,
        cardinality=cardinality,
        verified=True,
        verification_sql="select 1",
    )


def _role(
    column: str,
    role: ColumnRoleName,
    *,
    provenance: str = "inferred",
) -> ColumnRole:
    return ColumnRole(
        column=column,
        role=role,
        confidence=0.9,
        provenance=provenance,  # type: ignore[arg-type]
        verified_by=["test"],
    )


def _facts(
    name: str,
    *,
    dtype: str = "int64",
    semantic_type: str = "datetime",
    samples: list[str],
) -> ColumnFacts:
    return ColumnFacts(
        name=name,
        dtype=dtype,
        semantic_type=semantic_type,
        row_count=1000,
        unique_count=900,
        unique_percent=90.0,
        sample_values=samples,
    )


_EMPTY_TABLE = TableFacts(dataset="table.csv", row_count=0, columns=[])


# --- 1. join proposal quality gate: role veto --------------------------------


def test_timestamp_join_key_proposal_is_rejected() -> None:
    """The `order_approved_at -> order_id` class of absurd proposals dies at
    the proposal exit when the role layer verified a timestamp on one side."""
    absurd = _candidate("orders.csv", "order_approved_at", "items.csv", "order_id")
    sane = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    role_sets = {
        "orders.csv": ColumnRoleSet(
            dataset="orders.csv",
            roles=[_role("order_approved_at", ColumnRoleName.TIMESTAMP)],
        )
    }

    proposals = propose_join_candidates(
        _candidate_set(absurd, sane), role_sets=role_sets
    )

    labels = [entry.label() for entry in proposals]
    assert absurd.pair.label() not in labels
    assert sane.pair.label() in labels


def test_measure_join_key_proposal_is_rejected_on_either_side() -> None:
    candidate = _candidate("orders.csv", "order_id", "items.csv", "price")
    role_sets = {
        "items.csv": ColumnRoleSet(
            dataset="items.csv", roles=[_role("price", ColumnRoleName.MEASURE)]
        )
    }

    proposals = propose_join_candidates(_candidate_set(candidate), role_sets=role_sets)

    assert proposals == []


def test_unverified_role_never_vetoes_a_proposal() -> None:
    """DI8-B red line: unverified hypotheses gate nothing, not even here."""
    candidate = _candidate("orders.csv", "order_approved_at", "items.csv", "order_id")
    role_sets = {
        "orders.csv": ColumnRoleSet(
            dataset="orders.csv",
            roles=[
                _role(
                    "order_approved_at",
                    ColumnRoleName.TIMESTAMP,
                    provenance="unverified",
                )
            ],
        )
    }

    proposals = propose_join_candidates(_candidate_set(candidate), role_sets=role_sets)

    assert [entry.label() for entry in proposals] == [candidate.pair.label()]


def test_gate_degrades_to_noop_without_role_sets() -> None:
    candidate = _candidate("orders.csv", "order_approved_at", "items.csv", "order_id")

    proposals = propose_join_candidates(_candidate_set(candidate))

    assert [entry.label() for entry in proposals] == [candidate.pair.label()]


# --- 2. m:n without id naming is downweighted --------------------------------


def test_many_to_many_without_id_pattern_gets_low_quality_and_sinks() -> None:
    murky = _candidate(
        "orders.csv", "status", "items.csv", "category", confidence="medium",
        right_unique_rate=0.2,
    )
    sane = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")

    proposals = propose_join_candidates(_candidate_set(murky, sane))

    by_label = {entry.label(): entry for entry in proposals}
    assert by_label[murky.pair.label()].quality == "low"
    assert by_label[sane.pair.label()].quality == "normal"
    # Low quality sorts last regardless of label order.
    assert proposals[-1].label() == murky.pair.label()
    # Low-quality proposals never auto-confirm.
    assert by_label[murky.pair.label()].status == "proposed"


def test_many_to_many_with_id_pattern_on_one_side_stays_normal_quality() -> None:
    candidate = _candidate(
        "orders.csv", "order_id", "items.csv", "category", confidence="medium",
        right_unique_rate=0.2,
    )

    proposals = propose_join_candidates(_candidate_set(candidate))

    assert proposals[0].quality == "normal"
    assert proposals[0].status == "proposed"  # medium confidence: no auto-confirm


# --- 3. high-confidence auto-confirmation ------------------------------------


def test_high_confidence_id_named_non_mn_join_auto_confirms() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_one")]
    )

    proposals = propose_join_candidates(_candidate_set(candidate), validations)

    entry = proposals[0]
    assert entry.status == "auto_confirmed"
    assert entry.confirmed_by == "auto"
    assert entry.confirmed_at is not None

    whitelist = JoinWhitelist()
    assert whitelist.merge_proposals(proposals) == 1
    assert entry.label() in whitelist.usable_labels()
    # confirmed_labels is the converged alias: auto_confirmed counts.
    assert entry.label() in whitelist.confirmed_labels()
    assert whitelist.auto_confirmed_labels() == {entry.label()}


def test_auto_confirm_requires_id_naming_on_both_sides() -> None:
    one_sided = _candidate("orders.csv", "customer_id", "customers.csv", "segment")

    proposals = propose_join_candidates(_candidate_set(one_sided))

    assert proposals[0].status == "proposed"


def test_auto_confirm_requires_full_validation() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")

    proposals = propose_join_candidates(_candidate_set(candidate))

    assert proposals[0].status == "proposed"
    assert proposals[0].confirmed_by == ""


def test_unvalidated_proposal_cannot_be_human_confirmed(tmp_path: Path) -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    whitelist = JoinWhitelist()
    whitelist.merge_proposals(propose_join_candidates(_candidate_set(candidate)))
    save_join_whitelist(tmp_path, whitelist)

    try:
        confirm_join(tmp_path, candidate.pair.label())
        raise AssertionError("expected validation gate")
    except ValueError as exc:
        assert "requires full validation" in str(exc)


def test_join_authorization_is_bound_to_current_dataset_ids() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_one")]
    )
    whitelist = JoinWhitelist()
    whitelist.merge_proposals(
        propose_join_candidates(_candidate_set(candidate), validations)
    )
    entry = whitelist.entries[0]
    current = {
        entry.left_dataset: candidate.pair.left_dataset_id,
        entry.right_dataset: candidate.pair.right_dataset_id,
    }

    assert entry.validation_freshness(current) == "fresh"
    assert whitelist.confirmed_labels(current) == {entry.label()}

    changed = {**current, entry.right_dataset: "changed_dataset_id"}
    assert entry.validation_freshness(changed) == "stale"
    assert whitelist.confirmed_labels(changed) == set()
    assert whitelist.confirmed_labels({}) == set()


def test_legacy_confirmed_join_is_visible_but_runtime_unverifiable() -> None:
    entry = JoinWhitelistEntry(
        left_dataset="orders.csv",
        left_columns=["customer_id"],
        right_dataset="customers.csv",
        right_columns=["customer_id"],
        cardinality="many_to_one",
        status="confirmed",
    )
    whitelist = JoinWhitelist(entries=[entry])

    assert whitelist.confirmed_labels() == {entry.label()}
    assert entry.validation_freshness(
        {"orders.csv": "new_orders", "customers.csv": "new_customers"}
    ) == "unverifiable"
    assert whitelist.confirmed_labels(
        {"orders.csv": "new_orders", "customers.csv": "new_customers"}
    ) == set()


def test_revalidation_of_changed_data_invalidates_human_confirmation() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_one")]
    )
    proposal = propose_join_candidates(_candidate_set(candidate), validations)[0]
    proposal.status = "confirmed"
    proposal.confirmed_by = "user"
    proposal.left_dataset_id = "old_left"
    proposal.right_dataset_id = "old_right"
    whitelist = JoinWhitelist(entries=[proposal])

    assert whitelist.merge_proposals(
        propose_join_candidates(_candidate_set(candidate), validations)
    ) == 0

    refreshed = whitelist.entries[0]
    assert refreshed.status != "confirmed"
    assert refreshed.confirmed_by != "user"
    if refreshed.status == "auto_confirmed":
        assert refreshed.confirmed_by == "auto"
    else:
        assert refreshed.confirmed_at is None
    assert refreshed.usage_count == 0
    assert refreshed.left_dataset_id == candidate.pair.left_dataset_id
    assert refreshed.right_dataset_id == candidate.pair.right_dataset_id


def test_auto_confirm_refuses_many_to_many_cardinality() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_many")]
    )

    proposals = propose_join_candidates(_candidate_set(candidate), validations)

    assert proposals[0].status == "proposed"


def test_disclosure_notes_cover_auto_confirmed_joins() -> None:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_one")]
    )
    whitelist = JoinWhitelist()
    whitelist.merge_proposals(
        propose_join_candidates(_candidate_set(candidate), validations)
    )

    label = candidate.pair.label()
    notes = whitelist.disclosure_notes([label])
    assert notes == [
        f"Join {label} was auto-confirmed (high confidence); "
        "review it on the Knowledge page."
    ]
    # Labels not auto-confirmed produce no note.
    assert whitelist.disclosure_notes(["unknown -> label"]) == []


# --- 4. revocation and promotion ---------------------------------------------


def _seed_auto_confirmed(project_dir: Path) -> str:
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    validations = RelationshipValidationSet(
        validations=[_validation(candidate, "many_to_one")]
    )
    whitelist = JoinWhitelist()
    whitelist.merge_proposals(
        propose_join_candidates(_candidate_set(candidate), validations)
    )
    save_join_whitelist(project_dir, whitelist)
    return candidate.pair.label()


def test_auto_confirmed_join_is_revocable_and_stays_revoked(tmp_path: Path) -> None:
    label = _seed_auto_confirmed(tmp_path)

    whitelist = revoke_auto_confirmation(tmp_path, label)

    entry = whitelist.entry(label)
    assert entry is not None
    assert entry.status == "proposed"
    assert entry.confirmed_by == ""
    assert entry.confirmed_at is None
    assert whitelist.usable_labels() == set()

    # A later run re-proposing the same label must NOT re-auto-confirm it:
    # merge_proposals never touches known labels, so the revocation sticks.
    reloaded = load_join_whitelist(tmp_path)
    candidate = _candidate("orders.csv", "customer_id", "customers.csv", "customer_id")
    assert reloaded.merge_proposals(propose_join_candidates(_candidate_set(candidate))) == 0
    revoked = reloaded.entry(label)
    assert revoked is not None
    assert revoked.status == "proposed"


def test_auto_confirmed_join_can_be_promoted_to_user_confirmed(tmp_path: Path) -> None:
    label = _seed_auto_confirmed(tmp_path)

    whitelist = confirm_join(tmp_path, label, confirmed_by="user")

    entry = whitelist.entry(label)
    assert entry is not None
    assert entry.status == "confirmed"
    assert entry.confirmed_by == "user"
    assert label in whitelist.usable_labels()


def test_revoke_refuses_user_confirmed_and_unknown_labels(tmp_path: Path) -> None:
    label = _seed_auto_confirmed(tmp_path)
    confirm_join(tmp_path, label, confirmed_by="user")

    try:
        revoke_auto_confirmation(tmp_path, label)
        raise AssertionError("expected ValueError for user-confirmed entry")
    except ValueError:
        pass
    try:
        revoke_auto_confirmation(tmp_path, "nope -> nope")
        raise AssertionError("expected ValueError for unknown label")
    except ValueError:
        pass


# --- 5. dataset-scoped whitelist access --------------------------------------


def _olist_entry() -> JoinWhitelistEntry:
    return JoinWhitelistEntry(
        left_dataset="olist_orders_dataset.csv",
        left_columns=["customer_id"],
        right_dataset="olist_customers_dataset.csv",
        right_columns=["customer_id"],
        cardinality="many_to_one",
        status="confirmed",
    )


def test_entries_for_returns_nothing_for_foreign_datasets() -> None:
    """The cross-project pollution fix: olist joins are invisible to a
    creditcard run even though they share the project whitelist file."""
    whitelist = JoinWhitelist(entries=[_olist_entry()])

    assert whitelist.entries_for({"creditcard.csv"}) == []


def test_entries_for_requires_both_sides_in_scope() -> None:
    whitelist = JoinWhitelist(entries=[_olist_entry()])

    assert whitelist.entries_for({"olist_orders_dataset.csv"}) == []
    scoped = whitelist.entries_for(
        {"olist_orders_dataset.csv", "olist_customers_dataset.csv", "creditcard.csv"}
    )
    assert [entry.label() for entry in scoped] == [_olist_entry().label()]


# --- 6. numeric-offset Time columns are not timestamps -----------------------


def test_creditcard_time_offset_is_not_a_timestamp() -> None:
    """creditcard `Time` = integer seconds since the first transaction: the
    profiler flags it datetime (name + parseable ints), but the role layer
    must veto — it is a numeric offset, not a point in time."""
    facts = _facts("Time", samples=["0", "406", "7891", "172792"])

    assert check_timestamp(facts, _EMPTY_TABLE) is None
    rationale = timestamp_numeric_offset_rationale(facts)
    assert rationale is not None
    assert "numeric offset" in rationale


def test_creditcard_float_time_offset_is_not_a_timestamp() -> None:
    """The live Kaggle CSV profile is float64 with integer-valued strings."""
    facts = _facts(
        "Time", dtype="float64", samples=["0.0", "406.0", "7891.0", "172792.0"]
    )

    assert check_timestamp(facts, _EMPTY_TABLE) is None
    assert timestamp_numeric_offset_rationale(facts) is not None


def test_time_offset_column_lands_in_measure_via_inference() -> None:
    column = ColumnProfile(
        name="Time",
        dtype="int64",
        semantic_type="datetime",
        missing_count=0,
        missing_percent=0.0,
        unique_count=900,
        unique_percent=90.0,
        sample_values=["0", "406", "7891", "172792"],
    )
    profile = DatasetProfile(
        dataset_id="ds_credit",
        name="creditcard.csv",
        rows=1000,
        columns=1,
        column_names=["Time"],
        dtypes={"Time": "int64"},
        missing_values={"Time": 0},
        missing_percent={"Time": 0.0},
        numeric_columns=[],
        categorical_columns=[],
        columns_detail=[column],
    )

    role_set = infer_column_roles(profile)

    role = role_set.role_of("Time")
    assert role is not None
    assert role.role is ColumnRoleName.MEASURE


def test_true_epoch_seconds_still_verify_as_timestamp() -> None:
    """10-digit values inside the 2000-2035 epoch range are genuine times."""
    facts = _facts("timestamp", samples=["1600000000", "1600000100", "1712345678"])

    checks = check_timestamp(facts, _EMPTY_TABLE)
    assert checks is not None
    assert "timestamp_epoch_seconds_range" in checks
    assert timestamp_numeric_offset_rationale(facts) is None


def test_veto_only_applies_to_bare_time_names_and_integer_dtypes() -> None:
    # A descriptive name keeps the normal profiled-datetime path.
    named = _facts("order_purchase_timestamp", samples=["0", "406"])
    assert check_timestamp(named, _EMPTY_TABLE) == ["timestamp_profiled_datetime"]
    # A string datetime column named "time" is not integer-shaped: no veto.
    stringy = _facts(
        "time", dtype="object", samples=["2026-01-01 10:00:00", "2026-01-02 11:30:00"]
    )
    assert check_timestamp(stringy, _EMPTY_TABLE) == ["timestamp_profiled_datetime"]
    # A real datetime64 dtype is always a timestamp.
    dt = _facts("time", dtype="datetime64[ns]", samples=["2026-01-01"])
    assert check_timestamp(dt, _EMPTY_TABLE) == ["timestamp_dtype"]


# --- 7. time charts need at least two periods --------------------------------


def test_single_period_timeline_chart_is_skipped(tmp_path: Path) -> None:
    # Amounts must be genuinely continuous, or the column is charted as one bar
    # per value and this test would assert on the wrong chart form.
    csv_path = tmp_path / "single_day.csv"
    rows = ["order_date,amount"]
    for index in range(40):
        rows.append(f"2026-01-01,{10 + index * 3.5:.2f}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_single")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )

    titles = {ChartSpec.model_validate(a.payload).title for a in artifacts}
    assert "Records over order_date" not in titles
    # The non-time charts still render.
    assert "Distribution of amount" in titles


def test_creditcard_shaped_time_column_produces_no_timeline(tmp_path: Path) -> None:
    """Integer second-offsets parse to a single 1970 day: no line chart."""
    csv_path = tmp_path / "creditcard.csv"
    rows = ["Time,Amount"] + [f"{i * 37},{10.5 + i}" for i in range(30)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_csv(csv_path, dataset_id="ds_credit")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )

    titles = {ChartSpec.model_validate(a.payload).title for a in artifacts}
    assert "Records over Time" not in titles


def test_two_period_timeline_chart_still_renders(tmp_path: Path) -> None:
    csv_path = tmp_path / "two_days.csv"
    csv_path.write_text(
        "order_date,amount\n"
        "2026-01-01,10\n"
        "2026-01-02,20\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_two")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")

    artifacts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )

    titles = {ChartSpec.model_validate(a.payload).title for a in artifacts}
    assert "Records over order_date" in titles

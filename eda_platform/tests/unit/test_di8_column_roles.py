"""DI8-B — deterministic column-role validators and the ColumnRoleSet cache.

Positive/negative cases per validator, with the Olist regressions the sprint
plan names explicitly: ``order_item_id``/``payment_sequential`` must verify as
sequence, zip prefixes must land in code (never measure), and review_score
sits on the measure/dimension boundary (both checks pass; measure is the
deterministic default).
"""

from __future__ import annotations

import pandas as pd
from semantic_test_helpers import load_seeds, save_seeds

from eda_platform.core.column_roles import (
    ColumnFacts,
    ColumnRole,
    ColumnRoleName,
    ColumnRoleSet,
    TableFacts,
    apply_role_seeds,
    check_code,
    check_identifier,
    check_measure,
    check_sequence,
    column_role_set_artifact,
    infer_column_roles,
    verify_role,
)
from eda_platform.core.semantic import ColumnRoleSeed, SemanticSeeds
from eda_platform.schemas.artifacts import ArtifactType, ColumnProfile, DatasetProfile

# --- fixtures ---------------------------------------------------------------


def _col(
    name: str,
    *,
    dtype: str = "str",
    semantic_type: str = "categorical",
    unique_count: int = 1,
    unique_percent: float = 0.0,
    missing_count: int = 0,
    samples: list[str] | None = None,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype=dtype,
        semantic_type=semantic_type,  # type: ignore[arg-type]
        missing_count=missing_count,
        missing_percent=0.0,
        unique_count=unique_count,
        unique_percent=unique_percent,
        sample_values=samples or [],
    )


def _profile(
    columns: list[ColumnProfile], *, rows: int, name: str = "table.csv"
) -> DatasetProfile:
    return DatasetProfile(
        dataset_id=f"ds_{name}",
        name=name,
        rows=rows,
        columns=len(columns),
        column_names=[column.name for column in columns],
        dtypes={column.name: column.dtype for column in columns},
        missing_values={column.name: column.missing_count for column in columns},
        missing_percent={column.name: column.missing_percent for column in columns},
        numeric_columns=[c.name for c in columns if c.semantic_type == "numeric"],
        categorical_columns=[c.name for c in columns if c.semantic_type == "categorical"],
        columns_detail=columns,
    )


def _facts(profile_column: ColumnProfile, *, rows: int) -> ColumnFacts:
    return ColumnFacts(
        name=profile_column.name,
        dtype=profile_column.dtype,
        semantic_type=profile_column.semantic_type,
        row_count=rows,
        missing_count=profile_column.missing_count,
        unique_count=profile_column.unique_count,
        unique_percent=profile_column.unique_percent,
        sample_values=list(profile_column.sample_values),
    )


_EMPTY_TABLE = TableFacts(dataset="table.csv", row_count=0, columns=[])


def _olist_items_frame() -> pd.DataFrame:
    """order_items-shaped frame: 10 orders x 3 line items."""
    order_ids = [f"g{index:02d}" for index in range(10) for _ in range(3)]
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "order_item_id": [1, 2, 3] * 10,
            "payment_sequential": [1, 2, 3] * 10,
            "price": [round(10.0 + i * 1.5, 2) for i in range(30)],
        }
    )


def _olist_items_profile() -> DatasetProfile:
    return _profile(
        [
            _col(
                "order_id",
                semantic_type="id",
                unique_count=10,
                unique_percent=33.33,
                samples=["g00", "g01", "g02"],
            ),
            _col(
                "order_item_id",
                dtype="int64",
                semantic_type="numeric",
                unique_count=3,
                unique_percent=10.0,
                samples=["1", "2", "3"],
            ),
            _col(
                "payment_sequential",
                dtype="int64",
                semantic_type="numeric",
                unique_count=3,
                unique_percent=10.0,
                samples=["1", "1", "2"],
            ),
            _col(
                "price",
                dtype="float64",
                semantic_type="numeric",
                unique_count=30,
                unique_percent=100.0,
                samples=["10.0", "11.5", "13.0"],
            ),
        ],
        rows=30,
        name="olist_order_items.csv",
    )


# --- identifier -------------------------------------------------------------


def test_identifier_verified_by_name_pattern_and_uniqueness() -> None:
    facts = _facts(
        _col(
            "order_id",
            semantic_type="id",
            unique_count=100,
            unique_percent=100.0,
            samples=["e481f51cbdc54678", "53cdb2fc8bc7dce0"],
        ),
        rows=100,
    )
    assert check_identifier(facts, _EMPTY_TABLE) == ["id_name_pattern", "id_unique_ratio"]


def test_identifier_generic_path_for_unnamed_near_unique_keys() -> None:
    facts = _facts(
        _col(
            "product_category_name",
            unique_count=71,
            unique_percent=100.0,
            samples=["beleza_saude", "automotivo"],
        ),
        rows=71,
    )
    assert check_identifier(facts, _EMPTY_TABLE) == ["id_unique_ratio", "id_short_values"]


def test_identifier_rejects_low_uniqueness_even_with_id_name() -> None:
    # Olist regression root cause: order_item_id is id-NAMED but is a per-order
    # line counter (uniqueness ~0.02%), so the identifier check must fail.
    facts = _facts(
        _col("order_item_id", dtype="int64", semantic_type="numeric", unique_count=3,
             unique_percent=10.0, samples=["1", "2", "3"]),
        rows=30,
    )
    assert check_identifier(facts, _EMPTY_TABLE) is None


def test_identifier_rejects_missing_values() -> None:
    facts = _facts(
        _col("review_id", semantic_type="id", unique_count=99, unique_percent=99.0,
             missing_count=1, samples=["a", "b"]),
        rows=100,
    )
    assert check_identifier(facts, _EMPTY_TABLE) is None


# --- sequence ---------------------------------------------------------------


def test_sequence_verified_strict_1n_within_identifier_group() -> None:
    profile = _olist_items_profile()
    role_set = infer_column_roles(profile, frame=_olist_items_frame())

    for column in ("order_item_id", "payment_sequential"):
        role = role_set.role_of(column)
        assert role is not None, column
        assert role.role is ColumnRoleName.SEQUENCE
        assert role.provenance == "inferred"
        assert role.verified_by == ["sequence_strict_1n_within_group:order_id"]


def test_sequence_rejects_gaps_and_constant_ones() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "a", "b", "b"],
            "gapped": [1, 3, 1, 2],  # 1,3 breaks strict 1..n
            "constant": [1, 1, 1, 1],  # degenerate: no order information
        }
    )
    columns = [
        _facts(_col("order_id", semantic_type="id", unique_count=2, unique_percent=50.0), rows=4),
        _facts(
            _col("gapped", dtype="int64", semantic_type="numeric", unique_count=3,
                 unique_percent=75.0),
            rows=4,
        ),
        _facts(
            _col("constant", dtype="int64", semantic_type="numeric", unique_count=1,
                 unique_percent=25.0),
            rows=4,
        ),
    ]
    table = TableFacts(dataset="t.csv", row_count=4, columns=columns, frame=frame)

    assert check_sequence(columns[1], table) is None
    assert check_sequence(columns[2], table) is None


def test_sequence_tolerates_a_sliver_of_dirty_groups() -> None:
    # Live-Olist regression (2026-07-18): 82 of 103 886 payment rows start at 2
    # because the first installment record is missing. A conformity ratio below
    # 100% but >= _SEQUENCE_CONFORMITY_MIN must still verify — otherwise a true
    # counter stays "unverified" and re-enters the stats candidate pools.
    groups = [f"g{i}" for i in range(500)]
    order_ids = [g for g in groups for _ in range(2)]
    seq = [1, 2] * 500
    seq[0], seq[1] = 2, 3  # one broken group out of 500 (99.8% conformity)
    frame = pd.DataFrame({"order_id": order_ids, "payment_sequential": seq})
    columns = [
        _facts(
            _col("order_id", semantic_type="id", unique_count=500, unique_percent=50.0),
            rows=1000,
        ),
        _facts(
            _col(
                "payment_sequential",
                dtype="int64",
                semantic_type="numeric",
                unique_count=3,
                unique_percent=0.3,
            ),
            rows=1000,
        ),
    ]
    table = TableFacts(dataset="t.csv", row_count=1000, columns=columns, frame=frame)

    assert check_sequence(columns[1], table) == [
        "sequence_strict_1n_within_group:order_id"
    ]

    # Heavy dirt (half the groups broken) must NOT verify.
    bad_seq = [2 if i % 2 == 0 else 1 for i in range(500)]
    bad_frame = pd.DataFrame({"order_id": groups, "payment_sequential": bad_seq})
    bad_table = TableFacts(
        dataset="t.csv", row_count=500, columns=columns, frame=bad_frame
    )
    assert check_sequence(columns[1], bad_table) is None


def test_sequence_needs_the_frame_and_never_falls_through_to_measure() -> None:
    # Without a frame the strict per-group check cannot run; the id-named
    # counter must NOT verify as anything else (especially not measure).
    profile = _olist_items_profile()
    role_set = infer_column_roles(profile)  # no frame

    assert role_set.role_of("order_item_id") is None
    facts = _facts(profile.columns_detail[1], rows=profile.rows)
    assert check_measure(facts, _EMPTY_TABLE) is None


def test_sequence_named_counter_never_verifies_as_measure_without_frame() -> None:
    # Real-Olist regression: payment_sequential has no id token, and with
    # profile stats only (no frame) its 29 distinct values would pass the
    # dispersion check. The sequence-name guard keeps it out of measure AND
    # dimension; only the strict frame check may assign it a (sequence) role.
    facts = _facts(
        _col("payment_sequential", dtype="int64", semantic_type="numeric", unique_count=29,
             unique_percent=0.03, samples=["1", "1", "2"]),
        rows=103886,
    )
    assert check_measure(facts, _EMPTY_TABLE) is None
    assert verify_role(ColumnRoleName.DIMENSION, facts, _EMPTY_TABLE) is None


# --- code vs measure (zip / numeric-shaped strings) -------------------------


def test_zip_prefix_is_code_not_measure() -> None:
    facts = _facts(
        _col("seller_zip_code_prefix", dtype="int64", semantic_type="numeric",
             unique_count=2246, unique_percent=72.57, samples=["13023", "1037", "20031"]),
        rows=3095,
    )
    assert check_code(facts, _EMPTY_TABLE) == ["code_name_pattern", "code_digit_values"]
    assert check_measure(facts, _EMPTY_TABLE) is None


def test_leading_zero_digit_strings_are_code_without_a_code_name() -> None:
    facts = _facts(
        _col("part_ref", unique_count=40, unique_percent=80.0,
             samples=["01037", "00453", "09210"]),
        rows=50,
    )
    assert check_code(facts, _EMPTY_TABLE) == ["code_leading_zero_digits"]


def test_plain_float_values_are_not_code() -> None:
    facts = _facts(
        _col("price", dtype="float64", semantic_type="numeric", unique_count=30,
             unique_percent=100.0, samples=["58.9", "239.9"]),
        rows=30,
    )
    assert check_code(facts, _EMPTY_TABLE) is None
    assert check_measure(facts, _EMPTY_TABLE) == [
        "measure_numeric_dtype",
        "measure_fractional_values",
    ]


# --- review_score: the measure/dimension boundary ---------------------------


def test_review_score_verifies_as_ordinal_measure_by_default() -> None:
    profile = _profile(
        [
            _col("review_score", dtype="int64", semantic_type="numeric", unique_count=5,
                 unique_percent=0.01, samples=["4", "5", "1"]),
        ],
        rows=99224,
        name="olist_order_reviews.csv",
    )
    role_set = infer_column_roles(profile)
    role = role_set.role_of("review_score")

    assert role is not None
    assert role.role is ColumnRoleName.MEASURE
    assert "measure_ordinal_scale" in role.verified_by


def test_review_score_dimension_reading_is_also_verifiable() -> None:
    # Boundary case: a 1-5 scale legitimately verifies as EITHER measure or
    # dimension. Mutual exclusion is resolved by whichever role is
    # hypothesised; the deterministic default (above) prefers measure.
    facts = _facts(
        _col("review_score", dtype="int64", semantic_type="numeric", unique_count=5,
             unique_percent=0.01, samples=["4", "5", "1"]),
        rows=99224,
    )
    assert verify_role(ColumnRoleName.MEASURE, facts, _EMPTY_TABLE) is not None
    assert verify_role(ColumnRoleName.DIMENSION, facts, _EMPTY_TABLE) == [
        "dimension_low_cardinality"
    ]


# --- timestamp / geo / text / dimension -------------------------------------


def test_timestamp_geo_text_dimension_positive_and_negative() -> None:
    rows = 1000
    profile = _profile(
        [
            _col("order_purchase_timestamp", semantic_type="datetime", unique_count=990,
                 unique_percent=99.0, samples=["2017-10-02 10:56:33"]),
            _col("geolocation_lat", dtype="float64", semantic_type="numeric",
                 unique_count=700, unique_percent=70.0, samples=["-23.54", "-22.11"]),
            _col("customer_state", semantic_type="categorical", unique_count=27,
                 unique_percent=2.7, samples=["SP", "RJ"]),
            _col("review_comment_message", semantic_type="text", unique_count=400,
                 unique_percent=40.0,
                 samples=["Recebi bem antes do prazo estipulado, recomendo a loja."]),
            _col("order_status", semantic_type="categorical", unique_count=8,
                 unique_percent=0.8, samples=["delivered", "shipped"]),
        ],
        rows=rows,
    )
    role_set = infer_column_roles(profile)

    def _role(name: str) -> ColumnRoleName:
        role = role_set.role_of(name)
        assert role is not None, name
        return role.role

    assert _role("order_purchase_timestamp") is ColumnRoleName.TIMESTAMP
    assert _role("geolocation_lat") is ColumnRoleName.GEO
    assert _role("customer_state") is ColumnRoleName.GEO
    assert _role("review_comment_message") is ColumnRoleName.TEXT
    assert _role("order_status") is ColumnRoleName.DIMENSION


def test_latency_is_not_geo_and_out_of_bounds_coordinates_fail() -> None:
    latency = _facts(
        _col("latency", dtype="float64", semantic_type="numeric", unique_count=900,
             unique_percent=90.0, samples=["12.5", "300.9"]),
        rows=1000,
    )
    bad_lat = _facts(
        _col("geolocation_lat", dtype="float64", semantic_type="numeric", unique_count=900,
             unique_percent=90.0, samples=["-233.5", "412.0"]),
        rows=1000,
    )
    assert verify_role(ColumnRoleName.GEO, latency, _EMPTY_TABLE) is None
    assert verify_role(ColumnRoleName.GEO, bad_lat, _EMPTY_TABLE) is None


# --- consumption API: stats exclusion + impact weights ----------------------


def test_excluded_from_stats_and_impact_weight_gate_only_verified_roles() -> None:
    profile = _olist_items_profile()
    role_set = infer_column_roles(profile, frame=_olist_items_frame())

    excluded = role_set.excluded_from_stats()
    assert "order_item_id" in excluded
    assert "payment_sequential" in excluded
    assert "price" not in excluded

    assert role_set.impact_weight("order_item_id") == 0.0
    assert role_set.impact_weight("payment_sequential") == 0.0
    assert role_set.impact_weight("price") == 1.0
    assert role_set.impact_weight("never_seen_column") == 1.0


def test_unverified_roles_never_gate_anything() -> None:
    # Red line: an unverified hypothesis affects wording only. It must not
    # remove a column from candidate pools nor zero its impact weight.
    role_set = ColumnRoleSet(
        dataset="t.csv",
        roles=[
            ColumnRole(
                column="maybe_seq",
                role=ColumnRoleName.SEQUENCE,
                confidence=0.4,
                provenance="unverified",
            )
        ],
    )
    assert role_set.excluded_from_stats() == set()
    assert role_set.impact_weight("maybe_seq") == 1.0


# --- seeds: seeded > inferred ----------------------------------------------


def test_seed_overrides_inferred_role_and_invalid_seed_is_skipped() -> None:
    profile = _olist_items_profile()
    seeds = SemanticSeeds(
        column_role_seeds=[
            ColumnRoleSeed(
                dataset="olist_order_items.csv",
                column="price",
                role="dimension",
                note="Pinned for a pricing-tier study.",
            ),
            ColumnRoleSeed(dataset="olist_order_items.csv", column="order_id", role="florble"),
            ColumnRoleSeed(dataset="other.csv", column="price", role="measure"),
        ]
    )
    role_set = infer_column_roles(profile, frame=_olist_items_frame(), seeds=seeds)

    price = role_set.role_of("price")
    assert price is not None
    assert price.role is ColumnRoleName.DIMENSION
    assert price.provenance == "seeded"
    assert price.confidence == 1.0
    assert price.verified_by == ["human_seed"]
    # Invalid role value never extends the vocabulary; other-dataset seed ignored.
    assert role_set.role_of("order_id") is None


def test_seed_adds_role_for_column_without_inference() -> None:
    role_set = ColumnRoleSet(dataset="t.csv", roles=[])
    seeds = SemanticSeeds(
        column_role_seeds=[ColumnRoleSeed(dataset="t.csv", column="mystery", role="measure")]
    )
    apply_role_seeds(role_set, seeds)

    role = role_set.role_of("mystery")
    assert role is not None
    assert role.provenance == "seeded"
    assert role.role is ColumnRoleName.MEASURE


def test_column_role_seeds_persist_via_seed_store(tmp_path) -> None:
    seeds = SemanticSeeds(
        column_role_seeds=[
            ColumnRoleSeed(dataset="orders.csv", column="gmv", role="measure", note="Core KPI.")
        ]
    )
    save_seeds(tmp_path, seeds)
    loaded = load_seeds(tmp_path)

    assert loaded.column_role_seeds[0].column == "gmv"
    assert loaded.column_role_seeds[0].role == "measure"


# --- artifact form: the rebuildable cache -----------------------------------


def test_role_set_round_trips_through_artifact_payload() -> None:
    profile = _olist_items_profile()
    role_set = infer_column_roles(profile, frame=_olist_items_frame())
    artifact = column_role_set_artifact(
        role_set, project_id="project_demo", session_id="run_demo", parents=["prof_1"]
    )

    assert artifact.type is ArtifactType.COLUMN_ROLE_SET
    assert artifact.parents == ["prof_1"]
    restored = ColumnRoleSet.model_validate(artifact.payload)
    assert restored.dataset == "olist_order_items.csv"
    assert restored.model_version == "deterministic"
    assert restored.generated_at is not None
    assert restored.excluded_from_stats() == role_set.excluded_from_stats()


def test_adoption_lifecycle_field_defaults_to_zero() -> None:
    role = ColumnRole(
        column="c", role=ColumnRoleName.MEASURE, confidence=0.4, provenance="unverified"
    )
    assert role.adoption_count == 0

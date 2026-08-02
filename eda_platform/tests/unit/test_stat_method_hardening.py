"""E1.5 method hardening for stat_tests: effect/CI coverage matrix, signed
effects (R4), contingency effect CIs, and the hardened correlation screen."""

from __future__ import annotations

import math
from typing import get_args

import pandas as pd
import pytest

from eda_platform.schemas.stats import StatTestType
from eda_platform.tools.stat_tests import (
    TEST_PUBLISHABILITY,
    effect_ci_supported,
    run_stat_test,
    screen_correlations,
)

# ---------------------------------------------------------------------------
# P0-9: publishability matrix
# ---------------------------------------------------------------------------


def test_publishability_matrix_covers_every_test_type_with_no_dark_path() -> None:
    assert set(TEST_PUBLISHABILITY) == set(get_args(StatTestType))
    assert set(TEST_PUBLISHABILITY.values()) <= {"confirmatory_ready", "descriptive_only"}
    for test_type, lane in TEST_PUBLISHABILITY.items():
        # No dark path: confirmatory lane if and only if an effect CI exists.
        assert (lane == "confirmatory_ready") == effect_ci_supported(test_type), test_type


# ---------------------------------------------------------------------------
# R4: signed effect sizes
# ---------------------------------------------------------------------------


def _two_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "value": [9.0, 10, 11, 10, 12, 11, 9, 10, 11, 12]
            + [21.0, 20, 22, 23, 19, 21, 22, 20, 23, 21],
        }
    )


def test_independent_t_effect_size_keeps_direction() -> None:
    result = run_stat_test(
        _two_group_frame(),
        dataset_id="d",
        test_type="independent_t_test",
        group_column="group",
        value_column="value",
        effect_ci=True,
    )
    # Group "a" is far below "b": Cohen's d must be negative, not |d|.
    assert result.effect_size is not None and result.effect_size < -2.0
    assert result.effect_ci_low is not None and result.effect_ci_high is not None
    assert result.effect_ci_low <= result.effect_size <= result.effect_ci_high
    assert result.effect_ci_high < 0.0


def test_mann_whitney_rank_biserial_keeps_direction() -> None:
    frame = pd.DataFrame(
        {"g": ["a"] * 5 + ["b"] * 5, "v": [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    )
    low_first = run_stat_test(
        frame, dataset_id="d", test_type="mann_whitney_u", group_column="g", value_column="v"
    )
    assert low_first.effect_size == pytest.approx(-1.0)

    flipped = pd.DataFrame(
        {"g": ["a"] * 5 + ["b"] * 5, "v": [5.0, 6, 7, 8, 9, 0, 1, 2, 3, 4]}
    )
    high_first = run_stat_test(
        flipped, dataset_id="d", test_type="mann_whitney_u", group_column="g", value_column="v"
    )
    assert high_first.effect_size == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# P0-9: paired t effect (signed dz) + CI + mean-difference CI note
# ---------------------------------------------------------------------------


def _paired_frame(direction: float) -> pd.DataFrame:
    subjects = list(range(24))
    before = [10.0 + (i % 5) for i in subjects]
    after = [b + direction * (3.0 + 0.1 * (i % 3)) for i, b in enumerate(before)]
    return pd.DataFrame(
        {
            "subject": subjects * 2,
            "phase": ["after"] * 24 + ["before"] * 24,
            "value": after + before,
        }
    )


def test_paired_t_reports_signed_dz_with_ci_and_mean_difference_note() -> None:
    result = run_stat_test(
        _paired_frame(+1.0),
        dataset_id="d",
        test_type="paired_t_test",
        group_column="phase",
        value_column="value",
        pair_column="subject",
        effect_ci=True,
    )
    # Labels sort as (after, before); after > before -> positive dz.
    assert result.effect_size is not None and result.effect_size > 0.0
    assert result.effect_ci_low is not None and result.effect_ci_high is not None
    assert result.effect_ci_low <= result.effect_size <= result.effect_ci_high
    assert result.effect_ci_low > 0.0
    assert any(w.code == "paired_mean_difference_ci" for w in result.warnings)

    reversed_result = run_stat_test(
        _paired_frame(-1.0),
        dataset_id="d",
        test_type="paired_t_test",
        group_column="phase",
        value_column="value",
        pair_column="subject",
        effect_ci=True,
    )
    assert reversed_result.effect_size == pytest.approx(-result.effect_size)


# ---------------------------------------------------------------------------
# P0-9: chi-square bias-corrected Cramér's V + CI
# ---------------------------------------------------------------------------


def test_chi_square_effect_is_bias_corrected_cramers_v() -> None:
    # 2x2 with phi2 = 0.01 < (r-1)(k-1)/(n-1) = 1/39: the corrected V is exactly 0
    # while the uncorrected V would be 0.1.
    frame = pd.DataFrame(
        {
            "g": ["x"] * 20 + ["y"] * 20,
            "c": ["yes"] * 11 + ["no"] * 9 + ["yes"] * 9 + ["no"] * 11,
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="d",
        test_type="chi_square_independence",
        group_column="g",
        category_column="c",
    )
    assert result.test_type == "chi_square_independence"
    assert result.effect_size == pytest.approx(0.0)


def test_chi_square_strong_association_has_effect_ci() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["organic"] * 30 + ["paid"] * 30,
            "converted": ["yes"] * 24 + ["no"] * 6 + ["yes"] * 8 + ["no"] * 22,
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="d",
        test_type="chi_square_independence",
        group_column="channel",
        category_column="converted",
        effect_ci=True,
    )
    assert result.effect_size is not None and result.effect_size > 0.3
    assert result.effect_ci_low is not None and result.effect_ci_high is not None
    assert result.effect_ci_low <= result.effect_size <= result.effect_ci_high


# ---------------------------------------------------------------------------
# P0-9: Fisher exact fallback reports odds ratio + Woolf CI
# ---------------------------------------------------------------------------


def test_fisher_fallback_effect_is_odds_ratio_with_ci() -> None:
    # Category labels chosen so the alphabetically sorted crosstab keeps the
    # intended orientation: rows (x, y) x columns (hit, miss) = [[7,1],[1,7]].
    frame = pd.DataFrame(
        {
            "g": ["x"] * 8 + ["y"] * 8,
            "c": ["hit"] * 7 + ["miss"] * 1 + ["hit"] * 1 + ["miss"] * 7,
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="d",
        test_type="chi_square_independence",
        group_column="g",
        category_column="c",
        effect_ci=True,
    )
    assert result.test_type == "fisher_exact"
    assert result.effect_size == pytest.approx(49.0)
    assert result.effect_size is not None
    assert result.effect_ci_low is not None and result.effect_ci_high is not None
    assert result.effect_ci_low <= result.effect_size <= result.effect_ci_high
    assert result.effect_ci_low > 1.0


def test_fisher_zero_cell_uses_haldane_anscombe_for_or_and_ci() -> None:
    # Sorted crosstab is [[8, 0], [2, 6]]: the zero cell forces the correction.
    frame = pd.DataFrame(
        {
            "g": ["x"] * 8 + ["y"] * 8,
            "c": ["hit"] * 8 + ["hit"] * 2 + ["miss"] * 6,
        }
    )
    result = run_stat_test(
        frame,
        dataset_id="d",
        test_type="chi_square_independence",
        group_column="g",
        category_column="c",
        effect_ci=True,
    )
    assert result.test_type == "fisher_exact"
    # (8.5 * 6.5) / (0.5 * 2.5) = 44.2 with the 0.5 continuity correction.
    assert result.effect_size == pytest.approx(44.2)
    assert result.effect_size is not None
    assert result.effect_ci_low is not None and result.effect_ci_high is not None
    assert math.isfinite(result.effect_ci_low) and math.isfinite(result.effect_ci_high)
    assert result.effect_ci_low <= result.effect_size <= result.effect_ci_high
    assert any("Haldane" in w.message for w in result.warnings)


# ---------------------------------------------------------------------------
# P1: hardened correlation screen (Holm default, Spearman, min pairwise n)
# ---------------------------------------------------------------------------


def _corr_frame(n: int = 40) -> pd.DataFrame:
    x = [float(i) for i in range(n)]
    return pd.DataFrame(
        {
            "x": x,
            "neg": [50.0 - v + ((i % 3) - 1) * 0.5 for i, v in enumerate(x)],
            "noise": [((i * 7919) % 97) / 97.0 for i in range(n)],
        }
    )


def test_screen_correlations_defaults_to_holm_and_keeps_sign() -> None:
    result = screen_correlations(
        _corr_frame(), dataset_id="ds", dataset_name="corr.csv"
    )
    assert result.correction_method == "holm"
    assert result.correlation_method == "pearson"
    assert result.min_pairwise_n >= 3
    assert result.pairs_tested == 3
    rows = result.table.rows
    top = rows[0]
    # x vs neg is a strong NEGATIVE correlation; the sign must survive.
    assert {top["column_a"], top["column_b"]} == {"x", "neg"}
    assert top["coefficient"] < -0.9
    assert top["p_value"] is not None and top["adjusted_p"] is not None
    assert top["adjusted_p"] >= top["p_value"]
    assert top["missing_policy"] == "pairwise_complete"
    # P2 cleanup: no duplicated derived/repeated keys in every row.
    assert "abs_pearson" not in top and "pearson" not in top
    assert "correction_method" not in top and "pairs_tested" not in top


def test_screen_correlations_supports_spearman_for_monotone_nonlinear() -> None:
    frame = pd.DataFrame(
        {
            "x": [float(i) for i in range(30)],
            "expy": [math.exp(0.3 * i) for i in range(30)],
        }
    )
    result = screen_correlations(
        frame, dataset_id="ds", dataset_name="mono.csv", method="spearman"
    )
    assert result.correlation_method == "spearman"
    assert result.table.rows[0]["coefficient"] == pytest.approx(1.0)


def test_screen_correlations_marks_insufficient_pairs_instead_of_p_values() -> None:
    n = 30
    frame = _corr_frame(n)
    sparse = [float(i) if i < 4 else None for i in range(n)]
    frame["sparse"] = sparse
    result = screen_correlations(
        frame, dataset_id="ds", dataset_name="corr.csv", min_pairwise_n=10
    )
    assert result.pairs_insufficient_n == 3
    assert result.pairs_tested == 3
    insufficient = [row for row in result.table.rows if row["insufficient_n"]]
    assert len(insufficient) == 3
    for row in insufficient:
        assert row["p_value"] is None and row["adjusted_p"] is None
        assert row["coefficient"] is None
        assert row["pairwise_complete_n"] < 10
    tested = [row for row in result.table.rows if not row["insufficient_n"]]
    assert all(row["p_value"] is not None for row in tested)


def test_screen_correlations_rejects_unknown_method_and_correction() -> None:
    frame = _corr_frame()
    with pytest.raises(ValueError, match="method"):
        screen_correlations(
            frame, dataset_id="ds", dataset_name="x", method="kendall"
        )
    with pytest.raises(ValueError, match="correction"):
        screen_correlations(
            frame, dataset_id="ds", dataset_name="x", correction_method="bonferroni"
        )

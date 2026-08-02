from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from eda_platform.core.ids import make_artifact_id
from eda_platform.core.provenance import code_ref
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_exists,
    check_column_semantic_type,
    check_enum,
    check_non_empty,
    check_range,
    raise_for_violations,
)
from eda_platform.schemas.artifacts import AnalysisTable, Artifact, ArtifactType
from eda_platform.schemas.charts import ChartSpec
from eda_platform.schemas.stats import (
    StatAssumptionCheck,
    StatTestResult,
    StatTestType,
    StatWarning,
)

_STAT_TEST_TYPES: tuple[StatTestType, ...] = (
    "independent_t_test",
    "paired_t_test",
    "chi_square_independence",
    "one_way_anova",
    "welch_anova",
    "mann_whitney_u",
    "kruskal_wallis",
)
_VALUE_TESTS: frozenset[str] = frozenset(
    {
        "independent_t_test",
        "paired_t_test",
        "one_way_anova",
        "welch_anova",
        "mann_whitney_u",
        "kruskal_wallis",
    }
)
_MAX_BOXPLOT_GROUPS = 20
_MAX_BOXPLOT_VALUES_PER_GROUP = 50

# Cochran's rule of thumb: chi-square is unreliable once more than 20% of
# expected cell frequencies fall below 5.
_CHI2_LOW_EXPECTED_SHARE = 0.20

_EFFECT_CI_RESAMPLES = 2_000
_EFFECT_CI_MAX_TOTAL_N = 8_000
_EFFECT_CI_SEED = 0

# Contingency tests get their effect CI from dedicated paths rather than the
# (group, value) sample bootstrap in _EFFECT_CI_STATISTICS.
_EFFECT_CI_SPECIAL: frozenset[str] = frozenset(
    {"paired_t_test", "chi_square_independence", "fisher_exact"}
)

# Per-test-type publishability for the claim gate: a test type may enter the
# confirmatory lane only if it produces an effect size WITH a CI; anything
# else must be explicitly descriptive_only (no silent dead-end lane).
TEST_PUBLISHABILITY: dict[str, str] = {
    "independent_t_test": "confirmatory_ready",
    "paired_t_test": "confirmatory_ready",
    "chi_square_independence": "confirmatory_ready",
    "fisher_exact": "confirmatory_ready",
    "one_way_anova": "confirmatory_ready",
    "welch_anova": "confirmatory_ready",
    "mann_whitney_u": "confirmatory_ready",
    "kruskal_wallis": "confirmatory_ready",
}


def effect_ci_supported(test_type: str) -> bool:
    """True when run_stat_test can attach an effect-size CI for this test type."""
    return test_type in _EFFECT_CI_STATISTICS or test_type in _EFFECT_CI_SPECIAL


def run_stat_test(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    test_type: StatTestType,
    group_column: str | None = None,
    value_column: str | None = None,
    category_column: str | None = None,
    pair_column: str | None = None,
    comparison_count: int = 1,
    effect_ci: bool = False,
) -> StatTestResult:
    guard_stat_test_params(
        frame,
        test_type=test_type,
        group_column=group_column,
        value_column=value_column,
        category_column=category_column,
        pair_column=pair_column,
        comparison_count=comparison_count,
    )
    with warnings.catch_warnings():
        # Degenerate inputs make scipy emit "unreliable result" RuntimeWarnings; such non-
        # finite results are dropped by the skip gate below, so the warning is pure noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _dispatch_stat_test(
            frame,
            dataset_id=dataset_id,
            test_type=test_type,
            group_column=group_column,
            value_column=value_column,
            category_column=category_column,
            pair_column=pair_column,
        )

    # Skip gate: a non-computable test (constant column / empty group -> non-finite or
    # None statistic) is not valid evidence.
    if result.statistic is None or result.p_value is None:
        raise ValueError(
            f"{test_type} is not computable on this data (non-finite result); "
            "likely a constant column or an empty group."
        )
    if effect_ci and result.effect_size is not None:
        ci_low, ci_high, ci_warnings = _effect_size_ci(
            frame,
            test_type=result.test_type,
            group_column=group_column,
            value_column=value_column,
            category_column=category_column,
            pair_column=pair_column,
        )
        result.effect_ci_low = ci_low
        result.effect_ci_high = ci_high
        result.warnings.extend(ci_warnings)
    if comparison_count > 1:
        result.adjusted_p_value = min(1.0, float(result.p_value) * comparison_count)
        result.correction_method = "bonferroni"
        result.warnings.append(
            StatWarning(
                code="multiple_comparisons",
                message=(
                    f"{comparison_count} comparisons were requested in this run; "
                    f"Bonferroni-adjusted p={result.adjusted_p_value:.6g}."
                ),
            )
        )
    return result


def guard_stat_test_params(
    frame: pd.DataFrame,
    *,
    test_type: Any,
    group_column: Any,
    value_column: Any,
    category_column: Any,
    pair_column: Any = None,
    comparison_count: Any,
) -> None:
    violations: list[GuardViolation | None] = [
        check_enum("test_type", test_type, _STAT_TEST_TYPES),
        check_range(
            "comparison_count",
            comparison_count,
            minimum=1.0,
            fix_hint="Set `comparison_count` to an integer greater than or equal to 1.",
        ),
    ]
    if test_type in _VALUE_TESTS:
        violations.extend(
            [
                check_non_empty("group_column", group_column),
                check_non_empty("value_column", value_column),
            ]
        )
        if _is_non_empty_string(group_column):
            violations.append(
                check_column_semantic_type(
                    "group_column",
                    group_column,
                    frame,
                    allowed_semantic_types=("categorical",),
                )
            )
        if _is_non_empty_string(value_column):
            violations.append(
                check_column_semantic_type(
                    "value_column",
                    value_column,
                    frame,
                    allowed_semantic_types=("numeric",),
                )
            )
        if test_type == "paired_t_test":
            violations.append(
                check_non_empty(
                    "pair_column",
                    pair_column,
                    fix_hint=(
                        "Provide the entity or subject key that identifies matched pairs."
                    ),
                )
            )
            if _is_non_empty_string(pair_column):
                violations.append(
                    check_column_exists("pair_column", pair_column, frame.columns)
                )
    elif test_type == "chi_square_independence":
        violations.extend(
            [
                check_non_empty("group_column", group_column),
                check_non_empty("category_column", category_column),
            ]
        )
        if _is_non_empty_string(group_column):
            violations.append(
                check_column_semantic_type(
                    "group_column",
                    group_column,
                    frame,
                    allowed_semantic_types=("categorical",),
                )
            )
        if _is_non_empty_string(category_column):
            violations.append(
                check_column_semantic_type(
                    "category_column",
                    category_column,
                    frame,
                    allowed_semantic_types=("categorical",),
                )
            )
    raise_for_violations("run_stat_test", violations)


def _dispatch_stat_test(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    test_type: StatTestType,
    group_column: str | None,
    value_column: str | None,
    category_column: str | None,
    pair_column: str | None,
) -> StatTestResult:
    if test_type == "independent_t_test":
        return _independent_t_test(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
        )
    if test_type == "paired_t_test":
        return _paired_t_test(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
            pair_column=_required(pair_column, "pair_column"),
        )
    if test_type == "chi_square_independence":
        return _chi_square_independence(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            category_column=_required(category_column, "category_column"),
        )
    if test_type == "one_way_anova":
        return _one_way_anova(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
        )
    if test_type == "welch_anova":
        return _welch_anova(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
        )
    if test_type == "mann_whitney_u":
        return _mann_whitney_u(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
        )
    if test_type == "kruskal_wallis":
        return _kruskal_wallis(
            frame,
            dataset_id=dataset_id,
            group_column=_required(group_column, "group_column"),
            value_column=_required(value_column, "value_column"),
        )
    raise ValueError(f"Unsupported stat test type: {test_type}")


def create_stat_test_artifact(
    result: StatTestResult,
    *,
    project_id: str,
    session_id: str,
    parents: list[str] | None = None,
) -> Artifact:
    payload = result.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("stat", payload),
        type=ArtifactType.STAT_TEST_RESULT,
        project_id=project_id,
        session_id=session_id,
        parents=parents or [],
        payload=payload,
        code_ref=code_ref(run_stat_test),
        plain_language=_stat_plain_language(result),
    )


def create_anova_boxplot_artifact(
    frame: pd.DataFrame,
    result: StatTestResult,
    *,
    project_id: str,
    session_id: str,
    parents: list[str] | None = None,
) -> Artifact | None:
    """Create a bounded Vega-Lite boxplot companion for a one-way ANOVA."""
    if (
        result.test_type not in {"one_way_anova", "welch_anova"}
        or result.group_column is None
        or result.value_column is None
    ):
        return None
    grouped = _numeric_groups(
        frame,
        group_column=result.group_column,
        value_column=result.value_column,
    )
    selected = list(grouped.items())[:_MAX_BOXPLOT_GROUPS]
    values = [
        {result.group_column: label, result.value_column: float(value)}
        for label, group_values in selected
        for value in _display_sample(group_values, _MAX_BOXPLOT_VALUES_PER_GROUP)
    ]
    if not values:
        return None
    sample_sizes = ", ".join(f"{label}={len(group)}" for label, group in selected)
    group_note = f"{len(selected)} of {len(grouped)} groups"
    spec = ChartSpec(
        dataset_id=result.dataset_id,
        title=f"{result.value_column} by {result.group_column} (ANOVA groups)",
        description=(
            f"Boxplots for {group_note}; up to {_MAX_BOXPLOT_VALUES_PER_GROUP} "
            f"values sampled per group. Original sample sizes: {sample_sizes}."
        ),
        category="comparison",
        mark="boxplot",
        data={"values": values},
        encoding={
            "x": {"field": result.group_column, "type": "nominal"},
            "y": {"field": result.value_column, "type": "quantitative"},
        },
    )
    payload = spec.model_dump(mode="json")
    return Artifact(
        id=make_artifact_id("chart", payload),
        type=ArtifactType.CHART_SPEC,
        project_id=project_id,
        session_id=session_id,
        parents=parents or [],
        payload=payload,
        code_ref=code_ref(create_anova_boxplot_artifact),
        plain_language=(
            f"ANOVA boxplots show {group_note}, capped at "
            f"{_MAX_BOXPLOT_VALUES_PER_GROUP} values per group; original sample sizes: "
            f"{sample_sizes}."
        ),
    )


def _stat_plain_language(result: StatTestResult) -> str:
    """One-sentence, deterministic summary of a statistical test result."""
    label = result.test_type.replace("_", " ")
    if result.category_column:
        subject = f"{result.group_column} vs {result.category_column}"
    else:
        subject = f"{result.value_column} across {result.group_column}"
    statistic = "n/a" if result.statistic is None else f"{result.statistic:.4g}"
    if result.p_value is None:
        p_value = "n/a"
    elif result.p_value <= 0:
        p_value = "< floating-point resolution"
    else:
        p_value = f"{result.p_value:.4g}"
    return f"{label} of {subject}: statistic={statistic}, p={p_value}, n={result.sample_size}."


def _independent_t_test(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    if len(grouped) != 2:
        raise ValueError("independent_t_test requires exactly two groups")
    labels = list(grouped)
    left = grouped[labels[0]]
    right = grouped[labels[1]]
    test = cast(Any, stats.ttest_ind(left, right, equal_var=False, nan_policy="omit"))
    assumptions = [
        *_normality_checks(stats, grouped),
        StatAssumptionCheck(
            name="variance_model",
            status="passed",
            message="Welch's t-test does not assume equal group variances.",
        ),
        StatAssumptionCheck(
            name="independence",
            status="not_applicable",
            message="Group independence must be established from study design.",
        ),
    ]
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="independent_t_test",
        group_column=group_column,
        value_column=value_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        effect_size=_round_float(_cohens_d(left, right)),
        sample_size=int(len(left) + len(right)),
        groups={label: int(len(values)) for label, values in grouped.items()},
        assumptions=assumptions,
    )


def _matched_pair_values(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    pair_column: str,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """Complete matched pairs as (left, right, sorted labels); left/right follow label order."""
    working = cast(
        pd.DataFrame,
        frame[[pair_column, group_column, value_column]],
    ).dropna().copy()
    working[group_column] = working[group_column].astype(str)
    if working.duplicated(subset=[pair_column, group_column]).any():
        raise ValueError(
            "paired_t_test requires one observation per pair key and group"
        )
    labels = sorted(working[group_column].drop_duplicates().tolist())
    if len(labels) != 2:
        raise ValueError("paired_t_test requires exactly two groups")
    pivot = working.pivot(
        index=pair_column,
        columns=group_column,
        values=value_column,
    )
    paired = pivot[labels].apply(pd.to_numeric, errors="coerce").dropna()
    if len(paired) < 2:
        raise ValueError("paired_t_test requires at least two complete matched pairs")
    paired_left = cast(pd.Series, paired[labels[0]]).reset_index(drop=True)
    paired_right = cast(pd.Series, paired[labels[1]]).reset_index(drop=True)
    return paired_left, paired_right, labels


def _paired_t_test(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
    pair_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    paired_left, paired_right, labels = _matched_pair_values(
        frame,
        group_column=group_column,
        value_column=value_column,
        pair_column=pair_column,
    )
    differences = paired_left - paired_right
    test = cast(Any, stats.ttest_rel(paired_left, paired_right, nan_policy="omit"))
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="paired_t_test",
        group_column=group_column,
        value_column=value_column,
        pair_column=pair_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        effect_size=_round_float(_paired_cohens_d(paired_left, paired_right)),
        sample_size=int(len(paired_left)),
        groups={label: int(len(paired_left)) for label in labels},
        assumptions=[
            *_normality_checks(stats, {"paired_differences": differences}),
            StatAssumptionCheck(
                name="pairing_key",
                status="passed",
                message=f"Matched observations by explicit key {pair_column}.",
            ),
        ],
    )


def _chi_square_independence(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    category_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    working = cast(pd.DataFrame, frame[[group_column, category_column]]).dropna()
    table = pd.crosstab(working[group_column], working[category_column])
    statistic, p_value, dof, expected = cast(
        tuple[Any, Any, Any, Any],
        stats.chi2_contingency(table),
    )
    expected_array = np.asarray(expected, dtype=float)
    low_expected_share = float((expected_array < 5).mean())
    if low_expected_share > _CHI2_LOW_EXPECTED_SHARE:
        if table.shape == (2, 2):
            return _fisher_exact_2x2(
                stats,
                table,
                dataset_id=dataset_id,
                group_column=group_column,
                category_column=category_column,
                low_expected_share=low_expected_share,
            )
        raise ValueError(
            "chi_square_independence is unreliable here: "
            f"{low_expected_share:.0%} of expected cell frequencies are below 5 and the "
            f"table is {table.shape[0]}x{table.shape[1]}, so the Fisher exact fallback "
            "(2x2 only) does not apply. Merge sparse categories or collect more data."
        )
    warnings: list[StatWarning] = []
    if bool(cast(Any, expected < 5).any()):
        warnings.append(
            StatWarning(
                code="low_expected_frequency",
                message="At least one expected cell frequency is below 5.",
            )
        )
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="chi_square_independence",
        group_column=group_column,
        category_column=category_column,
        statistic=_round_float(statistic),
        p_value=_finite_float(p_value),
        effect_size=_round_float(_corrected_cramers_v_from_counts(table.to_numpy())),
        degrees_of_freedom=int(dof),
        sample_size=int(table.to_numpy().sum()),
        groups={str(index): int(table.loc[index].sum()) for index in table.index},
        assumptions=[
            StatAssumptionCheck(
                name="expected_frequency",
                status="warn" if warnings else "passed",
                message="Expected frequencies should generally be at least 5.",
            )
        ],
        warnings=warnings,
    )


def _fisher_exact_2x2(
    stats,
    table: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    category_column: str,
    low_expected_share: float,
) -> StatTestResult:
    """Deterministic fallback when chi-square expected counts are too sparse."""
    counts = table.to_numpy()
    _, p_value = cast(tuple[Any, Any], stats.fisher_exact(counts))
    odds = _haldane_anscombe_odds_ratio(counts)
    correction_note = ""
    if bool((counts == 0).any()):
        # A zero cell makes the sample odds ratio 0 or infinite; the
        # Haldane-Anscombe 0.5 correction keeps the estimate (and its Woolf
        # CI) finite and non-degenerate.
        correction_note = (
            " The odds ratio is Haldane-Anscombe corrected because the table "
            "contains a zero cell."
        )
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="fisher_exact",
        group_column=group_column,
        category_column=category_column,
        statistic=_round_float(odds),
        p_value=_finite_float(p_value),
        # The odds ratio is the reported effect for a 2x2 table; its Woolf CI
        # comes from _effect_size_ci when effect_ci is requested.
        effect_size=_round_float(odds),
        sample_size=int(counts.sum()),
        groups={str(index): int(table.loc[index].sum()) for index in table.index},
        assumptions=[
            StatAssumptionCheck(
                name="expected_frequency",
                status="warn",
                message=(
                    f"{low_expected_share:.0%} of expected cell frequencies are below 5; "
                    "the Fisher exact test replaces chi-square on this 2x2 table."
                ),
            )
        ],
        warnings=[
            StatWarning(
                code="fisher_exact_fallback",
                message=(
                    "Chi-square expected-frequency assumption failed; the Fisher exact "
                    "test was used instead." + correction_note
                ),
            )
        ],
    )


# Tests whose effect size can be recomputed from (group, value) samples; the
# paired test and contingency tests use the dedicated paths in _effect_size_ci.
_EFFECT_CI_STATISTICS: dict[str, Callable[..., float]] = {}


def _effect_size_ci(
    frame: pd.DataFrame,
    *,
    test_type: str,
    group_column: str | None,
    value_column: str | None,
    category_column: str | None = None,
    pair_column: str | None = None,
) -> tuple[float | None, float | None, list[StatWarning]]:
    """Effect-size interval for the reported effect (bounded, seeded bootstrap
    for resampling paths; analytic Woolf interval for the Fisher odds ratio)."""
    notes: list[StatWarning] = []
    if test_type == "paired_t_test" and group_column and value_column and pair_column:
        return _paired_effect_ci(
            frame,
            group_column=group_column,
            value_column=value_column,
            pair_column=pair_column,
        )
    if test_type == "chi_square_independence" and group_column and category_column:
        return _contingency_effect_ci(
            frame, group_column=group_column, category_column=category_column
        )
    if test_type == "fisher_exact" and group_column and category_column:
        return _fisher_effect_ci(
            frame, group_column=group_column, category_column=category_column
        )
    statistic = _EFFECT_CI_STATISTICS.get(test_type)
    if statistic is None or not group_column or not value_column:
        notes.append(
            StatWarning(
                code="effect_ci_unavailable",
                severity="info",
                message=f"No bootstrap effect-size interval is defined for {test_type}.",
            )
        )
        return None, None, notes
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    series_list = list(grouped.values())
    total = sum(len(values) for values in series_list)
    if total > _EFFECT_CI_MAX_TOTAL_N:
        scale = _EFFECT_CI_MAX_TOTAL_N / total
        series_list = [
            values.sample(
                n=max(2, min(len(values), int(len(values) * scale))),
                random_state=_EFFECT_CI_SEED,
            )
            for values in series_list
        ]
        notes.append(
            StatWarning(
                code="effect_ci_subsampled",
                severity="info",
                message=(
                    f"Effect-size bootstrap used a deterministic subsample of "
                    f"{sum(len(values) for values in series_list)} of {total} rows."
                ),
            )
        )
    samples = tuple(values.to_numpy(dtype=float) for values in series_list)
    low, high, bootstrap_notes = _bounded_bootstrap_ci(samples, statistic, paired=False)
    notes.extend(bootstrap_notes)
    return low, high, notes


def _bounded_bootstrap_ci(
    samples: tuple[np.ndarray, ...],
    statistic: Callable[..., float],
    *,
    paired: bool,
) -> tuple[float | None, float | None, list[StatWarning]]:
    notes: list[StatWarning] = []
    stats = _scipy_stats()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.bootstrap(
                samples,
                statistic,
                n_resamples=_EFFECT_CI_RESAMPLES,
                method="BCa",
                paired=paired,
                vectorized=False,
                random_state=np.random.default_rng(_EFFECT_CI_SEED),
            )
    except Exception:
        notes.append(
            StatWarning(
                code="effect_ci_failed",
                severity="info",
                message="The BCa bootstrap did not converge on this data; no interval reported.",
            )
        )
        return None, None, notes
    low = _round_float(result.confidence_interval.low)
    high = _round_float(result.confidence_interval.high)
    if low is None or high is None:
        notes.append(
            StatWarning(
                code="effect_ci_failed",
                severity="info",
                message="The BCa bootstrap produced non-finite bounds; no interval reported.",
            )
        )
        return None, None, notes
    return low, high, notes


def _paired_effect_ci(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    pair_column: str,
) -> tuple[float | None, float | None, list[StatWarning]]:
    """Bootstrap CI for Cohen's dz plus an analytic mean-difference CI note."""
    notes: list[StatWarning] = []
    paired_left, paired_right, _ = _matched_pair_values(
        frame,
        group_column=group_column,
        value_column=value_column,
        pair_column=pair_column,
    )
    differences = cast(pd.Series, paired_left - paired_right)
    if len(differences) > _EFFECT_CI_MAX_TOTAL_N:
        differences = differences.sample(
            n=_EFFECT_CI_MAX_TOTAL_N, random_state=_EFFECT_CI_SEED
        )
        notes.append(
            StatWarning(
                code="effect_ci_subsampled",
                severity="info",
                message=(
                    f"Effect-size bootstrap used a deterministic subsample of "
                    f"{_EFFECT_CI_MAX_TOTAL_N} of {len(paired_left)} pairs."
                ),
            )
        )
    values = differences.to_numpy(dtype=float)
    mean_ci = _mean_difference_ci(values)
    if mean_ci is not None:
        notes.append(
            StatWarning(
                code="paired_mean_difference_ci",
                severity="info",
                message=(
                    f"Mean paired difference {np.mean(values):.6g} with 95% CI "
                    f"[{mean_ci[0]:.6g}, {mean_ci[1]:.6g}]."
                ),
            )
        )
    low, high, bootstrap_notes = _bounded_bootstrap_ci(
        (values,), _cohens_dz_np, paired=False
    )
    notes.extend(bootstrap_notes)
    return low, high, notes


def _mean_difference_ci(differences: np.ndarray) -> tuple[float, float] | None:
    stats = _scipy_stats()
    n = differences.size
    if n < 2:
        return None
    spread = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))
    margin = float(stats.t.ppf(0.975, n - 1)) * spread / math.sqrt(n)
    if not math.isfinite(margin):
        return None
    return mean - margin, mean + margin


def _contingency_effect_ci(
    frame: pd.DataFrame,
    *,
    group_column: str,
    category_column: str,
) -> tuple[float | None, float | None, list[StatWarning]]:
    """Paired-rows bootstrap CI for the bias-corrected Cramér's V."""
    notes: list[StatWarning] = []
    working = cast(pd.DataFrame, frame[[group_column, category_column]]).dropna()
    if len(working) > _EFFECT_CI_MAX_TOTAL_N:
        working = working.sample(n=_EFFECT_CI_MAX_TOTAL_N, random_state=_EFFECT_CI_SEED)
        notes.append(
            StatWarning(
                code="effect_ci_subsampled",
                severity="info",
                message=(
                    f"Effect-size bootstrap used a deterministic subsample of "
                    f"{_EFFECT_CI_MAX_TOTAL_N} rows."
                ),
            )
        )
    group_codes = pd.factorize(working[group_column].astype(str))[0]
    category_codes = pd.factorize(working[category_column].astype(str))[0]
    low, high, bootstrap_notes = _bounded_bootstrap_ci(
        (group_codes, category_codes), _corrected_cramers_v_np, paired=True
    )
    notes.extend(bootstrap_notes)
    return low, high, notes


def _fisher_effect_ci(
    frame: pd.DataFrame,
    *,
    group_column: str,
    category_column: str,
) -> tuple[float | None, float | None, list[StatWarning]]:
    """Woolf logit interval for the (Haldane-Anscombe corrected) odds ratio."""
    notes: list[StatWarning] = []
    working = cast(pd.DataFrame, frame[[group_column, category_column]]).dropna()
    counts = pd.crosstab(working[group_column], working[category_column]).to_numpy()
    if counts.shape != (2, 2):
        notes.append(
            StatWarning(
                code="effect_ci_unavailable",
                severity="info",
                message="The odds-ratio interval is only defined for a 2x2 table.",
            )
        )
        return None, None, notes
    odds = _haldane_anscombe_odds_ratio(counts)
    shift = 0.5 if bool((counts == 0).any()) else 0.0
    cells = counts.astype(float).ravel() + shift
    if bool((cells <= 0).any()) or odds <= 0:
        notes.append(
            StatWarning(
                code="effect_ci_failed",
                severity="info",
                message="The Woolf odds-ratio interval is undefined on this table.",
            )
        )
        return None, None, notes
    standard_error = math.sqrt(float((1.0 / cells).sum()))
    z_975 = 1.959963984540054
    log_odds = math.log(odds)
    return (
        _round_float(math.exp(log_odds - z_975 * standard_error)),
        _round_float(math.exp(log_odds + z_975 * standard_error)),
        notes,
    )


def _haldane_anscombe_odds_ratio(counts: np.ndarray) -> float:
    """Sample odds ratio; 0.5 added to every cell when any cell is zero."""
    shift = 0.5 if bool((counts == 0).any()) else 0.0
    a, b = float(counts[0][0]) + shift, float(counts[0][1]) + shift
    c, d = float(counts[1][0]) + shift, float(counts[1][1]) + shift
    return (a * d) / (b * c)


def _cohens_dz_np(differences: np.ndarray) -> float:
    spread = float(np.std(differences, ddof=1))
    if spread == 0:
        return 0.0
    return float(np.mean(differences)) / spread


def _corrected_cramers_v_np(group_codes: np.ndarray, category_codes: np.ndarray) -> float:
    group_codes = np.asarray(group_codes, dtype=int)
    category_codes = np.asarray(category_codes, dtype=int)
    table = np.zeros((int(group_codes.max()) + 1, int(category_codes.max()) + 1))
    np.add.at(table, (group_codes, category_codes), 1.0)
    return _corrected_cramers_v_from_counts(table)


def _corrected_cramers_v_from_counts(counts: np.ndarray) -> float:
    """Bergsma's bias-corrected Cramér's V (degrees-of-freedom corrected)."""
    table = np.asarray(counts, dtype=float)
    n = float(table.sum())
    row_totals = table.sum(axis=1, keepdims=True)
    column_totals = table.sum(axis=0, keepdims=True)
    rows_observed = int((row_totals > 0).sum())
    columns_observed = int((column_totals > 0).sum())
    if n <= 1 or rows_observed < 2 or columns_observed < 2:
        return 0.0
    expected = row_totals @ column_totals / n
    mask = expected > 0
    chi_square = float(
        (((table - expected) ** 2 / np.where(mask, expected, 1.0))[mask]).sum()
    )
    phi2 = chi_square / n
    phi2_corrected = max(
        0.0, phi2 - (rows_observed - 1) * (columns_observed - 1) / (n - 1)
    )
    rows_corrected = rows_observed - (rows_observed - 1) ** 2 / (n - 1)
    columns_corrected = columns_observed - (columns_observed - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    if denominator <= 0:
        return 0.0
    return math.sqrt(phi2_corrected / denominator)


def _cohens_d_np(left: np.ndarray, right: np.ndarray) -> float:
    left_n, right_n = len(left), len(right)
    pooled = (
        ((left_n - 1) * float(np.var(left, ddof=1)) + (right_n - 1) * float(np.var(right, ddof=1)))
        / (left_n + right_n - 2)
    ) ** 0.5
    if pooled == 0:
        return 0.0
    return (float(np.mean(left)) - float(np.mean(right))) / pooled


def _rank_biserial_np(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([left, right]))
    left_n, right_n = len(left), len(right)
    u_statistic = float(ranks[:left_n].sum()) - left_n * (left_n + 1) / 2
    return (2.0 * u_statistic) / (left_n * right_n) - 1.0


def _eta_squared_np(*groups: np.ndarray) -> float:
    all_values = np.concatenate(groups)
    grand_mean = float(all_values.mean())
    between = sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in groups)
    total = float(((all_values - grand_mean) ** 2).sum())
    return 0.0 if total == 0 else between / total


def _epsilon_squared_np(*groups: np.ndarray) -> float:
    from scipy.stats import kruskal

    statistic = float(kruskal(*groups).statistic)
    sample_size = sum(len(group) for group in groups)
    group_count = len(groups)
    if sample_size <= group_count:
        return 0.0
    return max(0.0, (statistic - group_count + 1) / (sample_size - group_count))


_EFFECT_CI_STATISTICS.update(
    {
        "independent_t_test": _cohens_d_np,
        "mann_whitney_u": _rank_biserial_np,
        "one_way_anova": _eta_squared_np,
        "welch_anova": _eta_squared_np,
        "kruskal_wallis": _epsilon_squared_np,
    }
)


def _one_way_anova(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    if len(grouped) < 2:
        raise ValueError("one_way_anova requires at least two groups")
    test = cast(Any, stats.f_oneway(*grouped.values()))
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="one_way_anova",
        group_column=group_column,
        value_column=value_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        effect_size=_round_float(_eta_squared(grouped)),
        sample_size=int(sum(len(values) for values in grouped.values())),
        groups={label: int(len(values)) for label, values in grouped.items()},
        assumptions=[*_normality_checks(stats, grouped), _variance_check(stats, grouped)],
    )


def _welch_anova(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
) -> StatTestResult:
    """Welch's unequal-variance ANOVA for an explicit mean estimand."""
    stats = _scipy_stats()
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    if len(grouped) < 2:
        raise ValueError("welch_anova requires at least two groups")
    test = cast(Any, stats.f_oneway(*grouped.values(), equal_var=False))
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="welch_anova",
        group_column=group_column,
        value_column=value_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        # Descriptive separation only; this is not a Welch-specific CI.
        effect_size=_round_float(_eta_squared(grouped)),
        sample_size=int(sum(len(values) for values in grouped.values())),
        groups={label: int(len(values)) for label, values in grouped.items()},
        assumptions=[
            *_normality_checks(stats, grouped),
            StatAssumptionCheck(
                name="variance_model",
                status="passed",
                message="Welch ANOVA does not assume equal group variances.",
            ),
            StatAssumptionCheck(
                name="independence",
                status="not_applicable",
                message="Group independence must be established from study design.",
            ),
        ],
    )


def _mann_whitney_u(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    if len(grouped) != 2:
        raise ValueError("mann_whitney_u requires exactly two groups")
    left, right = list(grouped.values())
    test = cast(Any, stats.mannwhitneyu(left, right, alternative="two-sided"))
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="mann_whitney_u",
        group_column=group_column,
        value_column=value_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        effect_size=_round_float(_rank_biserial(float(test.statistic), left, right)),
        sample_size=int(len(left) + len(right)),
        groups={label: int(len(values)) for label, values in grouped.items()},
        assumptions=[
            StatAssumptionCheck(
                name="non_parametric",
                status="not_applicable",
                message="Mann-Whitney U does not assume normally distributed values.",
            )
        ],
    )


def _kruskal_wallis(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
    grouped = _numeric_groups(frame, group_column=group_column, value_column=value_column)
    if len(grouped) < 2:
        raise ValueError("kruskal_wallis requires at least two groups")
    test = cast(Any, stats.kruskal(*grouped.values()))
    return StatTestResult(
        dataset_id=dataset_id,
        test_type="kruskal_wallis",
        group_column=group_column,
        value_column=value_column,
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        effect_size=_round_float(_kruskal_epsilon_squared(float(test.statistic), grouped)),
        sample_size=int(sum(len(values) for values in grouped.values())),
        groups={label: int(len(values)) for label, values in grouped.items()},
        assumptions=[
            StatAssumptionCheck(
                name="non_parametric",
                status="not_applicable",
                message="Kruskal-Wallis does not assume normally distributed values.",
            )
        ],
    )


def _display_sample(values: pd.Series, limit: int) -> pd.Series:
    """Deterministic uniform sample, avoiding row-order bias in display artifacts."""
    if len(values) <= limit:
        return values
    return values.sample(n=limit, random_state=0, replace=False).sort_index()


def _numeric_groups(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
) -> dict[str, pd.Series]:
    working = pd.DataFrame(
        {
            "group": frame[group_column],
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    ).dropna(subset=["group", "value"])
    working["group"] = working["group"].astype(str)
    grouped = {
        str(label): cast(pd.Series, group["value"]).reset_index(drop=True)
        for label, group in working.groupby("group", dropna=False)
    }
    if any(len(values) < 2 for values in grouped.values()):
        raise ValueError("Each group must contain at least two numeric values")
    return grouped


_SHAPIRO_MAX_RELIABLE_N = 5_000
_SHAPIRO_SAMPLE_SEED = 0


def _normality_checks(stats, grouped: dict[str, pd.Series]) -> list[StatAssumptionCheck]:
    checks: list[StatAssumptionCheck] = []
    for label, values in grouped.items():
        if len(values) < 3:
            checks.append(
                StatAssumptionCheck(
                    name="normality",
                    status="not_applicable",
                    message=f"Group {label} has fewer than 3 observations.",
                )
            )
            continue
        # Shapiro-Wilk divides by the sample variance (W = numer / denom); a
        # constant/near-constant group makes denom ~0 -> NaN + RuntimeWarning
        if _is_degenerate(values):
            checks.append(
                StatAssumptionCheck(
                    name="normality",
                    status="not_applicable",
                    message=f"Group {label} has zero variance; normality is undefined.",
                )
            )
            continue
        tested_values = values
        sample_note = ""
        if len(values) > _SHAPIRO_MAX_RELIABLE_N:
            tested_values = values.sample(
                n=_SHAPIRO_MAX_RELIABLE_N,
                random_state=_SHAPIRO_SAMPLE_SEED,
                replace=False,
            ).reset_index(drop=True)
            sample_note = f" Deterministic sample n={_SHAPIRO_MAX_RELIABLE_N} from N={len(values)}."
        test = _safe_scipy(lambda v=tested_values: stats.shapiro(v))
        if test is None:
            checks.append(
                StatAssumptionCheck(
                    name="normality",
                    status="not_applicable",
                    message=f"Group {label} is near-constant; normality is undefined.",
                )
            )
            continue
        checks.append(
            StatAssumptionCheck(
                name="normality",
                status=_assumption_status(test.pvalue),
                statistic=_round_float(test.statistic),
                p_value=_finite_float(test.pvalue),
                message=(f"Shapiro-Wilk normality check for group {label}.{sample_note}"),
            )
        )
    return checks


def _variance_check(stats, grouped: dict[str, pd.Series]) -> StatAssumptionCheck:
    # Levene also divides by within-group spread; a zero-variance group (or fewer
    # than two usable groups) yields NaN + RuntimeWarning, so skip cleanly.
    usable = [values for values in grouped.values() if len(values) >= 2]
    if len(usable) < 2 or any(_is_degenerate(values) for values in usable):
        return StatAssumptionCheck(
            name="variance_homogeneity",
            status="not_applicable",
            message="Variance homogeneity is undefined for constant or tiny groups.",
        )
    test = _safe_scipy(lambda: stats.levene(*usable))
    if test is None:
        return StatAssumptionCheck(
            name="variance_homogeneity",
            status="not_applicable",
            message="Variance homogeneity is undefined for near-constant groups.",
        )
    return StatAssumptionCheck(
        name="variance_homogeneity",
        status=_assumption_status(test.pvalue),
        statistic=_round_float(test.statistic),
        p_value=_finite_float(test.pvalue),
        message="Levene variance homogeneity check across groups.",
    )


def _is_degenerate(values: pd.Series) -> bool:
    """A group scipy's variance-based checks cannot handle: constant values."""
    return values.nunique(dropna=True) < 2


def _safe_scipy(func: Callable[[], Any]) -> Any | None:
    """Run a SciPy test and reject degenerate warnings or NaN results."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            test = func()
        except (RuntimeWarning, ValueError):
            return None
    pvalue = float(test.pvalue)
    if pvalue != pvalue:  # NaN slipped through without a warning
        return None
    return test


def _assumption_status(p_value: Any) -> Literal["passed", "warn", "not_applicable"]:
    """Map a p-value to an assumption-check status."""
    numeric = float(p_value)
    if numeric != numeric:  # NaN
        return "not_applicable"
    return "passed" if numeric >= 0.05 else "warn"


def _cohens_d(left: pd.Series, right: pd.Series) -> float:
    left_std = float(cast(float, left.std(ddof=1)))
    right_std = float(cast(float, right.std(ddof=1)))
    pooled = (
        ((len(left) - 1) * left_std**2 + (len(right) - 1) * right_std**2)
        / (len(left) + len(right) - 2)
    ) ** 0.5
    if pooled == 0:
        return 0.0
    # R4: the sign (first sorted group minus second) is the direction claim.
    return (float(left.mean()) - float(right.mean())) / pooled


def _paired_cohens_d(left: pd.Series, right: pd.Series) -> float:
    differences = left.reset_index(drop=True) - right.reset_index(drop=True)
    standard_deviation = float(cast(float, differences.std(ddof=1)))
    if standard_deviation == 0:
        return 0.0
    return float(differences.mean()) / standard_deviation


def _eta_squared(grouped: dict[str, pd.Series]) -> float:
    all_values = pd.concat(list(grouped.values()), ignore_index=True)
    grand_mean = float(all_values.mean())
    between_groups = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2 for values in grouped.values()
    )
    total = float(((all_values - grand_mean) ** 2).sum())
    return 0.0 if total == 0 else between_groups / total


def _rank_biserial(statistic: float, left: pd.Series, right: pd.Series) -> float:
    # Signed: positive when the first sorted group is stochastically larger.
    return (2.0 * statistic) / (len(left) * len(right)) - 1.0


def _kruskal_epsilon_squared(
    statistic: float,
    grouped: dict[str, pd.Series],
) -> float:
    sample_size = sum(len(values) for values in grouped.values())
    group_count = len(grouped)
    if sample_size <= group_count:
        return 0.0
    return max(0.0, (statistic - group_count + 1) / (sample_size - group_count))


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _round_float(value: Any, *, digits: int = 6) -> float | None:
    """Round finite values and collapse non-finite values to ``None``."""
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _finite_float(value: Any) -> float | None:
    """Return finite values unchanged so tiny probabilities retain precision."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _scipy_stats() -> Any:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing.
        raise RuntimeError(
            "scipy is required for M5 statistical tests. Install project dependencies."
        ) from exc
    return stats


# ---------------------------------------------------------------------------
# Hardened correlation screen (E1.5): Holm default, Spearman, min pairwise n.
# ---------------------------------------------------------------------------

_CORRELATION_METHODS = ("pearson", "spearman")
_CORRELATION_CORRECTIONS = ("holm", "fdr_bh")
_CORRELATION_MAX_COLUMNS = 24
_CORRELATION_MAX_PUBLISHED_ROWS = 50
_CORRELATION_TRIVIAL_ABS = 0.999
_CORRELATION_DEFAULT_MIN_PAIRWISE_N = 10


@dataclass(slots=True)
class CorrelationScreenResult:
    """Correlation table plus the method/missingness facts a receipt needs."""

    table: AnalysisTable
    correlation_method: str
    correction_method: str
    min_pairwise_n: int
    columns_considered: list[str]
    pairs_tested: int
    pairs_insufficient_n: int
    pairs_degenerate: int


def screen_correlations(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    correction_method: str = "holm",
    min_pairwise_n: int = _CORRELATION_DEFAULT_MIN_PAIRWISE_N,
) -> CorrelationScreenResult:
    """Pairwise correlation screen with explicit method and missingness semantics.

    Holm is the default correction: it controls the family-wise error rate with
    no positive-dependence assumption, which an arbitrary correlation family
    does not guarantee for Benjamini-Hochberg. Pairs with fewer than
    ``min_pairwise_n`` complete rows are marked ``insufficient_n`` instead of
    receiving a p-value.
    """
    if method not in _CORRELATION_METHODS:
        raise ValueError("method must be `pearson` or `spearman`.")
    if correction_method not in _CORRELATION_CORRECTIONS:
        raise ValueError("correction_method must be `holm` or `fdr_bh`.")
    if min_pairwise_n < 3:
        raise ValueError("min_pairwise_n must be at least 3.")
    if columns is None:
        resolved = [
            str(column) for column in frame.columns if is_numeric_dtype(frame[column])
        ][:_CORRELATION_MAX_COLUMNS]
    else:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Columns not found in the dataset: {missing}.")
        if len(columns) > _CORRELATION_MAX_COLUMNS:
            raise ValueError(
                f"At most {_CORRELATION_MAX_COLUMNS} columns can be tested in one call."
            )
        resolved = [str(column) for column in columns]
    if len(resolved) < 2:
        raise ValueError("Correlation needs at least two numeric columns.")
    numeric_by_column = {
        column: cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))
        for column in resolved
    }
    stats = _scipy_stats()
    tested: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    p_values: list[float] = []
    degenerate = 0
    for column_a, column_b in combinations(resolved, 2):
        paired = pd.DataFrame(
            {"a": numeric_by_column[column_a], "b": numeric_by_column[column_b]}
        ).dropna()
        base_row: dict[str, Any] = {
            "column_a": column_a,
            "column_b": column_b,
            "dataset": dataset_name,
            "pairwise_complete_n": int(len(paired)),
            "excluded_pair_n": int(len(frame) - len(paired)),
            "missing_policy": "pairwise_complete",
        }
        if len(paired) < min_pairwise_n:
            insufficient.append(
                {
                    **base_row,
                    "coefficient": None,
                    "p_value": None,
                    "adjusted_p": None,
                    "insufficient_n": True,
                }
            )
            continue
        series_a = cast(pd.Series, paired["a"])
        series_b = cast(pd.Series, paired["b"])
        if series_a.nunique() < 2 or series_b.nunique() < 2:
            degenerate += 1
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            if method == "pearson":
                test = cast(Any, stats.pearsonr(series_a.to_numpy(), series_b.to_numpy()))
            else:
                test = cast(Any, stats.spearmanr(series_a.to_numpy(), series_b.to_numpy()))
        coefficient = float(test.statistic)
        p_value = float(test.pvalue)
        if pd.isna(coefficient) or pd.isna(p_value):
            degenerate += 1
            continue
        tested.append(
            {
                **base_row,
                "coefficient": _round_float(coefficient, digits=4),
                "p_value": p_value,
                "insufficient_n": False,
                "is_trivial_pair": abs(coefficient) >= _CORRELATION_TRIVIAL_ABS,
            }
        )
        p_values.append(p_value)
    if not tested and not insufficient:
        raise ValueError(
            "No column pair had enough pairwise-complete numeric rows to test."
        )
    if tested:
        from statsmodels.stats.multitest import multipletests

        _, adjusted, _, _ = multipletests(p_values, method=correction_method)
        for row, adjusted_p in zip(tested, adjusted, strict=True):
            row["adjusted_p"] = float(adjusted_p)
        tested.sort(
            key=lambda row: (-abs(float(row["coefficient"])), str(row["column_a"]))
        )
    published = (tested + insufficient)[:_CORRELATION_MAX_PUBLISHED_ROWS]
    table = AnalysisTable(
        dataset_id=dataset_id,
        title=f"{dataset_name} - Correlation screen (multiplicity adjusted)",
        kind="correlation",
        description=(
            f"{method} correlations over {len(tested)} tested pairs in {dataset_name}; "
            f"p-values adjusted with {correction_method} across every tested pair. "
            f"{len(insufficient)} pair(s) below the minimum pairwise n of "
            f"{min_pairwise_n} are marked insufficient_n without a p-value; "
            f"{degenerate} degenerate pair(s) were skipped. "
            f"Showing {len(published)} of {len(tested) + len(insufficient)} rows."
        ),
        rows=published,
    )
    return CorrelationScreenResult(
        table=table,
        correlation_method=method,
        correction_method=correction_method,
        min_pairwise_n=min_pairwise_n,
        columns_considered=resolved,
        pairs_tested=len(tested),
        pairs_insufficient_n=len(insufficient),
        pairs_degenerate=degenerate,
    )

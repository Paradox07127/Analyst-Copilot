from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import pandas as pd

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
from eda_platform.schemas.artifacts import Artifact, ArtifactType
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


def _paired_t_test(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    value_column: str,
    pair_column: str,
) -> StatTestResult:
    stats = _scipy_stats()
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
        sample_size=int(len(paired)),
        groups={label: int(len(paired)) for label in labels},
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
    table = pd.crosstab(frame[group_column], frame[category_column])
    statistic, p_value, dof, expected = cast(
        tuple[Any, Any, Any, Any],
        stats.chi2_contingency(table),
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
        effect_size=_round_float(_cramers_v(float(statistic), table)),
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
    return abs(float(left.mean()) - float(right.mean())) / pooled


def _paired_cohens_d(left: pd.Series, right: pd.Series) -> float:
    differences = left.reset_index(drop=True) - right.reset_index(drop=True)
    standard_deviation = float(cast(float, differences.std(ddof=1)))
    if standard_deviation == 0:
        return 0.0
    return abs(float(differences.mean())) / standard_deviation


def _cramers_v(statistic: float, table: pd.DataFrame) -> float:
    sample_size = int(table.to_numpy().sum())
    denominator_dimension = min(table.shape[0] - 1, table.shape[1] - 1)
    if sample_size == 0 or denominator_dimension <= 0:
        return 0.0
    return math.sqrt(statistic / (sample_size * denominator_dimension))


def _eta_squared(grouped: dict[str, pd.Series]) -> float:
    all_values = pd.concat(list(grouped.values()), ignore_index=True)
    grand_mean = float(all_values.mean())
    between_groups = sum(
        len(values) * (float(values.mean()) - grand_mean) ** 2 for values in grouped.values()
    )
    total = float(((all_values - grand_mean) ** 2).sum())
    return 0.0 if total == 0 else between_groups / total


def _rank_biserial(statistic: float, left: pd.Series, right: pd.Series) -> float:
    return abs(1.0 - (2.0 * statistic) / (len(left) * len(right)))


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

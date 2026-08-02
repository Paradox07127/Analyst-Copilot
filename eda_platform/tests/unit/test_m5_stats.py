import pandas as pd

from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.stat_tests import (
    _stat_plain_language,
    create_stat_test_artifact,
    run_stat_test,
)


def test_t_test_returns_effect_size_assumptions_and_multiple_comparison_warning() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 10 + ["B"] * 10,
            "revenue": [9, 10, 11, 10, 12, 11, 9, 10, 11, 12]
            + [21, 20, 22, 23, 19, 21, 22, 20, 23, 21],
        }
    )

    result = run_stat_test(
        frame,
        dataset_id="ds_sales",
        test_type="independent_t_test",
        group_column="segment",
        value_column="revenue",
        comparison_count=3,
    )

    assert result.test_type == "independent_t_test"
    assert result.group_column == "segment"
    assert result.value_column == "revenue"
    assert result.sample_size == 20
    assert result.p_value is not None and result.p_value < 0.001
    # Signed Cohen's d (R4): group A sits far below group B.
    assert result.effect_size is not None and result.effect_size < -5.0
    assert any(warning.code == "multiple_comparisons" for warning in result.warnings)
    assert result.correction_method == "bonferroni"
    assert result.adjusted_p_value == min(1.0, result.p_value * 3)
    assert {check.name for check in result.assumptions} >= {
        "normality",
        "variance_model",
        "independence",
    }


def test_chi_square_independence_returns_expected_schema_and_artifact() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["organic"] * 30 + ["paid"] * 30,
            "converted": ["yes"] * 24 + ["no"] * 6 + ["yes"] * 8 + ["no"] * 22,
        }
    )

    result = run_stat_test(
        frame,
        dataset_id="ds_marketing",
        test_type="chi_square_independence",
        group_column="channel",
        category_column="converted",
    )
    artifact = create_stat_test_artifact(
        result,
        project_id="project_demo",
        session_id="run_demo",
        parents=["prof_123"],
    )
    restored = StatTestResult.model_validate(artifact.payload)

    assert result.p_value is not None and result.p_value < 0.001
    assert restored.test_type == "chi_square_independence"
    assert restored.degrees_of_freedom == 1
    assert artifact.type is ArtifactType.STAT_TEST_RESULT
    assert artifact.parents == ["prof_123"]


def test_degenerate_input_is_skipped_not_crashed() -> None:
    """A constant column makes scipy return a non-finite (or None) statistic;
    run_stat_test must raise ValueError (which the pipeline step catches and
    skips) rather than crash StatTestResult validation. Regression for a live
    'EDA pipeline failed: 2 validation errors for StatTestResult' crash."""
    import pytest

    frame = pd.DataFrame({"grp": ["a"] * 5 + ["b"] * 5, "val": [5.0] * 10})
    for test_type in ("independent_t_test", "one_way_anova"):
        with pytest.raises(ValueError, match="not computable"):
            run_stat_test(
                frame,
                dataset_id="d",
                test_type=test_type,  # type: ignore[arg-type]
                group_column="grp",
                value_column="val",
            )


def test_degenerate_input_emits_no_runtime_warning() -> None:
    import warnings

    frame = pd.DataFrame({"grp": ["a"] * 5 + ["b"] * 5, "val": [3.0] * 5 + [3, 3, 3, 3, 7.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            run_stat_test(
                frame,
                dataset_id="d",
                test_type="independent_t_test",
                group_column="grp",
                value_column="val",
            )
        except ValueError:
            pass  # skipped is fine; the point is no RuntimeWarning leaked


def test_large_group_shapiro_uses_deterministic_auditable_sample() -> None:
    group_size = 5_001
    values_a = [(index % 97) + index / 10_000 for index in range(group_size)]
    values_b = [(index % 89) + index / 9_000 + 2 for index in range(group_size)]
    frame = pd.DataFrame(
        {
            "segment": ["A"] * group_size + ["B"] * group_size,
            "value": values_a + values_b,
        }
    )

    first = run_stat_test(
        frame,
        dataset_id="ds_large",
        test_type="independent_t_test",
        group_column="segment",
        value_column="value",
    )
    second = run_stat_test(
        frame,
        dataset_id="ds_large",
        test_type="independent_t_test",
        group_column="segment",
        value_column="value",
    )
    first_normality = [check for check in first.assumptions if check.name == "normality"]
    second_normality = [check for check in second.assumptions if check.name == "normality"]

    assert len(first_normality) == 2
    assert first_normality == second_normality
    assert all(
        "Deterministic sample n=5000 from N=5001" in check.message for check in first_normality
    )


def test_welch_anova_uses_unequal_variance_model() -> None:
    from scipy import stats

    frame = pd.DataFrame(
        {
            "segment": ["A"] * 12 + ["B"] * 18 + ["C"] * 25,
            "value": [10 + index * 0.1 for index in range(12)]
            + [20 + index * 1.5 for index in range(18)]
            + [40 + index * 4.0 for index in range(25)],
        }
    )

    result = run_stat_test(
        frame,
        dataset_id="ds_welch",
        test_type="welch_anova",
        group_column="segment",
        value_column="value",
    )
    expected = stats.f_oneway(
        frame.loc[frame["segment"] == "A", "value"],
        frame.loc[frame["segment"] == "B", "value"],
        frame.loc[frame["segment"] == "C", "value"],
        equal_var=False,
    )

    assert result.test_type == "welch_anova"
    assert result.statistic == round(float(expected.statistic), 6)
    assert any(check.name == "variance_model" for check in result.assumptions)
    assert not any(check.name == "variance_homogeneity" for check in result.assumptions)


def test_underflowed_p_value_is_never_rendered_as_zero() -> None:
    result = StatTestResult(
        dataset_id="ds_underflow",
        test_type="welch_anova",
        group_column="segment",
        value_column="value",
        statistic=123.0,
        p_value=0.0,
        effect_size=0.2,
        sample_size=30,
        groups={"A": 10, "B": 10, "C": 10},
    )

    text = _stat_plain_language(result)

    assert "floating-point resolution" in text
    assert "p=0" not in text

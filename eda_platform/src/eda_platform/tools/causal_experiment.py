"""Two-tier causal_experiment executor: design check (A) and randomized ATE (B).

Tier A (default) is an honest observational design check: a Welch t / chi-square
group contrast plus a covariate SMD balance table, with a fixed disclaimer that
the difference cannot be attributed to the treatment. Tier B runs only under a
user-issued randomized-design confirmation credential (data alone cannot prove
randomization): the two-group mean difference is the ATE with a Welch
(unequal-variance) CI from statsmodels ``CompareMeans.tconfint_diff``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from eda_platform.core.permissions import action_hash
from eda_platform.core.tool_guard import (
    GuardViolation,
    check_column_exists,
    check_column_semantic_type,
    infer_column_semantic_type,
    raise_for_violations,
)
from eda_platform.schemas.artifacts import AnalysisTable
from eda_platform.schemas.stats import StatAssumptionCheck, StatTestResult, StatWarning
from eda_platform.tools.ml_baseline import _is_id_like
from eda_platform.tools.stat_tests import run_stat_test

ExperimentDesign = Literal["observational", "randomized"]

# Claim-gate whitelist anchors: only a committed run_causal_experiment receipt
# with this family plus a matching credential id licenses causal wording.
OBSERVATIONAL_DESIGN_FAMILY = "causal_design_check"
RANDOMIZED_EXPERIMENT_FAMILY = "randomized_experiment"

# The credential is issued through the existing approval mechanism (a pending
# action the user explicitly registers); its hash binds dataset + treatment.
RANDOMIZED_DESIGN_APPROVAL_KIND = "randomized_design_confirmation"

OBSERVATIONAL_CONTRAST_DISCLAIMER = (
    "This is an observed group contrast. Group assignment was not verified as "
    "randomized; the difference cannot be attributed to the treatment, and "
    "unobserved confounding cannot be ruled out even when every covariate "
    "looks balanced."
)
RANDOMIZED_DESIGN_ASSUMPTION = (
    "Treatment assignment was confirmed as randomized by the user; the "
    "confirmation credential is recorded on this receipt. The estimate is an "
    "average treatment difference under that confirmed randomization."
)
RANDOMIZED_DOWNGRADE_WARNING = (
    "design='randomized' was requested, but no active user confirmation "
    "credential exists for this dataset and treatment column, so the "
    "observational design check ran instead. Ask the user to confirm the "
    "randomized assignment before requesting the experiment analysis."
)
COVARIATE_IMBALANCE_WARNING = (
    "covariate imbalance is larger than expected under randomization; "
    "check the assignment mechanism"
)

# Standard balance diagnostic line (|SMD| > 0.1), used as a hint, never a proof.
BALANCE_SMD_THRESHOLD = 0.1
_MAX_AUTO_COVARIATES = 10
_MAX_CATEGORICAL_LEVELS = 20
_MAX_CATEGORICAL_COVARIATE_CARDINALITY = 10
_MIN_GROUP_ROWS = 2


def randomized_design_action(*, dataset_id: str, treatment_column: str) -> dict[str, Any]:
    """Canonical approval action for one randomized-design confirmation."""
    return {
        "type": RANDOMIZED_DESIGN_APPROVAL_KIND,
        "dataset_id": dataset_id,
        "treatment_column": treatment_column,
    }


def randomized_design_credential_id(*, dataset_id: str, treatment_column: str) -> str:
    """The credential id is the approval action hash: it binds dataset and
    treatment column, so the claim gate can re-derive and verify it from the
    receipt alone."""
    return action_hash(
        randomized_design_action(dataset_id=dataset_id, treatment_column=treatment_column)
    )


@dataclass(slots=True)
class CausalExperimentResult:
    design: ExperimentDesign
    requested_design: ExperimentDesign
    outcome_kind: Literal["numeric", "binary"]
    treatment_column: str
    outcome_column: str
    control_label: str
    treated_label: str
    covariate_columns: list[str]
    balance_rows: list[dict[str, Any]]
    imbalanced_covariates: list[str]
    ci_method: str
    disclaimer: str
    stat: StatTestResult
    ate_definition: str | None = None
    balance_table: AnalysisTable | None = None
    notes: list[str] = field(default_factory=list)


def run_causal_experiment(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    treatment_column: str,
    outcome_column: str,
    covariate_columns: list[str] | None = None,
    design: ExperimentDesign = "observational",
    requested_design: ExperimentDesign | None = None,
    comparison_count: int = 1,
) -> CausalExperimentResult:
    """Run one tier of the causal_experiment method family.

    ``design`` is the effective tier decided by the caller after credential
    verification; ``requested_design`` records what the agent asked for so a
    downgrade is stated explicitly in the durable result.
    """
    guard_causal_experiment_params(
        frame,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        covariate_columns=covariate_columns,
    )
    requested = requested_design or design
    outcome_kind = _resolve_outcome_kind(cast(pd.Series, frame[outcome_column]))
    covariates = _resolve_covariates(
        frame,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        covariate_columns=covariate_columns,
    )
    labels = sorted(
        str(value) for value in frame[treatment_column].dropna().unique()
    )
    control_label, treated_label = labels[0], labels[1]

    ate_definition: str | None = None
    if design == "randomized":
        stat, ate_definition = _randomized_ate(
            frame,
            dataset_id=dataset_id,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_kind=outcome_kind,
            control_label=control_label,
            treated_label=treated_label,
            comparison_count=comparison_count,
        )
        ci_method = "welch_tconfint_unequal"
        disclaimer = RANDOMIZED_DESIGN_ASSUMPTION
    else:
        stat = run_stat_test(
            frame,
            dataset_id=dataset_id,
            test_type=(
                "independent_t_test" if outcome_kind == "numeric" else "chi_square_independence"
            ),
            group_column=treatment_column,
            value_column=outcome_column if outcome_kind == "numeric" else None,
            category_column=outcome_column if outcome_kind == "binary" else None,
            comparison_count=comparison_count,
            effect_ci=True,
        )
        ci_method = observational_ci_method(stat.test_type)
        disclaimer = OBSERVATIONAL_CONTRAST_DISCLAIMER
        stat.assumptions.append(
            StatAssumptionCheck(
                name="assignment_mechanism",
                status="warn",
                message=(
                    "The assignment mechanism is unverified; this contrast is a "
                    "design check, not causal evidence."
                ),
            )
        )

    balance_rows, imbalanced = _covariate_balance(
        frame,
        treatment_column=treatment_column,
        covariates=covariates,
        control_label=control_label,
        treated_label=treated_label,
    )
    notes: list[str] = []
    if requested == "randomized" and design == "observational":
        stat.warnings.append(
            StatWarning(
                code="randomized_design_unconfirmed",
                message=RANDOMIZED_DOWNGRADE_WARNING,
            )
        )
        notes.append(RANDOMIZED_DOWNGRADE_WARNING)
    if imbalanced:
        imbalance_message = (
            f"{COVARIATE_IMBALANCE_WARNING}: {', '.join(imbalanced)}"
            if design == "randomized"
            else "Imbalanced covariates (|SMD| > "
            f"{BALANCE_SMD_THRESHOLD}): {', '.join(imbalanced)}. They are "
            "candidate confounders for this contrast."
        )
        stat.warnings.append(
            StatWarning(code="covariate_imbalance", message=imbalance_message)
        )
    stat.warnings.append(
        StatWarning(
            code=(
                "randomized_design" if design == "randomized" else "observational_design"
            ),
            severity="info" if design == "randomized" else "warn",
            message=disclaimer,
        )
    )

    result = CausalExperimentResult(
        design=design,
        requested_design=requested,
        outcome_kind=outcome_kind,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        control_label=control_label,
        treated_label=treated_label,
        covariate_columns=covariates,
        balance_rows=balance_rows,
        imbalanced_covariates=imbalanced,
        ci_method=ci_method,
        disclaimer=disclaimer,
        stat=stat,
        ate_definition=ate_definition,
        notes=notes,
    )
    result.balance_table = _balance_table(
        result, dataset_id=dataset_id, dataset_name=dataset_name
    )
    return result


def observational_ci_method(test_type: str) -> str:
    """Name the effect-CI machinery run_stat_test used for the tier-A contrast."""
    return "woolf" if test_type == "fisher_exact" else "bca_bootstrap"


def guard_causal_experiment_params(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    covariate_columns: list[str] | None,
) -> None:
    violations: list[GuardViolation | None] = []
    treatment_ok = False
    violation = check_column_semantic_type(
        "treatment_column",
        treatment_column,
        frame,
        allowed_semantic_types=["categorical"],
        fix_hint="Choose the two-valued assignment column (categorical or boolean).",
    )
    if violation is not None:
        violations.append(violation)
    else:
        distinct = int(
            frame[treatment_column].dropna().astype(str).nunique()
        )
        if distinct != 2:
            violations.append(
                GuardViolation(
                    field="treatment_column",
                    got=f"{treatment_column} ({distinct} distinct non-null values)",
                    allowed="a column with exactly two distinct non-null values",
                    fix_hint=(
                        "Filter or recode the assignment column to exactly two "
                        "groups (treated vs control)."
                    ),
                    problem="the treatment column does not define two groups.",
                )
            )
        else:
            treatment_ok = True
    exists = check_column_exists(
        "outcome_column", outcome_column, [str(name) for name in frame.columns]
    )
    if exists is not None:
        violations.append(exists)
    elif outcome_column == treatment_column:
        violations.append(
            GuardViolation(
                field="outcome_column",
                got=outcome_column,
                allowed="a column different from treatment_column",
                fix_hint="Name the measured outcome, not the assignment column.",
                problem="outcome_column equals treatment_column.",
            )
        )
    elif _resolve_outcome_kind_or_none(cast(pd.Series, frame[outcome_column])) is None:
        violations.append(
            GuardViolation(
                field="outcome_column",
                got=outcome_column,
                allowed="a numeric outcome or a two-valued (binary) outcome",
                fix_hint=(
                    "Choose a numeric measure or a binary flag; multi-category "
                    "outcomes are not supported by this two-group contrast."
                ),
                problem="the outcome is neither numeric nor two-valued.",
            )
        )
    if covariate_columns is not None:
        if len(covariate_columns) != len(set(covariate_columns)):
            violations.append(
                GuardViolation(
                    field="covariate_columns",
                    got=covariate_columns,
                    allowed="distinct column names",
                    fix_hint="Remove the duplicated covariate names.",
                    problem="covariate_columns contains duplicates.",
                )
            )
        for column in covariate_columns:
            exists = check_column_exists(
                "covariate_columns", column, [str(name) for name in frame.columns]
            )
            if exists is not None:
                violations.append(exists)
                continue
            if column in {treatment_column, outcome_column}:
                violations.append(
                    GuardViolation(
                        field="covariate_columns",
                        got=column,
                        allowed="columns other than the treatment and the outcome",
                        fix_hint="Drop it; the treatment and outcome are not covariates.",
                        problem="covariate repeats the treatment or outcome column.",
                    )
                )
                continue
            if infer_column_semantic_type(cast(pd.Series, frame[column])) == "datetime":
                violations.append(
                    GuardViolation(
                        field="covariate_columns",
                        got=column,
                        allowed="numeric or categorical covariates",
                        fix_hint="Drop the datetime column; balance is not defined on it.",
                        problem="covariate is a datetime column.",
                    )
                )
    if treatment_ok and not violations:
        working = frame[frame[treatment_column].notna()]
        counts = working[treatment_column].astype(str).value_counts()
        if int(counts.min()) < _MIN_GROUP_ROWS:
            violations.append(
                GuardViolation(
                    field="treatment_column",
                    got=f"smallest group has {int(counts.min())} row(s)",
                    allowed=f"at least {_MIN_GROUP_ROWS} rows per group",
                    fix_hint="Both groups need enough rows for a variance estimate.",
                    problem="one treatment group is too small.",
                )
            )
    raise_for_violations("run_causal_experiment", violations)


def _resolve_outcome_kind_or_none(
    series: pd.Series,
) -> Literal["numeric", "binary"] | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    semantic = infer_column_semantic_type(series)
    if semantic == "numeric":
        return "numeric"
    if semantic == "categorical" and int(non_null.astype(str).nunique()) == 2:
        return "binary"
    return None


def _resolve_outcome_kind(series: pd.Series) -> Literal["numeric", "binary"]:
    kind = _resolve_outcome_kind_or_none(series)
    assert kind is not None  # the guard rejected the other shapes
    return kind


def _resolve_covariates(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    covariate_columns: list[str] | None,
) -> list[str]:
    if covariate_columns is not None:
        return list(covariate_columns)
    selected: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in {treatment_column, outcome_column}:
            continue
        series = cast(pd.Series, frame[name])
        semantic = infer_column_semantic_type(series)
        if semantic == "datetime":
            continue
        if int(series.dropna().nunique()) <= 1:
            continue
        if semantic == "numeric":
            if _is_id_like(series, name):
                continue
        elif int(series.dropna().nunique()) > _MAX_CATEGORICAL_COVARIATE_CARDINALITY:
            continue
        selected.append(name)
        if len(selected) >= _MAX_AUTO_COVARIATES:
            break
    return selected


def _covariate_balance(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    covariates: list[str],
    control_label: str,
    treated_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    working = frame[frame[treatment_column].notna()]
    groups = working[treatment_column].astype(str)
    control_mask = groups == control_label
    treated_mask = groups == treated_label
    rows: list[dict[str, Any]] = []
    imbalanced: list[str] = []
    for covariate in covariates:
        series = cast(pd.Series, working[covariate])
        kind = (
            "numeric"
            if infer_column_semantic_type(series) == "numeric"
            else "categorical"
        )
        if kind == "numeric":
            smd, row_imbalanced, note = _numeric_smd(
                pd.to_numeric(series[control_mask], errors="coerce").dropna(),
                pd.to_numeric(series[treated_mask], errors="coerce").dropna(),
            )
        else:
            smd, row_imbalanced, note = _categorical_share_difference(
                series[control_mask].dropna().astype(str),
                series[treated_mask].dropna().astype(str),
            )
        rows.append(
            {
                "covariate": covariate,
                "kind": kind,
                "smd": smd,
                "imbalanced": row_imbalanced,
                "note": note,
            }
        )
        if row_imbalanced:
            imbalanced.append(covariate)
    return rows, imbalanced


def _numeric_smd(
    control: pd.Series, treated: pd.Series
) -> tuple[float | None, bool, str | None]:
    if len(control) < _MIN_GROUP_ROWS or len(treated) < _MIN_GROUP_ROWS:
        return None, False, "too few non-null values per group to measure balance"
    mean_diff = float(treated.mean() - control.mean())
    pooled = float(np.sqrt((float(treated.var(ddof=1)) + float(control.var(ddof=1))) / 2.0))
    if pooled == 0.0:
        if mean_diff == 0.0:
            return 0.0, False, None
        return None, True, "both groups are constant at different values"
    smd = round(mean_diff / pooled, 6)
    return smd, abs(smd) > BALANCE_SMD_THRESHOLD, None


def _categorical_share_difference(
    control: pd.Series, treated: pd.Series
) -> tuple[float | None, bool, str | None]:
    if control.empty or treated.empty:
        return None, False, "too few non-null values per group to measure balance"
    overall = pd.concat([control, treated]).value_counts()
    levels = [str(level) for level in overall.index[:_MAX_CATEGORICAL_LEVELS]]
    control_share = control.value_counts(normalize=True)
    treated_share = treated.value_counts(normalize=True)
    difference = max(
        abs(float(treated_share.get(level, 0.0)) - float(control_share.get(level, 0.0)))
        for level in levels
    )
    value = round(difference, 6)
    note = None
    if len(overall) > _MAX_CATEGORICAL_LEVELS:
        note = f"only the {_MAX_CATEGORICAL_LEVELS} most frequent levels were compared"
    return value, value > BALANCE_SMD_THRESHOLD, note


def _randomized_ate(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    treatment_column: str,
    outcome_column: str,
    outcome_kind: Literal["numeric", "binary"],
    control_label: str,
    treated_label: str,
    comparison_count: int,
) -> tuple[StatTestResult, str]:
    working = cast(pd.DataFrame, frame[[treatment_column, outcome_column]]).dropna()
    groups = working[treatment_column].astype(str)
    if outcome_kind == "binary":
        outcome_values = working[outcome_column].astype(str)
        positive = sorted(outcome_values.unique())[-1]
        y = (outcome_values == positive).astype(float)
        ate_definition = (
            f"share of {outcome_column}={positive} in {treatment_column}="
            f"{treated_label} minus {treatment_column}={control_label} (risk difference)"
        )
    else:
        y = pd.to_numeric(working[outcome_column], errors="coerce")
        ate_definition = (
            f"mean {outcome_column} in {treatment_column}={treated_label} "
            f"minus {treatment_column}={control_label}"
        )
    treated = y[groups == treated_label].dropna().to_numpy(dtype=float)
    control = y[groups == control_label].dropna().to_numpy(dtype=float)
    if len(treated) < _MIN_GROUP_ROWS or len(control) < _MIN_GROUP_ROWS:
        raise ValueError(
            "two_sample_ate needs at least "
            f"{_MIN_GROUP_ROWS} non-null outcome rows in each treatment group."
        )

    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    compare = CompareMeans(DescrStatsW(treated), DescrStatsW(control))
    statistic, p_value, dof = compare.ttest_ind(usevar="unequal")
    ci_low, ci_high = compare.tconfint_diff(alpha=0.05, usevar="unequal")
    ate = float(treated.mean() - control.mean())
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        raise ValueError(
            "two_sample_ate is not computable on this data (non-finite result); "
            "likely a constant outcome in one group."
        )
    warnings: list[StatWarning] = []
    if outcome_kind == "binary":
        warnings.append(
            StatWarning(
                code="binary_outcome_risk_difference",
                severity="info",
                message=(
                    "The outcome is binary, so the ATE is a risk difference and "
                    "the Welch interval is the unpooled Wald interval."
                ),
            )
        )
    result = StatTestResult(
        dataset_id=dataset_id,
        test_type="two_sample_ate",
        group_column=treatment_column,
        value_column=outcome_column,
        statistic=round(float(statistic), 6),
        p_value=min(1.0, max(0.0, float(p_value))),
        effect_size=round(ate, 6),
        effect_ci_low=round(float(ci_low), 6),
        effect_ci_high=round(float(ci_high), 6),
        degrees_of_freedom=int(dof),
        sample_size=int(len(treated) + len(control)),
        groups={control_label: int(len(control)), treated_label: int(len(treated))},
        assumptions=[
            StatAssumptionCheck(
                name="randomized_assignment",
                status="passed",
                message=(
                    "Randomization was confirmed by the user, not inferred "
                    "from the data."
                ),
            ),
            StatAssumptionCheck(
                name="variance_model",
                status="passed",
                message="Welch/Satterthwaite: no equal-variance assumption.",
            ),
        ],
        warnings=warnings,
    )
    if comparison_count > 1 and result.p_value is not None:
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
    return result, ate_definition


def _balance_table(
    result: CausalExperimentResult,
    *,
    dataset_id: str,
    dataset_name: str,
) -> AnalysisTable:
    tier = (
        "randomized experiment (user-confirmed design)"
        if result.design == "randomized"
        else "observational design check"
    )
    return AnalysisTable(
        dataset_id=dataset_id,
        title=f"{dataset_name} - Covariate balance ({result.treatment_column})",
        kind="numeric_summary",
        description=(
            f"Standardized mean difference (numeric) or largest absolute share "
            f"difference (categorical) between {result.treatment_column}="
            f"{result.treated_label} and {result.treatment_column}="
            f"{result.control_label}; |SMD| > {BALANCE_SMD_THRESHOLD} is flagged "
            f"as imbalanced. Computed for the {tier}. {result.disclaimer}"
        ),
        rows=list(result.balance_rows),
    )

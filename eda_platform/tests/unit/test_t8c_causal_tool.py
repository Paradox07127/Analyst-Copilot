"""run_causal_experiment two-tier executor (T8c).

Tier A numbers (Welch t, SMD balance) are recomputed independently in the
tests; tier B's ATE and Welch CI are checked against statsmodels reference
values and a hand-computed mean difference. The credential chain is exercised
on both sides (credential present -> tier B; absent -> explicit downgrade),
and the claim-gate causal whitelist is probed with forged parameters.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from eda_platform.agents.data_tools import (
    DataToolContext,
    RunCausalExperimentArguments,
    build_data_tools,
)
from eda_platform.agents.receipts import build_receipt
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.core.claim_gates import run_claim_gates
from eda_platform.core.claim_language import (
    asserts_model_capability,
    implies_causation,
)
from eda_platform.core.methods import METHOD_REGISTRY
from eda_platform.core.store import ArtifactStore
from eda_platform.core.tool_guard import ToolGuardError
from eda_platform.drivers.question_exec import _method_contract_failure
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.claims import Claim, ClaimBundle
from eda_platform.schemas.datasets import DatasetRecord
from eda_platform.schemas.questions import QuestionCandidate, QuestionScore
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptFact,
    ReceiptMethod,
    ReceiptScope,
    verify_receipt_digest,
)
from eda_platform.schemas.stats import StatTestResult
from eda_platform.tools.causal_experiment import (
    OBSERVATIONAL_CONTRAST_DISCLAIMER,
    RANDOMIZED_DESIGN_APPROVAL_KIND,
    RANDOMIZED_DESIGN_ASSUMPTION,
    RANDOMIZED_DOWNGRADE_WARNING,
    RANDOMIZED_EXPERIMENT_FAMILY,
    randomized_design_action,
    randomized_design_credential_id,
)
from eda_platform.tools.loader import LoadedDataset
from eda_platform.tools.sql_runner import build_catalog

SESSION_ID = "run_t8c"
SOURCE_SESSION_ID = "run_t8c_source"


def _dataset(frame: pd.DataFrame) -> LoadedDataset:
    return LoadedDataset(
        record=DatasetRecord(
            dataset_id="ds_exp",
            name="experiment.csv",
            path=Path("/data/experiment.csv"),
            content_hash="hash_ds_exp",
        ),
        frame=frame,
    )


def _context(
    frame: pd.DataFrame,
    *,
    store: ArtifactStore | None = None,
    session_id: str = SESSION_ID,
    source_session_id: str | None = None,
) -> DataToolContext:
    datasets = [_dataset(frame)]
    return DataToolContext(
        datasets=datasets,
        catalog=build_catalog(datasets),
        project_id="project_t8c",
        session_id=session_id,
        store=store,
        payload_policy="schema+aggregates",
        artifacts=[],
        source_session_id=source_session_id,
    )


def _tool(context: DataToolContext) -> Any:
    return next(
        tool
        for tool in build_data_tools(context)
        if tool.name == "run_causal_experiment"
    )


def _receipt(context: DataToolContext) -> EvidenceReceipt:
    artifact = [
        a for a in context.artifacts if a.type is ArtifactType.EVIDENCE_RECEIPT
    ][-1]
    return EvidenceReceipt.model_validate(artifact.payload)


def _fact(receipt: EvidenceReceipt, fact_id: str) -> Any:
    return next(fact for fact in receipt.facts if fact.fact_id == fact_id).value


def _stat_payload(context: DataToolContext) -> StatTestResult:
    artifact = [
        a for a in context.artifacts if a.type is ArtifactType.STAT_TEST_RESULT
    ][-1]
    return StatTestResult.model_validate(artifact.payload)


def _observational_frame() -> pd.DataFrame:
    """Hand-checkable geometry: 20 rows per group, exact means and variances."""
    return pd.DataFrame(
        {
            "plan": ["basic"] * 20 + ["promo"] * 20,
            "spend": [10.0, 12.0, 14.0, 16.0, 18.0] * 4
            + [20.0, 22.0, 24.0, 26.0, 28.0] * 4,
            "age": [30.0] * 10 + [40.0] * 10 + [35.0] * 10 + [45.0] * 10,
            "channel": ["web"] * 12 + ["phone"] * 8 + ["web"] * 4 + ["phone"] * 16,
        }
    )


def _randomized_frame(effect: float = 2.5, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    treat = rng.integers(0, 2, n)
    outcome = 5.0 + effect * treat + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "arm": np.where(treat == 1, "treated", "control"),
            "spend": outcome,
            "age": rng.normal(40.0, 10.0, n),
        }
    )


def _store(tmp_path: Path, *, session_ids: tuple[str, ...] = (SESSION_ID,)) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("project_t8c", name="T8c")
    for session_id in session_ids:
        store.start_session("project_t8c", session_id)
    return store


def _issue_credential(
    store: ArtifactStore,
    *,
    session_id: str = SESSION_ID,
    dataset_id: str = "ds_exp",
    treatment_column: str = "arm",
    ttl_seconds: int = 1800,
) -> str:
    digest, _generation, _expires = ApprovalService(
        store, ttl_seconds=ttl_seconds
    ).register(
        kind=RANDOMIZED_DESIGN_APPROVAL_KIND,
        session_id=session_id,
        project_id="project_t8c",
        action=randomized_design_action(
            dataset_id=dataset_id, treatment_column=treatment_column
        ),
        payload={
            "project_id": "project_t8c",
            "session_id": session_id,
            "dataset_id": dataset_id,
            "treatment_column": treatment_column,
        },
    )
    return digest


def test_observational_contrast_is_hand_recomputable() -> None:
    context = _context(_observational_frame())
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="plan",
            outcome_column="spend",
            covariate_columns=["age", "channel"],
        )
    )
    content = cast(dict[str, Any], result.content)
    stat = _stat_payload(context)

    # Independent Welch t on the exact group values.
    basic = np.array([10.0, 12.0, 14.0, 16.0, 18.0] * 4)
    promo = np.array([20.0, 22.0, 24.0, 26.0, 28.0] * 4)
    se = math.sqrt(basic.var(ddof=1) / 20 + promo.var(ddof=1) / 20)
    expected_t = (basic.mean() - promo.mean()) / se
    assert content["design"] == "observational"
    assert stat.test_type == "independent_t_test"
    assert stat.statistic == pytest.approx(expected_t, abs=1e-6)
    pooled = math.sqrt((basic.var(ddof=1) + promo.var(ddof=1)) / 2)
    assert stat.effect_size == pytest.approx(
        (basic.mean() - promo.mean()) / pooled, abs=1e-6
    )
    assert stat.sample_size == 40

    # SMD balance recomputed from the same formula, by hand.
    ages_basic = np.array([30.0] * 10 + [40.0] * 10)
    ages_promo = np.array([35.0] * 10 + [45.0] * 10)
    expected_smd = (ages_promo.mean() - ages_basic.mean()) / math.sqrt(
        (ages_basic.var(ddof=1) + ages_promo.var(ddof=1)) / 2
    )
    balance = {row["covariate"]: row for row in content["balance"]}
    assert balance["age"]["smd"] == pytest.approx(expected_smd, abs=1e-6)
    assert balance["age"]["imbalanced"] is True
    # channel: web share 12/20 vs 4/20 -> difference 0.4 for both levels.
    assert balance["channel"]["smd"] == pytest.approx(0.4, abs=1e-9)
    assert balance["channel"]["imbalanced"] is True
    assert content["imbalanced_covariates"] == ["age", "channel"]

    # The fixed disclaimer is in the durable payload AND on the receipt.
    assert content["disclaimer"] == OBSERVATIONAL_CONTRAST_DISCLAIMER
    assert OBSERVATIONAL_CONTRAST_DISCLAIMER in content["warnings"]
    assert any(
        w.code == "observational_design" and w.message == OBSERVATIONAL_CONTRAST_DISCLAIMER
        for w in stat.warnings
    )
    receipt = _receipt(context)
    assert verify_receipt_digest(receipt)
    assert receipt.method.family == "causal_design_check"
    assert receipt.method.assumptions == (OBSERVATIONAL_CONTRAST_DISCLAIMER,)
    assert receipt.method.parameters["experiment_design"] == "observational"
    assert "design_confirmation_id" not in receipt.method.parameters
    assert _fact(receipt, "design") == "observational"
    assert _fact(receipt, "imbalanced_covariate_count") == 2
    assert _fact(receipt, "covariate0.name") == "age"
    assert _fact(receipt, "covariate0.smd") == balance["age"]["smd"]


def test_binary_outcome_runs_chi_square_in_tier_a() -> None:
    frame = pd.DataFrame(
        {
            "plan": ["basic"] * 60 + ["promo"] * 60,
            "converted": ["yes"] * 30 + ["no"] * 30 + ["yes"] * 45 + ["no"] * 15,
        }
    )
    context = _context(frame)
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="plan",
            outcome_column="converted",
            covariate_columns=[],
        )
    )
    content = cast(dict[str, Any], result.content)
    stat = _stat_payload(context)
    from scipy import stats as scipy_stats

    table = pd.crosstab(frame["plan"], frame["converted"])
    expected_chi2, expected_p, _dof, _expected = scipy_stats.chi2_contingency(table)
    assert content["outcome_kind"] == "binary"
    assert stat.test_type == "chi_square_independence"
    assert stat.statistic == pytest.approx(float(expected_chi2), abs=1e-6)
    assert stat.p_value == pytest.approx(float(expected_p), rel=1e-9)


def test_repeat_contrast_tightens_its_own_family() -> None:
    context = _context(_observational_frame())
    tool = _tool(context)
    arguments = RunCausalExperimentArguments(
        dataset_id="ds_exp",
        treatment_column="plan",
        outcome_column="spend",
        covariate_columns=["age"],
    )
    tool.execute(arguments)
    first = _receipt(context)
    tool.execute(arguments)
    second = _receipt(context)

    assert first.statistics is not None and second.statistics is not None
    assert first.statistics.sequence_index == 1
    assert second.statistics.sequence_index == 2
    assert first.statistics.statistical_family_id == second.statistics.statistical_family_id
    assert first.statistics.adjusted_p_value is None
    assert second.statistics.p_value is not None
    assert second.statistics.adjusted_p_value == pytest.approx(
        min(1.0, second.statistics.p_value * 2), rel=1e-12
    )
    registry = context.stat_registry
    assert registry is not None
    assert len(registry.attempts()) == 2
    assert all(attempt.status == "completed" for attempt in registry.attempts())


def test_randomized_without_credential_downgrades_and_says_so(tmp_path: Path) -> None:
    store = _store(tmp_path)
    context = _context(_randomized_frame(), store=store)
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    content = cast(dict[str, Any], result.content)
    assert content["design"] == "observational"
    assert content["requested_design"] == "randomized"
    assert content["test_type"] == "independent_t_test"
    assert RANDOMIZED_DOWNGRADE_WARNING in content["warnings"]
    receipt = _receipt(context)
    assert receipt.method.family == "causal_design_check"
    assert receipt.method.parameters["requested_design"] == "randomized"
    assert receipt.method.parameters["experiment_design"] == "observational"
    assert "design_confirmation_id" not in receipt.method.parameters
    assert RANDOMIZED_DOWNGRADE_WARNING in receipt.method.warnings


def test_expired_credential_does_not_admit_tier_b(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _issue_credential(store, ttl_seconds=0)
    context = _context(_randomized_frame(), store=store)
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    assert cast(dict[str, Any], result.content)["design"] == "observational"


def test_randomized_tier_matches_statsmodels_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    credential_id = _issue_credential(store)
    frame = _randomized_frame(effect=2.5)
    context = _context(frame, store=store)
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    content = cast(dict[str, Any], result.content)
    stat = _stat_payload(context)
    assert content["design"] == "randomized"
    assert stat.test_type == "two_sample_ate"

    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    treated = frame.loc[frame["arm"] == "treated", "spend"].to_numpy()
    control = frame.loc[frame["arm"] == "control", "spend"].to_numpy()
    compare = CompareMeans(DescrStatsW(treated), DescrStatsW(control))
    ref_low, ref_high = compare.tconfint_diff(alpha=0.05, usevar="unequal")
    ref_t, ref_p, _df = compare.ttest_ind(usevar="unequal")
    assert stat.effect_size == pytest.approx(
        float(treated.mean() - control.mean()), abs=1e-6
    )
    assert stat.effect_ci_low == pytest.approx(float(ref_low), abs=1e-6)
    assert stat.effect_ci_high == pytest.approx(float(ref_high), abs=1e-6)
    assert stat.statistic == pytest.approx(float(ref_t), abs=1e-6)
    assert stat.p_value == pytest.approx(float(ref_p), rel=1e-6, abs=1e-300)
    # The CI covers the true simulated effect.
    assert stat.effect_ci_low is not None and stat.effect_ci_high is not None
    assert stat.effect_ci_low <= 2.5 <= stat.effect_ci_high
    assert stat.sample_size == 400

    receipt = _receipt(context)
    assert receipt.method.family == RANDOMIZED_EXPERIMENT_FAMILY
    assert receipt.method.parameters["experiment_design"] == "randomized"
    assert receipt.method.parameters["design_confirmation_id"] == credential_id
    assert receipt.method.parameters["ci_method"] == "welch_tconfint_unequal"
    statistics = receipt.statistics
    assert statistics is not None
    # The full five-piece statistics block the statistical gate requires.
    assert statistics.p_value is not None
    assert statistics.effect_size is not None
    assert statistics.ci_low is not None and statistics.ci_high is not None
    assert statistics.sample_size == 400
    assert statistics.sequence_index == 1
    assert content["disclaimer"] == RANDOMIZED_DESIGN_ASSUMPTION


def test_credential_on_the_source_session_admits_a_derived_run(tmp_path: Path) -> None:
    store = _store(tmp_path, session_ids=(SOURCE_SESSION_ID, "qsess_derived"))
    _issue_credential(store, session_id=SOURCE_SESSION_ID)
    context = _context(
        _randomized_frame(),
        store=store,
        session_id="qsess_derived",
        source_session_id=SOURCE_SESSION_ID,
    )
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    assert cast(dict[str, Any], result.content)["design"] == "randomized"


def test_confirm_randomized_design_service_credential_admits_tier_b(
    tmp_path: Path,
) -> None:
    """The user-facing service issues the same credential the tool verifies."""
    from eda_platform.application.services.question_service import (
        QuestionService,
        QuestionValidationError,
    )
    from eda_platform.tools.profiler import profile_dataset

    store = _store(tmp_path)
    frame = _randomized_frame()
    profile_artifact = profile_dataset(
        _dataset(frame), project_id="project_t8c", session_id=SESSION_ID
    )
    store.save_artifact(profile_artifact)
    service = QuestionService(store, ApprovalService(store), cast(Any, None))

    confirmed = service.confirm_randomized_design(
        SESSION_ID, dataset_id="ds_exp", treatment_column="arm"
    )
    assert confirmed.credential_id == randomized_design_credential_id(
        dataset_id="ds_exp", treatment_column="arm"
    )
    with pytest.raises(QuestionValidationError, match="does not exist"):
        service.confirm_randomized_design(
            SESSION_ID, dataset_id="ds_exp", treatment_column="not_a_column"
        )
    with pytest.raises(QuestionValidationError, match="no profile"):
        service.confirm_randomized_design(
            SESSION_ID, dataset_id="ds_missing", treatment_column="arm"
        )

    context = _context(frame, store=store)
    context.add_artifact(profile_artifact, persist=False)
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    content = cast(dict[str, Any], result.content)
    assert content["design"] == "randomized"
    receipt = _receipt(context)
    assert (
        receipt.method.parameters["design_confirmation_id"] == confirmed.credential_id
    )


def _causal_claim(receipt: EvidenceReceipt, text: str, **overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_id": "c_causal",
        "claim_type": "causal",
        "claim_text": text,
        "support_type": "direct",
        "evidence_fact_ids": (f"{receipt.receipt_id}:effect_size",),
        "statistics_receipt_ids": (receipt.receipt_id,),
    }
    fields.update(overrides)
    return Claim.model_validate(fields)


def _gate_report(
    claim: Claim,
    receipt: EvidenceReceipt,
    *,
    lane: str = "confirmatory",
) -> Any:
    statistics = receipt.statistics
    assert statistics is not None
    family_id = statistics.statistical_family_id
    assert family_id is not None
    bundle = ClaimBundle.model_validate(
        {
            "claim_bundle_id": "clb_t8c",
            "hypothesis_id": "hyp_t8c",
            "evidence_lane": lane,
            "claims": (claim,),
        }
    )
    return run_claim_gates(
        bundle,
        committed_receipts={receipt.receipt_id: receipt},
        run_witness=receipt.data_state_witness,
        stat_attempt_counts={family_id: statistics.sequence_index or 1},
    )


def _codes(report: Any, gate: str) -> set[str]:
    for verdict in report.verdicts:
        if verdict.gate == gate:
            return {violation.code for violation in verdict.violations}
    return set()


def test_claim_gate_licenses_causal_wording_only_for_the_credentialed_receipt(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _issue_credential(store)
    context = _context(_randomized_frame(), store=store)
    _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="arm",
            outcome_column="spend",
            design="randomized",
        )
    )
    receipt = _receipt(context)
    effect = _fact(receipt, "effect_size")
    text = (
        "In the user-confirmed randomized experiment, the treatment caused "
        f"an average spend increase of {effect}."
    )
    assert implies_causation(text)  # the wording is genuinely causal

    report = _gate_report(_causal_claim(receipt, text), receipt)
    assert report.passed, [
        (verdict.gate, verdict.violations)
        for verdict in report.verdicts
        if not verdict.passed
    ]

    # Causal wording without claim_type=causal is licensed by the same receipt.
    observational_wording = _causal_claim(
        receipt, text, claim_type="observation", claim_id="c_obs"
    )
    assert _codes(_gate_report(observational_wording, receipt), "entity") == set()


def test_claim_gate_still_rejects_tier_a_receipts_with_causal_text() -> None:
    context = _context(_observational_frame())
    _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="plan",
            outcome_column="spend",
            covariate_columns=["age"],
        )
    )
    receipt = _receipt(context)
    effect = _fact(receipt, "effect_size")
    text = f"The promo plan caused a spend difference of {effect}."

    typed = _gate_report(_causal_claim(receipt, text), receipt, lane="exploratory")
    assert "causal_claim_rejected" in _codes(typed, "entity")
    worded = _gate_report(
        _causal_claim(receipt, text, claim_type="observation"),
        receipt,
        lane="exploratory",
    )
    assert "causal_language" in _codes(worded, "entity")


def test_claim_gate_rejects_forged_randomized_parameters() -> None:
    """A receipt that claims the randomized tier but whose credential id was
    minted for a different treatment binding must not license causal wording."""
    forged = build_receipt(
        tool_call_id="call_forged",
        tool_name="run_causal_experiment",
        tool_version="1",
        arguments={"dataset_id": "ds_exp"},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(
            dataset_ids=("ds_exp",), columns=("arm", "spend"), scope_resolution="explicit"
        ),
        facts=(
            ReceiptFact(
                fact_id="effect_size", name="effect_size", value=2.6, value_type="number"
            ),
        ),
        method=ReceiptMethod(
            family=RANDOMIZED_EXPERIMENT_FAMILY,
            parameters={
                "experiment_design": "randomized",
                "treatment_column": "arm",
                # Credential minted for another column: the gate's re-derivation
                # from (dataset, treatment) must refuse it.
                "design_confirmation_id": randomized_design_credential_id(
                    dataset_id="ds_exp", treatment_column="other_column"
                ),
            },
        ),
        data_state_witness="dsw1_" + "a" * 64,
        created_at="2026-08-26T00:00:00Z",
    )
    claim = Claim.model_validate(
        {
            "claim_id": "c_forged",
            "claim_type": "causal",
            "claim_text": "The treatment caused a 2.6 increase.",
            "support_type": "direct",
            "evidence_fact_ids": (f"{forged.receipt_id}:effect_size",),
        }
    )
    bundle = ClaimBundle.model_validate(
        {
            "claim_bundle_id": "clb_forged",
            "hypothesis_id": "hyp_forged",
            "evidence_lane": "exploratory",
            "claims": (claim,),
        }
    )
    report = run_claim_gates(
        bundle,
        committed_receipts={forged.receipt_id: forged},
        run_witness=forged.data_state_witness,
        stat_attempt_counts={},
    )
    assert "causal_claim_rejected" in _codes(report, "entity")

    # Same forgery with a missing credential id is rejected as well.
    parameters = dict(forged.method.parameters)
    parameters.pop("design_confirmation_id")
    no_credential = build_receipt(
        tool_call_id="call_forged_2",
        tool_name="run_causal_experiment",
        tool_version="1",
        arguments={"dataset_id": "ds_exp"},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=forged.scope,
        facts=forged.facts,
        method=ReceiptMethod(family=RANDOMIZED_EXPERIMENT_FAMILY, parameters=parameters),
        data_state_witness=forged.data_state_witness,
        created_at="2026-08-26T00:00:00Z",
    )
    claim2 = claim.model_copy(
        update={"evidence_fact_ids": (f"{no_credential.receipt_id}:effect_size",)}
    )
    report2 = run_claim_gates(
        bundle.model_copy(update={"claims": (claim2,)}),
        committed_receipts={no_credential.receipt_id: no_credential},
        run_witness=no_credential.data_state_witness,
        stat_attempt_counts={},
    )
    assert "causal_claim_rejected" in _codes(report2, "entity")


def test_guard_rejects_bad_shapes() -> None:
    frame = pd.DataFrame(
        {
            "plan": ["a", "b", "c"] * 10,
            "flag": ["x", "y"] * 15,
            "spend": [float(v) for v in range(30)],
            "color": ["red", "green", "blue"] * 10,
        }
    )
    context = _context(frame)
    tool = _tool(context)

    with pytest.raises(ToolGuardError, match="two groups"):
        tool.execute(
            RunCausalExperimentArguments(
                dataset_id="ds_exp", treatment_column="plan", outcome_column="spend"
            )
        )
    with pytest.raises(ToolGuardError, match="semantic type"):
        tool.execute(
            RunCausalExperimentArguments(
                dataset_id="ds_exp", treatment_column="spend", outcome_column="flag"
            )
        )
    with pytest.raises(ToolGuardError, match="neither numeric nor two-valued"):
        tool.execute(
            RunCausalExperimentArguments(
                dataset_id="ds_exp", treatment_column="flag", outcome_column="color"
            )
        )
    with pytest.raises(ToolGuardError, match="not covariates"):
        tool.execute(
            RunCausalExperimentArguments(
                dataset_id="ds_exp",
                treatment_column="flag",
                outcome_column="spend",
                covariate_columns=["flag"],
            )
        )
    with pytest.raises(ValidationError):
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="flag",
            outcome_column="spend",
            covariate_columns=["age", "age"],
        )
    with pytest.raises(ValidationError):
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="flag",
            outcome_column="spend",
            design="quasi",  # type: ignore[arg-type]
        )
    assert not [
        a for a in context.artifacts if a.type is ArtifactType.STAT_TEST_RESULT
    ]


def _causal_candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_causal",
        question_en="Does the promo plan raise spend?",
        origin="llm",
        analysis_mode="causal_experiment",
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.1,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
    )


def test_causal_answer_contract_is_satisfied_by_the_tool_output() -> None:
    context = _context(_observational_frame())
    result = _tool(context).execute(
        RunCausalExperimentArguments(
            dataset_id="ds_exp",
            treatment_column="plan",
            outcome_column="spend",
            covariate_columns=["age"],
        )
    )
    failure = _method_contract_failure(
        _causal_candidate(),
        evidence_artifacts=cast(list[Any], result.artifacts),
        tool_names=["run_causal_experiment"],
    )
    assert failure is None

    # A generic table alone still cannot answer a causal question.
    balance_only = [a for a in result.artifacts if a.type is ArtifactType.TABLE]
    failure = _method_contract_failure(
        _causal_candidate(),
        evidence_artifacts=cast(list[Any], balance_only),
        tool_names=["run_sql"],
    )
    assert failure is not None and failure.code == "method_contract_failed"


def test_language_stays_honest_and_family_is_supported() -> None:
    context = _context(_observational_frame())
    tool_description = _tool(context).description
    for text in (
        OBSERVATIONAL_CONTRAST_DISCLAIMER,
        RANDOMIZED_DESIGN_ASSUMPTION,
        RANDOMIZED_DOWNGRADE_WARNING,
        tool_description,
    ):
        assert not implies_causation(text)
        assert not asserts_model_capability(text)
    assert METHOD_REGISTRY["causal_experiment"].supported is True

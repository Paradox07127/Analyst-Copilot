"""DI sprint-3 method-family execution (DI3-B).

These cover the three real method executors added to the investigation
orchestrator: group comparison (stat tests), outcome prediction (baseline model +
leakage guard), and anomaly detection. The DI3-A reducers
(``tools.method_findings``) and anomaly tool (``tools.anomaly``) are built in
parallel; assertions that need them ``importorskip`` so the suite is green before
merge and exercises the real path afterward. The execution-level guarantees that
depend only on the read-only tools already on disk (stat tests, ml_baseline) and
the sprint-2 trust boundaries are asserted unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.investigations import (
    InvestigationPlan,
    InvestigationRecord,
    ValidatedFinding,
)
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionScore,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _prepare_source(
    tmp_path: Path,
    filename: str,
    frame: pd.DataFrame,
    *,
    analysis_mode: str,
    data_requirements: list[str] | None = None,
    referenced_columns: dict[str, list[str]] | None = None,
) -> tuple[AutoEDAResult, QuestionCandidate]:
    """Build a real source run from a crafted CSV, then inject one candidate.

    ``run_auto_eda`` produces the real DatasetProfile + upload the orchestrator
    reloads; the crafted candidate carries a valid deterministic score and is
    pointed at ``analysis_mode`` so the plan routes to the target method family.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / filename
    frame.to_csv(csv, index=False)
    source = run_auto_eda(
        [csv],
        workspace=tmp_path / "ws",
        project_id="project_demo",
        session_id="source_run",
    )
    store = ArtifactStore(source.workspace)
    cs_artifact = next(
        store.get_artifact(item.id)
        for item in source.artifacts
        if item.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate_set = QuestionCandidateSet.model_validate(cs_artifact.payload)
    candidate = QuestionCandidate(
        question_id="q_di3_method",
        question_en="How does the measure differ across the approved data?",
        origin="template",
        target_datasets=[filename],
        analysis_mode=analysis_mode,  # type: ignore[arg-type]
        data_requirements=data_requirements or [],
        referenced_columns=referenced_columns or {},
        score=QuestionScore(
            data_availability=0.9,
            statistical_signal=0.7,
            quality_risk=0.2,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
    )
    candidate_set = candidate_set.model_copy(update={"candidates": [candidate]})
    cs_artifact.payload = candidate_set.model_dump(mode="json")
    store.save_artifact(cs_artifact)
    return source, candidate


def _plan(source: AutoEDAResult, question_id: str) -> tuple[Artifact, InvestigationPlan]:
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = next(
        item for item in planned.artifacts if item.type is ArtifactType.INVESTIGATION_PLAN
    )
    return plan_artifact, InvestigationPlan.model_validate(plan_artifact.payload)


def _approve_and_execute(source: AutoEDAResult, plan_artifact: Artifact) -> list[Artifact]:
    approve_plan(
        project_id=source.project_id,
        plan_session_id="plan_run",
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id="plan_run",
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    return completed.artifacts


def _record(artifacts: list[Artifact]) -> InvestigationRecord:
    return next(
        InvestigationRecord.model_validate(item.payload)
        for item in artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
    )


def _finding(artifacts: list[Artifact]) -> ValidatedFinding | None:
    for item in artifacts:
        if item.type is ArtifactType.VALIDATED_FINDING:
            return ValidatedFinding.model_validate(item.payload)
    return None


def _group_frame() -> pd.DataFrame:
    # Three groups (>= 5 expected rows each) with deliberately unequal variance so
    # the Levene homogeneity check warns -> the finding must carry the warning.
    north = [10, 10, 11, 9, 10, 11, 9, 10, 10, 11]
    south = [1, 50, 100, 2, 80, 5, 95, 3, 70, 60]
    east = [30, 32, 28, 31, 29, 33, 27, 30, 31, 29]
    rows = (
        [("north", v) for v in north]
        + [("south", v) for v in south]
        + [("east", v) for v in east]
    )
    return pd.DataFrame(rows, columns=["region", "revenue"])


def _prediction_frame(*, leakage: bool) -> pd.DataFrame:
    if leakage:
        # Only an id-like column and a perfect proxy of the target -> every feature
        # is excluded by the leakage guard and no honest baseline can be built.
        churn = [i % 2 for i in range(60)]
        return pd.DataFrame(
            {
                "customer_id": [f"CU{i:04d}" for i in range(60)],
                "churn_copy": churn,
                "churn": churn,
            }
        )
    # Low-cardinality features whose values are independent of a randomly drawn
    # target, so no column is an identifier or a proxy of the outcome.
    import numpy as np

    rng = np.random.default_rng(0)
    churn = rng.integers(0, 2, size=60).tolist()
    age = [(20, 30, 40, 50)[i % 4] for i in range(60)]
    tenure = [(1, 2, 3)[i % 3] for i in range(60)]
    return pd.DataFrame({"age": age, "tenure": tenure, "churn": churn})


def _anomaly_frame(*, constant: bool) -> pd.DataFrame:
    if constant:
        return pd.DataFrame(
            {"label": [f"r{i}" for i in range(40)], "value": [7.0] * 40}
        )
    values = [10.0 + (i % 5) for i in range(38)] + [500.0, -400.0]
    return pd.DataFrame({"label": [f"r{i}" for i in range(40)], "value": values})


# --------------------------------------------------------------------------- #
# Group comparison
# --------------------------------------------------------------------------- #
def test_group_comparison_routes_and_produces_stat_evidence(tmp_path: Path) -> None:
    source, candidate = _prepare_source(
        tmp_path,
        "di3_groups.csv",
        _group_frame(),
        analysis_mode="diagnostic",
        referenced_columns={"di3_groups.csv": ["region", "revenue"]},
    )
    plan_artifact, plan = _plan(source, candidate.question_id)
    assert plan.method_family == "group_comparison"
    assert plan.execution_ready is True

    artifacts = _approve_and_execute(source, plan_artifact)
    stat = next(item for item in artifacts if item.type is ArtifactType.STAT_TEST_RESULT)
    # ANOVA (3 groups) always emits a bounded boxplot companion.
    assert any(item.type is ArtifactType.CHART_SPEC for item in artifacts)
    # C4: the stat result is scoped to the single approved dataset.
    assert stat.payload["group_column"] == "region"
    assert stat.payload["value_column"] == "revenue"


def test_group_comparison_finding_carries_assumption_warning(tmp_path: Path) -> None:
    pytest.importorskip("eda_platform.tools.method_findings")
    source, candidate = _prepare_source(
        tmp_path,
        "di3_groups.csv",
        _group_frame(),
        analysis_mode="diagnostic",
        referenced_columns={"di3_groups.csv": ["region", "revenue"]},
    )
    plan_artifact, _ = _plan(source, candidate.question_id)
    artifacts = _approve_and_execute(source, plan_artifact)

    finding = _finding(artifacts)
    assert finding is not None
    assert finding.claim_class == "observed"
    # Broken-assumption data -> a warning must survive into the limitations.
    assert any(
        "assumption" in limitation.lower() or "variance" in limitation.lower()
        for limitation in finding.limitations
    )
    # Every reducer statement resolves to a stored evidence artifact.
    stored = {item.id for item in artifacts}
    for statement in finding.findings:
        assert statement.evidence
        assert all(ref.artifact_id in stored for ref in statement.evidence)


# --------------------------------------------------------------------------- #
# Outcome prediction
# --------------------------------------------------------------------------- #
def test_prediction_leakage_yields_record_not_finding(tmp_path: Path) -> None:
    source, candidate = _prepare_source(
        tmp_path,
        "di3_leaky.csv",
        _prediction_frame(leakage=True),
        analysis_mode="prediction",
        data_requirements=["target column: churn"],
    )
    plan_artifact, plan = _plan(source, candidate.question_id)
    assert plan.method_family == "predictive_modeling"

    artifacts = _approve_and_execute(source, plan_artifact)
    assert _finding(artifacts) is None
    record = _record(artifacts)
    assert record.status == "failed"
    assert record.reason_code == "method_gate_failed"
    assert "leakage" in record.reason.lower()
    assert any(
        gate.name == "method" and gate.status == "failed"
        for gate in record.validation_gates
    )


def test_prediction_success_finding_is_predictive_and_non_causal(tmp_path: Path) -> None:
    pytest.importorskip("eda_platform.tools.method_findings")
    source, candidate = _prepare_source(
        tmp_path,
        "di3_predict.csv",
        _prediction_frame(leakage=False),
        analysis_mode="prediction",
        data_requirements=["target column: churn"],
    )
    plan_artifact, _ = _plan(source, candidate.question_id)
    artifacts = _approve_and_execute(source, plan_artifact)

    assert any(item.type is ArtifactType.MODEL_CARD for item in artifacts)
    finding = _finding(artifacts)
    assert finding is not None
    assert finding.claim_class == "predictive"
    # The non-causal baseline caveat is force-included regardless of eligibility.
    assert any(
        "not a causal" in limitation.lower() or "非因果" in limitation
        for limitation in finding.limitations
    )


def test_prediction_baseline_model_card_is_produced_when_reducer_absent(
    tmp_path: Path,
) -> None:
    # Even before the reducer lands, the guarded baseline executes and its
    # ModelCard is emitted; only the claim sentence waits on the reducer.
    source, candidate = _prepare_source(
        tmp_path,
        "di3_predict.csv",
        _prediction_frame(leakage=False),
        analysis_mode="prediction",
        data_requirements=["target column: churn"],
    )
    plan_artifact, _ = _plan(source, candidate.question_id)
    artifacts = _approve_and_execute(source, plan_artifact)
    assert any(item.type is ArtifactType.MODEL_CARD for item in artifacts)


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
def test_anomaly_constant_column_is_handled(tmp_path: Path) -> None:
    pytest.importorskip("eda_platform.tools.anomaly")
    pytest.importorskip("eda_platform.tools.method_findings")
    source, candidate = _prepare_source(
        tmp_path,
        "di3_constant.csv",
        _anomaly_frame(constant=True),
        analysis_mode="anomaly",
        referenced_columns={"di3_constant.csv": ["value"]},
    )
    plan_artifact, plan = _plan(source, candidate.question_id)
    assert plan.method_family == "anomaly_detection"

    # A constant numeric column (MAD == 0) must not crash the run.
    artifacts = _approve_and_execute(source, plan_artifact)
    record = _record(artifacts)
    assert record.investigation_id == plan.investigation_id
    finding = _finding(artifacts)
    if finding is not None:
        assert finding.claim_class == "observed"


def test_anomaly_finding_evidence_resolves(tmp_path: Path) -> None:
    pytest.importorskip("eda_platform.tools.anomaly")
    pytest.importorskip("eda_platform.tools.method_findings")
    source, candidate = _prepare_source(
        tmp_path,
        "di3_anomaly.csv",
        _anomaly_frame(constant=False),
        analysis_mode="anomaly",
        referenced_columns={"di3_anomaly.csv": ["value"]},
    )
    plan_artifact, _ = _plan(source, candidate.question_id)
    artifacts = _approve_and_execute(source, plan_artifact)
    assert any(item.type is ArtifactType.ANOMALY_SCREEN_RESULT for item in artifacts)
    finding = _finding(artifacts)
    if finding is not None:
        stored = {item.id for item in artifacts}
        for statement in finding.findings:
            assert statement.evidence
            assert all(ref.artifact_id in stored for ref in statement.evidence)


# --------------------------------------------------------------------------- #
# Scope is refused on every new route (before any executor runs)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("filename", "frame", "mode", "requirements", "referenced"),
    [
        (
            "di3_groups.csv",
            _group_frame(),
            "diagnostic",
            None,
            {"di3_groups.csv": ["region", "revenue"]},
        ),
        (
            "di3_predict.csv",
            _prediction_frame(leakage=False),
            "prediction",
            ["target column: churn"],
            None,
        ),
        (
            "di3_anomaly.csv",
            _anomaly_frame(constant=False),
            "anomaly",
            None,
            {"di3_anomaly.csv": ["value"]},
        ),
    ],
)
def test_out_of_scope_dataset_refused_on_every_method_route(
    tmp_path: Path,
    filename: str,
    frame: pd.DataFrame,
    mode: str,
    requirements: list[str] | None,
    referenced: dict[str, list[str]] | None,
) -> None:
    source, candidate = _prepare_source(
        tmp_path,
        filename,
        frame,
        analysis_mode=mode,
        data_requirements=requirements,
        referenced_columns=referenced,
    )
    plan_artifact, plan = _plan(source, candidate.question_id)
    # Narrow the approved scope to a dataset the Card never referenced; execution
    # must not widen scope back to the source dataset.
    store = ArtifactStore(source.workspace)
    plan.target_datasets = ["dataset_outside_the_approved_scope.csv"]
    plan_artifact.payload = plan.model_dump(mode="json")
    store.save_artifact(plan_artifact)

    artifacts = _approve_and_execute(source, plan_artifact)
    assert _finding(artifacts) is None
    record = _record(artifacts)
    assert record.status == "rejected"
    assert record.reason_code == "scope_violation"
    assert any(
        item.type
        in {
            ArtifactType.STAT_TEST_RESULT,
            ArtifactType.MODEL_CARD,
            ArtifactType.ANOMALY_SCREEN_RESULT,
        }
        for item in artifacts
    ) is False

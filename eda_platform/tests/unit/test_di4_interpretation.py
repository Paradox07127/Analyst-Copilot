"""DI sprint-4 (DI4-B): Level-1 calibrated interpretation and its wiring.

The interpretation gate is the red line in code: the LLM writes "what this means"
but every number it prints must trace to deterministic evidence, it may not assert
causation, and it must stay short. These tests drive the gate adversarially (a
model that fabricates a number, one that rewrites in causal language, one that
runs long) and confirm each is discarded to ``fallback`` with the text withheld,
that a well-behaved interpretation is admitted and stamped onto the artifact, and
that offline runs never call the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from eda_platform.agents.interpretation import InterpretationResult, interpret_findings
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.drivers.question_exec import execute_question_candidate
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionExecutionResult,
    QuestionFinding,
    QuestionScore,
)
from eda_platform.tools.loader import LoadedDataset, load_csv

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Fake / spy LLM clients
# --------------------------------------------------------------------------- #
class FakeInterpretationLLM:
    """A live-looking client that returns a fixed interpretation string."""

    def __init__(self, interpretation: str) -> None:
        self._interpretation = interpretation
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        return schema(interpretation=self._interpretation)  # type: ignore[call-arg]

    def text(self, *, task: str, payload: dict) -> str:
        return self._interpretation

    def last_usage(self) -> None:
        return None


class SpyOfflineLLM(OfflineLLMClient):
    """Offline client that records whether ``structured`` was ever invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        return super().structured(task=task, schema=schema, payload=payload)


def _ranked_findings() -> list[QuestionFinding]:
    """One deterministic finding whose evidence and text share the same numbers."""
    return [
        QuestionFinding(
            text=(
                "Which PCA components separate fraud best? The strongest is V3 "
                "(separation 7.045), followed by V14 (separation 6.984)."
            ),
            evidence=[
                EvidenceRef(
                    kind="sql",
                    artifact_id="sql_a",
                    locator="rows_preview[0].separation",
                    value=7.045,
                ),
                EvidenceRef(
                    kind="sql",
                    artifact_id="sql_a",
                    locator="rows_preview[1].separation",
                    value=6.984,
                ),
            ],
        )
    ]


# --------------------------------------------------------------------------- #
# Adversarial gate behaviour
# --------------------------------------------------------------------------- #
def test_fabricated_number_is_rejected_to_fallback() -> None:
    llm = FakeInterpretationLLM(
        "V3 dominates the fraud signal with a separation of 9.999, far above the rest."
    )
    result = interpret_findings(
        llm, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result.status == "fallback"
    assert result.text == ""
    assert "unsupported_number" in result.reject_reason
    assert llm.structured_calls == 1


def test_percent_cannot_be_washed_by_raw_evidence() -> None:
    """A raw evidence value must not legitimize the same digits with a % suffix.

    2026-07-17 trust audit: the gate keeps percent and raw pools separate,
    mirroring the report validator and decision-report gates.
    """
    llm = FakeInterpretationLLM(
        "About 7.045% of transactions separate fraud, led by component V3."
    )
    result = interpret_findings(
        llm, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result.status == "fallback"
    assert "unsupported_number" in result.reject_reason
    assert "%" in result.reject_reason


def test_causal_rewrite_is_rejected_to_fallback() -> None:
    llm = FakeInterpretationLLM(
        "Component V3 drives the fraud outcome, so it causes most flagged transactions."
    )
    result = interpret_findings(
        llm, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result.status == "fallback"
    assert result.text == ""
    assert "causal_language" in result.reject_reason


def test_overlong_interpretation_is_rejected_to_fallback() -> None:
    # No digits and no causal terms, so length is the only failing gate.
    long_text = ("The separation signal concentrates in a handful of components. " * 20).strip()
    assert len(long_text) > 600
    llm = FakeInterpretationLLM(long_text)
    result = interpret_findings(
        llm, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result.status == "fallback"
    assert result.text == ""
    assert "too_long" in result.reject_reason


def test_empty_interpretation_is_rejected_to_fallback() -> None:
    llm = FakeInterpretationLLM("   ")
    result = interpret_findings(
        llm, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result.status == "fallback"
    assert result.reject_reason == "empty_interpretation"


def test_valid_interpretation_echoing_evidence_is_validated() -> None:
    text = (
        "Fraud separation concentrates in V3 at 7.045 with V14 close behind at 6.984, "
        "so a cheap pre-screen on these components looks worthwhile before costly models."
    )
    llm = FakeInterpretationLLM(text)
    result = interpret_findings(
        llm,
        question="Which components separate fraud?",
        findings=_ranked_findings(),
        method_context="ranked component separation",
        limitations=["Observed on the current sample only."],
    )
    assert result.status == "validated"
    assert result.text == text
    assert result.reject_reason == ""


def test_offline_client_returns_absent_without_calling_the_model() -> None:
    spy = SpyOfflineLLM()
    result = interpret_findings(
        spy, question="Which components separate fraud?", findings=_ranked_findings()
    )
    assert result == InterpretationResult(status="absent")
    assert result.text == ""
    assert spy.structured_calls == 0


def test_no_findings_returns_absent_without_calling_the_model() -> None:
    llm = FakeInterpretationLLM("anything")
    result = interpret_findings(llm, question="q", findings=[])
    assert result.status == "absent"
    assert llm.structured_calls == 0


# --------------------------------------------------------------------------- #
# question_exec wiring
# --------------------------------------------------------------------------- #
def _tiny_dataset(tmp_path: Path) -> LoadedDataset:
    csv = tmp_path / "tiny.csv"
    csv.write_text("a\n1\n2\n", encoding="utf-8")
    return load_csv(csv, dataset_id="ds_tiny")


def _template_candidate() -> QuestionCandidate:
    return QuestionCandidate(
        question_id="q_value",
        question_en="What is the value?",
        origin="template",
        template_id="group_difference",
        target_datasets=["tiny.csv"],
        sql_template="SELECT 42 AS value",
        score=QuestionScore(
            data_availability=1.0,
            statistical_signal=0.5,
            quality_risk=0.0,
            join_risk=0.0,
            deterministic_score=0.8,
        ),
    )


def _qexec_from(artifacts: list[Artifact]) -> QuestionExecutionResult:
    artifact = next(
        item for item in artifacts if item.type is ArtifactType.QUESTION_EXECUTION_RESULT
    )
    return QuestionExecutionResult.model_validate(artifact.payload)


def test_question_exec_stamps_validated_interpretation(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    llm = FakeInterpretationLLM("The returned value settles at 42, a modest figure to monitor.")
    artifacts = execute_question_candidate(
        _template_candidate(),
        datasets=[dataset],
        project_id="p",
        session_id="r",
        parent_ids=["src"],
        llm=llm,
    )
    qexec = _qexec_from(artifacts)
    assert qexec.status == "succeeded"
    assert qexec.interpretation_status == "validated"
    assert "42" in qexec.interpretation
    assert llm.structured_calls == 1


def test_question_exec_offline_stamps_absent_without_model_call(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    spy = SpyOfflineLLM()
    artifacts = execute_question_candidate(
        _template_candidate(),
        datasets=[dataset],
        project_id="p",
        session_id="r",
        parent_ids=["src"],
        llm=spy,
    )
    qexec = _qexec_from(artifacts)
    assert qexec.status == "succeeded"
    assert qexec.interpretation_status == "absent"
    assert qexec.interpretation == ""
    assert spy.structured_calls == 0


def test_question_exec_none_client_stamps_absent(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    artifacts = execute_question_candidate(
        _template_candidate(),
        datasets=[dataset],
        project_id="p",
        session_id="r",
        parent_ids=["src"],
        llm=None,
    )
    qexec = _qexec_from(artifacts)
    assert qexec.interpretation_status == "absent"
    assert qexec.interpretation == ""


# --------------------------------------------------------------------------- #
# Orchestrator wiring (offline end-to-end)
# --------------------------------------------------------------------------- #
def _source(tmp_path: Path) -> AutoEDAResult:
    return run_auto_eda(
        [GOLDEN_DATA / "ecommerce_orders.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="source_run",
    )


def test_orchestrator_offline_stamps_absent_on_validated_finding(tmp_path: Path) -> None:
    source = _source(tmp_path)
    candidate_set = QuestionCandidateSet.model_validate(
        next(
            item.payload
            for item in source.artifacts
            if item.type is ArtifactType.QUESTION_CANDIDATE_SET
        )
    )
    candidate = next(
        item
        for item in candidate_set.candidates
        if item.origin == "template" and item.sql_template is not None
    )
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[candidate.question_id],
        workspace=source.workspace,
        session_id="plan_run",
    )
    plan_artifact = next(
        item for item in planned.artifacts if item.type is ArtifactType.INVESTIGATION_PLAN
    )
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    completed = execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
    )
    finding = ValidatedFinding.model_validate(
        next(
            item.payload
            for item in completed.artifacts
            if item.type is ArtifactType.VALIDATED_FINDING
        )
    )
    # Offline run: no model was available, so interpretation is absent, not fabricated.
    assert finding.interpretation_status == "absent"
    assert finding.interpretation == ""

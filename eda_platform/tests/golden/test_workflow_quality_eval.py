from __future__ import annotations

from pathlib import Path

from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.drivers.workflow_eval import run_fresh_workflow_eval_case
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.questions import QuestionExecutionResult
from eda_platform.schemas.workflow_eval import WorkflowEvalSpec
from eda_platform.tools.workflow_eval import evaluate_workflow_run

_EVAL_DIR = Path(__file__).parents[1] / "evals" / "workflow_quality"


def test_numeric_postal_code_guardrail_passes_whole_workflow_eval(tmp_path: Path) -> None:
    spec = WorkflowEvalSpec.model_validate_json(
        (_EVAL_DIR / "cases" / "semantic_guardrails.json").read_text(encoding="utf-8")
    )
    files = [_EVAL_DIR / "data" / filename for filename in spec.input_files]
    result = run_auto_eda(
        files,
        workspace=tmp_path / "workspace",
        project_id="workflow_eval_semantic_guardrails",
        business_context=spec.business_context,
        llm=OfflineLLMClient(),
    )
    artifacts = ArtifactStore(result.workspace).list_artifacts(
        project_id=result.project_id,
        session_id=result.session_id,
    )

    evaluation = evaluate_workflow_run(artifacts, spec)

    assert evaluation.passed, evaluation.gate_failures
    assert evaluation.answered_count == 0
    assert evaluation.semantic_escape_rate == 0.0
    assert evaluation.report_dataset_coverage == 1.0
    assert evaluation.total_tokens == 0


def test_contract_matrix_abstains_through_real_batch_path(tmp_path: Path) -> None:
    spec = WorkflowEvalSpec.model_validate_json(
        (_EVAL_DIR / "cases" / "contract_abstention.json").read_text(encoding="utf-8")
    )
    artifact_runs = run_fresh_workflow_eval_case(
        spec,
        input_dir=_EVAL_DIR / "data",
        workspace=tmp_path / "contract_workspace",
        repeat=1,
    )

    evaluation = evaluate_workflow_run(artifact_runs[0], spec)

    assert evaluation.passed, evaluation.gate_failures
    assert evaluation.answered_count == 1
    assert evaluation.abstained_count == 7
    assert evaluation.abstention_precision == 1.0
    assert evaluation.abstention_recall == 1.0
    assert evaluation.duration_seconds > 0
    assert evaluation.semantic_escape_rate == 0.0
    assert evaluation.total_tokens == 0
    results = [
        QuestionExecutionResult.model_validate(artifact.payload)
        for artifact in artifact_runs[0]
        if artifact.type is ArtifactType.QUESTION_EXECUTION_RESULT
    ]
    assert {
        result.abstention_code for result in results if result.outcome == "abstained"
    } == {
        "answer_schema_mismatch",
        "duration_out_of_range",
        "empty_query_result",
        "hhi_out_of_range",
        "missing_metric_output",
        "metric_unit_mismatch",
        "non_finite_metric_output",
    }
    seeded_currency = next(
        result for result in results if "seeded-currency GMV" in result.question
    )
    assert seeded_currency.outcome == "answered"
    assert any("100 BRL" in finding.text for finding in seeded_currency.findings)
    assert any(
        evidence.unit == "currency"
        and evidence.unit_label == "BRL"
        and evidence.unit_reference == "ISO 4217 List One@2026-01-01"
        for finding in seeded_currency.findings
        for evidence in finding.evidence
        if evidence.locator.endswith(".gmv_total")
    )

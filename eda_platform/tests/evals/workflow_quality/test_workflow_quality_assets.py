from __future__ import annotations

import re
from pathlib import Path

from eda_platform.core.ids import stable_hash
from eda_platform.schemas.workflow_eval import WorkflowEvalSpec, WorkflowEvalSuiteResult

_ROOT = Path(__file__).parent


def test_workflow_quality_case_specs_are_typed_and_regexes_compile() -> None:
    case_paths = sorted((_ROOT / "cases").glob("*.json"))
    assert {path.stem for path in case_paths} == {
        "contract_abstention",
        "credit",
        "olist",
        "semantic_guardrails",
    }

    for path in case_paths:
        spec = WorkflowEvalSpec.model_validate_json(path.read_text(encoding="utf-8"))
        patterns = [
            *(answer.question_pattern for answer in spec.expected_answers),
            *(
                pattern
                for answer in spec.expected_answers
                for pattern in answer.required_output_patterns
            ),
            *(abstention.question_pattern for abstention in spec.expected_abstentions),
            *spec.required_executive_summary_patterns,
            *spec.forbidden_output_patterns,
        ]
        for pattern in patterns:
            re.compile(pattern, re.IGNORECASE)


def test_repository_guardrail_case_inputs_exist() -> None:
    for case_name in ("contract_abstention", "semantic_guardrails"):
        spec = WorkflowEvalSpec.model_validate_json(
            (_ROOT / "cases" / f"{case_name}.json").read_text(encoding="utf-8")
        )
        assert all((_ROOT / "data" / filename).is_file() for filename in spec.input_files)


def test_repository_baselines_match_their_case_specs() -> None:
    for baseline_path in sorted((_ROOT / "baselines").glob("*.json")):
        spec = WorkflowEvalSpec.model_validate_json(
            (_ROOT / "cases" / baseline_path.name).read_text(encoding="utf-8")
        )
        baseline = WorkflowEvalSuiteResult.model_validate_json(
            baseline_path.read_text(encoding="utf-8")
        )
        assert baseline.case_name == spec.name
        assert baseline.spec_digest == stable_hash(
            spec.model_dump(mode="json"), length=32
        )
        assert baseline.passed

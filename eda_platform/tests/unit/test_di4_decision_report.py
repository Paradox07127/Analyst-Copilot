from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eda_platform.core.publication_fingerprint import DECISION_REPORT_POLICY_VERSION
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.decision_report import _text_numbers_resolve, create_decision_report
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef
from eda_platform.schemas.decision_report import DecisionReport
from eda_platform.schemas.investigations import ValidatedFinding
from eda_platform.schemas.quality_context import QualityContext
from eda_platform.schemas.questions import QuestionFinding
from eda_platform.schemas.synthesis import SynthesisBrief, SynthesisStoryBeat
from eda_platform.tools.exporter import decision_report_to_markdown

_NUMBER_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")
_CAUSAL_WORDS = ("causes", "caused", "causal", "drives", "because of", "leads to")


class FakeLLM:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response
        self.calls = 0

    def structured(self, *, task: str, schema: type[Any], payload: dict[str, Any]) -> Any:
        self.calls += 1
        return schema.model_validate(self.response)

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        raise AssertionError("Decision report refinement must use structured output.")

    def last_usage(self) -> None:
        return None


def _finding(
    *,
    finding_id: str,
    question: str,
    claims: list[QuestionFinding],
    limitations: list[str] | None = None,
    interpretation: str = "",
) -> ValidatedFinding:
    return ValidatedFinding(
        finding_id=finding_id,
        investigation_id=f"inv_{finding_id}",
        question_id=f"q_{finding_id}",
        question=question,
        quality_context=[
            QualityContext(
                context_id=f"context_{finding_id}",
                dataset_id="ds_orders",
                dataset_name="orders.csv",
                issue_code="high_missing",
                severity="warn",
                column="amount",
                observation="Amount completeness was reviewed.",
                report_limitation="Amount coverage may narrow the interpretation.",
            )
        ],
        claim_class="observed",
        findings=claims,
        evidence_support="high",
        analytical_reliability="high",
        decision_readiness="medium",
        limitations=limitations or [],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
        report_readiness_reason="Validated with disclosed data conditions.",
        interpretation=interpretation,
        interpretation_status="validated" if interpretation else "absent",
    )


def _seed_store(tmp_path: Path) -> tuple[ArtifactStore, str, list[str]]:
    store = ArtifactStore(tmp_path / "workspace")
    project_id = "project_di4"
    session_id = "synthesis_run"
    store.ensure_project(project_id, "DI4")
    store.start_session(project_id, session_id)

    findings = [
        _finding(
            finding_id="finding_orders",
            question="How do order values vary by channel?",
            claims=[
                QuestionFinding(
                    text="The observed average order value is 125.5.",
                    evidence=[
                        EvidenceRef(
                            kind="table",
                            artifact_id="table_orders",
                            locator="rows[0].average_order_value",
                            value=125.5,
                        )
                    ],
                ),
                QuestionFinding(
                    text="A fabricated subgroup contains 999 orders.",
                    evidence=[],
                ),
            ],
            limitations=["Channel labels require review."],
            interpretation="The 125.5 observation should be read with channel coverage in mind.",
        ),
        _finding(
            finding_id="finding_returns",
            question="What share of orders are returned?",
            claims=[
                QuestionFinding(
                    text="The observed return share is 8%.",
                    evidence=[
                        EvidenceRef(
                            kind="stat",
                            artifact_id="stat_returns",
                            locator="return_share",
                            value=8,
                            unit="percent",
                        )
                    ],
                )
            ],
        ),
    ]
    finding_ids: list[str] = []
    for index, finding in enumerate(findings, start=1):
        artifact = Artifact(
            id=f"vf_{index}",
            type=ArtifactType.VALIDATED_FINDING,
            project_id=project_id,
            session_id="finding_run",
            payload=finding.model_dump(mode="json"),
        )
        store.start_session(project_id, artifact.session_id)
        store.save_artifact(artifact)
        finding_ids.append(artifact.id)

    brief = SynthesisBrief(
        brief_id="brief_di4",
        project_id=project_id,
        selected_finding_artifact_ids=finding_ids,
        decision_context="Which evidence should guide the channel decision?",
        headline="The observed average order value is 125.5.",
        storyline=[
            SynthesisStoryBeat(
                title="Evidence",
                body="Validated findings are ready for review.",
                finding_artifact_ids=finding_ids,
            )
        ],
        limitations=["Channel labels require review."],
        investigation_gaps=["Validate return patterns in later periods."],
        report_eligible=True,
        report_readiness="eligible_with_limitations",
    )
    brief_artifact = Artifact(
        id="sbrief_di4",
        type=ArtifactType.SYNTHESIS_BRIEF,
        project_id=project_id,
        session_id=session_id,
        parents=finding_ids,
        payload=brief.model_dump(mode="json"),
    )
    store.save_artifact(brief_artifact)
    return store, brief_artifact.id, finding_ids


def _assert_every_number_resolves(report: DecisionReport, findings: list[ValidatedFinding]) -> None:
    evidence: list[tuple[float, str]] = []
    for finding in findings:
        for claim in finding.findings:
            for ref in claim.evidence:
                if isinstance(ref.value, int | float) and not isinstance(ref.value, bool):
                    evidence.append((float(ref.value), ref.unit))
    texts = [
        report.scqa.situation,
        report.scqa.complication,
        report.scqa.question,
        report.scqa.answer,
        *(section.body for section in report.sections),
    ]
    for text in texts:
        for match in _NUMBER_PATTERN.finditer(text):
            token = match.group(0)
            number = float(token.removesuffix("%"))
            unit = "percent" if token.endswith("%") else "raw"
            pool = [value for value, evidence_unit in evidence if evidence_unit == unit]
            if number.is_integer():
                assert number in pool
            else:
                assert any(abs(number - value) <= max(abs(value) * 0.01, 0.01) for value in pool)


def _load_report(store: ArtifactStore, artifact_id: str) -> DecisionReport:
    return DecisionReport.model_validate(store.get_artifact(artifact_id).payload)


def test_decision_report_number_gate_accepts_only_matching_exact_currency() -> None:
    evidence = [(120.0, "currency", "BRL")]

    assert _text_numbers_resolve("GMV is 120 BRL.", evidence) is True
    assert _text_numbers_resolve("GMV is BRL 120.", evidence) is True
    assert _text_numbers_resolve("GMV is 120 USD.", evidence) is False
    assert _text_numbers_resolve("GMV is 120.", evidence) is False


def test_deterministic_assembly_is_evidence_bounded_and_persisted(tmp_path: Path) -> None:
    store, brief_id, finding_ids = _seed_store(tmp_path)
    report_id = create_decision_report(store, project_id="project_di4", brief_artifact_id=brief_id)
    artifact = store.get_artifact(report_id)
    report = DecisionReport.model_validate(artifact.payload)

    assert artifact.type is ArtifactType.DECISION_REPORT
    assert artifact.parents == [brief_id, *finding_ids]
    assert report.source_finding_artifact_ids == finding_ids
    assert report.report_policy_version == DECISION_REPORT_POLICY_VERSION
    assert report.publication_input_fingerprint is not None
    assert len(report.publication_input_fingerprint) == 64
    assert report.narrative_status == "deterministic"
    assert "orders.csv" in report.scqa.situation
    assert report.scqa.question == "Which evidence should guide the channel decision?"
    assert "These observations come with conditions:" in report.scqa.complication
    assert [section.title for section in report.sections] == [
        "How do order values vary by channel?",
        "What share of orders are returned?",
    ]
    assert "125.5 observation" in report.sections[0].body
    assert report.limitations == ["Channel labels require review."]
    assert report.investigation_gaps == ["Validate return patterns in later periods."]
    assert report.report_readiness == "eligible_with_limitations"

    findings = [
        ValidatedFinding.model_validate(store.get_artifact(item).payload) for item in finding_ids
    ]
    _assert_every_number_resolves(report, findings)


def test_unresolved_number_sentence_is_stripped_and_answer_has_no_causal_wording(
    tmp_path: Path,
) -> None:
    store, brief_id, _ = _seed_store(tmp_path)
    brief_artifact = store.get_artifact(brief_id)
    brief_artifact.payload["headline"] = "The outage causes higher returns of 777%."
    store.save_artifact(brief_artifact)

    report = _load_report(
        store,
        create_decision_report(store, project_id="project_di4", brief_artifact_id=brief_id),
    )
    rendered = " ".join(
        [report.scqa.situation, report.scqa.complication, report.scqa.answer]
        + [section.body for section in report.sections]
    )
    assert "999" not in rendered
    assert "777" not in rendered
    assert all(word not in report.scqa.answer.lower() for word in _CAUSAL_WORDS)


def test_llm_refinement_is_accepted_only_when_number_and_causal_gates_pass(
    tmp_path: Path,
) -> None:
    store, brief_id, _ = _seed_store(tmp_path)
    fabricated = FakeLLM(
        {
            "situation": "A fabricated 4567-unit situation.",
            "complication": "Coverage needs review.",
            "answer": "Review the observations.",
        }
    )
    rejected = _load_report(
        store,
        create_decision_report(
            store,
            project_id="project_di4",
            brief_artifact_id=brief_id,
            llm=fabricated,
        ),
    )
    assert fabricated.calls == 1
    assert rejected.narrative_status == "deterministic"
    assert "4567" not in rejected.scqa.situation

    accepted_llm = FakeLLM(
        {
            "situation": "The selected analyses frame the current decision.",
            "complication": "Coverage conditions narrow interpretation.",
            "answer": "Use the validated observations in the decision review.",
        }
    )
    accepted = _load_report(
        store,
        create_decision_report(
            store,
            project_id="project_di4",
            brief_artifact_id=brief_id,
            llm=accepted_llm,
        ),
    )
    assert accepted.narrative_status == "llm_refined"
    assert accepted.scqa.situation == "The selected analyses frame the current decision."


def test_markdown_contains_scqa_headers(tmp_path: Path) -> None:
    store, brief_id, _ = _seed_store(tmp_path)
    report = _load_report(
        store,
        create_decision_report(store, project_id="project_di4", brief_artifact_id=brief_id),
    )
    markdown = decision_report_to_markdown(report)
    for heading in ("## Situation", "## Complication", "## Question", "## Answer"):
        assert heading in markdown
    assert "## Limitations" in markdown
    assert "## Investigation Gaps" in markdown

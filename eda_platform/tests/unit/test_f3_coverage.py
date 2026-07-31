"""F3: quantitative coverage gap — "delete the number to stay safe" made visible.

A claim in a quantitative section with zero number_verified tokens is recorded
as a coverage gap: an independent disclosure metric, never a finding, never a
rewrite trigger, never a gate_verdict change.
"""

from __future__ import annotations

from eda_platform.agents.reporting import _bundle_from_plan
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
)
from eda_platform.tools.evidence import (
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.report_validator import validate_report_bundle


def test_numberless_claim_in_quantitative_section_is_a_gap() -> None:
    # R1: §5.1 sanitized-draft style — no numbers at all, resolvable evidence.
    claim = ReportClaim(
        text="Age shows very high missing rates.",
        evidence=[
            EvidenceRef(
                kind="profile_field",
                artifact_id="prof_hr",
                locator="missing_percent.Age",
            )
        ],
        referenced_datasets=["hr.csv"],
        referenced_columns=["Age"],
        quality_issue_refs=["qi_age_missing"],
    )
    audit = validate_report_bundle(
        _bundle_with_claim("Key EDA Insights", claim), _pack()
    )

    assert claim.quantitative_coverage_gap is True
    assert audit.quantitative_coverage_gap_count == 1
    # Disclosure only: no finding, no rewrite trigger, no verdict change.
    assert [f for f in audit.findings if "coverage" in f.code] == []
    assert audit.gate_verdict == "pass"


def test_same_text_outside_quantitative_sections_is_not_a_gap() -> None:
    # R2: identical claim in a non-quantitative section (Dataset Overview
    # joined QUANTITATIVE_SECTIONS 2026-07-23; Selected Analysis Focus did not).
    claim = ReportClaim(
        text="Age shows very high missing rates.",
        evidence=[
            EvidenceRef(
                kind="profile_field",
                artifact_id="prof_hr",
                locator="missing_percent.Age",
            )
        ],
        referenced_datasets=["hr.csv"],
        referenced_columns=["Age"],
        quality_issue_refs=["qi_age_missing"],
    )
    audit = validate_report_bundle(
        _bundle_with_claim("Selected Analysis Focus", claim), _pack()
    )

    assert claim.quantitative_coverage_gap is False
    assert audit.quantitative_coverage_gap_count == 0


def test_verified_number_closes_the_gap_but_all_unverified_does_not() -> None:
    # R3a: a number_verified token means the section claim has coverage.
    verified = ReportClaim(
        text="The file contains 1470 employee records.",
        evidence=[
            EvidenceRef(kind="artifact", artifact_id="prof_hr", locator="rows")
        ],
        referenced_datasets=["hr.csv"],
    )
    audit = validate_report_bundle(
        _bundle_with_claim("Business Findings", verified), _pack()
    )
    assert verified.numeric_rollup == "number_verified"
    assert verified.quantitative_coverage_gap is False
    assert audit.quantitative_coverage_gap_count == 0

    # R3b: numbers present but all unverified -> still a coverage gap.
    unverified = ReportClaim(
        text="Total sessions reached 9999.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="qexec_1",
                locator="findings[0]",
                value=9999,
            )
        ],
        referenced_datasets=["hr.csv"],
    )
    audit = validate_report_bundle(
        _bundle_with_claim("Business Findings", unverified), _pack()
    )
    assert unverified.numeric_rollup == "unverified"
    assert unverified.quantitative_coverage_gap is True
    assert audit.quantitative_coverage_gap_count == 1


def test_deterministic_source_claims_are_exempt() -> None:
    # R4: the platform's own fallback claim (QualityIssueSet evidence is
    # unresolvable by design) must not be punished by the platform's metric.
    claim = ReportClaim(
        id="quality_issue_count",
        text="hr.csv quality scan found 11 issues.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="quality_1",
                locator="issues",
                value=11,
            )
        ],
        referenced_datasets=["hr.csv"],
        quality_issue_refs=["quality_1"],
        deterministic_source=True,
    )
    audit = validate_report_bundle(
        _bundle_with_claim("Data Quality Findings", claim), _pack()
    )

    assert claim.numeric_rollup == "unverified"
    assert claim.quantitative_coverage_gap is False
    assert audit.quantitative_coverage_gap_count == 0


def test_plan_authored_deterministic_source_cannot_dodge_the_gap() -> None:
    # R5: deterministic_source is set only at platform generation sites. A
    # plan claim asserting it must lose the flag in _report_claim_from_plan's
    # whitelist; this pins the escape hatch shut against a future refactor to
    # a model_dump full copy.
    draft = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Key EDA Insights",
                id="c1",
                text="Age shows very high missing rates.",
                evidence=[
                    EvidenceRef(
                        kind="profile_field",
                        artifact_id="prof_hr",
                        locator="missing_percent.Age",
                    )
                ],
                referenced_datasets=["hr.csv"],
                referenced_columns=["Age"],
                quality_issue_refs=["qi_age_missing"],
                deterministic_source=True,
            )
        ],
    )
    bundle, _dropped = _bundle_from_plan(draft, project_id="project_demo", session_id="run_demo")
    audit = validate_report_bundle(bundle, _pack())

    claim = next(
        claim
        for section in bundle.sections
        for claim in section.claims
        if claim.id == "c1"
    )
    assert claim.deterministic_source is False
    assert claim.quantitative_coverage_gap is True
    assert audit.quantitative_coverage_gap_count == 1


def test_revalidation_is_idempotent() -> None:
    claim = ReportClaim(
        text="Age shows very high missing rates.",
        evidence=[
            EvidenceRef(
                kind="profile_field",
                artifact_id="prof_hr",
                locator="missing_percent.Age",
            )
        ],
        referenced_datasets=["hr.csv"],
        referenced_columns=["Age"],
        quality_issue_refs=["qi_age_missing"],
    )
    bundle = _bundle_with_claim("Key EDA Insights", claim)
    validate_report_bundle(bundle, _pack())
    audit = validate_report_bundle(bundle, _pack())
    assert claim.quantitative_coverage_gap is True
    assert audit.quantitative_coverage_gap_count == 1


def _bundle_with_claim(section_title: str, claim: ReportClaim) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    section = next(s for s in bundle.sections if s.title == section_title)
    section.claims.append(claim)
    return bundle


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "prof_hr": EvidenceArtifactSummary(
                artifact_id="prof_hr",
                artifact_type="DatasetProfile",
                title="hr.csv",
                dataset_id="ds_hr",
            ),
            "qexec_1": EvidenceArtifactSummary(
                artifact_id="qexec_1",
                artifact_type="QuestionExecutionResult",
                title="Question execution",
                dataset_id="ds_hr",
            ),
            "quality_1": EvidenceArtifactSummary(
                artifact_id="quality_1",
                artifact_type="QualityIssueSet",
                title="hr.csv quality issues",
                dataset_id="ds_hr",
            ),
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_hr",
                dataset_id="ds_hr",
                name="hr.csv",
                row_count=1470,
                column_count=35,
                columns=["Age", "Attrition"],
                dtypes={"Age": "int64", "Attrition": "object"},
                missing_percent={"Age": 93.27},
            ),
        ],
    )

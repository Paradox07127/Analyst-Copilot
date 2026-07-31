"""Evidence readability layer: deterministic per-claim provenance lines."""

from __future__ import annotations

from eda_platform.schemas.artifacts import (
    AnalysisTable,
    Artifact,
    ArtifactType,
    EvidenceRef,
)
from eda_platform.schemas.reports import ReportBundle, ReportClaim
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
    EvidenceQualityIssue,
    build_evidence_pack,
)
from eda_platform.tools.evidence_display import evidence_display_context, evidence_lines
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html
from eda_platform.tools.report_validator import validate_report_bundle


def _bundle_with_claim(claim: ReportClaim) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for section in bundle.sections:
        section.body = f"{section.title} body."
    bundle.sections[0].claims.append(claim)
    return bundle


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "prof_1": EvidenceArtifactSummary(
                artifact_id="prof_1",
                artifact_type="DatasetProfile",
                title="hr.csv",
                dataset_id="ds_hr",
            ),
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_hr",
            ),
            "qual_1": EvidenceArtifactSummary(
                artifact_id="qual_1",
                artifact_type="QualityIssueSet",
                title="Quality issues",
                dataset_id="ds_hr",
            ),
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_hr",
                name="hr.csv",
                row_count=1470,
                column_count=3,
                columns=["age", "salary", "score"],
                dtypes={"age": "int64", "salary": "float64", "score": "float64"},
            )
        ],
        quality_issues=[
            # Legacy issue: prose figures only, no structured mirrors, so the
            # ref resolves nothing and the claim publishes as unverified.
            EvidenceQualityIssue(
                artifact_id="qual_1",
                dataset_id="ds_hr",
                severity="warn",
                code="high_missing",
                column="score",
                message="score is 89% missing.",
                recommendation="Treat score findings as limited.",
            )
        ],
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_hr",
                title="Numeric summary",
                kind="aggregation",
                description="Salary summary",
                rows=[{"revenue": 120, "discount": 40}],
            )
        ],
    )


def test_verified_number_line_has_value_type_and_locator() -> None:
    claim = ReportClaim(
        id="c_rows",
        text="hr.csv contains 1470 rows.",
        evidence=[
            EvidenceRef(kind="profile_field", artifact_id="prof_1", locator="row_count")
        ],
        referenced_datasets=["hr.csv"],
    )
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    validate_report_bundle(bundle, pack)

    lines = evidence_lines(claim, pack, {})

    assert lines[0] == "✓ 1 of 1 figure verified"
    ref_line = next(line for line in lines[1:] if line.startswith("✓"))
    assert "1470" in ref_line
    assert "dataset profile" in ref_line
    assert "'hr.csv'" in ref_line
    assert "row_count" in ref_line
    assert "(exact)" in ref_line


def test_unverified_quality_prose_ref_is_labeled_truthfully() -> None:
    claim = ReportClaim(
        id="c_quality",
        text="score is 89% missing.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="qual_1",
                locator="quality_issue:high_missing:score",
            )
        ],
        quality_issue_refs=["high_missing"],
    )
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    validate_report_bundle(bundle, pack)

    lines = evidence_lines(claim, pack, {})

    assert lines[0].startswith("◌")
    assert "0 of 1 figure verified" in lines[0]
    ref_line = next(line for line in lines[1:] if "unresolvable" in line)
    assert "quality issues" in ref_line
    assert "quality_issue:high_missing:score" in ref_line
    assert "prose figure, unverified" in ref_line


def test_failed_line_contains_expected_pool() -> None:
    claim = ReportClaim(
        id="c_failed",
        text="Revenue is 999.",
        evidence=[EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")],
    )
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    validate_report_bundle(bundle, pack)

    lines = evidence_lines(claim, pack, {})

    assert lines[0].startswith("✗")
    failed_line = next(line for line in lines[1:] if line.startswith("✗"))
    assert "999" in failed_line
    assert "40" in failed_line
    assert "120" in failed_line


def test_claim_without_numbers_gets_dash_summary() -> None:
    claim = ReportClaim(
        id="c_plain",
        text="Salary distributions look plausible.",
        evidence=[EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")],
    )
    bundle = _bundle_with_claim(claim)
    pack = _pack()
    validate_report_bundle(bundle, pack)

    lines = evidence_lines(claim, pack, {})

    assert lines[0].startswith("–")


def _exportable_bundle_and_artifacts() -> tuple[ReportBundle, list[Artifact]]:
    table_artifact = Artifact(
        id="table_1",
        type=ArtifactType.TABLE,
        project_id="project_demo",
        session_id="run_demo",
        payload=AnalysisTable(
            dataset_id="ds_hr",
            title="Numeric summary",
            kind="numeric_summary",
            description="Salary summary",
            rows=[{"revenue": 120, "discount": 40}],
        ).model_dump(),
    )
    claim = ReportClaim(
        id="c_export",
        text="Revenue is 120.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0].revenue")
        ],
    )
    bundle = _bundle_with_claim(claim)
    validate_report_bundle(bundle, build_evidence_pack([table_artifact]))
    return bundle, [table_artifact]


def test_markdown_export_contains_evidence_lines() -> None:
    bundle, artifacts = _exportable_bundle_and_artifacts()

    markdown = report_bundle_to_markdown(bundle, artifacts=artifacts)

    ledger = markdown[markdown.index("### Claim Ledger") :]
    assert "✓ 1 of 1 figure verified" in ledger
    assert "  - ✓ 120 — analysis table 'Numeric summary' · rows[0].revenue (exact)" in ledger


def test_html_export_contains_evidence_lines() -> None:
    bundle, artifacts = _exportable_bundle_and_artifacts()

    html = export_report_html(bundle, artifacts=artifacts)

    ledger = html[html.index("Claim Ledger") :]
    assert "✓ 1 of 1 figure verified" in ledger
    assert "analysis table &#x27;Numeric summary&#x27;" in ledger

    # Without artifacts the exporter still reports the claim-level rollup but
    # cannot (and must not pretend to) resolve per-ref values.
    degraded = export_report_html(bundle)
    degraded_ledger = degraded[degraded.index("Claim Ledger") :]
    assert "✓ 1 of 1 figure verified" in degraded_ledger
    assert "analysis table" not in degraded_ledger


# --- E1: display must rebuild the pack under the validation policy -----------


def _schema_only_bundle_and_artifacts() -> tuple[ReportBundle, list[Artifact]]:
    """A bundle whose numeric statuses were produced under schema_only."""
    table_artifact = Artifact(
        id="table_1",
        type=ArtifactType.TABLE,
        project_id="project_demo",
        session_id="run_demo",
        payload=AnalysisTable(
            dataset_id="ds_hr",
            title="Numeric summary",
            kind="numeric_summary",
            description="Salary summary",
            rows=[{"revenue": 120, "discount": 40}],
        ).model_dump(),
    )
    claim = ReportClaim(
        id="c_policy",
        text="Revenue is 120.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0].revenue")
        ],
    )
    bundle = _bundle_with_claim(claim)
    validate_report_bundle(
        bundle, build_evidence_pack([table_artifact], payload_policy="schema_only")
    )
    return bundle, [table_artifact]


def test_display_context_honors_the_validation_policy() -> None:
    bundle, artifacts = _schema_only_bundle_and_artifacts()
    claim = bundle.sections[0].claims[0]

    pack, sql_results = evidence_display_context(artifacts, payload_policy="schema_only")
    lines = evidence_lines(claim, pack, sql_results)

    # The statuses were computed under schema_only; the display pack must agree
    # — no source line may show a green resolved value the validator never saw.
    assert not any(line.startswith("✓") for line in lines)
    assert any("unresolvable" in line for line in lines)


def test_markdown_export_accepts_the_validation_policy() -> None:
    bundle, artifacts = _schema_only_bundle_and_artifacts()

    markdown = report_bundle_to_markdown(
        bundle, artifacts=artifacts, payload_policy="schema_only"
    )

    ledger = markdown[markdown.index("### Claim Ledger") :]
    assert "unresolvable" in ledger
    assert "✓ 120" not in ledger


def test_html_export_accepts_the_validation_policy() -> None:
    bundle, artifacts = _schema_only_bundle_and_artifacts()

    html = export_report_html(bundle, artifacts=artifacts, payload_policy="schema_only")

    ledger = html[html.index("Claim Ledger") :]
    assert "unresolvable" in ledger
    assert "✓ 120" not in ledger

from __future__ import annotations

from eda_platform.agents.reporting import _validation_trace_event
from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportBundle, ReportClaim
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.report_validator import validate_report_bundle


def test_numeric_mismatch_finding_carries_failed_number_and_evidence() -> None:
    bundle = _bundle_with_claims(
        [
            ReportClaim(
                text="Revenue is 9999.",
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id="table_1",
                        locator="rows[0]",
                        value=1470,
                    )
                ],
                referenced_datasets=["sales.csv"],
                referenced_columns=["revenue"],
            )
        ]
    )

    audit = validate_report_bundle(bundle, _pack())

    mismatches = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(mismatches) == 1
    finding = mismatches[0]
    assert finding.numeric_details
    detail = finding.numeric_details[0]
    assert detail.number == 9999
    assert detail.reason == "outside_tolerance"
    assert 1470 in detail.evidence_values
    assert any(
        source.artifact_id == "table_1" and source.locator == "rows[0]"
        for source in detail.sources
    )
    assert "9999" in finding.message


def test_empty_evidence_pool_is_unverified_not_a_mismatch() -> None:
    # F1 third state: an unresolvable pool means "cannot verify", not "wrong".
    claim = ReportClaim(
        text="Revenue is 120.",
        evidence=[
            EvidenceRef(
                kind="chart",
                artifact_id="chart_1",
                locator="series[0]",
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    assert [f for f in audit.findings if f.code == "numeric_mismatch"] == []
    assert claim.numeric_rollup == "unverified"
    assert audit.numeric_unverified_claim_count == 1


def test_trace_event_serializes_all_findings_with_structure() -> None:
    claims = [
        ReportClaim(
            id=f"claim_{index}",
            text="Revenue is 9999.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=1470,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["revenue"],
        )
        for index in range(12)
    ]
    bundle = _bundle_with_claims(claims)

    audit = validate_report_bundle(bundle, _pack())
    event = _validation_trace_event(audit, bundle=bundle, attempt=1)

    assert len(audit.findings) == 12
    # No truncation: every finding is serialized, both flat and structured.
    assert len(event.findings) == len(audit.findings)
    assert len(event.structured_findings) == len(audit.findings)
    structured = event.structured_findings[0]
    assert structured["code"] == "numeric_mismatch"
    assert structured["numeric_details"][0]["number"] == 9999


def test_overlong_number_token_does_not_crash_validation() -> None:
    # A 400-digit token parses to float("inf"); message formatting must not
    # raise OverflowError (regression found in the 2026-07-23 codex review).
    bundle = _bundle_with_claims(
        [
            ReportClaim(
                text="value=" + "9" * 400,
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id="table_1",
                        locator="rows[0]",
                        value=1470,
                    )
                ],
                referenced_datasets=["sales.csv"],
                referenced_columns=["revenue"],
            )
        ]
    )

    audit = validate_report_bundle(bundle, _pack())

    mismatches = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].numeric_details[0].number == float("inf")


def _bundle_with_claims(claims: list[ReportClaim]) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for claim in claims:
        bundle.sections[0].claims.append(claim)
    return bundle


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_sales",
            ),
            "chart_1": EvidenceArtifactSummary(
                artifact_id="chart_1",
                artifact_type="Chart",
                title="Revenue chart",
                dataset_id="ds_sales",
            ),
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_sales",
                name="sales.csv",
                row_count=10,
                column_count=3,
                columns=["region", "revenue", "discount"],
                dtypes={"region": "object", "revenue": "float64", "discount": "float64"},
            )
        ],
        # F1: pools only hold resolved payloads; table_1 resolves to 1470.
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_sales",
                title="Numeric summary",
                kind="aggregation",
                description="Revenue summary",
                rows=[{"revenue": 1470}],
            )
        ],
    )

from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef, SqlResult
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSeverity, ReportStatus
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
    EvidenceQualityIssue,
)
from eda_platform.tools.report_validator import validate_report_bundle


def test_validator_blocks_missing_required_sections() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections = bundle.sections[:-1]

    audit = validate_report_bundle(bundle, _pack())

    assert audit.status is ReportStatus.NEEDS_REVISION
    assert _codes(audit) == {"missing_required_section"}


def test_validator_blocks_claim_without_existing_evidence() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Revenue is 120.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="missing",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["revenue"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert audit.status is ReportStatus.NEEDS_REVISION
    assert "missing_evidence_artifact" in _codes(audit)


def test_validator_blocks_numeric_mismatch() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Revenue is 999.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["revenue"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert audit.status is ReportStatus.NEEDS_REVISION
    assert "numeric_mismatch" in _codes(audit)


def test_validator_blocks_specific_currency_change_or_omission() -> None:
    evidence = EvidenceRef(
        kind="stat",
        artifact_id="table_1",
        locator="rows[0]",
        value=120,
        unit="currency",
        unit_label="BRL",
        unit_reference="ISO 4217 List One@2026-01-01",
    )
    common = {
        "evidence": [evidence],
        "referenced_datasets": ["sales.csv"],
        "referenced_columns": ["revenue"],
    }

    changed = validate_report_bundle(
        _bundle_with_claim(ReportClaim(text="Revenue is 120 USD.", **common)),
        _pack(),
    )
    omitted = validate_report_bundle(
        _bundle_with_claim(ReportClaim(text="Revenue is 120.", **common)),
        _pack(),
    )

    assert "currency_unit_mismatch" in _codes(changed)
    assert "currency_unit_mismatch" in _codes(omitted)
    mismatch = next(
        finding for finding in changed.findings if finding.code == "currency_unit_mismatch"
    )
    assert mismatch.repair_mode == "llm"


def test_validator_accepts_supported_currency_prefix_or_suffix() -> None:
    evidence = EvidenceRef(
        kind="stat",
        artifact_id="table_1",
        locator="rows[0]",
        value=120,
        unit="currency",
        unit_label="BRL/order",
    )
    for text in ("AOV is 120 BRL per order.", "AOV is BRL 120 per order."):
        audit = validate_report_bundle(
            _bundle_with_claim(
                ReportClaim(
                    text=text,
                    evidence=[evidence],
                    referenced_datasets=["sales.csv"],
                    referenced_columns=["revenue"],
                )
            ),
            _pack(),
        )
        assert audit.status is ReportStatus.VALIDATED, audit.findings


def test_resolved_sql_locator_overrides_forged_inline_value() -> None:
    claim = ReportClaim(
        text="Revenue is 999999.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id="sql_1",
                locator="rows_preview[0].revenue",
                value=999999,
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    sql_result = SqlResult(
        sql="select 120 as revenue",
        columns=["revenue"],
        dtypes={"revenue": "DOUBLE"},
        rows_preview=[{"revenue": 120}],
        row_count=1,
    )

    audit = validate_report_bundle(
        _bundle_with_claim(claim),
        _pack(),
        sql_results={"sql_1": sql_result},
    )

    assert "numeric_mismatch" in _codes(audit)


def test_sql_result_unit_is_authoritative_for_currency_gate() -> None:
    claim = ReportClaim(
        text="Revenue is 120 USD.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id="sql_1",
                locator="rows_preview[0].revenue",
                value=120,
                unit="currency",
                unit_label="USD",
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    sql_result = SqlResult(
        sql="select 120 as revenue",
        columns=["revenue"],
        dtypes={"revenue": "DOUBLE"},
        units={"revenue": "BRL"},
        rows_preview=[{"revenue": 120}],
        row_count=1,
    )

    audit = validate_report_bundle(
        _bundle_with_claim(claim),
        _pack(),
        sql_results={"sql_1": sql_result},
    )

    assert "currency_unit_mismatch" in _codes(audit)


def test_invalid_sql_locator_fails_closed_instead_of_selecting_all_rows() -> None:
    claim = ReportClaim(
        text="Revenue is 120.",
        evidence=[
            EvidenceRef(
                kind="sql",
                artifact_id="sql_1",
                locator="not_a_real_locator",
                value=120,
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    sql_result = SqlResult(
        sql="select 120 as revenue",
        columns=["revenue"],
        dtypes={"revenue": "DOUBLE"},
        rows_preview=[{"revenue": 120}],
        row_count=1,
    )

    audit = validate_report_bundle(
        _bundle_with_claim(claim),
        _pack(),
        sql_results={"sql_1": sql_result},
    )

    assert "invalid_evidence_locator" in _codes(audit)


def test_currency_gate_ignores_unrelated_uppercase_acronyms() -> None:
    audit = validate_report_bundle(
        _bundle_with_claim(
            ReportClaim(
                text="The run used 10 SQL queries; revenue is 120 BRL.",
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id="table_1",
                        locator="rows[0]",
                        value=120,
                        unit="currency",
                        unit_label="BRL",
                    ),
                    EvidenceRef(
                        kind="stat",
                        artifact_id="table_1",
                        locator="rows[1]",
                        value=10,
                    ),
                ],
                referenced_datasets=["sales.csv"],
                referenced_columns=["revenue"],
            )
        ),
        _pack(),
    )

    assert audit.status is ReportStatus.VALIDATED, audit.findings


def test_validator_blocks_unknown_dataset_or_column_references() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Margin is 120.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["finance.csv"],
            referenced_columns=["margin"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert {"unknown_dataset", "unknown_column"}.issubset(_codes(audit))


def test_validator_blocks_causal_language_without_causal_evidence() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Region caused revenue to increase by 120.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["region", "revenue"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert "causal_overclaim" in _codes(audit)


def test_validator_blocks_high_risk_column_without_quality_ref() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Discount is 40.",
            evidence=[EvidenceRef(kind="stat", artifact_id="table_1", locator="rows[0]", value=40)],
            referenced_datasets=["sales.csv"],
            referenced_columns=["discount"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert "missing_quality_warning" in _codes(audit)


def test_validator_validates_supported_report() -> None:
    bundle = _bundle_with_claim(
        ReportClaim(
            text="Revenue is 120.",
            evidence=[
                EvidenceRef(
                    kind="stat",
                    artifact_id="table_1",
                    locator="rows[0]",
                    value=120,
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["revenue"],
        )
    )

    audit = validate_report_bundle(bundle, _pack())

    assert audit.status is ReportStatus.VALIDATED
    assert audit.findings == []


def _bundle_with_claim(claim: ReportClaim) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].claims.append(claim)
    for section in bundle.sections:
        section.body = section.structural_body()
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
            "qual_1": EvidenceArtifactSummary(
                artifact_id="qual_1",
                artifact_type="QualityIssueSet",
                title="Quality issues",
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
        quality_issues=[
            EvidenceQualityIssue(
                artifact_id="qual_1",
                dataset_id="ds_sales",
                severity="warn",
                code="high_missing",
                column="discount",
                message="discount has high missingness.",
                recommendation="Treat discount findings as limited.",
            )
        ],
        # F1: pools only hold resolved payloads, so table_1 must resolve for
        # the numeric assertions in this module (inline values no longer count).
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_sales",
                title="Numeric summary",
                kind="aggregation",
                description="Revenue summary",
                rows=[{"revenue": 120, "discount": 40}, {"queries": 10}],
            )
        ],
    )


def _codes(audit) -> set[str]:
    assert all(finding.severity is ReportSeverity.CRITICAL for finding in audit.findings)
    return {finding.code for finding in audit.findings}

"""F1: type-dispatched evidence resolution + three-state numeric verification.

The numeric gate must not let an unresolvable reference self-certify its own
inline value, and every claim-text number gets exactly one of
number_verified / unverified / failed.
"""

from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportBundle, ReportClaim
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.report_validator import validate_report_bundle


def test_inline_self_certified_number_is_unverified_not_a_finding() -> None:
    # R1: kind="artifact" + fabricated 9999 + self-consistent inline value,
    # artifact type is unresolvable -> unverified, and NOT numeric_mismatch.
    claim = ReportClaim(
        text="Total sessions reached 9999.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="qexec_1",
                locator="findings[0]",
                value=9999,
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    assert [f for f in audit.findings if f.code == "numeric_mismatch"] == []
    assert [status.status for status in claim.numeric_statuses] == ["unverified"]
    assert claim.numeric_rollup == "unverified"
    assert audit.numeric_unverified_claim_count == 1


def test_dataset_profile_summary_locator_verifies_row_and_column_counts() -> None:
    # R2: HR-corpus style "1470 employee records with 35 attributes".
    claim = ReportClaim(
        text="The file contains 1470 employee records with 35 attributes.",
        evidence=[
            EvidenceRef(kind="artifact", artifact_id="prof_hr", locator="summary")
        ],
        referenced_datasets=["hr.csv"],
        referenced_columns=[],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    assert [f for f in audit.findings if f.code == "numeric_mismatch"] == []
    assert [status.status for status in claim.numeric_statuses] == [
        "number_verified",
        "number_verified",
    ]
    assert claim.numeric_rollup == "number_verified"
    assert audit.numeric_unverified_claim_count == 0


def test_dispatch_follows_artifact_type_not_ref_kind() -> None:
    # A mislabeled kind must still resolve through the artifact's real type.
    claim = ReportClaim(
        text="Revenue is 120.",
        evidence=[
            EvidenceRef(kind="stat", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    validate_report_bundle(_bundle_with_claims([claim]), _pack())

    assert claim.numeric_rollup == "number_verified"


def test_fabricated_percent_with_only_raw_evidence_is_failed() -> None:
    # The claim resolves raw values, so an invented percent must not hide in
    # the unverified band: empty own-unit pool -> failed / no_evidence_values.
    claim = ReportClaim(
        text="Revenue rose 9999%.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(numeric) == 1
    detail = numeric[0].numeric_details[0]
    assert detail.number == 9999
    assert detail.is_percent is True
    assert detail.reason == "no_evidence_values"
    assert detail.evidence_values == []
    assert detail.evidence_value_count == 0
    assert [status.status for status in claim.numeric_statuses] == ["failed"]
    assert claim.numeric_rollup == "failed"
    assert audit.numeric_unverified_claim_count == 0


def test_fabricated_raw_with_only_percent_evidence_is_failed() -> None:
    # Symmetric direction: percent evidence resolves, invented raw count fails.
    claim = ReportClaim(
        text="Missing values affect 9999 employees.",
        evidence=[
            EvidenceRef(
                kind="profile_field",
                artifact_id="prof_hr",
                locator="missing_percent.Age",
            )
        ],
        referenced_datasets=["hr.csv"],
        referenced_columns=["Age"],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(numeric) == 1
    detail = numeric[0].numeric_details[0]
    assert detail.number == 9999
    assert detail.is_percent is False
    assert detail.reason == "no_evidence_values"
    assert detail.evidence_values == []
    assert detail.evidence_value_count == 0
    assert [status.status for status in claim.numeric_statuses] == ["failed"]
    assert claim.numeric_rollup == "failed"
    assert audit.numeric_unverified_claim_count == 0


def test_nonempty_pool_mismatch_is_failed_with_finding() -> None:
    claim = ReportClaim(
        text="Revenue is 999.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    audit = validate_report_bundle(_bundle_with_claims([claim]), _pack())

    numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(numeric) == 1
    detail = numeric[0].numeric_details[0]
    assert detail.number == 999
    assert detail.reason == "outside_tolerance"
    assert 120 in detail.evidence_values
    assert [status.status for status in claim.numeric_statuses] == ["failed"]
    assert claim.numeric_rollup == "failed"
    assert audit.numeric_unverified_claim_count == 0


def test_resolving_claim_never_leaves_tokens_unverified() -> None:
    # Once any pool resolves, every token is verified or failed — never
    # unverified: 999 -> outside_tolerance, 15% -> no_evidence_values.
    mixed = ReportClaim(
        text="Revenue is 999, up 15%.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    validate_report_bundle(_bundle_with_claims([mixed]), _pack())
    assert [status.status for status in mixed.numeric_statuses] == [
        "failed",
        "failed",
    ]
    assert mixed.numeric_rollup == "failed"

    partial = ReportClaim(
        text="Revenue is 120, up 15%.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    audit = validate_report_bundle(_bundle_with_claims([partial]), _pack())
    assert [status.status for status in partial.numeric_statuses] == [
        "number_verified",
        "failed",
    ]
    assert partial.numeric_rollup == "failed"
    assert audit.numeric_unverified_claim_count == 0


def test_unvalidated_claim_defaults_to_not_evaluated() -> None:
    # A bundle that never went through the validator must not masquerade as
    # an evaluated "no_numbers" claim.
    claim = ReportClaim(text="Revenue is 120.")
    assert claim.numeric_rollup == "not_evaluated"

    numberless = ReportClaim(
        text="Revenue data is fully documented.",
        evidence=[
            EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    validate_report_bundle(_bundle_with_claims([numberless]), _pack())
    assert numberless.numeric_rollup == "no_numbers"


def test_revalidation_is_idempotent() -> None:
    claim = ReportClaim(
        text="Total sessions reached 9999.",
        evidence=[
            EvidenceRef(
                kind="artifact",
                artifact_id="qexec_1",
                locator="findings[0]",
                value=9999,
            )
        ],
        referenced_datasets=["sales.csv"],
        referenced_columns=["revenue"],
    )
    bundle = _bundle_with_claims([claim])
    validate_report_bundle(bundle, _pack())
    first = (list(claim.numeric_statuses), claim.numeric_rollup)
    audit = validate_report_bundle(bundle, _pack())
    assert (list(claim.numeric_statuses), claim.numeric_rollup) == first
    assert audit.numeric_unverified_claim_count == 1


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
            "qexec_1": EvidenceArtifactSummary(
                artifact_id="qexec_1",
                artifact_type="QuestionExecutionResult",
                title="Question execution",
                dataset_id="ds_sales",
            ),
            "prof_hr": EvidenceArtifactSummary(
                artifact_id="prof_hr",
                artifact_type="DatasetProfile",
                title="hr.csv",
                dataset_id="ds_hr",
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
            ),
            EvidenceDataset(
                artifact_id="prof_hr",
                dataset_id="ds_hr",
                name="hr.csv",
                row_count=1470,
                column_count=35,
                columns=["Age", "Attrition"],
                dtypes={"Age": "int64", "Attrition": "object"},
                missing_percent={"Age": 12.5},
            ),
        ],
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_sales",
                title="Numeric summary",
                kind="aggregation",
                description="Revenue summary",
                rows=[{"revenue": 120}],
            )
        ],
    )

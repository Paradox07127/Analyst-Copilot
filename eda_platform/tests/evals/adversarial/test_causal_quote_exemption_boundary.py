"""Adversarial boundary cases for the quoted-span causal exemption.

M4 plan §11 queued "引号内因果文本豁免边界" as an eval-week adversarial case:
``report_validator`` exempts double-quoted spans in *claim* text from the
causal-language gate (a verbatim quoted question is a citation, not the
report's own assertion). These tests pin the boundary so the exemption can
never quietly widen into a causal-overclaim bypass:

- quoted causal text alone is exempt;
- causal text outside the quotes is still blocked, even when a quoted span
  is present in the same claim;
- an unbalanced quote strips nothing (conservative default -> blocked);
- section *bodies* get no exemption at all (deliberate: bodies must stay
  qualitative).
"""

from __future__ import annotations

from eda_platform.schemas.artifacts import EvidenceRef
from eda_platform.schemas.reports import ReportBundle, ReportClaim, ReportSeverity
from eda_platform.tools.evidence import (
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidencePack,
)
from eda_platform.tools.report_validator import validate_report_bundle


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_sales",
            )
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_sales",
                name="sales.csv",
                row_count=10,
                column_count=2,
                columns=["region", "revenue"],
                dtypes={"region": "object", "revenue": "float64"},
            )
        ],
        quality_issues=[],
    )


def _bundle_with_claim_text(text: str) -> ReportBundle:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].claims.append(
        ReportClaim(
            text=text,
            evidence=[
                EvidenceRef(
                    kind="stat", artifact_id="table_1", locator="rows[0]", value=120
                )
            ],
            referenced_datasets=["sales.csv"],
            referenced_columns=["region", "revenue"],
        )
    )
    for section in bundle.sections:
        section.body = section.structural_body()
    return bundle


def _codes(bundle: ReportBundle) -> set[str]:
    audit = validate_report_bundle(bundle, _pack())
    assert all(finding.severity is ReportSeverity.CRITICAL for finding in audit.findings)
    return {finding.code for finding in audit.findings}


def test_causal_language_fully_inside_quotes_is_exempt() -> None:
    bundle = _bundle_with_claim_text(
        'The executed question was "Which region drives revenue?" '
        "and the measured revenue is 120."
    )
    assert "causal_overclaim" not in _codes(bundle)


def test_causal_language_outside_quotes_is_still_blocked() -> None:
    bundle = _bundle_with_claim_text("Region caused revenue to reach 120.")
    assert "causal_overclaim" in _codes(bundle)


def test_exemption_does_not_leak_past_the_quoted_span() -> None:
    bundle = _bundle_with_claim_text(
        'We ran "revenue by region" and conclude the region drives revenue to 120.'
    )
    assert "causal_overclaim" in _codes(bundle)


def test_unbalanced_quote_strips_nothing_and_stays_blocked() -> None:
    bundle = _bundle_with_claim_text(
        'An unterminated citation "which region drives revenue, value 120.'
    )
    assert "causal_overclaim" in _codes(bundle)


def test_section_body_gets_no_quoted_exemption() -> None:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    for section in bundle.sections:
        section.body = section.structural_body()
    bundle.sections[0].body = 'Context: "discount drives revenue" per the operator.'
    codes = _codes(bundle)
    assert "causal_overclaim" in codes

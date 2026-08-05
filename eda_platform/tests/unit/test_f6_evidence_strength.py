"""F6 — evidence-strength three-tier confidence labels.

The confidence_label axis is re-based from "did the LLM propose the question"
to evidence strength (analysis-v3 F6):

- ref strength from the persisted artifact type (resolved refs only):
  DatasetProfile / Table / QualityIssueSet / StatTestResult -> strong,
  SqlResult / ModelCard -> indicative;
- claim tier = strongest resolved ref, zero resolved refs -> exploratory;
- bundle verdict = strong-claim ratio against STRONG_RATIO_CUT (rejected
  logic untouched);
- legacy labels ("verified" / "low_relevance") stay loadable and render as
  before; new code writes only the three new tiers.
"""

from __future__ import annotations

import math
from typing import Any

from eda_platform.schemas.artifacts import EvidenceRef, SqlResult
from eda_platform.schemas.reports import (
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportSeverity,
    ReportStatus,
    ReportValidationFinding,
)
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidenceDataset,
    EvidenceModelCard,
    EvidencePack,
    EvidenceStatTest,
)
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html
from eda_platform.tools.report_validator import (
    STRONG_RATIO_CUT,
    apply_semantic_gate,
    evidence_strength_label,
    strong_ratio_verdict,
)


def _pack() -> EvidencePack:
    return EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "prof_1": EvidenceArtifactSummary(
                artifact_id="prof_1",
                artifact_type="DatasetProfile",
                title="Profile",
                dataset_id="ds_sales",
            ),
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                dataset_id="ds_sales",
            ),
            "quality_1": EvidenceArtifactSummary(
                artifact_id="quality_1",
                artifact_type="QualityIssueSet",
                title="Quality issues",
                dataset_id="ds_sales",
            ),
            "stat_1": EvidenceArtifactSummary(
                artifact_id="stat_1",
                artifact_type="StatTestResult",
                title="ANOVA",
                dataset_id="ds_sales",
            ),
            "model_1": EvidenceArtifactSummary(
                artifact_id="model_1",
                artifact_type="ModelCard",
                title="Baseline model",
                dataset_id="ds_sales",
            ),
        },
        datasets=[
            EvidenceDataset(
                artifact_id="prof_1",
                dataset_id="ds_sales",
                name="sales.csv",
                row_count=1470,
                column_count=35,
                columns=["revenue"],
                dtypes={"revenue": "float64"},
            )
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
        stat_tests=[
            EvidenceStatTest(
                artifact_id="stat_1",
                dataset_id="ds_sales",
                test_type="one_way_anova",
                statistic=8.1,
                p_value=0.002,
                effect_size=0.2,
                sample_size=120,
            )
        ],
        model_cards=[
            EvidenceModelCard(
                artifact_id="model_1",
                dataset_id="ds_sales",
                task_type="classification",
                target_column="churn",
                split_strategy="holdout",
                model_type="logreg",
                metrics={"auc": 0.81},
            )
        ],
    )


def _sql_results() -> dict[str, SqlResult]:
    return {
        "sql_1": SqlResult(
            sql="SELECT region, revenue FROM t",
            columns=["region", "revenue"],
            dtypes={"region": "VARCHAR", "revenue": "DOUBLE"},
            rows_preview=[{"region": "East", "revenue": 40.0}],
            row_count=1,
        )
    }


def _claim(claim_id: str, refs: list[EvidenceRef], text: str = "Revenue moves.") -> ReportClaim:
    return ReportClaim(id=claim_id, text=text, evidence=refs)


_PROFILE_REF = EvidenceRef(kind="stat", artifact_id="prof_1", locator="rows")
_SQL_REF = EvidenceRef(kind="table", artifact_id="sql_1", locator="rows[0].revenue")
_GHOST_REF = EvidenceRef(kind="artifact", artifact_id="ghost_1", locator="", value=9999)


# --------------------------------------------------------------------------- #
# R1 — the three tiers exist as schema values.
# --------------------------------------------------------------------------- #
def test_three_tier_labels_are_valid_schema_values() -> None:
    for label in ("strong", "indicative", "exploratory"):
        claim = ReportClaim(text="x", confidence_label=label)  # type: ignore[arg-type]
        assert claim.confidence_label == label


# --------------------------------------------------------------------------- #
# R2 — claim tier = strongest resolved ref; zero resolved refs -> exploratory.
# --------------------------------------------------------------------------- #
def test_mixed_profile_and_sql_evidence_is_strong() -> None:
    claim = _claim("mixed", [_PROFILE_REF, _SQL_REF])
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "strong"


def test_pure_sql_evidence_is_indicative() -> None:
    claim = _claim("sql_only", [_SQL_REF])
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "indicative"


def test_zero_resolved_refs_is_exploratory() -> None:
    # Inline evidence.value never counts: an unresolvable ref proves nothing.
    claim = _claim("ghost", [_GHOST_REF])
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "exploratory"


def test_no_evidence_at_all_is_exploratory() -> None:
    claim = _claim("bare", [])
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "exploratory"


def test_type_dispatch_strength_map() -> None:
    # QualityIssueSet numbers live in prose (unresolvable by design) but the
    # artifact is a full-table deterministic scan: an artifact_index hit
    # counts as resolved (analysis-v3 F6 relaxation).
    cases = {
        "quality_1": "strong",
        "stat_1": "strong",
        "table_1": "strong",
        "model_1": "indicative",
    }
    for artifact_id, expected in cases.items():
        claim = _claim(
            artifact_id, [EvidenceRef(kind="artifact", artifact_id=artifact_id, locator="")]
        )
        label = evidence_strength_label(
            claim, evidence_pack=_pack(), sql_results=_sql_results()
        )
        assert label == expected, artifact_id


# --------------------------------------------------------------------------- #
# R2b — number binding (cross-review fix 1): when the claim asserts numbers,
# only refs that actually verified >=1 token contribute strength.
# --------------------------------------------------------------------------- #
def test_number_verified_only_by_sql_is_indicative_despite_profile_ref() -> None:
    # The profile ref resolves 1470 (rows) which supports nothing in the text;
    # the only verifiable figure (40.0) comes from the sql cell, so an
    # unrelated strong ref must not launder the claim to strong.
    claim = _claim(
        "sql_verified",
        [_PROFILE_REF, _SQL_REF],
        text="Average revenue reached 40.0 in the east region.",
    )
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "indicative"


def test_number_verified_by_profile_keeps_strong() -> None:
    # Control probe: the same ref mix stays strong when the strong ref is the
    # one that verified the asserted number.
    claim = _claim(
        "prof_verified",
        [_PROFILE_REF, _SQL_REF],
        text="The dataset has 1470 rows.",
    )
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "strong"


def test_unverified_numbers_are_exploratory_even_with_quality_ref() -> None:
    # hr_attrition c4 class: the claim asserts a figure no ref resolves; a
    # QualityIssueSet artifact_index hit must not back an unsupported number.
    claim = _claim(
        "unverified_num",
        [EvidenceRef(kind="artifact", artifact_id="quality_1", locator="")],
        text="About 12% of employee rows are affected.",
    )
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "exploratory"


def test_numberless_claim_keeps_strongest_resolved_ref() -> None:
    # Control probe: qualitative conclusions are still backed by the strongest
    # resolved ref (QualityIssueSet index hit included).
    claim = _claim(
        "qualitative",
        [EvidenceRef(kind="artifact", artifact_id="quality_1", locator=""), _SQL_REF],
        text="Data quality issues cluster in the revenue column.",
    )
    label = evidence_strength_label(
        claim, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert label == "strong"


# --------------------------------------------------------------------------- #
# R3 — bundle verdict flips on the strong ratio around STRONG_RATIO_CUT.
# --------------------------------------------------------------------------- #
def _bundle_with_claims(claims: list[ReportClaim]) -> tuple[ReportBundle, ReportAudit]:
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].claims.extend(claims)
    return bundle, ReportAudit(status=ReportStatus.VALIDATED)


def _mixed_strength_claims(strong: int, total: int) -> list[ReportClaim]:
    claims = [_claim(f"strong_{i}", [_PROFILE_REF]) for i in range(strong)]
    claims.extend(
        _claim(f"weak_{i}", [_GHOST_REF]) for i in range(total - strong)
    )
    return claims


def test_verdict_flips_at_strong_ratio_cut() -> None:
    total = 5
    at_cut = math.ceil(STRONG_RATIO_CUT * total)

    bundle, audit = _bundle_with_claims(_mixed_strength_claims(at_cut, total))
    outcome = apply_semantic_gate(
        bundle, audit, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert outcome.verdict == "pass"
    assert audit.gate_verdict == "pass"

    bundle, audit = _bundle_with_claims(_mixed_strength_claims(at_cut - 1, total))
    outcome = apply_semantic_gate(
        bundle, audit, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert outcome.verdict == "degraded"
    assert audit.gate_verdict == "degraded"


def test_gate_writes_only_new_tier_values_and_counts_them() -> None:
    claims = [
        _claim("s", [_PROFILE_REF]),
        _claim("i", [_SQL_REF]),
        _claim("e", [_GHOST_REF]),
    ]
    bundle, audit = _bundle_with_claims(claims)

    outcome = apply_semantic_gate(
        bundle, audit, evidence_pack=_pack(), sql_results=_sql_results()
    )

    assert [claim.confidence_label for claim in claims] == [
        "strong",
        "indicative",
        "exploratory",
    ]
    assert (outcome.strong_claims, outcome.indicative_claims, outcome.exploratory_claims) == (
        1,
        1,
        1,
    )
    # 1/3 strong < cut -> degraded by ratio, not by any per-claim flag.
    assert outcome.verdict == "degraded"
    assert outcome.degraded_claim_count == 0


def test_strong_ratio_verdict_helper_boundaries() -> None:
    # Cross-review fix 5: an empty report proves nothing -> never a vacuous pass.
    assert strong_ratio_verdict(0, 0) == "degraded"
    assert strong_ratio_verdict(3, 5, cut=0.6) == "pass"
    assert strong_ratio_verdict(2, 4, cut=0.6) == "degraded"


def test_gate_denominator_excludes_legacy_qfocus_claims() -> None:
    # Cross-review fix 4: legacy pre-F4 qfocus_* claims leave the strong-ratio
    # denominator (scoreboard alignment) but still get claim-level labels.
    claims = [
        _claim("strong_1", [_PROFILE_REF]),
        _claim("strong_2", [_PROFILE_REF]),
        _claim("qfocus_legacy_1", [_GHOST_REF]),
        _claim("qfocus_legacy_2", [_GHOST_REF]),
        _claim("qfocus_legacy_3", [_GHOST_REF]),
    ]
    bundle, audit = _bundle_with_claims(claims)

    outcome = apply_semantic_gate(
        bundle, audit, evidence_pack=_pack(), sql_results=_sql_results()
    )

    # 2/2 strong, not 2/5.
    assert outcome.verdict == "pass"
    assert audit.gate_verdict == "pass"
    assert (
        outcome.strong_claims,
        outcome.indicative_claims,
        outcome.exploratory_claims,
    ) == (2, 0, 0)
    assert claims[2].confidence_label == "exploratory"


def test_semantic_gate_replay_is_idempotent_and_clears_stale_state() -> None:
    # Cross-review fix 3: re-gating an already-gated bundle must reset the
    # gate's own claim state and audit products instead of contradicting or
    # duplicating them.
    claim = _claim("legacy", [_PROFILE_REF])
    claim.confidence_label = "low_relevance"
    claim.gate_verdict = "degraded"
    claim.gate_flags = ["low_impact_column:ghost_col"]
    claim.time_boundary_flag = "partial_periods:1999-01"
    bundle, audit = _bundle_with_claims([claim])
    audit.findings.append(
        ReportValidationFinding(
            severity=ReportSeverity.WARN,
            code="semantic_degraded",
            message=(
                "Claim published degraded with semantic gate flag(s): "
                "low_impact_column:ghost_col"
            ),
            section_title=bundle.sections[0].title,
            claim_id="legacy",
        )
    )
    audit.semantic_notes.extend(
        [
            "Evidence strength: 0 strong / 1 indicative / 0 exploratory claim(s); "
            "verdict 'degraded' at strong-ratio cut 60%.",
            "Semantic soft gate: 1 claim(s) published degraded with structured "
            "gate flags (not disclaimers).",
        ]
    )

    apply_semantic_gate(bundle, audit, evidence_pack=_pack(), sql_results=_sql_results())
    first = (bundle.model_dump(mode="json"), audit.model_dump(mode="json"))
    apply_semantic_gate(bundle, audit, evidence_pack=_pack(), sql_results=_sql_results())
    second = (bundle.model_dump(mode="json"), audit.model_dump(mode="json"))

    assert first == second
    assert claim.gate_flags == []  # no stale flag residue
    assert claim.gate_verdict == "pass"
    assert claim.time_boundary_flag == ""
    assert claim.confidence_label == "strong"
    strength_notes = [
        note for note in audit.semantic_notes if note.startswith("Evidence strength:")
    ]
    assert len(strength_notes) == 1
    assert not any(f.code == "semantic_degraded" for f in audit.findings)


def test_hard_gate_rejection_still_wins_over_ratio() -> None:
    bundle, audit = _bundle_with_claims(_mixed_strength_claims(5, 5))
    audit.findings.append(
        # Simulate a hard-gate CRITICAL surviving into the gate call.
        __import__(
            "eda_platform.schemas.reports", fromlist=["ReportValidationFinding"]
        ).ReportValidationFinding(
            severity="critical",  # type: ignore[arg-type]
            code="numeric_mismatch",
            message="x",
        )
    )
    outcome = apply_semantic_gate(
        bundle, audit, evidence_pack=_pack(), sql_results=_sql_results()
    )
    assert outcome.verdict == "rejected"
    assert audit.gate_verdict == "rejected"


# --------------------------------------------------------------------------- #
# R4 — legacy bundles load and render unchanged; new tiers render prefixes.
# --------------------------------------------------------------------------- #
def _legacy_bundle_payload() -> dict[str, Any]:
    return {
        "project_id": "p",
        "session_id": "r",
        "sections": [
            {
                "title": "Key EDA Insights",
                "claims": [
                    {
                        "id": "c_verified",
                        "text": "Legacy verified claim.",
                        "confidence_label": "verified",
                    },
                    {
                        "id": "c_lowrel",
                        "text": "Legacy low relevance claim.",
                        "confidence_label": "low_relevance",
                    },
                ],
            }
        ],
    }


def test_legacy_confidence_labels_load_and_render_unchanged() -> None:
    bundle = ReportBundle.model_validate(_legacy_bundle_payload())
    markdown = report_bundle_to_markdown(bundle)
    assert "- Legacy verified claim." in markdown  # no prefix for "verified"
    assert "[Low relevance] Legacy low relevance claim." in markdown
    # Cross-review fix 6: HTML matches the markdown legacy rendering.
    html = export_report_html(bundle)
    assert "<li>Legacy verified claim.</li>" in html
    assert "<li>[Low relevance] Legacy low relevance claim.</li>" in html


def test_exporters_render_strength_prefixes() -> None:
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    bundle.sections[0].claims.extend(
        [
            ReportClaim(id="s", text="Strong claim.", confidence_label="strong"),
            ReportClaim(id="i", text="Indicative claim.", confidence_label="indicative"),
            ReportClaim(id="e", text="Exploratory claim.", confidence_label="exploratory"),
        ]
    )

    markdown = report_bundle_to_markdown(bundle)
    assert "- Strong claim." in markdown  # strong carries no prefix
    assert "- [Indicative] Indicative claim." in markdown
    assert "- [Exploratory — hypothesis-generating] Exploratory claim." in markdown

    html = export_report_html(bundle)
    assert "<li>Strong claim.</li>" in html
    assert "<li>[Indicative] Indicative claim.</li>" in html
    assert "<li>[Exploratory — hypothesis-generating] Exploratory claim.</li>" in html


# --------------------------------------------------------------------------- #
# Registry-authored SQL: the cap exists because a SqlResult cannot prove its own
# coverage. When the platform wrote the query, the platform knows the coverage.
# --------------------------------------------------------------------------- #
def test_registry_written_sql_is_strong() -> None:
    """A domain metric's SQL is a registry template, not a model's plan.

    `_time_coverage_sql` and friends aggregate a named table the platform chose;
    that is the same footing as the profiler scan, which is already strong. The
    cap was blanket, so these read as "[Indicative]" while stating exact facts.
    """
    claim = _claim("sql_only", [_SQL_REF])
    label = evidence_strength_label(
        claim,
        evidence_pack=_pack(),
        sql_results=_sql_results(),
        platform_sql_ids={_SQL_REF.artifact_id},
    )
    assert label == "strong"


def test_a_planned_query_is_not_promoted_by_the_exception() -> None:
    """The set is the whole permission: an id outside it stays indicative."""
    claim = _claim("sql_only", [_SQL_REF])
    label = evidence_strength_label(
        claim,
        evidence_pack=_pack(),
        sql_results=_sql_results(),
        platform_sql_ids={"sql_some_other_result"},
    )
    assert label == "indicative"

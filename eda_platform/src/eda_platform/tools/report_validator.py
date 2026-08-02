from __future__ import annotations

import math
import operator
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from eda_platform.core.claim_language import REPORT_BODY_CAUSAL_TERMS
from eda_platform.core.column_roles import ColumnRoleSet
from eda_platform.schemas.artifacts import Artifact, ArtifactType, EvidenceRef, SqlResult
from eda_platform.schemas.reports import (
    GateVerdict,
    NumericEvidenceSource,
    NumericMismatchDetail,
    NumericRollup,
    NumericSourceValue,
    NumericTokenStatus,
    ReportAudit,
    ReportBundle,
    ReportClaim,
    ReportSeverity,
    ReportStatus,
    ReportValidationFinding,
    required_report_sections,
)
from eda_platform.tools.evidence import EvidencePack
from eda_platform.tools.time_boundary import (
    DEFAULT_MIN_BUCKETS,
    DEFAULT_PARTIAL_RATIO,
    TimeBoundaryAssessment,
    assess_sql_result,
    split_edge_values,
)

# Thousands separators are part of the number: without the group-of-3 branch
# "1,470" read as 1 and 470, so any rendered claim looked unsupported. The
# scientific branch is first so "1e-10" is one token, not a truncated "1".
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.-])-?\d+(?:\.\d+)?[eE][-+]?\d+"
    r"|(?<![\w.-])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\w.-])-?\d+(?:\.\d+)?%?"
)
_ROW_LOCATOR_PATTERN = re.compile(
    r"^(?:rows|rows_preview)(?:\[(?P<index>\d+)\])?"
    r"(?:\.(?P<field>[A-Za-z_][\w]*))?$"
)
# A comparison operator immediately before a number token marks an inequality
# assertion ("p < 0.0001"): verified against the bound, not by equality.
_THRESHOLD_PREFIX_PATTERN = re.compile(r"(<=|>=|<|>)\s*$")
_THRESHOLD_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
_CAUSAL_TERMS = REPORT_BODY_CAUSAL_TERMS
# F3: sections whose claims are expected to carry at least one verified number.
# Executive Summary / Dataset Overview added 2026-07-23 (cross-review): the §5.1
# HR sanitization was published in Executive Summary, outside the original list.
QUANTITATIVE_SECTIONS = frozenset(
    {
        "Executive Summary",
        "Dataset Overview",
        "Business Findings",
        "Key EDA Insights",
        "Data Quality Findings",
    }
)
_HIGH_RISK_QUALITY_CODES = {"high_missing", "mixed_type_string", "outlier_detected"}
_QUOTED_SPAN_PATTERN = re.compile(r"\"[^\"]*\"")
_CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?:"
    r"(?P<prefix_code>[A-Z]{3})\s+(?P<prefix_value>-?\d+(?:\.\d+)?)"
    r"|"
    r"(?P<suffix_value>-?\d+(?:\.\d+)?)\s+(?P<suffix_code>[A-Z]{3})"
    r"(?:/order|\s+per\s+order)?"
    r")"
)


def validate_report_bundle(
    bundle: ReportBundle,
    evidence_pack: EvidencePack,
    *,
    numeric_tolerance: float = 0.01,
    sql_results: dict[str, SqlResult] | None = None,
) -> ReportAudit:
    sql_results = sql_results or {}
    findings: list[ReportValidationFinding] = []
    findings.extend(_section_findings(bundle))
    findings.extend(_body_findings(bundle))
    known_artifact_ids = set(evidence_pack.artifact_index) | set(sql_results)
    known_dataset_refs = {
        ref
        for dataset in evidence_pack.datasets
        for ref in (dataset.dataset_id, dataset.name)
    }
    known_columns = {column for dataset in evidence_pack.datasets for column in dataset.columns}
    high_risk_columns = {
        issue.column
        for issue in evidence_pack.quality_issues
        if issue.column and issue.code in _HIGH_RISK_QUALITY_CODES
    }

    numeric_unverified_claims = 0
    quantitative_coverage_gaps = 0
    for section in bundle.sections:
        for claim_index, claim in enumerate(section.claims):
            claim_id = claim.id or f"{section.title}:{claim_index}"
            statuses, numeric_details = _numeric_gate_outcome(
                claim,
                evidence_pack=evidence_pack,
                numeric_tolerance=numeric_tolerance,
                sql_results=sql_results,
            )
            # Written directly onto the claim (same pattern as gate_flags in
            # apply_semantic_gate); re-validation overwrites idempotently.
            claim.numeric_statuses = statuses
            claim.numeric_rollup = _numeric_rollup(statuses)
            if claim.numeric_rollup == "unverified":
                numeric_unverified_claims += 1
            # F3 coverage gap: quantitative-section claim with zero verified
            # numbers (no numbers at all, or all unverified/failed). Disclosure
            # only — never a finding, never a rewrite trigger. Platform
            # deterministic fallback claims are exempt until QualityIssue is
            # structured (analysis-v3 §11.3).
            verified_numeric_count = sum(
                1 for status in statuses if status.status == "number_verified"
            )
            claim.quantitative_coverage_gap = (
                section.title in QUANTITATIVE_SECTIONS
                and verified_numeric_count == 0
                and not claim.deterministic_source
            )
            if claim.quantitative_coverage_gap:
                quantitative_coverage_gaps += 1
            findings.extend(
                _claim_findings(
                    claim,
                    section_title=section.title,
                    claim_id=claim_id,
                    known_artifact_ids=known_artifact_ids,
                    known_dataset_refs=known_dataset_refs,
                    known_columns=known_columns,
                    high_risk_columns=high_risk_columns,
                    numeric_details=numeric_details,
                    numeric_tolerance=numeric_tolerance,
                    sql_results=sql_results,
                )
            )

    status = ReportStatus.NEEDS_REVISION if findings else ReportStatus.VALIDATED
    return ReportAudit(
        status=status,
        findings=findings,
        numeric_unverified_claim_count=numeric_unverified_claims,
        quantitative_coverage_gap_count=quantitative_coverage_gaps,
    )


def _section_findings(bundle: ReportBundle) -> list[ReportValidationFinding]:
    section_titles = [section.title for section in bundle.sections]
    findings: list[ReportValidationFinding] = []
    for title in required_report_sections():
        if title not in section_titles:
            findings.append(
                _critical(
                    "missing_required_section",
                    f"Required report section is missing: {title}",
                    section_title=title,
                )
            )
    return findings


def _body_findings(bundle: ReportBundle) -> list[ReportValidationFinding]:
    """Only structural display prose may live outside checked claims."""
    findings: list[ReportValidationFinding] = []
    for section in bundle.sections:
        if not section.body:
            continue
        if section.body != section.structural_body():
            findings.append(
                _critical(
                    "unsupported_section_body",
                    "Section body is presentation-only; move factual statements "
                    "into evidence-backed claims.",
                    section_title=section.title,
                )
            )
        if _contains_causal_language(section.body):
            findings.append(
                _critical(
                    "causal_overclaim",
                    "Section body uses causal language; causal statements require "
                    "evidence and must be claims.",
                    section_title=section.title,
                )
            )
    return findings


def _claim_findings(
    claim: ReportClaim,
    *,
    section_title: str,
    claim_id: str,
    known_artifact_ids: set[str],
    known_dataset_refs: set[str],
    known_columns: set[str],
    high_risk_columns: set[str],
    numeric_details: list[NumericMismatchDetail],
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> list[ReportValidationFinding]:
    findings: list[ReportValidationFinding] = []
    if not claim.evidence:
        findings.append(
            _critical(
                "missing_evidence",
                "Claim has no evidence references.",
                section_title=section_title,
                claim_id=claim_id,
            )
        )
    for evidence in claim.evidence:
        if evidence.artifact_id and evidence.artifact_id not in known_artifact_ids:
            findings.append(
                _critical(
                    "missing_evidence_artifact",
                    f"Evidence artifact does not exist: {evidence.artifact_id}",
                    section_title=section_title,
                    claim_id=claim_id,
                )
            )
        if (
            evidence.artifact_id in sql_results
            and not _valid_sql_result_locator(evidence, sql_results[evidence.artifact_id])
        ):
            findings.append(
                _critical(
                    "invalid_evidence_locator",
                    f"Evidence locator does not resolve in SQL result: {evidence.locator}",
                    section_title=section_title,
                    claim_id=claim_id,
                )
            )
    for dataset in claim.referenced_datasets:
        if dataset not in known_dataset_refs:
            findings.append(
                _critical(
                    "unknown_dataset",
                    f"Claim references an unknown dataset: {dataset}",
                    section_title=section_title,
                    claim_id=claim_id,
                )
            )
    for column in claim.referenced_columns:
        if column not in known_columns:
            findings.append(
                _critical(
                    "unknown_column",
                    f"Claim references an unknown column: {column}",
                    section_title=section_title,
                    claim_id=claim_id,
                )
            )
    if numeric_details:
        findings.append(
            _critical(
                "numeric_mismatch",
                _numeric_mismatch_message(numeric_details),
                section_title=section_title,
                claim_id=claim_id,
                numeric_details=numeric_details,
            )
        )
    if _has_currency_unit_mismatch(
        claim,
        numeric_tolerance=numeric_tolerance,
        sql_results=sql_results,
    ):
        findings.append(
            _critical(
                "currency_unit_mismatch",
                "Claim text omits or changes a specific currency unit carried by evidence.",
                section_title=section_title,
                claim_id=claim_id,
            )
        )
    # Double-quoted spans are citations (e.g. a verbatim question the agent ran),
    # not the report's own causal assertion, so exclude them from the causal check.
    if _contains_causal_language(_strip_quoted_spans(claim.text)):
        findings.append(
            _critical(
                "causal_overclaim",
                "Claim uses causal language without causal evidence.",
                section_title=section_title,
                claim_id=claim_id,
            )
        )
    risky_refs = high_risk_columns.intersection(claim.referenced_columns)
    if risky_refs and not claim.quality_issue_refs:
        columns = ", ".join(sorted(risky_refs))
        findings.append(
            _critical(
                "missing_quality_warning",
                f"Claim references high-risk column(s) without quality warning: {columns}",
                section_title=section_title,
                claim_id=claim_id,
            )
        )
    return findings


def _has_numeric_mismatch(
    claim: ReportClaim,
    *,
    evidence_pack: EvidencePack,
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> bool:
    """Thin wrapper kept for external probe scripts."""
    return bool(
        _numeric_mismatch_details(
            claim,
            evidence_pack=evidence_pack,
            numeric_tolerance=numeric_tolerance,
            sql_results=sql_results,
        )
    )


_NUMERIC_DETAIL_VALUE_CAP = 20
_NUMERIC_SOURCE_VALUE_CAP = 8


def _numeric_mismatch_details(
    claim: ReportClaim,
    *,
    evidence_pack: EvidencePack,
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> list[NumericMismatchDetail]:
    """Thin wrapper kept for external probe scripts."""
    return _numeric_gate_outcome(
        claim,
        evidence_pack=evidence_pack,
        numeric_tolerance=numeric_tolerance,
        sql_results=sql_results,
    )[1]


def _numeric_token_statuses(
    claim: ReportClaim,
    *,
    evidence_pack: EvidencePack,
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> list[NumericTokenStatus]:
    """Per-token verification states in `_numbers_from_text` order."""
    return _numeric_gate_outcome(
        claim,
        evidence_pack=evidence_pack,
        numeric_tolerance=numeric_tolerance,
        sql_results=sql_results,
    )[0]


def _numeric_gate_outcome(
    claim: ReportClaim,
    *,
    evidence_pack: EvidencePack,
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> tuple[list[NumericTokenStatus], list[NumericMismatchDetail]]:
    """Three-state numeric gate: verified / unverified / failed.

    Matching is value-tiered (F2): each pool value carries a policy — "exact"
    for integer counts, "rounded" for continuous stats (half-ulp of the token's
    displayed decimals). A token preceded by </>/<=/>= is an inequality, but
    only threshold-eligible values (stat p_value/statistic/effect_size
    locators) in the token's own unit pool may satisfy it; without eligible
    values the token is judged by the plain exact/rounded rules, so a bare
    "< 9999" cannot wash a fabrication through any resolvable pool.
    `numeric_tolerance` is no longer used here (kept for signature
    compatibility; the currency and time-boundary gates still take it).

    "Unverified" is reserved for claims where no ref resolves any value at all
    (cannot verify anything). Once the claim resolves values in either unit, a
    token whose own-unit pool is empty is a fabrication, not a resolution gap,
    so it fails with reason "no_evidence_values".
    """
    tokens = _numeric_tokens_from_text(claim.text)
    if not tokens:
        return [], []
    evidence_values, sources = _numeric_evidence_values_with_sources(
        claim, evidence_pack, sql_results
    )
    # Separate percentages from raw counts so a "89% missing" figure cannot be
    # "washed" by unrelated raw evidence like "89 rows" (and vice versa).
    percent_pool = [
        (value, policy, eligible)
        for value, unit, policy, eligible in evidence_values
        if unit == "percent"
    ]
    raw_pool = [
        (value, policy, eligible)
        for value, unit, policy, eligible in evidence_values
        if unit != "percent"
    ]
    claim_resolves_nothing = not evidence_values
    statuses: list[NumericTokenStatus] = []
    details: list[NumericMismatchDetail] = []
    for token in tokens:
        pool = percent_pool if token.is_percent else raw_pool
        eligible_values = (
            [value for value, _policy, eligible in pool if eligible]
            if token.threshold_op is not None
            else []
        )
        if claim_resolves_nothing:
            status: Literal["number_verified", "unverified", "failed"] = "unverified"
        elif eligible_values:
            if _satisfies_threshold(token, eligible_values):
                status = "number_verified"
            else:
                status = "failed"
                details.append(
                    NumericMismatchDetail(
                        number=token.value,
                        is_percent=token.is_percent,
                        reason="outside_tolerance",
                        evidence_values=sorted(set(eligible_values))[
                            :_NUMERIC_DETAIL_VALUE_CAP
                        ],
                        evidence_value_count=len(eligible_values),
                        sources=sources,
                    )
                )
        elif not pool:
            status = "failed"
            details.append(
                NumericMismatchDetail(
                    number=token.value,
                    is_percent=token.is_percent,
                    reason="no_evidence_values",
                    evidence_values=[],
                    evidence_value_count=0,
                    sources=sources,
                )
            )
        elif any(
            _value_supports_token(token, value, policy) for value, policy, _eligible in pool
        ):
            status = "number_verified"
        else:
            status = "failed"
            details.append(
                NumericMismatchDetail(
                    number=token.value,
                    is_percent=token.is_percent,
                    reason="outside_tolerance",
                    evidence_values=sorted({value for value, _policy, _eligible in pool})[
                        :_NUMERIC_DETAIL_VALUE_CAP
                    ],
                    evidence_value_count=len(pool),
                    sources=sources,
                )
            )
        statuses.append(
            NumericTokenStatus(
                number=token.value, is_percent=token.is_percent, status=status
            )
        )
    return statuses, details


def _value_supports_token(token: _NumericToken, value: float, policy: str) -> bool:
    if policy == "exact":
        # Float == is exact only below 2^53; local dataset cardinalities sit
        # many orders of magnitude under that, so the limit is unreachable here.
        return value == token.value
    # rounded: half-ulp of the token's last displayed digit (comma separators
    # never count as decimals). The old global ±1% window and its 0.01
    # absolute floor are gone (analysis-v3 §5.2/§5.7); the slack is relative
    # so deep-decimal tokens are not widened by an absolute epsilon.
    return abs(value - token.value) <= 0.5 * 10.0 ** (-token.decimals) * (1 + 1e-9)


def _satisfies_threshold(token: _NumericToken, values: Iterable[float]) -> bool:
    compare = _THRESHOLD_OPS[token.threshold_op or "<"]
    return any(compare(value, token.value) for value in values)


def _numeric_rollup(statuses: list[NumericTokenStatus]) -> NumericRollup:
    if not statuses:
        return "no_numbers"
    if any(status.status == "failed" for status in statuses):
        return "failed"
    if any(status.status == "unverified" for status in statuses):
        return "unverified"
    return "number_verified"


def _numeric_mismatch_message(details: list[NumericMismatchDetail]) -> str:
    # Per-number pool sizes: percent and raw pools are checked separately, so a
    # merged total would misstate what each number was compared against.
    numbers = ", ".join(
        f"{_format_claim_number(detail.number, detail.is_percent)}"
        f" [{'percent' if detail.is_percent else 'raw'} pool: {detail.evidence_value_count}]"
        for detail in details
    )
    return f"Claim text contains number(s) not supported by numeric evidence: {numbers}"


def _format_claim_number(number: float, is_percent: bool) -> str:
    if not math.isfinite(number):
        text = str(number)
    elif number == int(number):
        text = str(int(number))
    else:
        text = str(number)
    return f"{text}%" if is_percent else text


def _numeric_evidence_values(
    claim: ReportClaim,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> list[tuple[float, str, str, bool]]:
    values, _ = _numeric_evidence_values_with_sources(claim, evidence_pack, sql_results)
    return values


def _numeric_evidence_values_with_sources(
    claim: ReportClaim,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> tuple[list[tuple[float, str, str, bool]], list[NumericEvidenceSource]]:
    values: list[tuple[float, str, str, bool]] = []
    sources: list[NumericEvidenceSource] = []
    for evidence in claim.evidence:
        # Only resolved persisted payloads feed the pool; the model-authored
        # inline evidence.value must never self-certify a claim number (F1).
        contributed = _resolve_evidence_numbers(evidence, evidence_pack, sql_results)
        values.extend(contributed)
        sources.append(
            NumericEvidenceSource(
                artifact_id=evidence.artifact_id,
                locator=evidence.locator,
                kind=evidence.kind,
                resolved=bool(contributed),
                value_count=len(contributed),
                values=[
                    NumericSourceValue(value=value, unit=unit, policy=policy)
                    for value, unit, policy, _eligible in contributed[
                        :_NUMERIC_SOURCE_VALUE_CAP
                    ]
                ],
            )
        )
    return values, sources


def numeric_evidence_values_with_sources(
    claim: ReportClaim,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> tuple[list[tuple[float, str, str, bool]], list[NumericEvidenceSource]]:
    """Public shell over the per-ref numeric resolver (tools/evidence_display)."""
    return _numeric_evidence_values_with_sources(claim, evidence_pack, sql_results)


def _has_currency_unit_mismatch(
    claim: ReportClaim,
    *,
    numeric_tolerance: float,
    sql_results: dict[str, SqlResult],
) -> bool:
    """Bind user-visible currency labels to the exact evidence value."""
    evidence_pairs: list[tuple[float, str]] = []
    for evidence in claim.evidence:
        if evidence.artifact_id in sql_results:
            evidence_pairs.extend(
                _sql_result_currency_pairs(evidence, sql_results[evidence.artifact_id])
            )
            continue
        if not isinstance(evidence.value, int | float) or isinstance(evidence.value, bool):
            continue
        if evidence.unit != "currency" or evidence.unit_label is None:
            continue
        code = evidence.unit_label.removesuffix("/order")
        if re.fullmatch(r"[A-Z]{3}", code) is not None:
            evidence_pairs.append((float(evidence.value), code))
    if not evidence_pairs:
        return False

    assertion_text = _strip_quoted_spans(claim.text)
    text_pairs = _currency_amounts(assertion_text)
    for value, code in text_pairs:
        if not _matches_any_evidence(
            value,
            [evidence_value for evidence_value, _ in evidence_pairs],
            numeric_tolerance=numeric_tolerance,
        ):
            # Uppercase three-letter domain acronyms next to unrelated counts
            # (for example "10 SQL queries") are not currency assertions.
            continue
        if not any(
            code == evidence_code
            and _matches_any_evidence(
                value,
                [evidence_value],
                numeric_tolerance=numeric_tolerance,
            )
            for evidence_value, evidence_code in evidence_pairs
        ):
            return True

    text_numbers = [
        value for value, is_percent in _numbers_from_text(assertion_text) if not is_percent
    ]
    for evidence_value, evidence_code in evidence_pairs:
        if not _matches_any_evidence(
            evidence_value,
            text_numbers,
            numeric_tolerance=numeric_tolerance,
        ):
            continue
        if not any(
            code == evidence_code
            and _matches_any_evidence(
                value,
                [evidence_value],
                numeric_tolerance=numeric_tolerance,
            )
            for value, code in text_pairs
        ):
            return True
    return False


def _currency_amounts(text: str) -> list[tuple[float, str]]:
    amounts: list[tuple[float, str]] = []
    for match in _CURRENCY_AMOUNT_PATTERN.finditer(text):
        value = match.group("prefix_value") or match.group("suffix_value")
        code = match.group("prefix_code") or match.group("suffix_code")
        if value is not None and code is not None:
            amounts.append((float(value), code))
    return amounts


def full_coverage_evidence_refs(artifacts: Sequence[Artifact]) -> list[EvidenceRef]:
    """Cite every number each payload can resolve; the caller picks no locator.

    Kept beside the resolvers below so the two stay in step. Types absent here
    resolve nothing, so a figure drawn from one is refused rather than admitted
    unchecked.
    """
    refs: list[EvidenceRef] = []
    for artifact in artifacts:
        if artifact.type is ArtifactType.SQL_RESULT:
            refs.append(EvidenceRef(kind="sql", artifact_id=artifact.id, locator="rows"))
        elif artifact.type is ArtifactType.DATASET_PROFILE:
            refs.extend(
                EvidenceRef(kind="profile_field", artifact_id=artifact.id, locator=locator)
                for locator in ("summary", "missing_percent")
            )
        elif artifact.type is ArtifactType.STAT_TEST_RESULT:
            refs.append(EvidenceRef(kind="stat", artifact_id=artifact.id, locator=""))
        elif artifact.type is ArtifactType.MODEL_CARD:
            refs.append(EvidenceRef(kind="artifact", artifact_id=artifact.id, locator=""))
        elif artifact.type is ArtifactType.TABLE:
            refs.append(EvidenceRef(kind="table", artifact_id=artifact.id, locator="rows"))
    return refs


def _resolve_evidence_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> list[tuple[float, str, str, bool]]:
    """Dispatch on the artifact's persisted type, never the model-authored kind.

    Each entry is (value, unit, policy, threshold_eligible). Policy is derived
    deterministically — producer-declared count units and locator semantics
    first (counts exact, stats rounded), JSON native type otherwise (int exact,
    float rounded) — never from any LLM-writable field. threshold_eligible is
    True only for stat p_value/statistic/effect_size locator values: they are
    the only quantities an inequality token may verify against.
    """
    if not evidence.artifact_id:
        return []
    # SQL_RESULT payloads resolve regardless of ref kind (table/sql): the
    # question-execution chain cites result cells the same way table evidence does.
    if evidence.artifact_id in sql_results:
        return _sql_result_numbers_with_policy(evidence, sql_results[evidence.artifact_id])
    summary = evidence_pack.artifact_index.get(evidence.artifact_id)
    if summary is None:
        return []
    if summary.artifact_type == ArtifactType.DATASET_PROFILE.value:
        return _profile_numbers(evidence, evidence_pack)
    if summary.artifact_type == ArtifactType.STAT_TEST_RESULT.value:
        return _stat_test_numbers(evidence, evidence_pack)
    if summary.artifact_type == ArtifactType.MODEL_CARD.value:
        return _model_card_numbers(evidence, evidence_pack)
    if summary.artifact_type == ArtifactType.TABLE.value:
        return _table_numbers(evidence, evidence_pack)
    if summary.artifact_type == ArtifactType.QUALITY_ISSUE_SET.value:
        return _quality_issue_numbers(evidence, evidence_pack)
    return []


def _sql_result_numbers_with_policy(
    evidence: EvidenceRef,
    sql_result: SqlResult,
) -> list[tuple[float, str, str, bool]]:
    """Resolve numeric evidence from a SqlResult preview."""
    return [
        (value, _coarse_numeric_unit(unit_label), policy, False)
        for value, unit_label, policy in _sql_result_number_cells(evidence, sql_result)
    ]


def _sql_result_numbers(
    evidence: EvidenceRef,
    sql_result: SqlResult,
) -> list[tuple[float, str]]:
    """Policy-free (value, unit) view kept for evidence_interleave grants."""
    return [
        (value, unit)
        for value, unit, _policy, _eligible in _sql_result_numbers_with_policy(
            evidence, sql_result
        )
    ]


def _valid_sql_result_locator(evidence: EvidenceRef, sql_result: SqlResult) -> bool:
    locator = evidence.locator.strip()
    if not locator or locator in {"rows", "rows_preview"}:
        return True
    if locator in sql_result.columns:
        return True
    match = _ROW_LOCATOR_PATTERN.fullmatch(locator)
    if match is None:
        return False
    index_text = match.group("index")
    field = match.group("field")
    if index_text is not None and int(index_text) >= len(sql_result.rows_preview):
        return False
    return field is None or field in sql_result.columns


def _sql_result_number_cells(
    evidence: EvidenceRef,
    sql_result: SqlResult,
) -> list[tuple[float, str | None, str]]:
    """Resolve SQL cells together with their authoritative produced units."""
    if not _valid_sql_result_locator(evidence, sql_result):
        return []
    locator = evidence.locator.strip()
    match = _ROW_LOCATOR_PATTERN.fullmatch(locator)
    field = match.group("field") if match is not None else None
    index_text = match.group("index") if match is not None else None
    columns = (
        [locator]
        if locator in sql_result.columns
        else ([field] if field else sql_result.columns)
    )
    rows = sql_result.rows_preview
    if index_text is not None:
        rows = [rows[int(index_text)]]
    cells: list[tuple[float, str | None, str]] = []
    for row in rows:
        for column in columns:
            unit_label = sql_result.units.get(column)
            for value, policy in _typed_numbers_from_object(row.get(column)):
                # Producer-declared "count" is authoritative over the JSON
                # type: a float-serialized count is still an exact cardinality.
                if unit_label == "count":
                    policy = "exact"
                cells.append((value, unit_label, policy))
    return cells


def _sql_result_currency_pairs(
    evidence: EvidenceRef,
    sql_result: SqlResult,
) -> list[tuple[float, str]]:
    pairs: list[tuple[float, str]] = []
    for value, unit_label, _policy in _sql_result_number_cells(evidence, sql_result):
        if unit_label is None:
            continue
        code = unit_label.removesuffix("/order")
        if re.fullmatch(r"[A-Z]{3}", code) is not None:
            pairs.append((value, code))
    return pairs


def _coarse_numeric_unit(unit_label: str | None) -> str:
    if unit_label is None:
        return "raw"
    if unit_label in {"percent", "%"}:
        return "percent"
    code = unit_label.removesuffix("/order")
    return "currency" if re.fullmatch(r"[A-Z]{3}", code) is not None else "raw"


def _profile_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
) -> list[tuple[float, str, str, bool]]:
    dataset = next(
        (
            dataset
            for dataset in evidence_pack.datasets
            if dataset.artifact_id == evidence.artifact_id
        ),
        None,
    )
    if dataset is None:
        return []

    # Locator semantics: row/column counts are cardinalities (exact); missing
    # rates are continuous display stats (rounded).
    locator = evidence.locator.strip()
    if locator == "summary":
        return [
            (float(dataset.row_count), "raw", "exact", False),
            (float(dataset.column_count), "raw", "exact", False),
        ]
    if locator in {"rows", "row_count"}:
        return [(float(dataset.row_count), "raw", "exact", False)]
    if locator in {"columns", "column_count"}:
        return [(float(dataset.column_count), "raw", "exact", False)]
    if locator in {"missing_percent", "missing_percent.*"}:
        return [
            (float(value), "percent", "rounded", False)
            for value in dataset.missing_percent.values()
        ]
    if locator.startswith("missing_percent."):
        column = locator.removeprefix("missing_percent.")
        value = dataset.missing_percent.get(column)
        return [] if value is None else [(float(value), "percent", "rounded", False)]
    return []


def _stat_test_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
) -> list[tuple[float, str, str, bool]]:
    stat_test = next(
        (
            stat_test
            for stat_test in evidence_pack.stat_tests
            if stat_test.artifact_id == evidence.artifact_id
        ),
        None,
    )
    if stat_test is None:
        return []
    values = {
        "statistic": stat_test.statistic,
        "p_value": stat_test.p_value,
        "effect_size": stat_test.effect_size,
        "sample_size": stat_test.sample_size,
    }
    if evidence.locator in values and values[evidence.locator] is not None:
        # sample_size is a cardinality; the test statistics are continuous and
        # the only values inequality tokens may verify against.
        if evidence.locator == "sample_size":
            return [(float(stat_test.sample_size), "raw", "exact", False)]
        return [(float(values[evidence.locator]), "raw", "rounded", True)]
    return [
        (value, "raw", policy, False)
        for value, policy in _typed_numbers_from_object(stat_test.model_dump())
    ]


def _model_card_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
) -> list[tuple[float, str, str, bool]]:
    model_card = next(
        (
            model_card
            for model_card in evidence_pack.model_cards
            if model_card.artifact_id == evidence.artifact_id
        ),
        None,
    )
    if model_card is None:
        return []
    # Model metrics are continuous scores: rounded on every path.
    locator = evidence.locator.strip()
    if locator.startswith("metrics."):
        metric_name = locator.removeprefix("metrics.")
        value = model_card.metrics.get(metric_name)
        return [] if value is None else [(float(value), "raw", "rounded", False)]
    return [
        (value, "raw", "rounded", False)
        for value, _policy in _typed_numbers_from_object(model_card.metrics)
    ]


def _table_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
) -> list[tuple[float, str, str, bool]]:
    table = next(
        (
            table
            for table in evidence_pack.analysis_tables
            if table.artifact_id == evidence.artifact_id
        ),
        None,
    )
    if table is None:
        return []

    selected = _select_table_locator(table.rows, evidence.locator)
    return [
        (value, "raw", policy, False)
        for value, policy in _typed_numbers_from_object(selected)
    ]


def _quality_issue_numbers(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack,
) -> list[tuple[float, str, str, bool]]:
    """Structured QualityIssue figures (analysis-v3 §11.3).

    Locators follow the produced grammar (agents/reporting.py emits
    "quality_issue:{code}:{column}"; the prefix-free "{code}:{column}" is
    accepted too, with an empty column part for dataset-level issues). The
    never-produced "{column}:{code}" form is retired. "" or "issues" selects
    the whole set but resolves only its cardinality — per-issue metrics
    require a per-issue locator, so a whole-set pool cannot verify a figure
    misattributed to another column. Only the structured fields resolve —
    issues from legacy artifacts carry None and contribute nothing, so frozen
    pre-§11.3 corpora keep their unverified status. Policies follow the
    JSON-type rule: affected_count is an exact cardinality, metric_value a
    display-rounded stat.
    """
    issues = [
        issue
        for issue in evidence_pack.quality_issues
        if issue.artifact_id == evidence.artifact_id
    ]
    structured = [
        issue
        for issue in issues
        if issue.metric_value is not None or issue.affected_count is not None
    ]
    if not structured:
        return []
    locator = evidence.locator.strip()
    if locator in {"", "issues"}:
        return [(float(len(issues)), "raw", "exact", False)]
    code, separator, column = locator.removeprefix("quality_issue:").partition(":")
    if not separator:
        return []
    values: list[tuple[float, str, str, bool]] = []
    for issue in structured:
        if issue.code != code or (issue.column or "") != column:
            continue
        if issue.metric_value is not None:
            values.append((issue.metric_value, issue.metric_unit, "rounded", False))
        if issue.affected_count is not None:
            values.append((float(issue.affected_count), "raw", "exact", False))
    return values


def _select_table_locator(rows: list[dict[str, Any]], locator: str) -> Any:
    match = _ROW_LOCATOR_PATTERN.fullmatch(locator.strip())
    if not match:
        return rows
    index_text = match.group("index")
    field = match.group("field")
    if index_text is None:
        selected: Any = rows
    else:
        index = int(index_text)
        selected = rows[index] if 0 <= index < len(rows) else None
    if field and isinstance(selected, dict):
        return selected.get(field)
    return selected


def _typed_numbers_from_object(value: Any) -> list[tuple[float, str]]:
    """Numbers with the match policy of their JSON-native type: int values are
    exact cardinalities, float values are display-rounded stats (bool excluded)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [(float(value), "exact")]
    if isinstance(value, float):
        return [(value, "rounded")]
    if isinstance(value, dict):
        numbers: list[tuple[float, str]] = []
        for child in value.values():
            numbers.extend(_typed_numbers_from_object(child))
        return numbers
    if isinstance(value, list):
        numbers: list[tuple[float, str]] = []
        for child in value:
            numbers.extend(_typed_numbers_from_object(child))
        return numbers
    return []


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Public wrapper over the internal number extractor."""
    return _numbers_from_text(text)


@dataclass(frozen=True)
class _NumericToken:
    """One claim-text number with its display precision and, when directly
    preceded by a comparison operator, the asserted inequality."""

    value: float
    is_percent: bool
    decimals: int
    threshold_op: str | None = None


def _numeric_tokens_from_text(text: str) -> list[_NumericToken]:
    tokens: list[_NumericToken] = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw_token = match.group(0)
        is_percent = raw_token.endswith("%")
        core = raw_token.removesuffix("%")
        try:
            value = float(core.replace(",", ""))
        except ValueError:
            continue
        # Scientific tokens count mantissa decimals; thresholds only use value.
        mantissa = re.split(r"[eE]", core, maxsplit=1)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        prefix = _THRESHOLD_PREFIX_PATTERN.search(text[: match.start()])
        tokens.append(
            _NumericToken(
                value=value,
                is_percent=is_percent,
                decimals=decimals,
                threshold_op=prefix.group(1) if prefix else None,
            )
        )
    return tokens


def _numbers_from_text(text: str) -> list[tuple[float, bool]]:
    return [(token.value, token.is_percent) for token in _numeric_tokens_from_text(text)]


def _matches_any_evidence(
    number: float,
    evidence_numbers: Iterable[float],
    *,
    numeric_tolerance: float,
) -> bool:
    """Relative-window matcher used by the currency and time-boundary gates
    only; the numeric gate matches per value policy (_value_supports_token)."""
    for evidence_number in evidence_numbers:
        absolute_tolerance = max(abs(evidence_number) * numeric_tolerance, numeric_tolerance)
        if abs(number - evidence_number) <= absolute_tolerance:
            return True
    return False


def _contains_causal_language(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _CAUSAL_TERMS)


def _strip_quoted_spans(text: str) -> str:
    """Drop double-quoted citations so quoted questions do not read as assertions."""
    return _QUOTED_SPAN_PATTERN.sub(" ", text)


def _critical(
    code: str,
    message: str,
    *,
    section_title: str | None = None,
    claim_id: str | None = None,
    numeric_details: list[NumericMismatchDetail] | None = None,
) -> ReportValidationFinding:
    return ReportValidationFinding(
        severity=ReportSeverity.CRITICAL,
        code=code,
        message=message,
        section_title=section_title,
        claim_id=claim_id,
        repair_mode=_repair_mode_for_code(code),
        numeric_details=numeric_details or [],
    )


def _repair_mode_for_code(code: str) -> Literal["deterministic", "llm", "prune"]:
    if code in {"missing_quality_warning", "unsupported_section_body"}:
        return "deterministic"
    if code in {"numeric_mismatch", "currency_unit_mismatch", "causal_overclaim"}:
        return "llm"
    return "prune"


# DI8-D semantic soft gate — an INDEPENDENT code path from the correctness hard gate
# above.

_SEMANTIC_DEGRADED_CODE = "semantic_degraded"
_TIME_BOUNDARY_CODE = "time_boundary_partial_period"
# Products apply_semantic_gate writes into the audit; removed before a re-gate
# so replaying the gate is idempotent.
_SEMANTIC_GATE_FINDING_CODES = frozenset({_SEMANTIC_DEGRADED_CODE, _TIME_BOUNDARY_CODE})
_SEMANTIC_GATE_NOTE_PREFIXES = (
    "Evidence strength: ",
    "Semantic soft gate: ",
    "Time boundary: ",
)
# Legacy pre-F4 catalog claims (their synthetic evidence chain was deleted);
# excluded from the strong-ratio denominator to match the scoreboard.
_LEGACY_QFOCUS_PREFIX = "qfocus_"

# F6 evidence-strength tiers (claim-level confidence_label; the numeric
# verification axis above is separate and keeps its own vocabulary).
EvidenceStrength = Literal["strong", "indicative", "exploratory"]

# F6 placeholder: the final cut is adjudicated from the relabeled corpus
# distribution (analysis-v3 §11.2). Keep it defined in exactly this one place.
STRONG_RATIO_CUT = 0.60

# Full-table deterministic aggregations (no sampling branch in their producers).
_STRONG_ARTIFACT_TYPES = frozenset(
    {
        ArtifactType.DATASET_PROFILE.value,
        ArtifactType.TABLE.value,
        ArtifactType.STAT_TEST_RESULT.value,
    }
)


def evidence_strength_label(
    claim: ReportClaim,
    *,
    evidence_pack: EvidencePack | None,
    sql_results: dict[str, SqlResult],
) -> EvidenceStrength:
    """F6 claim tier with number binding (cross-review 2026-07-23).

    - Claim text asserts numbers and >=1 verifies: tier = strongest ref that
      actually verified a token; an unrelated strong ref cannot launder an
      sql-verified figure.
    - Numbers but zero verified tokens: exploratory (the asserted figures
      have no support).
    - No numbers: strongest resolved ref, QualityIssueSet artifact_index hit
      included (qualitative conclusions are backed by the full-table scan).
    """
    tokens = _numeric_tokens_from_text(claim.text)
    if tokens:
        verifying = _token_verifying_ref_indexes(
            claim, tokens, evidence_pack, sql_results
        )
        if not verifying:
            return "exploratory"
        strengths = {
            _evidence_ref_strength(claim.evidence[index], evidence_pack, sql_results)
            for index in verifying
        }
    else:
        strengths = {
            _evidence_ref_strength(evidence, evidence_pack, sql_results)
            for evidence in claim.evidence
        }
    if "strong" in strengths:
        return "strong"
    if "indicative" in strengths:
        return "indicative"
    return "exploratory"


def _token_verifying_ref_indexes(
    claim: ReportClaim,
    tokens: list[_NumericToken],
    evidence_pack: EvidencePack | None,
    sql_results: dict[str, SqlResult],
) -> set[int]:
    """Indexes of refs whose resolved values verified >=1 token, mirroring the
    _numeric_gate_outcome matching rules (unit pools, thresholds, policies)."""
    per_ref: list[list[tuple[float, str, str, bool]]] = []
    for evidence in claim.evidence:
        if evidence_pack is not None:
            per_ref.append(_resolve_evidence_numbers(evidence, evidence_pack, sql_results))
        elif evidence.artifact_id and evidence.artifact_id in sql_results:
            per_ref.append(
                _sql_result_numbers_with_policy(evidence, sql_results[evidence.artifact_id])
            )
        else:
            per_ref.append([])
    verifying: set[int] = set()
    for token in tokens:
        pool = [
            (ref_index, value, policy, eligible)
            for ref_index, values in enumerate(per_ref)
            for value, unit, policy, eligible in values
            if (unit == "percent") == token.is_percent
        ]
        if token.threshold_op is not None:
            compare = _THRESHOLD_OPS[token.threshold_op]
            eligible_entries = [
                (ref_index, value)
                for ref_index, value, _policy, eligible in pool
                if eligible
            ]
            if eligible_entries:
                verifying.update(
                    ref_index
                    for ref_index, value in eligible_entries
                    if compare(value, token.value)
                )
                continue
        verifying.update(
            ref_index
            for ref_index, value, policy, _eligible in pool
            if _value_supports_token(token, value, policy)
        )
    return verifying


def _evidence_ref_strength(
    evidence: EvidenceRef,
    evidence_pack: EvidencePack | None,
    sql_results: dict[str, SqlResult],
) -> EvidenceStrength | None:
    """Strength one ref contributes, or None when it resolves nothing.

    Typed dispatch mirrors _resolve_evidence_numbers (F1). QualityIssueSet
    numbers live in prose (unresolvable by design) but the artifact is a
    full-table deterministic scan, so an artifact_index hit counts as resolved.
    SqlResult/ModelCard cannot deterministically prove full-table aggregation
    (truncated/row_count are not proof — analysis-v3 F6 trap), so they cap at
    indicative.
    """
    if not evidence.artifact_id:
        return None
    if evidence.artifact_id in sql_results:
        cells = _sql_result_numbers_with_policy(
            evidence, sql_results[evidence.artifact_id]
        )
        return "indicative" if cells else None
    if evidence_pack is None:
        return None
    summary = evidence_pack.artifact_index.get(evidence.artifact_id)
    if summary is None:
        return None
    if summary.artifact_type == ArtifactType.QUALITY_ISSUE_SET.value:
        return "strong"
    if not _resolve_evidence_numbers(evidence, evidence_pack, sql_results):
        return None
    if summary.artifact_type in _STRONG_ARTIFACT_TYPES:
        return "strong"
    if summary.artifact_type == ArtifactType.MODEL_CARD.value:
        return "indicative"
    return None


def strong_ratio_verdict(
    strong_claims: int,
    total_claims: int,
    *,
    cut: float = STRONG_RATIO_CUT,
) -> Literal["pass", "degraded"]:
    """F6 bundle verdict: pass iff the strong-claim ratio reaches the cut.

    An empty denominator is degraded: a report that published nothing must not
    vacuously pass (cross-review 2026-07-23).
    """
    if total_claims == 0:
        return "degraded"
    return "pass" if strong_claims / total_claims >= cut else "degraded"


@dataclass
class SemanticGateOutcome:
    """Rollup of one semantic-gate pass (also written back onto the audit)."""

    verdict: GateVerdict
    degraded_claim_count: int = 0
    time_boundary_truncations: int = 0
    strong_claims: int = 0
    indicative_claims: int = 0
    exploratory_claims: int = 0
    findings: list[ReportValidationFinding] = field(default_factory=list)


def apply_semantic_gate(
    bundle: ReportBundle,
    audit: ReportAudit,
    *,
    evidence_pack: EvidencePack | None = None,
    role_sets: Sequence[ColumnRoleSet] | None = None,
    sql_results: dict[str, SqlResult] | None = None,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO,
    min_buckets: int = DEFAULT_MIN_BUCKETS,
    strong_ratio_cut: float = STRONG_RATIO_CUT,
) -> SemanticGateOutcome:
    """Label claims with F6 evidence-strength tiers and grade the bundle.

    The bundle verdict is the strong-claim ratio against the cut; the
    low_impact_column / time_boundary soft flags stay on their own per-claim
    axis (gate_flags + claim.gate_verdict) and no longer drive the verdict.
    Replay-safe: the gate first clears its own products from a previous pass.
    """
    sql_results = sql_results or {}
    _reset_semantic_gate_state(bundle, audit)
    if audit.has_critical_findings:
        # Hard-gate failure: the report is rejected as-is.
        audit.gate_verdict = "rejected"
        return SemanticGateOutcome(verdict="rejected")

    outcome = SemanticGateOutcome(verdict="pass")
    total_claims = 0
    for section in bundle.sections:
        for claim in section.claims:
            strength = evidence_strength_label(
                claim, evidence_pack=evidence_pack, sql_results=sql_results
            )
            claim.confidence_label = strength
            if not (claim.id or "").startswith(_LEGACY_QFOCUS_PREFIX):
                total_claims += 1
                if strength == "strong":
                    outcome.strong_claims += 1
                elif strength == "indicative":
                    outcome.indicative_claims += 1
                else:
                    outcome.exploratory_claims += 1
            flags = _semantic_flags(claim, role_sets)
            time_flag = _time_boundary_flag(
                claim,
                sql_results,
                partial_ratio_threshold=partial_ratio_threshold,
                min_buckets=min_buckets,
            )
            if time_flag:
                claim.time_boundary_flag = time_flag
                flags.append(time_flag)
                outcome.time_boundary_truncations += 1
            if not flags:
                continue
            claim.gate_flags = flags
            claim.gate_verdict = "degraded"
            outcome.degraded_claim_count += 1
            outcome.findings.append(
                ReportValidationFinding(
                    severity=ReportSeverity.WARN,
                    code=_TIME_BOUNDARY_CODE if time_flag else _SEMANTIC_DEGRADED_CODE,
                    message=(
                        "Claim published degraded with semantic gate flag(s): "
                        + "; ".join(flags)
                    ),
                    section_title=section.title,
                    claim_id=claim.id or None,
                )
            )

    outcome.verdict = strong_ratio_verdict(
        outcome.strong_claims, total_claims, cut=strong_ratio_cut
    )
    audit.gate_verdict = outcome.verdict
    audit.degraded_claim_count = outcome.degraded_claim_count
    audit.time_boundary_truncations = outcome.time_boundary_truncations
    audit.findings.extend(outcome.findings)
    if total_claims:
        audit.semantic_notes.append(
            f"Evidence strength: {outcome.strong_claims} strong / "
            f"{outcome.indicative_claims} indicative / "
            f"{outcome.exploratory_claims} exploratory claim(s); verdict "
            f"'{outcome.verdict}' at strong-ratio cut {strong_ratio_cut:.0%}."
        )
    if outcome.degraded_claim_count:
        audit.semantic_notes.append(
            f"Semantic soft gate: {outcome.degraded_claim_count} claim(s) published "
            "degraded with structured gate flags (not disclaimers)."
        )
    if outcome.time_boundary_truncations:
        audit.semantic_notes.append(
            f"Time boundary: {outcome.time_boundary_truncations} trend claim(s) rest "
            "on partial edge periods and were degraded with a time_boundary_flag."
        )
    return outcome


def _reset_semantic_gate_state(bundle: ReportBundle, audit: ReportAudit) -> None:
    """Clear the soft-gate claim state and this gate's own audit products so a
    re-gate starts clean instead of contradicting or duplicating a prior pass."""
    for section in bundle.sections:
        for claim in section.claims:
            claim.confidence_label = "verified"
            claim.gate_verdict = "pass"
            claim.gate_flags = []
            claim.time_boundary_flag = ""
    audit.degraded_claim_count = 0
    audit.time_boundary_truncations = 0
    audit.findings = [
        finding
        for finding in audit.findings
        if not (
            finding.severity is ReportSeverity.WARN
            and finding.code in _SEMANTIC_GATE_FINDING_CODES
        )
    ]
    audit.semantic_notes = [
        note
        for note in audit.semantic_notes
        if not note.startswith(_SEMANTIC_GATE_NOTE_PREFIXES)
    ]


def _semantic_flags(
    claim: ReportClaim,
    role_sets: Sequence[ColumnRoleSet] | None,
) -> list[str]:
    flags: list[str] = []
    for column in claim.referenced_columns:
        if _impact_weight(column, role_sets) == 0.0:
            flags.append(f"low_impact_column:{column}")
    return flags


def _impact_weight(column: str, role_sets: Sequence[ColumnRoleSet] | None) -> float:
    """Business-impact weight for a column; graceful default 1.0 without roles."""
    if not role_sets:
        return 1.0
    return min(role_set.impact_weight(column) for role_set in role_sets)


def _time_boundary_flag(
    claim: ReportClaim,
    sql_results: dict[str, SqlResult],
    *,
    partial_ratio_threshold: float,
    min_buckets: int,
) -> str:
    """Mandatory partial-period check for trend-type claims."""
    for evidence in claim.evidence:
        if not evidence.artifact_id or evidence.artifact_id not in sql_results:
            continue
        sql_result = sql_results[evidence.artifact_id]
        assessment = assess_sql_result(
            sql_result,
            partial_ratio_threshold=partial_ratio_threshold,
            min_buckets=min_buckets,
        )
        if assessment is None or not assessment.flagged:
            continue
        if _claim_leans_on_partial_periods(claim, sql_result, assessment):
            return "partial_periods:" + ",".join(assessment.partial_edge_labels)
    return ""


def _claim_leans_on_partial_periods(
    claim: ReportClaim,
    sql_result: SqlResult,
    assessment: TimeBoundaryAssessment,
) -> bool:
    text = claim.text
    if any(label and label in text for label in assessment.partial_edge_labels):
        return True
    text_numbers = _numbers_from_text(text)
    if not text_numbers:
        return True
    partial_values, complete_values = split_edge_values(sql_result, assessment)
    for number, _is_percent in text_numbers:
        in_partial = _matches_any_evidence(
            number, partial_values, numeric_tolerance=0.01
        )
        in_complete = _matches_any_evidence(
            number, complete_values, numeric_tolerance=0.01
        )
        if in_partial and not in_complete:
            return True
    return False

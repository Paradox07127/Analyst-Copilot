"""Deterministic, zero-LLM rendering of claim evidence provenance.

One line per evidence ref answers "where does this figure come from and did it
verify" in the claim ledger (UI, markdown, and HTML exports).
"""

from __future__ import annotations

import math

from pydantic import ValidationError

from eda_platform.schemas.artifacts import Artifact, ArtifactType, SqlResult
from eda_platform.schemas.reports import NumericEvidenceSource, ReportClaim
from eda_platform.tools.evidence import EvidencePack, PayloadPolicy, build_evidence_pack
from eda_platform.tools.report_validator import numeric_evidence_values_with_sources

_ARTIFACT_TYPE_PHRASES = {
    ArtifactType.DATASET_PROFILE.value: "dataset profile",
    ArtifactType.RAW_DATASET_PROFILE.value: "raw dataset profile",
    ArtifactType.QUALITY_ISSUE_SET.value: "quality issues",
    ArtifactType.SQL_RESULT.value: "query result",
    ArtifactType.TABLE.value: "analysis table",
    ArtifactType.STAT_TEST_RESULT.value: "statistical test",
    ArtifactType.MODEL_CARD.value: "model card",
    ArtifactType.CHART_SPEC.value: "chart",
}
_VALUE_DISPLAY_CAP = 3
_POOL_DISPLAY_CAP = 6


def evidence_display_context(
    artifacts: list[Artifact],
    *,
    payload_policy: PayloadPolicy = "schema+aggregates",
) -> tuple[EvidencePack, dict[str, SqlResult]]:
    """Build the evidence pack + SQL-result map evidence_lines resolves against.

    Contract: pass the payload_policy the claims were validated under. The
    persisted numeric statuses came from a pack built with that policy;
    rebuilding under a different one makes per-source lines contradict them
    (e.g. an unverified claim shown with green resolved sources).
    """
    pack = build_evidence_pack(artifacts, payload_policy=payload_policy)
    sql_results: dict[str, SqlResult] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.SQL_RESULT:
            try:
                sql_results[artifact.id] = SqlResult.model_validate(artifact.payload)
            except (ValidationError, TypeError):
                continue
    return pack, sql_results


def evidence_lines(
    claim: ReportClaim,
    evidence_pack: EvidencePack | None,
    sql_results: dict[str, SqlResult] | None,
) -> list[str]:
    """Per-claim provenance lines: token-status summary first, then one line
    per evidence ref. Without a pack only the persisted summary is rendered."""
    lines = [_summary_line(claim)]
    if evidence_pack is None:
        return lines
    sql_results = sql_results or {}
    values, sources = numeric_evidence_values_with_sources(
        claim, evidence_pack, sql_results
    )
    lines.extend(_failed_token_lines(claim, values))
    lines.extend(_source_line(source, evidence_pack, sql_results) for source in sources)
    return lines


def _summary_line(claim: ReportClaim) -> str:
    statuses = claim.numeric_statuses
    if not statuses:
        if claim.numeric_rollup == "no_numbers":
            return "– No figures in this claim"
        return "– Figures not evaluated"
    total = len(statuses)
    verified = sum(1 for status in statuses if status.status == "number_verified")
    failed = sum(1 for status in statuses if status.status == "failed")
    unverified = total - verified - failed
    noun = "figure" if total == 1 else "figures"
    if failed:
        return f"✗ {verified} of {total} {noun} verified · {failed} failed"
    if unverified:
        return f"◌ {verified} of {total} {noun} verified · {unverified} unverified"
    return f"✓ {verified} of {total} {noun} verified"


def _failed_token_lines(
    claim: ReportClaim,
    values: list[tuple[float, str, str, bool]],
) -> list[str]:
    """One line per failed token showing the pool it was checked against."""
    failed = [status for status in claim.numeric_statuses if status.status == "failed"]
    if not failed:
        return []
    percent_pool = sorted({value for value, unit, _p, _e in values if unit == "percent"})
    raw_pool = sorted({value for value, unit, _p, _e in values if unit != "percent"})
    lines: list[str] = []
    for status in failed:
        pool = percent_pool if status.is_percent else raw_pool
        unit = "percent" if status.is_percent else "raw"
        token = _format_value(status.number, unit)
        if pool:
            shown = ", ".join(_format_value(value, unit) for value in pool[:_POOL_DISPLAY_CAP])
            extra = len(pool) - _POOL_DISPLAY_CAP
            tail = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"✗ {token} — outside evidence pool: {shown}{tail}")
        else:
            unit_word = "percent" if status.is_percent else "numeric"
            lines.append(f"✗ {token} — no {unit_word} values resolved from evidence")
    return lines


def _source_line(
    source: NumericEvidenceSource,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> str:
    identity = _source_identity(source, evidence_pack, sql_results)
    if source.resolved:
        shown = ", ".join(
            _format_value(value.value, value.unit)
            for value in source.values[:_VALUE_DISPLAY_CAP]
        )
        extra = source.value_count - min(len(source.values), _VALUE_DISPLAY_CAP)
        tail = f" (+{extra} more)" if extra > 0 else ""
        policies = sorted({value.policy for value in source.values})
        policy = "/".join(policies) if policies else "exact"
        return f"✓ {shown}{tail} — {identity} ({policy})"
    annotation = _unresolved_annotation(source, evidence_pack, sql_results)
    return f"◌ unresolvable — {identity} ({annotation})"


def _source_identity(
    source: NumericEvidenceSource,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> str:
    summary = (
        evidence_pack.artifact_index.get(source.artifact_id)
        if source.artifact_id
        else None
    )
    if summary is not None:
        phrase = _ARTIFACT_TYPE_PHRASES.get(summary.artifact_type, summary.artifact_type)
        title = summary.title.strip()
        # Skip titles that just repeat the type ("Quality issues", "SqlResult").
        if title.lower() in {phrase.lower(), summary.artifact_type.lower()}:
            title = ""
    elif source.artifact_id and source.artifact_id in sql_results:
        phrase = _ARTIFACT_TYPE_PHRASES[ArtifactType.SQL_RESULT.value]
        title = ""
    else:
        phrase = source.kind or "evidence"
        title = ""
    identity = f"{phrase} '{title}'" if title else phrase
    if source.locator:
        identity += f" · {source.locator}"
    return identity


def _unresolved_annotation(
    source: NumericEvidenceSource,
    evidence_pack: EvidencePack,
    sql_results: dict[str, SqlResult],
) -> str:
    if not source.artifact_id:
        return "inline value, unverified"
    summary = evidence_pack.artifact_index.get(source.artifact_id)
    if summary is None and source.artifact_id not in sql_results:
        return "unknown artifact"
    if summary is not None and summary.artifact_type == ArtifactType.QUALITY_ISSUE_SET.value:
        return "prose figure, unverified"
    return "no numeric values, unverified"


def _format_value(value: float, unit: str) -> str:
    if math.isfinite(value) and value == int(value):
        text = str(int(value))
    else:
        text = f"{value:g}"
    return f"{text}%" if unit == "percent" else text

"""Build the final, compact AgentHandoff after all Auto-EDA stages finish."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from typing import Any

from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.handoff import (
    AgentDataset,
    AgentHandoffV3,
    AgentReadiness,
    ArtifactCatalogEntry,
    DatasetQualitySummary,
    DatasetReadiness,
    HandoffContextPolicy,
    HandoffGate,
    HandoffNextAction,
    HandoffQuestionResult,
    HandoffReport,
    HandoffRun,
    RelationshipReadiness,
)
from eda_platform.schemas.questions import QuestionExecutionResult

_MAX_INITIAL_BYTES = 64 * 1024
_MAX_INITIAL_TOKENS = 16_000
_MAX_HANDOFF_ARTIFACT_BYTES = 128 * 1024
_MAX_CATALOG_ENTRIES = 128
_MAX_CATALOG_CHART_ENTRIES = 32
_MAX_QUESTION_RESULTS = 32
_MAX_HANDOFF_PARENTS = 128
_MAX_DATASET_COLLECTION_REFS = 16
_MAX_PII_COLUMNS = 128
_PII_DATA_TYPES = {
    ArtifactType.CHART_SPEC,
    ArtifactType.RAW_CHART_SPEC,
    ArtifactType.RAW_DATA_PREVIEW,
    ArtifactType.SQL_RESULT,
    ArtifactType.CODE_EXECUTION_RESULT,
    ArtifactType.TABLE,
}
_SINGLETON_DATASET_KEYS = {
    ArtifactType.DATASET_PROFILE: "profile",
    ArtifactType.QUALITY_ISSUE_SET: "quality",
    ArtifactType.QUALITY_CONTEXT_SET: "quality_context",
    ArtifactType.PII_REPORT: "pii",
    ArtifactType.CLEANING_RECIPE: "cleaning_recipe",
    ArtifactType.COLUMN_ROLE_SET: "column_roles",
}
_COLLECTION_DATASET_KEYS = {
    ArtifactType.CHART_SPEC: "charts",
    ArtifactType.RAW_CHART_SPEC: "raw_charts",
    ArtifactType.TABLE: "tables",
    ArtifactType.STAT_TEST_RESULT: "stat_tests",
    ArtifactType.MODEL_CARD: "models",
}
_RAW_SINGLETON_DATASET_KEYS = {
    ArtifactType.RAW_DATASET_PROFILE: "raw_profile",
    ArtifactType.RAW_DATA_PREVIEW: "raw_preview",
    ArtifactType.PII_REPORT: "raw_pii",
}
_REFERENCE_ONLY_CATALOG_TYPES = {
    ArtifactType.EDA_HANDOFF,
    ArtifactType.HTML_REPORT,
    ArtifactType.MARKDOWN_REPORT,
    ArtifactType.QUESTION_CANDIDATE_SET,
    ArtifactType.REPORT_BUNDLE,
    ArtifactType.SESSION_SUMMARY,
}


def create_agent_handoff_artifact(
    artifacts: Sequence[Artifact],
    *,
    project_id: str,
    session_id: str,
    producer_version: str,
    execution_fingerprint: str,
    input_hashes: dict[str, str],
    generated_at: datetime | None = None,
    external_artifacts: Sequence[Artifact] = (),
    fetch_session_id: str | None = None,
) -> Artifact:
    """Create one deterministic, final manifest without embedding large payloads."""
    persisted_source = sorted(
        (artifact for artifact in artifacts if artifact.type is not ArtifactType.AGENT_HANDOFF),
        key=lambda artifact: (artifact.type.value, artifact.id),
    )
    persisted_ids = {artifact.id for artifact in persisted_source}
    external_by_id: dict[str, Artifact] = {}
    for artifact in sorted(
        external_artifacts,
        key=lambda item: (item.type.value, item.id, item.session_id),
    ):
        if artifact.type is ArtifactType.AGENT_HANDOFF or artifact.id in persisted_ids:
            continue
        external_by_id.setdefault(artifact.id, artifact)
    external = sorted(
        external_by_id.values(),
        key=lambda artifact: (artifact.type.value, artifact.id),
    )
    referenced = [*persisted_source, *external]
    published_session_id = fetch_session_id or session_id
    profiles = [
        (artifact, DatasetProfile.model_validate(artifact.payload))
        for artifact in referenced
        if artifact.type is ArtifactType.DATASET_PROFILE
    ]
    quality = _quality_by_dataset(referenced)
    legacy_datasets = _legacy_dataset_entries(referenced)
    raw_to_clean = _raw_to_clean_lineage(legacy_datasets)
    dataset_artifact_ids, dataset_artifact_omitted = _dataset_artifact_index(
        referenced, raw_to_clean
    )
    datasets = [
        _dataset_summary(
            profile,
            quality.get(profile.dataset_id, []),
            raw_dataset_id=legacy_datasets.get(profile.dataset_id, {}).get("raw_dataset_id"),
            artifact_ids=dataset_artifact_ids.get(profile.dataset_id, {}),
            artifact_omitted_counts=dataset_artifact_omitted.get(profile.dataset_id, {}),
        )
        for _, profile in profiles
    ]
    relationship = _relationship_readiness(referenced, len(profiles))
    report_gate, report = _report_summary(referenced)
    resource_preflight_status = _resource_preflight_status(referenced)
    quality_gate = _quality_gate(datasets)
    readiness, readiness_reasons = _overall_readiness(
        datasets, relationship, report_gate, referenced
    )
    sensitivity_by_id = _artifact_sensitivity(
        referenced, profiles=profiles, raw_to_clean=raw_to_clean
    )
    catalog = _bounded_catalog(
        referenced,
        raw_to_clean=raw_to_clean,
        sensitivity_by_id=sensitivity_by_id,
        source_session_id=session_id,
        fetch_session_id=published_session_id,
    )
    questions, question_total = _question_results(referenced)
    context_policy = _context_policy(
        catalog,
        referenced_count=len(referenced),
        included_question_count=len(questions),
        total_question_count=question_total,
    )
    next_actions = _next_actions(
        datasets,
        relationship,
        report,
        session_id,
        resource_preflight_status=resource_preflight_status,
    )
    counts = Counter(artifact.type.value for artifact in persisted_source)
    counts[ArtifactType.AGENT_HANDOFF.value] = 1
    pipeline_count = sum(
        1 for artifact in persisted_source if artifact.type is not ArtifactType.SESSION_METRICS
    )
    source_inventory_digest = _inventory_digest(persisted_source)
    external_inventory_digest = _inventory_digest(external)
    candidate_parent_ids = sorted({artifact.id for artifact in referenced})
    parent_ids = candidate_parent_ids[:_MAX_HANDOFF_PARENTS]
    payload = AgentHandoffV3(
        generated_at=generated_at or datetime.now(UTC),
        run=HandoffRun(
            project_id=project_id,
            session_id=session_id,
            status="completed" if readiness == "ready" else "completed_with_limits",
            producer_version=producer_version,
            execution_fingerprint=execution_fingerprint,
            input_hashes=dict(sorted(input_hashes.items())),
            pipeline_artifact_count=pipeline_count,
            persisted_source_artifact_count=len(persisted_source),
            referenced_external_artifact_count=len(external),
            artifact_count=len(persisted_source) + 1,
            artifact_counts=dict(sorted(counts.items())),
            source_inventory_count=len(persisted_source),
            source_inventory_digest=source_inventory_digest,
            external_inventory_digest=external_inventory_digest,
            lineage_candidate_parent_count=len(candidate_parent_ids),
            lineage_parent_count=len(parent_ids),
            lineage_parents_truncated=len(parent_ids) < len(candidate_parent_ids),
        ),
        readiness=AgentReadiness(
            status=readiness,
            reasons=readiness_reasons,
            quality_gate=quality_gate,
            report_gate=report_gate,
            cross_dataset_relationships=relationship,
        ),
        capabilities=_capabilities(
            referenced,
            len(profiles),
            relationship,
            resource_preflight_status=resource_preflight_status,
        ),
        datasets=datasets,
        artifact_catalog=catalog,
        question_results=questions,
        report=report,
        next_actions=next_actions,
        context_policy=context_policy,
    )
    _stabilize_context_budget(payload)
    # Revalidate after in-memory budget compaction; assignment validation is
    # intentionally off on the shared Pydantic base model.
    payload = AgentHandoffV3.model_validate(payload.model_dump(mode="json"))
    dumped = payload.model_dump(mode="json")
    result = Artifact(
        id=make_artifact_id(
            "agent_handoff",
            {"project_id": project_id, "session_id": session_id},
        ),
        type=ArtifactType.AGENT_HANDOFF,
        project_id=project_id,
        session_id=session_id,
        created_at=payload.generated_at,
        parents=parent_ids,
        payload=dumped,
        plain_language=(
            "Final, typed Auto-EDA handoff with readiness gates and lazy artifact references."
        ),
    )
    if len(result.model_dump_json().encode("utf-8")) > _MAX_HANDOFF_ARTIFACT_BYTES:
        raise ValueError(
            "AgentHandoff exceeds the 128 KiB contract budget after bounded compaction."
        )
    return result


def _quality_by_dataset(artifacts: Sequence[Artifact]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUALITY_ISSUE_SET:
            continue
        dataset_id = artifact.payload.get("dataset_id")
        issues = artifact.payload.get("issues", [])
        if isinstance(dataset_id, str) and isinstance(issues, list):
            result[dataset_id] = [issue for issue in issues if isinstance(issue, dict)]
    return result


def _legacy_dataset_entries(artifacts: Sequence[Artifact]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if artifact.type is not ArtifactType.EDA_HANDOFF:
            continue
        for item in artifact.payload.get("datasets", []):
            if isinstance(item, dict) and isinstance(item.get("dataset_id"), str):
                entries[item["dataset_id"]] = item
    return entries


def _raw_to_clean_lineage(
    legacy_datasets: dict[str, dict[str, Any]],
) -> dict[str, str]:
    return {
        raw_dataset_id: clean_dataset_id
        for clean_dataset_id, entry in legacy_datasets.items()
        for raw_dataset_id in [entry.get("raw_dataset_id")]
        if isinstance(raw_dataset_id, str)
    }


def _inventory_digest(artifacts: Sequence[Artifact]) -> str:
    inventory = [
        {
            "artifact_id": artifact.id,
            "type": artifact.type.value,
            "origin_session_id": artifact.session_id,
            "content_sha256": sha256(artifact.model_dump_json().encode("utf-8")).hexdigest(),
        }
        for artifact in sorted(
            artifacts, key=lambda item: (item.type.value, item.id, item.session_id)
        )
    ]
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _dataset_artifact_index(
    artifacts: Sequence[Artifact],
    raw_to_clean: dict[str, str],
) -> tuple[
    dict[str, dict[str, str | list[str]]],
    dict[str, dict[str, int]],
]:
    result: dict[str, dict[str, str | list[str]]] = defaultdict(dict)
    omitted: dict[str, Counter[str]] = defaultdict(Counter)
    chart_categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for artifact in artifacts:
        dataset_id = artifact.payload.get("dataset_id")
        if not isinstance(dataset_id, str):
            continue
        clean_dataset_id = raw_to_clean.get(dataset_id, dataset_id)
        is_raw = dataset_id in raw_to_clean
        singleton = (
            _RAW_SINGLETON_DATASET_KEYS.get(artifact.type)
            if is_raw
            else _SINGLETON_DATASET_KEYS.get(artifact.type)
        )
        collection = _COLLECTION_DATASET_KEYS.get(artifact.type)
        if singleton:
            result[clean_dataset_id][singleton] = artifact.id
        elif collection:
            existing = result[clean_dataset_id].setdefault(collection, [])
            if isinstance(existing, list):
                if artifact.type in {ArtifactType.CHART_SPEC, ArtifactType.RAW_CHART_SPEC}:
                    category = str(artifact.payload.get("category", "distribution"))
                    category_key = (clean_dataset_id, collection)
                    if category in chart_categories[category_key]:
                        omitted[clean_dataset_id][collection] += 1
                        continue
                    chart_categories[category_key].add(category)
                if len(existing) >= _MAX_DATASET_COLLECTION_REFS:
                    omitted[clean_dataset_id][collection] += 1
                    continue
                existing.append(artifact.id)
    indexed = {
        dataset_id: {
            key: sorted(value) if isinstance(value, list) else value
            for key, value in sorted(index.items())
        }
        for dataset_id, index in sorted(result.items())
    }
    omitted_counts = {
        dataset_id: dict(sorted(counts.items()))
        for dataset_id, counts in sorted(omitted.items())
        if counts
    }
    return indexed, omitted_counts


def _dataset_summary(
    profile: DatasetProfile,
    issues: list[dict[str, Any]],
    *,
    raw_dataset_id: Any,
    artifact_ids: dict[str, str | list[str]],
    artifact_omitted_counts: dict[str, int],
) -> AgentDataset:
    severities = Counter(str(issue.get("severity")) for issue in issues)
    material_codes = sorted(
        {
            str(issue.get("code"))
            for issue in issues
            if issue.get("severity") in {"critical", "warn"} and issue.get("code")
        }
    )
    reasons: list[str] = []
    if profile.rows == 0:
        reasons.append("dataset_has_no_rows")
    if severities["critical"]:
        reasons.append(f"critical_quality_issues:{severities['critical']}")
    if severities["warn"]:
        reasons.append(f"quality_warnings:{severities['warn']}")
    if not profile.grain or profile.grain.startswith("No column combination"):
        reasons.append("grain_not_established")
    status = (
        "blocked"
        if profile.rows == 0 or severities["critical"]
        else "limited"
        if reasons
        else "ready"
    )
    pii_items = sorted(profile.pii_columns.items())
    bounded_pii_items = pii_items[:_MAX_PII_COLUMNS]
    return AgentDataset(
        dataset_id=profile.dataset_id,
        raw_dataset_id=raw_dataset_id if isinstance(raw_dataset_id, str) else None,
        name=profile.name[:255],
        content_hash=profile.content_hash,
        rows=profile.rows,
        columns=profile.columns,
        grain=profile.grain[:500] if profile.grain else None,
        semantic_type_counts=dict(sorted(profile.semantic_type_counts.items())),
        primary_key_candidates=sorted(profile.primary_key_candidates)[:32],
        composite_key_candidates=sorted(profile.composite_key_candidates)[:16],
        pii_columns=dict(bounded_pii_items),
        pii_column_count=len(pii_items),
        pii_columns_omitted=len(pii_items) - len(bounded_pii_items),
        quality=DatasetQualitySummary(
            critical=severities["critical"],
            warn=severities["warn"],
            info=severities["info"],
            material_codes=material_codes,
        ),
        readiness=DatasetReadiness(status=status, reasons=reasons),
        artifact_ids=artifact_ids,
        artifact_omitted_counts=artifact_omitted_counts,
    )


def _relationship_readiness(
    artifacts: Sequence[Artifact], dataset_count: int
) -> RelationshipReadiness:
    candidate_ids = [
        artifact.id
        for artifact in artifacts
        if artifact.type is ArtifactType.RELATIONSHIP_CANDIDATE_SET
    ]
    validation_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.type is ArtifactType.RELATIONSHIP_VALIDATION_SET
    ]
    validation_ids = [artifact.id for artifact in validation_artifacts]
    verified = any(
        isinstance(validation, dict) and validation.get("verified") is True
        for artifact in validation_artifacts
        for validation in artifact.payload.get("validations", [])
    )
    artifact_ids = sorted([*candidate_ids, *validation_ids])
    if dataset_count < 2:
        return RelationshipReadiness(status="not_applicable", cross_table_claims_allowed=True)
    if verified:
        return RelationshipReadiness(
            status="validated",
            cross_table_claims_allowed=True,
            artifact_ids=artifact_ids,
        )
    if candidate_ids:
        return RelationshipReadiness(
            status="materialized",
            cross_table_claims_allowed=False,
            action="Validate and confirm a relationship before cross-table claims.",
            artifact_ids=artifact_ids,
        )
    return RelationshipReadiness(
        status="deferred",
        cross_table_claims_allowed=False,
        action="Run bounded relationship discovery before cross-table analysis.",
        artifact_ids=artifact_ids,
    )


def _quality_gate(datasets: Sequence[AgentDataset]) -> HandoffGate:
    critical = sum(dataset.quality.critical for dataset in datasets)
    warnings = sum(dataset.quality.warn for dataset in datasets)
    reasons = []
    if critical:
        reasons.append(f"critical_quality_issues:{critical}")
    if warnings:
        reasons.append(f"quality_warnings:{warnings}")
    status = "fail" if critical else "warn" if warnings else "pass"
    return HandoffGate(status=status, reasons=reasons)


def _latest_artifact(artifacts: Sequence[Artifact], artifact_type: ArtifactType) -> Artifact | None:
    matches = [artifact for artifact in artifacts if artifact.type is artifact_type]
    return (
        max(matches, key=lambda artifact: (artifact.created_at, artifact.id)) if matches else None
    )


def _report_summary(artifacts: Sequence[Artifact]) -> tuple[HandoffGate, HandoffReport]:
    audit = _latest_artifact(artifacts, ArtifactType.REPORT_AUDIT)
    bundle = _latest_artifact(artifacts, ArtifactType.REPORT_BUNDLE)
    markdown = _latest_artifact(artifacts, ArtifactType.MARKDOWN_REPORT)
    html = _latest_artifact(artifacts, ArtifactType.HTML_REPORT)
    if audit is None:
        return HandoffGate(status="not_run"), HandoffReport(status="not_generated")
    verdict = audit.payload.get("gate_verdict")
    audit_status = str(audit.payload.get("status") or "draft")
    findings = audit.payload.get("findings", [])
    reasons = sorted(
        {
            str(finding.get("code"))
            for finding in findings
            if isinstance(finding, dict) and finding.get("code")
        }
    )
    gate_status = (
        "fail"
        if verdict == "rejected" or audit_status == "blocked_for_review"
        else "warn"
        if verdict == "degraded" or audit_status in {"draft", "needs_revision"}
        else "pass"
        if verdict == "pass" and audit_status == "validated"
        else "warn"
    )
    if audit_status != "validated":
        reasons.append(f"report_status:{audit_status}")
    report_status = (
        "failed" if gate_status == "fail" else "limited" if gate_status == "warn" else "ready"
    )
    return (
        HandoffGate(status=gate_status, reasons=sorted(set(reasons)), artifact_id=audit.id),
        HandoffReport(
            status=report_status,
            audit_artifact_id=audit.id,
            bundle_artifact_id=bundle.id if bundle else None,
            markdown_artifact_id=markdown.id if markdown else None,
            html_artifact_id=html.id if html else None,
        ),
    )


def _resource_preflight_status(artifacts: Sequence[Artifact]) -> str | None:
    preflight = next(
        (
            artifact
            for artifact in reversed(artifacts)
            if artifact.type.value == "ResourcePreflight"
        ),
        None,
    )
    status = preflight.payload.get("status") if preflight is not None else None
    return status if status in {"accepted", "limited", "rejected"} else None


def _overall_readiness(
    datasets: Sequence[AgentDataset],
    relationship: RelationshipReadiness,
    report_gate: HandoffGate,
    artifacts: Sequence[Artifact],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    resource_preflight_status = _resource_preflight_status(artifacts)
    if not datasets:
        reasons.append("no_profiled_datasets")
    if resource_preflight_status in {"limited", "rejected"}:
        reasons.append(f"resource_preflight_{resource_preflight_status}")
    if any(dataset.readiness.status == "blocked" for dataset in datasets):
        reasons.append("one_or_more_datasets_blocked")
    elif any(dataset.readiness.status == "limited" for dataset in datasets):
        reasons.append("one_or_more_datasets_limited")
    if relationship.status in {"deferred", "materialized"}:
        reasons.append(f"cross_dataset_relationships_{relationship.status}")
    if report_gate.status in {"warn", "fail", "not_run"}:
        reasons.append(f"report_gate_{report_gate.status}")
    metrics = next(
        (
            artifact
            for artifact in reversed(artifacts)
            if artifact.type is ArtifactType.SESSION_METRICS
        ),
        None,
    )
    if metrics is None:
        reasons.append("session_metrics_unavailable")
    else:
        if metrics.payload.get("degraded"):
            reasons.append("session_metrics_degraded")
        if metrics.payload.get("coverage_limited"):
            reasons.append("session_coverage_limited")
        if metrics.payload.get("publication_blocked"):
            reasons.append("publication_blocked")
    blocked = (
        not datasets
        or any(dataset.readiness.status == "blocked" for dataset in datasets)
        or report_gate.status == "fail"
        or "publication_blocked" in reasons
    )
    return ("blocked" if blocked else "limited" if reasons else "ready", sorted(set(reasons)))


def _artifact_sensitivity(
    artifacts: Sequence[Artifact],
    *,
    profiles: Sequence[tuple[Artifact, DatasetProfile]],
    raw_to_clean: dict[str, str],
) -> dict[str, str]:
    pii_clean_ids = {profile.dataset_id for _, profile in profiles if profile.pii_columns}
    pii_dataset_ids = {
        *pii_clean_ids,
        *(
            raw_dataset_id
            for raw_dataset_id, clean_dataset_id in raw_to_clean.items()
            if clean_dataset_id in pii_clean_ids
        ),
    }
    tainted_ids: set[str] = set()
    for artifact in artifacts:
        dataset_id = artifact.payload.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id in pii_dataset_ids:
            tainted_ids.add(artifact.id)
        elif (
            artifact.type in _PII_DATA_TYPES and not isinstance(dataset_id, str) and pii_dataset_ids
        ):
            # SQL/code/table results often omit dataset_id. In a PII-bearing
            # run, unscoped data-bearing output is restricted until lineage
            # proves otherwise.
            tainted_ids.add(artifact.id)
    changed = True
    while changed:
        changed = False
        for artifact in artifacts:
            if artifact.id in tainted_ids or not set(artifact.parents) & tainted_ids:
                continue
            tainted_ids.add(artifact.id)
            changed = True
    sensitivity: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.type is ArtifactType.PII_REPORT:
            sensitivity[artifact.id] = "sensitive"
        elif artifact.id in tainted_ids:
            sensitivity[artifact.id] = (
                "pii_restricted" if artifact.type in _PII_DATA_TYPES else "sensitive"
            )
        else:
            sensitivity[artifact.id] = "internal"
    return sensitivity


def _catalog_entry(
    artifact: Artifact,
    *,
    raw_to_clean: dict[str, str],
    sensitivity_by_id: dict[str, str],
    source_session_id: str,
    fetch_session_id: str,
) -> ArtifactCatalogEntry:
    raw = artifact.model_dump_json().encode("utf-8")
    dataset_id = artifact.payload.get("dataset_id")
    dataset_id = dataset_id if isinstance(dataset_id, str) else None
    normalized_dataset_id = raw_to_clean.get(dataset_id, dataset_id) if dataset_id else None
    stage, role, priority = _catalog_semantics(artifact.type)
    required = (
        artifact.type
        in {
            ArtifactType.DATASET_PROFILE,
            ArtifactType.PII_REPORT,
            ArtifactType.QUALITY_ISSUE_SET,
            ArtifactType.QUALITY_CONTEXT_SET,
            ArtifactType.REPORT_AUDIT,
        }
        or artifact.type.value == "ResourcePreflight"
    )
    title = artifact.payload.get("title") or artifact.payload.get("name")
    fetch_origin_session_id = (
        fetch_session_id
        if artifact.session_id == source_session_id
        else artifact.session_id
    )
    return ArtifactCatalogEntry(
        artifact_id=artifact.id,
        type=artifact.type,
        origin_session_id=artifact.session_id,
        stage=stage,
        role=role,
        dataset_id=normalized_dataset_id,
        title=title[:200] if isinstance(title, str) else None,
        priority=priority,
        required=required,
        # Source artifacts may be published under an explicit public session,
        # while external/derived artifacts must be fetched from their actual
        # origin partition. Pointing every entry at the source session makes
        # external catalog links deterministically return 404.
        fetch=f"/api/v1/sessions/{fetch_origin_session_id}/artifacts/{artifact.id}",
        content_sha256=sha256(raw).hexdigest(),
        content_bytes=len(raw),
        estimated_tokens=ceil(len(raw) / 4),
        parent_count=len(set(artifact.parents)),
        parents=(sorted(set(artifact.parents)) if len(set(artifact.parents)) <= 8 else []),
        evidence_count=len(artifact.evidence),
        warning_count=len(artifact.warnings),
        sensitivity=sensitivity_by_id.get(artifact.id, "internal"),
    )


def _bounded_catalog(
    artifacts: Sequence[Artifact],
    *,
    raw_to_clean: dict[str, str],
    sensitivity_by_id: dict[str, str],
    source_session_id: str,
    fetch_session_id: str,
) -> list[ArtifactCatalogEntry]:
    """Bound the complete catalog; counts/digests preserve omitted inventory."""
    selected: list[Artifact] = []
    chart_keys: set[tuple[str, str]] = set()
    latest_audit = _latest_artifact(artifacts, ArtifactType.REPORT_AUDIT)
    for artifact in artifacts:
        if artifact.type in _REFERENCE_ONLY_CATALOG_TYPES:
            continue
        if (
            artifact.type is ArtifactType.REPORT_AUDIT
            and latest_audit is not None
            and artifact.id != latest_audit.id
        ):
            continue
        if artifact.type is not ArtifactType.CHART_SPEC:
            selected.append(artifact)
            continue
        dataset_id = artifact.payload.get("dataset_id")
        category = artifact.payload.get("category", "distribution")
        key = (str(dataset_id), str(category))
        if key in chart_keys:
            continue
        chart_keys.add(key)
        selected.append(artifact)
    entries = [
        _catalog_entry(
            artifact,
            raw_to_clean=raw_to_clean,
            sensitivity_by_id=sensitivity_by_id,
            source_session_id=source_session_id,
            fetch_session_id=fetch_session_id,
        )
        for artifact in selected
    ]

    # Reserve a small purpose-balanced visual slice instead of allowing a large
    # question/result inventory to crowd every chart out of the bounded index.
    def sort_key(item: ArtifactCatalogEntry) -> tuple[int, str, str]:
        return (
            {"critical": 0, "high": 1, "normal": 2, "on_demand": 3}[item.priority],
            item.type.value,
            item.artifact_id,
        )

    balanced_charts = sorted(
        (
            entry
            for entry in entries
            if entry.type in {ArtifactType.CHART_SPEC, ArtifactType.RAW_CHART_SPEC}
        ),
        key=sort_key,
    )[:_MAX_CATALOG_CHART_ENTRIES]
    non_charts = sorted(
        (
            entry
            for entry in entries
            if entry.type not in {ArtifactType.CHART_SPEC, ArtifactType.RAW_CHART_SPEC}
        ),
        key=sort_key,
    )
    chosen = [
        *non_charts[: max(0, _MAX_CATALOG_ENTRIES - len(balanced_charts))],
        *balanced_charts,
    ]
    return sorted(chosen, key=sort_key)


def _catalog_semantics(artifact_type: ArtifactType) -> tuple[str, str, str]:
    if artifact_type.value == "ResourcePreflight":
        return "ingest", "gate", "critical"
    if artifact_type in {ArtifactType.DATASET_PROFILE, ArtifactType.RAW_DATASET_PROFILE}:
        return "profile", "summary", "critical"
    if artifact_type in {
        ArtifactType.QUALITY_ISSUE_SET,
        ArtifactType.QUALITY_CONTEXT_SET,
        ArtifactType.PII_REPORT,
        ArtifactType.REPORT_AUDIT,
    }:
        return "quality", "gate", "critical"
    if artifact_type in {ArtifactType.CHART_SPEC, ArtifactType.RAW_CHART_SPEC}:
        return "exploration", "visual", "on_demand"
    if artifact_type is ArtifactType.TABLE:
        return "exploration", "evidence", "high"
    if artifact_type in {ArtifactType.STAT_TEST_RESULT, ArtifactType.MODEL_CARD}:
        return "statistics", "evidence", "high"
    if artifact_type in {ArtifactType.VALUE_MAP, ArtifactType.COLUMN_ROLE_SET}:
        return "semantic", "summary", "high"
    if artifact_type is ArtifactType.QUESTION_CANDIDATE_SET:
        return "question_planning", "plan", "on_demand"
    if artifact_type in {ArtifactType.QUESTION_EXECUTION_RESULT, ArtifactType.SQL_RESULT}:
        priority = (
            "high" if artifact_type is ArtifactType.QUESTION_EXECUTION_RESULT else "on_demand"
        )
        return "question_execution", "result", priority
    if artifact_type in {
        ArtifactType.HTML_REPORT,
        ArtifactType.MARKDOWN_REPORT,
        ArtifactType.REPORT_BUNDLE,
        ArtifactType.SESSION_SUMMARY,
    }:
        return "reporting", "presentation", "on_demand"
    if artifact_type is ArtifactType.SESSION_METRICS:
        return "observability", "metric", "normal"
    if artifact_type in {
        ArtifactType.CLEANING_RECIPE,
        ArtifactType.CLEANING_PREVIEW,
        ArtifactType.RAW_DATA_PREVIEW,
    }:
        return "ingest", "evidence", "normal"
    return "exploration", "evidence", "normal"


def _context_policy(
    catalog: Sequence[ArtifactCatalogEntry],
    *,
    referenced_count: int,
    included_question_count: int,
    total_question_count: int,
) -> HandoffContextPolicy:
    excluded = {ArtifactType.HTML_REPORT.value, ArtifactType.SESSION_METRICS.value}
    on_demand = {
        ArtifactType.CHART_SPEC.value,
        ArtifactType.HTML_REPORT.value,
        ArtifactType.MARKDOWN_REPORT.value,
        ArtifactType.QUESTION_CANDIDATE_SET.value,
        ArtifactType.SQL_RESULT.value,
    }
    candidates = sorted(
        (
            item
            for item in catalog
            if item.type.value not in excluded
            and item.priority in {"critical", "high"}
            and item.sensitivity != "pii_restricted"
        ),
        key=lambda item: (
            {"critical": 0, "high": 1}.get(item.priority, 2),
            item.content_bytes,
            item.artifact_id,
        ),
    )
    chosen: list[str] = []
    byte_total = 0
    token_total = 0
    for item in candidates:
        if (
            byte_total + item.content_bytes > _MAX_INITIAL_BYTES
            or token_total + item.estimated_tokens > _MAX_INITIAL_TOKENS
        ):
            continue
        chosen.append(item.artifact_id)
        byte_total += item.content_bytes
        token_total += item.estimated_tokens
    return HandoffContextPolicy(
        default_artifact_ids=chosen,
        on_demand_types=sorted(
            (ArtifactType(value) for value in on_demand), key=lambda item: item.value
        ),
        excluded_by_default_types=sorted(
            (ArtifactType(value) for value in excluded), key=lambda item: item.value
        ),
        max_initial_bytes=_MAX_INITIAL_BYTES,
        max_initial_estimated_tokens=_MAX_INITIAL_TOKENS,
        cataloged_artifact_count=len(catalog),
        default_artifact_count=len(chosen),
        omitted_artifact_count=max(0, referenced_count - len(catalog)),
        included_question_result_count=included_question_count,
        omitted_question_result_count=max(0, total_question_count - included_question_count),
        default_artifact_bytes=byte_total,
        default_artifact_estimated_tokens=token_total,
        serialized_bytes=0,
        estimated_tokens=0,
        initial_context_bytes=byte_total,
        initial_context_estimated_tokens=token_total,
    )


def _stabilize_context_budget(payload: AgentHandoffV3) -> None:
    """Reach a byte/token fixed point, shedding lowest-value defaults first."""
    policy = payload.context_policy
    priority_rank = {"critical": 0, "high": 1, "normal": 2, "on_demand": 3}
    for _ in range(512):
        catalog_by_id = {entry.artifact_id: entry for entry in payload.artifact_catalog}
        default_entries = [
            catalog_by_id[artifact_id]
            for artifact_id in policy.default_artifact_ids
            if artifact_id in catalog_by_id
        ]
        default_bytes = sum(entry.content_bytes for entry in default_entries)
        default_tokens = sum(entry.estimated_tokens for entry in default_entries)
        serialized_bytes = len(payload.model_dump_json().encode("utf-8"))
        estimated_tokens = ceil(serialized_bytes / 4)
        policy.default_artifact_count = len(default_entries)
        policy.default_artifact_bytes = default_bytes
        policy.default_artifact_estimated_tokens = default_tokens
        policy.serialized_bytes = serialized_bytes
        policy.estimated_tokens = estimated_tokens
        policy.initial_context_bytes = default_bytes + serialized_bytes
        policy.initial_context_estimated_tokens = default_tokens + estimated_tokens

        # Updating the resource figures can change their own digit width. Do
        # not make a budget decision until the serialized size is at a fixed point.
        if len(payload.model_dump_json().encode("utf-8")) != serialized_bytes:
            continue
        within_budget = (
            policy.initial_context_bytes <= policy.max_initial_bytes
            and policy.initial_context_estimated_tokens <= policy.max_initial_estimated_tokens
        )
        if within_budget:
            return
        if default_entries:
            remove = max(
                default_entries,
                key=lambda entry: (
                    priority_rank[entry.priority],
                    entry.content_bytes,
                    entry.artifact_id,
                ),
            )
            policy.default_artifact_ids = [
                artifact_id
                for artifact_id in policy.default_artifact_ids
                if artifact_id != remove.artifact_id
            ]
            continue
        if payload.artifact_catalog:
            remove = max(
                payload.artifact_catalog,
                key=lambda entry: (
                    not entry.required,
                    entry.type not in {ArtifactType.CHART_SPEC, ArtifactType.RAW_CHART_SPEC},
                    priority_rank[entry.priority],
                    entry.content_bytes,
                    entry.artifact_id,
                ),
            )
            payload.artifact_catalog = [
                entry
                for entry in payload.artifact_catalog
                if entry.artifact_id != remove.artifact_id
            ]
            policy.cataloged_artifact_count = len(payload.artifact_catalog)
            policy.omitted_artifact_count += 1
            continue
        raise ValueError("AgentHandoff payload alone exceeds the initial context budget.")
    raise ValueError("AgentHandoff context budget did not reach a stable fixed point.")


def _question_results(
    artifacts: Sequence[Artifact],
) -> tuple[list[HandoffQuestionResult], int]:
    results: list[HandoffQuestionResult] = []
    for artifact in artifacts:
        if artifact.type is not ArtifactType.QUESTION_EXECUTION_RESULT:
            continue
        execution = QuestionExecutionResult.model_validate(artifact.payload)
        results.append(
            HandoffQuestionResult(
                question_id=execution.question_id,
                status=execution.outcome or "failed",
                execution_artifact_id=artifact.id,
                sql_artifact_id=execution.sql_result_artifact_id,
                chart_artifact_id=execution.chart_artifact_id,
                finding_count=len(execution.findings),
                limitation_count=len(execution.limitations),
                exploratory=execution.exploratory,
            )
        )
    ordered = sorted(
        results,
        key=lambda result: (
            {"answered": 0, "abstained": 1, "awaiting_approval": 2, "failed": 3}[result.status],
            -result.finding_count,
            result.question_id,
            result.execution_artifact_id,
        ),
    )
    return ordered[:_MAX_QUESTION_RESULTS], len(ordered)


def _next_actions(
    datasets: Sequence[AgentDataset],
    relationship: RelationshipReadiness,
    report: HandoffReport,
    session_id: str,
    *,
    resource_preflight_status: str | None,
) -> list[HandoffNextAction]:
    actions: list[HandoffNextAction] = []
    resource_limited = resource_preflight_status in {"limited", "rejected"} and not datasets
    if resource_limited:
        actions.append(
            HandoffNextAction(
                action="partition_or_convert_dataset_and_rerun_auto_eda",
                priority="critical",
                blocking=True,
                reason=(
                    "Resource preflight allowed metadata only; partition the input or "
                    "convert it to a supported columnar workflow before rerunning Auto-EDA."
                ),
            )
        )
    if any(dataset.readiness.status != "ready" for dataset in datasets):
        actions.append(
            HandoffNextAction(
                action="review_material_quality_conditions",
                priority="high",
                blocking=any(dataset.readiness.status == "blocked" for dataset in datasets),
                reason="One or more datasets have material quality or grain limitations.",
                endpoint=f"/api/v1/sessions/{session_id}/quality",
            )
        )
    if relationship.status == "deferred":
        actions.append(
            HandoffNextAction(
                action="discover_cross_dataset_relationships",
                priority="high",
                blocking=True,
                reason=relationship.action,
                endpoint=f"/api/v1/sessions/{session_id}/relationships/discover",
            )
        )
    elif relationship.status == "materialized":
        actions.append(
            HandoffNextAction(
                action="validate_cross_dataset_relationship",
                priority="high",
                blocking=True,
                reason=relationship.action,
                endpoint=f"/api/v1/sessions/{session_id}/relationships",
            )
        )
    if report.status == "not_generated" and not resource_limited:
        actions.append(
            HandoffNextAction(
                action="generate_report_if_needed",
                priority="low",
                blocking=False,
                endpoint=f"/api/v1/sessions/{session_id}/report/generate",
            )
        )
    return actions


def _capabilities(
    artifacts: Sequence[Artifact],
    dataset_count: int,
    relationship: RelationshipReadiness,
    *,
    resource_preflight_status: str | None,
) -> dict[str, str]:
    types = {artifact.type for artifact in artifacts}
    relationship_status = (
        "not_applicable"
        if dataset_count < 2
        else "available"
        if relationship.status == "validated"
        else "deferred"
    )
    capabilities = {
        "cleaning": "available" if ArtifactType.CLEANING_RECIPE in types else "not_run",
        "profiling": "available" if ArtifactType.DATASET_PROFILE in types else "failed",
        "quality": "available" if ArtifactType.QUALITY_ISSUE_SET in types else "failed",
        "visualization": "available" if ArtifactType.CHART_SPEC in types else "not_run",
        "statistics": "available" if ArtifactType.STAT_TEST_RESULT in types else "not_run",
        "modeling": "available" if ArtifactType.MODEL_CARD in types else "not_run",
        "relationships": relationship_status,
        "questions": (
            "available" if ArtifactType.QUESTION_EXECUTION_RESULT in types else "not_run"
        ),
        "report": "available" if ArtifactType.MARKDOWN_REPORT in types else "not_run",
        "metrics": "available" if ArtifactType.SESSION_METRICS in types else "failed",
    }
    if resource_preflight_status in {"limited", "rejected"} and dataset_count == 0:
        for capability in (
            "profiling",
            "quality",
            "visualization",
            "statistics",
            "questions",
            "report",
        ):
            capabilities[capability] = "deferred"
    return capabilities

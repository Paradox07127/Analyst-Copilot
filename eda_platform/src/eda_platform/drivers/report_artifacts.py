"""Shared construction of agentic report artifacts after generation.

``auto_eda.ExportAgenticReportStep`` and ``question_exec._regenerate_report``
both turn an ``AgenticReportResult`` into the same artifact set. Trace emission
stays in the auto_eda step; only the artifact payload assembly is shared here.
"""

from __future__ import annotations

from eda_platform.agents.reporting import AgenticReportResult
from eda_platform.core.ids import make_artifact_id
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.charts import ChartSpec
from eda_platform.tools.evidence import PayloadPolicy
from eda_platform.tools.exporter import report_bundle_to_markdown
from eda_platform.tools.html_exporter import export_report_html


def build_agentic_report_artifacts(
    report: AgenticReportResult,
    artifacts: list[Artifact],
    *,
    project_id: str,
    session_id: str,
    payload_policy: PayloadPolicy,
) -> list[Artifact]:
    """Serialize a generated report into the standard report artifact set."""
    chart_specs = [
        ChartSpec.model_validate(artifact.payload)
        for artifact in artifacts
        if artifact.type is ArtifactType.CHART_SPEC
    ]
    markdown = report_bundle_to_markdown(
        report.bundle, artifacts=artifacts, payload_policy=payload_policy
    )
    html = export_report_html(
        report.bundle,
        charts=chart_specs,
        artifacts=artifacts,
        payload_policy=payload_policy,
    )
    parent_ids = [artifact.id for artifact in artifacts]
    bundle_payload = report.bundle.model_dump(mode="json")
    audit_payload = report.audit.model_dump(mode="json")
    report_identity = {
        "session_id": session_id,
        "artifact_ids": parent_ids,
        "status": report.bundle.status.value,
    }
    total_tokens = sum(call.usage.total_tokens for call in report.llm_calls)
    costs = [
        call.estimated_cost_usd for call in report.llm_calls if call.estimated_cost_usd is not None
    ]
    usage_payload = {
        "session_id": session_id,
        "report_status": report.bundle.status.value,
        "used_fallback": report.used_fallback,
        "llm_call_count": len(report.llm_calls),
        "prompt_tokens": sum(call.usage.prompt_tokens for call in report.llm_calls),
        "completion_tokens": sum(call.usage.completion_tokens for call in report.llm_calls),
        "total_tokens": total_tokens,
        "cached_tokens": sum(call.usage.cached_tokens for call in report.llm_calls),
        "estimated_cost_usd": round(sum(costs), 6) if costs else None,
        "model": report.llm_calls[-1].model if report.llm_calls else "offline",
    }
    result: list[Artifact] = [
        Artifact(
            id=make_artifact_id("runsummary", usage_payload),
            type=ArtifactType.SESSION_SUMMARY,
            project_id=project_id,
            session_id=session_id,
            parents=parent_ids,
            payload=usage_payload,
        ),
        Artifact(
            id=make_artifact_id("bundle", bundle_payload),
            type=ArtifactType.REPORT_BUNDLE,
            project_id=project_id,
            session_id=session_id,
            parents=parent_ids,
            payload=bundle_payload,
        ),
        Artifact(
            id=make_artifact_id("audit", audit_payload),
            type=ArtifactType.REPORT_AUDIT,
            project_id=project_id,
            session_id=session_id,
            parents=parent_ids,
            payload=audit_payload,
        ),
        Artifact(
            id=make_artifact_id("report", report_identity),
            type=ArtifactType.MARKDOWN_REPORT,
            project_id=project_id,
            session_id=session_id,
            parents=parent_ids,
            payload={"markdown": markdown},
        ),
        Artifact(
            id=make_artifact_id("html", report_identity),
            type=ArtifactType.HTML_REPORT,
            project_id=project_id,
            session_id=session_id,
            parents=parent_ids,
            payload={"html": html},
        ),
    ]
    if report.interleave_transcript is not None and report.interleave_transcript.exchanges:
        transcript_payload = report.interleave_transcript.model_dump(mode="json")
        result.append(
            Artifact(
                id=make_artifact_id("interleave", transcript_payload),
                type=ArtifactType.EVIDENCE_INTERLEAVE_TRANSCRIPT,
                project_id=project_id,
                session_id=session_id,
                parents=parent_ids,
                payload=transcript_payload,
            )
        )
    return result

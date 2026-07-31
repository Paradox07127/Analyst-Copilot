from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.reporting import (
    _apply_deterministic_repairs,
    _claim_plan_from_bundle,
    generate_agentic_report,
)
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.schemas.artifacts import Artifact, EvidenceRef
from eda_platform.schemas.reports import (
    NumericTokenStatus,
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
    ReportStatus,
)
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.report_validator import validate_report_bundle

T = TypeVar("T", bound=BaseModel)


class FakeReportPlanLLM:
    def __init__(self, plan: ReportPlanDraft | list[ReportPlanDraft]) -> None:
        self.plans = plan if isinstance(plan, list) else [plan]
        self.payloads: list[dict[str, Any]] = []
        self.schemas: list[str] = []
        self.call_count = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.payloads.append({"task": task, "payload": payload})
        self.schemas.append(schema.__name__)
        index = min(self.call_count, len(self.plans) - 1)
        self.call_count += 1
        return cast(T, self.plans[index])

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def test_claim_plan_from_bundle_strips_validator_derived_numeric_state() -> None:
    # The repair LLM must not see validator-derived numeric states: it could
    # otherwise rewrite unverified claims that no finding asked it to touch.
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    bundle.sections[0].claims.append(
        ReportClaim(
            id="c1",
            text="Revenue is 120, up 15%.",
            evidence=[EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")],
            numeric_statuses=[
                NumericTokenStatus(number=120, status="number_verified"),
                NumericTokenStatus(number=15, is_percent=True, status="unverified"),
            ],
            numeric_rollup="unverified",
            quantitative_coverage_gap=True,
            deterministic_source=True,
        )
    )

    rows = _claim_plan_from_bundle(bundle)

    row = next(r for r in rows if r["id"] == "c1")
    assert row["numeric_statuses"] == []
    assert row["numeric_rollup"] == "not_evaluated"
    assert row["quantitative_coverage_gap"] is False
    assert row["deterministic_source"] is False


def test_generate_agentic_report_requests_compact_claim_plan(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Dataset Overview",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = FakeReportPlanLLM(plan)

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.schemas == ["ReportPlanDraft"]
    assert "required_sections" not in llm.payloads[0]["payload"]
    assert "evidence_pack" not in llm.payloads[0]["payload"]
    assert "evidence_manifest" in llm.payloads[0]["payload"]
    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.bundle.sections[1].claims[0].id == "rows"


def test_generate_agentic_report_validates_fake_llm_plan(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = FakeReportPlanLLM(plan)

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.audit.status is ReportStatus.VALIDATED
    assert llm.payloads[0]["payload"]["business_context"] == "Revenue analysis"
    assert result.evidence_pack.datasets[0].sample_rows == []


def test_generate_agentic_report_injects_sections_without_llm_bodies(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    assert len(result.bundle.sections) == 11
    assert all(section.body or section.claims for section in result.bundle.sections)
    assert any(
        "No validated conclusion is available" in section.body
        for section in result.bundle.sections
        if not section.claims
    )
    assert result.validation_events[0].normalized_body_count == 0


def test_generate_agentic_report_repairs_quality_warnings_without_retry(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    quality = artifacts[1]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Data Quality Findings",
                id="revenue_missing",
                text="Revenue is 33.33% missing.",
                evidence=[
                    EvidenceRef(
                        kind="stat",
                        artifact_id=profile.id,
                        locator="missing_percent.revenue",
                        value=33.33,
                        unit="percent",
                    )
                ],
                referenced_datasets=["sales.csv"],
                referenced_columns=["revenue"],
            )
        ],
    )
    llm = FakeReportPlanLLM(plan)

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    claim = result.bundle.sections[3].claims[0]
    assert result.bundle.status is ReportStatus.VALIDATED
    assert llm.call_count == 1
    assert claim.quality_issue_refs == [quality.id]
    assert any(ref.artifact_id == quality.id and ref.kind == "artifact" for ref in claim.evidence)
    assert result.validation_events[-1].deterministic_repair_count > 0


def test_generate_agentic_report_retries_with_delta_payload_only(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    first = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="bad_rows",
                text="The dataset has 999 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    second = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="good_rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = FakeReportPlanLLM([first, second])

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    assert llm.call_count == 2
    retry_payload = llm.payloads[1]["payload"]
    assert "evidence_pack" not in retry_payload
    assert "evidence_manifest" not in retry_payload
    assert "previous_claim_plan" in retry_payload
    assert "previous_validator_findings" in retry_payload
    assert any(
        "numeric_mismatch" in finding
        for finding in retry_payload["previous_validator_findings"]
    )


def test_generate_agentic_report_stops_after_identical_no_progress_retry(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    unchanged_invalid_plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="bad_rows",
                text="The dataset has 999 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )
    llm = FakeReportPlanLLM(unchanged_invalid_plan)

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 2
    assert any(event.stopped_no_progress for event in result.validation_events)
    assert result.bundle.status is ReportStatus.VALIDATED
    assert all(
        claim.id != "bad_rows"
        for section in result.bundle.sections
        for claim in section.claims
    )
    assert (
        "Report repair stopped early after an identical no-progress retry."
        in result.audit.semantic_notes
    )


def test_generate_agentic_report_revalidates_when_hard_gate_prunes_all_claims(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                text="The finance dataset has 999 rows.",
                evidence=[EvidenceRef(kind="stat", artifact_id="missing", locator="rows", value=3)],
                referenced_datasets=["finance.csv"],
            )
        ],
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    retained_ids = [claim.id for section in result.bundle.sections for claim in section.claims]
    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.audit.status is ReportStatus.VALIDATED
    assert result.audit.findings == []
    assert retained_ids == [
        "exec_summary_dataset_overview_ds_sales",
        "dataset_overview_ds_sales",
    ]
    assert result.validation_events[-1].status == "validated"
    assert result.validation_events[-1].pruned_claim_count == 1
    assert "Hard validator removed 1 unsupported claim(s)." in result.audit.semantic_notes


def test_generate_agentic_report_prunes_invalid_claims_after_repair(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="supported",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            ),
            ReportPlanClaim(
                section_title="Executive Summary",
                id="unsupported",
                text="The finance dataset has 999 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id="missing", locator="rows", value=3)
                ],
                referenced_datasets=["finance.csv"],
            ),
        ]
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    retained_ids = [claim.id for section in result.bundle.sections for claim in section.claims]
    assert result.bundle.status is ReportStatus.VALIDATED
    assert retained_ids == ["supported", "dataset_overview_ds_sales"]
    assert result.validation_events[-1].pruned_claim_count == 1
    assert "Hard validator removed 1 unsupported claim(s)." in result.audit.semantic_notes


def test_generate_agentic_report_trace_reports_rendered_section_coverage(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    profile = artifacts[0]
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Executive Summary",
                id="rows",
                text="The dataset has 3 rows.",
                evidence=[
                    EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)
                ],
                referenced_datasets=["sales.csv"],
            )
        ],
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=FakeReportPlanLLM(plan),
    )

    assert result.validation_events[-1].section_coverage == 1.0
    assert result.validation_events[-1].claim_section_coverage < 1.0


def test_deterministic_repair_scrubs_unsupported_section_body(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    evidence_pack = build_evidence_pack(artifacts)
    profile = artifacts[0]
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    bundle.sections[0].body = "This section claims 999 rows without claim evidence."
    bundle.sections[0].claims.append(
        ReportClaim(
            id="supported_rows",
            text="The dataset has 3 rows.",
            evidence=[EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)],
            referenced_datasets=["sales.csv"],
        )
    )
    audit = validate_report_bundle(bundle, evidence_pack)

    repair_count = _apply_deterministic_repairs(bundle, audit, evidence_pack=evidence_pack)
    repaired_audit = validate_report_bundle(bundle, evidence_pack)

    assert repair_count == 1
    assert bundle.sections[0].body == ""
    assert repaired_audit.status is ReportStatus.VALIDATED


def test_generate_agentic_report_uses_deterministic_fallback_when_llm_is_offline(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=OfflineLLMClient(),
    )

    assert result.bundle.status is ReportStatus.VALIDATED
    assert result.audit.status is ReportStatus.VALIDATED
    assert any(section.claims for section in result.bundle.sections)
    assert any("Deterministic fallback" in note for note in result.audit.semantic_notes)


def test_offline_report_covers_every_dataset_in_file_sections(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "return_id,reason\n1,damaged\n2,late\n",
        encoding="utf-8",
    )
    returns = load_csv(returns_path, dataset_id="ds_returns")
    artifacts.append(
        profile_dataset(returns, project_id="project_demo", session_id="run_demo")
    )

    result = generate_agentic_report(
        artifacts,
        project_id="project_demo",
        session_id="run_demo",
        business_context="Revenue analysis",
        llm=OfflineLLMClient(),
    )

    overview = next(
        section for section in result.bundle.sections if section.title == "Dataset Overview"
    )
    file_summary = next(
        section
        for section in result.bundle.sections
        if section.title == "File-by-File EDA Summary"
    )
    assert {claim.referenced_datasets[0] for claim in overview.claims} == {
        "sales.csv",
        "returns.csv",
    }
    assert {claim.referenced_datasets[0] for claim in file_summary.claims} == {
        "sales.csv",
        "returns.csv",
    }
    quality_claims = next(
        section
        for section in result.bundle.sections
        if section.title == "Data Quality Findings"
    ).claims
    assert quality_claims
    for claim in quality_claims:
        evidence = claim.evidence[0]
        actual_count = sum(
            issue.artifact_id == evidence.artifact_id
            for issue in result.evidence_pack.quality_issues
        )
        assert evidence.value == actual_count
        assert claim.referenced_datasets
        assert not claim.referenced_columns
    assert result.bundle.status is ReportStatus.VALIDATED


def _artifacts(tmp_path: Path) -> list[Artifact]:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,order_date,revenue,region\n"
        "1,2026-01-01,10,East\n"
        "2,2026-01-02,,West\n"
        "3,2026-01-03,30,East\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path, dataset_id="ds_sales")
    profile = profile_dataset(loaded, project_id="project_demo", session_id="run_demo")
    quality = scan_quality(profile, project_id="project_demo", session_id="run_demo")
    charts = create_chart_specs(
        loaded, profile, project_id="project_demo", session_id="run_demo"
    )
    tables = create_analysis_tables(
        loaded,
        profile,
        project_id="project_demo",
        session_id="run_demo",
    )
    return [profile, quality, *charts, *tables]

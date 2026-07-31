from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.narrative_reviewer import review_narrative
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.schemas.artifacts import Artifact, EvidenceRef
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportSection,
    ReportStatus,
)
from eda_platform.tools.evidence import EvidencePack, build_evidence_pack
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.report_validator import validate_report_bundle

T = TypeVar("T", bound=BaseModel)

_ORIGINAL_FINDINGS_BODY = "Validated evidence-backed findings are listed below."
_CLEAN_POLISH_BODY = (
    "The revenue narrative highlights distinct regional behaviours, giving management "
    "a richer and more concrete view of where commercial attention belongs."
)
# Injects both a fresh number and fresh causal language -> must be rejected.
_MALICIOUS_BODY = (
    "Revenue rose 45% last quarter because of a seasonal demand surge that reshaped "
    "regional totals."
)


class FakeNarrativeLLM:
    """Returns a canned ``NarrativeRewrite`` body per section, tracking calls."""

    def __init__(self, bodies: dict[str, str] | str) -> None:
        self.bodies = bodies
        self.call_count = 0
        self.payloads: list[dict[str, Any]] = []
        self.schemas: list[str] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.call_count += 1
        self.payloads.append(payload)
        self.schemas.append(schema.__name__)
        title = payload.get("section_title", "")
        body = self.bodies.get(title, "") if isinstance(self.bodies, dict) else self.bodies
        return cast(T, schema(body=body))

    def text(self, *, task: str, payload: dict) -> str:
        return "fake"

    def last_usage(self) -> None:
        return None


def _find(bundle: ReportBundle, title: str) -> ReportSection:
    return next(section for section in bundle.sections if section.title == title)


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
    return [profile, quality]


def _bundle_with_business_findings(tmp_path: Path) -> tuple[ReportBundle, EvidencePack]:
    artifacts = _artifacts(tmp_path)
    evidence_pack = build_evidence_pack(artifacts)
    profile = artifacts[0]
    bundle = ReportBundle.empty(project_id="project_demo", session_id="run_demo")
    findings = _find(bundle, "Business Findings")
    findings.body = _ORIGINAL_FINDINGS_BODY
    findings.claims.append(
        ReportClaim(
            id="biz_rows",
            text="The dataset has 3 rows.",
            evidence=[EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)],
            referenced_datasets=["sales.csv"],
            confidence="high",
        )
    )
    # Sanity: the starting bundle is a valid release candidate.
    assert validate_report_bundle(bundle, evidence_pack).status is ReportStatus.VALIDATED
    return bundle, evidence_pack


def test_structural_body_is_not_rewritten(tmp_path: Path) -> None:
    bundle, evidence_pack = _bundle_with_business_findings(tmp_path)
    original_claims = [
        claim.model_copy(deep=True) for claim in _find(bundle, "Business Findings").claims
    ]
    llm = FakeNarrativeLLM({"Business Findings": _CLEAN_POLISH_BODY})

    result = review_narrative(
        bundle,
        evidence_pack=evidence_pack,
        llm=llm,
        target_sections=["Business Findings"],
    )

    reviewed = _find(result.bundle, "Business Findings")
    assert reviewed.body == _ORIGINAL_FINDINGS_BODY
    assert llm.schemas == []
    assert reviewed.claims == original_claims
    assert reviewed.claims[0].evidence == original_claims[0].evidence
    assert _find(bundle, "Business Findings").body == _ORIGINAL_FINDINGS_BODY
    assert validate_report_bundle(result.bundle, evidence_pack).status is ReportStatus.VALIDATED
    assert [event.status for event in result.events] == ["skipped"]


def test_malicious_rewrite_is_discarded_and_reverted(tmp_path: Path) -> None:
    bundle, evidence_pack = _bundle_with_business_findings(tmp_path)
    llm = FakeNarrativeLLM({"Business Findings": _MALICIOUS_BODY})

    result = review_narrative(
        bundle,
        evidence_pack=evidence_pack,
        llm=llm,
        target_sections=["Business Findings"],
    )

    reviewed = _find(result.bundle, "Business Findings")
    assert reviewed.body == _ORIGINAL_FINDINGS_BODY
    assert "45%" not in reviewed.body
    assert "because of" not in reviewed.body
    assert [event.status for event in result.events] == ["skipped"]
    # The final bundle is still a valid release candidate.
    assert validate_report_bundle(result.bundle, evidence_pack).status is ReportStatus.VALIDATED


def test_offline_or_missing_llm_skips_and_returns_bundle_unchanged(tmp_path: Path) -> None:
    bundle, evidence_pack = _bundle_with_business_findings(tmp_path)

    none_result = review_narrative(bundle, evidence_pack=evidence_pack, llm=None)
    offline_result = review_narrative(
        bundle, evidence_pack=evidence_pack, llm=OfflineLLMClient()
    )

    for result in (none_result, offline_result):
        assert result.bundle is bundle
        assert result.events == []
        assert result.llm_calls == 0
    assert _find(bundle, "Business Findings").body == _ORIGINAL_FINDINGS_BODY


def test_single_round_calls_llm_at_most_once_per_section(tmp_path: Path) -> None:
    bundle, evidence_pack = _bundle_with_business_findings(tmp_path)
    business_titles = ["Executive Summary", "Business Findings", "Business Recommendations"]
    for title in business_titles:
        section = _find(bundle, title)
        section.body = section.structural_body()
    llm = FakeNarrativeLLM(
        {
            title: f"Deepened qualitative narrative for {title} that stays evidence-free."
            for title in business_titles
        }
    )

    result = review_narrative(bundle, evidence_pack=evidence_pack, llm=llm)

    assert llm.call_count == 0
    assert result.llm_calls == 0
    assert sum(1 for event in result.events if event.status == "skipped") == len(
        business_titles
    )
    assert validate_report_bundle(result.bundle, evidence_pack).status is ReportStatus.VALIDATED

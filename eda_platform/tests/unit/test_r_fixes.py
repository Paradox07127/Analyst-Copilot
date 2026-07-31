"""Regression tests for the M2.5 hardening fixes (review findings R1-R19)."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

import pandas as pd
import pytest
from pydantic import BaseModel

from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.core.budget import Budget, BudgetExceeded
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.query import DuckDBQueryEngine, UnsafeQueryError
from eda_platform.schemas.artifacts import Artifact, EvidenceRef, QualityIssueSet
from eda_platform.schemas.reports import (
    ReportBundle,
    ReportClaim,
    ReportPlanDraft,
    ReportSection,
    ReportStatus,
)
from eda_platform.tools.evidence import (
    EvidenceAnalysisTable,
    EvidenceArtifactSummary,
    EvidencePack,
    build_evidence_pack,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.report_validator import validate_report_bundle

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"


def _profile_and_pack(tmp_path: Path) -> tuple[Artifact, EvidencePack]:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "order_id,amount\n1,10\n2,20\n3,30\n",
        encoding="utf-8",
    )
    profile = profile_dataset(
        load_csv(csv_path, dataset_id="ds_sales"),
        project_id="p",
        session_id="r",
    )
    pack = build_evidence_pack([profile], payload_policy="schema+aggregates")
    return profile, pack


# --- R1: section.body can no longer smuggle unverifiable facts past the gate ---


def test_r1_body_number_is_rejected(tmp_path: Path) -> None:
    _, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    _find(bundle, "Executive Summary").body = "The dataset has 3 rows."

    audit = validate_report_bundle(bundle, pack)

    assert audit.status is ReportStatus.NEEDS_REVISION
    assert any(f.code == "unsupported_section_body" for f in audit.findings)


def test_r1_body_causal_language_is_rejected(tmp_path: Path) -> None:
    _, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    _find(bundle, "Business Findings").body = "Revenue fell because of the price change."

    audit = validate_report_bundle(bundle, pack)

    assert any(f.code == "unsupported_section_body" for f in audit.findings)


def test_r1_qualitative_fact_cannot_bypass_claim_validation(tmp_path: Path) -> None:
    _, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    _find(bundle, "Business Findings").body = (
        "Revenue is concentrated in the enterprise segment."
    )

    audit = validate_report_bundle(bundle, pack)

    assert audit.status is ReportStatus.NEEDS_REVISION
    assert any(f.code == "unsupported_section_body" for f in audit.findings)


def test_r1_structural_section_body_is_allowed(tmp_path: Path) -> None:
    profile, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    section = _find(bundle, "Dataset Overview")
    section.body = "Validated evidence-backed findings are listed below."
    section.claims.append(
        ReportClaim(
            id="rows",
            text="The dataset has 3 rows.",
            evidence=[
                EvidenceRef(
                    kind="stat", artifact_id=profile.id, locator="rows", value=3
                )
            ],
        )
    )

    assert validate_report_bundle(bundle, pack).status is ReportStatus.VALIDATED


def test_r1_qualitative_body_with_grounded_claim_passes(tmp_path: Path) -> None:
    profile, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    _find(bundle, "Dataset Overview").claims.append(
        ReportClaim(
            id="rows",
            text="The dataset has 3 rows.",
            evidence=[EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)],
        )
    )

    audit = validate_report_bundle(bundle, pack)

    assert audit.status is ReportStatus.VALIDATED


def test_report_evidence_allows_analysis_table_kind() -> None:
    evidence = EvidenceRef(kind="table", artifact_id="table_1", locator="rows.0")

    assert evidence.kind == "table"


def test_report_validator_resolves_numbers_from_table_evidence() -> None:
    pack = EvidencePack(
        payload_policy="schema+aggregates",
        artifact_index={
            "table_1": EvidenceArtifactSummary(
                artifact_id="table_1",
                artifact_type="Table",
                title="Numeric summary",
                summary="Summary statistics",
            )
        },
        analysis_tables=[
            EvidenceAnalysisTable(
                artifact_id="table_1",
                dataset_id="ds_sales",
                title="Numeric summary",
                kind="numeric_summary",
                description="Summary statistics",
                rows=[{"column": "revenue", "mean": 125.0, "max": 150.0}],
            )
        ],
    )
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    _find(bundle, "File-by-File EDA Summary").claims.append(
        ReportClaim(
            id="mean_revenue",
            text="Revenue mean is 125 and max is 150.",
            evidence=[EvidenceRef(kind="table", artifact_id="table_1", locator="rows[0]")],
        )
    )

    audit = validate_report_bundle(bundle, pack)

    assert not any(f.code == "numeric_mismatch" for f in audit.findings)


# --- R6: percentages and raw counts cannot launder each other ---


def test_r6_percent_claim_not_matched_by_raw_evidence(tmp_path: Path) -> None:
    # F1 fix: the raw pool resolves (row count 3), so a "3%" claim with an
    # empty percent pool is a fabrication -> failed / no_evidence_values.
    profile, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    claim = ReportClaim(
        id="pct",
        text="Missing rate is 3% for the column.",
        evidence=[
            EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", unit="raw")
        ],
    )
    _find(bundle, "Data Quality Findings").claims.append(claim)

    audit = validate_report_bundle(bundle, pack)

    numeric = [f for f in audit.findings if f.code == "numeric_mismatch"]
    assert len(numeric) == 1
    assert numeric[0].numeric_details[0].reason == "no_evidence_values"
    assert claim.numeric_rollup == "failed"


def test_r6_percent_claim_matched_by_percent_evidence(tmp_path: Path) -> None:
    profile, pack = _profile_and_pack(tmp_path)
    bundle = ReportBundle.empty(project_id="p", session_id="r")
    claim = ReportClaim(
        id="pct",
        text="Missing rate is 0% for the column.",
        evidence=[
            EvidenceRef(
                kind="stat",
                artifact_id=profile.id,
                locator="missing_percent.amount",
                unit="percent",
            )
        ],
    )
    _find(bundle, "Data Quality Findings").claims.append(claim)

    audit = validate_report_bundle(bundle, pack)

    assert not any(f.code == "numeric_mismatch" for f in audit.findings)
    assert claim.numeric_rollup == "number_verified"


# --- R19: token budget is enforced ---


def test_r19_token_budget_is_enforced() -> None:
    budget = Budget(max_tokens=10)
    budget.add_tokens(6)
    assert budget.remaining_tokens() == 4
    with pytest.raises(BudgetExceeded):
        budget.add_tokens(6)


# --- R8: the read-only SQL runner blocks escapes, not just non-SELECT prefixes ---


def _engine() -> DuckDBQueryEngine:
    engine = DuckDBQueryEngine()
    engine.register_frame("orders", pd.DataFrame({"a": [1, 2, 3]}))
    return engine


def test_r8_allows_plain_select() -> None:
    result = _engine().execute_select("select count(*) as n from orders")
    assert result.to_dict("records") == [{"n": 3}]


@pytest.mark.parametrize(
    "sql",
    [
        "select 1; drop table orders",
        "copy (select 1) to '/tmp/x.csv'",
        "select * from read_csv('/etc/passwd')",
        "update orders set a = 1",
        "select * from read_parquet('/tmp/x.parquet')",
    ],
)
def test_r8_blocks_unsafe_queries(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        _engine().execute_select(sql)


# --- R13: outlier and empty-column rules exist (validator references them) ---


def test_r13_quality_flags_outlier_and_empty_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "dirty.csv"
    csv_path.write_text(
        "amount,note,blank\n1,a,\n2,b,\n3,c,\n4,d,\n100,e,\n",
        encoding="utf-8",
    )
    profile = profile_dataset(load_csv(csv_path, dataset_id="ds"), project_id="p", session_id="r")
    issues = QualityIssueSet.model_validate(
        scan_quality(profile, project_id="p", session_id="r").payload
    )
    codes = {issue.code for issue in issues.issues}

    assert "outlier_detected" in codes
    assert "empty_column" in codes


# --- R3 / R7: LLM failures degrade gracefully; usage is recorded ---


T = TypeVar("T", bound=BaseModel)


class _RaisingLLM:
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        raise RuntimeError("transport error")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        return None


class _BadJsonLLM:
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        ReportPlanDraft.model_validate({"claims": [{"section_title": "Executive Summary"}]})
        raise AssertionError("unreachable")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        return None


class _BadJsonWithUsageLLM:
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        ReportPlanDraft.model_validate({"claims": [{"section_title": "Executive Summary"}]})
        raise AssertionError("unreachable")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        return LLMResultMetadata(
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
        )


class _UsageLLM:
    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        draft = ReportPlanDraft(
            claims=[],
        )
        return cast(T, draft)

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        return LLMResultMetadata(
            provider="openai",
            model="gpt-test",
            usage=LLMUsage(prompt_tokens=30, completion_tokens=12, total_tokens=42),
        )


def test_r3_transport_error_falls_back_without_crashing(tmp_path: Path) -> None:
    profile, _ = _profile_and_pack(tmp_path)
    result = generate_agentic_report(
        [profile], project_id="p", session_id="r", business_context="", llm=_RaisingLLM()
    )
    assert result.used_fallback is True
    assert result.bundle.status is ReportStatus.VALIDATED
    assert len(result.llm_events) == 3
    assert result.llm_events[0].status == "error"
    assert result.llm_events[0].error_type == "RuntimeError"


def test_r3_invalid_json_falls_back_without_crashing(tmp_path: Path) -> None:
    profile, _ = _profile_and_pack(tmp_path)
    result = generate_agentic_report(
        [profile], project_id="p", session_id="r", business_context="", llm=_BadJsonLLM()
    )
    assert result.used_fallback is True


def test_r3_invalid_json_still_records_provider_usage(tmp_path: Path) -> None:
    profile, _ = _profile_and_pack(tmp_path)
    result = generate_agentic_report(
        [profile],
        project_id="p",
        session_id="r",
        business_context="",
        llm=_BadJsonWithUsageLLM(),
    )

    assert result.used_fallback is True
    assert len(result.llm_calls) == 3
    assert result.llm_calls[0].usage.total_tokens == 300
    assert result.llm_events[0].status == "error"
    assert result.llm_events[0].usage is not None


def test_r7_llm_usage_is_recorded(tmp_path: Path) -> None:
    profile, _ = _profile_and_pack(tmp_path)
    result = generate_agentic_report(
        [profile], project_id="p", session_id="r", business_context="", llm=_UsageLLM()
    )
    assert result.used_fallback is False
    assert len(result.llm_calls) == 1
    assert result.llm_calls[0].usage.total_tokens == 42
    assert result.llm_events[0].status == "success"
    assert result.llm_events[0].usage is not None


# --- R11: non-UTF-8 (GBK) files with Chinese headers load correctly ---


def test_r11_gbk_chinese_csv_is_decoded() -> None:
    loaded = load_csv(GOLDEN_DATA / "chinese_sales_gbk.csv", dataset_id="ds_zh")

    assert loaded.record.encoding == "gb18030"
    assert "地区" in loaded.frame.columns
    assert "华东" in loaded.frame["地区"].tolist()


def _find(bundle: ReportBundle, title: str) -> ReportSection:
    return next(section for section in bundle.sections if section.title == title)

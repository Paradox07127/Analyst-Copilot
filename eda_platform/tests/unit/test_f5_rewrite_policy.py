"""F5 rewrite policy: narrowed trigger surface, token budget breaker, and
best-attempt selection (analysis-v3 F5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.schemas.artifacts import Artifact, EvidenceRef
from eda_platform.schemas.reports import (
    EvidenceRequest,
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
)
from eda_platform.tools.analysis import create_analysis_tables
from eda_platform.tools.chart_specs import create_chart_specs
from eda_platform.tools.evidence import build_evidence_pack
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset
from eda_platform.tools.quality import scan_quality
from eda_platform.tools.report_validator import validate_report_bundle

T = TypeVar("T", bound=BaseModel)


class FakePlanLLM:
    """Returns scripted plans; optionally scripted per-call usage."""

    def __init__(
        self,
        plans: list[ReportPlanDraft],
        usages: list[LLMUsage] | None = None,
    ) -> None:
        self.plans = plans
        self.usages = usages
        self.call_count = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        # Report generation also narrates the finished bundle. Only plan
        # attempts spend the rewrite budget these tests are about, so a
        # non-plan request is answered without touching the counter.
        if schema is not ReportPlanDraft:
            return cast(T, schema())
        index = min(self.call_count, len(self.plans) - 1)
        self.call_count += 1
        return cast(T, self.plans[index])

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata | None:
        if self.usages is None:
            return None
        index = min(self.call_count - 1, len(self.usages) - 1)
        return LLMResultMetadata(
            provider="fake", model="fake-model", usage=self.usages[index]
        )


class FlakyPlanLLM(FakePlanLLM):
    """FakePlanLLM whose ``None`` plan entries raise a parse-style error."""

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        index = min(self.call_count, len(self.plans) - 1)
        self.call_count += 1
        plan = self.plans[index]
        if plan is None:
            raise RuntimeError("scripted parse failure")
        return cast(T, plan)


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
    profile = profile_dataset(loaded, project_id="p", session_id="r")
    quality = scan_quality(profile, project_id="p", session_id="r")
    charts = create_chart_specs(loaded, profile, project_id="p", session_id="r")
    tables = create_analysis_tables(loaded, profile, project_id="p", session_id="r")
    return [profile, quality, *charts, *tables]


def _good_claim(claim_id: str, profile_id: str) -> ReportPlanClaim:
    return ReportPlanClaim(
        section_title="Key EDA Insights",
        id=claim_id,
        text="The dataset has 3 rows.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)],
        referenced_datasets=["sales.csv"],
    )


def _bad_numeric_claim(claim_id: str, profile_id: str, wrong: int) -> ReportPlanClaim:
    return ReportPlanClaim(
        section_title="Key EDA Insights",
        id=claim_id,
        text=f"The dataset has {wrong} rows.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)],
        referenced_datasets=["sales.csv"],
    )


def _published_claim_ids(result: Any) -> set[str]:
    return {
        claim.id
        for section in result.bundle.sections
        for claim in section.claims
        if claim.id
    }


# --- R1: best-attempt selection ---


def test_r1_publishes_best_attempt_not_last(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    # Attempt 1: 2 CRITICAL (numeric_mismatch) + one good claim.
    plan_a = ReportPlanDraft(
        claims=[
            _good_claim("good_a", profile_id),
            _bad_numeric_claim("bad_a1", profile_id, 991),
            _bad_numeric_claim("bad_a2", profile_id, 992),
        ]
    )
    # Attempt 2 (worse): 3 CRITICAL + one good claim. Attempt 3 repeats it, so
    # the loop stops on no-progress after three calls.
    plan_b = ReportPlanDraft(
        claims=[
            _good_claim("good_b", profile_id),
            _bad_numeric_claim("bad_b1", profile_id, 993),
            _bad_numeric_claim("bad_b2", profile_id, 994),
            _bad_numeric_claim("bad_b3", profile_id, 995),
        ]
    )
    llm = FakePlanLLM([plan_a, plan_b, plan_b])

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count <= 3
    published = _published_claim_ids(result)
    assert "good_a" in published, "best attempt (fewest CRITICAL) must be published"
    assert "good_b" not in published
    assert any(event.selected_attempt == 1 for event in result.validation_events)


# --- R2: rewrite token budget breaker ---


def test_r2_budget_breaker_stops_rewrites(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    plans = [
        ReportPlanDraft(
            claims=[
                _good_claim("good_1", profile_id),
                _bad_numeric_claim("bad_1", profile_id, 991),
            ]
        ),
        ReportPlanDraft(
            claims=[
                _good_claim("good_2", profile_id),
                _bad_numeric_claim("bad_2", profile_id, 992),
                _bad_numeric_claim("bad_2b", profile_id, 993),
            ]
        ),
        ReportPlanDraft(claims=[_good_claim("good_3", profile_id)]),
    ]
    # Draft: 1000 total tokens. Repair 1: 1600 prompt+completion > 1.5 * 1000.
    usages = [
        LLMUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
        LLMUsage(prompt_tokens=1200, completion_tokens=400, total_tokens=1600),
        LLMUsage(prompt_tokens=1200, completion_tokens=400, total_tokens=1600),
    ]
    llm = FakePlanLLM(plans, usages=usages)

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 2, "no third LLM call after the budget is exhausted"
    assert any(event.budget_stopped for event in result.validation_events)
    # Attempt 1 (1 CRITICAL) beats attempt 2 (2 CRITICAL).
    published = _published_claim_ids(result)
    assert "good_1" in published
    assert "good_2" not in published
    assert any(event.selected_attempt == 1 for event in result.validation_events)


def test_r2_budget_under_threshold_allows_rewrites(tmp_path: Path) -> None:
    # Control probe: identical plans, usage below 1.5x -> rewrites continue.
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    plans = [
        ReportPlanDraft(claims=[_bad_numeric_claim("bad_1", profile_id, 991)]),
        ReportPlanDraft(claims=[_bad_numeric_claim("bad_2", profile_id, 992)]),
        ReportPlanDraft(claims=[_good_claim("good_3", profile_id)]),
    ]
    usages = [
        LLMUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
        LLMUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
        LLMUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
    ]
    llm = FakePlanLLM(plans, usages=usages)

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 3
    assert not any(event.budget_stopped for event in result.validation_events)
    assert "good_3" in _published_claim_ids(result)


def test_r2_budget_breaker_counts_error_retries(tmp_path: Path) -> None:
    # Cross-review fix 2a: a failed (parse-error) repair call burns tokens too;
    # the budget gate must run before the next plan call, not only after a
    # successful attempt.
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    plans = [
        ReportPlanDraft(claims=[_bad_numeric_claim("bad_1", profile_id, 991)]),
        None,  # repair call raises after burning 1600 tokens
        ReportPlanDraft(claims=[_good_claim("good_3", profile_id)]),
    ]
    usages = [
        LLMUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
        LLMUsage(prompt_tokens=1200, completion_tokens=400, total_tokens=1600),
        LLMUsage(prompt_tokens=1200, completion_tokens=400, total_tokens=1600),
    ]
    llm = FlakyPlanLLM(plans, usages=usages)

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 2, "no plan call after error retries burn the budget"
    assert any(event.budget_stopped for event in result.validation_events)
    assert any(event.selected_attempt == 1 for event in result.validation_events)


def test_r2_budget_counts_all_attempt_calls_not_just_last(tmp_path: Path) -> None:
    # Cross-review fix 2b: an interleave attempt makes several provider calls;
    # the spend is their sum, not the last call's usage.
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    plans = [
        ReportPlanDraft(claims=[_bad_numeric_claim("bad_1", profile_id, 991)]),
        ReportPlanDraft(
            evidence_requests=[
                EvidenceRequest(artifact_id=profile_id, locator="rows")
            ]
        ),
        ReportPlanDraft(claims=[_bad_numeric_claim("bad_2", profile_id, 992)]),
        ReportPlanDraft(claims=[_good_claim("good_4", profile_id)]),
    ]
    # Draft: 1000. Repair attempt: 800 + 800 = 1600 > 1.5x1000, while the
    # last call alone (800) would sneak under the budget.
    usages = [
        LLMUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
        LLMUsage(prompt_tokens=700, completion_tokens=100, total_tokens=800),
        LLMUsage(prompt_tokens=700, completion_tokens=100, total_tokens=800),
        LLMUsage(prompt_tokens=700, completion_tokens=100, total_tokens=800),
    ]
    llm = FakePlanLLM(plans, usages=usages)

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 3, "attempt 3 must not run once the sum exceeds budget"
    assert any(event.budget_stopped for event in result.validation_events)


# --- R3: unverified tokens + coverage gap never trigger a rewrite ---


def test_r3_unverified_and_coverage_gap_do_not_rewrite(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    chart_id = next(a.id for a in artifacts if a.id.startswith("chart_"))
    plan = ReportPlanDraft(
        claims=[
            ReportPlanClaim(
                section_title="Key EDA Insights",
                id="soft_only",
                # Chart evidence resolves no numeric values -> token stays
                # unverified (not failed) and the claim is a coverage gap.
                text="Revenue clusters around 42 in the east region.",
                evidence=[
                    EvidenceRef(kind="chart", artifact_id=chart_id, locator="chart")
                ],
                referenced_datasets=["sales.csv"],
            )
        ]
    )
    llm = FakePlanLLM([plan])

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 1, "exactly one claim-plan call; no rewrite"
    claim = next(
        claim
        for section in result.bundle.sections
        for claim in section.claims
        if claim.id == "soft_only"
    )
    assert claim.numeric_rollup == "unverified"
    assert claim.quantitative_coverage_gap is True


# --- R4: prune-mode CRITICAL (missing_evidence) never triggers a rewrite ---


def test_r4_prune_mode_critical_prunes_without_rewrite(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    profile_id = artifacts[0].id
    plan = ReportPlanDraft(
        claims=[
            _good_claim("good_kept", profile_id),
            ReportPlanClaim(
                section_title="Key EDA Insights",
                id="no_evidence",
                text="Something without any evidence reference.",
                evidence=[],
            ),
        ]
    )
    llm = FakePlanLLM([plan])

    result = generate_agentic_report(
        artifacts,
        project_id="p",
        session_id="r",
        business_context="Revenue analysis",
        llm=llm,
    )

    assert llm.call_count == 1, "prune-mode CRITICAL must not spend an LLM rewrite"
    published = _published_claim_ids(result)
    assert "good_kept" in published
    assert "no_evidence" not in published


# --- Ordering key: lexicographic, deterministic ---


def test_attempt_sort_key_orders_lexicographically(tmp_path: Path) -> None:
    # Deferred import: the helper only exists once F5 lands (red phase keeps
    # the module importable for the R3/R4 pinning tests above).
    from eda_platform.agents.reporting import _attempt_sort_key

    artifacts = _artifacts(tmp_path)
    evidence_pack = build_evidence_pack(artifacts)
    profile_id = artifacts[0].id

    def bundle_with(claims: list[ReportClaim]) -> tuple[ReportBundle, Any]:
        bundle = ReportBundle.empty(project_id="p", session_id="r")
        bundle.sections[0].claims.extend(claims)
        audit = validate_report_bundle(bundle, evidence_pack)
        return bundle, audit

    verified = ReportClaim(
        id="v",
        text="The dataset has 3 rows.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)],
        referenced_datasets=["sales.csv"],
    )
    mismatched = ReportClaim(
        id="m",
        text="The dataset has 999 rows.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)],
        referenced_datasets=["sales.csv"],
    )

    clean_bundle, clean_audit = bundle_with([verified])
    dirty_bundle, dirty_audit = bundle_with([verified, mismatched])
    # Key 1: fewer CRITICAL findings wins even with fewer claims.
    assert _attempt_sort_key(clean_bundle, clean_audit) < _attempt_sort_key(
        dirty_bundle, dirty_audit
    )

    one_verified_bundle, one_verified_audit = bundle_with([verified])
    two_verified_bundle, two_verified_audit = bundle_with(
        [verified, verified.model_copy(update={"id": "v2"})]
    )
    # Key 2: more number_verified tokens wins at equal CRITICAL count.
    assert _attempt_sort_key(two_verified_bundle, two_verified_audit) < _attempt_sort_key(
        one_verified_bundle, one_verified_audit
    )

    no_number = ReportClaim(
        id="n",
        text="Sales are recorded per region.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile_id, locator="rows", value=3)],
        referenced_datasets=["sales.csv"],
    )
    one_claim_bundle, one_claim_audit = bundle_with([no_number])
    two_claim_bundle, two_claim_audit = bundle_with(
        [no_number, no_number.model_copy(update={"id": "n2"})]
    )
    # Key 3: more claims wins at equal CRITICAL and verified-token counts.
    assert _attempt_sort_key(two_claim_bundle, two_claim_audit) < _attempt_sort_key(
        one_claim_bundle, one_claim_audit
    )

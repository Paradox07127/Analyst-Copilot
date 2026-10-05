"""Fault injection at task and remote commit boundaries of report generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from eda_platform.agents.evidence_interleave import (
    EvidenceInterleaveSession,
    InMemoryEvidenceResolver,
)
from eda_platform.agents.narrator import narrate_report
from eda_platform.agents.report_graph import ReportWorkflow
from eda_platform.agents.reporting import generate_agentic_report
from eda_platform.core.graph_execution import (
    GraphEffectUncertain,
    GraphIdentityError,
    GraphPersistence,
)
from eda_platform.core.llm import LLMResultMetadata, LLMSettings, LLMUsage
from eda_platform.schemas.artifacts import Artifact, EvidenceRef
from eda_platform.schemas.reports import (
    EvidenceRequest,
    ReportBundle,
    ReportClaim,
    ReportPlanClaim,
    ReportPlanDraft,
)
from eda_platform.tools.loader import load_csv
from eda_platform.tools.profiler import profile_dataset


class WorkerLost(BaseException):
    pass


class ReportLLM:
    def __init__(self, plans: list[ReportPlanDraft]) -> None:
        self.plans = plans
        self.calls: list[str] = []
        self.settings = LLMSettings(max_tokens=1000)
        self.usage: LLMResultMetadata | None = None

    def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append(task)
        self.usage = LLMResultMetadata(
            provider="test", model="test", request_id=f"call-{len(self.calls)}",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        if task == "report_section_narrative":
            return schema.model_validate({
                "text": payload["claims"][0]["text"],
                "cited_claim_ids": [payload["claims"][0]["id"]],
            })
        return schema.model_validate(self.plans.pop(0).model_dump())

    def last_usage(self) -> LLMResultMetadata | None:
        return self.usage

    def text(self, *, task: str, payload: dict) -> str:
        raise AssertionError("Unexpected text request")


def artifacts(tmp_path: Path) -> list[Artifact]:
    path = tmp_path / "data.csv"
    path.write_text("value,group\n1,A\n2,B\n3,A\n")
    return [profile_dataset(load_csv(path, dataset_id="d"), project_id="p", session_id="s")]


def row_plan(profile: Artifact) -> ReportPlanDraft:
    return ReportPlanDraft(claims=[ReportPlanClaim(
        id="rows", section_title="Dataset Overview", text="The dataset has 3 rows.",
        evidence=[EvidenceRef(kind="stat", artifact_id=profile.id, locator="rows", value=3)],
    )])


def generate(data: list[Artifact], llm: ReportLLM, persistence: GraphPersistence):
    return generate_agentic_report(
        data, project_id="p", session_id="s", business_context="Describe data",
        llm=llm, persistence=persistence,
    )


@pytest.mark.parametrize("boundary", ["remote_commit", "claim_plan_attempt", "validate_claim_plan"])
def test_report_resume_reuses_paid_calls_and_restores_interleave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    data = artifacts(tmp_path)
    request = ReportPlanDraft(evidence_requests=[
        EvidenceRequest(artifact_id=data[0].id, locator="rows"),
    ])
    llm = ReportLLM([request, row_plan(data[0])])
    persistence = GraphPersistence(tmp_path, f"report-{boundary}")
    crashed = False
    if boundary == "remote_commit":
        original = ReportWorkflow.structured

        def structured(self: ReportWorkflow, **kwargs: Any) -> dict[str, Any]:
            nonlocal crashed
            value = original(self, **kwargs)
            if kwargs["key"] == "plan:1:1" and not crashed:
                crashed = True
                raise WorkerLost()
            return value

        monkeypatch.setattr(ReportWorkflow, "structured", structured)
    else:
        original_step = ReportWorkflow.step

        def step(self: ReportWorkflow, name: str, run: Any) -> dict[str, Any]:
            nonlocal crashed
            value = original_step(self, name, run)
            if name == boundary and not crashed:
                crashed = True
                raise WorkerLost()
            return value

        monkeypatch.setattr(ReportWorkflow, "step", step)
    with pytest.raises(WorkerLost):
        generate(data, llm, persistence)
    assert llm.settings.max_tokens == 1000
    # Use a fresh client, as a restarted worker would: cached usage must come
    # from task/effect records, never the previous process's last_usage object.
    resumed_client = ReportLLM([])
    result = generate(data, resumed_client, persistence)
    assert len(llm.calls) == 2
    assert not resumed_client.calls
    assert result.interleave_transcript is not None
    assert result.interleave_transcript.granted_count == 1
    assert len(result.interleave_transcript.exchanges) == 1
    assert len(result.llm_calls) == 2
    assert [item.request_id for item in result.llm_calls] == ["call-1", "call-2"]
    assert len(result.llm_events) == 2
    assert [event.operation_id for event in result.llm_events] == ["plan:1:0", "plan:1:1"]
    assert [event.call_index for event in result.llm_events] == [1, 2]
    assert sum(event.usage.usage.total_tokens for event in result.llm_events if event.usage) == 20
    assert all(event.status == "success" for event in result.llm_events)
    again = generate(data, resumed_client, persistence)
    assert again == result
    assert not resumed_client.calls


def test_report_identity_binds_evidence_and_model_settings(tmp_path: Path) -> None:
    data = artifacts(tmp_path)
    persistence = GraphPersistence(tmp_path, "report")
    client = ReportLLM([row_plan(data[0])])
    generate(data, client, persistence)
    changed = data[0].model_copy(deep=True)
    changed.payload["rows"] = 99
    with pytest.raises(GraphIdentityError):
        generate([changed], client, persistence)
    client.settings.max_tokens += 1
    with pytest.raises(GraphIdentityError):
        generate(data, client, persistence)


def test_interleave_restore_preserves_consumed_and_refused_quotas() -> None:
    session = EvidenceInterleaveSession(InMemoryEvidenceResolver([]), total_limit=2)
    for _ in range(3):
        session.request(EvidenceRequest(artifact_id="absent", locator="rows"), section="a")
    restored = EvidenceInterleaveSession(InMemoryEvidenceResolver([]), total_limit=2)
    restored.restore(session.transcript)
    assert restored.remaining_total == 0
    assert restored.remaining_for_section("a") == 0
    assert restored.transcript == session.transcript


def narrative_bundle() -> ReportBundle:
    bundle = ReportBundle.empty(project_id="p", session_id="s")
    for section in bundle.sections[:2]:
        section.claims = [
            ReportClaim(id="a", text="There are 3 rows.", evidence=[]),
            ReportClaim(id="b", text="There are 2 columns.", evidence=[]),
        ]
    return bundle


def test_narration_resumes_at_section_boundary_and_accounts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = GraphPersistence(tmp_path, "narrative")
    llm = ReportLLM([])
    original = ReportWorkflow.step
    crashed = False

    def step(self: ReportWorkflow, name: str, run: Any) -> dict[str, Any]:
        nonlocal crashed
        value = original(self, name, run)
        if name == "narrate_section" and not crashed:
            crashed = True
            raise WorkerLost()
        return value

    monkeypatch.setattr(ReportWorkflow, "step", step)
    with pytest.raises(WorkerLost):
        narrate_report(narrative_bundle(), llm=llm, persistence=persistence)
    resumed = ReportLLM([])
    bundle = narrative_bundle()
    outcome = narrate_report(bundle, llm=resumed, persistence=persistence)
    assert len(llm.calls) == len(resumed.calls) == 1
    assert outcome.written == 2
    assert len(outcome.llm_calls) == len(outcome.llm_events) == 2
    assert all(section.narrative for section in bundle.sections[:2])
    assert narrate_report(bundle, llm=resumed, persistence=persistence) == outcome
    assert len(resumed.calls) == 1


def test_uncertain_provider_outcome_is_never_retried(tmp_path: Path) -> None:
    class TimeoutLLM(ReportLLM):
        def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
            self.calls.append(task)
            try:
                raise TimeoutError("Response was lost")
            except TimeoutError as exc:
                raise RuntimeError("Provider timeout") from exc

    data = artifacts(tmp_path)
    client = TimeoutLLM([])
    persistence = GraphPersistence(tmp_path, "uncertain")
    for _ in range(2):
        with pytest.raises(GraphEffectUncertain):
            generate(data, client, persistence)
    assert len(client.calls) == 1


def test_truncation_retry_restores_completion_cap_and_usage_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TruncatingLLM(ReportLLM):
        def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
            if not self.calls:
                self.calls.append(task)
                self.usage = LLMResultMetadata(
                    provider="test", model="test", request_id="truncated",
                    usage=LLMUsage(prompt_tokens=10, completion_tokens=1000, total_tokens=1010),
                )
                raise RuntimeError("Incomplete JSON response")
            assert self.settings.max_tokens == 1500
            return super().structured(task=task, schema=schema, payload=payload)

    data = artifacts(tmp_path)
    persistence = GraphPersistence(tmp_path, "truncation")
    llm = TruncatingLLM([row_plan(data[0])])
    original = ReportWorkflow.step
    completed_attempts = 0

    def step(self: ReportWorkflow, name: str, run: Any) -> dict[str, Any]:
        nonlocal completed_attempts
        saved = original(self, name, run)
        if name == "claim_plan_attempt":
            completed_attempts += 1
            if completed_attempts == 2:
                raise WorkerLost()
        return saved

    monkeypatch.setattr(ReportWorkflow, "step", step)
    with pytest.raises(WorkerLost):
        generate(data, llm, persistence)
    assert llm.settings.max_tokens == 1000
    resumed = TruncatingLLM([])
    result = generate(data, resumed, persistence)
    assert not resumed.calls
    assert resumed.settings.max_tokens == 1000
    assert len(result.llm_calls) == 2
    assert result.llm_calls[0].request_id == "truncated"
    assert len(result.llm_events) == 2
    assert result.llm_events[0].error_type == "truncation"
    assert result.llm_events[1].status == "success"


@pytest.mark.parametrize("narration", [False, True])
def test_report_cancellation_is_not_repaired_or_cached_as_fallback(
    tmp_path: Path, narration: bool,
) -> None:
    from eda_platform.core.cancellation import CancellationContext, CancellationError

    cancellation = CancellationContext()
    cancellation.request_cancel("User stopped report")

    class CancelledLLM(ReportLLM):
        def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
            self.calls.append(task)
            cancellation.checkpoint()
            raise AssertionError("Provider must not continue")

    llm = CancelledLLM([])
    persistence = GraphPersistence(tmp_path, "cancel-report")
    data = artifacts(tmp_path)

    def run():
        if narration:
            return narrate_report(narrative_bundle(), llm=llm, persistence=persistence)
        return generate(data, llm, persistence)

    with pytest.raises(CancellationError):
        run()
    assert len(llm.calls) == 1
    # A cancelled in-flight effect is unresolved, never a completed degraded report.
    with pytest.raises(GraphEffectUncertain):
        run()
    assert len(llm.calls) == 1


def test_interleave_failure_preserves_each_paid_call_and_status(tmp_path: Path) -> None:
    class MalformedSecondCall(ReportLLM):
        def structured[T: BaseModel](self, *, task: str, schema: type[T], payload: dict) -> T:
            if len(self.calls) == 1:
                self.calls.append(task)
                self.usage = LLMResultMetadata(
                    provider="test", model="test", request_id="call-2",
                    usage=LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                )
                raise RuntimeError("Malformed JSON")
            return super().structured(task=task, schema=schema, payload=payload)

    data = artifacts(tmp_path)
    request = ReportPlanDraft(evidence_requests=[
        EvidenceRequest(artifact_id=data[0].id, locator="rows"),
    ])
    provider = MalformedSecondCall([request, row_plan(data[0])])
    result = generate(data, provider, GraphPersistence(tmp_path, "interleave-error"))
    assert len(provider.calls) == len(result.llm_calls) == len(result.llm_events) == 3
    assert [event.status for event in result.llm_events] == ["success", "error", "success"]
    assert [event.operation_id for event in result.llm_events] == [
        "plan:1:0", "plan:1:1", "plan:2:0",
    ]
    assert [event.usage.request_id for event in result.llm_events if event.usage] == [
        "call-1", "call-2", "call-3",
    ]
    assert sum(event.usage.usage.total_tokens for event in result.llm_events if event.usage) == 30

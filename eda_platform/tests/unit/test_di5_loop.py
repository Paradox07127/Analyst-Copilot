"""DI sprint-5 (DI5-B): the Level-2 bounded investigation loop, driven adversarially.

The loop is the platform's controlled-autonomy surface: the LLM plans typed probes
and writes a (gated) conclusion, but every number comes from executing read-only SQL
and reducing it, and termination is guaranteed by hard caps. These tests script fake
LLMs to attack each exit and boundary:

- the LLM never stops (step cap) — exactly ``max_steps`` probes, then ``step_cap``;
- the LLM keeps erroring (retries burn calls) — ``llm_cap_reached`` at the cap;
- a probe names a table outside the approved scope — a failed step that never
  reaches the engine, counted as a probe error;
- two consecutive probe errors — ``probe_error_cap_reached``;
- a conclusion that fabricates a number — the note is discarded but the loop still
  exits ``concluded``; a clean conclusion is admitted;
- offline — zero model calls, empty transcript;
and the orchestrator wiring: a deep plan end-to-end yields a DEEP_INVESTIGATION_RESULT
artifact plus a validated finding whose merged probe findings are all evidence-backed,
while a non-deep plan is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from eda_platform.agents.investigation_loop import run_bounded_loop
from eda_platform.core.budget import BudgetExceeded
from eda_platform.core.llm import OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import AutoEDAResult, run_auto_eda
from eda_platform.drivers.investigation_orchestrator import (
    approve_plan,
    create_investigation_plans,
    execute_investigation_plans,
)
from eda_platform.schemas.artifacts import ArtifactType, EvidenceRef
from eda_platform.schemas.investigations import InvestigationPlan, ValidatedFinding
from eda_platform.schemas.questions import QuestionCandidateSet, QuestionFinding
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.sql_runner import SqlCatalog, build_catalog

GOLDEN_DATA = Path(__file__).parents[1] / "golden" / "data"

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Fake / spy LLM clients
# --------------------------------------------------------------------------- #
class ScriptedLoopLLM:
    """Returns pre-scripted probe/conclude decisions; counts every structured call."""

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        if not self._decisions:
            raise RuntimeError("scripted loop LLM exhausted its decisions.")
        return schema(**self._decisions.pop(0))  # type: ignore[return-value]

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class AlwaysRaisingLLM:
    """Every structured call fails to parse — used to burn the LLM-call budget."""

    def __init__(self) -> None:
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        raise RuntimeError("model returned unparseable output.")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class SpyOfflineLLM(OfflineLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        return super().structured(task=task, schema=schema, payload=payload)


class ScriptedOrchestratorLLM:
    """Serves scripted probes for the loop and a fixed interpretation for L1."""

    def __init__(self, probe_decisions: list[dict], interpretation: str) -> None:
        self._probes = list(probe_decisions)
        self._interpretation = interpretation
        self.probe_calls = 0
        self.interp_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        if task == "di5_bounded_probe":
            self.probe_calls += 1
            return schema(**self._probes.pop(0))  # type: ignore[return-value]
        if task == "di4_l1_interpretation":
            self.interp_calls += 1
            return schema(interpretation=self._interpretation)  # type: ignore[call-arg]
        raise RuntimeError(f"unexpected task: {task}")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _catalog(tmp_path: Path) -> tuple[SqlCatalog, LoadedDataset]:
    csv = tmp_path / "tiny.csv"
    csv.write_text("amount\n10\n20\n30\n", encoding="utf-8")
    dataset = load_csv(csv, dataset_id="ds_tiny")
    return build_catalog([dataset]), dataset


def _plan() -> InvestigationPlan:
    return InvestigationPlan(
        investigation_id="inv_test",
        source_session_id="src_run",
        question_id="q_test",
        card_version=1,
        candidate_fingerprint="fingerprint",
        question="How large is the amount?",
        target_datasets=["tiny.csv"],
        method_family="descriptive",
        method_recipe="A single read-only SELECT over the approved dataset.",
        allowed_tools=["read_only_sql", "llm_probe_planner"],
        feasibility="ready",
        status="planned",
        status_reason="Ready for controlled execution.",
    )


def _primary_finding(value: float = 42.0) -> QuestionFinding:
    return QuestionFinding(
        text=f"The primary metric settled at {value:g}.",
        evidence=[
            EvidenceRef(kind="sql", artifact_id="primary", locator="rows_preview[0].v", value=value)
        ],
    )


def _spy_engine(catalog: SqlCatalog) -> list[str]:
    """Record every SQL that reaches ``execute_select`` (i.e. is actually run)."""
    executed: list[str] = []
    original = catalog.engine.execute_select

    def spy(sql: str):  # noqa: ANN202
        executed.append(sql)
        return original(sql)

    catalog.engine.execute_select = spy  # type: ignore[method-assign]
    return executed


# --------------------------------------------------------------------------- #
# Loop-level adversarial behaviour
# --------------------------------------------------------------------------- #
def test_offline_skips_loop_with_zero_calls(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    spy = SpyOfflineLLM()
    result = run_bounded_loop(
        spy,
        plan=_plan(),
        primary_findings=[_primary_finding()],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "offline"
    assert result.llm_calls_used == 0
    assert result.steps == []
    assert spy.structured_calls == 0


def test_step_cap_stops_after_exactly_max_steps_probes(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    # Distinct SQL per probe: identical probes would now trip the DI8 repeat
    # detector; this test is about the step cap alone.
    probes = [
        {"action": "probe", "purpose": "scan amount", "sql": "SELECT amount FROM tiny"},
        {"action": "probe", "purpose": "max amount", "sql": "SELECT max(amount) AS m FROM tiny"},
        {"action": "probe", "purpose": "min amount", "sql": "SELECT min(amount) AS m FROM tiny"},
    ]
    llm = ScriptedLoopLLM(probes)
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
        max_steps=3,
    )
    assert result.exit_reason == "step_cap_reached"
    probe_steps = [step for step in result.steps if step.action == "probe"]
    assert len(probe_steps) == 3
    assert all(step.status == "succeeded" for step in probe_steps)
    # No 4th planning call: the step cap is checked before asking again.
    assert llm.structured_calls == 3


def test_llm_call_cap_stops_when_retries_burn_the_budget(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = AlwaysRaisingLLM()
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
        max_steps=3,
        llm_call_cap=8,
    )
    assert result.exit_reason == "llm_cap_reached"
    assert result.llm_calls_used == 8
    assert llm.structured_calls == 8
    assert result.steps == []


def test_run_budget_exception_is_not_retried_as_model_output_error(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)

    class BudgetRejectingLLM(AlwaysRaisingLLM):
        def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
            self.structured_calls += 1
            raise BudgetExceeded("run request budget exhausted")

    llm = BudgetRejectingLLM()
    with pytest.raises(BudgetExceeded):
        run_bounded_loop(
            llm,
            plan=_plan(),
            primary_findings=[],
            query_engine=catalog.engine,
            catalog=catalog,
        )

    assert llm.structured_calls == 1


def test_invalid_action_is_rejected_and_retry_receives_validation_feedback(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = ScriptedLoopLLM(
        [
            {"action": "continue", "purpose": "scan", "sql": "SELECT amount FROM tiny"},
            {"action": "conclude", "rationale": "Nothing further to add."},
        ]
    )
    payloads: list[dict] = []
    original = llm.structured

    def record_payload(*, task: str, schema: type[T], payload: dict) -> T:
        payloads.append(payload)
        return original(task=task, schema=schema, payload=payload)

    llm.structured = record_payload  # type: ignore[method-assign]
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )

    assert result.exit_reason == "concluded"
    assert result.llm_calls_used == 2
    assert len(result.steps) == 1
    assert "decision_validation_error" not in payloads[0]
    assert "decision_validation_error" in payloads[1]


def test_probe_without_sql_is_rejected_before_execution(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    executed = _spy_engine(catalog)
    llm = ScriptedLoopLLM(
        [
            {"action": "probe", "purpose": "scan", "sql": ""},
            {"action": "conclude", "rationale": "No safe follow-up was proposed."},
        ]
    )

    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )

    assert result.exit_reason == "concluded"
    assert result.llm_calls_used == 2
    assert executed == []
    assert all(step.action == "conclude" for step in result.steps)


def test_out_of_scope_probe_fails_without_touching_the_engine(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    executed = _spy_engine(catalog)
    llm = ScriptedLoopLLM(
        [
            {"action": "probe", "purpose": "peek", "sql": "SELECT * FROM secret_table"},
            {"action": "conclude", "rationale": "Nothing further to add."},
        ]
    )
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "concluded"
    assert result.probe_errors == 1
    probe_step = result.steps[0]
    assert probe_step.action == "probe"
    assert probe_step.status == "failed"
    assert "scope" in probe_step.error
    # The out-of-scope statement never reached the engine.
    assert not any("secret_table" in sql.lower() for sql in executed)


def test_two_consecutive_probe_errors_exit_probe_error_cap(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    # Two DIFFERENT out-of-scope probes: an identical repeat would now be
    # rejected dry by the DI8 repeat detector instead of erroring twice.
    bad_a = {"action": "probe", "purpose": "peek", "sql": "SELECT * FROM secret_table_a"}
    bad_b = {"action": "probe", "purpose": "peek", "sql": "SELECT * FROM secret_table_b"}
    llm = ScriptedLoopLLM([bad_a, bad_b])
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "probe_error_cap_reached"
    assert result.probe_errors == 2
    assert len(result.steps) == 2
    assert all(step.status == "failed" for step in result.steps)


def test_conclude_with_fabricated_number_discards_the_note(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = ScriptedLoopLLM(
        [{"action": "conclude", "rationale": "The metric soared to 9999 this quarter."}]
    )
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[_primary_finding(42.0)],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "concluded"
    # 9999 is not traceable to any finding evidence — the note is withheld.
    assert result.conclusion_note == ""


def test_clean_conclude_admits_the_note(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    rationale = "The follow-up probes surface nothing beyond the primary result."
    llm = ScriptedLoopLLM([{"action": "conclude", "rationale": rationale}])
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[_primary_finding(42.0)],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "concluded"
    assert result.conclusion_note == rationale


def test_llm_number_in_probe_purpose_never_reaches_claim_text(tmp_path: Path) -> None:
    """2026-07-17 trust audit: the LLM-authored purpose is interpolated into
    reducer claim text, so its number tokens are stripped before reduction —
    a fabricated "87%" must not ride into a finding sentence."""
    catalog, _ = _catalog(tmp_path)
    llm = ScriptedLoopLLM(
        [
            {
                "action": "probe",
                "purpose": "Investigate the 87% amount spike",
                "sql": "SELECT max(amount) AS max_amount FROM tiny",
            },
            {"action": "conclude", "rationale": "The peak matches expectations."},
        ]
    )
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    probe_step = next(step for step in result.steps if step.action == "probe")
    assert probe_step.status == "succeeded"
    assert probe_step.findings
    for finding in probe_step.findings:
        assert "87" not in finding.text
        assert "[n]" in finding.text


def test_successful_probe_produces_evidence_backed_findings(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = ScriptedLoopLLM(
        [
            {
                "action": "probe",
                "purpose": "largest amount",
                "sql": "SELECT max(amount) AS max_amount FROM tiny",
            },
            {"action": "conclude", "rationale": "The peak matches expectations."},
        ]
    )
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "concluded"
    probe_step = next(step for step in result.steps if step.action == "probe")
    assert probe_step.status == "succeeded"
    assert probe_step.result_artifact_id is not None
    assert probe_step.findings
    assert all(finding.evidence for finding in probe_step.findings)


# --------------------------------------------------------------------------- #
# Plan-creation deep marker
# --------------------------------------------------------------------------- #
def _source(tmp_path: Path) -> AutoEDAResult:
    return run_auto_eda(
        [GOLDEN_DATA / "ecommerce_orders.csv"],
        workspace=tmp_path / "workspace",
        project_id="project_demo",
        session_id="source_run",
    )


def _first_template_candidate_id(source: AutoEDAResult) -> str:
    candidate_set = QuestionCandidateSet.model_validate(
        next(
            item.payload
            for item in source.artifacts
            if item.type is ArtifactType.QUESTION_CANDIDATE_SET
        )
    )
    candidate = next(
        item
        for item in candidate_set.candidates
        if item.origin == "template" and item.sql_template is not None
    )
    return candidate.question_id


def test_deep_plan_carries_visible_marker_and_tool(tmp_path: Path) -> None:
    source = _source(tmp_path)
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[_first_template_candidate_id(source)],
        workspace=source.workspace,
        session_id="plan_run",
        deep=True,
    )
    plan = InvestigationPlan.model_validate(
        next(
            item.payload
            for item in planned.artifacts
            if item.type is ArtifactType.INVESTIGATION_PLAN
        )
    )
    assert "llm_probe_planner" in plan.allowed_tools
    assert any("Deep investigation enabled" in line for line in plan.assumptions)


# --------------------------------------------------------------------------- #
# Orchestrator wiring
# --------------------------------------------------------------------------- #
def _run_plan(
    source: AutoEDAResult,
    *,
    deep: bool,
    llm,  # noqa: ANN001
):
    planned = create_investigation_plans(
        project_id=source.project_id,
        source_session_id=source.session_id,
        question_ids=[_first_template_candidate_id(source)],
        workspace=source.workspace,
        session_id="plan_run",
        deep=deep,
    )
    plan_artifact = next(
        item for item in planned.artifacts if item.type is ArtifactType.INVESTIGATION_PLAN
    )
    approve_plan(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_id=plan_artifact.id,
        workspace=source.workspace,
    )
    return execute_investigation_plans(
        project_id=source.project_id,
        plan_session_id=planned.session_id,
        plan_ids=[plan_artifact.id],
        workspace=source.workspace,
        llm=llm,
    )


def test_deep_plan_end_to_end_merges_evidence_backed_probe_findings(tmp_path: Path) -> None:
    source = _source(tmp_path)
    llm = ScriptedOrchestratorLLM(
        probe_decisions=[
            {
                "action": "probe",
                "purpose": "largest order amount",
                "sql": "SELECT max(amount) AS max_amount FROM ecommerce_orders",
            },
            {"action": "conclude", "rationale": "The peak order is consistent; stopping."},
        ],
        interpretation="The result is consistent with the observed pattern.",
    )
    completed = _run_plan(source, deep=True, llm=llm)

    deep_artifact = next(
        item
        for item in completed.artifacts
        if item.type is ArtifactType.DEEP_INVESTIGATION_RESULT
    )
    finding_artifact = next(
        item for item in completed.artifacts if item.type is ArtifactType.VALIDATED_FINDING
    )
    finding = ValidatedFinding.model_validate(finding_artifact.payload)

    # The probe finding was merged, every merged finding carries evidence, and the
    # deep transcript is recorded as a source artifact.
    assert len(finding.findings) >= 2
    assert all(item.evidence for item in finding.findings)
    assert deep_artifact.id in finding.source_artifact_ids
    assert llm.probe_calls == 2
    assert llm.interp_calls >= 1


def test_merged_probe_evidence_refs_resolve_against_the_store(tmp_path: Path) -> None:
    """DI7-B item ④: probe findings execute through the loop's store-less engine, so
    their in-memory result ids never persist. After the deep artifact is stored, every
    merged evidence ref must be repointed at it (locator prefixed ``step{i}.``) so the
    whole ValidatedFinding resolves against the persisted store."""
    source = _source(tmp_path)
    llm = ScriptedOrchestratorLLM(
        probe_decisions=[
            {
                "action": "probe",
                "purpose": "largest order amount",
                "sql": "SELECT max(amount) AS max_amount FROM ecommerce_orders",
            },
            {"action": "conclude", "rationale": "The peak order is consistent; stopping."},
        ],
        interpretation="The result is consistent with the observed pattern.",
    )
    completed = _run_plan(source, deep=True, llm=llm)

    deep_artifact = next(
        item
        for item in completed.artifacts
        if item.type is ArtifactType.DEEP_INVESTIGATION_RESULT
    )
    finding_artifact = next(
        item for item in completed.artifacts if item.type is ArtifactType.VALIDATED_FINDING
    )
    finding = ValidatedFinding.model_validate(finding_artifact.payload)

    store = ArtifactStore(source.workspace)
    # Every merged evidence ref must resolve to a persisted artifact.
    all_refs = [ref for item in finding.findings for ref in item.evidence]
    assert all_refs
    for ref in all_refs:
        assert ref.artifact_id is not None
        store.get_artifact(ref.artifact_id)  # raises KeyError if unresolved

    # At least one ref is a repointed probe ref: it targets the persisted deep
    # artifact and its locator carries the step-index prefix.
    probe_refs = [
        ref
        for ref in all_refs
        if ref.artifact_id == deep_artifact.id and ref.locator.startswith("step")
    ]
    assert probe_refs, "expected merged probe refs repointed at the deep artifact"

    # The record persisted at the marker id carries the same resolvable refs.
    record_artifact = next(
        item
        for item in completed.artifacts
        if item.type is ArtifactType.INVESTIGATION_RECORD
        and item.payload.get("finding_artifact_id") == finding_artifact.id
    )
    for ref in record_artifact.evidence:
        if ref.artifact_id is not None:
            store.get_artifact(ref.artifact_id)


def test_non_deep_plan_is_unchanged(tmp_path: Path) -> None:
    source = _source(tmp_path)
    llm = ScriptedOrchestratorLLM(
        probe_decisions=[],
        interpretation="The result is consistent with the observed pattern.",
    )
    completed = _run_plan(source, deep=False, llm=llm)

    assert not any(
        item.type is ArtifactType.DEEP_INVESTIGATION_RESULT for item in completed.artifacts
    )
    assert any(item.type is ArtifactType.VALIDATED_FINDING for item in completed.artifacts)
    # The bounded loop was never planned.
    assert llm.probe_calls == 0

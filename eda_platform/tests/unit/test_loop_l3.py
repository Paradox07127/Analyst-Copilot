"""L3 of the analysis macro-loop: orchestrator return edge, typed exits, ledger artifact.

Design source: docs/archive/2026-07/base/eda-agent-platform-analysis-loop-design-2026-07-23.md
(§2 return edge, §5.2 round ledger, §8 validation bridge / pre-authorization).
Mock LLM + synthetic artifacts only; no live sessions.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest

import eda_platform.drivers.investigation_orchestrator as orchestrator
from eda_platform.core.budget import SessionBudgetPolicy
from eda_platform.core.ids import make_artifact_id, stable_hash
from eda_platform.core.kernel import SessionCancelled
from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.loop_fingerprint import question_fingerprint
from eda_platform.core.session_metrics import summarize_session
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.investigation_orchestrator import (
    MACRO_LOOP_ROUND_EVENT,
    create_investigation_plans,
    preauthorize_macro_loop,
    run_macro_loop,
)
from eda_platform.schemas.artifacts import (
    Artifact,
    ArtifactType,
    DatasetProfile,
    EvidenceRef,
    SqlResult,
)
from eda_platform.schemas.loop import LoopLedger, LoopRoundRecord
from eda_platform.schemas.questions import (
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionExecutionResult,
    QuestionFinding,
    QuestionScore,
)

_PROJECT = "proj_loop"
_SOURCE_RUN = "src_run"
_PLAN_RUN = "plan_run"
_BASE_QUESTION = "What is total revenue by region?"

# The bridge's deterministic fallback finding id for the base question.
_BASE_FINDING_ID = "finding_" + stable_hash({"question_id": "q_base"}, length=12)

# ------------------------------------------------------------------ helpers


class _TaskDispatchLLM:
    """Serves canned follow-up generations; refuses every other task.

    A list serves one response per generation round; a dict repeats forever.
    """

    def __init__(
        self, followup_response: dict[str, Any] | list[dict[str, Any]] | Exception
    ) -> None:
        self._followup = followup_response
        self.tasks: list[str] = []

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        self.tasks.append(task)
        if task == "l2_followup_generation":
            if isinstance(self._followup, Exception):
                raise self._followup
            response = self._followup.pop(0) if isinstance(self._followup, list) else self._followup
            return schema.model_validate(response)
        raise RuntimeError(f"task {task!r} is not served by this test double")

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


class _UsageTaskLLM(_TaskDispatchLLM):
    """Task-dispatch double that reports provider usage after every call."""

    def __init__(
        self,
        followup_response: dict[str, Any] | list[dict[str, Any]] | Exception,
        *,
        tokens_per_call: int,
    ) -> None:
        super().__init__(followup_response)
        self._tokens_per_call = tokens_per_call
        self._usage: LLMResultMetadata | None = None

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        result = super().structured(task=task, schema=schema, payload=payload)
        self._usage = LLMResultMetadata(
            provider="test",
            model="test",
            usage=LLMUsage(total_tokens=self._tokens_per_call),
        )
        return result

    def last_usage(self) -> LLMResultMetadata | None:
        return self._usage


class _UsageSequenceLLM:
    """Live-shaped provider double with fresh, call-scoped usage metadata."""

    def __init__(
        self,
        responses: list[dict[str, Any] | Exception],
        usage: list[LLMResultMetadata],
    ) -> None:
        self._responses = list(responses)
        self._usage_rows = list(usage)
        self._usage: LLMResultMetadata | None = None
        self.tasks: list[str] = []

    def structured(self, *, task: str, schema: type, payload: dict) -> Any:
        self.tasks.append(task)
        self._usage = self._usage_rows.pop(0)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)

    def text(self, *, task: str, payload: dict) -> str:
        raise RuntimeError(f"task {task!r} is not served by this test double")

    def last_usage(self) -> LLMResultMetadata | None:
        return self._usage


def _score() -> QuestionScore:
    return QuestionScore(
        data_availability=0.9,
        statistical_signal=0.6,
        quality_risk=0.2,
        join_risk=0.0,
        deterministic_score=0.7,
    )


def _setup(tmp_path: Path) -> tuple[ArtifactStore, Path]:
    """Synthetic source run (candidate set + profile + CSV) and a real plan run."""
    workspace = tmp_path / "ws"
    store = ArtifactStore(workspace)
    store.start_session(_PROJECT, _SOURCE_RUN)
    candidate = QuestionCandidate(
        question_id="q_base",
        question_en=_BASE_QUESTION,
        origin="template",
        template_id="agg_by_category",
        sql_template=("SELECT region, SUM(revenue) AS total_revenue FROM orders GROUP BY region"),
        target_datasets=["orders.csv"],
        score=_score(),
    )
    candidate_set = QuestionCandidateSet(candidates=[candidate])
    store.save_artifact(
        Artifact(
            id=make_artifact_id("qcand", {"session_id": _SOURCE_RUN}),
            type=ArtifactType.QUESTION_CANDIDATE_SET,
            project_id=_PROJECT,
            session_id=_SOURCE_RUN,
            payload=candidate_set.model_dump(mode="json"),
        )
    )
    profile = DatasetProfile(
        dataset_id="ds_orders",
        name="orders.csv",
        rows=4,
        columns=2,
        column_names=["region", "revenue"],
        dtypes={"region": "object", "revenue": "float64"},
        missing_values={"region": 0, "revenue": 0},
        missing_percent={"region": 0.0, "revenue": 0.0},
        numeric_columns=["revenue"],
        categorical_columns=["region"],
    )
    store.save_artifact(
        Artifact(
            id=make_artifact_id("profile", {"session_id": _SOURCE_RUN, "dataset": "orders.csv"}),
            type=ArtifactType.DATASET_PROFILE,
            project_id=_PROJECT,
            session_id=_SOURCE_RUN,
            payload=profile.model_dump(mode="json"),
        )
    )
    csv_path = store.project_dir(_PROJECT) / "uploads" / "ds_orders" / "v1" / "orders.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("region,revenue\nA,120.5\nB,80.0\nA,60.0\nB,40.0\n", encoding="utf-8")
    create_investigation_plans(
        project_id=_PROJECT,
        source_session_id=_SOURCE_RUN,
        question_ids=["q_base"],
        workspace=workspace,
        session_id=_PLAN_RUN,
    )
    return store, workspace


def _seed_execution_results(
    store: ArtifactStore, *, include_good: bool = True, include_bad: bool = True
) -> None:
    """Persist synthetic execution results on the plan run for the bridge to judge."""
    sql_result = SqlResult(
        sql="SELECT region, SUM(revenue) AS total_revenue FROM orders GROUP BY region",
        columns=["region", "total_revenue"],
        dtypes={"region": "object", "total_revenue": "float64"},
        rows_preview=[
            {"region": "A", "total_revenue": 180.5},
            {"region": "B", "total_revenue": 120.0},
        ],
        row_count=2,
    )
    sql_artifact = Artifact(
        id=make_artifact_id("sqlres", {"session_id": _PLAN_RUN, "question": "q_base"}),
        type=ArtifactType.SQL_RESULT,
        project_id=_PROJECT,
        session_id=_PLAN_RUN,
        payload=sql_result.model_dump(mode="json"),
    )
    store.save_artifact(sql_artifact)
    if include_good:
        good = QuestionExecutionResult(
            question_id="q_base",
            question=_BASE_QUESTION,
            origin="template",
            findings=[
                QuestionFinding(
                    text="Region A total revenue is 180.5.",
                    evidence=[
                        EvidenceRef(
                            kind="sql",
                            artifact_id=sql_artifact.id,
                            locator="rows[0].total_revenue",
                            value=180.5,
                        )
                    ],
                )
            ],
            status="succeeded",
        )
        store.save_artifact(
            Artifact(
                id=make_artifact_id("qexec", {"session_id": _PLAN_RUN, "question": "q_base"}),
                type=ArtifactType.QUESTION_EXECUTION_RESULT,
                project_id=_PROJECT,
                session_id=_PLAN_RUN,
                payload=good.model_dump(mode="json"),
            )
        )
    if include_bad:
        # Numbers that resolve against nothing: the numeric gate reports
        # "unverified", so the bridge must discard this result.
        bad = QuestionExecutionResult(
            question_id="q_unresolvable",
            question="Which region grew fastest?",
            origin="llm",
            findings=[
                QuestionFinding(
                    text="Region B revenue grew 999 units.",
                    evidence=[
                        EvidenceRef(
                            kind="sql",
                            artifact_id="sqlres_missing",
                            locator="rows[0].growth",
                            value=999,
                        )
                    ],
                )
            ],
            status="succeeded",
        )
        store.save_artifact(
            Artifact(
                id=make_artifact_id(
                    "qexec", {"session_id": _PLAN_RUN, "question": "q_unresolvable"}
                ),
                type=ArtifactType.QUESTION_EXECUTION_RESULT,
                project_id=_PROJECT,
                session_id=_PLAN_RUN,
                payload=bad.model_dump(mode="json"),
            )
        )


def _followup_payload() -> dict[str, Any]:
    return {
        "concluded": False,
        "proposals": [
            # Repeats the already-executed base question: fingerprint duplicate.
            {"question_text": _BASE_QUESTION, "parent_finding_id": _BASE_FINDING_ID},
            {
                "question_text": "Why does region A lead total revenue?",
                "parent_finding_id": _BASE_FINDING_ID,
            },
        ],
    }


def _run(workspace: Path, llm: Any, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "project_id": _PROJECT,
        "plan_session_id": _PLAN_RUN,
        "workspace": workspace,
        "llm": llm,
        "depth": 2,
    }
    kwargs.update(overrides)
    return run_macro_loop(**kwargs)


def _save_qexec(
    store: ArtifactStore,
    *,
    question_id: str,
    question: str,
    findings: list[QuestionFinding],
    status: Literal["succeeded", "failed"] = "succeeded",
) -> None:
    qexec = QuestionExecutionResult(
        question_id=question_id,
        question=question,
        origin="llm",
        findings=findings,
        status=status,
    )
    store.save_artifact(
        Artifact(
            id=make_artifact_id("qexec", {"session_id": _PLAN_RUN, "question": question_id}),
            type=ArtifactType.QUESTION_EXECUTION_RESULT,
            project_id=_PROJECT,
            session_id=_PLAN_RUN,
            payload=qexec.model_dump(mode="json"),
        )
    )


def _verifiable_followup_batch(
    candidates: list[QuestionCandidate], round_id: int
) -> list[Artifact]:
    """One succeeded follow-up result whose number resolves inside the batch."""
    candidate = candidates[0]
    value = 60.5 + round_id  # distinct per round so multi-round admissions never dedup
    session_id = f"{_PLAN_RUN}_macro_r{round_id}__internal"
    sql_result = SqlResult(
        sql="SELECT MAX(revenue) - MIN(revenue) AS lead_gap FROM orders",
        columns=["lead_gap"],
        dtypes={"lead_gap": "float64"},
        rows_preview=[{"lead_gap": value}],
        row_count=1,
    )
    sql_artifact = Artifact(
        id=make_artifact_id(
            "sqlres", {"session_id": session_id, "question": candidate.question_id}
        ),
        type=ArtifactType.SQL_RESULT,
        project_id=_PROJECT,
        session_id=session_id,
        payload=sql_result.model_dump(mode="json"),
    )
    qexec = QuestionExecutionResult(
        question_id=candidate.question_id,
        question=candidate.question_en,
        origin="llm",
        findings=[
            QuestionFinding(
                text=f"Region A leads by {value}.",
                evidence=[
                    EvidenceRef(
                        kind="sql",
                        artifact_id=sql_artifact.id,
                        locator="rows[0].lead_gap",
                        value=value,
                    )
                ],
            )
        ],
        status="succeeded",
    )
    qexec_artifact = Artifact(
        id=make_artifact_id("qexec", {"session_id": session_id, "question": candidate.question_id}),
        type=ArtifactType.QUESTION_EXECUTION_RESULT,
        project_id=_PROJECT,
        session_id=session_id,
        payload=qexec.model_dump(mode="json"),
    )
    return [sql_artifact, qexec_artifact]


def _patch_followup_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: list[int] | None = None,
    error: Exception | None = None,
) -> None:
    """Replace the funnel execution with a canned verifiable batch (or a crash)."""

    def fake(store: ArtifactStore, **kwargs: Any) -> tuple[list[Artifact], int, list[str]]:
        round_id = kwargs["round_id"]
        if calls is not None:
            calls.append(round_id)
        if error is not None:
            raise error
        session_ids = [
            f"{_PLAN_RUN}_macro_r{round_id}_source__internal",
            f"{_PLAN_RUN}_macro_r{round_id}__internal",
        ]
        candidates = list(kwargs["candidates"])
        return _verifiable_followup_batch(candidates, round_id), len(candidates), session_ids

    monkeypatch.setattr(orchestrator, "_execute_followup_round", fake)


def _workspace_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------------------- R1: depth 0/1 no-op


def test_depth0_and_depth1_are_noops(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    digest_before = _workspace_digest(workspace)
    for depth in (0, 1):
        assert _run(workspace, llm=None, depth=depth) is None
    # Byte-level no-op: every workspace file is untouched, none added or removed.
    assert _workspace_digest(workspace) == digest_before


# --------------------------------------- R2: one depth-2 round, bridge to execution


def test_depth2_round_bridges_dedupes_and_executes(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _TaskDispatchLLM(_followup_payload())
    result = _run(workspace, llm)
    assert result is not None
    # The follow-up executed but its result failed the bridge, so the loop
    # honestly reports no_new_information for its last round.
    assert result.exit_reason == "no_new_information"

    ledger = result.ledger
    assert ledger.validated_finding_ids == [_BASE_FINDING_ID]
    assert len(ledger.finding_fingerprints) == 1
    assert [row.round_id for row in ledger.rounds] == [0, 1]
    seed, executed = ledger.rounds
    assert seed.new_validated_findings == 1
    assert seed.discarded_findings == 1
    assert seed.redundant_findings == 0
    assert seed.disposition == "keep"
    assert seed.exit_reason == "continue"
    # One of the two proposals repeats an executed question: pruned by fingerprint.
    assert executed.executed_questions == 1
    assert executed.new_validated_findings == 0  # offline funnel execution failed validation
    assert executed.disposition == "discard"

    events = store.list_trace_events(project_id=_PROJECT, session_id=_PLAN_RUN)
    round_events = [e for e in events if e.event_type == MACRO_LOOP_ROUND_EVENT]
    assert len(round_events) == 2
    assert round_events[0].summary["new_validated_findings"] == 1
    assert round_events[-1].summary["exit_reason"] == "no_new_information"

    # The follow-up went through the existing funnel: a plan artifact plus a
    # fingerprint-bound approval exist on the macro round's internal plan run.
    macro_artifacts = store.list_artifacts(
        project_id=_PROJECT, session_id=f"{_PLAN_RUN}_macro_r1__internal"
    )
    macro_types = {artifact.type for artifact in macro_artifacts}
    assert ArtifactType.INVESTIGATION_PLAN in macro_types
    assert ArtifactType.INVESTIGATION_APPROVAL in macro_types


def test_macro_loop_cancellation_after_generation_skips_execution_and_terminal_ledger(
    tmp_path: Path,
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT,
        plan_session_id=_PLAN_RUN,
        workspace=workspace,
        depth=2,
    )
    llm = _TaskDispatchLLM(_followup_payload())

    with pytest.raises(SessionCancelled):
        _run(
            workspace,
            llm,
            cancel_check=lambda: bool(llm.tasks),
        )

    assert llm.tasks == ["l2_followup_generation"]
    artifacts = store.list_artifacts(project_id=_PROJECT, session_id=_PLAN_RUN)
    assert not any(artifact.type is ArtifactType.LOOP_LEDGER for artifact in artifacts)


# ----------------------------------------------------------- R3: typed exits


def test_concluded_bottom_symbol_exits(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _TaskDispatchLLM({"concluded": True, "conclusion_reason": "Nothing left."})
    result = _run(workspace, llm)
    assert result.exit_reason == "concluded"
    # Seed bridge row (round 0) plus the concluded round.
    assert [row.round_id for row in result.ledger.rounds] == [0, 1]
    assert result.ledger.rounds[-1].exit_reason == "concluded"
    assert result.ledger.rounds[-1].executed_questions == 0


def test_zero_new_findings_exits_no_new_information(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store, include_good=False)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _TaskDispatchLLM(_followup_payload())
    result = _run(workspace, llm)
    assert result.exit_reason == "no_new_information"
    row = result.ledger.rounds[0]
    assert row.new_validated_findings == 0
    assert row.discarded_findings == 1
    assert row.disposition == "discard"
    assert llm.tasks == []  # no follow-up generation for an empty round


def test_all_pruned_followups_exit_no_new_information(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _TaskDispatchLLM(
        {
            "concluded": False,
            "proposals": [{"question_text": _BASE_QUESTION, "parent_finding_id": _BASE_FINDING_ID}],
        }
    )
    result = _run(workspace, llm)
    assert result.exit_reason == "no_new_information"
    assert result.ledger.rounds[0].executed_questions == 0
    assert result.ledger.rounds[0].disposition == "keep"  # the bridge still admitted one


def test_round_cap_is_the_default_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Depth 2 caps at one round: a productive round still ends at the cap.
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _patch_followup_execution(monkeypatch)
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    assert result.exit_reason == "round_cap"
    assert [row.round_id for row in result.ledger.rounds] == [0, 1]


def test_exhausted_budget_exits_budget_cap(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _TaskDispatchLLM(_followup_payload())
    result = _run(
        workspace,
        llm,
        budget_policy=SessionBudgetPolicy(max_requests=0),
    )
    assert result.exit_reason == "budget_cap"
    assert result.ledger.rounds[0].exit_reason == "budget_cap"
    assert llm.tasks == []  # exhausted before any spend


def test_followup_crash_records_crash_round_without_raising(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    # TypeError is outside the follow-up agent's fail-safe net, so it crashes the round.
    result = _run(workspace, _TaskDispatchLLM(TypeError("boom")))
    assert result is not None
    assert result.exit_reason == "crash"
    row = result.ledger.rounds[-1]
    assert row.exit_reason == "crash"
    assert row.disposition == "crash"
    # The ledger still persists on a crashed run.
    ledger_artifact = store.get_artifact(result.ledger_artifact_id)
    assert ledger_artifact.type is ArtifactType.LOOP_LEDGER


# ------------------------------------- R4: ledger persistence + SessionMetrics rollup


def test_loop_ledger_artifact_round_trips(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    artifact = store.get_artifact(result.ledger_artifact_id)
    assert artifact.type is ArtifactType.LOOP_LEDGER
    assert artifact.session_id == _PLAN_RUN
    assert LoopLedger.model_validate(artifact.payload) == result.ledger


def test_run_metrics_rollup_derives_from_loop_ledger(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _run(workspace, _TaskDispatchLLM(_followup_payload()))
    metrics = summarize_session(store, _PROJECT, _PLAN_RUN)
    # Seed round (keep) + the executed round whose result failed the bridge (discard).
    assert metrics.macro_loop_rounds == 2
    assert metrics.macro_loop_new_findings == 1
    assert metrics.macro_loop_discard_rounds == 1


def test_run_metrics_counts_discard_rounds(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store, include_good=False)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _run(workspace, _TaskDispatchLLM(_followup_payload()))
    metrics = summarize_session(store, _PROJECT, _PLAN_RUN)
    assert metrics.macro_loop_rounds == 1
    assert metrics.macro_loop_new_findings == 0
    assert metrics.macro_loop_discard_rounds == 1


# ------------------------------------------------- R5: pre-authorization gate


def test_depth2_without_preauthorization_is_refused(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    with pytest.raises(ValueError, match="pre-authorization"):
        _run(workspace, _TaskDispatchLLM(_followup_payload()))


def test_preauthorization_is_depth_bound(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=3
    )
    # A depth-3 grant does not authorize a depth-2 loop: fingerprints differ.
    with pytest.raises(ValueError, match="pre-authorization"):
        _run(workspace, _TaskDispatchLLM(_followup_payload()), depth=2)
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()), depth=3)
    assert result is not None


def test_preauthorize_requires_depth_at_least_two(tmp_path: Path) -> None:
    _store, workspace = _setup(tmp_path)
    with pytest.raises(ValueError, match="depth >= 2"):
        preauthorize_macro_loop(
            project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=1
        )


# ------------------------------- R6: every executed round is bridged (finding 1)


def test_last_round_results_are_bridged_before_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _patch_followup_execution(monkeypatch)
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    assert result.exit_reason == "round_cap"
    # Seed admission + the last (only) round's executed result both reached the ledger.
    assert len(result.ledger.finding_fingerprints) == 2
    assert len(result.ledger.validated_finding_ids) == 2
    executed_round = result.ledger.rounds[-1]
    assert executed_round.round_id == 1
    assert executed_round.executed_questions == 1
    assert executed_round.new_validated_findings == 1
    assert executed_round.disposition == "keep"


def test_depth3_bridges_every_round_including_the_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=3
    )
    _patch_followup_execution(monkeypatch)
    responses = [
        {
            "concluded": False,
            "proposals": [{"question_text": text, "parent_finding_id": _BASE_FINDING_ID}],
        }
        for text in (
            "Why does region A lead total revenue?",
            "How concentrated is revenue within region A?",
            "Does region A lead order counts too?",
        )
    ]
    result = _run(workspace, _TaskDispatchLLM(responses), depth=3)
    assert result.exit_reason == "round_cap"
    executed_rounds = [row for row in result.ledger.rounds if row.executed_questions]
    assert [row.round_id for row in executed_rounds] == [1, 2, 3]
    assert all(row.new_validated_findings == 1 for row in executed_rounds)
    # Seed + three per-round admissions: the third round's result is on the ledger.
    assert len(result.ledger.finding_fingerprints) == 4


# ------------------------- R7: no_numbers claims need resolvable evidence (finding 2)


def test_bridge_discards_no_number_findings_without_resolvable_evidence(
    tmp_path: Path,
) -> None:
    store, workspace = _setup(tmp_path)
    _save_qexec(
        store,
        question_id="q_no_evidence",
        question="Which region leads revenue?",
        findings=[QuestionFinding(text="Region A leads total revenue overall.", evidence=[])],
    )
    _save_qexec(
        store,
        question_id="q_dangling",
        question="Which region trails revenue?",
        findings=[
            QuestionFinding(
                text="Region B trails the pack.",
                evidence=[
                    EvidenceRef(kind="sql", artifact_id="sqlres_missing", locator="rows[0].region")
                ],
            )
        ],
    )
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    result = _run(workspace, _TaskDispatchLLM({"concluded": True}))
    assert result.exit_reason == "no_new_information"
    row = result.ledger.rounds[0]
    assert row.new_validated_findings == 0
    assert row.discarded_findings == 2
    assert result.ledger.finding_fingerprints == []


# --------------------- R8: fingerprint family key separates questions (finding 3)


def test_same_evidence_from_different_questions_is_not_deduped(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store, include_bad=False)
    # A different question citing the exact same evidence values as q_base.
    sql_artifact_id = make_artifact_id("sqlres", {"session_id": _PLAN_RUN, "question": "q_base"})
    _save_qexec(
        store,
        question_id="q_other",
        question="How large is region A's revenue in absolute terms?",
        findings=[
            QuestionFinding(
                text="Region A books 180.5 in revenue.",
                evidence=[
                    EvidenceRef(
                        kind="sql",
                        artifact_id=sql_artifact_id,
                        locator="rows[0].total_revenue",
                        value=180.5,
                    )
                ],
            )
        ],
    )
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    result = _run(workspace, _TaskDispatchLLM({"concluded": True}))
    row = result.ledger.rounds[0]
    assert row.new_validated_findings == 2
    assert row.redundant_findings == 0
    assert len(set(result.ledger.finding_fingerprints)) == 2


# ------------------------------------- R9: budget gates inside a round (finding 4)


def test_round_tokens_and_cost_come_from_retry_ledger_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _patch_followup_execution(monkeypatch)
    llm = _UsageSequenceLLM(
        [
            {"proposals": "not-a-list"},
            _followup_payload(),
        ],
        [
            LLMResultMetadata(
                provider="test",
                model="test",
                usage=LLMUsage(
                    prompt_tokens=6,
                    completion_tokens=4,
                    total_tokens=10,
                    cached_tokens=2,
                ),
                estimated_cost_usd=0.01,
            ),
            LLMResultMetadata(
                provider="test",
                model="test",
                usage=LLMUsage(
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                    cached_tokens=3,
                ),
                estimated_cost_usd=0.02,
            ),
        ],
    )
    result = _run(workspace, llm)
    usage_events = [
        event
        for event in store.list_trace_events(project_id=_PROJECT, session_id=_PLAN_RUN)
        if event.event_type == "llm_usage" and event.summary.get("task") == "l2_followup_generation"
    ]
    assert len(usage_events) == 2
    assert sum(int(event.summary["total_tokens"]) for event in usage_events) == 30
    assert sum(int(event.summary["cached_tokens"]) for event in usage_events) == 5
    total_cost = sum(float(event.summary["estimated_cost_usd"]) for event in usage_events)
    assert total_cost == pytest.approx(0.03)
    assert result.ledger.rounds[-1].tokens == 30


def test_budget_gate_blocks_execution_within_a_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    calls: list[int] = []
    _patch_followup_execution(monkeypatch, calls=calls)
    llm = _UsageTaskLLM(_followup_payload(), tokens_per_call=120)
    result = _run(
        workspace,
        llm,
        budget_policy=SessionBudgetPolicy(max_requests=0),
    )
    # An exhausted restored budget blocks generation and execution.
    assert calls == []
    assert llm.tasks == []
    assert result.exit_reason == "budget_cap"
    assert result.ledger.rounds[-1].tokens == 0


def test_failed_provider_call_is_recorded_by_unified_ledger(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    llm = _UsageSequenceLLM(
        [RuntimeError("provider failed after burning tokens")],
        [
            LLMResultMetadata(
                provider="test",
                model="test",
                usage=LLMUsage(
                    prompt_tokens=40,
                    completion_tokens=30,
                    total_tokens=70,
                    cached_tokens=5,
                ),
                estimated_cost_usd=0.07,
            )
        ],
    )
    result = _run(workspace, llm)
    usage_events = [
        event
        for event in store.list_trace_events(project_id=_PROJECT, session_id=_PLAN_RUN)
        if event.event_type == "llm_usage"
    ]
    assert len(usage_events) == 1
    assert usage_events[0].summary["status"] == "RuntimeError"
    assert usage_events[0].summary["total_tokens"] == 70
    assert result.ledger.rounds[-1].tokens == 70


def test_macro_loop_restart_restores_request_budget(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    policy = SessionBudgetPolicy(max_requests=1)
    first_llm = _UsageSequenceLLM(
        [{"concluded": True, "conclusion_reason": "Complete."}],
        [
            LLMResultMetadata(
                provider="test",
                model="test",
                usage=LLMUsage(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                ),
                estimated_cost_usd=0.01,
            )
        ],
    )
    first = _run(workspace, first_llm, budget_policy=policy)
    assert first.exit_reason == "budget_cap"
    assert first_llm.tasks == ["l2_followup_generation"]

    restarted_llm = _UsageTaskLLM(_followup_payload(), tokens_per_call=10)
    restarted = _run(workspace, restarted_llm, budget_policy=policy)
    assert restarted.exit_reason == "budget_cap"
    assert restarted_llm.tasks == []
    usage_events = [
        event
        for event in store.list_trace_events(project_id=_PROJECT, session_id=_PLAN_RUN)
        if event.event_type == "llm_usage"
    ]
    assert len(usage_events) == 1


# -------------------------- R10: crash hygiene for question fingerprints (finding 5)


def test_crashed_round_leaves_no_dead_question_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    _patch_followup_execution(monkeypatch, error=RuntimeError("execution crashed mid-round"))
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    assert result.exit_reason == "crash"
    assert result.ledger.rounds[-1].disposition == "crash"
    fingerprints = set(result.ledger.question_fingerprints)
    assert question_fingerprint(_BASE_QUESTION) in fingerprints
    # The never-executed follow-up must not block a retry with a dead fingerprint.
    assert question_fingerprint("Why does region A lead total revenue?") not in fingerprints


# ------------------------------------ R11: derived-run isolation (finding 6)


def test_derived_followup_runs_are_marked_internal(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    assert result.executed_session_ids
    assert all("__internal" in session_id for session_id in result.executed_session_ids)
    macro_artifacts = store.list_artifacts(
        project_id=_PROJECT, session_id=f"{_PLAN_RUN}_macro_r1__internal"
    )
    assert ArtifactType.INVESTIGATION_PLAN in {artifact.type for artifact in macro_artifacts}


def test_run_metrics_counts_only_latest_loop_ledger(tmp_path: Path) -> None:
    store, workspace = _setup(tmp_path)
    _seed_execution_results(store)
    preauthorize_macro_loop(
        project_id=_PROJECT, plan_session_id=_PLAN_RUN, workspace=workspace, depth=2
    )
    result = _run(workspace, _TaskDispatchLLM(_followup_payload()))
    # A stale ledger from an earlier loop invocation at another depth.
    stale = LoopLedger(
        depth=3,
        rounds=[
            LoopRoundRecord(round_id=1, new_validated_findings=2, disposition="keep"),
            LoopRoundRecord(round_id=2, new_validated_findings=1, disposition="keep"),
        ],
    )
    store.save_artifact(
        Artifact(
            id=make_artifact_id("loopledger", {"session_id": _PLAN_RUN, "depth": 3}),
            type=ArtifactType.LOOP_LEDGER,
            project_id=_PROJECT,
            session_id=_PLAN_RUN,
            created_at=datetime.now(UTC) - timedelta(hours=1),
            payload=stale.model_dump(mode="json"),
        )
    )
    metrics = summarize_session(store, _PROJECT, _PLAN_RUN)
    assert metrics.macro_loop_rounds == len(result.ledger.rounds)
    assert metrics.macro_loop_new_findings == sum(
        row.new_validated_findings for row in result.ledger.rounds
    )

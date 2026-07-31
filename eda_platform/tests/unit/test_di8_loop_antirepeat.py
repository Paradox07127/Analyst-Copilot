"""DI sprint-8 (DI8-A/L2): anti-rut mechanisms of the bounded investigation loop.

Two additive defenses on top of the unchanged hard caps (3 probes / 8 LLM calls /
2 consecutive errors):

- *repeated-action detection* — a probe whose normalized SQL fingerprint was
  already seen in this loop is rejected WITHOUT execution and the rejection is
  fed back to the LLM ("换方向 or conclude"); the second rejection forces the
  typed ``repeated_action`` exit;
- *error feed-back* — every probe failure is compressed to one line and the
  most recent 5 entries are injected into the next planning payload.

All LLMs are scripted fakes; every assertion runs against the typed transcript,
the captured planning payloads, or a spy trace sink.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from eda_platform.agents.investigation_loop import _build_payload, run_bounded_loop
from eda_platform.core.trace import (
    LOOP_FAILURE_HISTORY_INJECTED,
    PROBE_REPEATED_REJECTED,
)
from eda_platform.schemas.investigations import InvestigationPlan
from eda_platform.schemas.sessions import TraceEvent
from eda_platform.tools.loader import LoadedDataset, load_csv
from eda_platform.tools.sql_runner import SqlCatalog, build_catalog

T = TypeVar("T", bound=BaseModel)


class PayloadRecordingLLM:
    """Scripted probe/conclude decisions; records every planning payload."""

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.payloads: list[dict] = []
        self.structured_calls = 0

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.structured_calls += 1
        self.payloads.append(payload)
        if not self._decisions:
            raise RuntimeError("scripted loop LLM exhausted its decisions.")
        return schema(**self._decisions.pop(0))  # type: ignore[return-value]

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> None:
        return None


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


def _spy_engine(catalog: SqlCatalog) -> list[str]:
    """Record every SQL that reaches ``execute_select`` (i.e. is actually run)."""
    executed: list[str] = []
    original = catalog.engine.execute_select

    def spy(sql: str):  # noqa: ANN202
        executed.append(sql)
        return original(sql)

    catalog.engine.execute_select = spy  # type: ignore[method-assign]
    return executed


_PROBE = {"action": "probe", "purpose": "scan amount", "sql": "SELECT amount FROM tiny"}
# Same statement after normalization (case / whitespace / trailing semicolon).
_PROBE_COSMETIC_DUP = {
    "action": "probe",
    "purpose": "scan amount again",
    "sql": "select   AMOUNT   from  tiny ;",
}
_CONCLUDE = {"action": "conclude", "rationale": "Nothing further to add."}


# --------------------------------------------------------------------------- #
# Repeated-action detection
# --------------------------------------------------------------------------- #
def test_repeated_probe_is_rejected_without_execution_and_fed_back(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    executed = _spy_engine(catalog)
    llm = PayloadRecordingLLM([dict(_PROBE), dict(_PROBE_COSMETIC_DUP), dict(_CONCLUDE)])

    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )

    # The duplicate never reached the engine: exactly ONE probe executed
    # (``run_sql`` issues two engine statements per probe: count + preview).
    assert len(executed) == 2
    assert result.exit_reason == "concluded"
    assert [step.status for step in result.steps] == ["succeeded", "skipped", "succeeded"]
    rejected = result.steps[1]
    assert rejected.action == "probe"
    assert "different direction" in rejected.error

    # The rejection is fed back to the LLM in the NEXT planning payload and it
    # tells the model to change direction or conclude.
    followup_payload = llm.payloads[2]
    assert "different direction" in followup_payload["repeated_probe_notice"]
    assert "conclude" in followup_payload["repeated_probe_notice"]
    # The rejection also lands in the failure history feed-back.
    assert any("repeated probe" in entry for entry in followup_payload["failed_probes"])
    # The rejection round still consumed LLM budget (every planning call counts).
    assert result.llm_calls_used == 3
    # Hard error-cap counters are untouched by the new mechanism.
    assert result.probe_errors == 0


def test_second_repeat_forces_repeated_action_exit(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    executed = _spy_engine(catalog)
    llm = PayloadRecordingLLM([dict(_PROBE), dict(_PROBE), dict(_PROBE_COSMETIC_DUP)])

    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )

    assert result.exit_reason == "repeated_action"
    assert [step.status for step in result.steps] == ["succeeded", "skipped", "skipped"]
    # Only the first proposal ever executed (2 engine statements per probe);
    # both repeats were rejected dry.
    assert len(executed) == 2
    assert llm.structured_calls == 3
    assert result.probe_errors == 0


def test_repeat_rejections_do_not_consume_probe_slots(tmp_path: Path) -> None:
    """The hard step cap still buys exactly ``max_steps`` EXECUTED probes: a
    rejected duplicate costs LLM budget but no probe slot."""
    catalog, _ = _catalog(tmp_path)
    other_a = {"action": "probe", "purpose": "max", "sql": "SELECT max(amount) AS m FROM tiny"}
    other_b = {"action": "probe", "purpose": "min", "sql": "SELECT min(amount) AS m FROM tiny"}
    llm = PayloadRecordingLLM(
        [dict(_PROBE), dict(_PROBE_COSMETIC_DUP), dict(other_a), dict(other_b)]
    )

    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
        max_steps=3,
    )

    assert result.exit_reason == "step_cap_reached"
    executed_steps = [step for step in result.steps if step.status == "succeeded"]
    assert len(executed_steps) == 3
    assert [step.status for step in result.steps] == [
        "succeeded",
        "skipped",
        "succeeded",
        "succeeded",
    ]


# --------------------------------------------------------------------------- #
# Error feed-back
# --------------------------------------------------------------------------- #
def test_probe_failure_history_is_injected_into_next_payload(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    bad = {"action": "probe", "purpose": "peek", "sql": "SELECT * FROM secret_table"}
    llm = PayloadRecordingLLM([dict(bad), dict(_PROBE), dict(_CONCLUDE)])

    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )

    assert result.exit_reason == "concluded"
    # First planning round: no failures yet, so no history keys at all.
    assert "failed_probes" not in llm.payloads[0]
    assert "failed_probes_note" not in llm.payloads[0]
    # After the guard rejection, the next payload carries the compressed history.
    followup_payload = llm.payloads[1]
    assert "Do not repeat" in followup_payload["failed_probes_note"]
    entries = followup_payload["failed_probes"]
    assert len(entries) == 1
    assert "scope" in entries[0]
    assert entries[0].startswith("[fp:")
    # Each entry is compressed to one short line (token discipline).
    assert "\n" not in entries[0]
    assert len(entries[0]) <= 160


def test_failure_history_is_truncated_to_last_five_entries() -> None:
    history = [f"entry {index}" for index in range(7)]
    payload = _build_payload(
        _plan(),
        primary_findings=[],
        steps=[],
        allowed_tables={"tiny"},
        probes_remaining=3,
        llm_calls_remaining=8,
        failure_history=history,
    )
    assert payload["failed_probes"] == ["entry 2", "entry 3", "entry 4", "entry 5", "entry 6"]
    assert len(payload["failed_probes"]) == 5


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #
def test_trace_events_for_rejection_and_history_injection(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = PayloadRecordingLLM([dict(_PROBE), dict(_PROBE_COSMETIC_DUP), dict(_CONCLUDE)])
    events: list[TraceEvent] = []

    run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
        trace_sink=events.append,
    )

    rejected = [event for event in events if event.event_type == PROBE_REPEATED_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].summary["duplicate_of_step"] == 0
    assert rejected[0].summary["repeat_rejections"] == 1
    assert rejected[0].summary["fingerprint"]

    injected = [
        event for event in events if event.event_type == LOOP_FAILURE_HISTORY_INJECTED
    ]
    assert len(injected) == 1
    assert injected[0].summary["entries"] == 1
    assert injected[0].summary["chars"] > 0


def test_no_trace_sink_keeps_the_loop_silent_and_working(tmp_path: Path) -> None:
    catalog, _ = _catalog(tmp_path)
    llm = PayloadRecordingLLM([dict(_PROBE), dict(_PROBE_COSMETIC_DUP), dict(_CONCLUDE)])
    result = run_bounded_loop(
        llm,
        plan=_plan(),
        primary_findings=[],
        query_engine=catalog.engine,
        catalog=catalog,
    )
    assert result.exit_reason == "concluded"

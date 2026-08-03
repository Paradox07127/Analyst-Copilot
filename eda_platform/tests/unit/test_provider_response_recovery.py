"""A served-but-unusable provider answer must not end an exploration run.

Covers the provider seam (schema violation, malformed response shape) and the
transport seam (a timeout that never reached the model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar
from urllib import error

import pytest
from pydantic import BaseModel

from eda_platform.agents.exploration.supervisor import (
    CandidateBatch,
    PhaseContext,
    SupervisorPhase,
)
from eda_platform.agents.exploration.workflow import JournaledCandidateGenerator
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.exploration_profiles import build_exploration_policy
from eda_platform.core.llm import (
    LLMSettings,
    MalformedProviderResponseError,
    OpenAICompatibleLLMClient,
    _anthropic_tool_response,
    _openai_tool_response,
)
from eda_platform.core.provider_registry import LLMProvider
from eda_platform.drivers.exploration import JsonSupervisorRecoveryStore
from eda_platform.schemas.exploration import InsightFamily
from eda_platform.schemas.hypotheses import (
    HypothesisPredicate,
    HypothesisProposal,
    HypothesisProposalBatch,
)

T = TypeVar("T", bound=BaseModel)

WITNESS = "dsw1_" + "a" * 32
STEP_ID = "xpl-recover:round:0:generate"


def _proposal(index: int = 0) -> HypothesisProposal:
    return HypothesisProposal(
        statement=f"Revenue differs by region {index}.",
        rationale="Regional mix shifted.",
        expected_evidence="A group comparison with an interval.",
        falsification_conditions=(f"No detectable difference {index}.",),
        family=InsightFamily.DIAGNOSTIC,
        method_family="compare_groups",
        dataset_ids=("ds-1",),
        columns=("region", "revenue"),
        probe_kind=f"region_difference_{index}",
        predicate=HypothesisPredicate(
            metric="revenue", operator="differs", left_operand="region"
        ),
    )


def _settings() -> LLMSettings:
    return LLMSettings(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-5.6-terra",
        max_tokens=256,
    )


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def _chat_completion(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {},
    }


# --- U1: a schema violation is a rejection, not an unknown outcome ------------


def test_thirteen_proposals_are_a_malformed_response_not_a_raw_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`proposals` is capped at 12. A 13-item answer is a served answer that
    breaks the contract, so it must surface as the recoverable provider error
    the executor and generator already know how to retry."""
    over_cap = json.dumps(
        {
            "concluded": False,
            "proposals": [
                json.loads(_proposal(index).model_dump_json()) for index in range(13)
            ],
        }
    )
    monkeypatch.setattr(
        "eda_platform.core.llm.request.urlopen",
        lambda req, timeout=0: _Resp(_chat_completion(over_cap)),  # noqa: ARG005
    )

    with pytest.raises(MalformedProviderResponseError):
        OpenAICompatibleLLMClient(_settings()).structured(
            task="t", schema=HypothesisProposalBatch, payload={}
        )


def _initialized_journal(tmp_path: Path) -> JsonlExplorationJournal:
    journal = JsonlExplorationJournal(tmp_path / "journal.jsonl")
    journal.initialize(
        exploration_id="xpl-recover",
        policy=build_exploration_policy(
            tier="quick",
            dataset_scope=("ds-1",),
            tool_capability_digest="tools-v1",
        ),
        code_fingerprint="code-v1",
        data_state_witness=WITNESS,
    )
    journal.claim_recovery()
    return journal


def _context() -> PhaseContext:
    return PhaseContext(
        exploration_id="xpl-recover",
        round_index=0,
        phase=SupervisorPhase.GENERATE,
        data_state_witness=WITNESS,
        soft_countdown_context="remaining={}",
        completed_step_ids=frozenset(),
    )


class _StructuredProvider:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.payloads: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict[str, Any]) -> T:
        del task
        self.payloads.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return schema.model_validate(outcome.model_dump(mode="json"))

    def text(self, *, task: str, payload: dict[str, Any]) -> str:
        raise AssertionError("unexpected text call")

    def last_usage(self) -> None:
        return None


def test_a_schema_violation_is_retried_once_with_the_error_fed_back(
    tmp_path: Path,
) -> None:
    journal = _initialized_journal(tmp_path)
    provider = _StructuredProvider(
        MalformedProviderResponseError(
            "structured response violates HypothesisProposalBatch: "
            "proposals: List should have at most 12 items"
        ),
        HypothesisProposalBatch(proposals=(_proposal(),)),
    )
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    batch = generator.generate(_context(), logical_step_id=STEP_ID)

    assert len(batch.candidates) == 1
    assert len(provider.payloads) == 2
    assert "at most 12 items" in json.dumps(provider.payloads[1])
    assert [event.event_type for event in journal.events()][-4:] == [
        "llm_call_started",
        "llm_call_rejected",
        "llm_call_started",
        "llm_call_completed",
    ]


def test_a_second_schema_violation_concludes_the_round_instead_of_killing_the_run(
    tmp_path: Path,
) -> None:
    journal = _initialized_journal(tmp_path)
    provider = _StructuredProvider(
        MalformedProviderResponseError("proposals: too many items"),
        MalformedProviderResponseError("proposals: too many items"),
    )
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    batch = generator.generate(_context(), logical_step_id=STEP_ID)

    assert isinstance(batch, CandidateBatch)
    assert batch.candidates == ()
    assert len(provider.payloads) == 2
    # The final attempt's call is settled by `llm_call_completed`, whose digest
    # covers the durable concluded batch; journaling a second rejection would
    # leave the generate step with no way to be marked durably complete.
    event_types = [event.event_type for event in journal.events()]
    assert event_types.count("llm_call_rejected") == 1
    assert event_types[-1] == "llm_call_completed"


def test_an_unknown_provider_failure_is_still_uncertain_and_terminal(
    tmp_path: Path,
) -> None:
    """Control: a transport failure leaves the outcome unknown, so the logical
    call must latch closed exactly as before."""
    journal = _initialized_journal(tmp_path)
    provider = _StructuredProvider(ConnectionError("connection reset by peer"))
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    with pytest.raises(ConnectionError):
        generator.generate(_context(), logical_step_id=STEP_ID)

    assert len(provider.payloads) == 1
    assert [event.event_type for event in journal.events()][-1] == "llm_call_uncertain"


def test_the_generate_instruction_states_the_proposal_cap(tmp_path: Path) -> None:
    journal = _initialized_journal(tmp_path)
    provider = _StructuredProvider(HypothesisProposalBatch(proposals=(_proposal(),)))
    generator = JournaledCandidateGenerator(
        provider=provider,
        journal=journal,
        recovery=JsonSupervisorRecoveryStore(tmp_path / "responses"),
        dataset_profiles=(),
    )

    generator.generate(_context(), logical_step_id=STEP_ID)

    assert "12" in provider.payloads[0]["instruction"]


# --- U2: a malformed response shape is a rejection ---------------------------


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"tool_calls": {}}}]},
        {"choices": [{"message": {"tool_calls": ["not-a-dict"]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"name": ""}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": "nope"}]}}]},
        {"choices": [{"message": "nope"}]},
        {"choices": []},
    ],
)
def test_a_malformed_openai_response_shape_is_a_provider_rejection(
    response: dict[str, Any],
) -> None:
    with pytest.raises(MalformedProviderResponseError):
        _openai_tool_response(response)


@pytest.mark.parametrize(
    "response",
    [
        {"content": "not-a-list"},
        {"content": [{"type": "tool_use", "name": "", "input": {}}]},
        {"content": []},
        {"content": [{"type": "thinking"}]},
    ],
)
def test_a_malformed_anthropic_response_shape_is_a_provider_rejection(
    response: dict[str, Any],
) -> None:
    with pytest.raises(MalformedProviderResponseError):
        _anthropic_tool_response(response)


# --- U8: a transport error that never reached the model is retried -----------


class _FlakyEndpoint:
    def __init__(self, *failures: BaseException) -> None:
        self.failures = list(failures)
        self.attempts = 0

    def __call__(self, req: Any, timeout: float = 0) -> Any:
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Resp(_chat_completion("ok"))


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "eda_platform.core.llm._TRANSPORT_RETRY_BACKOFF_SECONDS",
        (0.0, 0.0),
        raising=False,
    )


def test_two_transport_failures_are_retried_and_the_third_attempt_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _FlakyEndpoint(
        TimeoutError("timed out"),
        error.URLError("connection refused"),
    )
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", endpoint)

    text = OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    assert text == "ok"
    assert endpoint.attempts == 3


def test_a_transport_failure_that_never_clears_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: after the bounded retries the outcome is genuinely unknown, so
    the caller must still see a failure and latch the logical call closed."""
    endpoint = _FlakyEndpoint(
        TimeoutError("timed out"),
        TimeoutError("timed out"),
        TimeoutError("timed out"),
    )
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", endpoint)

    with pytest.raises(RuntimeError, match="did not respond within"):
        OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    assert endpoint.attempts == 3


def test_an_http_rejection_is_never_retried_as_a_transport_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: a 401 is a served response. Retrying it would multiply cost and
    hide the real cause."""
    detail = json.dumps({"error": {"message": "Incorrect API key provided."}})

    class _Unauthorized:
        def __init__(self) -> None:
            self.attempts = 0

        def __call__(self, req: Any, timeout: float = 0) -> Any:
            self.attempts += 1
            exc = error.HTTPError(
                url="https://api.example.com/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )
            exc.read = lambda: detail.encode("utf-8")  # type: ignore[method-assign]
            raise exc

    endpoint = _Unauthorized()
    monkeypatch.setattr("eda_platform.core.llm.request.urlopen", endpoint)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        OpenAICompatibleLLMClient(_settings()).text(task="t", payload={})

    assert endpoint.attempts == 1

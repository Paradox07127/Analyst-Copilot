"""E4b service contracts over the authoritative exploration journal."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from exploration_test_helpers import (
    TEST_BINDINGS,
    TEST_CAPS,
    TEST_RUNTIME_IDENTITY,
    TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
    release_certificate,
)

from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.exploration_service import (
    EXPLORATION_RELEASE_CERTIFICATE_ENV,
    EXPLORATION_RELEASE_TRUSTED_KEYS_ENV,
    ExplorationReleaseUnavailableError,
    ExplorationService,
    ExplorationSourceChangedError,
    ExplorationSourceSnapshot,
    assert_certificate_matches_runtime,
    load_configured_release_certificate,
    operator_pinned_release_public_keys,
    resolve_configured_release_trust,
)
from eda_platform.application.services.job_service import JobService
from eda_platform.core.exploration_journal import JsonlExplorationJournal
from eda_platform.core.exploration_shadow_store import shadow_run_root
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.exploration_budget import BudgetCapIncrease

SOURCE_SESSION_ID = "run_source"
PROJECT_ID = "demo"
DATASET_IDS = ("ds_orders",)
WITNESS = "dsw1_" + "a" * 32


class _RecordingBackend:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)

    def status(self, job_id: str) -> str:
        return "queued"


@dataclass
class _MutableSource:
    witness: str = WITNESS

    def __call__(
        self, session_id: str, dataset_ids: tuple[str, ...]
    ) -> ExplorationSourceSnapshot:
        assert session_id == SOURCE_SESSION_ID
        assert dataset_ids == DATASET_IDS
        return ExplorationSourceSnapshot(
            project_id=PROJECT_ID,
            dataset_ids=dataset_ids,
            data_state_witness=self.witness,
        )


@dataclass
class _Fixture:
    store: ArtifactStore
    backend: _RecordingBackend
    source: _MutableSource
    service: ExplorationService


@pytest.fixture
def exploration(tmp_path: Path) -> _Fixture:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT_ID, name="Demo")
    backend = _RecordingBackend(store)
    source = _MutableSource()
    service = ExplorationService(
        store,
        ApprovalService(store),
        JobService(store, backend),
        release_certificate=release_certificate(),
        trusted_release_public_keys=TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
        source_snapshot_resolver=source,
    )
    return _Fixture(store=store, backend=backend, source=source, service=service)


def _prepare_and_start(fixture: _Fixture, *, key: str = "start-key"):
    prepared = fixture.service.prepare(
        SOURCE_SESSION_ID,
        mode="open",
        goal=None,
        dataset_ids=DATASET_IDS,
        thinking_level="quick",
        provider="openai",
    )
    started = fixture.service.start(
        SOURCE_SESSION_ID,
        action_hash=prepared.action_hash,
        approval_token=prepared.approval_token,
        provider="openai",
        payload_policy="schema+aggregates",
        llm_env={"OPENAI_API_KEY": "secret-never-in-params"},
        idempotency_key=key,
    )
    return prepared, started


def _journal(fixture: _Fixture, exploration_id: str) -> JsonlExplorationJournal:
    return JsonlExplorationJournal(
        shadow_run_root(fixture.store.root, exploration_id) / "journal.jsonl"
    )


def _settle_job(fixture: _Fixture, job_id: str) -> None:
    assert JobLifecycleRepository(fixture.store).fail_active(
        job_id,
        error_code="test_settled",
        error_message="worker drained at pause checkpoint",
    )


def test_prepare_start_and_get_are_certificate_bound_and_journal_authoritative(
    exploration: _Fixture,
) -> None:
    prepared, started = _prepare_and_start(exploration)

    assert prepared.policy.thinking_level == "quick"
    assert prepared.cost_range.maximum_usd == prepared.policy.budget.llm.max_cost_usd
    assert prepared.release_certificate_digest == release_certificate().certificate_digest
    assert started.exploration.status == "running"
    assert started.exploration.last_seq == 0
    assert started.job.job_id == exploration.backend.commands[0].job_id
    assert exploration.backend.commands[0].kind == "exploration_run"
    params = json.loads(exploration.backend.commands[0].params_json)
    assert params["release_certificate_digest"] == prepared.release_certificate_digest
    assert "secret-never-in-params" not in exploration.backend.commands[0].params_json

    # A job projection can fail without rewriting exploration state.
    _settle_job(exploration, started.job.job_id)
    view = exploration.service.get(SOURCE_SESSION_ID, prepared.exploration_id)
    assert view.status == "running"
    assert view.job is not None and view.job.status == "failed"

    replay = exploration.service.start(
        SOURCE_SESSION_ID,
        action_hash=prepared.action_hash,
        approval_token=prepared.approval_token,
        provider="openai",
        payload_policy="schema+aggregates",
        llm_env={"OPENAI_API_KEY": "secret-never-in-params"},
        idempotency_key="start-key",
    )
    assert replay.job.job_id == started.job.job_id
    assert len(exploration.backend.commands) == 1


def test_budget_amendment_cannot_exceed_the_release_certificate(
    exploration: _Fixture,
) -> None:
    prepared, _started = _prepare_and_start(exploration)

    with pytest.raises(ExplorationReleaseUnavailableError, match="llm requests"):
        exploration.service.extend_budget(
            SOURCE_SESSION_ID,
            prepared.exploration_id,
            increase=BudgetCapIncrease(max_requests=29),
            reason="attempt to exceed certified hard cap",
            idempotency_key="over-certificate",
        )

    assert all(
        event.event_type != "budget_amended"
        for event in _journal(exploration, prepared.exploration_id).events()
    )


def test_missing_tampered_or_wrong_provider_certificate_fails_closed(
    exploration: _Fixture,
) -> None:
    closed = ExplorationService(
        exploration.store,
        ApprovalService(exploration.store),
        JobService(exploration.store, exploration.backend),
        release_certificate=None,
        trusted_release_public_keys=TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
        source_snapshot_resolver=exploration.source,
    )
    with pytest.raises(ExplorationReleaseUnavailableError):
        closed.prepare(
            SOURCE_SESSION_ID,
            mode="open",
            goal=None,
            dataset_ids=DATASET_IDS,
            thinking_level="quick",
            provider="openai",
        )

    untrusted = ExplorationService(
        exploration.store,
        ApprovalService(exploration.store),
        JobService(exploration.store, exploration.backend),
        release_certificate=release_certificate(),
        trusted_release_public_keys={},
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
        source_snapshot_resolver=exploration.source,
    )
    with pytest.raises(ExplorationReleaseUnavailableError, match="integrity"):
        untrusted.require_release_certificate()

    tampered = release_certificate().model_copy(update={"providers": ("anthropic",)})
    invalid = ExplorationService(
        exploration.store,
        ApprovalService(exploration.store),
        JobService(exploration.store, exploration.backend),
        release_certificate=tampered,
        trusted_release_public_keys=TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
        source_snapshot_resolver=exploration.source,
    )
    with pytest.raises(ExplorationReleaseUnavailableError, match="integrity"):
        invalid.require_release_certificate()
    with pytest.raises(ExplorationReleaseUnavailableError, match="not covered"):
        exploration.service.require_release_certificate(provider="anthropic")


def test_configured_certificate_loader_requires_a_pinned_release_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate = release_certificate()
    path = tmp_path / "release-certificate.json"
    path.write_text(certificate.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(EXPLORATION_RELEASE_CERTIFICATE_ENV, str(path))

    assert load_configured_release_certificate() is None
    loaded = load_configured_release_certificate(
        trusted_release_public_keys=TEST_TRUSTED_RELEASE_PUBLIC_KEYS,
        trusted_runtime_identity=TEST_RUNTIME_IDENTITY,
    )
    assert loaded == certificate


def _pinned_keys_value(key_id: str = "test-release-v1") -> str:
    return f"{key_id}:{TEST_TRUSTED_RELEASE_PUBLIC_KEYS[key_id].hex()}"


def test_operator_pinned_key_opens_the_gate_and_derives_the_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate = release_certificate()
    path = tmp_path / "release-certificate.json"
    path.write_text(certificate.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(EXPLORATION_RELEASE_CERTIFICATE_ENV, str(path))

    monkeypatch.delenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, raising=False)
    closed = resolve_configured_release_trust()
    assert closed.certificate is None
    assert closed.runtime_identity is None

    monkeypatch.setenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, _pinned_keys_value())
    opened = resolve_configured_release_trust()
    assert opened.certificate == certificate
    assert opened.runtime_identity == TEST_RUNTIME_IDENTITY


def test_operator_pinned_key_must_match_the_certificate_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate = release_certificate()
    path = tmp_path / "release-certificate.json"
    path.write_text(certificate.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv(EXPLORATION_RELEASE_CERTIFICATE_ENV, str(path))
    monkeypatch.setenv(
        EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, "test-release-v1:" + "00" * 32
    )

    assert resolve_configured_release_trust().certificate is None


@pytest.mark.parametrize(
    "value",
    ["not-a-pair", ":" + "00" * 32, "k:zz", "k:" + "00" * 31],
)
def test_malformed_pinned_trust_keys_raise_instead_of_downgrading(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXPLORATION_RELEASE_TRUSTED_KEYS_ENV, value)
    with pytest.raises(ValueError, match=EXPLORATION_RELEASE_TRUSTED_KEYS_ENV):
        operator_pinned_release_public_keys()


@pytest.mark.parametrize(
    "runtime",
    [
        replace(
            TEST_RUNTIME_IDENTITY,
            bindings=TEST_BINDINGS.model_copy(update={"checker_version": "other"}),
        ),
        replace(
            TEST_RUNTIME_IDENTITY,
            bindings=TEST_BINDINGS.model_copy(update={"code_fingerprint": "other"}),
        ),
        replace(
            TEST_RUNTIME_IDENTITY,
            bindings=TEST_BINDINGS.model_copy(
                update={"tool_capability_digest": "other"}
            ),
        ),
        replace(
            TEST_RUNTIME_IDENTITY,
            hard_caps=TEST_CAPS.model_copy(update={"max_llm_requests": 35}),
        ),
        replace(TEST_RUNTIME_IDENTITY, scoring_policy_version="other"),
        replace(TEST_RUNTIME_IDENTITY, statistical_policy_version="other"),
    ],
)
def test_certificate_must_match_every_trusted_runtime_identity(runtime) -> None:
    with pytest.raises(ValueError):
        assert_certificate_matches_runtime(release_certificate(), runtime)


def test_pause_is_not_stop_and_resume_witness_mismatch_stops_fail_closed(
    exploration: _Fixture,
) -> None:
    prepared, started = _prepare_and_start(exploration)
    paused_request = exploration.service.pause(
        SOURCE_SESSION_ID, prepared.exploration_id
    )
    assert paused_request.status == "pause_requested"
    assert paused_request.stop_reason is None

    journal = _journal(exploration, prepared.exploration_id)
    journal.append_new("paused")
    _settle_job(exploration, started.job.job_id)
    assert exploration.service.get(
        SOURCE_SESSION_ID, prepared.exploration_id
    ).status == "paused"

    exploration.source.witness = "dsw1_" + "b" * 32
    with pytest.raises(ExplorationSourceChangedError):
        exploration.service.resume(
            SOURCE_SESSION_ID,
            prepared.exploration_id,
            provider="openai",
            payload_policy="schema+aggregates",
            llm_env=None,
            idempotency_key="resume-key",
        )
    stopped = exploration.service.get(SOURCE_SESSION_ID, prepared.exploration_id)
    assert stopped.status == "stopped"
    assert stopped.stop_reason == "state_witness_changed"
    assert len(exploration.backend.commands) == 1


def test_resume_with_matching_witness_appends_resumed_and_queues_one_attempt(
    exploration: _Fixture,
) -> None:
    prepared, started = _prepare_and_start(exploration)
    exploration.service.pause(SOURCE_SESSION_ID, prepared.exploration_id)
    _journal(exploration, prepared.exploration_id).append_new("paused")
    _settle_job(exploration, started.job.job_id)

    resumed = exploration.service.resume(
        SOURCE_SESSION_ID,
        prepared.exploration_id,
        provider="openai",
        payload_policy="schema+aggregates",
        llm_env=None,
        idempotency_key="resume-key",
    )
    assert resumed.exploration.status == "running"
    assert resumed.exploration.last_seq == 3
    assert resumed.job.job_id != started.job.job_id
    assert len(exploration.backend.commands) == 2

    replay = exploration.service.resume(
        SOURCE_SESSION_ID,
        prepared.exploration_id,
        provider="openai",
        payload_policy="schema+aggregates",
        llm_env=None,
        idempotency_key="resume-key",
    )
    assert replay.job.job_id == resumed.job.job_id
    assert len(exploration.backend.commands) == 2


def test_budget_amendment_is_additive_system_derived_and_idempotent(
    exploration: _Fixture,
) -> None:
    prepared, _started = _prepare_and_start(exploration)
    increase = BudgetCapIncrease(max_requests=2, max_rounds=1)

    first = exploration.service.extend_budget(
        SOURCE_SESSION_ID,
        prepared.exploration_id,
        increase=increase,
        reason="User approved more coverage",
        idempotency_key="amend-key",
    )
    replay = exploration.service.extend_budget(
        SOURCE_SESSION_ID,
        prepared.exploration_id,
        increase=increase,
        reason="User approved more coverage",
        idempotency_key="amend-key",
    )

    assert first.amendment == replay.amendment
    assert first.amendment.amendment_id.startswith("xamend_")
    assert first.amendment.approved_by == "system:e4b-api"
    assert prepared.policy.budget.llm.max_requests is not None
    assert first.exploration.budget.max_llm_requests == (
        prepared.policy.budget.llm.max_requests + 2
    )
    events = _journal(exploration, prepared.exploration_id).events()
    assert [event.event_type for event in events].count("budget_amended") == 1
    assert first.effective_policy_fingerprint != prepared.policy.policy_fingerprint


def test_event_projection_uses_composite_journal_identity(
    exploration: _Fixture,
) -> None:
    prepared, _started = _prepare_and_start(exploration)
    exploration.service.pause(SOURCE_SESSION_ID, prepared.exploration_id)

    page = exploration.service.events_after(
        SOURCE_SESSION_ID, prepared.exploration_id, -1
    )
    assert [event.seq for event in page.events] == [0, 1]
    assert [event.event_id for event in page.events] == [
        f"{prepared.exploration_id}:0",
        f"{prepared.exploration_id}:1",
    ]


def _write_workflow_state(
    fixture: _Fixture, exploration_id: str, coverage: list[str]
) -> None:
    root = shadow_run_root(fixture.store.root, exploration_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow-state.json").write_text(
        json.dumps(
            {
                "committed_receipts": [],
                "admitted_bundles": [],
                "insights": [],
                "coverage_completed": coverage,
            }
        ),
        encoding="utf-8",
    )


def test_the_workflow_projection_is_reused_only_while_the_file_is_unchanged(
    exploration: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, _started = _prepare_and_start(exploration)
    exploration_id = prepared.exploration_id
    _write_workflow_state(exploration, exploration_id, ["region_difference"])

    reads = 0
    original = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal reads
        if self.name == "workflow-state.json":
            reads += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    first = exploration.service.get(SOURCE_SESSION_ID, exploration_id)
    assert first.coverage_completed == ("region_difference",)
    assert reads == 1

    # Every SSE frame refetches this view; an unchanged file must not be
    # re-parsed and every receipt in it re-verified.
    second = exploration.service.get(SOURCE_SESSION_ID, exploration_id)
    assert second.coverage_completed == ("region_difference",)
    assert reads == 1

    _write_workflow_state(exploration, exploration_id, ["region_difference", "spike_day"])
    third = exploration.service.get(SOURCE_SESSION_ID, exploration_id)
    assert third.coverage_completed == ("region_difference", "spike_day")
    assert reads == 2

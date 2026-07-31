from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import eda_platform.drivers.question_exec as question_exec_driver
from eda_platform.application.services.question_service import candidate_fingerprint
from eda_platform.core.cancellation import CancellationContext, cancellation_scope
from eda_platform.core.llm import CancellableLLMClient, OfflineLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository, LaunchClaim
from eda_platform.schemas.artifacts import Artifact, ArtifactType
from eda_platform.schemas.questions import QuestionCandidate, QuestionCandidateSet
from eda_platform.worker import runner

JOB_HANDLERS = {
    "auto_eda": "_run_auto_eda_job",
    "question_exec": "_run_question_exec_job",
    "skill_replay": "_run_skill_replay_job",
    "relationship_validate": "_run_relationship_validate_job",
    "relationship_discover": "_run_relationship_discover_job",
    "report_generate": "_run_report_generate_job",
    "session_fork": "_run_run_fork_job",
    "question_draft": "_run_question_draft_job",
    "investigation_plan": "_run_investigation_plan_job",
    "investigation_execute": "_run_investigation_execute_job",
    "macro_loop": "_run_macro_loop_job",
    "synthesis_brief_create": "_run_synthesis_brief_job",
    "decision_report_generate": "_run_decision_report_job",
}


def _launch(
    root: Path,
    *,
    job_id: str,
    kind: str,
    params: dict[str, Any] | None = None,
) -> tuple[ArtifactStore, JobLifecycleRepository, LaunchClaim]:
    store = ArtifactStore(root)
    store.ensure_project("demo", name="Demo")
    lifecycle = JobLifecycleRepository(store)
    lifecycle.create_queued_job(
        job_id=job_id,
        session_id=f"run_{job_id}",
        project_id="demo",
        kind=kind,
        params_json=json.dumps(params or {}),
        idempotency_key=None,
        lane_key=f"run_{job_id}",
        request_digest=f"digest-{job_id}",
        request_scope=f"run_{job_id}",
    )
    claim = lifecycle.claim_launch(job_id, owner="test")
    lifecycle.acknowledge_spawn(claim, pid=os.getpid(), birth_identity="test")
    return store, lifecycle, claim


@pytest.mark.parametrize("kind", JOB_HANDLERS)
def test_runner_forwards_cancel_callback_to_every_job_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    store, _lifecycle, claim = _launch(tmp_path, job_id=f"job_{kind}", kind=kind)
    observed: list[Callable[[], bool] | None] = []

    def handler(
        _store: ArtifactStore,
        _workspace: str,
        _job: dict,
        _params: dict,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        observed.append(cancel_check)
        assert cancel_check is not None
        assert cancel_check() is False

    monkeypatch.setattr(runner, JOB_HANDLERS[kind], handler)
    runner.run_job(
        str(tmp_path),
        claim.job_id,
        launch_token=claim.token,
        launch_attempt=claim.attempt,
    )

    assert len(observed) == 1
    final = store.get_job(claim.job_id)
    assert final is not None
    assert final["status"] == "completed"


def test_cancel_after_handler_before_terminal_cas_finishes_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, lifecycle, claim = _launch(
        tmp_path,
        job_id="job_terminal_window",
        kind="auto_eda",
    )

    def handler(
        _store: ArtifactStore,
        _workspace: str,
        _job: dict,
        _params: dict,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        assert cancel_check is not None
        assert lifecycle.request_cancel(claim.job_id)["status"] == "cancelling"

    monkeypatch.setattr(runner, "_run_auto_eda_job", handler)
    runner.run_job(
        str(tmp_path),
        claim.job_id,
        launch_token=claim.token,
        launch_attempt=claim.attempt,
    )

    final = store.get_job(claim.job_id)
    assert final is not None
    assert final["status"] == "cancelled"
    assert final["status"] != "completed"


@pytest.mark.parametrize("offline", [True, False])
def test_runner_build_llm_wraps_offline_and_live_clients_from_scope(
    monkeypatch: pytest.MonkeyPatch,
    offline: bool,
) -> None:
    provider = OfflineLLMClient()
    monkeypatch.setattr(runner, "create_llm_client", lambda _settings: provider)
    cancellation = CancellationContext()

    with cancellation_scope(cancellation):
        client = runner._build_llm({"llm": "offline"} if offline else {})

    assert isinstance(client, CancellableLLMClient)


def test_real_question_exec_durable_flag_finishes_cancelled_not_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_auto_eda(
        [Path(__file__).parents[1] / "golden" / "data" / "ecommerce_orders.csv"],
        workspace=tmp_path,
        project_id="demo",
        session_id="source_question_cancel",
        generate_report=False,
    )
    candidate_artifact = next(
        artifact
        for artifact in source.artifacts
        if artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
    )
    candidate = QuestionCandidateSet.model_validate(candidate_artifact.payload).candidates[0]
    params = {
        "source_session_id": source.session_id,
        "question_id": candidate.question_id,
        "candidate_fingerprint": candidate_fingerprint(candidate),
        "generate_report": False,
        "llm": "offline",
    }
    store, lifecycle, claim = _launch(
        tmp_path,
        job_id="job_question_running_cancel",
        kind="question_exec",
        params=params,
    )
    entered: list[str] = []

    def cancel_during_execution(
        executing: QuestionCandidate,
        **_kwargs: object,
    ) -> list[Artifact]:
        entered.append(executing.question_id)
        current = store.get_job(claim.job_id)
        assert current is not None and current["status"] == "running"
        assert lifecycle.request_cancel(claim.job_id)["status"] == "cancelling"
        return []

    monkeypatch.setattr(
        question_exec_driver,
        "execute_question_candidate",
        cancel_during_execution,
    )
    runner.run_job(
        str(tmp_path),
        claim.job_id,
        launch_token=claim.token,
        launch_attempt=claim.attempt,
    )

    assert entered == [candidate.question_id]
    final = store.get_job(claim.job_id)
    assert final is not None
    assert final["status"] == "cancelled"
    assert final["status"] != "completed"

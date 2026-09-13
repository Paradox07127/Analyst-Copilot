"""T7: chat turns must see conclusions from derived question runs.

Question executions land on qsess_* runs; the chat context previously loaded
only the source session's artifacts, so "what did my question conclude?" was
unanswerable."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

import eda_platform.drivers.chat as chat_driver_module
from eda_platform.application.services.approval_service import ApprovalService
from eda_platform.application.services.chat_service import ChatService
from eda_platform.core.store import ArtifactStore
from eda_platform.schemas.artifacts import Artifact, ArtifactType, DatasetProfile
from eda_platform.schemas.chat import ChatTurnResult
from eda_platform.schemas.plans import Intent
from eda_platform.schemas.sessions import SessionManifest

PROJECT = "proj_t7"
RUN = "run_t7"
DERIVED = "qsess_run_t7_1"
DATASET_ID = "ds_orders"
DATASET_NAME = "orders.csv"
DEADLINE_SECONDS = 10.0


def _profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id=DATASET_ID,
        name=DATASET_NAME,
        rows=2,
        columns=2,
        column_names=["region", "amount"],
        dtypes={"region": "object", "amount": "int64"},
        missing_values={"region": 0, "amount": 0},
        missing_percent={"region": 0.0, "amount": 0.0},
        numeric_columns=["amount"],
        categorical_columns=["region"],
    )


def _artifact(artifact_id: str, session_id: str, artifact_type: ArtifactType) -> Artifact:
    return Artifact(
        id=artifact_id,
        type=artifact_type,
        project_id=PROJECT,
        session_id=session_id,
        payload={"question_id": "q1", "status": "succeeded", "findings": []},
    )


def test_chat_turn_context_includes_derived_question_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_project(PROJECT, "T7")
    store.start_session(PROJECT, RUN)
    upload = tmp_path / "projects" / PROJECT / "uploads" / DATASET_ID / "v1" / DATASET_NAME
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_text("region,amount\nEast,10\nWest,20\n", encoding="utf-8")
    store.save_artifact(
        Artifact(
            id="prof_t7_orders",
            type=ArtifactType.DATASET_PROFILE,
            project_id=PROJECT,
            session_id=RUN,
            payload=_profile().model_dump(),
        )
    )

    store.start_session(PROJECT, DERIVED)
    store.write_manifest(
        SessionManifest(
            session_id=DERIVED,
            project_id=PROJECT,
            input_hashes={},
            code_version="test",
            source_session_id=RUN,
        )
    )
    store.save_artifact(
        _artifact("qexec_derived_1", DERIVED, ArtifactType.QUESTION_EXECUTION_RESULT)
    )
    # Raw derived output must stay behind: only structured results travel.
    store.save_artifact(_artifact("sqlres_derived_1", DERIVED, ArtifactType.SQL_RESULT))

    seen: list[list[str]] = []

    def capture(message: str, **kwargs: Any) -> ChatTurnResult:
        seen.append([artifact.id for artifact in kwargs.get("artifacts") or []])
        return ChatTurnResult(
            intent=Intent(kind="meta_help", confidence=1.0, raw_message=message),
            status="answer",
            message="ok",
        )

    monkeypatch.setattr(chat_driver_module, "run_chat_turn", capture)

    service = ChatService(store, ApprovalService(store))
    accepted = service.send_message(
        RUN, text="What did my follow-up question conclude?", llm="offline"
    )
    deadline = time.monotonic() + DEADLINE_SECONDS
    page = service.events_after(RUN, accepted.message_id, 0)
    while not page.done and time.monotonic() < deadline:
        time.sleep(0.02)
        page = service.events_after(RUN, accepted.message_id, 0)
    assert page.done, "the turn never settled"

    assert seen, "the turn never reached the driver"
    assert "prof_t7_orders" in seen[0]
    assert "qexec_derived_1" in seen[0]
    assert "sqlres_derived_1" not in seen[0]

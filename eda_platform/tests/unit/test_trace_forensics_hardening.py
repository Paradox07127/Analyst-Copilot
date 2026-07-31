"""Forensics reads must not leak a provider key that is no longer the active one,
must find `.env` without depending on the process CWD, and must stream exactly the
number of bytes the download declared."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services import trace_service
from eda_platform.application.services.trace_service import TraceService
from eda_platform.core.debug_log import DEBUG_LOG_FILENAME
from eda_platform.core.dev_log import LLM_DEBUG_FILENAME
from eda_platform.core.store import ArtifactStore

PROJECT = "demo"
RUN = "run_forensics"

ACTIVE_KEY = "sk-active-openai-key-000000001"
ROTATED_KEY = "sk-rotated-deepseek-key-00000002"
COMPATIBLE_KEY = "sk-old-compatible-key-000000003"

_KEY_VARS = (
    "EDA_LLM_API_KEY",
    "EDA_LLM_PROVIDER",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace nested under a repo-like directory, with the process CWD
    somewhere else entirely."""
    monkeypatch.setattr(trace_service, "DEFAULT_ENV_PATH", tmp_path / "missing.env")
    for name in _KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    repo = tmp_path / "repo"
    (repo / "workspace").mkdir(parents=True)
    return repo


@pytest.fixture
def store(workspace: Path) -> ArtifactStore:
    store = ArtifactStore(workspace / "workspace")
    store.ensure_project(PROJECT, name="Demo")
    store.start_session(PROJECT, RUN)
    return store


def _write_env(workspace: Path, body: str) -> None:
    (workspace / ".env").write_text(body, encoding="utf-8")


def _write_capture(store: ArtifactStore, status: str) -> None:
    path = store.session_dir(PROJECT, RUN) / LLM_DEBUG_FILENAME
    path.write_text(
        json.dumps(
            {
                "ts": "2026-07-25T10:00:00+00:00",
                "kind": "text",
                "task": "t",
                "status": status,
                "payload_preview": "",
                "response_preview": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_redacts_a_rotated_provider_key(workspace: Path, store: ArtifactStore) -> None:
    """The capture predates the current provider/key; every known key slot has to
    be masked, not just the one the active provider resolves to."""
    _write_env(
        workspace,
        f"EDA_LLM_PROVIDER=openai\n"
        f"EDA_LLM_API_KEY={ACTIVE_KEY}\n"
        f"DEEPSEEK_API_KEY={ROTATED_KEY}\n"
        f"OPENAI_COMPATIBLE_API_KEY={COMPATIBLE_KEY}\n",
    )
    _write_capture(store, f"error: AuthError: rejected {ROTATED_KEY} and {COMPATIBLE_KEY}")
    client = TestClient(create_app(store.root))
    item = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls").json()["items"][0]
    assert ROTATED_KEY not in item["status"]
    assert COMPATIBLE_KEY not in item["status"]
    assert item["status"].count("***") == 2


def test_env_file_is_found_from_the_workspace_not_the_cwd(
    workspace: Path, store: ArtifactStore
) -> None:
    """uvicorn started from another directory must not silently disable redaction."""
    _write_env(workspace, f"EDA_LLM_PROVIDER=openai\nEDA_LLM_API_KEY={ACTIVE_KEY}\n")
    _write_capture(store, f"error: AuthError: bad key {ACTIVE_KEY}")
    client = TestClient(create_app(store.root))
    item = client.get(f"/api/v1/sessions/{RUN}/debug/llm-calls").json()["items"][0]
    assert ACTIVE_KEY not in item["status"]
    assert "***" in item["status"]


def test_unavailable_key_is_reported(
    store: ArtifactStore, caplog: pytest.LogCaptureFixture
) -> None:
    """With no key resolvable, redaction cannot run — that must be visible rather
    than quietly serving raw captures."""
    _write_capture(store, "error: AuthError: bad key")
    with caplog.at_level(logging.WARNING, logger="eda_platform.application.services"):
        TraceService(store).list_llm_calls(RUN)
    assert any(
        "redact" in record.getMessage().lower() and RUN in record.getMessage()
        for record in caplog.records
    ), caplog.text


def test_empty_capture_page_does_not_warn(
    store: ArtifactStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing to redact is not a redaction failure."""
    with caplog.at_level(logging.WARNING, logger="eda_platform.application.services"):
        TraceService(store).list_llm_calls(RUN)
    assert not caplog.records


def test_debug_log_download_streams_a_fixed_snapshot(store: ArtifactStore) -> None:
    """A worker append after open is excluded without buffering the log in memory."""
    path = store.session_dir(PROJECT, RUN) / DEBUG_LOG_FILENAME
    path.write_text('{"a": 1}\n' * 50, encoding="utf-8")
    download = TraceService(store).open_debug_log(RUN)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"b": 2}\n' * 200)
    body = b"".join(download.chunks())
    assert len(body) == download.byte_size
    assert b'"b"' not in body

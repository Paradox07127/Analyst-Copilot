from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from eda_platform.application.ports import JobCommand
from eda_platform.core.llm import AnthropicLLMClient, OpenAICompatibleLLMClient
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.schemas.artifacts import ArtifactType


class _HangingProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _HangingHandler)
        self.received = threading.Event()
        self.release = threading.Event()


class _HangingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        server = self.server
        assert isinstance(server, _HangingProvider)
        server.received.set()
        if not server.release.wait(timeout=30):
            return
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "questions": [
                                        {
                                            "question_en": "Never persisted",
                                            "target_datasets": [],
                                            "llm_business_relevance": 0.8,
                                            "llm_actionability": 0.8,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _terminal_events(store: ArtifactStore, job_id: str) -> list[str]:
    with sqlite3.connect(store.db_path) as connection:
        rows = connection.execute(
            """
            select event_type from trace_events
            where job_id = ? and event_type in (
                'job.completed', 'job.failed', 'job.cancelled'
            )
            order by id
            """,
            (job_id,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def test_urllib_provider_hard_cancel_reaps_real_worker_without_result(
    tmp_path: Path,
) -> None:
    assert OpenAICompatibleLLMClient.active_abort_supported is False
    assert AnthropicLLMClient.active_abort_supported is False
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", name="Demo")
    store.start_session("demo", "source_hanging_provider")
    store.mark_session_status("demo", "source_hanging_provider", "completed")
    lifecycle = JobLifecycleRepository(store)
    job_id = "job_hanging_provider"
    session_id = "run_hanging_provider"
    params = {
        "source_session_id": "source_hanging_provider",
        "question": "Which segment should we prioritize?",
        "business_context": "",
        "llm": "live",
    }
    lifecycle.create_queued_job(
        job_id=job_id,
        session_id=session_id,
        project_id="demo",
        kind="question_draft",
        params_json=json.dumps(params),
        idempotency_key=None,
        lane_key=session_id,
        request_digest="digest-hanging-provider",
        request_scope=session_id,
    )
    provider = _HangingProvider()
    server_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    server_thread.start()
    backend = LocalProcessJobBackend(tmp_path, store)
    started = time.monotonic()
    try:
        reference = backend.enqueue(
            JobCommand(
                job_id=job_id,
                session_id=session_id,
                project_id="demo",
                kind="question_draft",
                params_json=json.dumps(params),
                env={
                    "EDA_LLM_PROVIDER": "openai_compatible",
                    "EDA_LLM_API_KEY": "test-key",
                    "EDA_LLM_BASE_URL": (
                        f"http://127.0.0.1:{provider.server_address[1]}"
                    ),
                    "EDA_LLM_MODEL": "hanging-test-model",
                    "EDA_LLM_TIMEOUT_SECONDS": "60",
                },
            )
        )
        assert reference.pid is not None
        assert provider.received.wait(timeout=10), "worker never entered provider request"
        backend.cancel(job_id)
        assert backend.join(job_id, timeout=10) is not None
        deadline = time.monotonic() + 3
        final = store.get_job(job_id)
        while final is not None and final["status"] not in {
            "completed",
            "failed",
            "cancelled",
        }:
            if time.monotonic() >= deadline:
                pytest.fail(f"job did not become terminal: {final['status']}")
            time.sleep(0.05)
            final = store.get_job(job_id)
    finally:
        provider.release.set()
        provider.shutdown()
        provider.server_close()
        server_thread.join(timeout=2)

    assert time.monotonic() - started < 14
    assert final is not None
    assert final["status"] == "cancelled"
    assert backend.join(job_id, timeout=0) is not None
    assert not any(
        artifact.type is ArtifactType.QUESTION_CANDIDATE_SET
        for artifact in store.list_artifacts(
            project_id="demo",
            session_id="source_hanging_provider",
        )
    )
    events = store.list_trace_events(project_id="demo", session_id=session_id)
    assert not any(event.event_type == "question.drafted" for event in events)
    assert not any(event.event_type == "step_completed" for event in events)
    assert _terminal_events(store, job_id) == ["job.cancelled"]

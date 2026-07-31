"""Test-only adapter for exercising a durable worker from a seeded queued row."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure.job_lifecycle import JobLifecycleRepository
from eda_platform.worker.runner import run_job


def run_claimed_job(
    workspace: Path,
    job_id: str,
    params_json: str,
) -> None:
    """Claim and run a legacy-seeded queued fixture through the production gate.

    A few validation tests intentionally seed rows with ``ArtifactStore.create_job``
    so they can manufacture invalid job kinds or stale payloads. Production
    creation uses ``JobLifecycleRepository.create_queued_job`` and already owns
    the run plus durable params; this adapter supplies those two lifecycle fields
    without restoring the removed ungated runner entry point.
    """

    store = ArtifactStore(workspace)
    job = store.get_job(job_id)
    if job is None:
        raise AssertionError(f"seeded job does not exist: {job_id}")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "update jobs set params_json = ? where job_id = ? and status = 'queued'",
            (params_json, job_id),
        )
        conn.execute(
            """
            update sessions set active_job_id = ?, state_version = state_version + 1
            where session_id = ? and active_job_id is null and storage_state = 'live'
            """,
            (job_id, str(job["session_id"])),
        )
    lifecycle = JobLifecycleRepository(store)
    claim = lifecycle.claim_launch(job_id, owner="test-direct-runner")
    lifecycle.acknowledge_spawn(
        claim,
        pid=os.getpid(),
        birth_identity="test-direct-runner",
    )
    run_job(
        str(workspace),
        job_id,
        launch_token=claim.token,
        launch_attempt=claim.attempt,
    )

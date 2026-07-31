from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response


def await_data_operation(
    client: TestClient,
    started: Response,
    result_path: str,
    *,
    timeout: float = 10.0,
) -> tuple[Response | None, dict[str, Any]]:
    """Wait for a 202 data operation and return (typed result, terminal job)."""
    assert started.status_code == 202, started.text
    job_id = str(started.json()["job"]["job_id"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}")
        assert status.status_code == 200, status.text
        job = status.json()
        if job["status"] == "completed":
            result = client.get(f"/api/v1/jobs/{job_id}/{result_path}")
            return result, job
        if job["status"] in {"failed", "cancelled"}:
            return None, job
        time.sleep(0.02)
    raise AssertionError(f"data operation {job_id} did not settle before timeout")


def operation_result_response(
    result: Response | None,
    job: dict[str, Any],
) -> Response:
    """Response-shaped assertion adapter after the new async contract is verified."""
    if result is not None:
        return result
    assert job["status"] in {"failed", "cancelled"}
    error_code = str(job.get("error_code") or "data_operation_failed")
    message = str(job.get("error_message") or error_code)
    public_code = {
        "ApprovalConsumedError": "approval_consumed",
        "ApprovalExpiredError": "approval_expired",
        "ApprovalNotFoundError": "approval_not_found",
        "approval_consumed": "approval_consumed",
        "approval_expired": "approval_expired",
        "approval_not_found": "approval_not_found",
        "CleaningSourceChangedError": "cleaning_source_changed",
        "cleaning_source_changed": "cleaning_source_changed",
        "DatasetNotFoundError": "dataset_not_found",
        "dataset_not_found": "dataset_not_found",
        "DatasetSourceMissingError": "dataset_source_missing",
        "dataset_source_missing": "dataset_source_missing",
        "JobIdempotencyMismatchError": "idempotency_key_reused",
        "idempotency_key_reused": "idempotency_key_reused",
        "SessionNotFoundError": "session_not_found",
        "session_not_found": "session_not_found",
        "cleaning_invalid": "cleaning_invalid",
        "cleaning_refused": "cleaning_refused",
        "custom_chart_invalid": "custom_chart_invalid",
    }.get(error_code, "custom_chart_invalid")
    status = 404 if error_code in {
        "SessionNotFoundError",
        "DatasetNotFoundError",
        "DatasetSourceMissingError",
        "session_not_found",
        "dataset_not_found",
        "dataset_source_missing",
    } else 410 if error_code in {
        "ApprovalExpiredError",
        "approval_expired",
    } else 409 if error_code in {
        "ApprovalConsumedError",
        "approval_consumed",
        "CleaningSourceChangedError",
        "cleaning_source_changed",
    } else 500 if error_code == "RuntimeError" else 422
    if "Approval not found" in message:
        status, public_code = 404, "approval_not_found"
    elif "No applicable cleaning operations" in message:
        status, public_code = 422, "cleaning_invalid"
    return Response(
        status_code=status,
        json={
            "error": {
                "code": public_code,
                "message": message,
            }
        },
    )

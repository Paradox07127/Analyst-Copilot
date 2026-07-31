"""Review round 3 F2: oversized bodies are refused at the ASGI layer, before
starlette spools the multipart payload to temp disk."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eda_platform.api.main import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path), raise_server_exceptions=False)


def test_declared_oversize_rejected_before_handler(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/api/v1/projects/demo/uploads",
        content=b"",
        headers={"content-length": str(10 << 30), "content-type": "multipart/form-data"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_streamed_oversize_aborts_mid_body(tmp_path: Path, monkeypatch) -> None:
    import eda_platform.api.main as api_main

    monkeypatch.setattr(api_main, "_BODY_LIMIT_SLACK", 0)
    monkeypatch.setattr(
        "eda_platform.api.main.MAX_UPLOAD_BYTES", 1024
    )
    client = _client(tmp_path)

    def chunks():
        # Chunked transfer: no Content-Length header to reject on.
        for _ in range(64):
            yield b"x" * 1024

    response = client.post(
        "/api/v1/projects/demo/uploads",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=xyz"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_oversize_json_uses_generic_request_error(tmp_path: Path, monkeypatch) -> None:
    import eda_platform.api.main as api_main

    monkeypatch.setattr(api_main, "_BODY_LIMIT_SLACK", 0)
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 64)
    client = _client(tmp_path)

    response = client.post(
        "/api/v1/projects",
        content=b'{"project_id":"demo","name":"' + (b"x" * 128) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_every_request_body_operation_declares_global_413(tmp_path: Path) -> None:
    schema = _client(tmp_path).get("/openapi.json").json()
    body_operations: list[tuple[str, str, dict]] = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if isinstance(operation, dict) and "requestBody" in operation:
                body_operations.append((path, method, operation))

    assert body_operations
    missing = [
        f"{method.upper()} {path}"
        for path, method, operation in body_operations
        if "413" not in operation["responses"]
    ]
    assert missing == []


def test_upload_service_keeps_upload_specific_413(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post(
        "/api/v1/projects",
        json={"project_id": "demo", "name": "Demo"},
    ).status_code == 201
    client.app.state.upload_service._max_bytes = 4  # type: ignore[attr-defined]  # noqa: SLF001

    response = client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_normal_small_request_passes_through(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/api/v1/projects")
    assert response.status_code == 200

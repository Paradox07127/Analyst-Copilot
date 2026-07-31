"""F-044/F-045: explicit remote boundary and durable upload abuse limits."""

from __future__ import annotations

import io
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.api.middleware import _trusted_client_id
from eda_platform.application.services.upload_service import (
    STAGING_DIRNAME,
    UploadConcurrentQuotaError,
    UploadFileQuotaError,
    UploadProjectByteQuotaError,
    UploadService,
)
from eda_platform.core.config import DeploymentConfigError, deployment_config
from eda_platform.core.query import TrustedFileQueryEngine
from eda_platform.core.store import ArtifactStore

CSV = b"a,b\n1,2\n"


def _service(store: ArtifactStore, **kwargs: int) -> UploadService:
    return UploadService(
        store,
        TrustedFileQueryEngine([store.root / "projects"]),
        parquet_enabled=False,
        **kwargs,
    )


def _store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.ensure_project("demo", "Demo")
    return store


def test_file_quota_is_persistent_and_rejection_leaves_no_orphan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = _service(store, project_file_quota=1)
    service.create_upload("demo", "one.csv", io.BytesIO(CSV))

    with pytest.raises(UploadFileQuotaError):
        service.create_upload("demo", "two.csv", io.BytesIO(b"a\n2\n"))

    assert list((tmp_path / STAGING_DIRNAME).iterdir()) == []
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("select count(*) from upload_usage").fetchone()[0] == 1
        assert conn.execute("select count(*) from upload_reservations").fetchone()[0] == 0


def test_byte_quota_aborts_stream_and_releases_reservation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = _service(store, project_byte_quota=4)

    with pytest.raises(UploadProjectByteQuotaError):
        service.create_upload("demo", "large.csv", io.BytesIO(CSV))

    assert list((tmp_path / STAGING_DIRNAME).iterdir()) == []
    assert not (store.project_dir("demo") / "uploads").exists()
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("select count(*) from upload_reservations").fetchone()[0] == 0


def test_concurrent_quota_is_an_atomic_sqlite_reservation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert (
        store.reserve_upload(
            "up_held",
            "demo",
            file_quota=10,
            concurrent_quota=1,
            now=time.time(),
        )
        is None
    )
    service = _service(store, concurrent_upload_quota=1)

    with pytest.raises(UploadConcurrentQuotaError):
        service.create_upload("demo", "blocked.csv", io.BytesIO(CSV))

    store.release_upload_reservation("up_held")
    assert service.create_upload("demo", "allowed.csv", io.BytesIO(CSV)).status == "completed"


def test_concurrent_byte_growth_cannot_cross_project_quota(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for upload_id in ("up_a", "up_b"):
        assert store.reserve_upload(
            upload_id,
            "demo",
            file_quota=10,
            concurrent_quota=2,
            now=100,
        ) is None
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def grow(upload_id: str) -> None:
        barrier.wait()
        outcomes.append(
            store.update_upload_reservation(
                upload_id,
                byte_size=8,
                byte_quota=10,
                now=101,
            )
        )

    threads = [threading.Thread(target=grow, args=(item,)) for item in ("up_a", "up_b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == [False, True]
    with sqlite3.connect(store.db_path) as conn:
        reserved = conn.execute(
            "select sum(reserved_bytes) from upload_reservations"
        ).fetchone()[0]
    assert reserved == 8


def test_startup_reconciles_legacy_canonical_uploads(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    legacy = store.project_dir("demo") / "uploads" / "ds_legacy" / "v1" / "old.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(CSV)
    monkeypatch.setenv("EDA_PROJECT_UPLOAD_FILE_QUOTA", "1")
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("new.csv", CSV, "text/csv")},
    )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "upload_file_quota"


def test_startup_reclaims_stale_reservation_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    assert store.reserve_upload(
        "up_stale",
        "demo",
        file_quota=10,
        concurrent_quota=2,
        now=0,
    ) is None
    staging = tmp_path / STAGING_DIRNAME / "up_stale"
    staging.mkdir(parents=True)
    (staging / "partial.csv").write_bytes(CSV)
    os.utime(staging, (0, 0))
    monkeypatch.setattr(time, "time", lambda: 100_000.0)

    TestClient(create_app(tmp_path))

    assert not staging.exists()
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute(
            "select count(*) from upload_reservations where upload_id = 'up_stale'"
        ).fetchone()[0] == 0


def test_reconcile_releases_quota_after_canonical_upload_is_deleted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = _service(store, project_file_quota=1)
    uploaded = service.create_upload("demo", "one.csv", io.BytesIO(CSV))
    assert uploaded.dataset is not None
    dataset_dir = (tmp_path / uploaded.dataset.original_uri).parent.parent
    shutil.rmtree(dataset_dir)
    store.reconcile_upload_quota()

    assert service.create_upload("demo", "two.csv", io.BytesIO(b"a\n2\n")).status == "completed"


def test_reconcile_cannot_erase_a_concurrent_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    assert store.reserve_upload(
        "up_live",
        "demo",
        file_quota=10,
        concurrent_quota=2,
        now=time.time(),
    ) is None
    uploads = store.project_dir("demo") / "uploads"
    canonical = uploads / "ds_live" / "v1" / "live.csv"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(CSV)
    scan_entered = threading.Event()
    continue_scan = threading.Event()
    original_iterdir = Path.iterdir

    def paused_iterdir(path: Path):
        if path == uploads and not scan_entered.is_set():
            scan_entered.set()
            assert continue_scan.wait(timeout=5)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", paused_iterdir)
    reconcile = threading.Thread(target=store.reconcile_upload_quota)
    reconcile.start()
    assert scan_entered.wait(timeout=5)
    completed = threading.Thread(
        target=store.complete_upload_reservation,
        args=("up_live", "ds_live", len(CSV)),
    )
    completed.start()
    time.sleep(0.05)
    assert completed.is_alive(), "completion should wait behind reconciliation's write lock"
    continue_scan.set()
    reconcile.join(timeout=5)
    completed.join(timeout=5)
    assert not reconcile.is_alive() and not completed.is_alive()
    with sqlite3.connect(store.db_path) as conn:
        usage = conn.execute(
            "select dataset_id, byte_size from upload_usage where project_id = 'demo'"
        ).fetchall()
        reservations = conn.execute(
            "select count(*) from upload_reservations"
        ).fetchone()[0]
    assert usage == [("ds_live", len(CSV))]
    assert reservations == 0


def _remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDA_DEPLOYMENT_MODE", "remote")
    monkeypatch.setenv("EDA_ALLOWED_HOSTS", "app.example,testserver")
    monkeypatch.setenv("EDA_ALLOWED_ORIGINS", "https://app.example")


def test_remote_mode_fails_closed_without_hosts_and_origins(monkeypatch) -> None:
    monkeypatch.setenv("EDA_DEPLOYMENT_MODE", "remote")
    monkeypatch.delenv("EDA_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("EDA_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(DeploymentConfigError):
        deployment_config()


def test_remote_mode_can_be_configured_from_repo_env_file(
    tmp_path: Path, monkeypatch
) -> None:
    for name in ("EDA_DEPLOYMENT_MODE", "EDA_ALLOWED_HOSTS", "EDA_ALLOWED_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "EDA_DEPLOYMENT_MODE=remote\n"
        "EDA_ALLOWED_HOSTS=app.example\n"
        "EDA_ALLOWED_ORIGINS=https://app.example\n",
        encoding="utf-8",
    )
    config = deployment_config(repo_root=tmp_path)
    assert config.remote is True
    assert config.allowed_hosts == ("app.example",)


def test_remote_config_normalizes_origin_slash_and_rejects_ambiguous_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDA_DEPLOYMENT_MODE", "remote")
    monkeypatch.setenv("EDA_ALLOWED_HOSTS", "APP.EXAMPLE")
    monkeypatch.setenv("EDA_ALLOWED_ORIGINS", "HTTPS://APP.EXAMPLE/")
    config = deployment_config()
    assert config.allowed_hosts == ("app.example",)
    assert config.allowed_origins == ("https://app.example",)

    for invalid in (
        "https://user@app.example",
        "https://*.example",
        "https://app.example:bad",
    ):
        monkeypatch.setenv("EDA_ALLOWED_ORIGINS", invalid)
        with pytest.raises(DeploymentConfigError):
            deployment_config()


def test_remote_host_and_unsafe_origin_header_contract(tmp_path: Path, monkeypatch) -> None:
    _remote_env(monkeypatch)
    client = TestClient(create_app(tmp_path), base_url="https://app.example")

    missing = client.post("/api/v1/projects", json={"project_id": "demo"})
    evil = client.post(
        "/api/v1/projects",
        json={"project_id": "demo"},
        headers={"Origin": "https://evil.example", "X-EDA-CSRF": "1"},
    )
    allowed = client.post(
        "/api/v1/projects",
        json={"project_id": "demo"},
        headers={"Origin": "https://app.example", "X-EDA-CSRF": "1"},
    )

    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "csrf_rejected"
    assert evil.status_code == 403
    assert allowed.status_code == 201
    assert client.get("/api/v1/projects").status_code == 200
    assert client.get("/api/v1/projects", headers={"Host": "evil.example"}).status_code == 400


def test_remote_cors_preflight_and_referer_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _remote_env(monkeypatch)
    client = TestClient(create_app(tmp_path), base_url="https://app.example")
    preflight = client.options(
        "/api/v1/projects",
        headers={
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-eda-csrf,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://app.example"
    referer = client.post(
        "/api/v1/projects",
        json={"project_id": "referer"},
        headers={
            "Referer": "https://app.example/projects",
            "X-EDA-CSRF": "1",
        },
    )
    assert referer.status_code == 201
    missing_signal = client.post(
        "/api/v1/projects",
        json={"project_id": "blocked"},
        headers={"Origin": "https://app.example"},
    )
    assert missing_signal.status_code == 403


def test_local_mode_keeps_loopback_mutation_compatibility(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/v1/projects",
        json={"project_id": "local"},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 201


def test_remote_upload_rate_limit_uses_trusted_client_and_expires(
    tmp_path: Path, monkeypatch
) -> None:
    _remote_env(monkeypatch)
    monkeypatch.setenv("EDA_UPLOAD_RATE_LIMIT", "1")
    client = TestClient(create_app(tmp_path), base_url="https://app.example")
    headers = {
        "Origin": "https://app.example",
        "X-EDA-CSRF": "1",
        "X-Forwarded-For": "192.0.2.10",
    }
    assert client.post(
        "/api/v1/projects", json={"project_id": "demo"}, headers=headers
    ).status_code == 201
    assert client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("one.csv", CSV, "text/csv")},
        headers=headers,
    ).status_code == 201

    limited = client.post(
        "/api/v1/projects/demo/uploads",
        files={"file": ("two.csv", CSV, "text/csv")},
        # The direct peer is not a configured trusted proxy, so spoofing a new
        # forwarded identity must not evade the direct-client rate bucket.
        headers={**headers, "X-Forwarded-For": "192.0.2.11"},
    )
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "upload_rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1


def test_trusted_proxy_walks_xff_from_the_socket_and_ignores_spoofed_prefix() -> None:
    scope = {"client": ("10.0.0.1", 1234)}
    trusted = frozenset({"10.0.0.1", "10.0.0.2"})

    assert _trusted_client_id(
        scope,
        {"x-forwarded-for": "198.51.100.99, 192.0.2.44"},
        trusted,
    ) == "192.0.2.44"
    assert _trusted_client_id(
        scope,
        {"x-forwarded-for": "198.51.100.99, 10.0.0.2"},
        trusted,
    ) == "198.51.100.99"


def test_untrusted_peer_and_malformed_xff_fail_closed_to_direct_identity() -> None:
    spoofed = {"x-forwarded-for": "192.0.2.1"}
    assert _trusted_client_id(
        {"client": ("203.0.113.8", 1234)},
        spoofed,
        frozenset({"10.0.0.1"}),
    ) == "203.0.113.8"
    assert _trusted_client_id(
        {"client": ("10.0.0.1", 1234)},
        {"x-forwarded-for": "not-an-ip"},
        frozenset({"10.0.0.1"}),
    ) == "10.0.0.1"


def test_rate_window_expiry_restores_capacity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.check_upload_rate_limit(
        "192.0.2.1", now=100, window_seconds=60, limit=1
    ) == (True, 0)
    allowed, retry = store.check_upload_rate_limit(
        "192.0.2.1", now=101, window_seconds=60, limit=1
    )
    assert allowed is False and retry > 0
    assert store.check_upload_rate_limit(
        "192.0.2.1", now=161, window_seconds=60, limit=1
    ) == (True, 0)


def test_upload_openapi_declares_remote_and_quota_errors(tmp_path: Path) -> None:
    operation = TestClient(create_app(tmp_path)).get("/openapi.json").json()["paths"][
        "/api/v1/projects/{project_id}/uploads"
    ]["post"]
    assert {"403", "413", "429"} <= operation["responses"].keys()


def test_remote_config_keeps_ipv6_origin_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urlsplit strips the brackets from an IPv6 host, but browsers send them.
    Rebuilding without them makes every IPv6 origin miss the allowlist."""
    monkeypatch.setenv("EDA_DEPLOYMENT_MODE", "remote")
    monkeypatch.setenv("EDA_ALLOWED_HOSTS", "app.example")
    monkeypatch.setenv("EDA_ALLOWED_ORIGINS", "https://[::1]:8443,http://[fe80::1]")

    config = deployment_config()

    assert config.allowed_origins == ("https://[::1]:8443", "http://[fe80::1]")

"""Settings API: provider metadata, session overrides, and the key-never-leaks
contract (plan §6.0/§14)."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.dto import SettingsPatch
from eda_platform.application.ports import JobCommand, JobRef
from eda_platform.application.services.job_service import JobService
from eda_platform.application.services.settings_service import (
    SettingsService,
    SettingsValidationError,
)
from eda_platform.core.llm import LLMProvider, LLMSettings
from eda_platform.core.process_identity import ProcessIdentity
from eda_platform.core.provider_registry import PROVIDER_REGISTRY
from eda_platform.core.request_dialect import (
    ParamRepair,
    forget_learned_repairs,
    learned_repairs,
    remember_repair,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.infrastructure import job_backend as backend_module
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.infrastructure.job_lifecycle import LaunchClaim
from eda_platform.infrastructure.launch_gate import acknowledge_and_wait

SECRET = "sk-live-supersecret-9911"


def test_settings_if_match_rejects_a_stale_writer(client: TestClient) -> None:
    initial = client.get("/api/v1/settings")
    assert initial.json()["version"] == 0
    first = client.put(
        "/api/v1/settings",
        json={"provider": "openai"},
        headers={"If-Match": '"0"'},
    )
    stale = client.put(
        "/api/v1/settings",
        json={"provider": "deepseek"},
        headers={"If-Match": '"0"'},
    )

    assert first.status_code == 200
    assert first.json()["version"] == 1
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "settings_version_conflict"
    assert client.get("/api/v1/settings").json()["provider"] == "openai"


def _mock_backend_spawn(
    monkeypatch: pytest.MonkeyPatch,
    seen: dict[str, object],
    *,
    pid: int,
) -> None:
    class _FakePopen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            self.pid = pid
            seen["argv"] = argv
            seen["env"] = kwargs.get("env")
            # An in-process fake shares the parent's descriptor table, so it
            # must copy the child side before the parent drops it.
            gate_fd, ready_fd = (int(part) for part in argv[-1].split(":")[1:])
            child_argument = f"fd:{os.dup(gate_fd)}:{os.dup(ready_fd)}"
            threading.Thread(
                target=lambda: acknowledge_and_wait(child_argument, argv[-3], 5.0),
                daemon=True,
            ).start()

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(backend_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        backend_module.JobLifecycleRepository,
        "claim_launch",
        lambda _self, job_id, **_kwargs: LaunchClaim(job_id, "token", 1),
    )
    monkeypatch.setattr(
        backend_module.JobLifecycleRepository,
        "acknowledge_spawn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        backend_module,
        "read_process_identity",
        lambda observed_pid: ProcessIdentity(
            pid=observed_pid,
            source="darwin-libproc",
            start_token="test",
        ),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ArtifactStore(tmp_path).ensure_project("demo", name="Demo")
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    app = create_app(workspace)
    # Deterministic defaults: the developer's real .env must not decide the test.
    app.state.settings_service = SettingsService(
        workspace=workspace, defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    return TestClient(app)


def test_providers_mark_entries_without_verified_agent_models(client: TestClient) -> None:
    body = client.get("/api/v1/settings/providers").json()
    assert len(body) == len(PROVIDER_REGISTRY)
    by_id = {item["provider"]: item for item in body}
    assert by_id["deepseek"]["display_name"] == "DeepSeek"
    assert by_id["deepseek"]["requires_api_key"] is True
    assert "deepseek-v4-flash" in by_id["deepseek"]["preset_models"]
    assert by_id["openai"]["agent_model_count"] > 0
    assert by_id["openai"]["capability_catalog_version"]
    assert by_id["azure_openai"]["agent_model_count"] == 0
    assert by_id["ollama"]["agent_model_count"] == 0


def test_get_settings_starts_from_env_defaults(client: TestClient) -> None:
    body = client.get("/api/v1/settings").json()
    assert body["provider"] == "offline"
    assert body["source"] == "env"
    assert body["api_key_set"] is False
    assert body["payload_policy"] == "schema+aggregates"


def test_update_switches_provider_and_seeds_its_preset_model(client: TestClient) -> None:
    body = client.put(
        "/api/v1/settings",
        json={"provider": "deepseek", "api_key": SECRET, "payload_policy": "schema_only"},
    ).json()
    assert body["provider"] == "deepseek"
    assert body["model"] == "deepseek-v4-flash"
    assert body["resolved_base_url"] == "https://api.deepseek.com"
    assert body["payload_policy"] == "schema_only"
    assert body["source"] == "session"
    assert body["is_ready_for_live_calls"] is True


def test_provider_switch_restores_each_provider_key(workspace: Path) -> None:
    openai_key = "sk-openai-secret-1111"
    deepseek_key = "sk-deepseek-secret-2222"
    service = SettingsService(
        workspace=workspace,
        defaults=LLMSettings(provider=LLMProvider.OFFLINE),
    )

    service.update_settings(SettingsPatch(provider="openai", api_key=openai_key))
    service.update_settings(SettingsPatch(provider="deepseek", api_key=deepseek_key))

    restored_openai = service.update_settings(SettingsPatch(provider="openai"))
    assert restored_openai.api_key_set is True
    assert restored_openai.api_key_last4 == "1111"
    assert service.resolve().llm.api_key == openai_key

    restored_deepseek = service.update_settings(SettingsPatch(provider="deepseek"))
    assert restored_deepseek.api_key_set is True
    assert restored_deepseek.api_key_last4 == "2222"
    assert service.resolve().llm.api_key == deepseek_key


def test_first_provider_switch_uses_its_env_key(workspace: Path) -> None:
    openai_key = "sk-openai-env-secret-3333"
    deepseek_key = "sk-deepseek-env-secret-4444"
    service = SettingsService(
        workspace=workspace,
        defaults=LLMSettings(provider=LLMProvider.OFFLINE),
        provider_api_keys={
            LLMProvider.OPENAI: openai_key,
            LLMProvider.DEEPSEEK: deepseek_key,
        },
    )

    openai = service.update_settings(SettingsPatch(provider="openai"))
    assert openai.api_key_last4 == "3333"
    assert service.resolve().llm.api_key == openai_key

    deepseek = service.update_settings(SettingsPatch(provider="deepseek"))
    assert deepseek.api_key_last4 == "4444"
    assert service.resolve().llm.api_key == deepseek_key


def test_provider_without_a_saved_or_env_key_does_not_reuse_current_key(
    workspace: Path,
) -> None:
    service = SettingsService(
        workspace=workspace,
        defaults=LLMSettings(provider=LLMProvider.OFFLINE),
    )
    service.update_settings(
        SettingsPatch(provider="openai", api_key="sk-openai-secret-5555")
    )

    switched = service.update_settings(SettingsPatch(provider="anthropic"))

    assert switched.api_key_set is False
    assert service.resolve().llm.api_key == ""


def test_api_key_is_never_echoed_in_any_response(client: TestClient) -> None:
    put = client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    get = client.get("/api/v1/settings")
    test = client.post("/api/v1/settings/test")
    for response in (put, get, test):
        assert SECRET not in response.text
    body = get.json()
    assert "api_key" not in body
    assert body["api_key_set"] is True
    assert body["api_key_last4"] == SECRET[-4:]


def test_failed_connection_test_does_not_echo_the_key(client: TestClient) -> None:
    """The probe hits a dead local port, so the error text is real transport
    output rather than a hand-written string."""
    client.put(
        "/api/v1/settings",
        json={
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "base_url": "http://127.0.0.1:9",
            "api_key": SECRET,
            "timeout_seconds": 10,
        },
    )
    response = client.post("/api/v1/settings/test")
    body = response.json()
    assert body["ok"] is False
    assert body["message"]
    assert SECRET not in response.text


def test_provider_error_echoing_the_key_is_scrubbed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control group for the redaction: some gateways mirror the request (and
    thus the Authorization header) into their error body."""
    echoed = f"HTTP 401: {{'sent_header': 'Bearer {SECRET}'}}"
    assert SECRET in echoed

    def exploding_client(_settings: object) -> object:
        raise RuntimeError(echoed)

    monkeypatch.setattr(
        "eda_platform.application.services.settings_service.create_llm_client",
        exploding_client,
    )
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    response = client.post("/api/v1/settings/test")
    assert response.json()["ok"] is False
    assert "***" in response.json()["message"]
    assert SECRET not in response.text


def test_offline_connection_test_sends_nothing(client: TestClient) -> None:
    body = client.post("/api/v1/settings/test").json()
    assert body["ok"] is True
    assert body["provider"] == "offline"
    assert "no request sent" in body["message"]


def test_api_key_never_reaches_disk(client: TestClient, workspace: Path) -> None:
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    client.post("/api/v1/settings/test")
    offenders = [
        path
        for path in workspace.rglob("*")
        if path.is_file() and SECRET in path.read_bytes().decode("utf-8", errors="ignore")
    ]
    assert offenders == []


def test_api_key_never_reaches_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
        client.post("/api/v1/settings/test")
        client.get("/api/v1/settings")
    assert SECRET not in caplog.text


def test_patch_repr_hides_the_key() -> None:
    """A traceback or debug dump that prints the request model must not carry it."""
    patch = SettingsPatch(provider="deepseek", api_key=SECRET)
    assert SECRET not in repr(patch)


def test_sessions_are_isolated(client: TestClient) -> None:
    client.put(
        "/api/v1/settings",
        json={"provider": "deepseek", "api_key": SECRET},
        headers={"X-EDA-Session": "alice"},
    )
    other = client.get("/api/v1/settings", headers={"X-EDA-Session": "bob"}).json()
    assert other["provider"] == "offline"
    assert other["api_key_set"] is False
    mine = client.get("/api/v1/settings", headers={"X-EDA-Session": "alice"}).json()
    assert mine["provider"] == "deepseek"


def test_reset_returns_to_env_defaults(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    body = client.delete("/api/v1/settings").json()
    assert body["provider"] == "offline"
    assert body["api_key_set"] is False
    assert body["source"] == "env"


def test_clear_api_key(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    body = client.put("/api/v1/settings", json={"clear_api_key": True}).json()
    assert body["api_key_set"] is False
    assert body["status_state"] == "incomplete"


def test_blank_api_key_keeps_the_stored_one(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    body = client.put("/api/v1/settings", json={"temperature": 0.5, "api_key": ""}).json()
    assert body["api_key_set"] is True
    assert body["temperature"] == 0.5


@pytest.mark.parametrize(
    "patch",
    [
        {"provider": "nope"},
        {"temperature": 9},
        {"max_tokens": 1},
        {"timeout_seconds": 1},
        {"payload_policy": "everything"},
        {"structured_output_mode": "yaml"},
        {"provider": "deepseek", "base_url": "ftp://elsewhere"},
        {"usd_per_1k_prompt": -0.1},
        {"usd_per_1k_completion": 5},
    ],
)
def test_invalid_patches_are_typed_errors(client: TestClient, patch: dict) -> None:
    response = client.put("/api/v1/settings", json=patch)
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"settings_invalid", "validation_error"}


def test_a_provider_without_a_verified_catalog_is_still_selectable(
    client: TestClient,
) -> None:
    """This used to 422. Twelve of eighteen providers were unreachable as a
    result, including every local one, whose model ids no catalog can enumerate."""
    switched = client.put("/api/v1/settings", json={"provider": "azure_openai"})

    assert switched.status_code == 200
    assert switched.json()["provider"] == "azure_openai"


def test_an_unverified_model_saves_and_reports_itself_as_unverified(
    client: TestClient,
) -> None:
    body = client.put(
        "/api/v1/settings",
        json={"provider": "openai", "model": "gpt-9-does-not-exist-yet", "api_key": SECRET},
    ).json()

    assert body["model"] == "gpt-9-does-not-exist-yet"
    assert body["model_verified"] is False
    assert any("verified catalog" in warning for warning in body["warnings"])
    # Advisory, not blocking: the probe decides before the run spends anything.
    assert body["is_ready_for_live_calls"] is True
    assert "agent_model" not in body["missing_fields"]


def test_a_verified_model_carries_no_warning(client: TestClient) -> None:
    body = client.put(
        "/api/v1/settings",
        json={"provider": "openai", "model": "gpt-4.1", "api_key": SECRET},
    ).json()

    assert body["model_verified"] is True
    assert body["warnings"] == []


def test_a_self_hosted_endpoint_is_configurable_end_to_end(client: TestClient) -> None:
    body = client.put(
        "/api/v1/settings",
        json={
            "provider": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "model": "my-finetune:latest",
        },
    ).json()

    assert body["status_state"] == "ready"
    assert body["is_ready_for_live_calls"] is True
    assert body["model"] == "my-finetune:latest"


def test_switching_to_a_preset_only_provider_seeds_a_model_it_serves(
    client: TestClient,
) -> None:
    """Gemini has no verified catalog, so the seed falls back to the registry's
    preset list rather than leaving the model field blank."""
    body = client.put("/api/v1/settings", json={"provider": "gemini"}).json()

    assert body["model"].startswith("gemini-")


def test_about_never_leaks_an_absolute_workspace_path(
    client: TestClient, workspace: Path
) -> None:
    body = client.get("/api/v1/settings").json()["about"]
    assert str(workspace) not in json.dumps(body)
    assert not body["workspace_label"].startswith("/")
    assert body["app_version"]


def test_short_key_does_not_expose_its_last_four(client: TestClient) -> None:
    body = client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": "abc123"}).json()
    assert body["api_key_set"] is True
    assert body["api_key_last4"] == ""


def test_resolve_feeds_the_worker_env_overlay(workspace: Path) -> None:
    service = SettingsService(
        workspace=workspace, defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    service.update_settings(
        SettingsPatch(provider="deepseek", api_key=SECRET, payload_policy="schema_only")
    )
    effective = service.resolve()
    assert effective.payload_policy == "schema_only"
    assert effective.env_overlay["EDA_LLM_PROVIDER"] == "deepseek"
    assert effective.env_overlay["EDA_LLM_MODEL"] == "deepseek-v4-flash"
    assert effective.env_overlay["EDA_LLM_API_KEY"] == SECRET


def test_offline_default_overlay_carries_no_key(workspace: Path) -> None:
    service = SettingsService(
        workspace=workspace, defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    assert "EDA_LLM_API_KEY" not in service.resolve().env_overlay


def test_expired_session_falls_back_to_defaults(workspace: Path) -> None:
    service = SettingsService(
        workspace=workspace,
        defaults=LLMSettings(provider=LLMProvider.OFFLINE),
        ttl_seconds=0,
    )
    service.update_settings(SettingsPatch(provider="deepseek", api_key=SECRET))
    assert service.get_settings().provider == "offline"


def test_service_rejects_unknown_provider_directly(workspace: Path) -> None:
    service = SettingsService(workspace=workspace)
    with pytest.raises(SettingsValidationError):
        service.update_settings(SettingsPatch(provider="not_a_provider"))


def test_cost_rates_round_trip_and_reach_the_worker_overlay(workspace: Path) -> None:
    service = SettingsService(
        workspace=workspace, defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    view = service.update_settings(
        SettingsPatch(usd_per_1k_prompt=0.0014, usd_per_1k_completion=0.0028)
    )
    assert view.usd_per_1k_prompt == pytest.approx(0.0014)
    assert view.usd_per_1k_completion == pytest.approx(0.0028)
    overlay = service.resolve().env_overlay
    assert overlay["EDA_LLM_USD_PER_1K_PROMPT"] == "0.0014"
    assert overlay["EDA_LLM_USD_PER_1K_COMPLETION"] == "0.0028"


def test_dev_mode_defaults_off_and_toggles_per_session(client: TestClient) -> None:
    assert client.get("/api/v1/settings").json()["dev_mode"] is False
    assert client.put("/api/v1/settings", json={"dev_mode": True}).json()["dev_mode"] is True
    assert client.get("/api/v1/settings").json()["dev_mode"] is True
    # Another session keeps its own value.
    other = client.get("/api/v1/settings", headers={"X-EDA-Session": "s2"}).json()
    assert other["dev_mode"] is False
    assert client.delete("/api/v1/settings").json()["dev_mode"] is False


class _CapturingBackend:
    """Records the JobCommand instead of spawning a worker."""

    def __init__(self) -> None:
        self.commands: list[JobCommand] = []

    def enqueue(self, command: JobCommand) -> JobRef:
        self.commands.append(command)
        return JobRef(job_id=command.job_id, pid=None)

    def cancel(self, job_id: str) -> None:
        pass

    def status(self, job_id: str) -> str:
        return "queued"


@pytest.fixture
def capturing_client(workspace: Path) -> tuple[TestClient, _CapturingBackend]:
    app = create_app(workspace)
    app.state.settings_service = SettingsService(
        workspace=workspace, defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    backend = _CapturingBackend()
    app.state.job_service = JobService(ArtifactStore(workspace), backend)
    seed = workspace / "seed" / "orders.csv"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("a,b\n1,2\n", encoding="utf-8")
    return TestClient(app), backend


def _start_run(client: TestClient, session_id: str, llm: str = "env") -> None:
    response = client.post(
        f"/api/v1/sessions/{session_id}/jobs",
        json={
            "kind": "auto_eda",
            "project_id": "demo",
            "datasets": ["seed/orders.csv"],
            "llm": llm,
            "generate_report": False,
        },
    )
    assert response.status_code == 201, response.text


def test_new_run_uses_the_session_payload_policy_and_provider(
    capturing_client: tuple[TestClient, _CapturingBackend],
) -> None:
    client, backend = capturing_client
    client.put(
        "/api/v1/settings",
        json={"provider": "deepseek", "api_key": SECRET, "payload_policy": "schema_only"},
    )
    _start_run(client, "run_new")
    command = backend.commands[-1]
    assert json.loads(command.params_json)["payload_policy"] == "schema_only"
    assert command.env is not None
    assert command.env["EDA_LLM_PROVIDER"] == "deepseek"
    assert command.env["EDA_LLM_MODEL"] == "deepseek-v4-flash"


def test_new_run_never_puts_the_key_in_worker_argv(
    capturing_client: tuple[TestClient, _CapturingBackend],
) -> None:
    """params_json becomes argv on the spawned worker, which `ps` exposes to
    every local user; the key must travel in the environment instead."""
    client, backend = capturing_client
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    _start_run(client, "run_new")
    command = backend.commands[-1]
    assert SECRET not in command.params_json
    assert command.env is not None
    assert command.env["EDA_LLM_API_KEY"] == SECRET


def test_offline_run_gets_no_llm_env_at_all(
    capturing_client: tuple[TestClient, _CapturingBackend],
) -> None:
    client, backend = capturing_client
    client.put("/api/v1/settings", json={"provider": "deepseek", "api_key": SECRET})
    _start_run(client, "run_offline", llm="offline")
    assert backend.commands[-1].env is None


def test_local_backend_hands_the_overlay_to_the_child_process(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closes the loop the capturing backend leaves open: JobCommand.env has to
    reach subprocess.Popen, or the worker never sees the session's provider."""
    seen: dict[str, object] = {}

    _mock_backend_spawn(monkeypatch, seen, pid=4321)
    LocalProcessJobBackend(workspace).enqueue(
        JobCommand(
            job_id="job_1",
            session_id="run_1",
            project_id="demo",
            kind="auto_eda",
            params_json='{"payload_policy": "schema_only"}',
            env={"EDA_LLM_API_KEY": SECRET, "EDA_LLM_PROVIDER": "deepseek"},
        )
    )
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["EDA_LLM_API_KEY"] == SECRET
    assert env["EDA_LLM_PROVIDER"] == "deepseek"
    # Inherited environment is preserved, not replaced.
    assert env.get("PATH") == os.environ.get("PATH")
    assert SECRET not in " ".join(str(item) for item in seen["argv"])  # type: ignore[union-attr]


def test_local_backend_inherits_the_environment_when_no_overlay(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    _mock_backend_spawn(monkeypatch, seen, pid=4322)
    LocalProcessJobBackend(workspace).enqueue(
        JobCommand(
            job_id="job_2",
            session_id="run_2",
            project_id="demo",
            kind="auto_eda",
            params_json="{}",
        )
    )
    assert seen["env"] is None


def test_repointing_the_same_model_at_a_new_endpoint_forgets_what_was_learned(
    tmp_path: Path,
) -> None:
    """Both caches key on (provider, model), which does not identify a server.
    Keeping a learned dialect across a base-URL change would send the previous
    endpoint's answer to a different one."""
    forget_learned_repairs()
    service = SettingsService(
        workspace=tmp_path.resolve(), defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    service.update_settings(
        SettingsPatch(
            provider="openai_compatible",
            base_url="http://host-a:8000/v1",
            model="my-finetune:latest",
        )
    )
    remember_repair(
        LLMProvider.OPENAI_COMPATIBLE,
        "my-finetune:latest",
        ParamRepair("rename", "max_tokens", "max_completion_tokens"),
    )

    service.update_settings(SettingsPatch(base_url="http://host-b:8000/v1"))

    assert learned_repairs(LLMProvider.OPENAI_COMPATIBLE, "my-finetune:latest") == ()


def test_an_unrelated_setting_change_keeps_the_learned_dialect(tmp_path: Path) -> None:
    """Re-discovering the dialect costs a paid round trip, so only a change of
    endpoint identity may throw it away."""
    forget_learned_repairs()
    service = SettingsService(
        workspace=tmp_path.resolve(), defaults=LLMSettings(provider=LLMProvider.OFFLINE)
    )
    service.update_settings(
        SettingsPatch(
            provider="openai_compatible",
            base_url="http://host-a:8000/v1",
            model="my-finetune:latest",
        )
    )
    repair = ParamRepair("rename", "max_tokens", "max_completion_tokens")
    remember_repair(LLMProvider.OPENAI_COMPATIBLE, "my-finetune:latest", repair)

    service.update_settings(SettingsPatch(temperature=0.7))

    assert learned_repairs(LLMProvider.OPENAI_COMPATIBLE, "my-finetune:latest") == (repair,)


# --------------------------------------------------------------------------- #
# Report narrative model: a second, optional choice alongside the workflow one
# --------------------------------------------------------------------------- #
def test_report_model_defaults_to_empty_meaning_same_as_the_workflow(
    client: TestClient,
) -> None:
    assert client.get("/api/v1/settings").json()["report_model"] == ""


def test_report_model_round_trips_and_reaches_the_worker_overlay(
    workspace: Path,
) -> None:
    service = SettingsService(
        defaults=LLMSettings(provider=LLMProvider.DEEPSEEK, model="deepseek-v4-flash")
    )
    view = service.update_settings(SettingsPatch(report_model="deepseek-v4"))

    assert view.report_model == "deepseek-v4"
    # The workflow model is untouched: two choices, not one moved.
    assert view.model == "deepseek-v4-flash"
    overlay = service.resolve().env_overlay
    assert overlay["EDA_REPORT_LLM_MODEL"] == "deepseek-v4"
    assert overlay["EDA_LLM_MODEL"] == "deepseek-v4-flash"
    _ = workspace


def test_an_unset_report_model_puts_nothing_in_the_overlay(workspace: Path) -> None:
    # An empty override must not reach the worker as an empty model id.
    service = SettingsService(
        defaults=LLMSettings(provider=LLMProvider.DEEPSEEK, model="deepseek-v4-flash")
    )
    assert "EDA_REPORT_LLM_MODEL" not in service.resolve().env_overlay
    _ = workspace


def test_clearing_the_report_model_returns_to_the_workflow_model(
    workspace: Path,
) -> None:
    service = SettingsService(
        defaults=LLMSettings(provider=LLMProvider.DEEPSEEK, model="deepseek-v4-flash")
    )
    service.update_settings(SettingsPatch(report_model="deepseek-v4"))
    view = service.update_settings(SettingsPatch(report_model=""))

    assert view.report_model == ""
    assert "EDA_REPORT_LLM_MODEL" not in service.resolve().env_overlay
    _ = workspace


def test_a_report_model_that_is_only_whitespace_is_rejected(workspace: Path) -> None:
    service = SettingsService(
        defaults=LLMSettings(provider=LLMProvider.DEEPSEEK, model="deepseek-v4-flash")
    )
    with pytest.raises(SettingsValidationError):
        service.update_settings(SettingsPatch(report_model="   x"))
    _ = workspace


def test_reset_clears_the_report_model(workspace: Path) -> None:
    service = SettingsService(
        defaults=LLMSettings(provider=LLMProvider.DEEPSEEK, model="deepseek-v4-flash")
    )
    service.update_settings(SettingsPatch(report_model="deepseek-v4"))
    assert service.reset().report_model == ""
    _ = workspace

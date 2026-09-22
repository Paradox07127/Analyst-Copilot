from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

import eda_platform.agents.code_graph as graph_module
from eda_platform.agents.code_agent import CodeAgent
from eda_platform.agents.runtime import AgentRuntime, AgentTool, AgentToolResult
from eda_platform.core.cancellation import CancellationContext
from eda_platform.core.graph_execution import GraphIdentityError, GraphPersistence
from eda_platform.core.llm import LLMToolCall, LLMToolResponse, OfflineLLMClient
from eda_platform.core.sandbox import ExecArtifact, SandboxBackendInfo
from eda_platform.core.store import ArtifactStore


class Crash(BaseException):
    pass


class Provider(OfflineLLMClient):
    calls = 0

    def structured(self, *, task, schema, payload):
        self.calls += 1
        return schema.model_validate({"code": "print('done')"})


class Backend:
    info = SandboxBackendInfo("test", True, True)

    def __init__(self, root: Path):
        self.root = root
        self.calls = 0

    def run_python(self, code, **kwargs):
        self.calls += 1
        output = self.root / "result.txt"
        output.write_text("done")
        return ExecArtifact(
            status="succeeded",
            backend="test",
            stdout="done",
            exit_code=0,
            output_files=[output],
            work_dir=self.root,
        )


def agent(root: Path, provider: Provider, backend: Backend) -> CodeAgent:
    return CodeAgent(
        llm=provider,
        backend=cast(Any, backend),
        persistence=GraphPersistence(root, "code:durable"),
    )


@pytest.mark.parametrize("boundary", ["sandbox", "validate"])
def test_code_restarts_at_checkpoint_without_repeating_completed_work(
    tmp_path: Path,
    monkeypatch,
    boundary: str,
) -> None:
    first, backend = Provider(), Backend(tmp_path)
    original = getattr(graph_module, "_" + boundary)

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(graph_module, "_" + boundary, crash)
    with pytest.raises(Crash):
        agent(tmp_path, first, backend).run(task="analyze", evidence_manifest={})
    assert first.calls == 1
    assert backend.calls == (1 if boundary == "validate" else 0)
    monkeypatch.setattr(graph_module, "_" + boundary, original)
    resumed = Provider()
    result = agent(tmp_path, resumed, backend).run(task="analyze", evidence_manifest={})
    assert result.status == "succeeded"
    assert resumed.calls == 0
    assert backend.calls == 1
    assert len(result.attempts) == 1
    assert agent(tmp_path, resumed, backend).run(task="analyze", evidence_manifest={}) == result
    assert resumed.calls == 0
    assert backend.calls == 1


def test_cancel_after_generation_does_not_run_sandbox(tmp_path: Path, monkeypatch) -> None:
    provider, backend = Provider(), Backend(tmp_path)
    original = graph_module._sandbox

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(graph_module, "_sandbox", crash)
    with pytest.raises(Crash):
        agent(tmp_path, provider, backend).run(task="analyze", evidence_manifest={})
    monkeypatch.setattr(graph_module, "_sandbox", original)
    cancellation = CancellationContext()
    cancellation.request_cancel("stop")
    result = agent(tmp_path, provider, backend).run(
        task="analyze",
        evidence_manifest={},
        cancellation=cancellation,
    )
    assert result.status == "failed"
    assert backend.calls == 0
    assert provider.calls == 1


@pytest.mark.parametrize("missing", [True, False])
def test_saved_code_outputs_must_still_match(tmp_path: Path, missing: bool) -> None:
    provider, backend = Provider(), Backend(tmp_path)
    agent(tmp_path, provider, backend).run(task="analyze", evidence_manifest={})
    output = tmp_path / "result.txt"
    if missing:
        output.unlink()
    else:
        output.write_text("changed")
    with pytest.raises(GraphIdentityError, match="outputs"):
        agent(tmp_path, provider, backend).run(task="analyze", evidence_manifest={})
    assert backend.calls == provider.calls == 1


def test_nested_code_graph_resumes_inside_parent_tool_without_repeating_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoArgs(BaseModel):
        pass

    class ParentProvider(OfflineLLMClient):
        def __init__(self, response: LLMToolResponse):
            super().__init__()
            self.response = response
            self.calls = 0

        def tool_call(self, **kwargs):
            self.calls += 1
            return self.response

    store = ArtifactStore(tmp_path / "workspace")
    child_provider, backend = Provider(), Backend(tmp_path)

    def execute(_args):
        result = agent(tmp_path, child_provider, backend).run(task="analyze", evidence_manifest={})
        return AgentToolResult({"status": result.status})

    def parent(provider):
        return AgentRuntime(
            llm=provider,
            tools=[AgentTool("code", "Analyze", NoArgs, execute, retry_safe=True)],
            persistence=GraphPersistence(tmp_path, "parent"),
            artifact_store=store,
        )

    first = ParentProvider(
        LLMToolResponse(
            tool_calls=[LLMToolCall(call_id="code-1", name="code", arguments={})],
        )
    )
    original = graph_module._sandbox

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(graph_module, "_sandbox", crash)
    with pytest.raises(Crash):
        parent(first).run(system_prompt="Analyze", user_message="Run code")
    monkeypatch.setattr(graph_module, "_sandbox", original)
    second = ParentProvider(LLMToolResponse(content="Done"))
    result = parent(second).run(system_prompt="Analyze", user_message="Run code")
    assert result.status == "completed"
    assert first.calls == second.calls == 1
    assert child_provider.calls == backend.calls == 1


class MeteredProvider(Provider):
    def last_usage(self):
        from eda_platform.core.llm import LLMResultMetadata, LLMUsage

        return LLMResultMetadata(
            provider="test", model="test",
            usage=LLMUsage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        )


@pytest.mark.parametrize("durable", [True, False])
def test_separate_code_executions_share_token_budget(tmp_path: Path, durable: bool) -> None:
    from eda_platform.core.budget import Budget

    provider, backend = MeteredProvider(), Backend(tmp_path)
    budget = Budget(max_tokens=75)

    def run(identity: str):
        return CodeAgent(
            llm=provider, backend=cast(Any, backend),
            persistence=GraphPersistence(tmp_path, identity) if durable else None,
        ).run(task="analyze", evidence_manifest={}, budget=budget)

    assert run("first").status == "succeeded"
    second = run("second")
    assert second.status == "failed"
    assert second.error_category == "budget_exhausted"
    assert budget.tokens_used == 100
    assert provider.calls == 2
    assert backend.calls == 1
    assert run("second").error_category == "budget_exhausted"
    assert budget.tokens_used == 100
    assert provider.calls == 2


def test_code_replay_accounts_only_its_own_spend(tmp_path: Path) -> None:
    from eda_platform.core.budget import Budget

    provider, backend = MeteredProvider(), Backend(tmp_path)
    budget = Budget(max_tokens=100)
    budget.add_tokens(20)
    runner = agent(tmp_path, provider, backend)
    for _ in range(2):
        assert runner.run(task="analyze", evidence_manifest={}, budget=budget).status == "succeeded"
        assert budget.tokens_used == 70
    restored_budget = Budget(max_tokens=60)
    restored_budget.add_tokens(20)
    for _ in range(2):
        result = runner.run(task="analyze", evidence_manifest={}, budget=restored_budget)
        assert result.error_category == "budget_exhausted"
        assert restored_budget.tokens_used == 70
    assert provider.calls == backend.calls == 1


def test_code_completed_budget_failure_restores_full_spend(tmp_path: Path) -> None:
    from eda_platform.core.budget import Budget

    provider, backend = MeteredProvider(), Backend(tmp_path)
    runner = agent(tmp_path, provider, backend)
    for budget in (Budget(max_tokens=25), Budget(max_tokens=25)):
        assert runner.run(task="analyze", evidence_manifest={}, budget=budget).error_category == (
            "budget_exhausted"
        )
        assert budget.tokens_used == 50
    assert provider.calls == 1
    assert backend.calls == 0


@pytest.mark.parametrize("mutation", ["content", "added", "removed", "renamed", "empty_dir"])
def test_code_directory_mount_change_rejects_cached_result(tmp_path: Path, mutation: str) -> None:
    from eda_platform.core.sandbox import SandboxMount

    source = tmp_path / "input"
    source.mkdir()
    data = source / "data.csv"
    data.write_text("a\n1\n")
    provider, backend = Provider(), Backend(tmp_path)
    runner = agent(tmp_path, provider, backend)
    runner.mounts = [SandboxMount(source=source, target="/data")]
    assert runner.run(task="analyze", evidence_manifest={}).status == "succeeded"
    # Timestamps are not part of the content witness.
    data.write_text("a\n1\n")
    assert runner.run(task="analyze", evidence_manifest={}).status == "succeeded"
    if mutation == "content":
        data.write_text("a\n2\n")
    elif mutation == "added":
        (source / "extra.csv").write_text("a\n3\n")
    elif mutation == "removed":
        data.unlink()
    elif mutation == "renamed":
        data.rename(source / "renamed.csv")
    else:
        (source / "empty").mkdir()
    with pytest.raises(GraphIdentityError):
        runner.run(task="analyze", evidence_manifest={})
    assert provider.calls == backend.calls == 1


@pytest.mark.parametrize("fresh_budget", [False, True])
def test_code_budget_survives_accounting_before_checkpoint_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_budget: bool,
) -> None:
    from eda_platform.core.budget import Budget

    original = Budget.account_execution_tokens
    crashed = False

    def account(self: Budget, execution_id: str, total_tokens: int) -> None:
        nonlocal crashed
        original(self, execution_id, total_tokens)
        if total_tokens and not crashed:
            crashed = True
            raise Crash()

    monkeypatch.setattr(Budget, "account_execution_tokens", account)
    provider, backend = MeteredProvider(), Backend(tmp_path)
    runner = agent(tmp_path, provider, backend)
    budget = Budget(max_tokens=75)
    with pytest.raises(Crash):
        runner.run(task="analyze", evidence_manifest={}, budget=budget)
    assert budget.tokens_used == 50
    if fresh_budget:
        budget = Budget(max_tokens=75)
    assert runner.run(task="analyze", evidence_manifest={}, budget=budget).status == "succeeded"
    assert budget.tokens_used == 50
    assert provider.calls == backend.calls == 1


def test_code_mount_symlink_is_not_an_untracked_input(tmp_path: Path) -> None:
    from eda_platform.core.sandbox import SandboxMount

    source = tmp_path / "input"
    source.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("value\n1\n")
    (source / "link.csv").symlink_to(outside)
    provider, backend = Provider(), Backend(tmp_path)
    runner = agent(tmp_path, provider, backend)
    runner.mounts = [SandboxMount(source=source, target="/data")]
    with pytest.raises(GraphIdentityError, match="symbolic"):
        runner.run(task="analyze", evidence_manifest={})
    assert provider.calls == backend.calls == 0

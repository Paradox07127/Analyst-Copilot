from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from eda_platform.core.llm import LLMResultMetadata, LLMUsage
from eda_platform.core.sandbox import (
    PortableSubprocessBackend,
    SandboxBackendInfo,
    SandboxLimits,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.drivers.chat import run_chat_turn
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.plans import Intent
from eda_platform.tools.loader import load_csv

T = TypeVar("T", bound=BaseModel)


class CodeChatLLM:
    def __init__(self, responses: list[BaseModel | dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, task: str, schema: type[T], payload: dict) -> T:
        self.calls.append({"task": task, "payload": payload})
        return schema.model_validate(self.responses.pop(0))

    def text(self, *, task: str, payload: dict) -> str:
        return ""

    def last_usage(self) -> LLMResultMetadata:
        return LLMResultMetadata(
            provider="fake",
            model="fake-model",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            estimated_cost_usd=0.001,
        )


class _SafeTestBackend(PortableSubprocessBackend):
    name = "safe_test_subprocess"

    @property
    def info(self) -> SandboxBackendInfo:
        return SandboxBackendInfo(
            name=self.name,
            safe_for_untrusted_code=True,
            available=True,
            detail="Test-only backend that exercises PortableSubprocessBackend behavior.",
        )


def test_run_chat_turn_open_analysis_repairs_code_and_stores_exec_artifact(
    tmp_path: Path,
) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "region,amount\nEast,10\nWest,20\nEast,5\n",
        encoding="utf-8",
    )
    store = ArtifactStore(tmp_path / "workspace")
    store.ensure_project("project_demo", name="Demo")
    store.start_session("project_demo", "run_demo")
    llm = CodeChatLLM(
        [
            Intent(
                kind="open_analysis",
                params={},
                confidence=0.92,
                raw_message="custom re-binning of amount",
            ),
            {"code": "raise RuntimeError('bad draft')"},
            {
                "code": (
                    "import json\n"
                    "import pandas as pd\n"
                    "df = pd.read_csv('inputs/orders.csv')\n"
                    "out = pd.DataFrame({'bin': ['low', 'high'], 'count': [2, 1]})\n"
                    "out.to_csv('rebinned.csv', index=False)\n"
                    "print(json.dumps({"
                    "'summary': 'rebinned amount into two groups', "
                    "'result_files': ['rebinned.csv'], "
                    "'metrics': {'rows': int(len(df))}"
                    "}))"
                )
            },
        ]
    )

    result = run_chat_turn(
        "custom re-binning of amount",
        datasets=[load_csv(orders, dataset_id="ds_orders")],
        project_id="project_demo",
        session_id="run_demo",
        llm=cast(Any, llm),
        store=store,
        code_backend=_SafeTestBackend(work_root=tmp_path / "code"),
        code_limits=SandboxLimits(timeout_seconds=5),
    )

    assert result.status == "answer"
    assert result.intent.kind == "open_analysis"
    artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.type is ArtifactType.CODE_EXECUTION_RESULT
    )
    assert artifact.payload["stdout_json"]["summary"] == "rebinned amount into two groups"
    assert artifact.payload["output_files"] == ["rebinned.csv"]
    assert artifact.evidence[0].kind == "code"
    assert "rebinned amount" in result.message

    events = store.list_trace_events(project_id="project_demo", session_id="run_demo")
    attempt_events = [event for event in events if event.event_type == "code_agent_attempt"]
    assert [event.summary["attempt"] for event in attempt_events] == [1, 2]
    assert [event.summary["status"] for event in attempt_events] == [
        "failed",
        "succeeded",
    ]
    repair_success_rate = sum(
        event.summary["status"] == "succeeded" for event in attempt_events
    ) / len(attempt_events)
    assert repair_success_rate == 0.5
    assert llm.calls[1]["payload"]["evidence_manifest"]["datasets"][0]["mount_path"] == (
        "inputs/orders.csv"
    )

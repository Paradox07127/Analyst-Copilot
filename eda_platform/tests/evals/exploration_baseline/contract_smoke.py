"""Real-provider contract smoke (doc 07-31 §13.3 subset; test doubles cannot
replace this gate).

Checks per model: capability-catalog hit, one real tool-call round trip
(request -> tool result -> final text), json_schema/json_object structured
output, one output-retry path (validation error fed back, second attempt must
comply — the urllib client has param repair but no output retry, so the retry
loop lives here at harness level, same as Eval-0 will use), usage/cost
accounting on every call, Chinese-input round trip.

Deliberately NOT covered this round (E4a scope): tier token/latency profiles,
rate-limit behavior, context-compaction recovery.

Usage:
  PYTHONPATH=<worktree>/eda_platform/src ./.venv/bin/python \
    eda_platform/tests/evals/exploration_baseline/contract_smoke.py \
    --provider openai --model gpt-5.6-luna \
    --env-file "/Users/taijial/VSCode/Analyst copilot/.env" --out results.json

Hard cap: MAX_CALLS API calls per invocation; exceeding it aborts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from eda_platform.core.env import (
    load_llm_settings_from_env_file,
    load_provider_api_keys_from_env_file,
)
from eda_platform.core.llm import LLMToolResponse, create_llm_client
from eda_platform.core.model_capabilities import (
    CAPABILITY_CATALOG_VERSION,
    agent_model_profile,
    is_verified_agent_model,
)
from eda_platform.core.provider_registry import LLMProvider

MAX_CALLS = 20
RETRY_MAGIC = "EVAL0-RETRY-OK"
ECHO_TOKEN = "EVAL0-ECHO-7Q4Z"


class SmokeSummary(BaseModel):
    verdict: Literal["ok", "fail"]
    row_count: int
    regions: list[str]


class RetryProbe(BaseModel):
    magic: str

    @field_validator("magic")
    @classmethod
    def _must_be_exact(cls, value: str) -> str:
        if value != RETRY_MAGIC:
            raise ValueError(f"magic must be exactly '{RETRY_MAGIC}', got '{value}'")
        return value


class ChineseProbe(BaseModel):
    column_name: str
    reason: str


class MeteredClient:
    def __init__(self, inner, cap: int) -> None:
        self.inner = inner
        self.cap = cap
        self.calls: list[dict] = []
        self._last_seen_meta = None

    @property
    def settings(self):
        return self.inner.settings

    def _record(self, label: str) -> None:
        # last_usage() still holding the PREVIOUS call's metadata means this
        # call raised before a response landed; record it as no_response so the
        # usage check does not blame accounting for a transport-level failure.
        meta = self.inner.last_usage()
        if meta is self._last_seen_meta:
            self.calls.append({"label": label, "no_response": True})
            return
        self._last_seen_meta = meta
        entry = {"label": label, "usage_reported": False}
        if meta is not None:
            entry.update(
                usage_reported=meta.usage_reported,
                prompt_tokens=meta.usage.prompt_tokens,
                completion_tokens=meta.usage.completion_tokens,
                total_tokens=meta.usage.total_tokens,
                estimated_cost_usd=meta.estimated_cost_usd,
                cost_basis=meta.cost_basis,
                finish_reason=meta.finish_reason,
                response_model=meta.model,
            )
        self.calls.append(entry)

    def _check_cap(self) -> None:
        if len(self.calls) >= self.cap:
            raise SystemExit(f"call cap {self.cap} reached — aborting to protect budget")

    def structured(self, *, label: str, **kwargs):
        self._check_cap()
        try:
            return self.inner.structured(**kwargs)
        finally:
            self._record(label)

    def text(self, *, label: str, **kwargs):
        self._check_cap()
        try:
            return self.inner.text(**kwargs)
        finally:
            self._record(label)

    def tool_call(self, *, label: str, **kwargs) -> LLMToolResponse:
        self._check_cap()
        try:
            return self.inner.tool_call(**kwargs)
        finally:
            self._record(label)

    def totals(self) -> dict:
        return {
            "n_calls": len(self.calls),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in self.calls),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in self.calls),
            "total_tokens": sum(c.get("total_tokens", 0) for c in self.calls),
            "estimated_cost_usd": round(
                sum(c.get("estimated_cost_usd") or 0.0 for c in self.calls), 6
            ),
        }


class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "fail", "skip"]
    detail: str = ""
    calls_used: int = 0


ECHO_TOOL = {
    "name": "echo_probe",
    "description": "Echo the provided token and number back to the caller.",
    "parameters": {
        "type": "object",
        "properties": {
            "token": {"type": "string", "description": "Token to echo back verbatim."},
            "number": {"type": "integer", "description": "Number to echo back."},
        },
        "required": ["token", "number"],
    },
}


def check_catalog(provider: LLMProvider, model: str) -> CheckResult:
    profile = agent_model_profile(provider, model)
    if profile is None or not is_verified_agent_model(provider, model):
        return CheckResult(
            name="capability_catalog",
            status="fail",
            detail=f"model {model} not in {CAPABILITY_CATALOG_VERSION}",
        )
    return CheckResult(
        name="capability_catalog",
        status="pass",
        detail=(
            f"{CAPABILITY_CATALOG_VERSION}: structured_output={profile.structured_output} "
            f"parallel={profile.parallel_tool_calling} tool_choice={profile.tool_choice_policy} "
            f"reasoning_state={profile.reasoning_state_policy}"
        ),
    )


def check_tool_roundtrip(client: MeteredClient) -> CheckResult:
    before = len(client.calls)
    messages = [
        {"role": "system", "content": "You are a tool-using probe. Use tools when asked."},
        {
            "role": "user",
            "content": (
                f"Call the echo_probe tool with token='{ECHO_TOKEN}' and number=7, "
                "then tell me what the tool echoed back."
            ),
        },
    ]
    response = client.tool_call(
        label="tool_leg1", task="smoke", messages=messages, tools=[ECHO_TOOL]
    )
    if not response.tool_calls:
        return CheckResult(
            name="tool_calling_roundtrip",
            status="fail",
            detail=f"no tool call returned (finish={response.finish_reason})",
            calls_used=len(client.calls) - before,
        )
    call = response.tool_calls[0]
    if call.name != "echo_probe":
        return CheckResult(
            name="tool_calling_roundtrip", status="fail", detail=f"wrong tool {call.name}"
        )
    if call.arguments.get("token") != ECHO_TOKEN or call.arguments.get("number") != 7:
        return CheckResult(
            name="tool_calling_roundtrip",
            status="fail",
            detail=f"bad arguments {call.arguments}",
            calls_used=len(client.calls) - before,
        )
    assistant_msg: dict = {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
        ],
    }
    # DeepSeek requires reasoning_content round-tripping via provider_state
    assistant_msg.update(response.provider_state)
    messages = [
        *messages,
        assistant_msg,
        {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": json.dumps({"echoed_token": ECHO_TOKEN, "echoed_number": 7}),
        },
    ]
    final = client.tool_call(label="tool_leg2", task="smoke", messages=messages, tools=[ECHO_TOOL])
    if final.tool_calls:
        return CheckResult(
            name="tool_calling_roundtrip",
            status="fail",
            detail="model kept calling tools after result was returned",
            calls_used=len(client.calls) - before,
        )
    if not final.content.strip():
        return CheckResult(
            name="tool_calling_roundtrip",
            status="fail",
            detail="empty final answer after tool result",
            calls_used=len(client.calls) - before,
        )
    return CheckResult(
        name="tool_calling_roundtrip",
        status="pass",
        detail=(
            "tool requested with exact args; tool result consumed; "
            f"final text mentions token={ECHO_TOKEN in final.content}"
        ),
        calls_used=len(client.calls) - before,
    )


def check_structured(client: MeteredClient) -> CheckResult:
    before = len(client.calls)
    result = client.structured(
        label="structured",
        task="smoke_structured",
        schema=SmokeSummary,
        payload={
            "instruction": (
                "Dataset has 1086 rows across regions North and South and looks healthy. "
                "Summarize with verdict 'ok'."
            )
        },
    )
    good = (
        result.verdict == "ok"
        and result.row_count == 1086
        and {"North", "South"} <= set(result.regions)
    )
    return CheckResult(
        name="structured_output",
        status="pass" if good else "fail",
        detail=f"parsed {result.model_dump()}",
        calls_used=len(client.calls) - before,
    )


def check_output_retry(client: MeteredClient) -> CheckResult:
    before = len(client.calls)
    try:
        client.structured(
            label="retry_attempt1",
            task="smoke_retry",
            schema=RetryProbe,
            payload={"instruction": "Set the field 'magic' to the placeholder string 'TBD'."},
        )
        return CheckResult(
            name="output_retry",
            status="fail",
            detail="first attempt unexpectedly satisfied the validator — probe inconclusive",
            calls_used=len(client.calls) - before,
        )
    except Exception as exc:  # noqa: BLE001 - validation failure is the expected red
        first_error = str(exc)[:300]
    result = client.structured(
        label="retry_attempt2",
        task="smoke_retry",
        schema=RetryProbe,
        payload={
            "previous_error": first_error,
            "instruction": (
                f"Your previous output failed validation. Set 'magic' to exactly '{RETRY_MAGIC}'."
            ),
        },
    )
    return CheckResult(
        name="output_retry",
        status="pass" if result.magic == RETRY_MAGIC else "fail",
        detail="attempt1 failed validation as planned; attempt2 complied after error feedback",
        calls_used=len(client.calls) - before,
    )


def check_chinese_roundtrip(client: MeteredClient) -> CheckResult:
    before = len(client.calls)
    result = client.structured(
        label="chinese",
        task="smoke_chinese",
        schema=ChineseProbe,
        payload={
            "任务": "下面是数据集的列名,选出表示营业收入的那一列,并用中文说明理由。",
            "列名": ["订单日期", "销售额", "地区", "客户年龄"],
        },
    )
    has_cjk = any("一" <= ch <= "鿿" for ch in result.reason)
    good = result.column_name == "销售额" and has_cjk
    return CheckResult(
        name="chinese_roundtrip",
        status="pass" if good else "fail",
        detail=f"column={result.column_name!r} reason_cjk={has_cjk}",
        calls_used=len(client.calls) - before,
    )


def check_usage_accounting(client: MeteredClient) -> CheckResult:
    answered = [c for c in client.calls if not c.get("no_response")]
    bad = [
        c["label"]
        for c in answered
        if not c.get("usage_reported")
        or c.get("prompt_tokens", 0) <= 0
        or c.get("completion_tokens", 0) <= 0
        or c.get("estimated_cost_usd") is None
    ]
    failed_transport = len(client.calls) - len(answered)
    suffix = f" ({failed_transport} call(s) raised before a response)" if failed_transport else ""
    return CheckResult(
        name="usage_accounting",
        status="pass" if not bad and answered else "fail",
        detail=(
            f"all {len(answered)} answered calls reported tokens and cost{suffix}"
            if not bad
            else f"missing usage on {bad}{suffix}"
        ),
        calls_used=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--patch-reasoning-effort-none",
        action="store_true",
        help=(
            "Diagnostic: inject reasoning_effort='none' into generation controls. "
            "Verifies the provider-documented workaround for models that reject "
            "function tools on /v1/chat/completions at default reasoning effort "
            "(observed on gpt-5.6-luna, 2026-08-01)."
        ),
    )
    args = parser.parse_args()
    if args.patch_reasoning_effort_none:
        import eda_platform.core.llm as llm_module

        original_controls = llm_module.build_generation_controls
        llm_module.build_generation_controls = lambda settings: {
            **original_controls(settings),
            "reasoning_effort": "none",
        }

    provider = LLMProvider(args.provider)
    settings = load_llm_settings_from_env_file(args.env_file)
    keys = load_provider_api_keys_from_env_file(args.env_file)
    if provider not in keys:
        raise SystemExit(f"no API key for provider {provider.value} in {args.env_file}")
    settings = settings.model_copy(
        update={"provider": provider, "model": args.model, "api_key": keys[provider]}
    )
    client = MeteredClient(create_llm_client(settings), cap=MAX_CALLS)

    results: list[CheckResult] = [check_catalog(provider, args.model)]
    for check in (check_tool_roundtrip, check_structured, check_output_retry,
                  check_chinese_roundtrip):
        try:
            results.append(check(client))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - record and continue to next check
            results.append(
                CheckResult(name=check.__name__.removeprefix("check_"), status="fail",
                            detail=repr(exc)[:300])
            )
    results.append(check_usage_accounting(client))
    for name in ("tier_token_latency_profile", "rate_limit_behavior", "context_compaction"):
        results.append(CheckResult(name=name, status="skip", detail="deferred to E4a"))

    report = {
        "provider": provider.value,
        "model": args.model,
        "patched_reasoning_effort_none": args.patch_reasoning_effort_none,
        "catalog_version": CAPABILITY_CATALOG_VERSION,
        "checks": [r.model_dump() for r in results],
        "calls": client.calls,
        "totals": client.totals(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0 if all(r.status != "fail" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

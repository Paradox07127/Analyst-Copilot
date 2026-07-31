"""Live NL2SQL execution-accuracy scoring (deferred until an API key exists).

This module is the *scoring* half of the M4 eval-week ruler. It is skipped by
default because the environment has no live LLM credentials; the offline
soundness of the ruler itself is proven in
``test_nl2sql_execution_accuracy.py``.

How to run manually (M4 DoD line 6 target: >= 80% execution accuracy)::

    EDA_LIVE_LLM_TEST=1 \
    EDA_LLM_PROVIDER=deepseek \
    DEEPSEEK_API_KEY=sk-... \
    EDA_LLM_MODEL=deepseek-v4-flash \
    .venv/bin/python -m pytest \
        eda_platform/tests/evals/nl2sql/test_nl2sql_live_execution_accuracy.py -q -s

Any OpenAI-compatible provider works — set ``EDA_LLM_PROVIDER`` /
``EDA_LLM_BASE_URL`` / ``EDA_LLM_API_KEY`` as in ``core/env.py`` (a project
``.env`` file is honoured too). Per-case results are printed and written to
``eda_platform/workspace/eval_results/`` for the eval report.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eda_platform.core.env import load_llm_settings_from_env_file
from eda_platform.core.llm import create_llm_client, llm_configuration_status

from .nl2sql_eval_harness import chat_sql_provider, load_golden_cases, run_execution_accuracy

LIVE_FLAG = "EDA_LIVE_LLM_TEST"
ACCURACY_TARGET = 0.80  # M4 plan §8 DoD line 6: 20 cases >= 80% execution accuracy
_RESULTS_DIR = Path(__file__).parents[3] / "workspace" / "eval_results"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_FLAG) != "1",
    reason=(
        "live LLM eval is opt-in: set EDA_LIVE_LLM_TEST=1 plus provider settings "
        "(e.g. EDA_LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=...), then run "
        ".venv/bin/python -m pytest "
        "eda_platform/tests/evals/nl2sql/test_nl2sql_live_execution_accuracy.py -q -s"
    ),
)


def test_live_nl2sql_execution_accuracy_meets_target() -> None:
    settings = load_llm_settings_from_env_file()
    status = llm_configuration_status(settings)
    if not status.is_ready_for_live_calls:
        pytest.fail(
            f"{LIVE_FLAG}=1 but the LLM is not configured for live calls: "
            f"{status.message} Set EDA_LLM_PROVIDER and the matching API key "
            "(see module docstring), or unset the flag to skip."
        )

    llm = create_llm_client(settings)
    cases = load_golden_cases()
    result = run_execution_accuracy(cases, chat_sql_provider(llm))

    print()
    print(result.summary_table())
    payload = {
        "ran_at": datetime.now(UTC).isoformat(),
        "provider": settings.provider.value,
        "model": settings.model,
        "accuracy_target": ACCURACY_TARGET,
        **result.to_json_payload(),
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        _RESULTS_DIR / f"nl2sql_live_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"per-case results written to {out_path}")

    assert result.accuracy >= ACCURACY_TARGET, (
        f"live execution accuracy {result.accuracy:.1%} is below the "
        f"{ACCURACY_TARGET:.0%} target; failed cases: "
        f"{[outcome.case_id for outcome in result.failed]} (details in {out_path})"
    )

"""F-023 cancellation contracts for every worker-owned driver entry point.

These tests deliberately describe the driver boundary before the worker wiring
is changed.  A driver may satisfy the checkpoint contract either by polling the
callback itself or by forwarding it to the long-running function it delegates
to (for example ``fork_session`` -> ``run_auto_eda``).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from collections.abc import Callable
from typing import Any

import pytest

from eda_platform.worker import runner

# Keep this table one-to-one with the worker dispatch.  ``question_draft`` is
# intentionally mapped to its blocking LLM proposal boundary; the following
# append is a short worker-owned persistence boundary and needs a second worker
# poll before it is called.
DRIVER_ENTRYPOINTS = {
    "auto_eda": ("eda_platform.drivers.auto_eda", "run_auto_eda"),
    "question_exec": ("eda_platform.drivers.question_exec", "run_question_batch"),
    "skill_replay": ("eda_platform.drivers.analysis_skill", "replay_skill"),
    "relationship_validate": (
        "eda_platform.drivers.auto_eda",
        "validate_relationship_candidate_on_demand",
    ),
    "relationship_discover": (
        "eda_platform.drivers.auto_eda",
        "discover_relationships_on_demand",
    ),
    "report_generate": (
        "eda_platform.drivers.auto_eda",
        "generate_report_on_demand",
    ),
    "session_fork": ("eda_platform.drivers.session_fork", "fork_session"),
    "question_draft": (
        "eda_platform.agents.question_agent",
        "propose_llm_question_candidates",
    ),
    "investigation_plan": (
        "eda_platform.drivers.investigation_orchestrator",
        "create_investigation_plans",
    ),
    "investigation_execute": (
        "eda_platform.drivers.investigation_orchestrator",
        "execute_investigation_plans",
    ),
    "macro_loop": (
        "eda_platform.drivers.investigation_orchestrator",
        "run_macro_loop",
    ),
    "synthesis_brief_create": (
        "eda_platform.drivers.synthesis_orchestrator",
        "create_synthesis_brief",
    ),
    "decision_report_generate": (
        "eda_platform.drivers.decision_report",
        "create_decision_report",
    ),
    "cleaning_preview": (
        "eda_platform.application.services.cleaning_service",
        "CleaningService.preview",
    ),
    "cleaning_apply": (
        "eda_platform.application.services.cleaning_service",
        "CleaningService.apply",
    ),
    "dataset_distributions": (
        "eda_platform.application.services.dataset_service",
        "DatasetService.get_distributions",
    ),
    "custom_chart": (
        "eda_platform.application.services.insight_service",
        "InsightService.build_custom_chart",
    ),
    "exploration_run": (
        "eda_platform.worker.exploration",
        "run_exploration_worker",
    ),
}


def _entrypoint(job_kind: str) -> Callable[..., Any]:
    module_name, function_name = DRIVER_ENTRYPOINTS[job_kind]
    value: Any = importlib.import_module(module_name)
    for part in function_name.split("."):
        value = getattr(value, part)
    return value


def _forwards_or_polls_cancel_check(function: Callable[..., Any]) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        referenced = [call.func, *call.args, *(item.value for item in call.keywords)]
        if any(
            isinstance(node, ast.Name) and node.id == "cancel_check"
            for root in referenced
            for node in ast.walk(root)
        ):
            return True
    return False


def test_contract_covers_every_runner_job_kind_exactly() -> None:
    source = inspect.getsource(runner.run_job)
    dispatched = set(re.findall(r'job\["kind"\]\s*==\s*"([^"]+)"', source))
    assert dispatched == set(DRIVER_ENTRYPOINTS)


@pytest.mark.parametrize("job_kind", DRIVER_ENTRYPOINTS)
def test_driver_entrypoint_accepts_optional_keyword_cancel_check(
    job_kind: str,
) -> None:
    parameter = inspect.signature(_entrypoint(job_kind)).parameters["cancel_check"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("job_kind", DRIVER_ENTRYPOINTS)
def test_driver_entrypoint_polls_or_forwards_cancel_check(job_kind: str) -> None:
    assert _forwards_or_polls_cancel_check(_entrypoint(job_kind)), (
        f"{job_kind} accepts cancellation but never polls or forwards it"
    )

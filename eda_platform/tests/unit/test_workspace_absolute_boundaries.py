from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from eda_platform.api.main import create_app
from eda_platform.application.job_results import read_job_result, write_job_result
from eda_platform.application.services.sandbox_status_service import SandboxStatusService
from eda_platform.application.services.settings_service import SettingsService
from eda_platform.application.services.upload_service import sweep_staging
from eda_platform.core.config import WorkspaceConfigError
from eda_platform.core.session_fence import session_key_lock
from eda_platform.core.session_loader import load_run
from eda_platform.core.store import ArtifactStore, session_dir_path
from eda_platform.drivers.auto_eda import run_auto_eda
from eda_platform.drivers.investigation_orchestrator import create_investigation_plans
from eda_platform.drivers.question_exec import run_question_batch
from eda_platform.drivers.synthesis_orchestrator import create_synthesis_brief
from eda_platform.drivers.workflow_eval import run_fresh_workflow_eval_case
from eda_platform.infrastructure.job_backend import LocalProcessJobBackend
from eda_platform.schemas.workflow_eval import WorkflowEvalSpec
from eda_platform.worker.runner import main as worker_main
from eda_platform.worker.runner import run_job

REPO_ROOT = Path(__file__).resolve().parents[3]


def _open_relative_run_lock() -> None:
    with session_key_lock("relative-workspace", "run"):
        pass


def _relative_boundary_calls(
    absolute_store: ArtifactStore,
) -> list[tuple[str, Callable[[], object]]]:
    return [
        ("artifact store", lambda: ArtifactStore("relative-workspace")),
        (
            "job result write",
            lambda: write_job_result("relative-workspace", "proj", "run", "job", "{}"),
        ),
        (
            "job result read",
            lambda: read_job_result("relative-workspace", "proj", "run", "job"),
        ),
        (
            "store run path",
            lambda: session_dir_path("relative-workspace", "project", "run"),
        ),
        ("run fence", _open_relative_run_lock),
        ("api", lambda: create_app("relative-workspace")),
        (
            "worker backend",
            lambda: LocalProcessJobBackend("relative-workspace", absolute_store),
        ),
        (
            "worker runner",
            lambda: run_job(
                "relative-workspace",
                "job_missing",
                launch_token="token",
                launch_attempt=1,
            ),
        ),
        (
            "worker argv",
            lambda: worker_main(
                ["relative-workspace", "job", "token", "1", "fd:-1:-1"]
            ),
        ),
        (
            "session loader",
            lambda: load_run("project", "run", workspace="relative-workspace"),
        ),
        ("upload staging sweep", lambda: sweep_staging("relative-workspace")),
        (
            "sandbox status service",
            lambda: SandboxStatusService(Path("relative-workspace")),
        ),
        (
            "settings service",
            lambda: SettingsService(workspace=Path("relative-workspace")),
        ),
        (
            "auto EDA",
            lambda: run_auto_eda([], workspace="relative-workspace"),
        ),
        (
            "investigation planning",
            lambda: create_investigation_plans(
                project_id="project",
                source_session_id="source",
                question_ids=[],
                workspace="relative-workspace",
            ),
        ),
        (
            "question execution",
            lambda: run_question_batch(
                project_id="project",
                source_session_id="source",
                question_ids=[],
                workspace="relative-workspace",
            ),
        ),
        (
            "synthesis",
            lambda: create_synthesis_brief(
                project_id="project",
                finding_artifact_ids=[],
                workspace="relative-workspace",
            ),
        ),
        (
            "workflow evaluation",
            lambda: run_fresh_workflow_eval_case(
                cast(WorkflowEvalSpec, object()),
                input_dir=Path("."),
                workspace=Path("relative-workspace"),
                repeat=1,
            ),
        ),
    ]


def test_all_public_workspace_boundaries_reject_relative_paths_from_any_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    absolute_store = ArtifactStore(tmp_path / "absolute-workspace")
    monkeypatch.chdir(elsewhere)

    for name, invoke in _relative_boundary_calls(absolute_store):
        with pytest.raises(WorkspaceConfigError, match="absolute"):
            invoke()
        assert not (elsewhere / "relative-workspace").exists(), name


@pytest.mark.parametrize(
    ("script_name", "extra_args"),
    [
        ("benchmark_offline.py", ("--files", "{csv}")),
        ("demo_j1.py", ("--files", "{csv}")),
        (
            "evaluate_workflow.py",
            (
                "--case",
                "{case}",
                "--input-dir",
                "{input_dir}",
            ),
        ),
    ],
)
def test_cli_rejects_relative_workspace_without_creating_it_under_cwd(
    script_name: str,
    extra_args: tuple[str, ...],
    tmp_path: Path,
) -> None:
    csv = tmp_path / "tiny.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    values = {
        "csv": str(csv),
        "case": str(
            REPO_ROOT
            / "eda_platform/tests/evals/workflow_quality/cases/semantic_guardrails.json"
        ),
        "input_dir": str(
            REPO_ROOT / "eda_platform/tests/evals/workflow_quality/data"
        ),
    }
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / script_name),
        "--workspace",
        "relative-workspace",
        *(part.format(**values) for part in extra_args),
    ]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "must be an absolute path" in completed.stderr
    assert not (tmp_path / "relative-workspace").exists()

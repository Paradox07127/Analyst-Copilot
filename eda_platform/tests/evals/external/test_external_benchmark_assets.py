"""Structure guards for the vendored external-benchmark samples (offline only).

Scoring against DABStep/KramaBench is deferred (needs their context data plus
a live LLM); these tests keep the vendored task samples intact and honestly
labelled so the deferred run starts from a known-good state.
"""

from __future__ import annotations

import json
from pathlib import Path

ASSET_DIR = Path(__file__).parent
DABSTEP_PATH = ASSET_DIR / "dabstep_hard_sample.json"
KRAMABENCH_PATH = ASSET_DIR / "kramabench_sample.json"
ADAPTER_NOTES_PATH = ASSET_DIR / "adapter_notes.md"


def test_dabstep_sample_has_5_to_10_hard_tasks_with_answers() -> None:
    data = json.loads(DABSTEP_PATH.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    assert 5 <= len(tasks) <= 10
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    for task in tasks:
        assert task["level"] == "hard", task["task_id"]
        assert task["question"].strip(), task["task_id"]
        assert str(task["answer"]).strip(), (
            f"{task['task_id']}: answer missing — dev split must be used "
            "(default split hides answers)"
        )
        assert task["guidelines"].strip(), task["task_id"]


def test_dabstep_meta_records_provenance_and_deferral() -> None:
    meta = json.loads(DABSTEP_PATH.read_text(encoding="utf-8"))["_meta"]
    assert "adyen/DABstep" in meta["source"]
    assert "split=dev" in meta["source"] or "dev" in meta["selection"]
    assert "NOT vendored" in meta["context_data"]


def test_kramabench_sample_has_3_to_5_hard_tasks_with_answers() -> None:
    data = json.loads(KRAMABENCH_PATH.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    assert 3 <= len(tasks) <= 5
    assert len({task["id"] for task in tasks}) == len(tasks)
    workloads = {task["workload"] for task in tasks}
    assert len(workloads) >= 3, "samples should span at least 3 domains"
    for task in tasks:
        assert "-hard-" in task["id"], task["id"]
        assert task["query"].strip(), task["id"]
        assert str(task["answer"]).strip() != "", task["id"]
        assert task["data_sources"], task["id"]
        assert task["subtasks"], f"{task['id']}: subtasks double as graded checkpoints"


def test_kramabench_meta_records_provenance_and_deferral() -> None:
    meta = json.loads(KRAMABENCH_PATH.read_text(encoding="utf-8"))["_meta"]
    assert "mitdbg/Kramabench" in meta["source"]
    assert "NOT vendored" in meta["context_data"]


def test_adapter_notes_cover_manual_steps_for_both_benchmarks() -> None:
    text = ADAPTER_NOTES_PATH.read_text(encoding="utf-8")
    for marker in (
        "snapshot_download",
        "git clone",
        "推迟",
        "dabstep_hard_sample.json",
        "kramabench_sample.json",
    ):
        assert marker in text, f"adapter_notes.md missing: {marker}"

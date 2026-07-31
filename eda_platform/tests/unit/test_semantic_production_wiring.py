"""Static guards for the single production semantic-seeds repository path."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_SRC = Path(__file__).parents[2] / "src" / "eda_platform"
_FORBIDDEN_CALLS = {
    "save_seeds",
    "add_verified_relation",
    "accept_proposal",
    "reject_proposal",
    "confirm_promotion",
    "save_meaning_proposals",
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_production_code_has_no_unversioned_semantic_seed_writer() -> None:
    violations: list[str] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in _FORBIDDEN_CALLS:
                violations.append(f"{path.relative_to(_SRC)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                if imported & _FORBIDDEN_CALLS:
                    violations.append(f"{path.relative_to(_SRC)}:{node.lineno}")
    assert violations == []


def test_legacy_writer_attributes_are_not_publicly_reachable() -> None:
    removed = {
        "eda_platform.core.semantic": {
            "save_seeds",
            "add_verified_relation",
        },
        "eda_platform.core.meaning_proposals": {
            "save_meaning_proposals",
            "upsert_proposals",
            "accept_proposal",
            "reject_proposal",
            "accept_all_verified",
        },
        "eda_platform.drivers.knowledge_promotion": {
            "confirm_promotion",
        },
    }
    for module_name, names in removed.items():
        module = importlib.import_module(module_name)
        assert all(not hasattr(module, name) for name in names)


def test_only_repository_owns_semantic_versions_file_in_production() -> None:
    owners: list[str] = []
    for path in _SRC.rglob("*.py"):
        if path == _SRC / "core" / "semantic_resources.py":
            continue
        source = path.read_text(encoding="utf-8")
        if '"versions.json"' in source or "'versions.json'" in source:
            owners.append(str(path.relative_to(_SRC)))
    assert owners == []


def test_runtime_seed_consumers_use_the_unified_safe_loader() -> None:
    readers = (
        _SRC / "drivers" / "auto_eda.py",
        _SRC / "drivers" / "question_exec.py",
        _SRC / "drivers" / "chat.py",
        _SRC / "drivers" / "investigation_orchestrator.py",
    )
    for path in readers:
        source = path.read_text(encoding="utf-8")
        assert "load_semantic_seeds_safe" in source
        assert "load_seeds(" not in source
        tree = ast.parse(source, filename=str(path))
        loader_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _call_name(node) == "load_semantic_seeds_safe"
        ]
        assert len(loader_calls) == 1

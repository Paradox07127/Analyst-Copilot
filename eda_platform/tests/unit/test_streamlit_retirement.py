"""Regression guards for the physical retirement of the legacy UI."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[3]
_PACKAGE = _ROOT / "eda_platform" / "src" / "eda_platform"
_WEB_SRC = _ROOT / "apps" / "web" / "src"

# Retired Streamlit module names that must not reappear in production source
# (comments, docstrings, or strings), with or without the ui/ prefix — bare
# `findings_ui.py` references survived the prefixed pattern for weeks.
_LEGACY_UI_MODULE_PATH = re.compile(r"\b(?:ui/)?[a-z0-9_]+_ui\.py\b", re.IGNORECASE)

_WEB_SKIP_DIR_NAMES = frozenset({"generated", "test", "e2e", "node_modules"})
_WEB_SOURCE_SUFFIXES = frozenset({".ts", ".tsx", ".css", ".js", ".jsx"})


def test_legacy_ui_source_config_and_dependency_are_absent() -> None:
    assert not (_PACKAGE / "ui").exists()
    assert not (_ROOT / ".streamlit").exists()

    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert not any(
        str(item).lower().startswith("streamlit") for item in dependencies
    )
    lock = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "streamlit"' not in lock


def test_production_python_has_no_streamlit_imports() -> None:
    offenders: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "streamlit" or name.startswith("streamlit.") for name in names):
                offenders.append(str(path.relative_to(_PACKAGE)))
    assert offenders == []


def test_launcher_and_current_docs_have_no_legacy_entrypoint() -> None:
    # launch.json and the cutover runbook are local ops artifacts, gitignored on
    # purpose (.claude/, /docs/) — validate them when present, skip otherwise.
    launch_path = _ROOT / ".claude" / "launch.json"
    runbook_path = _ROOT / "docs" / "infrastructure" / "react-cutover-runbook.md"

    if launch_path.is_file():
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        names = {str(item["name"]) for item in launch["configurations"]}
        assert "legacy-streamlit" not in names

    current_docs = [(_ROOT / "README.md").read_text(encoding="utf-8")]
    if runbook_path.is_file():
        current_docs.append(runbook_path.read_text(encoding="utf-8"))
    assert "uv run streamlit" not in "\n".join(current_docs)

    missing = [
        str(path.relative_to(_ROOT))
        for path in (launch_path, runbook_path)
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "local ops artifacts intentionally not committed: " + ", ".join(missing)
        )


def _legacy_ui_path_hits(path: Path) -> list[str]:
    """Return ``path:line: match`` for each retired ``ui/*_ui.py`` mention."""
    hits: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return hits
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in _LEGACY_UI_MODULE_PATH.finditer(line):
            hits.append(f"{path}:{line_no}: {match.group(0)}")
    return hits


def _iter_web_production_sources() -> list[Path]:
    if not _WEB_SRC.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(_WEB_SRC.rglob("*")):
        if not path.is_file() or path.suffix not in _WEB_SOURCE_SUFFIXES:
            continue
        if any(part in _WEB_SKIP_DIR_NAMES for part in path.parts):
            continue
        paths.append(path)
    return paths


def test_production_source_has_no_legacy_ui_module_paths() -> None:
    """Ban retired ``ui/*_ui.py`` paths from production comments/docstrings.

    After Streamlit retirement those modules are gone; mentioning them in
    production source re-introduces a false map of the codebase. Tests and
    archive docs may still describe the migration; this guard only covers
    shipping backend + web workbench sources.
    """
    offenders: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        offenders.extend(_legacy_ui_path_hits(path))
    for path in _iter_web_production_sources():
        offenders.extend(_legacy_ui_path_hits(path))
    assert offenders == [], (
        "retired legacy UI module paths must not appear in production source:\n"
        + "\n".join(offenders)
    )


def test_legacy_ui_path_scanner_matches_retired_module_names(
    tmp_path: Path,
) -> None:
    """Control: the ban pattern must catch real retired module paths."""
    probe = tmp_path / "commented.py"
    probe.write_text(
        "# see ui/synthesis_ui.py and ui/relationships_ui.py for the old flow\n"
        "value = 'ui/chat_ui.py'\n"
        "# not a hit: ui/helpers.py or streamlit_app.py\n",
        encoding="utf-8",
    )
    hits = _legacy_ui_path_hits(probe)
    matched = {hit.split(": ", 1)[1] for hit in hits}
    assert matched == {
        "ui/synthesis_ui.py",
        "ui/relationships_ui.py",
        "ui/chat_ui.py",
    }

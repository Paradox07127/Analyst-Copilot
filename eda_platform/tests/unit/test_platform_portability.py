"""Guards that keep the runtime importable and correct on Windows.

The platform split is allowed to exist, but only inside the few modules that
own it. Everything else must stay portable, because a POSIX-only import at
module scope makes the whole application fail to start on Windows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "eda_platform"

# Importing any of these at module scope raises ImportError on Windows.
POSIX_ONLY_MODULES = frozenset({"fcntl", "grp", "posix", "pwd", "termios"})

# Modules that own a platform split. Each must keep the import inside a
# platform branch, never at module scope.
PLATFORM_BOUNDARY_MODULES = frozenset({"core/file_lock.py"})

# The one module allowed to reach for a raw process primitive, because it is
# the module that chooses which primitive the platform actually supports.
PROCESS_CONTROL_MODULE = "core/process_control.py"


def _python_sources() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def _module_scope_imports(tree: ast.Module) -> set[str]:
    """Imports executed on import, ignoring those inside functions or branches."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_runtime_module_imports_a_posix_only_stdlib_module() -> None:
    offenders = {
        _relative(path): sorted(
            _module_scope_imports(ast.parse(path.read_text(encoding="utf-8")))
            & POSIX_ONLY_MODULES
        )
        for path in _python_sources()
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_posix_only_imports_stay_inside_the_platform_boundary_modules() -> None:
    """A conditional POSIX import elsewhere still splits the platform logic
    across the codebase; keep it where the fallback lives."""
    users = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: set[str] = set()
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module.split(".")[0]}
            if imported & POSIX_ONLY_MODULES:
                users.add(_relative(path))
    assert users <= PLATFORM_BOUNDARY_MODULES


@pytest.mark.parametrize(
    "forbidden",
    [
        # os.kill(pid, 0) is a liveness probe on POSIX; on Windows os.kill maps
        # every signal except the console events onto TerminateProcess, so this
        # would kill the process it claims to be probing.
        "os.kill(pid, 0)",
        # signal.SIGKILL does not exist on Windows.
        "signal.SIGKILL",
    ],
)
def test_runtime_avoids_windows_hostile_process_primitives(forbidden: str) -> None:
    offenders = [
        _relative(path)
        for path in _python_sources()
        if forbidden in path.read_text(encoding="utf-8")
        and _relative(path) != PROCESS_CONTROL_MODULE
    ]
    assert offenders == []


def test_process_control_guards_every_raw_primitive_behind_the_platform_check() -> None:
    """The one module holding the raw calls must branch on the platform, or the
    allowance above silently readmits the bug everywhere it is used."""
    source = (SOURCE_ROOT / PROCESS_CONTROL_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    branchless = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        if "signal.SIGKILL" in body and "_IS_WINDOWS" not in body:
            branchless.append(node.name)
    assert branchless == []


def test_no_module_reimplements_the_shared_platform_helpers() -> None:
    """Each of these has a Windows branch in exactly one place. A second copy
    is how the POSIX assumption creeps back in."""
    duplicates: dict[str, list[str]] = {}
    for marker, owner in (
        ("def fsync_directory", "core/fs.py"),
        ("def remove_tree", "core/fs.py"),
        ("def pid_is_alive", PROCESS_CONTROL_MODULE),
    ):
        found = [
            _relative(path)
            for path in _python_sources()
            if marker in path.read_text(encoding="utf-8")
        ]
        if found != [owner]:
            duplicates[marker] = found
    assert duplicates == {}


def test_descriptor_opens_of_regular_files_ask_for_binary_mode() -> None:
    """Windows os.open defaults to text mode and rewrites newlines, which
    corrupts staged payloads and breaks byte-exact identity readbacks."""
    offenders = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if "O_WRONLY" in line or "O_RDWR" in line:
                if "O_BINARY" not in source and "BINARY_FLAG" not in source:
                    offenders.append(_relative(path))
                break
    assert sorted(set(offenders)) == []

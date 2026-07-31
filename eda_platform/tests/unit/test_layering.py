"""Core must not import the application layer.

A core module reaching up into ``application`` is not just a style problem: it
already closed a session_deletion -> bounded_pagination -> session_service ->
session_deletion import cycle, which only stayed latent because of import order.
"""

from __future__ import annotations

import subprocess
import sys

CORE_MODULES = (
    "eda_platform.core.bounded_pagination",
    "eda_platform.core.session_deletion",
    "eda_platform.core.store",
)


def _application_modules_pulled_in(module: str) -> list[str]:
    script = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "print('\\n'.join(sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('eda_platform.application')\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_core_modules_do_not_import_the_application_layer() -> None:
    offenders = {
        module: _application_modules_pulled_in(module) for module in CORE_MODULES
    }
    assert offenders == dict.fromkeys(CORE_MODULES, [])


def test_core_session_deletion_imports_before_the_application_layer() -> None:
    """Importing core first must not raise: that is the cycle's failure mode."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import eda_platform.core.session_deletion\n"
            "import eda_platform.application.services.session_service\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

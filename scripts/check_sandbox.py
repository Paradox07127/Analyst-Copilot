"""Fail-closed operational check for the Docker CodeAgent sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eda_platform.core.sandbox import SandboxUnavailableError
from eda_platform.core.sandbox_broker import SandboxBroker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the effective Docker sandbox kernel controls."
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.cwd() / ".sandbox-preflight",
        help="directory used for the disposable canary execution",
    )
    args = parser.parse_args()
    try:
        info = SandboxBroker.from_env(
            work_root=args.work_root.resolve()
        ).require_safe_backend().info
    except SandboxUnavailableError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": info.available and info.safe_for_untrusted_code,
                "backend": info.name,
                "detail": info.detail,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

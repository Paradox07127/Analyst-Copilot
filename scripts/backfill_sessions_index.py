"""One-shot CLI: populate the runs-table session index for an existing workspace.

Usage:
    uv run python scripts/backfill_sessions_index.py [--workspace PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eda_platform.application.backfill import backfill_sessions_index
from eda_platform.core.config import resolve_workspace_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Absolute workspace root (default: EDA_WORKSPACE or the repository workspace)",
    )
    args = parser.parse_args()
    workspace = resolve_workspace_path(args.workspace)
    count = backfill_sessions_index(workspace)
    print(f"Indexed {count} runs in {workspace}")


if __name__ == "__main__":
    main()

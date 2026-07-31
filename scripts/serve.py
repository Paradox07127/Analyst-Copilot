"""Serve the production build on one port: FastAPI API + static React app.

Usage:
    npm run build --prefix apps/web
    uv run python scripts/serve.py [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from eda_platform.core.config import WorkspaceConfigError, resolve_workspace_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = REPO_ROOT / "apps" / "web" / "dist"


class ServeConfigError(Exception):
    pass


@dataclass(frozen=True)
class ServeConfig:
    host: str
    port: int
    dist: Path
    workspace: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the built web UI and the API from a single port."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--workspace",
        default=None,
        help="absolute workspace root; overrides EDA_WORKSPACE",
    )
    parser.add_argument(
        "--dist",
        default=None,
        help=f"web build directory (default: {DEFAULT_DIST})",
    )
    return parser


def resolve_workspace(cli_value: str | None) -> Path:
    """Resolve through the shared CWD-independent workspace configuration."""
    try:
        return resolve_workspace_path(cli_value, repo_root=REPO_ROOT)
    except WorkspaceConfigError as exc:
        raise ServeConfigError(str(exc)) from exc


def resolve_config(argv: list[str] | None = None) -> ServeConfig:
    args = build_parser().parse_args(argv)
    dist = Path(args.dist).expanduser().resolve() if args.dist else DEFAULT_DIST
    if not (dist / "index.html").is_file():
        raise ServeConfigError(
            f"web build not found at {dist} — run `npm run build --prefix apps/web` first"
        )
    workspace = resolve_workspace(args.workspace)
    return ServeConfig(host=args.host, port=args.port, dist=dist, workspace=workspace)


def main(argv: list[str] | None = None) -> int:
    try:
        config = resolve_config(argv)
    except ServeConfigError as exc:
        print(f"error: {exc}")
        return 1

    import uvicorn

    from eda_platform.api.main import create_app

    app = create_app(workspace=config.workspace, serve_web_dist=config.dist)
    print(f"workspace: {app.state.workspace}")
    print(f"web dist:  {config.dist}")
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

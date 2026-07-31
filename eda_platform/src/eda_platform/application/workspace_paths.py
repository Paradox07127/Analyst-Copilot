"""Rewrites absolute workspace paths in outgoing payloads to relative form,
shared by the artifact and insight read paths."""

from __future__ import annotations

import os
from pathlib import Path


def _relativize_str(node: str, prefix: str, prefix_sep: str) -> str:
    """Rewrite only genuine workspace paths: the root itself or root + separator.
    A bare startswith(prefix) would clobber siblings ("/…/ws_backup/x") and prose
    that merely begins with the root string ("/…/ws is configured")."""
    if node == prefix:
        return "."
    if node.startswith(prefix_sep):
        return node[len(prefix_sep) :].lstrip(os.sep) or "."
    return node


def relativize_workspace_paths(value: object, root: Path) -> dict:
    """Defense in depth: no scanned artifact carries server paths today, but a
    payload string that is a workspace path is rewritten relative."""
    prefix = str(root.resolve())
    prefix_sep = prefix + os.sep

    def walk(node: object) -> object:
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            return _relativize_str(node, prefix, prefix_sep)
        return node

    result = walk(dict(value) if isinstance(value, dict) else {})
    assert isinstance(result, dict)
    return result


def relativize_warnings(warnings: list[str], root: Path) -> list[str]:
    prefix = str(root.resolve())
    prefix_sep = prefix + os.sep
    return [_relativize_str(item, prefix, prefix_sep) for item in warnings]

"""Make the optional frozen scoreboard corpus explicit at collection time."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CORPUS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "scoreboard_corpus"
CORPUS_AVAILABLE = CORPUS_DIR.is_dir() and any(CORPUS_DIR.iterdir())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if CORPUS_AVAILABLE:
        return
    message = (
        "frozen scoreboard corpus is not present; provision "
        "eda_platform/tests/fixtures/scoreboard_corpus to run acceptance scoreboards"
    )
    if os.environ.get("EDA_REQUIRE_SCOREBOARD_CORPUS") == "1":
        raise pytest.UsageError(message)
    marker = pytest.mark.skip(reason=message)
    for item in items:
        if Path(str(item.path)).parent == Path(__file__).resolve().parent:
            item.add_marker(marker)

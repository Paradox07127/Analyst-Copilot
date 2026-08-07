"""The offline default must survive the live-eval opt-out.

`conftest.offline_llm_by_default` blanks every provider key so a developer with
real credentials cannot turn the suite into paid network calls. Letting the
NL2SQL eval through needed a hole in that guard, and a hole nobody watches is
how the guard quietly stops holding.

Opting in takes both halves -- the `live_llm` marker *and* `EDA_LIVE_LLM_TEST=1`
-- so setting the flag alone, which is what someone debugging that eval will
do, must leave every other test offline.
"""

from __future__ import annotations

import os

import pytest


def test_an_unmarked_test_is_offline_whatever_the_environment_says() -> None:
    assert os.environ["EDA_LLM_PROVIDER"] == "offline"
    assert os.environ["DEEPSEEK_API_KEY"] == ""
    assert os.environ["OPENAI_API_KEY"] == ""


@pytest.mark.live_llm
def test_the_marker_alone_does_not_unblank_the_keys() -> None:
    """Without the flag the marker is inert, so a stray marker cannot spend money."""
    if os.environ.get("EDA_LIVE_LLM_TEST") == "1":
        pytest.skip("the flag is set; this asserts the marker-only case")
    assert os.environ["EDA_LLM_PROVIDER"] == "offline"
    assert os.environ["DEEPSEEK_API_KEY"] == ""

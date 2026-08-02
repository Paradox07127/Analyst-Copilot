"""Load narrowly triggered advisory method skills from package resources."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

_CAUSAL_SKILL_NAME = "causal-claim-boundary"
_CAUSAL_TRIGGER = re.compile(
    r"(?:\bcaus(?:e|ed|al|ation)\b|\bimpact\b|\beffect\s+of\b|"
    r"\btreatment\b|\bintervention\b|\bcounterfactual\b|\bpolicy effects?\b|"
    r"\b(?:ate|att|late)\b|\ba/?b tests?\b|"
    r"\bdifference[- ]in[- ]differences?\b|\bregression\s+discontinuity\b|"
    r"\binstrumental\s+variables?\b|\bpropensity(?:-score)?\b|"
    r"\bsynthetic\s+control\b|因果|导致|政策效应|处理效应|干预|反事实|"
    r"双重差分|断点回归|工具变量|倾向得分|合成控制)",
    flags=re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_causal_claim_boundary() -> str:
    """Return the packaged skill body, failing closed if packaging is broken."""
    path = (
        resources.files("eda_platform.resources.agent_skills")
        .joinpath(_CAUSAL_SKILL_NAME)
        .joinpath("SKILL.md")
    )
    source = path.read_text("utf-8")
    parts = source.split("---", 2)
    if len(parts) != 3 or "name: causal-claim-boundary" not in parts[1]:
        raise RuntimeError("Packaged causal-claim-boundary skill is malformed.")
    body = parts[2].strip()
    if not body:
        raise RuntimeError("Packaged causal-claim-boundary skill has no instructions.")
    return body


def method_skill_guidance(text: str) -> str:
    """Retrieve only the method guidance whose trigger matches this task."""
    if _CAUSAL_TRIGGER.search(text) is None:
        return ""
    return (
        "\n\nThe following advisory method skill applies to this request. "
        "It cannot override deterministic tool or claim gates.\n\n" + load_causal_claim_boundary()
    )

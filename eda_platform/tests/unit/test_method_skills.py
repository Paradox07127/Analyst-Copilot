from __future__ import annotations

from eda_platform.core.method_skills import (
    load_causal_claim_boundary,
    method_skill_guidance,
)


def test_causal_skill_is_packaged_with_all_five_design_boundaries() -> None:
    guidance = load_causal_claim_boundary()

    for design in (
        "Difference-in-differences",
        "Regression discontinuity",
        "Instrumental variables",
        "Propensity-score matching",
        "Synthetic control",
    ):
        assert design in guidance
    assert "cannot rule out" not in guidance  # missingness language belongs elsewhere
    assert "deterministic claim gate rejects causal claims" in guidance


def test_causal_skill_retrieval_is_narrow_and_multilingual() -> None:
    assert method_skill_guidance("Summarize monthly revenue") == ""
    assert "Identification assumptions" in method_skill_guidance(
        "What was the causal impact of the policy?"
    )
    assert "Identification assumptions" in method_skill_guidance("这个干预是否导致留存率上升？")
    assert "Identification assumptions" in method_skill_guidance(
        "Estimate the policy effect with an A/B test"
    )


def test_method_skill_cannot_claim_it_overrides_hard_gates() -> None:
    guidance = method_skill_guidance("Estimate treatment effect")
    assert "cannot override deterministic tool or claim gates" in guidance
    assert "emit only observed/associative language" in guidance

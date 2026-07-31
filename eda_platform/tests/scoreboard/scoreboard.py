"""Offline scoreboard: fixed-corpus replay metrics for the validation gates.

Keep/revert discipline: every gate-affecting commit must hold every metric at
or above the frozen baseline; any regression is a revert signal, not a TODO.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus import CorpusRun, load_corpus

from eda_platform.tools import report_validator as rv

MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "scoreboard_corpus"
    / "expected_policy_manifest.json"
)
TOLERANCE = 0.01

MUTATIONS = (
    ("x1.05", lambda v: v * 1.05),
    ("x1.5", lambda v: v * 1.5),
    ("x2", lambda v: v * 2),
    ("plus1", lambda v: v + 1),
)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def policy_index(manifest: dict) -> dict[tuple[str, str, int], dict]:
    return {(e["run"], e["claim_id"], e["token_index"]): e for e in manifest["entries"]}


def _format_like(value: float, like_token: str) -> str:
    """Format a mutated value with the same decimals/percent shape as the original."""
    is_percent = like_token.endswith("%")
    core = like_token.rstrip("%")
    decimals = len(core.split(".")[1]) if "." in core else 0
    text = f"{value:.{decimals}f}" if decimals else str(int(round(value)))
    return text + ("%" if is_percent else "")


def compute_scoreboard(runs: list[CorpusRun] | None = None) -> dict:
    runs = runs if runs is not None else load_corpus()
    policies = policy_index(load_manifest())

    # Two detection scopes, ratcheted independently:
    # - hard: a finding attributes the exact mutated token (comparable to §5.7)
    # - disposition: hard OR the token's clean state is unverified (it was
    #   never published as verified, so the injection is not endorsed)
    hard_mutation: dict[str, list[int]] = {name: [0, 0] for name, _ in MUTATIONS}
    disp_mutation: dict[str, list[int]] = {name: [0, 0] for name, _ in MUTATIONS}
    hard_count_plus1 = [0, 0]
    disp_count_plus1 = [0, 0]
    clean_mismatch_claims = 0
    clean_unverified_tokens = 0
    collateral_only = 0
    resolved_refs = 0
    total_refs = 0
    # Prefix-injection wash probe: republish a clean verified token as
    # "< {2x}". Split by whether the own-unit pool holds threshold-eligible
    # (stat) values — without them the bound must be blocked; with them the
    # inequality is legal design and its pass-through is reported as-is.
    injection_no_eligible = [0, 0]  # [blocked, total]
    injection_eligible = [0, 0]  # [passed_by_design, total]
    # F3: quantitative-section claims with zero verified numbers on the clean
    # replay, measured through the production entry (validate_report_bundle).
    coverage_gaps_by_run: dict[str, int] = {}
    # F6: per-run evidence-strength distribution and the strong-ratio verdict.
    # Legacy qfocus_* claims (pre-F4 catalog form; their synthetic evidence
    # chain was deleted) are excluded from the denominator.
    strength_by_run: dict[str, dict] = {}

    for run in runs:
        tier_counts = {"strong": 0, "indicative": 0, "exploratory": 0}
        for section in run.bundle.sections:
            for claim in section.claims:
                if (claim.id or "").startswith("qfocus_"):
                    continue
                tier = rv.evidence_strength_label(
                    claim, evidence_pack=run.pack, sql_results=run.sql_results
                )
                tier_counts[tier] += 1
        strength_by_run[run.slug] = {
            **tier_counts,
            "verdict": rv.strong_ratio_verdict(
                tier_counts["strong"], sum(tier_counts.values())
            ),
        }
        audit = rv.validate_report_bundle(
            run.bundle, run.pack, sql_results=run.sql_results
        )
        coverage_gaps_by_run[run.slug] = audit.quantitative_coverage_gap_count
        for section in run.bundle.sections:
            for claim in section.claims:
                for evidence in claim.evidence:
                    total_refs += 1
                    if rv._resolve_evidence_numbers(evidence, run.pack, run.sql_results):
                        resolved_refs += 1
                tokens = list(rv._NUMBER_PATTERN.finditer(claim.text))
                if not tokens:
                    continue
                # Clean-replay token states; the pool does not change under
                # mutation, so an unverified token stays unverified when mutated.
                statuses = rv._numeric_token_statuses(
                    claim,
                    evidence_pack=run.pack,
                    numeric_tolerance=TOLERANCE,
                    sql_results=run.sql_results,
                )
                clean_unverified_tokens += sum(
                    1 for status in statuses if status.status == "unverified"
                )
                if rv._has_numeric_mismatch(
                    claim,
                    evidence_pack=run.pack,
                    numeric_tolerance=TOLERANCE,
                    sql_results=run.sql_results,
                ):
                    clean_mismatch_claims += 1
                    continue
                for index, match in enumerate(tokens):
                    token = match.group(0)
                    entry = policies.get((run.slug, claim.id, index))
                    if entry is not None and entry["policy"] == "threshold":
                        # Inequality assertions: a scaled value can still satisfy
                        # the stated bound, so equality-style mutation is not an error.
                        continue
                    try:
                        base_value = float(token.rstrip("%").replace(",", ""))
                    except ValueError:
                        continue
                    for name, mutate in MUTATIONS:
                        if name == "plus1" and base_value != int(base_value):
                            continue
                        new_token = _format_like(mutate(base_value), token)
                        if new_token == token:
                            continue  # rounding produced a no-op, not a real injection
                        mutated = claim.model_copy(
                            update={
                                "text": claim.text[: match.start()]
                                + new_token
                                + claim.text[match.end() :]
                            }
                        )
                        details = rv._numeric_mismatch_details(
                            mutated,
                            evidence_pack=run.pack,
                            numeric_tolerance=TOLERANCE,
                            sql_results=run.sql_results,
                        )
                        # Flagging some other (legit) number instead of the
                        # mutated token is collateral damage, not detection.
                        mutated_value = float(new_token.rstrip("%").replace(",", ""))
                        is_percent = new_token.endswith("%")
                        hard_caught = any(
                            detail.number == mutated_value
                            and detail.is_percent == is_percent
                            for detail in details
                        )
                        disp_caught = hard_caught or (
                            index < len(statuses)
                            and statuses[index].status == "unverified"
                        )
                        if details and not disp_caught:
                            collateral_only += 1
                        hard_mutation[name][1] += 1
                        hard_mutation[name][0] += int(hard_caught)
                        disp_mutation[name][1] += 1
                        disp_mutation[name][0] += int(disp_caught)
                        if (
                            name == "plus1"
                            and entry is not None
                            and entry["policy"] == "count"
                        ):
                            hard_count_plus1[1] += 1
                            hard_count_plus1[0] += int(hard_caught)
                            disp_count_plus1[1] += 1
                            disp_count_plus1[0] += int(disp_caught)
                parsed = rv._numeric_tokens_from_text(claim.text)
                pool_values = rv._numeric_evidence_values(
                    claim, run.pack, run.sql_results
                )
                for index, match in enumerate(tokens):
                    if (
                        index >= len(statuses)
                        or statuses[index].status != "number_verified"
                        or parsed[index].threshold_op is not None
                    ):
                        continue
                    token = match.group(0)
                    new_token = _format_like(
                        float(token.rstrip("%").replace(",", "")) * 2, token
                    )
                    if new_token == token:
                        continue  # doubling 0 injects no new value
                    injected = claim.model_copy(
                        update={
                            "text": claim.text[: match.start()]
                            + "< "
                            + new_token
                            + claim.text[match.end() :]
                        }
                    )
                    injected_statuses = rv._numeric_token_statuses(
                        injected,
                        evidence_pack=run.pack,
                        numeric_tolerance=TOLERANCE,
                        sql_results=run.sql_results,
                    )
                    blocked = injected_statuses[index].status != "number_verified"
                    has_eligible = any(
                        eligible
                        for _value, unit, _policy, eligible in pool_values
                        if (unit == "percent") == parsed[index].is_percent
                    )
                    if has_eligible:
                        injection_eligible[1] += 1
                        injection_eligible[0] += int(not blocked)
                    else:
                        injection_no_eligible[1] += 1
                        injection_no_eligible[0] += int(blocked)

    def _scope(mutation: dict[str, list[int]], count_plus1: list[int]) -> dict:
        return {
            "mutation_overall": [
                sum(stats[0] for stats in mutation.values()),
                sum(stats[1] for stats in mutation.values()),
            ],
            "mutation_by_class": {name: list(stats) for name, stats in mutation.items()},
            "count_plus1": list(count_plus1),
        }

    return {
        "hard_gate_detection": _scope(hard_mutation, hard_count_plus1),
        "not_verified_disposition": _scope(disp_mutation, disp_count_plus1),
        "resolvable_refs": [resolved_refs, total_refs],
        "clean_claim_mismatches": clean_mismatch_claims,
        "clean_unverified_tokens": clean_unverified_tokens,
        "collateral_only_detections": collateral_only,
        "threshold_injection": {
            "no_eligible_blocked": injection_no_eligible,
            "eligible_passed_by_design": injection_eligible,
        },
        "quantitative_coverage_gaps": {
            "total": sum(coverage_gaps_by_run.values()),
            "by_run": coverage_gaps_by_run,
        },
        "evidence_strength": {
            "strong_ratio_cut": rv.STRONG_RATIO_CUT,
            "by_run": strength_by_run,
        },
    }


if __name__ == "__main__":
    print(json.dumps(compute_scoreboard(), indent=1))

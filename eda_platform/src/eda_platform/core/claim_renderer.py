"""Deterministic report renderer over gate-passed ClaimBundles (R3.4, §4.6).

Pure function, zero LLM and zero I/O: when the SYNTHESIZE polish step is
unavailable the passed claims still ship. A final numeric rescan refuses any
token the renderer itself would introduce — the same ruler E4a polish must pass.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from eda_platform.core.claim_gates import GateReport, claim_bundle_digest
from eda_platform.schemas.claims import ClaimBundle, EvidenceLane
from eda_platform.tools.report_validator import (
    numeric_tokens_from_text,
    value_supports_token,
)

_HEADER_KEYS = ("exploration_id", "policy_fingerprint", "witness")
_MISSING_METADATA = "(not provided)"
_EMPTY_SECTION = "(none)"
_WITNESS_SHORT_CHARS = 12

class RenderedNumberLeakError(ValueError):
    """The rendered text contains a numeric token outside the allowed pool."""


class RenderedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str
    rendered_bundle_ids: tuple[str, ...]
    withheld_bundle_ids: tuple[str, ...]


def numeric_pool_from_texts(texts: Iterable[str]) -> frozenset[tuple[float, bool]]:
    """Allowed-number pool: every (value, is_percent) token in the given texts."""
    return frozenset(
        (token.value, token.is_percent)
        for text in texts
        for token in numeric_tokens_from_text(text)
    )


def assert_rendered_numbers(
    text: str, allowed_pool: Collection[tuple[float, bool]]
) -> list[str]:
    """§4.6 rescan of final rendered text: return tokens outside the pool.

    Exact-value match only (percent and raw pools stay apart): reformatting a
    pool value is admissible, adding precision or a new number is not.
    """
    offenders: list[str] = []
    for token in numeric_tokens_from_text(text):
        supported = any(
            is_percent == token.is_percent
            and value_supports_token(token, value, "exact")
            for value, is_percent in allowed_pool
        )
        if not supported:
            rendered = f"{token.value:g}"
            offenders.append(f"{rendered}%" if token.is_percent else rendered)
    return offenders


def render_claim_report(
    bundles: Sequence[tuple[ClaimBundle, GateReport]],
    *,
    run_metadata: Mapping[str, str],
) -> RenderedReport:
    """Render passed bundles into fixed-structure markdown, byte-deterministic."""
    if bundles and not run_metadata.get("witness"):
        raise ValueError("run metadata must include the witness used by claim gates.")
    for bundle, report in bundles:
        if bundle.claim_bundle_id != report.claim_bundle_id:
            raise ValueError(
                f"bundle {bundle.claim_bundle_id!r} is paired with the gate "
                f"report of {report.claim_bundle_id!r}."
            )
        if claim_bundle_digest(bundle) != report.claim_bundle_digest:
            raise ValueError(
                f"bundle {bundle.claim_bundle_id!r} does not match the exact "
                "content evaluated by its gate report."
            )
        metadata_witness = run_metadata.get("witness")
        if metadata_witness != report.run_witness:
            raise ValueError(
                f"bundle {bundle.claim_bundle_id!r} was gated against a different "
                "run witness than the report metadata."
            )
    ordered = sorted(bundles, key=lambda pair: pair[0].claim_bundle_id)
    rendered_pairs = [(b, r) for b, r in ordered if r.passed]
    withheld_ids = tuple(b.claim_bundle_id for b, r in ordered if not r.passed)

    header = {key: run_metadata.get(key, _MISSING_METADATA) for key in _HEADER_KEYS}
    if "witness" in run_metadata:
        header["witness"] = _short_witness(run_metadata["witness"])

    lines = ["# Exploration findings", ""]
    lines += [f"- {key}: {header[key]}" for key in _HEADER_KEYS]
    lines += ["", "## Confirmed findings", ""]
    lines += _lane_lines(rendered_pairs, "confirmatory")
    lines += ["", "## Exploratory observations", ""]
    lines += _lane_lines(rendered_pairs, "exploratory")
    lines += ["", "## Evidence trail", ""]
    lines += _trail_lines(rendered_pairs)
    lines += ["", "## Not rendered", ""]
    lines.append(f"- rendered bundles: {len(rendered_pairs)}")
    lines.append(f"- withheld bundles (rejected or abstained): {len(withheld_ids)}")
    markdown = "\n".join(lines) + "\n"

    offenders = assert_rendered_numbers(
        markdown, _allowed_pool(rendered_pairs, header, len(withheld_ids))
    )
    if offenders:
        raise RenderedNumberLeakError(
            "rendered report introduces numeric token(s) outside the evidence "
            f"pool: {offenders}"
        )
    return RenderedReport(
        markdown=markdown,
        rendered_bundle_ids=tuple(b.claim_bundle_id for b, _r in rendered_pairs),
        withheld_bundle_ids=withheld_ids,
    )


def _allowed_pool(
    rendered_pairs: list[tuple[ClaimBundle, GateReport]],
    header: Mapping[str, str],
    withheld_count: int,
) -> frozenset[tuple[float, bool]]:
    # Input-derived fragments plus the two tally counts; fixed literals never
    # feed the pool, so a digit sneaking into one raises at render time.
    texts = [str(len(rendered_pairs)), str(withheld_count)]
    texts.extend(header.values())
    for bundle, report in rendered_pairs:
        texts.append(bundle.claim_bundle_id)
        texts.extend(bundle.referenced_receipt_ids())
        for claim in bundle.claims:
            texts.append(claim.claim_id)
            texts.append(claim.claim_text)
        for verdict in report.verdicts:
            if verdict.gate == "statistical":
                texts.extend(v.code for v in verdict.violations)
    return numeric_pool_from_texts(texts)


def _lane_lines(
    pairs: list[tuple[ClaimBundle, GateReport]], lane: EvidenceLane
) -> list[str]:
    lines = [
        f"- ({bundle.claim_bundle_id}/{claim.claim_id}) {claim.claim_text}"
        for bundle, _report in pairs
        if bundle.evidence_lane == lane
        for claim in sorted(bundle.claims, key=lambda c: c.claim_id)
    ]
    return lines or [_EMPTY_SECTION]


def _trail_lines(pairs: list[tuple[ClaimBundle, GateReport]]) -> list[str]:
    lines = [
        f"- {bundle.claim_bundle_id}: "
        + ", ".join(sorted(bundle.referenced_receipt_ids()))
        for bundle, _report in pairs
    ]
    return lines or [_EMPTY_SECTION]


def _short_witness(value: str) -> str:
    prefix, sep, rest = value.partition("_")
    if not sep:
        return value if len(value) <= _WITNESS_SHORT_CHARS else value[:_WITNESS_SHORT_CHARS] + "…"
    if len(rest) <= _WITNESS_SHORT_CHARS:
        return value
    return f"{prefix}_{rest[:_WITNESS_SHORT_CHARS]}…"

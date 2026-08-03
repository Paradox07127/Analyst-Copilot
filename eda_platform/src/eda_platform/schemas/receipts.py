"""EvidenceReceipt — the typed, tamper-evident envelope of one tool call.

Only receipt facts (never tool prose or model text) feed the claim gates, so
the envelope must be self-consistent by construction: derivations reference
real facts, absence facts require an empty result, and `content_digest`
covers every other field. There is deliberately no HMAC in v1 — the threat
model is model fabrication, not a hostile local administrator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)


class ReceiptIntegrityError(ValueError):
    """A receipt failed digest, witness or structural verification."""


_RECEIPT_ID_RE = re.compile(r"^rcpt_[0-9a-f]{24}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class _FrozenParams(Mapping[str, Any]):
    """An immutable mapping of scalar parameters.

    Deliberately not a dict subclass: `|=`, `dict.update(d, ...)` and
    `dict.__init__(d, ...)` reach the C-level storage directly, so no
    dict-subclass blacklist can hold. The named mutators are still defined so
    the failure is a TypeError about immutability rather than an AttributeError.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("Receipt method parameters are immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable

    def __reduce__(self) -> tuple[Any, ...]:
        # mappingproxy cannot be pickled at all; this keeps deepcopy/pickle of a
        # receipt working without exposing a mutable view.
        return (_FrozenParams, (dict(self._data),))


FactSupportType = Literal["direct", "compression", "inference", "absence"]
FactValueType = Literal["number", "percent", "count", "string", "bool", "null"]
# R4: how this receipt's evidence relates to prior evidence for the same
# hypothesis. Only holdout/external_replication license an "independent
# replication" claim; a second query on the same snapshot is corroboration.
ReplicationKind = Literal[
    "same_snapshot_corroboration", "holdout", "external_replication"
]
DerivationOperator = Literal[
    "percentage", "difference", "relative_change", "weighted_average", "ratio"
]


class ReceiptFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    name: str
    value: float | int | str | bool | None
    value_type: FactValueType
    unit: str | None = None
    support_type: FactSupportType = "direct"


class ReceiptDerivation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    derived_fact_id: str
    operator: DerivationOperator
    # Inputs only; the validator recomputes the result and never trusts a
    # model-supplied value for it.
    input_fact_ids: tuple[str, ...]


class ReceiptExecution(BaseModel):
    """Executor-assigned call identity; never derivable from tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    provider_call_id: str
    logical_step_id: str
    attempt_epoch: int = 0
    sequence_index: int = 0


class ReceiptScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_ids: tuple[str, ...] = ()
    # Always the columns actually scanned; when the tool received no explicit
    # list, the wrapper resolves them first (scope_resolution == "resolved").
    # "whole_dataset" is the one case where an empty `columns` is a statement
    # rather than a gap: every column was scanned and none were enumerated.
    columns: tuple[str, ...] = ()
    omitted_columns: tuple[str, ...] = ()
    # None means "not recorded" and only survives for receipts persisted before
    # the rule; agents.receipts.build_receipt refuses it on new receipts.
    scope_resolution: Literal["explicit", "resolved", "whole_dataset"] | None = None
    filters: str | None = None
    time_range: str | None = None

    @model_validator(mode="after")
    def _resolution_matches_the_columns(self) -> ReceiptScope:
        if self.scope_resolution in {"explicit", "resolved"} and not self.columns:
            raise ValueError(
                f"scope_resolution {self.scope_resolution!r} claims a column "
                "scope but columns is empty; use 'whole_dataset' for a "
                "whole-dataset scan."
            )
        if self.scope_resolution == "whole_dataset" and (
            self.columns or self.omitted_columns
        ):
            raise ValueError(
                "scope_resolution 'whole_dataset' covers every column; list "
                "them under 'resolved' instead of mixing the two."
            )
        return self


class ReceiptManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    row_index: int
    status: Literal["evaluated", "unevaluated"]
    row_digest: str

    @model_validator(mode="after")
    def _well_formed(self) -> ReceiptManifestEntry:
        if self.row_index < 0:
            raise ValueError("row_index cannot be negative.")
        if not _SHA256_HEX_RE.fullmatch(self.row_digest):
            raise ValueError("row_digest must be a 64-character sha256 hex digest.")
        return self


class ReceiptFactManifest(BaseModel):
    """Bounded per-row index of a published result: every publishable row has
    a fact id; rows without inline facts are explicitly unevaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    total_rows: int
    unlisted_rows: int = 0
    entries: tuple[ReceiptManifestEntry, ...] = ()

    @model_validator(mode="after")
    def _bounded_and_consistent(self) -> ReceiptFactManifest:
        if self.total_rows < 0 or self.unlisted_rows < 0:
            raise ValueError("manifest row counts cannot be negative.")
        if len(self.entries) > 1024:
            raise ValueError("a fact manifest lists at most 1024 rows.")
        if len(self.entries) + self.unlisted_rows != self.total_rows:
            raise ValueError(
                "manifest entries + unlisted_rows must equal total_rows."
            )
        fact_ids = [entry.fact_id for entry in self.entries]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("manifest fact ids must be unique.")
        return self


class ReceiptMethod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    parameters: Mapping[str, str | int | float | bool | None] = {}
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def hypothesis_evidence_is_explicitly_invalid(self) -> bool:
        """Return the producer-owned validity decision for hypothesis evidence.

        Warning prose is intentionally not interpreted here. Tool adapters and
        executor-side adjudicators can set this structured parameter when a
        method precondition invalidates the receipt as hypothesis evidence.
        """
        return self.parameters.get("hypothesis_evidence_valid") is False

    @field_validator("parameters", mode="after")
    @classmethod
    def _freeze_parameters(
        cls, value: Mapping[str, str | int | float | bool | None]
    ) -> Mapping[str, str | int | float | bool | None]:
        # Pydantic's frozen config is shallow; this severs input aliasing and
        # makes the stored mapping immutable.
        return _FrozenParams(value)

    @field_serializer("parameters")
    def _dump_parameters(
        self, value: Mapping[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        # The digest is taken over this view; it must stay a plain dict so the
        # canonical JSON is byte-identical to receipts already on disk.
        return dict(value)


class ReceiptStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str | None = None
    statistical_family_id: str | None = None
    # Executor-owned relation between the measured result and the named
    # hypothesis. The reducer never infers this from prose, p-values, ordering,
    # or the model's requested tool sequence.
    hypothesis_outcome: Literal["supports", "contradicts"] | None = None
    test_name: str
    test_statistic: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    sample_size: int | None = None
    sequence_index: int | None = None

    @model_validator(mode="after")
    def _outcome_requires_a_hypothesis(self) -> ReceiptStatistics:
        if self.hypothesis_outcome is not None and not (self.hypothesis_id or "").strip():
            raise ValueError("hypothesis_outcome requires a non-empty hypothesis_id.")
        return self

    def has_valid_numeric_values(self) -> bool:
        """Whether populated statistics are finite and internally ordered."""
        values = (
            self.test_statistic,
            self.p_value,
            self.adjusted_p_value,
            self.effect_size,
            self.ci_low,
            self.ci_high,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            return False
        return not (
            self.ci_low is not None
            and self.ci_high is not None
            and self.ci_low > self.ci_high
        )


class EvidenceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str
    tool_call_id: str
    execution: ReceiptExecution | None = None
    tool_name: str
    tool_version: str
    input_digest: str
    output_digest: str
    artifact_ids: tuple[str, ...]
    result_count: int
    scope: ReceiptScope
    facts: tuple[ReceiptFact, ...]
    derivations: tuple[ReceiptDerivation, ...] = ()
    method: ReceiptMethod
    statistics: ReceiptStatistics | None = None
    fact_manifest: ReceiptFactManifest | None = None
    evidence_independence_key: str | None = None
    replication_kind: ReplicationKind | None = None
    data_state_witness: str
    created_at: str
    content_digest: str

    @model_validator(mode="after")
    def _internally_consistent(self) -> EvidenceReceipt:
        if not _RECEIPT_ID_RE.fullmatch(self.receipt_id):
            raise ValueError("receipt_id must match rcpt_<24 hex chars>.")
        for name in ("input_digest", "output_digest", "content_digest"):
            if not _SHA256_HEX_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a 64-character lowercase sha256 hex digest.")
        if not self.tool_call_id:
            raise ValueError("tool_call_id must be non-empty.")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique within a receipt.")
        facts_by_id = {fact.fact_id: fact for fact in self.facts}
        known = set(fact_ids)
        seen_derived: set[str] = set()
        for derivation in self.derivations:
            if derivation.derived_fact_id in seen_derived:
                raise ValueError(
                    f"duplicate derivation id {derivation.derived_fact_id!r}."
                )
            seen_derived.add(derivation.derived_fact_id)
            missing = [ref for ref in derivation.input_fact_ids if ref not in known]
            if missing:
                raise ValueError(
                    f"derivation {derivation.derived_fact_id!r} references "
                    f"unknown fact ids: {missing}."
                )
            if derivation.derived_fact_id in known:
                raise ValueError(
                    f"derived fact id {derivation.derived_fact_id!r} collides with a fact."
                )
            _check_derivation_semantics(derivation, facts_by_id)
        if self.fact_manifest is not None:
            for entry in self.fact_manifest.entries:
                if entry.status != "evaluated":
                    continue
                backed = entry.fact_id in known or any(
                    fact_id.startswith(entry.fact_id + ".") for fact_id in known
                )
                if not backed:
                    raise ValueError(
                        f"evaluated manifest entry {entry.fact_id!r} has no "
                        "backing inline fact."
                    )
        if self.result_count < 0:
            raise ValueError("result_count cannot be negative.")
        if self.result_count != 0:
            offenders = [f.fact_id for f in self.facts if f.support_type == "absence"]
            if offenders:
                # An absence fact asserts "nothing matched"; a non-empty result
                # contradicts it by construction (NabaOS abhava rule).
                raise ValueError(
                    f"absence facts {offenders} require result_count == 0, "
                    f"got {self.result_count}."
                )
        return self


_NUMERIC_VALUE_TYPES = {"number", "count", "percent"}
_BINARY_OPERATORS = {"percentage", "difference", "relative_change", "ratio"}
_SAME_UNIT_OPERATORS = {"percentage", "difference", "relative_change"}
_DENOMINATOR_OPERATORS = {"percentage", "relative_change", "ratio"}


def _check_derivation_semantics(
    derivation: ReceiptDerivation,
    facts_by_id: dict[str, ReceiptFact],
) -> None:
    """Operator arity, numeric inputs, unit agreement and divide-by-zero rules."""
    operator = derivation.operator
    arity = len(derivation.input_fact_ids)
    if operator in _BINARY_OPERATORS and arity != 2:
        raise ValueError(
            f"derivation {derivation.derived_fact_id!r}: operator {operator} "
            f"requires exactly 2 input facts, got {arity}."
        )
    if operator == "weighted_average" and (arity < 2 or arity % 2):
        raise ValueError(
            f"derivation {derivation.derived_fact_id!r}: weighted_average "
            f"requires an even number (>= 2) of input facts, got {arity}."
        )
    for ref in derivation.input_fact_ids:
        fact = facts_by_id[ref]
        if (
            fact.value_type not in _NUMERIC_VALUE_TYPES
            or isinstance(fact.value, bool)
            or not isinstance(fact.value, (int, float))
        ):
            raise ValueError(
                f"derivation {derivation.derived_fact_id!r}: input fact {ref!r} "
                "must be numeric."
            )
    if operator in _SAME_UNIT_OPERATORS:
        units = {facts_by_id[ref].unit for ref in derivation.input_fact_ids}
        if len(units) > 1:
            raise ValueError(
                f"derivation {derivation.derived_fact_id!r}: operator {operator} "
                f"requires one unit across inputs, got {sorted(str(u) for u in units)}."
            )
    if operator in _DENOMINATOR_OPERATORS:
        denominator = facts_by_id[derivation.input_fact_ids[1]]
        if float(denominator.value) == 0.0:  # type: ignore[arg-type]
            raise ValueError(
                f"derivation {derivation.derived_fact_id!r}: denominator fact "
                f"{denominator.fact_id!r} is zero."
            )


# Fields added after receipts were first persisted (R4). When None they are
# dropped from the digest view, so pre-R4 payloads verify byte-for-byte; any
# set value is covered, so forging, altering or stripping it breaks the digest.
_OMITTED_FROM_DIGEST_WHEN_NONE = frozenset(
    {"evidence_independence_key", "replication_kind"}
)


def receipt_content_digest(receipt_fields: dict[str, object]) -> str:
    """Digest over every field except content_digest itself.

    NabaOS's published signature leaves result_count and facts uncovered while
    claiming they are protected; covering the full canonical body closes that
    hole rather than reproducing it.
    """
    body = {
        key: value
        for key, value in receipt_fields.items()
        if key != "content_digest"
        and not (value is None and key in _OMITTED_FROM_DIGEST_WHEN_NONE)
    }
    # ``hypothesis_outcome`` was added after statistical receipts were already
    # persisted. Omit its unset form so those receipts retain their original
    # digest; any explicit supports/contradicts value remains covered.
    statistics = body.get("statistics")
    if isinstance(statistics, dict):
        body["statistics"] = {
            key: value
            for key, value in statistics.items()
            if not (
                value is None
                and key in {"hypothesis_outcome", "statistical_family_id"}
            )
        }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_receipt_digest(receipt: EvidenceReceipt) -> bool:
    return receipt.content_digest == receipt_content_digest(receipt.model_dump(mode="json"))


def load_verified_receipt(payload: Mapping[str, Any]) -> EvidenceReceipt:
    """Load a persisted receipt, refusing any payload whose digest fails."""
    receipt = EvidenceReceipt.model_validate(payload)
    if not verify_receipt_digest(receipt):
        raise ReceiptIntegrityError(
            "EvidenceReceipt content digest does not verify; the payload was "
            "tampered with or corrupted."
        )
    return receipt


WITNESS_SCHEMA_VERSION = 1

# (dataset_id, profile_artifact_id or None, manifest_input_hash)
WitnessEntry = tuple[str, "str | None", str]


def data_state_witness_digest(entries: Sequence[WitnessEntry]) -> str:
    """Versioned canonical digest of the {dataset, profile, manifest} triplets."""
    body = {
        "witness_schema_version": WITNESS_SCHEMA_VERSION,
        "datasets": sorted(
            (
                {
                    "dataset_id": dataset_id,
                    "profile_artifact_id": profile_artifact_id,
                    "manifest_input_hash": manifest_input_hash,
                }
                for dataset_id, profile_artifact_id, manifest_input_hash in entries
            ),
            key=lambda entry: str(entry["dataset_id"]),
        ),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return f"dsw{WITNESS_SCHEMA_VERSION}_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_data_state_witness(witness: str, entries: Sequence[WitnessEntry]) -> None:
    """Fail closed when the recorded witness no longer matches the data state."""
    expected = data_state_witness_digest(entries)
    if witness != expected:
        raise ReceiptIntegrityError(
            "data_state_witness mismatch: the dataset schema/profile state has "
            "changed since this receipt was produced."
        )

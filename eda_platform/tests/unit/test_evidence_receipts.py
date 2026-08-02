"""EvidenceReceipt: the typed envelope every analysis tool must emit.

The gate design (exploration plan §6) only admits claims whose numbers resolve
against receipt facts, so the receipt itself must be tamper-evident and
self-consistent: the content digest covers every field, derivations may only
reference facts that exist, and an absence fact is only constructible on an
empty result. NabaOS's published signature formula omits result_count/facts
from the MAC input while claiming they are protected — these tests pin the
corrected behaviour (digest covers everything) so we cannot inherit that bug.
"""

from __future__ import annotations

import copy
import operator
import pickle

import pytest
from pydantic import ValidationError

from eda_platform.agents.receipts import build_receipt
from eda_platform.schemas.artifacts import ArtifactType
from eda_platform.schemas.receipts import (
    EvidenceReceipt,
    ReceiptDerivation,
    ReceiptExecution,
    ReceiptFact,
    ReceiptIntegrityError,
    ReceiptMethod,
    ReceiptScope,
    load_verified_receipt,
    receipt_content_digest,
    verify_receipt_digest,
)


def _receipt(**overrides: object) -> EvidenceReceipt:
    fields: dict[str, object] = {
        "tool_call_id": "call_1",
        "tool_name": "run_sql",
        "tool_version": "1",
        "arguments": {"sql": "SELECT count(*) AS n FROM orders", "purpose": "count rows"},
        "raw_output": {"rows": [{"n": 42}]},
        "artifact_ids": ("sql_abc",),
        "result_count": 1,
        "scope": ReceiptScope(dataset_ids=("ds_orders",), columns=("n",)),
        "facts": (
            ReceiptFact(
                fact_id="f1",
                name="n",
                value=42,
                value_type="count",
                unit="raw",
                support_type="direct",
            ),
        ),
        "method": ReceiptMethod(family="sql_aggregation"),
        "data_state_witness": "witness_1",
        "created_at": "2026-08-01T00:00:00Z",
    }
    fields.update(overrides)
    return build_receipt(**fields)  # type: ignore[arg-type]


def test_the_digest_covers_every_field() -> None:
    """Tampering with any field — including result_count and facts — must show."""
    receipt = _receipt()
    assert verify_receipt_digest(receipt)

    for field, value in [
        ("result_count", 99),
        ("tool_name", "another_tool"),
        ("data_state_witness", "witness_2"),
        (
            "facts",
            (
                ReceiptFact(
                    fact_id="f1",
                    name="n",
                    value=777,
                    value_type="count",
                    unit="raw",
                    support_type="direct",
                ),
            ),
        ),
    ]:
        tampered = receipt.model_copy(update={field: value})
        assert not verify_receipt_digest(tampered), f"tampered {field} went undetected"


def test_the_same_inputs_produce_the_same_receipt_identity() -> None:
    """Receipt ids and digests are content-derived, not random."""
    first, second = _receipt(), _receipt()
    assert first.receipt_id == second.receipt_id
    assert first.content_digest == second.content_digest


def test_a_derivation_may_only_reference_existing_facts() -> None:
    receipt = _receipt()
    with pytest.raises(ValidationError, match="unknown fact"):
        EvidenceReceipt.model_validate(
            {
                **receipt.model_dump(),
                "derivations": [
                    {
                        "derived_fact_id": "d1",
                        "operator": "percentage",
                        "input_fact_ids": ["f1", "f_missing"],
                    }
                ],
            }
        )


def test_an_absence_fact_requires_an_empty_result() -> None:
    """`result_count == 0` is the only ground for an absence fact."""
    receipt = _receipt()
    with pytest.raises(ValidationError, match="absence"):
        EvidenceReceipt.model_validate(
            {
                **receipt.model_dump(),
                "facts": [
                    {
                        "fact_id": "f1",
                        "name": "no_nulls",
                        "value": None,
                        "value_type": "null",
                        "unit": None,
                        "support_type": "absence",
                    }
                ],
            }
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceReceipt.model_validate({**_receipt().model_dump(), "surprise": 1})


def _fact_for(
    fact_id: str, value: object, value_type: str, unit: str | None = "raw"
) -> ReceiptFact:
    return ReceiptFact(
        fact_id=fact_id,
        name=fact_id,
        value=value,  # type: ignore[arg-type]
        value_type=value_type,  # type: ignore[arg-type]
        unit=unit,
    )


def _receipt_with_derivations(
    facts: tuple[ReceiptFact, ...],
    derivations: list[dict[str, object]],
) -> EvidenceReceipt:
    return build_receipt(
        tool_call_id="call_1",
        tool_name="run_sql",
        tool_version="1",
        arguments={"sql": "SELECT 1"},
        raw_output={"rows": []},
        artifact_ids=(),
        result_count=1,
        scope=ReceiptScope(dataset_ids=("ds_orders",), scope_resolution="whole_dataset"),
        facts=facts,
        derivations=tuple(
            ReceiptDerivation.model_validate(derivation) for derivation in derivations
        ),
        method=ReceiptMethod(family="sql_aggregation"),
        data_state_witness="witness_1",
        created_at="2026-08-01T00:00:00Z",
    )


def test_load_verified_receipt_accepts_intact_and_rejects_tampered_payloads() -> None:
    receipt = _receipt()
    payload = receipt.model_dump(mode="json")
    assert load_verified_receipt(payload) == receipt

    tampered = dict(payload)
    tampered["result_count"] = 99
    with pytest.raises(ReceiptIntegrityError):
        load_verified_receipt(tampered)


def test_every_single_field_tamper_is_rejected_on_load() -> None:
    """Item 7: per-field tamper matrix over the whole envelope."""
    receipt = _receipt()
    payload = receipt.model_dump(mode="json")
    hex24 = "0" * 24 if not payload["receipt_id"].endswith("0" * 4) else "1" * 24
    hex64 = "f" * 64 if payload["input_digest"] != "f" * 64 else "e" * 64
    tampering: dict[str, object] = {
        "receipt_id": f"rcpt_{hex24}",
        "tool_call_id": payload["tool_call_id"] + "x",
        "execution": {
            "run_id": "run_forged",
            "provider_call_id": "call_forged",
            "logical_step_id": "step_forged",
        },
        "tool_name": "another_tool",
        "tool_version": "2",
        "input_digest": hex64,
        "output_digest": hex64,
        "artifact_ids": ["sql_zzz"],
        "result_count": 2,
        "scope": {"dataset_ids": ["ds_other"]},
        "facts": [{**payload["facts"][0], "value": 777}],
        "derivations": [
            {"derived_fact_id": "d1", "operator": "ratio", "input_fact_ids": ["f1", "f1"]}
        ],
        "method": {"family": "another_family"},
        "statistics": {"test_name": "forged_test"},
        "data_state_witness": "witness_2",
        "created_at": "2027-01-01T00:00:00Z",
        "evidence_independence_key": "forged_key",
        "replication_kind": "external_replication",
    }
    untouched = set(payload) - set(tampering) - {"content_digest", "fact_manifest"}
    assert not untouched, f"tamper matrix must cover every field, missing: {untouched}"
    for field, forged in tampering.items():
        tampered = {**payload, field: forged}
        with pytest.raises((ValidationError, ReceiptIntegrityError)):
            load_verified_receipt(tampered)


def test_strict_identifier_and_digest_formats() -> None:
    payload = _receipt().model_dump(mode="json")
    for field, bad in [
        ("receipt_id", "rcpt_NOTHEX"),
        ("receipt_id", "other_" + "0" * 24),
        ("input_digest", "abc"),
        ("output_digest", "Z" * 64),
        ("content_digest", "abc123"),
        ("tool_call_id", ""),
    ]:
        with pytest.raises(ValidationError):
            EvidenceReceipt.model_validate({**payload, field: bad})


def test_method_parameters_are_deeply_immutable() -> None:
    source = {"threshold": 3.5}
    method = ReceiptMethod(family="screen", parameters=source)
    source["threshold"] = 99.0
    assert method.parameters["threshold"] == 3.5, "input aliasing must be severed"
    with pytest.raises(TypeError):
        method.parameters["threshold"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        method.parameters.update({"threshold": 99.0})  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        method.parameters.pop("threshold")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        method.parameters.clear()  # type: ignore[attr-defined]
    # The frozen mapping must still serialize like a plain dict.
    assert method.model_dump(mode="json")["parameters"] == {"threshold": 3.5}


def test_duplicate_derived_fact_ids_are_rejected() -> None:
    facts = (_fact_for("fa", 42, "count"), _fact_for("fb", 84, "count"))
    with pytest.raises(ValidationError, match="duplicate"):
        _receipt_with_derivations(
            facts,
            [
                {"derived_fact_id": "d1", "operator": "ratio", "input_fact_ids": ["fa", "fb"]},
                {"derived_fact_id": "d1", "operator": "difference", "input_fact_ids": ["fa", "fb"]},
            ],
        )


def test_operator_arity_is_enforced() -> None:
    facts = (
        _fact_for("fa", 42, "count"),
        _fact_for("fb", 84, "count"),
        _fact_for("fc", 7, "count"),
    )
    with pytest.raises(ValidationError, match="exactly 2"):
        _receipt_with_derivations(
            facts,
            [{"derived_fact_id": "d1", "operator": "percentage", "input_fact_ids": ["fa"]}],
        )
    with pytest.raises(ValidationError, match="even"):
        _receipt_with_derivations(
            facts,
            [
                {
                    "derived_fact_id": "d1",
                    "operator": "weighted_average",
                    "input_fact_ids": ["fa", "fb", "fc"],
                }
            ],
        )


def test_unit_mismatch_and_division_by_zero_are_rejected() -> None:
    facts = (
        _fact_for("fa", 42, "count", unit="raw"),
        _fact_for("fusd", 10, "number", unit="usd"),
        _fact_for("fzero", 0, "number", unit="raw"),
        _fact_for("ftext", "n/a", "string"),
    )
    with pytest.raises(ValidationError, match="unit"):
        _receipt_with_derivations(
            facts,
            [{"derived_fact_id": "d1", "operator": "difference", "input_fact_ids": ["fa", "fusd"]}],
        )
    with pytest.raises(ValidationError, match="zero"):
        _receipt_with_derivations(
            facts,
            [{"derived_fact_id": "d1", "operator": "ratio", "input_fact_ids": ["fa", "fzero"]}],
        )
    with pytest.raises(ValidationError, match="numeric"):
        _receipt_with_derivations(
            facts,
            [
                {
                    "derived_fact_id": "d1",
                    "operator": "percentage",
                    "input_fact_ids": ["fa", "ftext"],
                }
            ],
        )


def test_fact_manifest_invariants_are_enforced() -> None:
    from eda_platform.schemas.receipts import ReceiptFactManifest

    entry = {
        "fact_id": "pair0",
        "row_index": 0,
        "status": "unevaluated",
        "row_digest": "a" * 64,
    }
    manifest = ReceiptFactManifest(total_rows=2, unlisted_rows=1, entries=(entry,))
    assert manifest.schema_version == 1

    with pytest.raises(ValidationError, match="total_rows"):
        ReceiptFactManifest(total_rows=5, unlisted_rows=0, entries=(entry,))
    with pytest.raises(ValidationError, match="unique"):
        ReceiptFactManifest(total_rows=2, unlisted_rows=0, entries=(entry, entry))
    with pytest.raises(ValidationError):
        ReceiptFactManifest(
            total_rows=1,
            unlisted_rows=0,
            entries=({**entry, "row_digest": "not-hex"},),
        )


def test_an_evaluated_manifest_entry_requires_a_backing_fact() -> None:
    from eda_platform.schemas.receipts import ReceiptFactManifest

    def _with_manifest(status: str, fact_id: str) -> EvidenceReceipt:
        return build_receipt(
            tool_call_id="call_1",
            tool_name="correlate_columns",
            tool_version="1",
            arguments={},
            raw_output={},
            artifact_ids=(),
            result_count=1,
            scope=ReceiptScope(dataset_ids=("ds_c",), columns=("x", "y")),
            facts=(_fact_for("pair0.pearson", 0.5, "number"),),
            method=ReceiptMethod(family="pearson_correlation_screen"),
            data_state_witness="witness_1",
            created_at="2026-08-01T00:00:00Z",
            fact_manifest=ReceiptFactManifest(
                total_rows=1,
                unlisted_rows=0,
                entries=(
                    {
                        "fact_id": fact_id,
                        "row_index": 0,
                        "status": status,
                        "row_digest": "b" * 64,
                    },
                ),
            ),
        )

    receipt = _with_manifest("evaluated", "pair0")
    assert verify_receipt_digest(receipt)
    with pytest.raises(ValidationError, match="backing"):
        _with_manifest("evaluated", "pair_unbacked")


def test_legacy_payloads_without_independence_fields_still_verify() -> None:
    """Receipts persisted before R4 lack the independence fields; adding the
    fields must not invalidate their digests (fail closed would lock out every
    historical artifact, not protect it)."""
    payload = _receipt().model_dump(mode="json")
    legacy = {
        key: value
        for key, value in payload.items()
        if key not in {"evidence_independence_key", "replication_kind"}
    }
    loaded = load_verified_receipt(legacy)
    assert loaded.evidence_independence_key is None
    assert loaded.replication_kind is None


def test_independence_fields_participate_in_the_digest_when_set() -> None:
    receipt = _receipt(
        evidence_independence_key="snapshot:ds_orders@v1",
        replication_kind="holdout",
    )
    assert receipt.replication_kind == "holdout"
    payload = receipt.model_dump(mode="json")
    assert load_verified_receipt(payload) == receipt
    for field, forged in [
        ("replication_kind", "external_replication"),
        ("evidence_independence_key", "snapshot:ds_orders@v2"),
        # Stripping a set field back to None must also break the digest.
        ("replication_kind", None),
        ("evidence_independence_key", None),
    ]:
        with pytest.raises(ReceiptIntegrityError):
            load_verified_receipt({**payload, field: forged})


def test_replication_kind_values_are_the_r4_enum() -> None:
    for kind in ("same_snapshot_corroboration", "holdout", "external_replication"):
        assert _receipt(replication_kind=kind).replication_kind == kind
    with pytest.raises(ValidationError):
        _receipt(replication_kind="reran_same_query")


def _execution(logical_step_id: str, sequence_index: int = 1) -> ReceiptExecution:
    return ReceiptExecution(
        run_id="run_abc",
        provider_call_id="tool_1",
        logical_step_id=logical_step_id,
        sequence_index=sequence_index,
    )


def test_the_receipt_id_covers_the_logical_step_id() -> None:
    """Two positions in one run with the same provider call id, tool, arguments
    and output are different calls; sharing a receipt_id would let one silently
    displace the other in the gate's receipt mapping."""
    first = _receipt(execution=_execution("step_one", 1))
    second = _receipt(execution=_execution("step_five", 5))
    assert first.receipt_id != second.receipt_id
    assert first.content_digest != second.content_digest


def test_replaying_one_logical_step_keeps_the_receipt_id_stable() -> None:
    """Crash replay rebuilds the receipt with a drifted created_at; the outbox's
    exactly-once commit depends on the identity not drifting with it."""
    execution = _execution("step_one", 1)
    first = _receipt(execution=execution, created_at="2026-08-01T00:00:00Z")
    second = _receipt(execution=execution, created_at="2026-08-01T00:09:11Z")
    assert first.receipt_id == second.receipt_id


def test_method_parameters_resist_c_level_dict_mutation() -> None:
    """`|=`, `dict.update` and `dict.__init__` bypass any dict-subclass
    blacklist, so the stored mapping must not be a dict at all."""
    method = ReceiptMethod(family="screen", parameters={"comparison_count": 40})
    params = method.parameters
    for label, mutate in (
        ("|=", lambda: operator.ior(params, {"comparison_count": 1})),
        ("dict.update", lambda: dict.update(params, {"comparison_count": 1})),  # type: ignore[arg-type]
        ("dict.__init__", lambda: dict.__init__(params, {"comparison_count": 1})),  # type: ignore[arg-type]
        ("dict.__setitem__", lambda: dict.__setitem__(params, "injected", 1)),  # type: ignore[arg-type]
    ):
        with pytest.raises(TypeError):
            mutate()
        assert method.parameters["comparison_count"] == 40, f"{label} mutated in place"
    assert "injected" not in method.parameters


def test_frozen_parameters_survive_copy_pickle_and_comparison() -> None:
    method = ReceiptMethod(family="screen", parameters={"threshold": 3.5})
    assert method.parameters == {"threshold": 3.5}
    assert dict(copy.deepcopy(method).parameters) == {"threshold": 3.5}
    assert dict(pickle.loads(pickle.dumps(method)).parameters) == {"threshold": 3.5}
    assert method.model_dump()["parameters"] == {"threshold": 3.5}
    assert method.model_dump(mode="json")["parameters"] == {"threshold": 3.5}


# Frozen on 2026-08-02 from the then-current implementation. Any change to the
# canonical body or its serialization breaks these two, which would invalidate
# every receipt already on disk.
_PRE_R4_PAYLOAD: dict[str, object] = {
    "artifact_ids": ["sql_abc"],
    "content_digest": "82b0ba12b42b870e5bfa6218f79613edb5214ed2c18ba51d55f59753ce76ce2a",
    "created_at": "2026-08-01T00:00:00Z",
    "data_state_witness": "witness_1",
    "derivations": [],
    "execution": None,
    "fact_manifest": None,
    "facts": [
        {
            "fact_id": "f1",
            "name": "n",
            "support_type": "direct",
            "unit": "raw",
            "value": 42,
            "value_type": "count",
        }
    ],
    "input_digest": "348b044aa2361064f4b83be93a77281729dfc01516398ac008c6e2fa19e66dea",
    "method": {
        "assumptions": [],
        "family": "sql_aggregation",
        "parameters": {"threshold": 3.5},
        "warnings": [],
    },
    "output_digest": "d816720db2e7d58f35475e0d5a1bded0625ed46311bf7abaeab08aba219d9ab2",
    "receipt_id": "rcpt_898f91c3532f336893fb3309",
    "result_count": 1,
    "scope": {
        "columns": [],
        "dataset_ids": ["ds_orders"],
        "filters": None,
        "omitted_columns": [],
        "scope_resolution": None,
        "time_range": None,
    },
    "statistics": None,
    "tool_call_id": "run_legacy/call_legacy#0",
    "tool_name": "run_sql",
    "tool_version": "1",
}


def test_a_receipt_persisted_before_r4_still_loads() -> None:
    """Its scope is the legacy unrecorded shape (columns=[], resolution null),
    so tightening scope recording must not lock historical receipts out."""
    loaded = load_verified_receipt(_PRE_R4_PAYLOAD)
    assert loaded.receipt_id == "rcpt_898f91c3532f336893fb3309"
    assert loaded.scope.scope_resolution is None
    assert loaded.scope.columns == ()


def test_the_content_digest_is_byte_stable() -> None:
    body = {key: value for key, value in _PRE_R4_PAYLOAD.items()}
    assert (
        receipt_content_digest(body)
        == "82b0ba12b42b870e5bfa6218f79613edb5214ed2c18ba51d55f59753ce76ce2a"
    )


def test_a_whole_dataset_scan_must_say_so_instead_of_shipping_an_empty_tuple() -> None:
    """`columns=()` with no resolution cannot be told apart from "scope was
    never recorded", which is what a coverage gate has to distinguish."""
    with pytest.raises(ValueError, match="scope_resolution"):
        _receipt(scope=ReceiptScope(dataset_ids=("ds_orders",)))
    receipt = _receipt(
        scope=ReceiptScope(dataset_ids=("ds_orders",), scope_resolution="whole_dataset")
    )
    assert receipt.scope.scope_resolution == "whole_dataset"


def test_a_declared_column_resolution_cannot_be_empty() -> None:
    for resolution in ("explicit", "resolved"):
        with pytest.raises(ValidationError, match="columns"):
            ReceiptScope(dataset_ids=("ds_orders",), scope_resolution=resolution)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="whole_dataset"):
        ReceiptScope(
            dataset_ids=("ds_orders",),
            columns=("amount",),
            scope_resolution="whole_dataset",
        )


def test_the_artifact_type_and_catalog_semantics_exist() -> None:
    """A receipt artifact must have explicit handoff catalog semantics, not the
    unknown-type fallback, so its priority is a decision rather than an accident."""
    from eda_platform.tools.agent_handoff import _catalog_semantics

    assert ArtifactType.EVIDENCE_RECEIPT.value == "EvidenceReceipt"
    stage, role, priority = _catalog_semantics(ArtifactType.EVIDENCE_RECEIPT)
    assert (stage, role, priority) == ("exploration", "evidence", "on_demand")

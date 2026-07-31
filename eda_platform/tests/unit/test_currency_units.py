from __future__ import annotations

from eda_platform.core.currency_units import (
    ISO_4217_LIST_ONE_SHA256,
    ISO_4217_LIST_THREE_SHA256,
    ISO_4217_SNAPSHOT_DATE,
    classify_currency_unit,
    currency_unit_display,
)


def test_current_historical_and_unlisted_currency_codes_are_distinguished() -> None:
    current = classify_currency_unit("BRL")
    historical = classify_currency_unit("BGN")
    custom = classify_currency_unit("BTC")

    assert current.status == "current"
    assert current.reference == "ISO 4217 List One@2026-01-01"
    assert historical.status == "historical"
    assert historical.reference == "ISO 4217 List Three@2026-01-01"
    assert custom.status == "unlisted"
    assert custom.reference == (
        "semantic seed (unlisted in ISO 4217 snapshot@2026-01-01)"
    )


def test_per_order_suffix_preserves_code_classification_and_display() -> None:
    classification = classify_currency_unit("BRL/order")

    assert classification.code == "BRL"
    assert classification.status == "current"
    assert currency_unit_display("BRL/order") == "BRL per order"
    assert currency_unit_display("BGN") == "BGN"


def test_non_code_units_are_not_misclassified_as_currency_codes() -> None:
    assert classify_currency_unit("kg").status == "not_code"
    assert classify_currency_unit("currency").status == "not_code"
    assert classify_currency_unit(None).status == "not_code"
    assert currency_unit_display("kg") is None


def test_snapshot_metadata_is_pinned_for_replay() -> None:
    assert ISO_4217_SNAPSHOT_DATE == "2026-01-01"
    assert len(ISO_4217_LIST_ONE_SHA256) == 64
    assert len(ISO_4217_LIST_THREE_SHA256) == 64

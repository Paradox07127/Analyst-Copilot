"""Versioned ISO 4217 classification for user-confirmed currency unit labels."""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

ISO_4217_SNAPSHOT_DATE = "2026-01-01"
ISO_4217_LIST_ONE_URL = (
    "https://www.six-group.com/dam/download/financial-information/"
    "data-center/iso-currrency/lists/list-one.xml"
)
ISO_4217_LIST_THREE_URL = (
    "https://www.six-group.com/dam/download/financial-information/"
    "data-center/iso-currrency/lists/list-three.xml"
)
ISO_4217_LIST_ONE_SHA256 = "838dfb991648cf36df939edd5fe3811737962b75a32252847d239cedd1e291c9"
ISO_4217_LIST_THREE_SHA256 = "98fde2423cdb916dd59dcf5fe96222edad8fa198d865c1c83dbc464b9cc52387"

# Generated mechanically from the two official XML snapshots above. Keep the
# source metadata and tests in lockstep when refreshing this data.
_CURRENT_CODES = frozenset(
    """
    AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND BOB BOV
    BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP
    CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF
    GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR
    KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT
    MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN
    PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE
    SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX
    USD USN UYI UYU UYW UZS VED VES VND VUV WST XAD XAF XAG XAU XBA XBB XBC
    XBD XCD XCG XDR XOF XPD XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG
    """.split()
)
_HISTORICAL_CODES = frozenset(
    """
    ADP AFA ALK ANG AOK AON AOR ARA ARP ARY ATS AYM AZM BAD BEC BEF BEL BGJ
    BGK BGL BGN BOP BRB BRC BRE BRN BRR BUK BYB BYR CHC CSD CSJ CSK CUC CYP
    DDM DEM ECS ECV EEK ESA ESB ESP EUR FIM FRF GEK GHC GHP GNE GNS GQE GRD
    GWE GWP HRD HRK IDR IEP ILP ILR ISJ ITL LAJ LSM LTL LTT LUC LUF LUL LVL
    LVR MGF MLF MRO MTL MTP MVQ MWK MXP MZE MZM NIC NLG PEH PEI PEN PES PLZ
    PTE RHD ROK ROL RON RUR SDD SDG SDP SIT SKK SLL SRG STD SUR SZL TJR TMM
    TPE TRL TRY UAK UGS UGW USS UYN UYP VEB VEF VNC XEU XFO XFU XRE YDD YUD
    YUM YUN ZAL ZMK ZRN ZRZ ZWC ZWD ZWL ZWN ZWR
    """.split()
)
_CURRENCY_UNIT_RE = re.compile(r"^(?P<code>[A-Z]{3})(?P<per_order>/order)?$")

CurrencyCodeStatus = Literal["current", "historical", "unlisted", "not_code"]


class CurrencyUnitClassification(NamedTuple):
    code: str | None
    status: CurrencyCodeStatus
    reference: str | None


def classify_currency_unit(unit: str | None) -> CurrencyUnitClassification:
    """Classify an exact unit label against the pinned SIX snapshot."""
    match = _CURRENCY_UNIT_RE.fullmatch(unit or "")
    if match is None:
        return CurrencyUnitClassification(None, "not_code", None)
    code = match.group("code")
    if code in _CURRENT_CODES:
        return CurrencyUnitClassification(
            code,
            "current",
            f"ISO 4217 List One@{ISO_4217_SNAPSHOT_DATE}",
        )
    if code in _HISTORICAL_CODES:
        return CurrencyUnitClassification(
            code,
            "historical",
            f"ISO 4217 List Three@{ISO_4217_SNAPSHOT_DATE}",
        )
    return CurrencyUnitClassification(
        code,
        "unlisted",
        f"semantic seed (unlisted in ISO 4217 snapshot@{ISO_4217_SNAPSHOT_DATE})",
    )


def currency_unit_display(unit: str | None) -> str | None:
    """Render a compact deterministic label for a specific currency unit."""
    classification = classify_currency_unit(unit)
    if classification.code is None:
        return None
    if unit is not None and unit.endswith("/order"):
        return f"{classification.code} per order"
    return classification.code

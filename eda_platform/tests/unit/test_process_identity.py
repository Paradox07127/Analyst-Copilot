from __future__ import annotations

import os
import sys

from eda_platform.core import process_identity as identities
from eda_platform.core.process_identity import (
    ProcessIdentity,
    ProcessIdentityComparison,
    check_process_identity,
    compare_process_identity,
    process_identity_matches,
    read_process_identity,
)


def test_current_process_has_authoritative_identity_on_supported_platform() -> None:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        return
    identity = read_process_identity(os.getpid())
    assert identity is not None
    assert check_process_identity(identity) is ProcessIdentityComparison.MATCH
    assert process_identity_matches(identity)


def test_pid_reuse_start_mismatch_fails_closed(
    monkeypatch,
) -> None:
    expected = ProcessIdentity(42, "boot-a:100", "linux-procfs")
    reused = ProcessIdentity(42, "boot-a:900", "linux-procfs")
    monkeypatch.setattr(identities, "read_process_identity", lambda _pid: reused)

    assert compare_process_identity(expected, reused) is ProcessIdentityComparison.MISMATCH
    assert check_process_identity(expected) is ProcessIdentityComparison.MISMATCH
    assert not process_identity_matches(expected)


def test_unknown_or_cross_source_identity_never_matches(monkeypatch) -> None:
    expected = ProcessIdentity(42, "boot-a:100", "linux-procfs")
    darwin = ProcessIdentity(42, "100:1", "darwin-libproc")
    monkeypatch.setattr(identities, "read_process_identity", lambda _pid: None)

    assert compare_process_identity(expected, None) is ProcessIdentityComparison.UNKNOWN
    assert compare_process_identity(expected, darwin) is ProcessIdentityComparison.UNKNOWN
    assert check_process_identity(expected) is ProcessIdentityComparison.UNKNOWN
    assert not process_identity_matches(expected)


def test_linux_stat_parser_handles_parentheses_in_process_name() -> None:
    suffix = ["S", *["0"] * 18, "12345", "0"]
    stat = f"77 (worker ) name) {' '.join(suffix)}"
    assert identities._linux_start_ticks(stat) == 12345
    assert identities._linux_start_ticks("malformed") is None


def test_invalid_or_unsupported_pid_identity_is_unknown(monkeypatch) -> None:
    assert read_process_identity(0) is None
    assert read_process_identity(True) is None
    monkeypatch.setattr(identities.sys, "platform", "unsupported")
    assert read_process_identity(1) is None

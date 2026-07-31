"""Fail-closed process identity checks for PID-reuse-safe signalling.

A PID alone is not an identity: after a process exits the operating system may
assign the same number to an unrelated process.  Callers that may signal a
worker should persist :class:`ProcessIdentity` when the worker starts, then
re-read and compare it immediately before signalling.

Linux uses ``/proc/<pid>/stat`` start ticks plus the kernel boot id.  macOS uses
``proc_pidinfo(PROC_PIDTBSDINFO)`` and its microsecond-resolution start time.
Windows uses ``GetProcessTimes`` on a query-only handle, whose creation time is
100-nanosecond resolution.  There is intentionally no low-resolution ``ps``
fallback: inability to obtain an authoritative identity is ``UNKNOWN`` and must
fail closed.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

ProcessIdentitySource = Literal["linux-procfs", "darwin-libproc", "windows-kernel32"]


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Stable process birth identity captured for one PID."""

    pid: int
    start_token: str
    source: ProcessIdentitySource

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not self.start_token:
            raise ValueError("start_token must not be empty")


class ProcessIdentityComparison(StrEnum):
    """Result of comparing an expected identity with a fresh observation."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


def read_process_identity(pid: int) -> ProcessIdentity | None:
    """Read an authoritative start identity for *pid*, or ``None``.

    ``None`` covers a missing process, insufficient permissions, malformed
    operating-system data, and unsupported platforms.  Callers must not treat
    it as evidence that the PID is safe to signal.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        return _read_linux_process_identity(pid)
    if sys.platform == "darwin":
        return _read_darwin_process_identity(pid)
    if sys.platform == "win32":
        return _read_windows_process_identity(pid)
    return None


def compare_process_identity(
    expected: ProcessIdentity,
    observed: ProcessIdentity | None,
) -> ProcessIdentityComparison:
    """Compare two observations without converting uncertainty into a match."""

    if observed is None or observed.source != expected.source:
        return ProcessIdentityComparison.UNKNOWN
    if observed.pid != expected.pid or observed.start_token != expected.start_token:
        return ProcessIdentityComparison.MISMATCH
    return ProcessIdentityComparison.MATCH


def check_process_identity(expected: ProcessIdentity) -> ProcessIdentityComparison:
    """Re-read *expected.pid* and compare it with the captured identity."""

    return compare_process_identity(expected, read_process_identity(expected.pid))


def process_identity_matches(expected: ProcessIdentity) -> bool:
    """Return true only for an authoritative match; unknown fails closed."""

    return check_process_identity(expected) is ProcessIdentityComparison.MATCH


def _read_linux_process_identity(pid: int) -> ProcessIdentity | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    start_ticks = _linux_start_ticks(stat)
    if start_ticks is None or not boot_id or any(char.isspace() for char in boot_id):
        return None
    return ProcessIdentity(
        pid=pid,
        start_token=f"{boot_id}:{start_ticks}",
        source="linux-procfs",
    )


def _linux_start_ticks(stat: str) -> int | None:
    """Extract field 22 while allowing spaces and ``)`` in the comm field."""

    comm_end = stat.rfind(")")
    if comm_end < 2:
        return None
    # The suffix begins at field 3 (state), so field 22 has index 19.
    fields = stat[comm_end + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        value = int(fields[19])
    except ValueError:
        return None
    return value if value >= 0 else None


class _ProcBsdInfo(ctypes.Structure):
    """The stable prefix/full layout of Darwin's ``struct proc_bsdinfo``."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _read_darwin_process_identity(pid: int) -> ProcessIdentity | None:
    # PROC_PIDTBSDINFO from <libproc.h>.  Loading by absolute path avoids
    # environment-dependent library resolution.
    proc_pidtbsdinfo = 3
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = _ProcBsdInfo()
        size = ctypes.sizeof(info)
        read_size = proc_pidinfo(
            pid,
            proc_pidtbsdinfo,
            0,
            ctypes.byref(info),
            size,
        )
    except (AttributeError, OSError):
        return None
    if read_size != size or int(info.pbi_pid) != pid or info.pbi_start_tvsec == 0:
        return None
    return ProcessIdentity(
        pid=pid,
        start_token=f"{int(info.pbi_start_tvsec)}:{int(info.pbi_start_tvusec)}",
        source="darwin-libproc",
    )


def _read_windows_process_identity(pid: int) -> ProcessIdentity | None:
    """Read the creation time of *pid* through a query-only process handle.

    ``PROCESS_QUERY_LIMITED_INFORMATION`` is deliberately narrower than the
    access ``os.kill`` would request: this call must never be able to affect the
    process it is identifying.
    """
    process_query_limited_information = 0x1000
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        creation = ctypes.c_uint64()
        exited = ctypes.c_uint64()
        kernel_time = ctypes.c_uint64()
        user_time = ctypes.c_uint64()
        ok = kernel32.GetProcessTimes(
            ctypes.c_void_p(handle),
            ctypes.byref(creation),
            ctypes.byref(exited),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    if not ok or creation.value == 0:
        return None
    return ProcessIdentity(
        pid=pid,
        start_token=str(creation.value),
        source="windows-kernel32",
    )

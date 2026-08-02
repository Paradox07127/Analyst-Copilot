"""Small, dependency-free process resource measurements."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

PeakRssMethod = Literal[
    "getrusage_ru_maxrss",
    "get_process_memory_info_peak_working_set",
    "unavailable",
]


@dataclass(frozen=True, slots=True)
class PeakRssMeasurement:
    bytes: int | None
    method: PeakRssMethod


def normalize_posix_maxrss(raw_maxrss: int, *, platform_name: str | None = None) -> int:
    """Normalize ``ru_maxrss`` to bytes (macOS reports bytes; Linux reports KiB)."""
    raw = max(0, int(raw_maxrss))
    platform = platform_name or sys.platform
    return raw if platform == "darwin" else raw * 1024


def process_peak_rss(*, platform_name: str | None = None) -> PeakRssMeasurement:
    """Return the process-lifetime peak resident set, or a typed unavailable value."""
    platform = platform_name or sys.platform
    if platform == "win32":
        value = _windows_peak_working_set_bytes()
        return PeakRssMeasurement(
            bytes=value,
            method=(
                "get_process_memory_info_peak_working_set"
                if value is not None
                else "unavailable"
            ),
        )
    raw = _posix_peak_maxrss()
    if raw is None:
        return PeakRssMeasurement(bytes=None, method="unavailable")
    return PeakRssMeasurement(
        bytes=normalize_posix_maxrss(raw, platform_name=platform),
        method="getrusage_ru_maxrss",
    )


def _posix_peak_maxrss() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None


def _windows_peak_working_set_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None

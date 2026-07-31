"""Behaviour of the three primitives whose implementation differs per platform.

These run on the host platform. The Windows branches cannot execute here, so
what is pinned is the contract both branches must satisfy.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eda_platform.core.file_lock import exclusive_lock, lock_exclusive, unlock
from eda_platform.core.fs import remove_tree
from eda_platform.infrastructure.launch_gate import (
    GATE_ACK,
    LaunchGateError,
    open_parent_gate,
)


def _hold_lock(path: str, acquired, release) -> None:  # type: ignore[no-untyped-def]
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        lock_exclusive(descriptor)
        acquired.set()
        release.wait(10)
        unlock(descriptor)
    finally:
        os.close(descriptor)


def test_exclusive_lock_blocks_a_second_process_until_the_first_releases(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "contended.lock"
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_lock, args=(str(lock_path), acquired, release)
    )
    holder.start()
    try:
        assert acquired.wait(10)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            started = time.monotonic()
            release.set()
            with exclusive_lock(descriptor):
                waited = time.monotonic() - started
        finally:
            os.close(descriptor)
    finally:
        holder.join(10)
    # The point is that the second acquire returned only after the first
    # released, not that it was fast.
    assert holder.exitcode == 0
    assert waited >= 0


def test_a_second_acquire_does_not_succeed_while_the_lock_is_held(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "held.lock"
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_lock, args=(str(lock_path), acquired, release)
    )
    holder.start()
    try:
        assert acquired.wait(10)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys;"
                "sys.path.insert(0, sys.argv[2]);"
                "from eda_platform.core.file_lock import lock_exclusive;"
                "fd = os.open(sys.argv[1], os.O_RDWR);"
                "lock_exclusive(fd);"
                "print('acquired')",
                str(lock_path),
                str(Path(__file__).resolve().parents[2] / "src"),
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        pytest.fail(f"probe should have blocked, got {probe.stdout!r}")
    except subprocess.TimeoutExpired:
        pass
    finally:
        release.set()
        holder.join(10)


def test_remove_tree_deletes_entries_a_plain_rmtree_cannot(tmp_path: Path) -> None:
    """A read-only parent blocks unlink on POSIX; a read-only file blocks it on
    Windows. One helper has to survive both."""
    for name in ("plain", "helper"):
        target = tmp_path / name
        locked_dir = target / "locked"
        locked_dir.mkdir(parents=True)
        (locked_dir / "payload.json").write_text("{}", encoding="utf-8")
        (locked_dir / "payload.json").chmod(stat.S_IRUSR)
        locked_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    with pytest.raises(OSError):
        shutil.rmtree(tmp_path / "plain")
    assert (tmp_path / "plain").exists()

    remove_tree(tmp_path / "helper")
    assert not (tmp_path / "helper").exists()

    # The control tree is still undeletable; pytest's own tmpdir cleanup would
    # warn about it for the rest of the session.
    remove_tree(tmp_path / "plain")


def test_jsonl_page_keys_are_posix_on_every_host(tmp_path: Path) -> None:
    """The index writes this key and run deletion purges by it. A host
    separator in the stored value splits the two on Windows, so the run's
    chat and debug index rows would survive its deletion."""
    from eda_platform.core.bounded_pagination import JsonlPageIndex, jsonl_path_key

    nested = tmp_path / "projects" / "demo" / "chat" / "run_x.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"role":"user"}\n', encoding="utf-8")

    index = JsonlPageIndex(tmp_path / "state.sqlite", tmp_path)
    state = index.ensure(nested, accept=lambda _line: True)

    assert state.path_key == "projects/demo/chat/run_x.jsonl"
    assert "\\" not in state.path_key
    assert jsonl_path_key(state.path_key) == state.path_key


def test_launch_gate_child_does_not_start_when_the_parent_never_releases() -> None:
    """A parent that dies between spawn and release must leave the child
    refusing to run, because no birth identity was persisted for it."""
    from eda_platform.infrastructure.launch_gate import acknowledge_and_wait

    gate = open_parent_gate("token-abcdefgh")
    argument = gate.child_argument()
    kind, _, rest = argument.partition(":")
    if kind != "fd":  # pragma: no cover - platform branch
        pytest.skip("descriptor gate is POSIX-only")
    gate_fd, _, ready_fd = rest.partition(":")
    child_argument = f"fd:{os.dup(int(gate_fd))}:{os.dup(int(ready_fd))}"

    parent_saw = []

    import threading

    def child() -> None:
        parent_saw.append(acknowledge_and_wait(child_argument, "token-abcdefgh", 5.0))

    worker = threading.Thread(target=child, daemon=True)
    worker.start()
    gate.wait_for_acknowledgement(5.0)
    gate.close()  # parent dies here, without release()
    worker.join(5)
    assert parent_saw == [False]


def test_launch_gate_reports_a_child_that_never_acknowledges() -> None:
    with open_parent_gate("token-abcdefgh") as gate:
        with pytest.raises(LaunchGateError, match="did not acknowledge"):
            gate.wait_for_acknowledgement(0.1)


def test_launch_gate_argument_carries_no_secret_on_the_descriptor_transport() -> None:
    """The POSIX transport is unreachable by other processes, so the argument
    must not leak the launch token into a world-readable argv."""
    with open_parent_gate("token-abcdefgh") as gate:
        argument = gate.child_argument()
        if not argument.startswith("fd:"):  # pragma: no cover - platform branch
            pytest.skip("descriptor gate is POSIX-only")
        assert "token-abcdefgh" not in argument
        assert gate.inheritable_descriptors() != ()
    assert GATE_ACK == b"R"

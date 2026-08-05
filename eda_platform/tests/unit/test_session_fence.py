from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import eda_platform.core.session_fence as session_fence
from eda_platform.core.session_fence import SessionFencePathError, session_key_lock


def _process_contender(workspace: str, start: Any) -> None:
    start.wait(timeout=10)
    counter = Path(workspace) / "counter.txt"
    with session_key_lock(workspace, "same_run"):
        current = int(counter.read_text(encoding="utf-8"))
        time.sleep(0.003)
        counter.write_text(str(current + 1), encoding="utf-8")


def test_first_creation_is_serialized_across_threads(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    barrier = threading.Barrier(24)

    def contender() -> None:
        barrier.wait(timeout=5)
        with session_key_lock(tmp_path, "same_run"):
            current = int(counter.read_text(encoding="utf-8"))
            time.sleep(0.001)
            counter.write_text(str(current + 1), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(contender) for _ in range(24)]
        for future in futures:
            future.result(timeout=10)

    assert counter.read_text(encoding="utf-8") == "24"


def test_first_creation_is_serialized_across_processes(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_process_contender, args=(str(tmp_path), start))
        for _ in range(8)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    assert counter.read_text(encoding="utf-8") == "8"


def test_legacy_windows_crlf_identity_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()
    info = workspace.stat()
    identity_path = workspace.parent / f".eda-workspace-{digest}.identity"
    identity_path.write_bytes(f"{info.st_dev}:{info.st_ino}\r\n".encode())
    monkeypatch.setattr(session_fence, "SUPPORTS_DIR_FD", False)

    with session_key_lock(workspace, "same_run"):
        pass


def test_recreated_locks_directory_is_rejected_while_old_lock_is_held(
    tmp_path: Path,
) -> None:
    with session_key_lock(tmp_path, "same_run"):
        locks = tmp_path / ".storage-operations" / "locks"
        old_locks = locks.with_name("locks.old")
        locks.rename(old_locks)
        locks.mkdir(mode=0o700)

        with pytest.raises(SessionFencePathError, match="identity changed"):
            with session_key_lock(tmp_path, "same_run"):
                pytest.fail("replacement directory must never produce a second lock")


def test_symlinked_locks_directory_is_rejected(tmp_path: Path) -> None:
    control = tmp_path / ".storage-operations"
    control.mkdir()
    target = tmp_path / "attacker-locks"
    target.mkdir()
    (control / "locks").symlink_to(target, target_is_directory=True)

    with pytest.raises(SessionFencePathError):
        with session_key_lock(tmp_path, "same_run"):
            pytest.fail("symlinked locks directory must be rejected")


def test_replaced_lock_file_is_rejected(tmp_path: Path) -> None:
    with session_key_lock(tmp_path, "same_run"):
        pass
    digest = hashlib.sha256(b"same_run").hexdigest()
    lock_path = (
        tmp_path / ".storage-operations" / "locks" / f"run-{digest}.lock"
    )
    lock_path.unlink()
    target = tmp_path / "attacker-file"
    target.write_text("", encoding="utf-8")
    os.symlink(target, lock_path)

    with pytest.raises(SessionFencePathError):
        with session_key_lock(tmp_path, "same_run"):
            pytest.fail("symlinked lock file must be rejected")


def test_control_directory_replacement_is_detected_before_yield(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    error: list[BaseException] = []

    def contender() -> None:
        try:
            with session_key_lock(tmp_path, "same_run"):
                entered.set()
                release.wait(timeout=5)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=contender)
    thread.start()
    assert entered.wait(timeout=5)
    control = tmp_path / ".storage-operations"
    shutil.move(control, tmp_path / ".storage-operations.old")
    control.mkdir()
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    with pytest.raises(SessionFencePathError):
        with session_key_lock(tmp_path, "same_run"):
            pytest.fail("replacement control root must be rejected")
    assert error == []


def test_workspace_root_replacement_cannot_create_a_second_lock(
    tmp_path: Path,
) -> None:
    original = tmp_path.with_name(f"{tmp_path.name}.old")
    try:
        with session_key_lock(tmp_path, "same_run"):
            tmp_path.rename(original)
            tmp_path.mkdir()
            with pytest.raises(SessionFencePathError, match="workspace root identity changed"):
                with session_key_lock(tmp_path, "same_run"):
                    pytest.fail("replacement workspace must not create a split lock")
    finally:
        if tmp_path.is_dir():
            tmp_path.rmdir()
        if original.exists():
            original.rename(tmp_path)


def test_a_remounted_volume_keeps_its_workspace(tmp_path: Path) -> None:
    """`st_dev` is a mount-instance number, not part of a directory's identity.

    Live failure 2026-08-05: creating a job raised "Session fence workspace root
    identity changed." with `16777232:98226571` recorded and `16777234:98226571`
    on disk -- the same inode, a device number macOS reassigned when the volume
    was remounted. The machine had already hit this once; a
    `.storage-operations.identity.pre-migration-` backup from an earlier mount
    holds the same inode under a third device number.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()
    identity_path = workspace.parent / f".eda-workspace-{digest}.identity"
    info = workspace.stat()
    identity_path.write_bytes(f"{info.st_dev + 2}:{info.st_ino}\n".encode())

    with session_key_lock(workspace, "same_run"):
        pass

    # Rewritten, so the next open compares against this mount, not the old one.
    assert identity_path.read_bytes() == f"{info.st_dev}:{info.st_ino}\n".encode()


def test_a_replaced_workspace_is_still_rejected(tmp_path: Path) -> None:
    """Tolerating the device number must not tolerate a different directory.

    A swap changes the inode, which is the half that identifies the object.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()
    identity_path = workspace.parent / f".eda-workspace-{digest}.identity"
    info = workspace.stat()
    identity_path.write_bytes(f"{info.st_dev}:{info.st_ino + 1}\n".encode())

    with pytest.raises(SessionFencePathError, match="workspace root identity changed"):
        with session_key_lock(workspace, "same_run"):
            pytest.fail("a different directory must never pass the fence")

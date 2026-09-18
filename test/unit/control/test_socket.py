from __future__ import annotations

from datetime import UTC, datetime
import errno
import json
import multiprocessing
import os
from pathlib import Path
import queue
import socket
import stat

import pytest

import claude_code_proxy.control.socket as socket_module
from claude_code_proxy.control.socket import SocketLease, resolve_socket_path


STARTED_AT = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)


def _private_path(tmp_path: Path) -> Path:
    directory = tmp_path / "runtime"
    directory.mkdir(mode=0o700)
    return directory / "control.sock"


def _claim_socket(
    path: str,
    start: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    if not start.wait(10):
        results.put(("unexpected", "TimeoutError", "start signal timed out"))
        return
    try:
        lease = SocketLease.acquire(Path(path))
    except RuntimeError as error:
        results.put(("error", type(error).__name__, str(error)))
        return
    except Exception as error:  # noqa: BLE001 - child reports unexpected outcome
        results.put(("unexpected", type(error).__name__, str(error)))
        return
    results.put(("acquired", "SocketLease", os.getpid()))
    if not release.wait(30):
        results.put(("unexpected", "TimeoutError", "release signal timed out"))
    lease.close()


def _with_uid(value: os.stat_result, uid: int) -> os.stat_result:
    fields = list(value)
    fields[4] = uid
    return os.stat_result(fields)


def _with_permissions(value: os.stat_result, mode: int) -> os.stat_result:
    fields = list(value)
    fields[0] = stat.S_IFMT(value.st_mode) | mode
    return os.stat_result(fields)


def _prior_metadata(
    path: Path,
    *,
    pid: int,
    boot_id: str,
    start_ticks: int,
    socket_dev: int | None = None,
    socket_ino: int | None = None,
) -> dict[str, object]:
    identity = path.lstat()
    return {
        "boot_id": boot_id,
        "pid": pid,
        "pid_start_ticks": start_ticks,
        "socket_dev": identity.st_dev if socket_dev is None else socket_dev,
        "socket_ino": identity.st_ino if socket_ino is None else socket_ino,
        "started_at": "2026-04-05T06:07:08Z",
    }


def _write_prior_metadata(path: Path, metadata: dict[str, object]) -> str:
    payload = json.dumps(metadata, separators=(",", ":")) + "\n"
    path.with_name("control.lock").write_text(payload)
    return payload


def _set_process_identity_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    boot_id: str = "boot-current",
    pid_ticks: dict[int, int | None] | None = None,
) -> None:
    ticks = {} if pid_ticks is None else pid_ticks
    monkeypatch.setattr(
        socket_module,
        "_read_boot_id",
        lambda: boot_id,
        raising=False,
    )
    monkeypatch.setattr(
        socket_module,
        "_read_pid_start_ticks",
        lambda pid: ticks.get(pid, 9001),
        raising=False,
    )


def _connect_and_accept(path: Path, listener: socket.socket) -> None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(path))
        accepted, _ = listener.accept()
        accepted.close()
    finally:
        client.close()


def _replace_leaf_during_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
):
    real_rename = socket_module._rename_leaf

    def replace_before_rename(dir_fd: int, source: str, target: str) -> None:
        os.unlink(source, dir_fd=dir_fd)
        replacement_fd = os.open(
            source,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
            dir_fd=dir_fd,
        )
        os.write(replacement_fd, payload)
        os.close(replacement_fd)
        real_rename(dir_fd, source, target)

    monkeypatch.setattr(socket_module, "_rename_leaf", replace_before_rename)
    return real_rename


def test_acquire_rejects_non_linux_before_filesystem_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "not-created" / "control.sock"
    monkeypatch.setattr(socket_module.sys, "platform", "darwin")

    with pytest.raises(
        RuntimeError,
        match=(
            "secure control socket requires Linux; "
            "use Docker on other platforms"
        ),
    ):
        SocketLease.acquire(path)

    assert not path.parent.exists()
    assert not path.exists()
    assert not path.with_name("control.lock").exists()


def test_resolve_socket_path_prefers_explicit_and_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_socket_path(
        Path("~/chosen/control.sock"),
        {"CONTROL_SOCKET_PATH": "   ", "XDG_RUNTIME_DIR": "/ignored"},
    )

    assert resolved == tmp_path / "chosen" / "control.sock"
    assert resolved.is_absolute()


def test_resolve_socket_path_uses_stripped_environment_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = resolve_socket_path(
        None,
        {"CONTROL_SOCKET_PATH": "  relative/control.sock  "},
    )

    assert resolved == tmp_path / "relative" / "control.sock"
    assert resolved.is_absolute()


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_socket_path_rejects_blank_environment_value(value: str) -> None:
    with pytest.raises(ValueError, match="CONTROL_SOCKET_PATH must not be empty"):
        resolve_socket_path(None, {"CONTROL_SOCKET_PATH": value})


def test_resolve_socket_path_prefers_safe_xdg_directory(
    tmp_path: Path,
) -> None:
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)

    resolved = resolve_socket_path(None, {"XDG_RUNTIME_DIR": str(xdg)})

    assert resolved == xdg / "claude-code-proxy" / "control.sock"


def test_resolve_socket_path_skips_relative_xdg_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relative = tmp_path / "relative-xdg"
    relative.mkdir(mode=0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", Path("/missing"))
    monkeypatch.setattr(socket_module.tempfile, "gettempdir", lambda: "/t")

    resolved = resolve_socket_path(None, {"XDG_RUNTIME_DIR": "relative-xdg"})

    assert resolved == Path(f"/t/claude-code-proxy-{os.getuid()}/control.sock")


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_resolve_socket_path_skips_insecure_xdg_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: int
) -> None:
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=mode)
    xdg.chmod(mode)
    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", Path("/missing"))
    monkeypatch.setattr(socket_module.tempfile, "gettempdir", lambda: "/t")

    resolved = resolve_socket_path(None, {"XDG_RUNTIME_DIR": str(xdg)})

    assert resolved == Path(f"/t/claude-code-proxy-{os.getuid()}/control.sock")


def test_resolve_socket_path_skips_insecure_run_user_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uid = 4242
    run_root = Path("/r")
    run_base = run_root / str(uid)
    insecure = _with_uid(_with_permissions(tmp_path.stat(), 0o755), uid)

    def simulated_lstat(candidate: os.PathLike[str] | str) -> os.stat_result:
        if Path(candidate) == run_base:
            return insecure
        raise FileNotFoundError(candidate)

    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", run_root)
    monkeypatch.setattr(socket_module.os, "lstat", simulated_lstat)
    monkeypatch.setattr(socket_module.tempfile, "gettempdir", lambda: "/t")

    resolved = resolve_socket_path(None, {}, uid=uid)

    assert resolved == Path("/t/claude-code-proxy-4242/control.sock")


def test_resolve_socket_path_skips_xdg_directory_with_wrong_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xdg = Path("/x")
    run_root = Path("/r")
    run_base = run_root / str(os.getuid())
    directory_stat = tmp_path.stat()

    def simulated_owners(path: os.PathLike[str] | str) -> os.stat_result:
        owner = os.getuid() + 1 if Path(path) == xdg else os.getuid()
        return _with_uid(directory_stat, owner)

    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", run_root)
    monkeypatch.setattr(socket_module.os, "lstat", simulated_owners)

    resolved = resolve_socket_path(None, {"XDG_RUNTIME_DIR": str(xdg)})

    assert resolved == run_base / "claude-code-proxy" / "control.sock"


def test_resolve_socket_path_uses_run_user_before_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uid = 4242
    run_root = Path("/r")
    run_base = run_root / str(uid)
    directory_stat = tmp_path.stat()

    def expected_owner(path: os.PathLike[str] | str) -> os.stat_result:
        return _with_uid(directory_stat, uid)

    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", run_root)
    monkeypatch.setattr(socket_module.os, "lstat", expected_owner)

    resolved = resolve_socket_path(None, {}, uid=uid)

    assert resolved == run_base / "claude-code-proxy" / "control.sock"


def test_resolve_socket_path_falls_back_to_private_temp_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uid = 4242
    monkeypatch.setattr(socket_module, "_RUN_USER_ROOT", tmp_path / "missing")
    monkeypatch.setattr(socket_module.tempfile, "gettempdir", lambda: str(tmp_path))

    resolved = resolve_socket_path(None, {}, uid=uid)

    assert resolved == tmp_path / f"claude-code-proxy-{uid}" / "control.sock"
    assert resolved.parent != tmp_path


@pytest.mark.parametrize(
    "value",
    [Path("bad\0name.sock"), Path("/" + "x" * 108)],
)
def test_resolve_socket_path_rejects_invalid_unix_socket_paths(value: Path) -> None:
    with pytest.raises(ValueError, match="socket path"):
        resolve_socket_path(value)


def test_acquire_creates_private_application_directory(tmp_path: Path) -> None:
    path = tmp_path / "new-runtime" / "control.sock"

    lease = SocketLease.acquire(path)
    try:
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    finally:
        lease.close()


def test_created_application_directory_is_0700_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    path = tmp_path / "new-runtime" / "control.sock"
    previous_umask = os.umask(0o777)
    try:
        lease = SocketLease.acquire(path)
    finally:
        os.umask(previous_umask)

    try:
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    finally:
        lease.close()


def test_acquire_accepts_existing_private_application_directory(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)

    with SocketLease.acquire(path) as lease:
        assert lease.path == path


@pytest.mark.parametrize("mode", [0o750, 0o707])
def test_acquire_rejects_insecure_application_directory_mode(
    tmp_path: Path, mode: int
) -> None:
    directory = tmp_path / "runtime"
    directory.mkdir(mode=mode)
    directory.chmod(mode)

    with pytest.raises(RuntimeError, match="permissions"):
        SocketLease.acquire(directory / "control.sock")


def test_acquire_rejects_symlink_application_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    directory = tmp_path / "runtime"
    directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        SocketLease.acquire(directory / "control.sock")


def test_acquire_rejects_non_directory_application_path(tmp_path: Path) -> None:
    directory = tmp_path / "runtime"
    directory.write_text("not a directory")

    with pytest.raises(RuntimeError, match="directory"):
        SocketLease.acquire(directory / "control.sock")


def test_acquire_rejects_wrong_owner_application_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_lstat = socket_module.os.lstat

    def wrong_owner(candidate: os.PathLike[str] | str) -> os.stat_result:
        result = real_lstat(candidate)
        if Path(candidate) == path.parent:
            return _with_uid(result, os.getuid() + 1)
        return result

    monkeypatch.setattr(socket_module.os, "lstat", wrong_owner)

    with pytest.raises(RuntimeError, match="owned"):
        SocketLease.acquire(path)


def test_acquire_verifies_directory_after_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "runtime" / "control.sock"
    real_mkdir = socket_module.os.mkdir

    def create_insecure(candidate: os.PathLike[str] | str, mode: int) -> None:
        real_mkdir(candidate, 0o755)
        os.chmod(candidate, 0o755)

    monkeypatch.setattr(socket_module.os, "mkdir", create_insecure)

    with pytest.raises(RuntimeError, match="permissions"):
        SocketLease.acquire(path)


def test_lease_binds_private_nonblocking_listener_and_writes_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    _set_process_identity_seams(monkeypatch, pid_ticks={2468: 777})

    with SocketLease.acquire(path, pid=2468, started_at=STARTED_AT) as lease:
        assert lease.path == path
        assert lease.socket.family == socket.AF_UNIX
        assert lease.socket.type & socket.SOCK_STREAM
        assert lease.socket.getblocking() is False
        bound = path.lstat()
        assert stat.S_ISSOCK(bound.st_mode)
        assert stat.S_IMODE(bound.st_mode) == 0o600

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(str(path))
        finally:
            client.close()

    assert not path.exists()
    lock_path = path.with_name("control.lock")
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert json.loads(lock_path.read_text()) == {
        "boot_id": "boot-current",
        "pid": 2468,
        "pid_start_ticks": 777,
        "socket_dev": bound.st_dev,
        "socket_ino": bound.st_ino,
        "started_at": "2026-04-05T06:07:08Z",
    }


@pytest.mark.parametrize("pid", [0, -1, True, "123"])
def test_acquire_rejects_invalid_pid_before_filesystem_changes(
    tmp_path: Path, pid: object
) -> None:
    path = tmp_path / "new-runtime" / "control.sock"

    with pytest.raises(ValueError, match="pid must be a positive integer"):
        SocketLease.acquire(path, pid=pid)  # type: ignore[arg-type]

    assert not path.parent.exists()


def test_acquire_validates_default_pid_before_filesystem_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "new-runtime" / "control.sock"
    monkeypatch.setattr(socket_module.os, "getpid", lambda: 0)

    with pytest.raises(ValueError, match="pid must be a positive integer"):
        SocketLease.acquire(path)

    assert not path.parent.exists()


def test_existing_same_owner_lock_file_is_forced_to_mode_0600(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    lock_path = path.with_name("control.lock")
    lock_path.write_text("old")
    lock_path.chmod(0o644)

    with SocketLease.acquire(path):
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_acquire_rejects_wrong_owner_lock_at_fstat_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_fstat = socket_module.os.fstat
    calls = 0

    def wrong_owner(lock_fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        result = real_fstat(lock_fd)
        if calls == 2:
            return _with_uid(result, os.getuid() + 1)
        return result

    monkeypatch.setattr(socket_module.os, "fstat", wrong_owner)
    with pytest.raises(RuntimeError, match="lock is not owned"):
        SocketLease.acquire(path)

    assert not path.exists()
    monkeypatch.setattr(socket_module.os, "fstat", real_fstat)
    with SocketLease.acquire(path):
        assert path.exists()


def test_acquire_rejects_symlink_lock_without_touching_target(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    target = tmp_path / "target"
    target.write_text("do not change")
    path.with_name("control.lock").symlink_to(target)

    with pytest.raises(RuntimeError, match="lock"):
        SocketLease.acquire(path)

    assert target.read_text() == "do not change"


def test_acquire_rejects_non_regular_lock_file(tmp_path: Path) -> None:
    path = _private_path(tmp_path)
    path.with_name("control.lock").mkdir()

    with pytest.raises(RuntimeError, match="regular"):
        SocketLease.acquire(path)


def test_started_at_must_be_timezone_aware_before_filesystem_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "new-runtime" / "control.sock"

    with pytest.raises(ValueError, match="timezone-aware"):
        SocketLease.acquire(path, started_at=datetime(2026, 1, 1))

    assert not path.parent.exists()


def test_second_lease_is_rejected_then_close_allows_new_lease(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    first = SocketLease.acquire(path)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            SocketLease.acquire(path)
    finally:
        first.close()

    with SocketLease.acquire(path):
        assert path.exists()


def test_simultaneous_process_claim_has_exactly_one_lock_winner(
    tmp_path: Path,
) -> None:
    path = _private_path(tmp_path)
    lock_path = path.with_name("control.lock")
    context = multiprocessing.get_context("fork")
    start = context.Event()
    release = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_claim_socket, args=(str(path), start, release, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()

    outcomes: list[tuple[str, str, object]] = []
    try:
        outcomes = [results.get(timeout=10), results.get(timeout=10)]
        winner = next(outcome for outcome in outcomes if outcome[0] == "acquired")
        loser = next(outcome for outcome in outcomes if outcome[0] == "error")
        assert loser == (
            "error",
            "RuntimeError",
            "Control socket lease is already held: control.lock",
        )
        assert stat.S_ISSOCK(path.lstat().st_mode)
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600
        metadata = json.loads(lock_path.read_text())
        assert metadata["pid"] == winner[2]
        started_at = datetime.fromisoformat(metadata["started_at"])
        assert started_at.utcoffset() == UTC.utcoffset(None)
    except (queue.Empty, StopIteration):
        pytest.fail(
            f"socket claim subprocesses did not report expected outcomes: {outcomes}"
        )
    finally:
        release.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    assert all(process.exitcode == 0 for process in processes)
    assert not path.exists()
    assert lock_path.exists()


def test_matching_dead_prior_metadata_reclaims_refused_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    dead_pid = 4242
    _write_prior_metadata(
        path,
        _prior_metadata(
            path,
            pid=dead_pid,
            boot_id="boot-current",
            start_ticks=111,
        ),
    )
    _set_process_identity_seams(monkeypatch, pid_ticks={dead_pid: None})

    with SocketLease.acquire(path) as lease:
        _connect_and_accept(path, lease.socket)
        metadata = json.loads(path.with_name("control.lock").read_text())
        identity = path.lstat()
        assert (metadata["socket_dev"], metadata["socket_ino"]) == (
            identity.st_dev,
            identity.st_ino,
        )


def test_bound_unlistening_socket_without_proof_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(path))
    inode = path.lstat().st_ino
    _set_process_identity_seams(monkeypatch)
    try:
        lease = None
        try:
            lease = SocketLease.acquire(path)
        except RuntimeError as error:
            assert "prove" in str(error)
        else:
            lease.close()
            pytest.fail("unproven refused socket was reclaimed")
        assert path.lstat().st_ino == inode
    finally:
        endpoint.close()
        path.unlink(missing_ok=True)


def test_matching_live_prior_process_identity_rejects_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(path))
    endpoint.close()
    owner_pid = 4242
    payload = _write_prior_metadata(
        path,
        _prior_metadata(
            path,
            pid=owner_pid,
            boot_id="boot-current",
            start_ticks=111,
        ),
    )
    _set_process_identity_seams(monkeypatch, pid_ticks={owner_pid: 111})

    with pytest.raises(RuntimeError, match="prove"):
        SocketLease.acquire(path)

    assert path.exists()
    assert path.with_name("control.lock").read_text() == payload


def test_pid_reuse_start_tick_mismatch_allows_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(path))
    endpoint.close()
    owner_pid = 4242
    _write_prior_metadata(
        path,
        _prior_metadata(
            path,
            pid=owner_pid,
            boot_id="boot-current",
            start_ticks=111,
        ),
    )
    _set_process_identity_seams(monkeypatch, pid_ticks={owner_pid: 222})

    with SocketLease.acquire(path) as lease:
        _connect_and_accept(path, lease.socket)
        metadata = json.loads(path.with_name("control.lock").read_text())
        identity = path.lstat()
        assert (metadata["socket_dev"], metadata["socket_ino"]) == (
            identity.st_dev,
            identity.st_ino,
        )


@pytest.mark.parametrize("metadata_kind", ["malformed", "wrong-inode"])
def test_invalid_prior_metadata_cannot_authorize_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metadata_kind: str
) -> None:
    path = _private_path(tmp_path)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(path))
    endpoint.close()
    if metadata_kind == "malformed":
        payload = "not-json\n"
        path.with_name("control.lock").write_text(payload)
    else:
        payload = _write_prior_metadata(
            path,
            _prior_metadata(
                path,
                pid=4242,
                boot_id="boot-old",
                start_ticks=111,
                socket_ino=path.lstat().st_ino + 1,
            ),
        )
    _set_process_identity_seams(monkeypatch)

    with pytest.raises(RuntimeError, match="prove"):
        SocketLease.acquire(path)

    assert path.exists()
    assert path.with_name("control.lock").read_text() == payload


def test_socket_disappearing_between_lstat_and_probe_is_recoverable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    real_probe = socket_module._probe_socket

    def disappear_then_probe(candidate: Path) -> bool:
        candidate.unlink()
        return real_probe(candidate)

    monkeypatch.setattr(socket_module, "_probe_socket", disappear_then_probe)

    with SocketLease.acquire(path):
        assert path.exists()


def test_failed_live_acquisition_preserves_previous_lock_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(path))
    live.listen()
    lock_path = path.with_name("control.lock")
    lock_path.write_text("previous metadata must survive\n")
    _set_process_identity_seams(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="live socket"):
            SocketLease.acquire(path)
        assert lock_path.read_text() == "previous metadata must survive\n"
    finally:
        live.close()
        path.unlink(missing_ok=True)


def test_live_legacy_socket_is_rejected_and_preserved(tmp_path: Path) -> None:
    path = _private_path(tmp_path)
    legacy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    legacy.bind(str(path))
    legacy.listen()
    inode = path.lstat().st_ino

    try:
        with pytest.raises(RuntimeError, match="live socket"):
            SocketLease.acquire(path)
        assert path.lstat().st_ino == inode
    finally:
        legacy.close()
        path.unlink()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_non_socket_paths_are_never_removed(tmp_path: Path, kind: str) -> None:
    path = _private_path(tmp_path)
    if kind == "file":
        path.write_text("preserve")
    else:
        target = tmp_path / "target"
        target.write_text("preserve")
        path.symlink_to(target)

    with pytest.raises(RuntimeError, match=kind):
        SocketLease.acquire(path)

    assert path.lexists() if hasattr(path, "lexists") else os.path.lexists(path)


def test_wrong_owner_socket_is_never_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    real_stat_leaf = socket_module._stat_leaf

    def wrong_socket_owner(dir_fd: int, leaf: str) -> os.stat_result:
        result = real_stat_leaf(dir_fd, leaf)
        if leaf == path.name:
            return _with_uid(result, os.getuid() + 1)
        return result

    monkeypatch.setattr(socket_module, "_stat_leaf", wrong_socket_owner)

    with pytest.raises(RuntimeError, match="owned"):
        SocketLease.acquire(path)

    assert path.exists()


def test_unexpected_probe_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    def fail_probe(candidate: Path) -> None:
        raise OSError(errno.EACCES, "denied", candidate)

    monkeypatch.setattr(socket_module, "_probe_socket", fail_probe)

    with pytest.raises(RuntimeError, match="probe"):
        SocketLease.acquire(path)

    assert path.exists()


def test_post_bind_wrong_owner_failure_removes_socket_and_releases_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_raw_identity = socket_module._raw_leaf_identity
    real_socket = socket_module.socket.socket
    listeners: list[socket.socket] = []
    socket_lstats = 0

    def tracked_socket(*args, **kwargs) -> socket.socket:
        listener = real_socket(*args, **kwargs)
        listeners.append(listener)
        return listener

    def wrong_owner_after_capture(
        dir_fd: int, leaf: str
    ) -> tuple[int, int, int, int]:
        nonlocal socket_lstats
        result = real_raw_identity(dir_fd, leaf)
        if leaf == path.name:
            socket_lstats += 1
            if socket_lstats == 2:
                return result[0], result[1], os.getuid() + 1, result[3]
        return result

    monkeypatch.setattr(socket_module.socket, "socket", tracked_socket)
    monkeypatch.setattr(socket_module, "_raw_leaf_identity", wrong_owner_after_capture)
    lease = None
    try:
        lease = SocketLease.acquire(path)
    except RuntimeError as error:
        assert "not owned" in str(error)
    else:
        lease.close()
        pytest.fail("post-bind owner validation did not run")

    assert not path.exists()
    assert listeners[0].fileno() == -1
    monkeypatch.setattr(socket_module.socket, "socket", real_socket)
    monkeypatch.setattr(socket_module, "_raw_leaf_identity", real_raw_identity)
    with SocketLease.acquire(path):
        assert path.exists()


def test_transient_first_post_bind_lstat_failure_leaves_stale_path_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_raw_identity = socket_module._raw_leaf_identity
    real_socket = socket_module.socket.socket
    listeners: list[socket.socket] = []
    failed = False

    def tracked_socket(*args, **kwargs) -> socket.socket:
        listener = real_socket(*args, **kwargs)
        listeners.append(listener)
        return listener

    def fail_first_socket_lstat(
        dir_fd: int, leaf: str
    ) -> tuple[int, int, int, int]:
        nonlocal failed
        result = real_raw_identity(dir_fd, leaf)
        if leaf == path.name and not failed:
            failed = True
            raise OSError(errno.EIO, "transient lstat failure")
        return result

    monkeypatch.setattr(socket_module.socket, "socket", tracked_socket)
    monkeypatch.setattr(socket_module, "_raw_leaf_identity", fail_first_socket_lstat)
    with pytest.raises(OSError, match="transient lstat failure"):
        SocketLease.acquire(path)

    assert stat.S_ISSOCK(path.lstat().st_mode)
    assert listeners[0].fileno() == -1
    monkeypatch.setattr(socket_module.socket, "socket", real_socket)
    monkeypatch.setattr(socket_module, "_raw_leaf_identity", real_raw_identity)
    with pytest.raises(RuntimeError, match="prove"):
        SocketLease.acquire(path)
    assert path.exists()


def test_missing_bound_identity_never_unlinks_live_same_uid_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_raw_identity = socket_module._raw_leaf_identity
    real_socket = socket_module.socket.socket
    original_listeners: list[socket.socket] = []
    replacement: socket.socket | None = None

    def tracked_socket(*args, **kwargs) -> socket.socket:
        listener = real_socket(*args, **kwargs)
        original_listeners.append(listener)
        return listener

    def replace_then_fail(
        dir_fd: int, leaf: str
    ) -> tuple[int, int, int, int]:
        nonlocal replacement
        result = real_raw_identity(dir_fd, leaf)
        if leaf == path.name and replacement is None:
            os.unlink(leaf, dir_fd=dir_fd)
            replacement = real_socket(socket.AF_UNIX, socket.SOCK_STREAM)
            replacement.bind(str(path))
            replacement.listen()
            raise OSError(errno.EIO, "identity capture failed")
        return result

    monkeypatch.setattr(socket_module.socket, "socket", tracked_socket)
    monkeypatch.setattr(socket_module, "_raw_leaf_identity", replace_then_fail)
    try:
        with pytest.raises(OSError, match="identity capture failed"):
            SocketLease.acquire(path)

        assert replacement is not None
        assert path.exists()
        assert replacement.fileno() >= 0
        assert original_listeners[0].fileno() == -1
        monkeypatch.setattr(socket_module.socket, "socket", real_socket)
        monkeypatch.setattr(socket_module, "_raw_leaf_identity", real_raw_identity)

        _connect_and_accept(path, replacement)

        with pytest.raises(RuntimeError, match="live socket"):
            SocketLease.acquire(path)
        assert path.exists()
    finally:
        if replacement is not None:
            replacement.close()
        path.unlink(missing_ok=True)


def test_post_bind_mode_validation_requires_exact_0600(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_chmod = socket_module.os.chmod
    previous_umask = os.umask(0)
    monkeypatch.setattr(socket_module.os, "chmod", lambda *args, **kwargs: None)
    lease = None
    try:
        try:
            lease = SocketLease.acquire(path)
        except RuntimeError as error:
            assert "permissions" in str(error)
        else:
            lease.close()
            pytest.fail("post-bind mode validation did not run")
    finally:
        os.umask(previous_umask)

    assert not path.exists()
    monkeypatch.setattr(socket_module.os, "chmod", real_chmod)
    with SocketLease.acquire(path):
        assert path.exists()


def test_replacement_inode_survives_close(tmp_path: Path) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    path.unlink()
    path.write_text("replacement")

    with pytest.raises(RuntimeError, match="changed before removal"):
        lease.close()

    assert path.read_text() == "replacement"


def test_close_is_idempotent(tmp_path: Path) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)

    lease.close()
    lease.close()

    assert not path.exists()


def test_directory_replaced_after_pin_fails_without_touching_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    original_directory = path.parent
    moved_directory = tmp_path / "pinned-runtime"
    sentinel = "replacement directory must stay untouched"
    real_acquire_lock = socket_module._acquire_lock
    replaced = False

    def replace_then_lock(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            original_directory.rename(moved_directory)
            original_directory.mkdir(mode=0o700)
            (original_directory / "sentinel").write_text(sentinel)
        return real_acquire_lock(*args, **kwargs)

    monkeypatch.setattr(socket_module, "_acquire_lock", replace_then_lock)
    lease = None
    try:
        lease = SocketLease.acquire(path)
    except RuntimeError as error:
        assert "directory changed" in str(error)
    else:
        lease.close()
        pytest.fail("acquisition succeeded through replacement directory")

    assert [entry.name for entry in original_directory.iterdir()] == ["sentinel"]
    assert (original_directory / "sentinel").read_text() == sentinel
    assert not (moved_directory / "control.sock").exists()


def test_recovery_quarantine_preserves_delete_window_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    dead_pid = 4242
    previous = _write_prior_metadata(
        path,
        _prior_metadata(
            path,
            pid=dead_pid,
            boot_id="boot-current",
            start_ticks=111,
        ),
    )
    _set_process_identity_seams(monkeypatch, pid_ticks={dead_pid: None})
    _replace_leaf_during_quarantine(monkeypatch, b"recovery replacement")

    with pytest.raises(RuntimeError, match="changed before removal"):
        SocketLease.acquire(path)

    assert path.read_text() == "recovery replacement"
    assert path.with_name("control.lock").read_text() == previous


def test_partial_acquire_quarantine_preserves_delete_window_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    real_chmod = socket_module.os.chmod
    real_rename = _replace_leaf_during_quarantine(
        monkeypatch,
        b"partial replacement",
    )

    def fail_chmod(*args, **kwargs) -> None:
        raise OSError(errno.EIO, "chmod")

    monkeypatch.setattr(socket_module.os, "chmod", fail_chmod)
    with pytest.raises(OSError, match="chmod"):
        SocketLease.acquire(path)

    assert path.read_text() == "partial replacement"
    monkeypatch.setattr(socket_module.os, "chmod", real_chmod)
    monkeypatch.setattr(socket_module, "_rename_leaf", real_rename)
    path.unlink()
    with SocketLease.acquire(path):
        assert path.exists()


def test_close_quarantine_preserves_delete_window_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    real_rename = _replace_leaf_during_quarantine(
        monkeypatch,
        b"close replacement",
    )

    with pytest.raises(RuntimeError, match="changed before removal"):
        lease.close()

    assert path.read_text() == "close replacement"
    monkeypatch.setattr(socket_module, "_rename_leaf", real_rename)
    path.unlink()
    with SocketLease.acquire(path):
        assert path.exists()


def _quarantine_names(directory: Path) -> list[Path]:
    return sorted(directory.glob(".ccp-quarantine-*"))


def test_probe_uses_pinned_directory_when_configured_path_is_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    original_directory = path.parent
    moved_directory = tmp_path / "pinned-runtime"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(path))
    live.listen()
    original_inode = path.lstat().st_ino
    real_probe = socket_module._probe_socket

    def replace_directory_then_probe(candidate: Path) -> object:
        original_directory.rename(moved_directory)
        original_directory.mkdir(mode=0o700)
        return real_probe(candidate)

    monkeypatch.setattr(socket_module, "_probe_socket", replace_directory_then_probe)
    try:
        with pytest.raises(RuntimeError) as captured:
            SocketLease.acquire(path)
        assert "live socket" in str(captured.value) or "directory changed" in str(
            captured.value
        )

        pinned_path = moved_directory / path.name
        assert pinned_path.lstat().st_ino == original_inode
        assert live.fileno() >= 0
        _connect_and_accept(pinned_path, live)
        assert not (original_directory / path.name).exists()
    finally:
        live.close()
        pinned_path = moved_directory / path.name
        pinned_path.unlink(missing_ok=True)


def test_quarantine_rename_failure_cleans_reservation_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    inode = path.lstat().st_ino

    def fail_rename(dir_fd: int, source: str, target: str) -> None:
        raise OSError(errno.EIO, "rename failure")

    monkeypatch.setattr(socket_module, "_rename_leaf", fail_rename)
    with pytest.raises(OSError, match="rename failure"):
        lease.close()

    assert path.lstat().st_ino == inode
    assert _quarantine_names(path.parent) == []


def test_quarantine_post_rename_stat_failure_restores_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    inode = path.lstat().st_ino
    real_raw_identity = socket_module._raw_leaf_identity
    failed = False

    def fail_quarantine_stat(
        dir_fd: int, leaf: str
    ) -> tuple[int, int, int, int]:
        nonlocal failed
        if leaf.startswith(".ccp-quarantine-") and not failed:
            failed = True
            raise OSError(errno.EIO, "stat failure")
        return real_raw_identity(dir_fd, leaf)

    monkeypatch.setattr(socket_module, "_raw_leaf_identity", fail_quarantine_stat)
    with pytest.raises(OSError, match="stat failure"):
        lease.close()

    assert path.lstat().st_ino == inode
    assert _quarantine_names(path.parent) == []


def test_quarantine_post_rename_unlink_failure_restores_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    inode = path.lstat().st_ino
    real_unlink = socket_module.os.unlink
    failed = False

    def fail_quarantine_unlink(
        candidate: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed
        if str(candidate).startswith(".ccp-quarantine-") and not failed:
            failed = True
            raise OSError(errno.EIO, "unlink failure")
        real_unlink(candidate, dir_fd=dir_fd)

    monkeypatch.setattr(socket_module.os, "unlink", fail_quarantine_unlink)
    with pytest.raises(OSError, match="unlink failure"):
        lease.close()

    assert path.lstat().st_ino == inode
    assert _quarantine_names(path.parent) == []


def test_quarantine_failed_restore_preserves_public_and_quarantine_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _private_path(tmp_path)
    lease = SocketLease.acquire(path)
    original_inode = path.lstat().st_ino
    real_raw_identity = socket_module._raw_leaf_identity
    failed = False

    def replace_public_then_fail(
        dir_fd: int, leaf: str
    ) -> tuple[int, int, int, int]:
        nonlocal failed
        if leaf.startswith(".ccp-quarantine-") and not failed:
            failed = True
            replacement_fd = os.open(
                path.name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                dir_fd=dir_fd,
            )
            os.write(replacement_fd, b"public replacement")
            os.close(replacement_fd)
            raise OSError(errno.EIO, "validation failure")
        return real_raw_identity(dir_fd, leaf)

    monkeypatch.setattr(socket_module, "_raw_leaf_identity", replace_public_then_fail)
    with pytest.raises(OSError, match="validation failure"):
        lease.close()

    assert path.read_text() == "public replacement"
    quarantines = _quarantine_names(path.parent)
    assert len(quarantines) == 1
    assert quarantines[0].lstat().st_ino == original_inode

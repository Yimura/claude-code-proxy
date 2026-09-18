"""Secure ownership of the local control-plane Unix socket."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
import errno
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import sys
import tempfile
from types import TracebackType
from typing import Self

if sys.platform == "linux":
    import fcntl


_APP_DIRECTORY_NAME = "claude-code-proxy"
_SOCKET_NAME = "control.sock"
_LOCK_NAME = "control.lock"
_LINUX_SUN_PATH_BYTES = 108
_MAX_LOCK_METADATA_BYTES = 1_048_576
_PROBE_TIMEOUT_SECONDS = 0.1
_RUN_USER_ROOT = Path("/run/user")
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


class _ProbeOutcome(Enum):
    LIVE = auto()
    DISAPPEARED = auto()
    INDETERMINATE = auto()


class _OwnerState(Enum):
    LIVE = auto()
    GONE = auto()
    INDETERMINATE = auto()


@dataclass(frozen=True)
class _ProcessIdentity:
    boot_id: str
    start_ticks: int


@dataclass(frozen=True)
class _LockMetadata:
    pid: int
    started_at: datetime
    process: _ProcessIdentity
    socket_dev: int
    socket_ino: int

    def to_dict(self) -> dict[str, object]:
        return {
            "boot_id": self.process.boot_id,
            "pid": self.pid,
            "pid_start_ticks": self.process.start_ticks,
            "socket_dev": self.socket_dev,
            "socket_ino": self.socket_ino,
            "started_at": self.started_at.isoformat().replace("+00:00", "Z"),
        }


@dataclass(frozen=True)
class _PreviousLock:
    raw: bytes
    metadata: _LockMetadata | None


@dataclass(frozen=True)
class _PinnedDirectory:
    fd: int
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class _AcquireContext:
    path: Path
    leaf: str
    uid: int
    pid: int
    started_at: datetime
    process: _ProcessIdentity
    directory: _PinnedDirectory


class SocketLease:
    """Exclusive lifetime ownership of a listening Unix-domain socket."""

    def __init__(
        self,
        path: Path,
        listener: socket.socket,
        lock_fd: int,
        dir_fd: int,
        identity: tuple[int, int, int],
        socket_leaf: str,
    ) -> None:
        self._path = path
        self._socket = listener
        self._lock_fd = lock_fd
        self._dir_fd = dir_fd
        self._identity = identity
        self._socket_leaf = socket_leaf
        self._closed = False

    @property
    def path(self) -> Path:
        """Return the absolute bound socket path."""
        return self._path

    @property
    def socket(self) -> socket.socket:
        """Return the nonblocking listening socket."""
        return self._socket

    @classmethod
    def acquire(
        cls,
        path: Path,
        *,
        uid: int | None = None,
        pid: int | None = None,
        started_at: datetime | None = None,
    ) -> Self:
        """Acquire the lock, recover a stale endpoint, and bind the socket."""
        _require_linux()
        expected_uid = os.getuid() if uid is None else uid
        socket_path = _absolute_socket_path(path)
        socket_leaf = _socket_leaf(socket_path)
        process_id = _validated_pid(os.getpid() if pid is None else pid)
        timestamp = _started_at(started_at)
        process_identity = _current_process_identity(process_id)
        dir_fd, directory_identity = _ensure_and_pin_directory(
            socket_path.parent,
            expected_uid,
        )
        context = _AcquireContext(
            socket_path,
            socket_leaf,
            expected_uid,
            process_id,
            timestamp,
            process_identity,
            _PinnedDirectory(dir_fd, directory_identity),
        )
        return cls._acquire_pinned(context)

    @classmethod
    def _acquire_pinned(cls, context: _AcquireContext) -> Self:
        path = context.path
        leaf = context.leaf
        uid = context.uid
        dir_fd = context.directory.fd
        lock_fd: int | None = None
        listener: socket.socket | None = None
        identity: tuple[int, int, int] | None = None
        try:
            _validate_proc_bind_path(dir_fd, leaf)
            lock_fd, previous = _acquire_lock(dir_fd, uid)
            _recover_existing_socket(dir_fd, leaf, path, uid, previous.metadata)
            listener, identity = _bind_listener(dir_fd, leaf, uid)
            _verify_directory_path(path.parent, context.directory.identity, uid)
            metadata = _LockMetadata(
                context.pid,
                context.started_at,
                context.process,
                identity[0],
                identity[1],
            )
            _write_lock_metadata(lock_fd, metadata, previous.raw)
            try:
                _verify_directory_path(path.parent, context.directory.identity, uid)
            except BaseException:
                _replace_lock_contents(lock_fd, previous.raw)
                raise
            return cls(path, listener, lock_fd, dir_fd, identity, leaf)
        except BaseException:
            _cleanup_failed_acquire(dir_fd, leaf, listener, identity, lock_fd)
            os.close(dir_fd)
            raise

    def close(self) -> None:
        """Release this lease, removing only the socket inode it created."""
        if self._closed:
            return
        self._closed = True

        first_error: BaseException | None = None
        try:
            _quarantine_and_remove(
                self._dir_fd,
                self._socket_leaf,
                self._identity,
            )
        except BaseException as error:
            first_error = error
        for operation in (
            self._socket.close,
            lambda: fcntl.flock(self._lock_fd, fcntl.LOCK_UN),
            lambda: os.close(self._lock_fd),
            lambda: os.close(self._dir_fd),
        ):
            try:
                operation()
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "secure control socket requires Linux; "
            "use Docker on other platforms"
        )


def resolve_socket_path(
    explicit: Path | None,
    environ: Mapping[str, str] | None = None,
    *,
    uid: int | None = None,
) -> Path:
    """Resolve the control socket location in deterministic priority order."""
    _require_linux()
    expected_uid = os.getuid() if uid is None else uid
    environment = os.environ if environ is None else environ
    if explicit is not None:
        return _absolute_socket_path(explicit)

    configured = environment.get("CONTROL_SOCKET_PATH")
    if configured is not None:
        stripped = configured.strip()
        if not stripped:
            raise ValueError("CONTROL_SOCKET_PATH must not be empty")
        return _absolute_socket_path(Path(stripped))

    xdg_base = _eligible_runtime_base(
        environment.get("XDG_RUNTIME_DIR"), expected_uid
    )
    if xdg_base is not None:
        return _absolute_socket_path(xdg_base / _APP_DIRECTORY_NAME / _SOCKET_NAME)

    run_base = _RUN_USER_ROOT / str(expected_uid)
    if _is_private_runtime_directory(run_base, expected_uid):
        return _absolute_socket_path(run_base / _APP_DIRECTORY_NAME / _SOCKET_NAME)

    temporary = Path(tempfile.gettempdir()) / f"{_APP_DIRECTORY_NAME}-{expected_uid}"
    return _absolute_socket_path(temporary / _SOCKET_NAME)


def _absolute_socket_path(path: Path) -> Path:
    value = os.fspath(path)
    if "\0" in value:
        raise ValueError("socket path must not contain NUL bytes")
    absolute = Path(value).expanduser().absolute()
    _validate_unix_socket_path(os.fspath(absolute))
    return absolute


def _validate_unix_socket_path(path: str) -> None:
    if "\0" in path:
        raise ValueError("socket path must not contain NUL bytes")
    if len(os.fsencode(path)) >= _LINUX_SUN_PATH_BYTES:
        raise ValueError("socket path is too long for Linux AF_UNIX")


def _socket_leaf(path: Path) -> str:
    leaf = path.name
    if not leaf or leaf in (".", ".."):
        raise ValueError("socket path must name a socket file")
    if leaf == _LOCK_NAME:
        raise ValueError("socket path must not collide with control.lock")
    return leaf


def _eligible_runtime_base(raw: str | None, uid: int) -> Path | None:
    if raw is None or not raw.strip() or "\0" in raw:
        return None
    base = Path(raw.strip()).expanduser()
    if not base.is_absolute():
        return None
    return base if _is_private_runtime_directory(base, uid) else None


def _is_private_runtime_directory(path: Path, uid: int) -> bool:
    try:
        details = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError):
        return False
    return (
        stat.S_ISDIR(details.st_mode)
        and details.st_uid == uid
        and stat.S_IMODE(details.st_mode) == 0o700
    )


def _ensure_and_pin_directory(
    directory: Path,
    uid: int,
) -> tuple[int, tuple[int, int, int]]:
    _ensure_application_directory(directory, uid)
    path_details = _validate_application_directory(directory, uid)
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        dir_fd = os.open(directory, flags)
    except OSError as error:
        raise RuntimeError(
            f"Cannot securely open control socket directory: {directory}"
        ) from error
    try:
        descriptor = _validate_directory_descriptor(dir_fd, uid, directory)
        path_identity = path_details.st_dev, path_details.st_ino
        if descriptor[:2] != path_identity:
            raise RuntimeError(
                f"Control socket directory changed while opening: {directory}"
            )
        _verify_directory_path(directory, descriptor, uid)
        return dir_fd, descriptor
    except BaseException:
        os.close(dir_fd)
        raise


def _ensure_application_directory(directory: Path, uid: int) -> None:
    created = False
    try:
        os.mkdir(directory, 0o700)
        created = True
    except FileExistsError:
        pass
    details = _validate_application_directory(directory, uid)
    if created:
        _set_created_directory_mode(directory, uid, details)


def _set_created_directory_mode(
    directory: Path,
    uid: int,
    created: os.stat_result,
) -> None:
    if stat.S_IMODE(created.st_mode) != 0o700:
        os.chmod(directory, 0o700, follow_symlinks=False)
    verified = _validate_application_directory(directory, uid)
    if _stat_identity(created) != _stat_identity(verified):
        raise RuntimeError(
            f"Control socket directory changed after creation: {directory}"
        )
    if stat.S_IMODE(verified.st_mode) != 0o700:
        raise RuntimeError(f"Cannot secure control socket directory: {directory}")


def _validate_application_directory(directory: Path, uid: int) -> os.stat_result:
    try:
        details = os.lstat(directory)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Control socket directory disappeared: {directory}"
        ) from error
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(
            f"Control socket directory must not be a symlink: {directory}"
        )
    if not stat.S_ISDIR(details.st_mode):
        raise RuntimeError(f"Control socket path is not a directory: {directory}")
    if details.st_uid != uid:
        raise RuntimeError(
            f"Control socket directory is not owned by UID {uid}: {directory}"
        )
    if stat.S_IMODE(details.st_mode) & 0o077:
        raise RuntimeError(
            f"Control socket directory has insecure permissions: {directory}"
        )
    return details


def _validate_directory_descriptor(
    dir_fd: int,
    uid: int,
    directory: Path,
) -> tuple[int, int, int]:
    details = os.fstat(dir_fd)
    if not stat.S_ISDIR(details.st_mode):
        raise RuntimeError(f"Pinned control path is not a directory: {directory}")
    if details.st_uid != uid:
        raise RuntimeError(
            f"Pinned control directory is not owned by UID {uid}: {directory}"
        )
    if stat.S_IMODE(details.st_mode) & 0o077:
        raise RuntimeError(
            f"Pinned control directory has insecure permissions: {directory}"
        )
    return details.st_dev, details.st_ino, details.st_uid


def _verify_directory_path(
    directory: Path,
    expected: tuple[int, int, int],
    uid: int,
) -> None:
    details = _validate_application_directory(directory, uid)
    if _stat_identity(details) != expected:
        raise RuntimeError(
            f"Control socket directory changed after pinning: {directory}"
        )


def _stat_identity(details: os.stat_result) -> tuple[int, int, int]:
    return details.st_dev, details.st_ino, details.st_uid


def _validated_pid(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("pid must be a positive integer")
    return value


def _started_at(value: datetime | None) -> datetime:
    timestamp = datetime.now(UTC) if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    return timestamp.astimezone(UTC)


def _current_process_identity(pid: int) -> _ProcessIdentity:
    try:
        boot_id = _read_boot_id()
        start_ticks = _read_pid_start_ticks(pid)
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot determine Linux process identity") from error
    if not boot_id or start_ticks is None:
        raise RuntimeError("Cannot determine Linux process identity")
    return _ProcessIdentity(boot_id, start_ticks)


def _read_boot_id() -> str | None:
    try:
        value = _BOOT_ID_PATH.read_text().strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _read_pid_start_ticks(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    closing_parenthesis = value.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("malformed /proc PID stat")
    fields = value[closing_parenthesis + 2 :].split()
    if len(fields) <= 19:
        raise ValueError("malformed /proc PID stat")
    return int(fields[19])


def _acquire_lock(dir_fd: int, uid: int) -> tuple[int, _PreviousLock]:
    _validate_existing_lock(dir_fd, uid)
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(_LOCK_NAME, flags, 0o600, dir_fd=dir_fd)
    except OSError as error:
        raise RuntimeError("Cannot securely open control lock") from error
    try:
        _validate_open_lock(lock_fd, uid)
        os.fchmod(lock_fd, 0o600)
        _lock_exclusively(lock_fd)
        raw = _read_lock_bytes(lock_fd)
        return lock_fd, _PreviousLock(raw, _parse_lock_metadata(raw))
    except BaseException:
        os.close(lock_fd)
        raise


def _validate_existing_lock(dir_fd: int, uid: int) -> None:
    try:
        details = os.stat(_LOCK_NAME, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError("Control lock must not be a symlink")
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeError("Control lock must be a regular file")
    if details.st_uid != uid:
        raise RuntimeError(f"Control lock is not owned by UID {uid}")


def _validate_open_lock(lock_fd: int, uid: int) -> None:
    details = os.fstat(lock_fd)
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeError("Control lock must be a regular file")
    if details.st_uid != uid:
        raise RuntimeError(f"Control lock is not owned by UID {uid}")


def _lock_exclusively(lock_fd: int) -> None:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN):
            raise RuntimeError(
                "Control socket lease is already held: control.lock"
            ) from error
        raise


def _read_lock_bytes(lock_fd: int) -> bytes:
    size = os.fstat(lock_fd).st_size
    if size > _MAX_LOCK_METADATA_BYTES:
        raise RuntimeError("Control lock metadata is too large")
    os.lseek(lock_fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(lock_fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_lock_metadata(raw: bytes) -> _LockMetadata | None:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        pid = _metadata_positive_int(value, "pid")
        ticks = _metadata_positive_int(value, "pid_start_ticks")
        socket_dev = _metadata_nonnegative_int(value, "socket_dev")
        socket_ino = _metadata_nonnegative_int(value, "socket_ino")
        boot_id = value["boot_id"]
        if not isinstance(boot_id, str) or not boot_id:
            return None
        started_at = datetime.fromisoformat(value["started_at"])
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            return None
        return _LockMetadata(
            pid,
            started_at.astimezone(UTC),
            _ProcessIdentity(boot_id, ticks),
            socket_dev,
            socket_ino,
        )
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None


def _metadata_positive_int(value: dict[str, object], key: str) -> int:
    candidate = value[key]
    if type(candidate) is not int or candidate <= 0:
        raise ValueError(key)
    return candidate


def _metadata_nonnegative_int(value: dict[str, object], key: str) -> int:
    candidate = value[key]
    if type(candidate) is not int or candidate < 0:
        raise ValueError(key)
    return candidate


def _write_lock_metadata(
    lock_fd: int,
    metadata: _LockMetadata,
    previous: bytes,
) -> None:
    payload = (json.dumps(metadata.to_dict(), separators=(",", ":")) + "\n").encode()
    try:
        _replace_lock_contents(lock_fd, payload)
    except BaseException:
        try:
            _replace_lock_contents(lock_fd, previous)
        except BaseException:
            pass
        raise


def _replace_lock_contents(lock_fd: int, payload: bytes) -> None:
    os.ftruncate(lock_fd, 0)
    os.lseek(lock_fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(lock_fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write to control lock")
        view = view[written:]
    os.fsync(lock_fd)


def _recover_existing_socket(
    dir_fd: int,
    leaf: str,
    path: Path,
    uid: int,
    previous: _LockMetadata | None,
) -> None:
    existing = _socket_identity(dir_fd, leaf, uid, path)
    if existing is None:
        return
    try:
        outcome = _probe_socket(Path(_validate_proc_bind_path(dir_fd, leaf)))
    except OSError as error:
        raise RuntimeError(
            f"Cannot safely probe existing control socket: {path}"
        ) from error
    if outcome is _ProbeOutcome.LIVE:
        raise RuntimeError(f"Control path has a live socket: {path}")
    if outcome is _ProbeOutcome.INDETERMINATE:
        if not _metadata_proves_stale(previous, existing):
            raise RuntimeError(f"Cannot prove existing control socket is stale: {path}")
    _quarantine_and_remove(dir_fd, leaf, existing)


def _metadata_proves_stale(
    metadata: _LockMetadata | None,
    socket_identity: tuple[int, int, int],
) -> bool:
    if metadata is None:
        return False
    if (metadata.socket_dev, metadata.socket_ino) != socket_identity[:2]:
        return False
    return _recorded_owner_state(metadata) is _OwnerState.GONE


def _recorded_owner_state(metadata: _LockMetadata) -> _OwnerState:
    try:
        boot_id = _read_boot_id()
        if boot_id is None:
            return _OwnerState.INDETERMINATE
        if boot_id != metadata.process.boot_id:
            return _OwnerState.GONE
        ticks = _read_pid_start_ticks(metadata.pid)
    except (OSError, ValueError):
        return _OwnerState.INDETERMINATE
    if ticks is None or ticks != metadata.process.start_ticks:
        return _OwnerState.GONE
    return _OwnerState.LIVE


def _socket_identity(
    dir_fd: int,
    leaf: str,
    uid: int,
    display_path: Path,
) -> tuple[int, int, int] | None:
    try:
        details = _stat_leaf(dir_fd, leaf)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(f"Control socket path is a symlink: {display_path}")
    if not stat.S_ISSOCK(details.st_mode):
        raise RuntimeError(f"Control socket path is a non-socket file: {display_path}")
    if details.st_uid != uid:
        raise RuntimeError(f"Control socket is not owned by UID {uid}: {display_path}")
    return _stat_identity(details)


def _stat_leaf(dir_fd: int, leaf: str) -> os.stat_result:
    return os.stat(leaf, dir_fd=dir_fd, follow_symlinks=False)


def _raw_leaf_identity(dir_fd: int, leaf: str) -> tuple[int, int, int, int]:
    details = _stat_leaf(dir_fd, leaf)
    return details.st_dev, details.st_ino, details.st_uid, details.st_mode


def _probe_socket(path: Path) -> _ProbeOutcome:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_PROBE_TIMEOUT_SECONDS)
    try:
        probe.connect(str(path))
    except FileNotFoundError:
        return _ProbeOutcome.DISAPPEARED
    except OSError as error:
        if error.errno == errno.ECONNREFUSED:
            return _ProbeOutcome.INDETERMINATE
        raise
    finally:
        probe.close()
    return _ProbeOutcome.LIVE


def _validate_proc_bind_path(dir_fd: int, leaf: str) -> str:
    path = f"/proc/self/fd/{dir_fd}/{leaf}"
    _validate_unix_socket_path(path)
    return path


def _bind_listener(
    dir_fd: int,
    leaf: str,
    uid: int,
) -> tuple[socket.socket, tuple[int, int, int]]:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound: tuple[int, int, int, int] | None = None
    try:
        listener.bind(_validate_proc_bind_path(dir_fd, leaf))
        bound = _raw_leaf_identity(dir_fd, leaf)
        os.chmod(leaf, 0o600, dir_fd=dir_fd, follow_symlinks=False)
        identity = _validate_bound_socket(dir_fd, leaf, uid, bound)
        listener.listen()
        listener.setblocking(False)
        return listener, identity
    except BaseException:
        try:
            if bound is not None:
                _quarantine_and_remove(dir_fd, leaf, bound[:3])
        except BaseException:
            pass
        finally:
            listener.close()
        raise


def _validate_bound_socket(
    dir_fd: int,
    leaf: str,
    uid: int,
    bound: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    current = _raw_leaf_identity(dir_fd, leaf)
    if current[:2] != bound[:2]:
        raise RuntimeError("Control socket changed after bind")
    if current[2] != uid:
        raise RuntimeError(f"Control socket is not owned by UID {uid}")
    if not stat.S_ISSOCK(current[3]):
        raise RuntimeError("Bound control path is not a socket")
    if stat.S_IMODE(current[3]) != 0o600:
        raise RuntimeError("Control socket has insecure permissions")
    return current[0], current[1], current[2]


def _quarantine_and_remove(
    dir_fd: int,
    leaf: str,
    expected: tuple[int, int, int],
) -> bool:
    quarantine = _reserve_quarantine_leaf(dir_fd)
    source_moved = False
    try:
        _rename_leaf(dir_fd, leaf, quarantine)
        source_moved = True
        moved = _raw_leaf_identity(dir_fd, quarantine)
        if moved[:3] != expected or not stat.S_ISSOCK(moved[3]):
            raise RuntimeError("Control socket changed before removal")
        os.unlink(quarantine, dir_fd=dir_fd)
        return True
    except FileNotFoundError:
        if not source_moved:
            _remove_reservation(dir_fd, quarantine)
            return False
        _restore_after_quarantine_error(dir_fd, quarantine, leaf)
        raise
    except BaseException:
        if source_moved:
            _restore_after_quarantine_error(dir_fd, quarantine, leaf)
        else:
            _remove_reservation(dir_fd, quarantine)
        raise


def _remove_reservation(dir_fd: int, quarantine: str) -> None:
    try:
        os.unlink(quarantine, dir_fd=dir_fd)
    except OSError:
        pass


def _restore_after_quarantine_error(
    dir_fd: int,
    quarantine: str,
    leaf: str,
) -> None:
    try:
        _restore_quarantined_leaf(dir_fd, quarantine, leaf)
    except OSError:
        pass


def _reserve_quarantine_leaf(dir_fd: int) -> str:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(8):
        leaf = f".ccp-quarantine-{secrets.token_hex(16)}"
        try:
            reservation_fd = os.open(leaf, flags, 0o600, dir_fd=dir_fd)
        except FileExistsError:
            continue
        os.close(reservation_fd)
        return leaf
    raise RuntimeError("Cannot reserve socket quarantine name")


def _rename_leaf(dir_fd: int, source: str, target: str) -> None:
    os.rename(source, target, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)


def _restore_quarantined_leaf(
    dir_fd: int,
    quarantine: str,
    leaf: str,
) -> None:
    try:
        os.link(
            quarantine,
            leaf,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return
    os.unlink(quarantine, dir_fd=dir_fd)


def _cleanup_failed_acquire(
    dir_fd: int,
    leaf: str,
    listener: socket.socket | None,
    identity: tuple[int, int, int] | None,
    lock_fd: int | None,
) -> None:
    if identity is not None:
        try:
            _quarantine_and_remove(dir_fd, leaf, identity)
        except BaseException:
            pass
    if listener is not None:
        try:
            listener.close()
        except BaseException:
            pass
    if lock_fd is None:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except BaseException:
        pass
    try:
        os.close(lock_fd)
    except BaseException:
        pass

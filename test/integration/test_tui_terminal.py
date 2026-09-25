from __future__ import annotations

import array
import fcntl
import os
from pathlib import Path
import pty
import selectors
import signal
import subprocess
import termios
import time

import pytest

from test.integration.control_plane_support import (
    PROCESS_TIMEOUT_SECONDS,
    RunningProxy,
    cli_executable,
    proxy_process,
    stop_process,
    write_mapping,
)

_ALT_SCREEN = b"\x1b[?1049h"
_OUTPUT_LIMIT = 32_768


def _prepare_proxy(tmp_path: Path) -> tuple[Path, Path]:
    mapping_path = tmp_path / "models.json"
    write_mapping(mapping_path)
    (tmp_path / "home").mkdir(exist_ok=True)
    return mapping_path, cli_executable()


def _make_controlling_terminal(slave_fd: int) -> None:
    os.setsid()
    fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
    os.tcsetpgrp(slave_fd, os.getpgrp())


def _launch_tui(
    tmp_path: Path,
    executable: Path,
    running: RunningProxy,
    slave_fd: int,
) -> subprocess.Popen[bytes]:
    environment = {
        **running.environment,
        "TERM": "xterm-256color",
        "TEXTUAL_DISABLE_KITTY_KEY": "1",
    }
    environment.pop("COLORTERM", None)
    return subprocess.Popen(
        [str(executable), "tui", "--socket", str(running.socket_path)],
        cwd=tmp_path,
        env=environment,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=lambda: _make_controlling_terminal(slave_fd),
    )


def _read_until(
    process: subprocess.Popen[bytes],
    master_fd: int,
    marker: bytes,
    *,
    timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> bytes:
    output = bytearray()
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    try:
        while time.monotonic() < deadline:
            if marker in output:
                return bytes(output)
            remaining = max(0.0, deadline - time.monotonic())
            events = selector.select(timeout=remaining)
            if not events:
                continue
            try:
                chunk = os.read(master_fd, 8192)
            except OSError as error:
                if error.errno == 5 and process.poll() is not None:
                    break
                raise
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > _OUTPUT_LIMIT:
                del output[:-_OUTPUT_LIMIT]
    finally:
        selector.close()
    raise AssertionError(
        f"TUI did not emit {marker!r} before deadline; "
        f"returncode={process.poll()}; output tail={bytes(output)!r}"
    )


def _read_until_quiet(
    process: subprocess.Popen[bytes],
    master_fd: int,
    output: bytes,
    *,
    timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> bytes:
    captured = bytearray(output[-_OUTPUT_LIMIT:])
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    try:
        while time.monotonic() < deadline:
            events = selector.select(timeout=0.1)
            if not events:
                return bytes(captured)
            if process.poll() is not None:
                break
            chunk = os.read(master_fd, 8192)
            captured.extend(chunk)
            if len(captured) > _OUTPUT_LIMIT:
                del captured[:-_OUTPUT_LIMIT]
    finally:
        selector.close()
    raise AssertionError(
        "TUI output did not settle before deadline; "
        f"returncode={process.poll()}; output tail={bytes(captured)!r}"
    )


def _read_some(
    process: subprocess.Popen[bytes],
    master_fd: int,
    *,
    timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> bytes:
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not selector.select(timeout=remaining):
                continue
            chunk = os.read(master_fd, 8192)
            if chunk:
                return chunk
            break
    finally:
        selector.close()
    raise AssertionError(
        "TUI emitted no redraw after resize before deadline; "
        f"returncode={process.poll()}"
    )


def _drain_available(master_fd: int) -> bytes:
    output = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    try:
        while selector.select(timeout=0):
            try:
                chunk = os.read(master_fd, 8192)
            except OSError as error:
                if error.errno == 5:
                    break
                raise
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > _OUTPUT_LIMIT:
                del output[:-_OUTPUT_LIMIT]
    finally:
        selector.close()
    return bytes(output)


def _set_window_size(slave_fd: int, rows: int, columns: int) -> None:
    size = array.array("H", [rows, columns, 0, 0])
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, size)


def _wait_for_exit(
    process: subprocess.Popen[bytes],
    master_fd: int,
    output: bytes,
) -> bytes:
    captured = bytearray(output[-_OUTPUT_LIMIT:])
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ)
    try:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                assert returncode == 0, (
                    f"TUI exited {returncode}; output tail={bytes(captured)!r}"
                )
                return bytes(captured)
            remaining = max(0.0, deadline - time.monotonic())
            for _key, _mask in selector.select(timeout=min(0.1, remaining)):
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError as error:
                    if error.errno == 5:
                        continue
                    raise
                captured.extend(chunk)
                if len(captured) > _OUTPUT_LIMIT:
                    del captured[:-_OUTPUT_LIMIT]
    finally:
        selector.close()
    raise AssertionError(
        "TUI did not exit before deadline; "
        f"returncode={process.poll()}; output tail={bytes(captured)!r}"
    )


@pytest.mark.parametrize("exit_action", ["q", "sigint"])
def test_tui_terminal_exit_restores_termios(
    tmp_path: Path,
    exit_action: str,
) -> None:
    mapping_path, executable = _prepare_proxy(tmp_path)
    with proxy_process(
        tmp_path,
        mapping_path,
        executable,
        performance="collector",
    ) as running:
        master_fd, slave_fd = pty.openpty()
        process: subprocess.Popen[bytes] | None = None
        try:
            _set_window_size(slave_fd, 30, 120)
            before = termios.tcgetattr(slave_fd)
            process = _launch_tui(tmp_path, executable, running, slave_fd)
            output = _read_until(process, master_fd, b"CONNECTED")
            output = _read_until_quiet(process, master_fd, output)
            if exit_action == "q":
                os.write(master_fd, b"q")
            else:
                process.send_signal(signal.SIGINT)
            output = _wait_for_exit(process, master_fd, output)
            output += _drain_available(master_fd)

            assert termios.tcgetattr(slave_fd) == before
            assert b"Traceback" not in output
        finally:
            if process is not None:
                stop_process(process)
            os.close(master_fd)
            os.close(slave_fd)


def test_sigwinch_resize_has_no_traceback_or_hang(tmp_path: Path) -> None:
    mapping_path, executable = _prepare_proxy(tmp_path)
    with proxy_process(
        tmp_path,
        mapping_path,
        executable,
        performance="collector",
    ) as running:
        master_fd, slave_fd = pty.openpty()
        process: subprocess.Popen[bytes] | None = None
        try:
            _set_window_size(slave_fd, 30, 120)
            process = _launch_tui(tmp_path, executable, running, slave_fd)
            output = _read_until(process, master_fd, b"CONNECTED")
            output = _read_until_quiet(process, master_fd, output)

            _set_window_size(slave_fd, 18, 72)
            process.send_signal(signal.SIGWINCH)
            output += _read_some(process, master_fd)
            assert process.poll() is None, f"TUI exited during resize: {output!r}"

            os.write(master_fd, b"q")
            output = _wait_for_exit(process, master_fd, output)
            output += _drain_available(master_fd)
            assert b"Traceback" not in output
        finally:
            if process is not None:
                stop_process(process)
            os.close(master_fd)
            os.close(slave_fd)


def test_non_tty_fails_before_alternate_screen(tmp_path: Path) -> None:
    executable = cli_executable()
    completed = subprocess.run(
        [str(executable), "tui"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=PROCESS_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 1
    assert b"interactive TTY" in completed.stderr
    assert _ALT_SCREEN not in completed.stdout
    assert _ALT_SCREEN not in completed.stderr
    assert b"Traceback" not in completed.stderr

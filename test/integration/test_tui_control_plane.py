from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import socket
import time
from typing import Any

import httpx
import pytest

from claude_code_proxy.tui.app import (
    AppResult,
    ConnectionPhase,
    TuiApp,
)
from test.integration.control_plane_support import (
    RunningProxy,
    cleanup_running_proxy,
    cli_executable,
    proxy_process,
    reserve_loopback_port,
    start_proxy_with_retries,
    write_mapping,
)

_WAIT_TIMEOUT_SECONDS = 10.0
_PROTOCOL_FAILURE = (
    "TUI requires control protocol v1 with performance and "
    "performance_events capabilities"
)


def _prepare_proxy(tmp_path: Path) -> tuple[Path, Path]:
    mapping_path = tmp_path / "models.json"
    write_mapping(mapping_path)
    (tmp_path / "home").mkdir(exist_ok=True)
    return mapping_path, cli_executable()


def _count_tokens(running: RunningProxy, session_id: str) -> None:
    with httpx.Client(timeout=5.0, trust_env=False) as client:
        response = client.post(
            f"http://127.0.0.1:{running.port}/v1/messages/count_tokens",
            headers={"x-claude-code-session-id": session_id},
            json={
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "count locally"}],
            },
        )
    assert response.status_code == 200, response.text


async def _wait_until(
    pilot: Any,
    predicate: Callable[[], bool],
    description: str,
) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {description}")


async def _wait_for_connected(
    app: TuiApp,
    pilot: Any,
    process_id: int,
) -> None:
    await _wait_until(
        pilot,
        lambda: (
            app.connection_status.phase is ConnectionPhase.CONNECTED
            and app.state.process is not None
            and app.state.process.pid == process_id
        ),
        f"CONNECTED reset from proxy PID {process_id}",
    )


@pytest.mark.asyncio
async def test_real_control_stream_populates_headless_tui(tmp_path: Path) -> None:
    mapping_path, executable = _prepare_proxy(tmp_path)
    with proxy_process(
        tmp_path,
        mapping_path,
        executable,
        performance="collector",
    ) as running:
        app = TuiApp(running.socket_path)
        async with app.run_test(size=(120, 32)) as pilot:
            await _wait_for_connected(app, pilot, running.process.pid)
            assert app.state.cursor == 0
            assert app.state.sessions == {}

            await asyncio.to_thread(
                _count_tokens, running, "control-plane-integration-session"
            )
            await _wait_until(
                pilot,
                lambda: len(app.state.sessions) == 1,
                "live count_tokens session row",
            )

            session = next(iter(app.state.sessions.values()))
            assert app.connection_status.phase is ConnectionPhase.CONNECTED
            assert session.performance.latest_request is not None
            assert session.performance.latest_request.operation == "count_tokens"
            assert session.session.requests == 1
            await pilot.press("q")

    assert app.return_value == AppResult(0)


@pytest.mark.asyncio
async def test_restart_replaces_stale_rows_with_lower_sequence_reset(
    tmp_path: Path,
) -> None:
    mapping_path, executable = _prepare_proxy(tmp_path)
    socket_dir = tmp_path / "stable-control"
    socket_dir.mkdir(mode=0o700)
    socket_path = socket_dir / "control.sock"
    reservation, port = reserve_loopback_port()
    reservation.close()
    first = start_proxy_with_retries(
        tmp_path,
        mapping_path,
        executable,
        performance="collector",
        socket_path=socket_path,
        port=port,
    )
    second: RunningProxy | None = None
    diagnostics: list[str] = []
    failure: BaseException | None = None
    try:
        app = TuiApp(socket_path)
        async with app.run_test(size=(120, 32)) as pilot:
            await _wait_for_connected(app, pilot, first.process.pid)
            await asyncio.to_thread(
                _count_tokens, first, "restart-integration-session"
            )
            await _wait_until(
                pilot,
                lambda: len(app.state.sessions) == 1,
                "pre-restart live session row",
            )
            stale_session_id = next(iter(app.state.sessions))
            app.state = app.state.select_session(stale_session_id)
            old_process = app.state.process
            old_cursor = app.state.cursor
            assert old_process is not None
            assert old_cursor > 0

            stdout, stderr = cleanup_running_proxy(first)
            diagnostics.append(
                f"first proxy stdout:\n{stdout}\nfirst proxy stderr:\n{stderr}"
            )
            await _wait_until(
                pilot,
                lambda: app.connection_status.phase
                in {ConnectionPhase.DISCONNECTED, ConnectionPhase.RECONNECTING},
                "stale reconnecting state after first proxy stopped",
            )
            assert stale_session_id in app.state.sessions
            assert app.state.process == old_process

            second = await asyncio.to_thread(
                start_proxy_with_retries,
                tmp_path,
                mapping_path,
                executable,
                performance="collector",
                socket_path=socket_path,
                port=port,
            )
            await _wait_for_connected(app, pilot, second.process.pid)
            await _wait_until(
                pilot,
                lambda: not app.state.sessions,
                "fresh reset replacing stale rows",
            )

            assert app.state.process != old_process
            assert app.state.cursor < old_cursor
            assert app.state.selected_session_id is None
            assert app.state.selected_request_id is None
            await pilot.press("q")
    except BaseException as error:
        failure = error
    finally:
        if first.process.poll() is None:
            stdout, stderr = cleanup_running_proxy(first)
            diagnostics.append(
                f"first proxy stdout:\n{stdout}\nfirst proxy stderr:\n{stderr}"
            )
        if second is not None:
            stdout, stderr = cleanup_running_proxy(second)
            diagnostics.append(
                f"second proxy stdout:\n{stdout}\nsecond proxy stderr:\n{stderr}"
            )
    if failure is not None:
        raise AssertionError(f"{failure}\n" + "\n".join(diagnostics)) from failure

    assert app.return_value == AppResult(0)


@pytest.mark.asyncio
async def test_missing_performance_capability_returns_fixed_safe_failure(
    tmp_path: Path,
) -> None:
    mapping_path, executable = _prepare_proxy(tmp_path)
    raw_marker = "raw provider body Authorization=secret"
    with proxy_process(tmp_path, mapping_path, executable) as running:
        app = TuiApp(running.socket_path)
        result = await asyncio.wait_for(
            asyncio.to_thread(app.run, headless=True),
            timeout=_WAIT_TIMEOUT_SECONDS,
        )

    assert result == AppResult(1, _PROTOCOL_FAILURE)
    assert "protocol v1" in result.message
    assert "performance" in result.message
    assert "performance_events" in result.message
    assert raw_marker not in result.message
    assert raw_marker not in repr(result)

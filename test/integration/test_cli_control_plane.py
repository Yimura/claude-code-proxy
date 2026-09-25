from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time
from typing import cast

import httpx
from pydantic import TypeAdapter
import pytest

from claude_code_proxy.control.schemas import (
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
)
from test.integration import control_plane_support
from test.integration.control_plane_support import (
    PROCESS_TIMEOUT_SECONDS as _PROCESS_TIMEOUT_SECONDS,
    RunningProxy as _RunningProxy,
    StartupExit as _StartupExit,
    StartupTimeout as _StartupTimeout,
    cleanup_running_proxy as _cleanup_running_proxy,
    cli_executable as _cli_executable,
    control_get as _control_get,
    launch_proxy_attempt as _launch_proxy_attempt,
    proxy_process as _proxy_process,
    public_get as _public_get,
    start_proxy_with_retries as _start_proxy_with_retries,
    stop_process as _stop_process,
    write_mapping as _write_mapping,
)


def _assert_http_contract(running: _RunningProxy) -> None:
    with httpx.Client(timeout=0.5, trust_env=False) as public_client:
        assert _public_get(public_client, running.port, "/").json() == {
            "message": "Anthropic Proxy for LiteLLM"
        }
        assert _public_get(
            public_client, running.port, "/v1/health"
        ).status_code == 404
        assert _public_get(
            public_client, running.port, "/v1/sessions"
        ).status_code == 404

    health = _control_get(running.socket_path, "/v1/health")
    sessions = _control_get(running.socket_path, "/v1/sessions")
    assert health.status_code == sessions.status_code == 200
    assert health.json()["protocol_version"] == 1
    assert health.json()["capabilities"] == ["sessions", "agents"]
    assert health.json()["pid"] == running.process.pid
    assert health.json()["inactive_limit"] == 2
    assert health.json()["sessions"] == {"active": 0, "retained": 0}
    assert sessions.json()["sessions"] == []
    assert _control_get(
        running.socket_path, "/v1/performance"
    ).status_code == 404
    assert _control_get(
        running.socket_path, "/v1/performance/events"
    ).status_code == 404


def _run_ps(
    tmp_path: Path,
    executable: Path,
    running: _RunningProxy,
    output_format: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(executable),
            "ps",
            "--socket",
            str(running.socket_path),
            "--format",
            output_format,
        ],
        cwd=tmp_path,
        env=running.environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=_PROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _run_perf(
    tmp_path: Path,
    executable: Path,
    running: _RunningProxy,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(executable),
            "perf",
            "--socket",
            str(running.socket_path),
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=running.environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=_PROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _assert_perf_watch_reset_and_signal_shutdown(
    tmp_path: Path,
    executable: Path,
    running: _RunningProxy,
) -> None:
    process = subprocess.Popen(
        [
            str(executable),
            "perf",
            "--watch",
            "--format",
            "json",
            "--socket",
            str(running.socket_path),
        ],
        cwd=tmp_path,
        env=running.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        assert selector.select(timeout=_PROCESS_TIMEOUT_SECONDS), (
            "perf --watch did not emit its initial reset before the deadline"
        )
        reset = PerformanceResetResponse.model_validate_json(
            process.stdout.readline()
        )
        assert reset.type == "reset"
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=_PROCESS_TIMEOUT_SECONDS) == 0
        _, stderr = process.communicate(timeout=1)
        assert stderr == ""
    finally:
        selector.close()
        _stop_process(process)


def _assert_ps_is_empty(
    tmp_path: Path,
    executable: Path,
    running: _RunningProxy,
) -> None:
    completed = _run_ps(tmp_path, executable, running, "json")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "[]\n"
    assert completed.stderr == ""


def _assert_clean_shutdown(running: _RunningProxy) -> None:
    running.process.send_signal(signal.SIGTERM)
    assert running.process.wait(timeout=_PROCESS_TIMEOUT_SECONDS) == 0
    assert not running.socket_path.exists()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", running.port), timeout=0.2)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _PollingProcess:
    def __init__(self, pid: int, polls: list[int | None]) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self._polls = iter(polls)

    def poll(self) -> int | None:
        try:
            self.returncode = next(self._polls)
        except StopIteration:
            pass
        return self.returncode


def _collision_proxies(
    tmp_path: Path,
) -> tuple[_RunningProxy, _RunningProxy, _PollingProcess]:
    first_process = _PollingProcess(1001, [None, 1])
    first = _RunningProxy(
        cast(subprocess.Popen[str], first_process),
        {},
        tmp_path / "first.sock",
        10001,
    )
    second = _RunningProxy(
        cast(subprocess.Popen[str], _PollingProcess(1002, [None, None])),
        {},
        tmp_path / "second.sock",
        10002,
    )
    return first, second, first_process


def test_responsive_port_collision_exits_during_stabilization_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[int] = []
    cleaned: list[_RunningProxy] = []
    first, second, first_process = _collision_proxies(tmp_path)
    by_socket = {first.socket_path: first, second.socket_path: second}

    def launch(_tmp_path, _mapping_path, _executable, attempt):
        starts.append(attempt)
        return first if attempt == 1 else second

    def control_get(socket_path, _path):
        return _FakeResponse({"pid": by_socket[socket_path].process.pid})

    def cleanup(running):
        cleaned.append(running)
        if running is first:
            return "", "[Errno 98] address already in use"
        return "", ""

    module = control_plane_support
    monkeypatch.setattr(module, "launch_proxy_attempt", launch)
    monkeypatch.setattr(
        module,
        "public_get",
        lambda _client, _port, _path: _FakeResponse(
            {"message": "Anthropic Proxy for LiteLLM"}
        ),
    )
    monkeypatch.setattr(module, "control_get", control_get)
    monkeypatch.setattr(module, "cleanup_running_proxy", cleanup)
    monkeypatch.setattr(module, "STARTUP_STABILITY_SECONDS", 0)

    running = _start_proxy_with_retries(
        tmp_path,
        tmp_path / "models.json",
        tmp_path / "claude-code-proxy",
        max_attempts=2,
    )
    control_plane_support.cleanup_running_proxy(running)

    assert running is second
    assert starts == [1, 2]
    assert cleaned == [first, second]
    assert first_process.poll() == 1


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(17)])
def test_startup_base_exception_cleans_once_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    starts: list[int] = []
    cleaned: list[_RunningProxy] = []
    running = _RunningProxy(
        cast(subprocess.Popen[str], object()), {}, tmp_path / "only.sock", 10001
    )

    def launch(_tmp_path, _mapping_path, _executable, attempt):
        starts.append(attempt)
        return running

    def wait(_running):
        raise interruption

    def cleanup(cleaned_running):
        cleaned.append(cleaned_running)
        return "", ""

    module = control_plane_support
    monkeypatch.setattr(module, "launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "wait_for_both_servers", wait)
    monkeypatch.setattr(module, "cleanup_running_proxy", cleanup)

    with pytest.raises(type(interruption)) as caught:
        _start_proxy_with_retries(
            tmp_path,
            tmp_path / "models.json",
            tmp_path / "claude-code-proxy",
            max_attempts=2,
        )

    assert caught.value is interruption
    assert starts == [1]
    assert cleaned == [running]


def test_startup_retries_after_address_in_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    cleaned: list[_RunningProxy] = []
    first = _RunningProxy(
        cast(subprocess.Popen[str], object()), {}, tmp_path / "first.sock", 10001
    )
    second = _RunningProxy(
        cast(subprocess.Popen[str], object()), {}, tmp_path / "second.sock", 10002
    )

    def launch(_tmp_path, _mapping_path, _executable, attempt):
        attempts.append(attempt)
        return first if attempt == 1 else second

    def wait(running):
        if running is first:
            raise _StartupExit("proxy exited during startup with 1")

    def cleanup(running):
        cleaned.append(running)
        return "", "[Errno 98] address already in use"

    module = control_plane_support
    monkeypatch.setattr(module, "launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "wait_for_both_servers", wait)
    monkeypatch.setattr(module, "cleanup_running_proxy", cleanup)

    running = _start_proxy_with_retries(
        tmp_path,
        tmp_path / "models.json",
        tmp_path / "claude-code-proxy",
        max_attempts=2,
    )

    assert running is second
    assert attempts == [1, 2]
    assert cleaned == [first]


@pytest.mark.parametrize(
    ("startup_error", "stderr"),
    [
        (_StartupExit("proxy exited during startup with 2"), "fatal startup error"),
        (_StartupTimeout("readiness timed out"), "[Errno 98] address already in use"),
    ],
)
def test_startup_does_not_retry_non_address_exit_or_live_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    startup_error: Exception,
    stderr: str,
) -> None:
    attempts: list[int] = []
    cleaned: list[_RunningProxy] = []
    running = _RunningProxy(
        cast(subprocess.Popen[str], object()), {}, tmp_path / "only.sock", 10001
    )

    def launch(_tmp_path, _mapping_path, _executable, attempt):
        attempts.append(attempt)
        return running

    def wait(_running):
        raise startup_error

    def cleanup(cleaned_running):
        cleaned.append(cleaned_running)
        return "", stderr

    module = control_plane_support
    monkeypatch.setattr(module, "launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "wait_for_both_servers", wait)
    monkeypatch.setattr(module, "cleanup_running_proxy", cleanup)

    with pytest.raises(AssertionError, match="attempt 1"):
        _start_proxy_with_retries(
            tmp_path,
            tmp_path / "models.json",
            tmp_path / "claude-code-proxy",
            max_attempts=2,
        )

    assert attempts == [1]
    assert cleaned == [running]


def test_foreground_proxy_serves_isolated_public_and_control_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    fake_marker = tmp_path / "fake-cli-used"
    fake_cli = hostile_bin / "claude-code-proxy"
    fake_cli.write_text(
        "#!/bin/sh\ntouch \"$FAKE_CLI_MARKER\"\nprintf '[]\\n'\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    monkeypatch.setenv("FAKE_CLI_MARKER", str(fake_marker))
    monkeypatch.setenv(
        "PATH",
        f"{hostile_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    )

    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()
    executable = _cli_executable()

    with _proxy_process(tmp_path, mapping_path, executable) as running:
        _assert_http_contract(running)
        _assert_ps_is_empty(tmp_path, executable, running)
        assert not fake_marker.exists(), "ambient PATH executable was invoked"
        _assert_clean_shutdown(running)


def _assert_performance_stream(
    running: _RunningProxy,
    snapshot: PerformanceListResponse,
    headers: dict[str, str],
    body: dict[str, object],
) -> None:
    transport = httpx.HTTPTransport(uds=str(running.socket_path))
    with httpx.Client(
        transport=transport,
        base_url="http://control",
        timeout=httpx.Timeout(2.0),
        trust_env=False,
    ) as control_client:
        with control_client.stream("GET", "/v1/performance/events") as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith(
                "application/x-ndjson"
            )
            lines = stream.iter_lines()
            reset = PerformanceResetResponse.model_validate_json(next(lines))
            assert reset.sequence == snapshot.cursor
            with httpx.Client(timeout=5.0, trust_env=False) as public_client:
                response = public_client.post(
                    f"http://127.0.0.1:{running.port}/v1/messages/count_tokens",
                    headers=headers,
                    json=body,
                )
            assert response.status_code == 200
            events: list[PerformanceEventResponse] = []
            for line in lines:
                if not line:
                    continue
                event = TypeAdapter(PerformanceStreamEvent).validate_json(line)
                assert isinstance(event, PerformanceEventResponse)
                events.append(event)
                if event.type == "completed":
                    break
            assert [event.sequence for event in events] == list(
                range(reset.sequence + 1, reset.sequence + len(events) + 1)
            )
            assert {event.type for event in events} >= {
                "request_started",
                "completed",
            }
            assert all(event.process == reset.process for event in events)


def test_performance_snapshot_and_events_stream_over_control_uds(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()
    executable = _cli_executable()
    headers = {"x-claude-code-session-id": "stream-integration-session"}
    body = {
        "model": "claude-haiku",
        "messages": [{"role": "user", "content": "count me"}],
    }

    with _proxy_process(
        tmp_path, mapping_path, executable, performance="collector"
    ) as running:
        health = _control_get(running.socket_path, "/v1/health")
        assert health.json()["capabilities"] == [
            "sessions", "agents", "performance", "performance_events"
        ]
        response = _control_get(running.socket_path, "/v1/performance")
        assert response.status_code == 200
        snapshot = PerformanceListResponse.model_validate(response.json())
        _assert_performance_stream(running, snapshot, headers, body)

        perf_result = _run_perf(tmp_path, executable, running)
        assert perf_result.returncode == 0, perf_result.stderr
        perf_payload = json.loads(perf_result.stdout)
        assert perf_payload["sessions"][0]["performance"][
            "latest_request"
        ]["operation"] == "count_tokens"
        _assert_perf_watch_reset_and_signal_shutdown(
            tmp_path, executable, running
        )
        assert _control_get(running.socket_path, "/v1/health").status_code == 200
        assert running.process.poll() is None
        _assert_clean_shutdown(running)
        stdout, stderr = running.process.communicate(timeout=1)
        assert "performance outcome=" not in stdout + stderr


def test_default_mode_keeps_perf_disabled_and_terminal_logs_quiet(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()
    executable = _cli_executable()
    running = _start_proxy_with_retries(tmp_path, mapping_path, executable)
    stdout = stderr = ""
    try:
        _assert_http_contract(running)
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.post(
                f"http://127.0.0.1:{running.port}/v1/messages/count_tokens",
                headers={"x-claude-code-session-id": "default-off-session"},
                json={
                    "model": "claude-haiku",
                    "messages": [{"role": "user", "content": "count me"}],
                },
            )
        assert response.status_code == 200
        sessions = _control_get(running.socket_path, "/v1/sessions").json()
        assert sessions["sessions"][0]["requests"] == 1

        perf_result = _run_perf(tmp_path, executable, running)
        assert perf_result.returncode == 1
        assert "proxy --performance collector" in perf_result.stderr
        _assert_clean_shutdown(running)
    finally:
        stdout, stderr = _cleanup_running_proxy(running)

    assert "performance outcome=" not in stdout + stderr


def test_logging_mode_emits_one_terminal_performance_record(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()
    executable = _cli_executable()
    running = _start_proxy_with_retries(
        tmp_path, mapping_path, executable, performance="logging"
    )
    stdout = stderr = ""
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.post(
                f"http://127.0.0.1:{running.port}/v1/messages/count_tokens",
                json={
                    "model": "claude-haiku",
                    "messages": [{"role": "user", "content": "count me"}],
                },
            )
        assert response.status_code == 200
        _assert_clean_shutdown(running)
    finally:
        stdout, stderr = _cleanup_running_proxy(running)

    assert (stdout + stderr).count("performance outcome=completed") == 1


def test_agent_identity_reaches_control_json_and_nested_ps(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()
    executable = _cli_executable()
    headers = {
        "x-claude-code-session-id": "integration-session",
        "x-claude-code-agent-id": "integration-agent",
        "x-claude-code-parent-agent-id": "integration-parent",
    }
    body = {
        "model": "claude-haiku",
        "messages": [{"role": "user", "content": "count me"}],
    }

    with _proxy_process(tmp_path, mapping_path, executable) as running:
        with httpx.Client(timeout=5.0, trust_env=False) as public_client:
            response = public_client.post(
                f"http://127.0.0.1:{running.port}/v1/messages/count_tokens",
                headers=headers,
                json=body,
            )
        assert response.status_code == 200

        json_result = _run_ps(tmp_path, executable, running, "json")
        table_result = _run_ps(tmp_path, executable, running, "table")

    assert json_result.returncode == 0, json_result.stderr
    payload = json.loads(json_result.stdout)
    assert len(payload) == 1
    assert payload[0]["requests"] == 1
    assert len(payload[0]["agents"]) == 1
    assert payload[0]["agents"][0]["requests"] == 1
    assert payload[0]["agents"][0]["parent_id"] is not None
    assert table_result.returncode == 0, table_result.stderr
    assert "SESSION / AGENT" in table_result.stdout
    assert "└─ " in table_result.stdout
    combined = json_result.stdout + table_result.stdout
    assert "integration-session" not in combined
    assert "integration-agent" not in combined
    assert "integration-parent" not in combined


def test_proxy_startup_uses_bundled_cost_map_without_remote_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counting_http_server,
) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_URL", counting_http_server.url)
    mapping_path = tmp_path / "models.json"
    _write_mapping(mapping_path)
    (tmp_path / "home").mkdir()

    started = time.monotonic()
    running = _start_proxy_with_retries(
        tmp_path,
        mapping_path,
        _cli_executable(),
    )
    ready_in = time.monotonic() - started
    try:
        assert ready_in < 5.0
        assert counting_http_server.request_count == 0
        _assert_http_contract(running)
        _assert_clean_shutdown(running)
    finally:
        stdout, stderr = _cleanup_running_proxy(running)

    output = f"{stdout}\n{stderr}".casefold()
    assert "model cost map" not in output
    assert "failed to fetch remote" not in output
    assert counting_http_server.request_count == 0

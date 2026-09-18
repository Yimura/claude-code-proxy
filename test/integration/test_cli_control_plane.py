from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import cast

import httpx
import pytest

_STARTUP_TIMEOUT_SECONDS = 15.0
_STARTUP_STABILITY_SECONDS = 0.5
_PROCESS_TIMEOUT_SECONDS = 10.0
_STARTUP_ATTEMPTS = 5
_ADDRESS_IN_USE_MARKERS = (
    "address already in use",
    "eaddrinuse",
    "errno 98",
    "errno 48",
    "winerror 10048",
)


class _StartupExit(RuntimeError):
    pass


class _StartupTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class _RunningProxy:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    socket_path: Path
    port: int


def _cli_executable() -> Path:
    executable = Path(sys.executable).with_name("claude-code-proxy")
    assert executable.is_file(), (
        f"console script not found beside current Python: {executable}"
    )
    assert os.access(executable, os.X_OK), (
        f"console script is not executable: {executable}"
    )
    return executable


def _reserve_loopback_port() -> tuple[socket.socket, int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    return listener, listener.getsockname()[1]


def _isolated_environment(
    tmp_path: Path,
    socket_path: Path,
    mapping_path: Path,
    runtime_path: Path,
    port: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    sensitive_fragments = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    for name in tuple(environment):
        if any(fragment in name.upper() for fragment in sensitive_fragments):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CONTROL_SOCKET_PATH": str(socket_path),
            "HOME": str(tmp_path / "home"),
            "MODEL_MAPPING_PATH": str(mapping_path),
            "OPENCODE_DATA_DIR": str(runtime_path),
            "OPENAI_TRANSPORT": "litellm",
            "PROXY_HOST": "127.0.0.1",
            "PROXY_PORT": str(port),
            "SESSION_RETENTION_LIMIT": "2",
            "USE_VERTEX_AUTH": "False",
            "VERTEX_LOCATION": "unset",
            "VERTEX_PROJECT": "unset",
        }
    )
    return environment


def _public_get(client: httpx.Client, port: int, path: str) -> httpx.Response:
    return client.get(f"http://127.0.0.1:{port}{path}")


def _control_get(socket_path: Path, path: str) -> httpx.Response:
    transport = httpx.HTTPTransport(uds=str(socket_path))
    with httpx.Client(
        transport=transport,
        base_url="http://control",
        timeout=0.5,
        trust_env=False,
    ) as client:
        return client.get(path)


def _require_running_process(running: _RunningProxy) -> None:
    if running.process.poll() is not None:
        raise _StartupExit(
            f"proxy exited during startup with {running.process.returncode}"
        )


def _public_is_ready(client: httpx.Client, running: _RunningProxy) -> bool:
    response = _public_get(client, running.port, "/")
    return response.status_code == 200 and response.json() == {
        "message": "Anthropic Proxy for LiteLLM"
    }


def _control_is_owned_by_child(running: _RunningProxy) -> bool:
    response = _control_get(running.socket_path, "/v1/health")
    payload = response.json()
    return (
        response.status_code == 200
        and isinstance(payload, dict)
        and payload.get("pid") == running.process.pid
    )


def _confirm_stable_start(
    running: _RunningProxy,
    public_client: httpx.Client,
) -> bool:
    """Catch delayed bind failure because public HTTP has no process identity."""
    deadline = time.monotonic() + _STARTUP_STABILITY_SECONDS
    while time.monotonic() < deadline:
        _require_running_process(running)
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    _require_running_process(running)
    return _control_is_owned_by_child(running) and _public_is_ready(
        public_client, running
    )


def _wait_for_both_servers(running: _RunningProxy) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    with httpx.Client(timeout=0.5, trust_env=False) as public_client:
        while time.monotonic() < deadline:
            _require_running_process(running)
            try:
                control_ready = _control_is_owned_by_child(running)
                public_ready = _public_is_ready(public_client, running)
                if control_ready and public_ready:
                    if _confirm_stable_start(running, public_client):
                        return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.05)
    raise _StartupTimeout("proxy did not make both servers ready before timeout")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_PROCESS_TIMEOUT_SECONDS)


def _cleanup_running_proxy(running: _RunningProxy) -> tuple[str, str]:
    _stop_process(running.process)
    stdout, stderr = running.process.communicate(timeout=1)
    running.socket_path.unlink(missing_ok=True)
    return stdout, stderr


def _launch_proxy_attempt(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
    attempt: int,
) -> _RunningProxy:
    attempt_path = tmp_path / f"proxy-attempt-{attempt}"
    attempt_path.mkdir(mode=0o700)
    attempt_path.chmod(0o700)
    socket_path = attempt_path / "control.sock"
    reservation, port = _reserve_loopback_port()
    environment = _isolated_environment(
        tmp_path,
        socket_path,
        mapping_path,
        attempt_path / "opencode",
        port,
    )
    reservation.close()
    process = subprocess.Popen(
        [str(executable), "proxy"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return _RunningProxy(process, environment, socket_path, port)


def _address_in_use(stdout: str, stderr: str) -> bool:
    output = f"{stdout}\n{stderr}".casefold()
    return any(marker in output for marker in _ADDRESS_IN_USE_MARKERS)


def _startup_diagnostic(
    attempt: int,
    error: Exception,
    stdout: str,
    stderr: str,
) -> str:
    return (
        f"attempt {attempt}: {error}\n"
        f"proxy stdout:\n{stdout}\nproxy stderr:\n{stderr}"
    )


def _start_proxy_with_retries(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
    max_attempts: int = _STARTUP_ATTEMPTS,
) -> _RunningProxy:
    address_conflicts: list[str] = []
    for attempt in range(1, max_attempts + 1):
        running = _launch_proxy_attempt(
            tmp_path,
            mapping_path,
            executable,
            attempt,
        )
        try:
            _wait_for_both_servers(running)
            return running
        except _StartupExit as error:
            stdout, stderr = _cleanup_running_proxy(running)
            diagnostic = _startup_diagnostic(attempt, error, stdout, stderr)
            if _address_in_use(stdout, stderr):
                address_conflicts.append(diagnostic)
                continue
            raise AssertionError(diagnostic) from error
        except Exception as error:
            stdout, stderr = _cleanup_running_proxy(running)
            diagnostic = _startup_diagnostic(attempt, error, stdout, stderr)
            raise AssertionError(diagnostic) from error
        except BaseException as error:
            try:
                _cleanup_running_proxy(running)
            except BaseException as cleanup_error:
                error.add_note(f"startup cleanup failure: {cleanup_error!r}")
            raise
    raise AssertionError(
        "proxy exhausted startup retries after address conflicts:\n"
        + "\n".join(address_conflicts)
    )


@contextmanager
def _proxy_process(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
) -> Iterator[_RunningProxy]:
    running = _start_proxy_with_retries(tmp_path, mapping_path, executable)
    failure: Exception | None = None
    try:
        yield running
    except Exception as error:
        failure = error
    finally:
        stdout, stderr = _cleanup_running_proxy(running)
    if failure is not None:
        raise AssertionError(
            f"{failure}\nproxy stdout:\n{stdout}\nproxy stderr:\n{stderr}"
        ) from failure


def _write_mapping(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "models": {
                    "smoke": {
                        "target": "openai/smoke-model",
                        "context_window": 1000,
                    }
                },
                "tiers": {"small": "smoke"},
                "mappings": {"haiku": {"tier": "small", "effort": "medium"}},
            }
        ),
        encoding="utf-8",
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


def test_responsive_port_collision_exits_during_stabilization_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[int] = []
    cleaned: list[_RunningProxy] = []
    first_process = _PollingProcess(1001, [None, 1])
    second_process = _PollingProcess(1002, [None, None])
    first = _RunningProxy(
        cast(subprocess.Popen[str], first_process),
        {},
        tmp_path / "first.sock",
        10001,
    )
    second = _RunningProxy(
        cast(subprocess.Popen[str], second_process),
        {},
        tmp_path / "second.sock",
        10002,
    )
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

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_launch_proxy_attempt", launch)
    monkeypatch.setattr(
        module,
        "_public_get",
        lambda _client, _port, _path: _FakeResponse(
            {"message": "Anthropic Proxy for LiteLLM"}
        ),
    )
    monkeypatch.setattr(module, "_control_get", control_get)
    monkeypatch.setattr(module, "_cleanup_running_proxy", cleanup)
    monkeypatch.setattr(module, "_STARTUP_STABILITY_SECONDS", 0)

    running = _start_proxy_with_retries(
        tmp_path,
        tmp_path / "models.json",
        tmp_path / "claude-code-proxy",
        max_attempts=2,
    )
    _cleanup_running_proxy(running)

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

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "_wait_for_both_servers", wait)
    monkeypatch.setattr(module, "_cleanup_running_proxy", cleanup)

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

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "_wait_for_both_servers", wait)
    monkeypatch.setattr(module, "_cleanup_running_proxy", cleanup)

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

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_launch_proxy_attempt", launch)
    monkeypatch.setattr(module, "_wait_for_both_servers", wait)
    monkeypatch.setattr(module, "_cleanup_running_proxy", cleanup)

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

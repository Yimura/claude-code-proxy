from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

STARTUP_TIMEOUT_SECONDS = 15.0
STARTUP_STABILITY_SECONDS = 0.5
PROCESS_TIMEOUT_SECONDS = 10.0
STARTUP_ATTEMPTS = 5
_ADDRESS_IN_USE_MARKERS = (
    "address already in use",
    "eaddrinuse",
    "errno 98",
    "errno 48",
    "winerror 10048",
)


class StartupExit(RuntimeError):
    pass


class StartupTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class RunningProxy:
    process: subprocess.Popen[str]
    environment: dict[str, str]
    socket_path: Path
    port: int


def cli_executable() -> Path:
    executable = Path(sys.executable).with_name("claude-code-proxy")
    assert executable.is_file(), (
        f"console script not found beside current Python: {executable}"
    )
    assert os.access(executable, os.X_OK), (
        f"console script is not executable: {executable}"
    )
    return executable


def reserve_loopback_port() -> tuple[socket.socket, int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    return listener, listener.getsockname()[1]


def isolated_environment(
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


def public_get(client: httpx.Client, port: int, path: str) -> httpx.Response:
    return client.get(f"http://127.0.0.1:{port}{path}")


def control_get(socket_path: Path, path: str) -> httpx.Response:
    transport = httpx.HTTPTransport(uds=str(socket_path))
    with httpx.Client(
        transport=transport,
        base_url="http://control",
        timeout=0.5,
        trust_env=False,
    ) as client:
        return client.get(path)


def require_running_process(running: RunningProxy) -> None:
    if running.process.poll() is not None:
        raise StartupExit(
            f"proxy exited during startup with {running.process.returncode}"
        )


def public_is_ready(client: httpx.Client, running: RunningProxy) -> bool:
    response = public_get(client, running.port, "/")
    return response.status_code == 200 and response.json() == {
        "message": "Anthropic Proxy for LiteLLM"
    }


def control_is_owned_by_child(running: RunningProxy) -> bool:
    response = control_get(running.socket_path, "/v1/health")
    payload = response.json()
    return (
        response.status_code == 200
        and isinstance(payload, dict)
        and payload.get("pid") == running.process.pid
    )


def confirm_stable_start(
    running: RunningProxy,
    public_client: httpx.Client,
) -> bool:
    """Catch delayed bind failure because public HTTP has no process identity."""
    deadline = time.monotonic() + STARTUP_STABILITY_SECONDS
    while time.monotonic() < deadline:
        require_running_process(running)
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    require_running_process(running)
    return control_is_owned_by_child(running) and public_is_ready(
        public_client, running
    )


def wait_for_both_servers(running: RunningProxy) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    with httpx.Client(timeout=0.5, trust_env=False) as public_client:
        while time.monotonic() < deadline:
            require_running_process(running)
            try:
                control_ready = control_is_owned_by_child(running)
                public_ready = public_is_ready(public_client, running)
                if control_ready and public_ready:
                    if confirm_stable_start(running, public_client):
                        return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.05)
    raise StartupTimeout("proxy did not make both servers ready before timeout")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_TIMEOUT_SECONDS)


def cleanup_running_proxy(running: RunningProxy) -> tuple[str, str]:
    stop_process(running.process)
    stdout, stderr = running.process.communicate(timeout=1)
    running.socket_path.unlink(missing_ok=True)
    return stdout, stderr


def launch_proxy_attempt(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
    attempt: int,
    *,
    performance: str | None = None,
    socket_path: Path | None = None,
    port: int | None = None,
) -> RunningProxy:
    attempt_path = tmp_path / f"proxy-attempt-{attempt}"
    attempt_path.mkdir(mode=0o700, exist_ok=True)
    attempt_path.chmod(0o700)
    effective_socket = socket_path or attempt_path / "control.sock"
    if port is None:
        reservation, effective_port = reserve_loopback_port()
        reservation.close()
    else:
        effective_port = port
    environment = isolated_environment(
        tmp_path,
        effective_socket,
        mapping_path,
        attempt_path / "opencode",
        effective_port,
    )
    command = [str(executable), "proxy"]
    if performance is not None:
        command.extend(("--performance", performance))
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return RunningProxy(process, environment, effective_socket, effective_port)


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


def start_proxy_with_retries(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
    max_attempts: int = STARTUP_ATTEMPTS,
    *,
    performance: str | None = None,
    socket_path: Path | None = None,
    port: int | None = None,
) -> RunningProxy:
    address_conflicts: list[str] = []
    for attempt in range(1, max_attempts + 1):
        launch_options: dict[str, object] = {}
        if performance is not None:
            launch_options["performance"] = performance
        if socket_path is not None:
            launch_options["socket_path"] = socket_path
        if port is not None:
            launch_options["port"] = port
        running = launch_proxy_attempt(
            tmp_path,
            mapping_path,
            executable,
            attempt,
            **launch_options,
        )
        try:
            wait_for_both_servers(running)
            return running
        except StartupExit as error:
            stdout, stderr = cleanup_running_proxy(running)
            diagnostic = _startup_diagnostic(attempt, error, stdout, stderr)
            if _address_in_use(stdout, stderr):
                address_conflicts.append(diagnostic)
                continue
            raise AssertionError(diagnostic) from error
        except Exception as error:
            stdout, stderr = cleanup_running_proxy(running)
            diagnostic = _startup_diagnostic(attempt, error, stdout, stderr)
            raise AssertionError(diagnostic) from error
        except BaseException as error:
            try:
                cleanup_running_proxy(running)
            except BaseException as cleanup_error:
                error.add_note(f"startup cleanup failure: {cleanup_error!r}")
            raise
    raise AssertionError(
        "proxy exhausted startup retries after address conflicts:\n"
        + "\n".join(address_conflicts)
    )


@contextmanager
def proxy_process(
    tmp_path: Path,
    mapping_path: Path,
    executable: Path,
    *,
    performance: str | None = None,
    socket_path: Path | None = None,
    port: int | None = None,
) -> Iterator[RunningProxy]:
    running = start_proxy_with_retries(
        tmp_path,
        mapping_path,
        executable,
        performance=performance,
        socket_path=socket_path,
        port=port,
    )
    failure: Exception | None = None
    try:
        yield running
    except Exception as error:
        failure = error
    finally:
        stdout, stderr = cleanup_running_proxy(running)
    if failure is not None:
        raise AssertionError(
            f"{failure}\nproxy stdout:\n{stdout}\nproxy stderr:\n{stderr}"
        ) from failure


def write_mapping(path: Path) -> None:
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

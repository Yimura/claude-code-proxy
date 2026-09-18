"""Regression coverage for dependency-free CLI and offline startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


_PROCESS_TIMEOUT_SECONDS = 3.0
_PROVIDER_IMPORT_TIMEOUT_SECONDS = 15.0
_FETCH_MESSAGE = "model cost map"
_TRACKED_IMPORTS = (
    "litellm",
    "claude_code_proxy.runtime",
    "claude_code_proxy.server",
    "claude_code_proxy.app",
    "claude_code_proxy.providers",
    "claude_code_proxy.providers.litellm",
)


def _cli_executable() -> Path:
    executable = Path(sys.executable).with_name("claude-code-proxy")
    assert executable.is_file()
    return executable


def _offline_environment(server_url: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    environment["LITELLM_MODEL_COST_MAP_URL"] = server_url
    return environment


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    cwd: Path,
    timeout: float = _PROCESS_TIMEOUT_SECONDS,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=timeout,
        check=False,
    )
    return completed, time.monotonic() - started


def _assert_no_fetch_output(completed: subprocess.CompletedProcess[str]) -> None:
    output = f"{completed.stdout}\n{completed.stderr}".casefold()
    assert _FETCH_MESSAGE not in output
    assert "failed to fetch remote" not in output


def test_importing_cli_does_not_load_runtime_or_provider_modules(
    tmp_path: Path,
    counting_http_server,
) -> None:
    script = """
import importlib
import json
import sys

importlib.import_module("claude_code_proxy.cli")
tracked = json.loads(sys.argv[1])
print(json.dumps({name: name in sys.modules for name in tracked}, sort_keys=True))
"""
    completed, elapsed = _run(
        [sys.executable, "-c", script, json.dumps(_TRACKED_IMPORTS)],
        environment=_offline_environment(counting_http_server.url),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert elapsed < _PROCESS_TIMEOUT_SECONDS
    assert json.loads(completed.stdout) == {
        name: False for name in sorted(_TRACKED_IMPORTS)
    }
    _assert_no_fetch_output(completed)
    assert counting_http_server.request_count == 0


@pytest.mark.parametrize(
    "arguments",
    [(), ("--help",), ("proxy", "--help"), ("ps", "--help")],
)
def test_help_commands_exit_quickly_without_loading_remote_cost_map(
    tmp_path: Path,
    counting_http_server,
    arguments: tuple[str, ...],
) -> None:
    completed, elapsed = _run(
        [str(_cli_executable()), *arguments],
        environment=_offline_environment(counting_http_server.url),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert elapsed < _PROCESS_TIMEOUT_SECONDS
    _assert_no_fetch_output(completed)
    assert counting_http_server.request_count == 0


def test_ps_missing_socket_exits_quickly_without_remote_cost_map(
    tmp_path: Path,
    counting_http_server,
) -> None:
    completed, elapsed = _run(
        [
            str(_cli_executable()),
            "ps",
            "--socket",
            str(tmp_path / "missing.sock"),
        ],
        environment=_offline_environment(counting_http_server.url),
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    assert elapsed < _PROCESS_TIMEOUT_SECONDS
    assert "Control API unavailable" in completed.stderr
    _assert_no_fetch_output(completed)
    assert counting_http_server.request_count == 0


def test_provider_import_defaults_to_bundled_cost_map(
    tmp_path: Path,
    counting_http_server,
) -> None:
    script = """
import importlib
import os
import sys

os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
sys.modules.pop("litellm", None)
sys.modules.pop("claude_code_proxy.providers.litellm", None)
importlib.import_module("claude_code_proxy.providers.litellm")
print(os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"))
"""
    completed, elapsed = _run(
        [sys.executable, "-c", script],
        environment=_offline_environment(counting_http_server.url),
        cwd=tmp_path,
        timeout=_PROVIDER_IMPORT_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    assert elapsed < _PROVIDER_IMPORT_TIMEOUT_SECONDS
    assert completed.stdout == "True\n"
    _assert_no_fetch_output(completed)
    assert counting_http_server.request_count == 0


@pytest.mark.parametrize("operator_value", ["TRUE", "false"])
def test_provider_import_preserves_operator_cost_map_override(
    tmp_path: Path,
    operator_value: str,
) -> None:
    script = """
import importlib
import os
import sys
import types

sys.modules["litellm"] = types.ModuleType("litellm")
importlib.import_module("claude_code_proxy.providers.litellm")
print(os.environ["LITELLM_LOCAL_MODEL_COST_MAP"])
"""
    environment = os.environ.copy()
    environment["LITELLM_LOCAL_MODEL_COST_MAP"] = operator_value
    completed, elapsed = _run(
        [sys.executable, "-c", script],
        environment=environment,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert elapsed < _PROCESS_TIMEOUT_SECONDS
    assert completed.stdout == f"{operator_value}\n"

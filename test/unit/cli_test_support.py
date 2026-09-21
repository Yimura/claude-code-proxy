"""Shared builders and process helpers for CLI tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import shutil
import socketserver
import subprocess
import sys
import threading

from typer.testing import CliRunner

from claude_code_proxy.config import Settings
from claude_code_proxy.control.schemas import (
    AgentResponse,
    SessionListResponse,
    SessionResponse,
)

runner = CliRunner()
ROOT = Path(__file__).parents[2]
CAPTURED_AT = datetime(2026, 1, 2, 12, tzinfo=UTC)


def settings(tmp_path: Path, **overrides: object) -> Settings:
    configured = Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
        vertex_project="unset",
        vertex_location="unset",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="codex",
        opencode_data_dir=tmp_path / "opencode",
        model_mapping_path=tmp_path / "models.json",
        proxy_host="env-host",
        proxy_port=9000,
        control_socket_path=tmp_path / "env.sock",
        session_retention_limit=27,
    )
    return replace(configured, **overrides)


def session(
    identifier: str,
    *,
    state: str = "idle",
    client_model: str = "claude-opus",
    model: str = "gpt-5.6-sol",
    effort: str = "high",
    context_window: int | None = 1_000_000,
    active_requests: int = 0,
    requests: int = 3,
    seconds_ago: int = 0,
    agents: tuple[AgentResponse, ...] = (),
) -> SessionResponse:
    return SessionResponse(
        id=identifier,
        state=state,
        active_requests=active_requests,
        requests=requests,
        client_model=client_model,
        model=model,
        provider="openai",
        transport="codex",
        effort=effort,
        context_window=context_window,
        first_seen=CAPTURED_AT - timedelta(hours=2),
        last_seen=CAPTURED_AT - timedelta(seconds=seconds_ago),
        elapsed_seconds=7200,
        last_result="completed",
        agents=agents,
    )


def agent(
    identifier: str,
    *,
    parent_id: str | None = None,
    seconds_ago: int = 0,
    state: str = "idle",
    requests: int = 1,
) -> AgentResponse:
    observed = CAPTURED_AT - timedelta(seconds=seconds_ago)
    return AgentResponse(
        id=identifier,
        parent_id=parent_id,
        state=state,
        active_requests=1 if state == "active" else 0,
        requests=requests,
        client_model="claude-sonnet",
        model="gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
        first_seen=observed,
        last_seen=observed,
        elapsed_seconds=1.0,
        last_result=None if state == "active" else "completed",
    )


def response(*sessions: SessionResponse) -> SessionListResponse:
    return SessionListResponse(captured_at=CAPTURED_AT, sessions=sessions)


def run_ps_subprocess(
    socket_path: Path,
    payloads: dict[str, object],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            request = b""
            self.request.settimeout(5)
            while b"\r\n\r\n" not in request:
                chunk = self.request.recv(4096)
                if not chunk:
                    return
                request += chunk
            path = request.split(b" ", 2)[1].decode("ascii").split("?", 1)[0]
            body = json.dumps(payloads[path]).encode("utf-8")
            headers = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
            )
            self.request.sendall(headers + body)

    server = socketserver.UnixStreamServer(str(socket_path), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    installed = shutil.which("claude-code-proxy")
    if installed is None:
        installed = str(Path(sys.executable).with_name("claude-code-proxy"))
    try:
        completed = subprocess.run(
            [installed, "ps", "--socket", str(socket_path), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        socket_path.unlink(missing_ok=True)

    assert not thread.is_alive()
    return completed


class FakeClient:
    instances: list["FakeClient"] = []
    result = response()
    error: Exception | None = None

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.filters: tuple[str, ...] | None = None
        self.entered = False
        self.exited = False
        type(self).instances.append(self)

    def __enter__(self) -> "FakeClient":
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True

    def sessions(self, filters: tuple[str, ...]) -> SessionListResponse:
        self.filters = filters
        if type(self).error is not None:
            raise type(self).error
        return type(self).result

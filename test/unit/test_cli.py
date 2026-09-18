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
import tomllib

import pytest
from click import unstyle
from wcwidth import wcswidth
from typer.testing import CliRunner

from claude_code_proxy import cli as cli_module
import claude_code_proxy.control.socket as socket_module
from claude_code_proxy.cli import app
from claude_code_proxy.config import Settings
from claude_code_proxy.control.client import (
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from claude_code_proxy.control.schemas import AgentResponse, SessionListResponse, SessionResponse


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


@pytest.fixture(autouse=True)
def reset_fake_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeClient.instances = []
    FakeClient.result = response()
    FakeClient.error = None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "ControlClient", FakeClient)


def test_bare_app_shows_help_successfully() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert "proxy" in result.stdout
    assert "ps" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["unknown"],
        ["ps", "--format", "yaml"],
        ["proxy", "--port", "0"],
        ["proxy", "--port", "65536"],
        ["proxy", "--session-limit", "-1"],
        ["proxy", "--session-limit", str(2**63)],
    ],
)
def test_invalid_command_or_option_exits_two(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2


def test_proxy_uses_environment_settings_and_calls_startup_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = settings(tmp_path)
    calls: list[tuple[str, object]] = []
    resolved = tmp_path / "resolved.sock"
    runtime = object()
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: calls.append(("settings", cls)) or configured),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda path: calls.append(("resolve", path)) or resolved,
    )
    monkeypatch.setattr(
        cli_module,
        "_configure_proxy_logging",
        lambda: calls.append(("logging", None)),
    )

    def create(value: Settings) -> object:
        calls.append(("runtime", value))
        return runtime

    def run(value: object, path: Path) -> None:
        calls.append(("run", (value, path)))

    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: calls.append(("load", None)) or (create, run),
    )

    result = runner.invoke(app, ["proxy"])

    assert result.exit_code == 0
    assert calls == [
        ("settings", cli_module.Settings),
        ("resolve", configured.control_socket_path),
        ("logging", None),
        ("load", None),
        ("runtime", configured),
        ("run", (runtime, resolved)),
    ]


def test_proxy_reports_non_linux_control_socket_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = settings(
        tmp_path, control_socket_path=tmp_path / "not-created" / "control.sock"
    )
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: configured),
    )
    monkeypatch.setattr(socket_module.sys, "platform", "darwin")
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: pytest.fail("runtime dependencies loaded on unsupported platform"),
    )

    result = runner.invoke(app, ["proxy"])

    assert result.exit_code == 1
    assert "secure control socket requires Linux" in result.stderr
    assert "use Docker on other platforms" in result.stderr
    assert "Traceback" not in result.stderr
    assert not configured.control_socket_path.parent.exists()


@pytest.mark.parametrize(
    ("arguments", "field", "expected"),
    [
        (["--host", "cli-host"], "proxy_host", "cli-host"),
        (["--port", "4123"], "proxy_port", 4123),
        (["--socket", "/cli/control.sock"], "control_socket_path", Path("/cli/control.sock")),
        (["--session-limit", "0"], "session_retention_limit", 0),
        (["--session-limit", str(2**63 - 1)], "session_retention_limit", 2**63 - 1),
    ],
)
def test_proxy_cli_options_override_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    field: str,
    expected: object,
) -> None:
    configured = settings(tmp_path)
    observed: list[Settings] = []
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: configured),
    )
    monkeypatch.setattr(cli_module, "resolve_socket_path", lambda path: path)
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: (
            lambda value: observed.append(value) or object(),
            lambda runtime, path: None,
        ),
    )

    result = runner.invoke(app, ["proxy", *arguments])

    assert result.exit_code == 0
    assert getattr(observed[0], field) == expected
    unchanged = {
        "proxy_host": configured.proxy_host,
        "proxy_port": configured.proxy_port,
        "control_socket_path": configured.control_socket_path,
        "session_retention_limit": configured.session_retention_limit,
    }
    unchanged.pop(field)
    for name, value in unchanged.items():
        assert getattr(observed[0], name) == value


@pytest.mark.parametrize(
    ("exported", "cli_socket", "expected_name"),
    [
        (None, None, "file.sock"),
        ("exported.sock", None, "exported.sock"),
        ("exported.sock", "cli.sock", "cli.sock"),
    ],
)
def test_proxy_socket_precedence_in_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exported: str | None,
    cli_socket: str | None,
    expected_name: str,
) -> None:
    file_socket = tmp_path / "file.sock"
    (tmp_path / ".env").write_text(f"CONTROL_SOCKET_PATH={file_socket}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)
    if exported is not None:
        monkeypatch.setenv("CONTROL_SOCKET_PATH", str(tmp_path / exported))

    observed_settings: list[Settings] = []
    observed_paths: list[Path] = []
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: (
            lambda value: observed_settings.append(value) or object(),
            lambda runtime, path: observed_paths.append(path),
        ),
    )
    arguments = ["proxy"]
    if cli_socket is not None:
        arguments.extend(["--socket", str(tmp_path / cli_socket)])

    result = runner.invoke(app, arguments)

    expected = (tmp_path / expected_name).absolute()
    assert result.exit_code == 0
    assert observed_settings[0].control_socket_path == expected
    assert observed_paths == [expected]


@pytest.mark.parametrize("stage", ["settings", "resolve", "load", "runtime", "run"])
def test_proxy_reports_ordinary_failures_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    configured = settings(tmp_path)

    def fail() -> None:
        raise RuntimeError("startup\nfailed")

    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: fail() if stage == "settings" else configured),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda path: fail() if stage == "resolve" else tmp_path / "control.sock",
    )
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)

    def load_runtime():
        if stage == "load":
            fail()
        return (
            lambda value: fail() if stage == "runtime" else object(),
            lambda runtime, path: fail() if stage == "run" else None,
        )

    monkeypatch.setattr(cli_module, "_load_proxy_runtime", load_runtime)

    result = runner.invoke(app, ["proxy"])

    assert result.exit_code == 1
    assert "startup\\x0afailed" in result.stderr
    assert "Traceback" not in result.stderr


def test_proxy_treats_keyboard_interrupt_as_clean_termination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: settings(tmp_path)),
    )
    monkeypatch.setattr(cli_module, "resolve_socket_path", lambda path: tmp_path / "control.sock")
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: (
            lambda value: object(),
            lambda runtime, path: (_ for _ in ()).throw(KeyboardInterrupt()),
        ),
    )

    result = runner.invoke(app, ["proxy"])

    assert result.exit_code == 0
    assert "Aborted" not in result.output


@pytest.mark.parametrize(
    ("exported", "cli_socket", "expected_name"),
    [
        (None, None, "file.sock"),
        ("exported.sock", None, "exported.sock"),
        ("exported.sock", "cli.sock", "cli.sock"),
    ],
)
def test_ps_socket_precedence_in_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exported: str | None,
    cli_socket: str | None,
    expected_name: str,
) -> None:
    file_socket = tmp_path / "file.sock"
    (tmp_path / ".env").write_text(f"CONTROL_SOCKET_PATH={file_socket}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)
    if exported is not None:
        monkeypatch.setenv("CONTROL_SOCKET_PATH", str(tmp_path / exported))

    arguments = ["ps"]
    if cli_socket is not None:
        arguments.extend(["--socket", str(tmp_path / cli_socket)])
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert FakeClient.instances[0].socket_path == (
        tmp_path / expected_name
    ).absolute()


def test_ps_loads_current_directory_dotenv_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cli_module,
        "load_dotenv",
        lambda *, dotenv_path, override: calls.append((dotenv_path, override)),
    )

    result = runner.invoke(app, ["ps", "--socket", str(tmp_path / "cli.sock")])

    assert result.exit_code == 0
    assert calls == [(tmp_path / ".env", False)]


@pytest.mark.parametrize("dotenv_contents", [None, "PROXY_PORT=9000\n"])
@pytest.mark.parametrize("command", ["proxy", "ps"])
def test_commands_ignore_ancestor_dotenv_when_cwd_dotenv_is_missing_or_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    dotenv_contents: str | None,
) -> None:
    ancestor = tmp_path / "ancestor"
    working_directory = ancestor / "working"
    working_directory.mkdir(parents=True)
    (ancestor / ".env").write_text(
        "PROXY_PORT=9001\nCONTROL_SOCKET_PATH=/ancestor.sock\n"
    )
    if dotenv_contents is not None:
        (working_directory / ".env").write_text(dotenv_contents)
    monkeypatch.chdir(working_directory)
    monkeypatch.delenv("PROXY_PORT", raising=False)
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)

    observed_socket_settings: list[Path | None] = []
    observed_proxy_settings: list[Settings] = []
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda value: observed_socket_settings.append(value)
        or working_directory / "fallback.sock",
    )
    monkeypatch.setattr(cli_module, "_configure_proxy_logging", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_load_proxy_runtime",
        lambda: (
            lambda value: observed_proxy_settings.append(value) or object(),
            lambda runtime, socket_path: None,
        ),
    )

    result = runner.invoke(app, [command])

    assert result.exit_code == 0
    assert observed_socket_settings == [None]
    if command == "proxy":
        expected_port = 9000 if dotenv_contents is not None else 8082
        assert observed_proxy_settings[0].proxy_port == expected_port


def test_ps_socket_option_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTROL_SOCKET_PATH", str(tmp_path / "env.sock"))
    result = runner.invoke(app, ["ps", "--socket", str(tmp_path / "cli.sock")])

    assert result.exit_code == 0
    assert FakeClient.instances[0].socket_path == (tmp_path / "cli.sock").absolute()


def test_ps_uses_environment_socket_without_loading_provider_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTROL_SOCKET_PATH", str(tmp_path / "env.sock"))
    monkeypatch.setattr(
        cli_module.Settings,
        "from_environment",
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("must not load settings"))),
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert FakeClient.instances[0].socket_path == (tmp_path / "env.sock").absolute()


def test_ps_uses_runtime_default_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CONTROL_SOCKET_PATH", raising=False)
    expected = tmp_path / "default.sock"
    observed: list[Path | None] = []
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda explicit: observed.append(explicit) or expected,
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert observed == [None]
    assert FakeClient.instances[0].socket_path == expected


@pytest.mark.parametrize(
    "filters",
    [
        ["state=active"] * 33,
        ["model=" + "x" * 251],
        ["missing-separator"],
        ["model=one=two"],
        [" =value"],
        ["model=  "],
        ["unknown=value"],
    ],
)
def test_ps_rejects_invalid_filters_before_network(filters: list[str]) -> None:
    arguments = ["ps"]
    for entry in filters:
        arguments.extend(["--filter", entry])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert FakeClient.instances == []


def test_ps_normalizes_and_forwards_filters_in_order(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ps",
            "--socket",
            str(tmp_path / "control.sock"),
            "--filter",
            " state = active ",
            "--filter",
            "model = opus",
            "--filter",
            "state=idle",
        ],
    )

    assert result.exit_code == 0
    assert FakeClient.instances[0].filters == (
        "state=active",
        "model=opus",
        "state=idle",
    )


def test_ps_accepts_filter_count_and_length_boundaries() -> None:
    supplied = [f" state = value-{index} " for index in range(31)]
    supplied.append("model=" + "x" * 250)
    arguments = ["ps"]
    for entry in supplied:
        arguments.extend(["--filter", entry])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert FakeClient.instances[0].filters == (
        *(f"state=value-{index}" for index in range(31)),
        "model=" + "x" * 250,
    )
    assert len(FakeClient.instances[0].filters[-1]) == 256



def test_ps_table_renders_nested_agents_parent_before_child() -> None:
    parent = agent("parent-agent")
    child = agent("child-agent", parent_id="parent-agent")
    FakeClient.result = response(
        session("session-id", agents=(child, parent))
    )

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    assert "SESSION / AGENT" in result.stdout
    assert result.stdout.index("parent-agent") < result.stdout.index("child-agent")
    assert "└─ parent-agent" in result.stdout
    assert "   └─ child-agent" in result.stdout


def test_ps_json_keeps_flat_agent_collection() -> None:
    FakeClient.result = response(
        session(
            "session-id",
            agents=(agent("child", parent_id="parent"),),
        )
    )

    result = runner.invoke(app, ["ps", "--format", "json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert len(payload[0]["agents"]) == 1
    assert payload[0]["agents"][0]["id"] == "child"
    assert payload[0]["agents"][0]["parent_id"] == "parent"



def test_agent_tree_handles_deep_lineage_without_recursion():
    agents = tuple(
        agent(
            f"agent-{index}",
            parent_id=(f"agent-{index - 1}" if index else None),
        )
        for index in range(1_100)
    )

    rows = cli_module._agent_rows(agents, CAPTURED_AT, no_trunc=True)

    assert len(rows) == len(agents)

def test_ps_agent_cycles_and_orphans_render_once() -> None:
    agents = (
        agent("orphan", parent_id="missing"),
        agent("self", parent_id="self"),
        agent("cycle-a", parent_id="cycle-b"),
        agent("cycle-b", parent_id="cycle-a"),
    )
    FakeClient.result = response(session("session-id", agents=agents))

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    for item in agents:
        assert result.stdout.count(item.id) == 1

def test_ps_table_has_exact_headers_and_preserves_server_order() -> None:
    FakeClient.result = response(
        session("b" * 64, model="newest-model", seconds_ago=0),
        session("a" * 64, model="older-model", seconds_ago=65),
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0].split() == [
        "SESSION",
        "/",
        "AGENT",
        "MODEL",
        "STATE",
        "EFFORT",
        "CONTEXT",
        "ACTIVE",
        "REQUESTS",
        "LAST",
        "SEEN",
    ]
    assert result.stdout.index("bbbbbbbbbbbb") < result.stdout.index("aaaaaaaaaaaa")
    assert "newest-model" in result.stdout
    assert "older-model" in result.stdout
    assert "1m" in result.stdout


def test_ps_empty_table_prints_headers_only() -> None:
    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1
    assert "SESSION" in result.stdout
    assert "LAST SEEN" in result.stdout


def test_ps_table_formats_ids_context_and_relative_time() -> None:
    identifier = "0123456789abcdef" * 4
    FakeClient.result = response(
        session(identifier, context_window=None, seconds_ago=-5),
        session("b" * 64, context_window=1_000_000, seconds_ago=59),
        session("c" * 64, context_window=200_000, seconds_ago=3600),
        session("d" * 64, context_window=8_000, seconds_ago=172800),
    )

    truncated = runner.invoke(app, ["ps"])
    full = runner.invoke(app, ["ps", "--no-trunc"])

    assert truncated.exit_code == full.exit_code == 0
    assert "0123456789ab" in truncated.stdout
    assert identifier not in truncated.stdout
    assert identifier in full.stdout
    assert "—" in truncated.stdout
    assert "1.0m" in truncated.stdout
    assert "200k" in truncated.stdout
    assert "8k" in truncated.stdout
    for value in ("now", "59s", "1h", "2d"):
        assert value in truncated.stdout


def test_context_formatter_uses_bounded_integer_only_output() -> None:
    maximum = 2**63 - 1

    assert cli_module._format_context(1_000_000) == "1.0m"
    assert cli_module._format_context(200_000) == "200k"
    assert cli_module._format_context(maximum) == "9223372036854.7m"
    assert cli_module._format_context(10**999) == "invalid"
    assert cli_module._format_context(-(10**999)) == "invalid"


def test_ps_table_bounds_numeric_fields_when_client_contract_is_bypassed() -> None:
    huge = 10**999
    item = session("a" * 64).model_copy(update={
        "active_requests": huge,
        "requests": huge,
        "context_window": huge,
    })
    FakeClient.result = response(item)

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert result.stdout.count("invalid") == 3
    assert len(result.stdout.splitlines()[1]) < 200


def test_ps_model_cell_uses_only_resolved_model() -> None:
    FakeClient.result = response(
        session(
            "a" * 64,
            client_model="must-not-be-displayed",
            model="resolved-model",
        )
    )

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    assert result.stdout.splitlines()[1].split()[1] == "resolved-model"
    assert "must-not-be-displayed" not in result.stdout
    assert "→" not in result.stdout


def test_ps_default_bounds_model_and_effort_but_no_trunc_keeps_them() -> None:
    resolved_model = "m" * 500
    effort = "e" * 500
    FakeClient.result = response(
        session(
            "a" * 64,
            client_model="ignored-client-model",
            model=resolved_model,
            effort=effort,
        )
    )

    truncated = runner.invoke(app, ["ps"])
    full = runner.invoke(app, ["ps", "--no-trunc"])

    assert truncated.exit_code == full.exit_code == 0
    cells = truncated.stdout.splitlines()[1].split()
    assert len(cells[1]) == 24
    assert cells[1].endswith("…")
    assert len(cells[3]) == 12
    assert cells[3].endswith("…")
    assert resolved_model not in truncated.stdout
    assert effort not in truncated.stdout
    assert resolved_model in full.stdout
    assert effort in full.stdout


def test_ps_default_bounds_combining_mark_flood_by_atoms_and_cells() -> None:
    combining = "́"
    flood = combining * 10_000
    FakeClient.result = response(
        session(
            "a" * 64,
            model="m" + flood,
            effort="e" + flood,
        )
    )

    truncated = runner.invoke(app, ["ps"])
    full = runner.invoke(app, ["ps", "--no-trunc"])

    assert truncated.exit_code == full.exit_code == 0
    cells = truncated.stdout.splitlines()[1].split()
    assert len(cells[1]) <= 96
    assert len(cells[3]) <= 48
    assert wcswidth(cells[1]) <= 24
    assert wcswidth(cells[3]) <= 12
    assert cells[1].endswith("…")
    assert cells[3].endswith("…")
    assert truncated.stdout.count(combining) < 200
    assert full.stdout.count(combining) == 20_000


def test_ps_table_escapes_terminal_control_characters() -> None:
    controls = "".join(chr(value) for value in [0, 9, 10, 13, 27, 31, 127, 128, 133, 159])
    FakeClient.result = response(
        session(
            "abc" + controls + "def" + "0" * 48,
            client_model="client" + controls,
            model="model" + controls,
            effort="high" + controls,
        )
    )

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    for value in [0, 9, 10, 13, 27, 31, 127, 128, 133, 159]:
        assert f"\\x{value:02x}" in result.stdout
    row = result.stdout.splitlines()[1]
    assert all(character not in row for character in controls)


def test_ps_table_escapes_surrogates_and_remains_utf8_encodable() -> None:
    FakeClient.result = response(
        session(
            "a" * 64,
            model="safe\ud800model\U000e0001",
        )
    )

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    assert "\\ud800" in result.stdout
    assert "\\U000e0001" in result.stdout
    assert "\ud800" not in result.stdout
    result.stdout.encode("utf-8", errors="strict")


def test_ps_table_never_splits_escaped_token_at_truncation_boundary() -> None:
    FakeClient.result = response(
        session("a" * 64, model="a" * 21 + "\x1b" + "tail")
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    model_cell = result.stdout.splitlines()[1].split()[1]
    assert model_cell == "a" * 21 + "…"
    assert "\\x…" not in model_cell


def test_ps_table_aligns_columns_by_display_cell_width() -> None:
    FakeClient.result = response(
        session("a" * 64, model="界界界界é"),
        session("b" * 64, model="123456789"),
    )

    result = runner.invoke(app, ["ps", "--no-trunc"])

    assert result.exit_code == 0
    rows = result.stdout.splitlines()[1:]
    state_offsets = [wcswidth(row[: row.index("idle")]) for row in rows]
    assert state_offsets == [state_offsets[0], state_offsets[0]]


def test_ps_json_rejects_non_finite_numbers_with_safe_error() -> None:
    item = session("a" * 64).model_copy(update={"elapsed_seconds": float("nan")})
    FakeClient.result = response(item)

    result = runner.invoke(app, ["ps", "--format", "json"])

    assert result.exit_code == 1
    assert "Control API returned an invalid sessions response" in result.stderr
    assert "Out of range float" not in result.stderr
    assert "NaN" not in result.stdout
    assert "Infinity" not in result.stdout
    assert "Traceback" not in result.stderr


def test_ps_table_naive_datetime_failure_is_safe() -> None:
    FakeClient.result = SessionListResponse(
        captured_at=datetime(2026, 1, 2, 12),
        sessions=(session("a" * 64),),
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert "Control API returned an invalid sessions response" in result.stderr
    assert "offset-naive" not in result.stderr
    assert "offset-aware" not in result.stderr
    assert "Traceback" not in result.stderr


def test_ps_json_is_exact_array_with_full_fields_and_standard_escaping() -> None:
    item = session(
        "a" * 64,
        client_model="line\nbreak",
        model="ansi\x1bmodel",
    )
    FakeClient.result = response(item)

    result = runner.invoke(app, ["ps", "--format", "json"])

    def reject_non_standard_number(value: str) -> None:
        raise AssertionError(f"non-standard JSON number {value}")

    assert result.exit_code == 0
    assert json.loads(
        result.stdout,
        parse_constant=reject_non_standard_number,
    ) == [item.model_dump(mode="json")]
    assert "line\\nbreak" in result.stdout
    assert "ansi\\u001bmodel" in result.stdout


def test_ps_json_empty_result_is_empty_array() -> None:
    result = runner.invoke(app, ["ps", "--format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ControlUnavailable(Path("/safe/control.sock"), "not found"), "/safe/control.sock"),
        (IncompatibleProtocol("protocol mismatch"), "protocol mismatch"),
        (ControlError("safe failure"), "safe failure"),
    ],
)
def test_ps_client_errors_exit_one_and_close_context(
    error: Exception, expected: str
) -> None:
    FakeClient.error = error

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
    assert FakeClient.instances[0].entered
    assert FakeClient.instances[0].exited
    if isinstance(error, ControlUnavailable):
        assert "start `claude-code-proxy proxy`" in result.stderr
        assert "docker compose exec proxy claude-code-proxy ps" in result.stderr


@pytest.mark.parametrize(
    "error",
    [
        IncompatibleProtocol("protocol version 2 is incompatible\nforged"),
        IncompatibleProtocol("missing sessions capability\x1b\u202e\ud800"),
    ],
)
def test_ps_incompatible_protocol_reports_safe_socket_and_guidance(
    error: IncompatibleProtocol, tmp_path: Path
) -> None:
    socket_path = tmp_path / "chosen\n\u202e.sock"
    FakeClient.error = error

    result = runner.invoke(
        app, ["ps", "--socket", str(socket_path)]
    )

    assert result.exit_code == 1
    assert "Incompatible control API" in result.stderr
    assert str(tmp_path) in result.stderr
    assert "chosen\\x0a\\u202e.sock" in result.stderr
    assert "protocol version" in result.stderr or "sessions capability" in result.stderr
    assert "forged" in result.stderr or "\\x1b\\u202e\\ud800" in result.stderr
    assert "source: start `claude-code-proxy proxy`" in result.stderr
    assert "docker compose exec proxy claude-code-proxy ps" in result.stderr
    assert "\nforged" not in result.stderr
    assert "\x1b" not in result.stderr
    assert "\u202e" not in result.stderr
    assert "\ud800" not in result.stderr
    result.stderr.encode("utf-8", errors="strict")


def test_ps_generic_control_error_has_no_incompatibility_guidance() -> None:
    FakeClient.error = ControlError("Control API returned HTTP 500")

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert "HTTP 500" in result.stderr
    assert "Incompatible" not in result.stderr
    assert "claude-code-proxy proxy" not in result.stderr
    assert "docker compose exec" not in result.stderr


def test_ps_closes_client_context_on_success() -> None:
    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert FakeClient.instances[0].entered
    assert FakeClient.instances[0].exited


def test_module_adapter_and_console_script_declaration() -> None:
    from claude_code_proxy import __main__ as main_module

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert main_module.app is app
    assert metadata["project"]["scripts"]["claude-code-proxy"] == "claude_code_proxy.cli:app"
    assert "Operating System :: POSIX :: Linux" in metadata["project"]["classifiers"]
    assert "typer>=0.21.1" in metadata["project"]["dependencies"]
    assert "wcwidth>=0.2.13" in metadata["project"]["dependencies"]


def test_installed_script_and_module_adapter_subprocess_help() -> None:
    installed = shutil.which("claude-code-proxy")
    if installed is None:
        installed = str(Path(sys.executable).with_name("claude-code-proxy"))
    assert Path(installed).is_file(), f"console script not found at {installed}"
    commands = (
        [installed, "--help"],
        [sys.executable, "-m", "claude_code_proxy", "--help"],
    )

    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "proxy" in completed.stdout
        assert "ps" in completed.stdout


def test_ps_subprocess_escapes_model_and_effort_terminal_data(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    observed = session(
        "a" * 64,
        model="model界é\n\r\x1b\x85\ud800tail",
        effort="effort界é\n\r\x1b\x85\ud800tail",
    )
    health = {
        "protocol_version": 1,
        "application_version": "0.1.0",
        "pid": 42,
        "started_at": "2026-01-02T03:04:05Z",
        "uptime_seconds": 10.5,
        "capabilities": ["sessions"],
        "sessions": {"active": 0, "retained": 1},
        "inactive_limit": 1000,
    }
    payloads = {
        "/v1/health": health,
        "/v1/sessions": {
            "captured_at": CAPTURED_AT.isoformat(),
            "sessions": [observed.model_dump(mode="json")],
        },
    }

    completed = run_ps_subprocess(socket_path, payloads, "--no-trunc")
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "model界é\\x0a\\x0d\\x1b\\x85\\ud800tail" in completed.stdout
    assert "effort界é\\x0a\\x0d\\x1b\\x85\\ud800tail" in completed.stdout
    assert "\\x1b" in completed.stdout
    assert "\\x85" in completed.stdout
    assert "\\ud800" in completed.stdout
    assert len(completed.stdout.splitlines()) == 2
    assert "\r" not in completed.stdout
    assert "\x1b" not in completed.stdout
    assert "\x85" not in completed.stdout
    assert "\ud800" not in completed.stdout
    completed.stdout.encode("utf-8", errors="strict")


def test_ps_subprocess_hides_utc_conversion_overflow(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    observed = session("a" * 64).model_dump(mode="json")
    observed["last_seen"] = "9999-12-31T23:59:59-23:59"
    payloads = {
        "/v1/health": {
            "protocol_version": 1,
            "application_version": "0.1.0",
            "pid": 42,
            "started_at": "2026-01-02T03:04:05Z",
            "uptime_seconds": 10.5,
            "capabilities": ["sessions"],
            "sessions": {"active": 0, "retained": 1},
            "inactive_limit": 1000,
        },
        "/v1/sessions": {
            "captured_at": CAPTURED_AT.isoformat(),
            "sessions": [observed],
        },
    }

    completed = run_ps_subprocess(socket_path, payloads)

    assert completed.returncode == 1, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Control API returned an invalid sessions response" in completed.stderr
    assert "date value out of range" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ps_subprocess_rejects_oversized_context_safely(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    observed = session("a" * 64).model_dump(mode="json")
    observed["context_window"] = 10**999
    payloads = {
        "/v1/health": {
            "protocol_version": 1,
            "application_version": "0.1.0",
            "pid": 42,
            "started_at": "2026-01-02T03:04:05Z",
            "uptime_seconds": 10.5,
            "capabilities": ["sessions"],
            "sessions": {"active": 0, "retained": 1},
            "inactive_limit": 1000,
        },
        "/v1/sessions": {
            "captured_at": CAPTURED_AT.isoformat(),
            "sessions": [observed],
        },
    }

    completed = run_ps_subprocess(socket_path, payloads)

    assert completed.returncode == 1, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Control API returned an invalid sessions response" in completed.stderr
    assert "integer division" not in completed.stderr
    assert "Traceback" not in completed.stderr
    assert len(completed.stdout) < 200
    assert len(completed.stderr) < 500


@pytest.mark.parametrize("malformed_health", [True, False])
def test_ps_subprocess_rejects_coerced_wire_types(
    tmp_path: Path,
    malformed_health: bool,
) -> None:
    socket_path = tmp_path / "control.sock"
    health: dict[str, object] = {
        "protocol_version": 1,
        "application_version": "0.1.0",
        "pid": 42,
        "started_at": "2026-01-02T03:04:05Z",
        "uptime_seconds": 10.5,
        "capabilities": ["sessions"],
        "sessions": {"active": 0, "retained": 1},
        "inactive_limit": 1000,
    }
    observed = session("a" * 64).model_dump(mode="json")
    expected = "invalid sessions response"
    if malformed_health:
        health.update({
            "pid": True,
            "inactive_limit": "1",
            "sessions": {"active": 1.0, "retained": 1},
        })
        expected = "invalid health response"
    else:
        observed.update({
            "active_requests": True,
            "requests": "1",
            "context_window": 1.0,
        })
    payloads = {
        "/v1/health": health,
        "/v1/sessions": {
            "captured_at": CAPTURED_AT.isoformat(),
            "sessions": [observed],
        },
    }

    completed = run_ps_subprocess(socket_path, payloads)

    assert completed.returncode == 1, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert expected in completed.stderr
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    "arguments", [["--help"], ["proxy", "--help"], ["ps", "--help"]]
)
def test_help_is_available_on_non_linux(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket_module.sys, "platform", "darwin")

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert "Usage" in result.stdout


@pytest.mark.parametrize("command", ["proxy", "ps"])
def test_command_help_lists_documented_options(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    help_text = unstyle(result.stdout)
    if command == "proxy":
        for option in ("--host", "--port", "--socket", "--session-limit"):
            assert option in help_text
    else:
        for option in ("--filter", "--format", "--no-trunc", "--socket"):
            assert option in help_text

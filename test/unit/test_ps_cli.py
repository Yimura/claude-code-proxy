from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest
from wcwidth import wcswidth

from claude_code_proxy import cli as cli_module
from claude_code_proxy import cli_common
from claude_code_proxy.cli import app
from claude_code_proxy.config import Settings
from claude_code_proxy.control.client import (
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from claude_code_proxy.control.schemas import SessionListResponse
from test.unit.cli_test_support import (
    CAPTURED_AT,
    FakeClient,
    agent,
    response,
    run_ps_subprocess,
    runner,
    session,
)


@pytest.fixture(autouse=True)
def reset_fake_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeClient.instances = []
    FakeClient.result = response()
    FakeClient.error = None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "ControlClient", FakeClient)


def test_ps_watch_table_rejects_non_tty_before_environment_or_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_load_current_directory_environment",
        lambda: pytest.fail("environment loaded before watch output validation"),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda _: pytest.fail("socket resolved before watch output validation"),
    )

    result = runner.invoke(app, ["ps", "--watch"])

    assert result.exit_code == 2
    assert "--watch" in result.stderr
    assert "--format" in result.stderr
    assert "json instead" in result.stderr
    assert FakeClient.instances == []


def test_ps_watch_json_delegates_normalized_options_in_one_client_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_socket = tmp_path / "environment.sock"
    explicit_socket = tmp_path / "explicit.sock"
    monkeypatch.setenv("CONTROL_SOCKET_PATH", str(environment_socket))
    calls: list[tuple[FakeClient, tuple[str, ...], cli_common.OutputFormat, bool]] = []

    def watch_sessions(
        client: FakeClient,
        filters: tuple[str, ...],
        output_format: cli_common.OutputFormat,
        no_trunc: bool,
        renderer: object,
    ) -> None:
        assert client.entered
        assert not client.exited
        assert renderer is cli_module._render_sessions
        calls.append((client, filters, output_format, no_trunc))

    monkeypatch.setattr(
        cli_module,
        "watch_sessions",
        watch_sessions,
        raising=False,
    )

    result = runner.invoke(
        app,
        [
            "ps",
            "--watch",
            "--format",
            "json",
            "--filter",
            " state = active ",
            "--filter",
            "model = opus",
            "--socket",
            str(explicit_socket),
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0
    assert len(FakeClient.instances) == 1
    client = FakeClient.instances[0]
    assert client.socket_path == explicit_socket.absolute()
    assert calls == [
        (
            client,
            ("state=active", "model=opus"),
            cli_common.OutputFormat.JSON,
            True,
        )
    ]
    assert client.entered
    assert client.exited


def test_ps_watch_control_error_is_reported_safely_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def watch_sessions(*_: object) -> None:
        raise ControlError("watch failed\nforged")

    monkeypatch.setattr(
        cli_module,
        "watch_sessions",
        watch_sessions,
        raising=False,
    )

    result = runner.invoke(app, ["ps", "--watch", "--format", "json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "watch failed\\x0aforged" in result.stderr
    assert "\nforged" not in result.stderr
    assert "Traceback" not in result.stderr
    assert FakeClient.instances[0].entered
    assert FakeClient.instances[0].exited


def test_ps_watch_keyboard_interrupt_exits_zero_and_closes_client() -> None:
    FakeClient.error = KeyboardInterrupt()

    result = runner.invoke(app, ["ps", "--watch", "--format", "json"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Aborted" not in result.output
    assert FakeClient.instances[0].entered
    assert FakeClient.instances[0].exited


def test_render_sessions_compact_json_matches_pretty_payload_on_one_line() -> None:
    snapshot = response(
        session(
            "session-id",
            client_model="line\nbreak",
            model="ansi\x1bmodel",
            agents=(agent("agent-id"),),
        )
    )

    pretty = cli_module._render_sessions(
        snapshot,
        cli_common.OutputFormat.JSON,
        False,
    )
    compact = cli_module._render_sessions(
        snapshot,
        cli_common.OutputFormat.JSON,
        False,
        True,
    )

    assert json.loads(compact) == json.loads(pretty)
    assert json.loads(compact) == [snapshot.sessions[0].model_dump(mode="json")]
    assert "\n" not in compact
    assert compact.startswith('[{"id":"session-id",')
    assert "\n  {" in pretty


def test_ps_success_output_is_byte_exact() -> None:
    FakeClient.result = response(session("session-1234567890"))

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 0
    assert result.stdout == (
        "SESSION / AGENT  MODEL        STATE  EFFORT  CONTEXT  ACTIVE  "
        "REQUESTS  LAST SEEN\n"
        "session-1234     gpt-5.6-sol  idle   high    1.0m     0       "
        "3         now\n"
    )
    assert result.stderr == ""


def test_ps_unavailable_output_is_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda path: Path("/safe/control.sock"),
    )
    FakeClient.error = ControlUnavailable(
        Path("/safe/control.sock"), "not found"
    )

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Error: Control API unavailable at /safe/control.sock: not found\n"
        "source: start `claude-code-proxy proxy`\n"
        "Docker: run `docker compose exec proxy claude-code-proxy ps`\n"
    )


def test_ps_incompatible_output_is_byte_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda path: Path("/safe/control.sock"),
    )
    FakeClient.error = IncompatibleProtocol("protocol mismatch")

    result = runner.invoke(app, ["ps"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Error: Incompatible control API at /safe/control.sock: "
        "protocol mismatch\n"
        "source: start `claude-code-proxy proxy`\n"
        "Docker: run `docker compose exec proxy claude-code-proxy ps`\n"
    )


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
        cli_common,
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
            "session_id = raw-session",
            "--filter",
            "state=idle",
        ],
    )

    assert result.exit_code == 0
    assert FakeClient.instances[0].filters == (
        "state=active",
        "model=opus",
        "session_id=raw-session",
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


def test_ps_json_pretty_output_is_byte_exact() -> None:
    item = session("session-id", agents=(agent("agent-id"),))
    FakeClient.result = response(item)

    result = runner.invoke(app, ["ps", "--format", "json"])

    assert result.exit_code == 0
    assert result.stdout == (
        json.dumps(
            [item.model_dump(mode="json")],
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    assert result.stderr == ""


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

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
from click import unstyle

from claude_code_proxy import cli as cli_module
import claude_code_proxy.control.socket as socket_module
from claude_code_proxy.cli import app
from claude_code_proxy.config import Settings
from test.unit.cli_test_support import ROOT, runner, settings


def test_bare_app_shows_help_successfully() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert "proxy" in result.stdout
    assert "ps" in result.stdout
    assert "perf" in result.stdout
    assert "tui" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["unknown"],
        ["ps", "--format", "yaml"],
        ["perf", "--format", "yaml"],
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
        assert "perf" in completed.stdout
        assert "tui" in completed.stdout



@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["proxy", "--help"],
        ["ps", "--help"],
        ["perf", "--help"],
        ["tui", "--help"],
    ],
)
def test_help_is_available_on_non_linux(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket_module.sys, "platform", "darwin")

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert "Usage" in result.stdout


@pytest.mark.parametrize("command", ["proxy", "ps", "perf", "tui"])
def test_command_help_lists_documented_options(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0
    help_text = unstyle(result.stdout)
    if command == "proxy":
        for option in ("--host", "--port", "--socket", "--session-limit"):
            assert option in help_text
        return
    if command == "tui":
        assert "--socket" in help_text
        assert "--filter" not in help_text
        return
    for option in ("--filter", "--format", "--no-trunc", "--socket"):
        assert option in help_text
    if command == "perf":
        assert "--watch" in help_text


def test_cli_import_and_help_keep_textual_app_lazy() -> None:
    script = """
import sys
from claude_code_proxy.cli import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, [\"--help\"])
assert result.exit_code == 0
assert \"claude_code_proxy.tui.app\" not in sys.modules
assert not any(name == \"textual\" or name.startswith(\"textual.\") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_tui_rejects_non_tty_before_loading_textual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_interactive_tty", lambda: False)
    monkeypatch.setattr(
        cli_module,
        "_load_tui_runner",
        lambda: pytest.fail("TUI loaded before interactive TTY check"),
    )

    result = runner.invoke(app, ["tui"])

    assert result.exit_code == 1
    assert "interactive TTY" in result.stderr
    assert len(result.stderr) < 300


def test_tui_loads_environment_resolves_socket_then_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    socket_path = tmp_path / "control.sock"
    monkeypatch.setattr(cli_module, "_interactive_tty", lambda: True)
    monkeypatch.setattr(
        cli_module,
        "_load_current_directory_environment",
        lambda: calls.append(("environment", None)),
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_socket_path",
        lambda value: calls.append(("socket", value)) or socket_path,
    )
    monkeypatch.setattr(
        cli_module,
        "_load_tui_runner",
        lambda: calls.append(("load", None))
        or (lambda value: calls.append(("run", value)) or SimpleNamespace(exit_code=0, message="")),
    )

    result = runner.invoke(app, ["tui", "--socket", str(socket_path)])

    assert result.exit_code == 0
    assert calls == [
        ("environment", None),
        ("socket", socket_path),
        ("load", None),
        ("run", socket_path),
    ]


def test_tui_nonzero_result_prints_safe_capability_and_docker_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_interactive_tty", lambda: True)
    monkeypatch.setattr(cli_module, "resolve_socket_path", lambda value: Path("/safe.sock"))
    monkeypatch.setattr(
        cli_module,
        "_load_tui_runner",
        lambda: lambda value: SimpleNamespace(
            exit_code=1,
            message=(
                "TUI requires control protocol v1 with performance and "
                "performance_events capabilities"
            ),
        ),
    )

    result = runner.invoke(app, ["tui"])

    assert result.exit_code == 1
    assert "protocol v1" in result.stderr
    assert "performance" in result.stderr
    assert "performance_events" in result.stderr
    assert "performance collector" in result.stderr
    assert "docker compose exec proxy claude-code-proxy tui" in result.stderr
    assert "Traceback" not in result.stderr

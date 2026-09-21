from dataclasses import replace
from pathlib import Path

from click import unstyle
import pytest
from typer.testing import CliRunner

from claude_code_proxy import cli as cli_module
from claude_code_proxy.cli import app
from claude_code_proxy.config import PerformanceMode, Settings

runner = CliRunner()


def settings(tmp_path: Path, *, mode: PerformanceMode) -> Settings:
    return Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
        vertex_project="unset",
        vertex_location="unset",
        use_vertex_auth=False,
        openai_base_url=None,
        openai_transport="litellm",
        opencode_data_dir=tmp_path,
        model_mapping_path=tmp_path / "models.json",
        control_socket_path=tmp_path / "control.sock",
        performance_mode=mode,
    )


def invoke_proxy(
    monkeypatch: pytest.MonkeyPatch,
    configured: Settings,
    arguments: list[str],
) -> tuple[object, list[Settings]]:
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
    return runner.invoke(app, ["proxy", *arguments]), observed


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], PerformanceMode.OFF),
        (["--performance", "collector"], PerformanceMode.COLLECTOR),
        (["--performance", "logging"], PerformanceMode.LOGGING),
    ],
)
def test_proxy_applies_explicit_performance_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    expected: PerformanceMode,
) -> None:
    configured = settings(tmp_path, mode=PerformanceMode.LOGGING)

    result, observed = invoke_proxy(monkeypatch, configured, arguments)

    assert result.exit_code == 0
    assert observed == [replace(configured, performance_mode=expected)]


def test_proxy_performance_option_requires_a_mode() -> None:
    result = runner.invoke(app, ["proxy", "--performance"])

    assert result.exit_code == 2


def test_proxy_help_documents_performance_modes() -> None:
    result = runner.invoke(app, ["proxy", "--help"])

    assert result.exit_code == 0
    help_text = unstyle(result.stdout).lower()
    assert "--performance" in help_text
    assert "off" in help_text
    assert "collector" in help_text
    assert "logging" in help_text

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from claude_code_proxy import cli_common
from claude_code_proxy.cli_common import (
    DISPLAY_ATOMS_PER_CELL,
    ERROR_DISPLAY_LENGTH,
    MAX_FILTER_LENGTH,
    MAX_FILTERS,
    SUPPORTED_FILTERS,
    OutputFormat,
    bounded_error,
    control_guidance,
    display_width,
    exit_with_error,
    load_current_directory_environment,
    render_table,
    report_unavailable,
    terminal_atom,
    terminal_text,
    validated_filter,
    validated_filters,
)
from claude_code_proxy.control.client import ControlUnavailable
from claude_code_proxy.performance_filters import FILTER_FIELDS


def test_shared_constants_preserve_cli_contract() -> None:
    assert MAX_FILTERS == 32
    assert MAX_FILTER_LENGTH == 256
    assert DISPLAY_ATOMS_PER_CELL == 4
    assert ERROR_DISPLAY_LENGTH == 400
    assert SUPPORTED_FILTERS is FILTER_FIELDS
    assert tuple(OutputFormat) == (OutputFormat.TABLE, OutputFormat.JSON)


def test_validated_filters_preserve_order_and_normalize_whitespace() -> None:
    assert validated_filters(
        [" state = active ", "model = opus", "state=idle"]
    ) == ("state=active", "model=opus", "state=idle")


def test_validated_filters_accept_count_and_length_boundaries() -> None:
    entries = ["state=active"] * (MAX_FILTERS - 1)
    entries.append("model=" + "x" * (MAX_FILTER_LENGTH - len("model=")))

    assert validated_filters(entries) == tuple(entries)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (["state=active"] * 33, "at most 32 filters may be supplied"),
        (["model=" + "x" * 251], "filter must be at most 256 characters"),
        (["missing-separator"], "filter must contain exactly one '='"),
        (["model=one=two"], "filter must contain exactly one '='"),
        ([" =value"], "filter key and value must not be empty"),
        (["model=  "], "filter key and value must not be empty"),
        (
            ["unknown=value"],
            "unsupported filter key 'unknown'; choose from "
            "effort, id, model, provider, session_id, state, transport",
        ),
    ],
)
def test_validated_filters_preserve_bad_parameter_text(
    entries: list[str], message: str
) -> None:
    with pytest.raises(typer.BadParameter) as raised:
        validated_filters(entries)

    assert raised.value.param_hint == "--filter"
    assert raised.value.message == message


def test_validated_filter_returns_exact_key_value_form() -> None:
    assert validated_filter(" session_id = raw-session ") == (
        "session_id=raw-session"
    )


def test_terminal_text_escapes_controls_and_indeterminate_scalars() -> None:
    value = "safe\x00\x1b\x85\ud800\U000e0001tail"

    rendered = terminal_text(value)

    assert rendered == "safe\\x00\\x1b\\x85\\ud800\\U000e0001tail"
    rendered.encode("utf-8", errors="strict")
    assert terminal_atom("\x1b") == "\\x1b"


def test_display_width_counts_wide_and_combining_characters() -> None:
    assert display_width("界e\N{COMBINING ACUTE ACCENT}") == 3


def test_terminal_text_bounds_zero_width_flood_by_atoms_and_cells() -> None:
    rendered = terminal_text(
        "m" + "\N{COMBINING ACUTE ACCENT}" * 10_000,
        maximum=24,
    )

    assert display_width(rendered) <= 24
    assert len(rendered) <= 24 * DISPLAY_ATOMS_PER_CELL
    assert rendered.endswith("…")


def test_terminal_text_truncates_only_at_whole_escape_atoms() -> None:
    rendered = terminal_text("a" * 21 + "\x1b" + "tail", maximum=24)

    assert rendered == "a" * 21 + "…"
    assert "\\x…" not in rendered


def test_render_table_aligns_cells_by_terminal_display_width() -> None:
    rendered = render_table(
        ("NAME", "VALUE"),
        [("界", "wide"), ("e\N{COMBINING ACUTE ACCENT}", "combining")],
    )

    assert rendered == (
        "NAME  VALUE\n"
        "界    wide\n"
        "e\N{COMBINING ACUTE ACCENT}     combining"
    )


def test_load_current_directory_environment_uses_only_cwd_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, bool]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_common,
        "load_dotenv",
        lambda *, dotenv_path, override: calls.append((dotenv_path, override)),
    )

    load_current_directory_environment()

    assert calls == [(tmp_path / ".env", False)]


def test_control_guidance_preserves_ps_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    control_guidance("not available")

    assert capsys.readouterr().err == (
        "Error: not available\n"
        "source: start `claude-code-proxy proxy`\n"
        "Docker: run `docker compose exec proxy claude-code-proxy ps`\n"
    )


def test_control_guidance_uses_command_name_and_escapes_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    control_guidance("bad\nmessage", command_name="perf")

    assert capsys.readouterr().err == (
        "Error: bad\\x0amessage\n"
        "source: start `claude-code-proxy proxy`\n"
        "Docker: run `docker compose exec proxy claude-code-proxy perf`\n"
    )


def test_report_unavailable_preserves_ps_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_unavailable(
        ControlUnavailable(Path("/safe/control.sock"), "not found"),
    )

    assert capsys.readouterr().err == (
        "Error: Control API unavailable at /safe/control.sock: not found\n"
        "source: start `claude-code-proxy proxy`\n"
        "Docker: run `docker compose exec proxy claude-code-proxy ps`\n"
    )


def test_exit_with_error_escapes_and_bounds_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as raised:
        exit_with_error("failure\n" + "x" * 1_000)

    assert raised.value.exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("Error: failure\\x0a")
    assert display_width(error.removeprefix("Error: ").rstrip("\n")) <= 400
    assert error.rstrip().endswith("…")
    assert bounded_error("failure\n") == "failure\\x0a"

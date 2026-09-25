from __future__ import annotations

from io import StringIO
import os

import pytest

from claude_code_proxy import ps_watch_cli


class RecordingOutput(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


@pytest.mark.parametrize(
    ("frame", "columns", "expected"),
    [
        ("", 80, 1),
        ("ascii", 5, 1),
        ("wrapped", 3, 3),
        ("one\n\n123456", 3, 4),
        ("界界a", 2, 3),
        ("e\N{COMBINING ACUTE ACCENT}", 1, 1),
    ],
)
def test_physical_rows_counts_logical_lines_and_display_cells(
    frame: str,
    columns: int,
    expected: int,
) -> None:
    assert ps_watch_cli.physical_rows(frame, columns) == expected


@pytest.mark.parametrize("columns", [0, -1])
def test_physical_rows_rejects_non_positive_columns(columns: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        ps_watch_cli.physical_rows("frame", columns)


def test_first_frame_is_written_without_ansi_prefix_and_flushed() -> None:
    output = RecordingOutput()
    writer = ps_watch_cli.TerminalFrameWriter(
        output,
        columns=lambda: (_ for _ in ()).throw(AssertionError("not needed")),
    )

    writer.write("first\nsecond")

    assert output.getvalue() == "first\nsecond\n"
    assert output.flush_count == 1


def test_later_frame_erases_entire_old_region_before_shrinking_output() -> None:
    output = RecordingOutput()
    writer = ps_watch_cli.TerminalFrameWriter(output, columns=lambda: 4)
    writer.write("abcdefgh\n123456")

    writer.write("x")

    erase_old_frame = (
        "\x1b[4F"
        "\x1b[2K\x1b[1E"
        "\x1b[2K\x1b[1E"
        "\x1b[2K\x1b[1E"
        "\x1b[2K"
        "\x1b[3F"
    )
    assert output.getvalue() == "abcdefgh\n123456\n" + erase_old_frame + "x\n"
    assert output.flush_count == 2


def test_later_frame_recomputes_old_rows_after_terminal_resize() -> None:
    current_columns = [8]
    column_calls = 0

    def columns() -> int:
        nonlocal column_calls
        column_calls += 1
        return current_columns[0]

    output = RecordingOutput()
    writer = ps_watch_cli.TerminalFrameWriter(output, columns=columns)
    writer.write("abcdefgh")
    current_columns[0] = 4

    writer.write("next")

    assert output.getvalue() == (
        "abcdefgh\n"
        "\x1b[2F\x1b[2K\x1b[1E\x1b[2K\x1b[1F"
        "next\n"
    )
    assert column_calls == 1


def test_default_columns_getter_uses_terminal_size_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallbacks: list[tuple[int, int]] = []

    def terminal_size(*, fallback: tuple[int, int]) -> os.terminal_size:
        fallbacks.append(fallback)
        return os.terminal_size((12, 24))

    monkeypatch.setattr(ps_watch_cli.shutil, "get_terminal_size", terminal_size)
    output = RecordingOutput()
    writer = ps_watch_cli.TerminalFrameWriter(output)
    writer.write("x" * 13)

    writer.write("new")

    assert output.getvalue().endswith(
        "\x1b[2F\x1b[2K\x1b[1E\x1b[2K\x1b[1Fnew\n"
    )
    assert fallbacks == [(80, 24)]


def test_frame_replacement_avoids_forbidden_terminal_controls() -> None:
    output = RecordingOutput()
    writer = ps_watch_cli.TerminalFrameWriter(output, columns=lambda: 2)
    writer.write("old frame")
    writer.write("new")

    rendered = output.getvalue()
    forbidden = (
        "\x1b[2J",
        "\x1b[3J",
        "\x1b[?1049h",
        "\x1b[?1049l",
        "\x1b[?25h",
        "\x1b[?25l",
    )
    assert all(control not in rendered for control in forbidden)

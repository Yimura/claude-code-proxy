from __future__ import annotations

from io import StringIO
import os

import pytest
import typer

from claude_code_proxy import ps_watch_cli
from claude_code_proxy.cli_common import OutputFormat
from claude_code_proxy.control.client import ControlError
from claude_code_proxy.control.schemas import SessionListResponse

from .cli_test_support import response


class RecordingOutput(StringIO):
    def __init__(self, *, tty: bool = False) -> None:
        super().__init__()
        self.flush_count = 0
        self._tty = tty

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.parametrize(
    ("output_format", "tty"),
    [
        (OutputFormat.TABLE, True),
        (OutputFormat.JSON, False),
    ],
)
def test_validate_watch_output_accepts_supported_streams(
    output_format: OutputFormat,
    tty: bool,
) -> None:
    ps_watch_cli.validate_watch_output(
        output_format,
        RecordingOutput(tty=tty),
    )


def test_validate_watch_output_rejects_table_on_non_tty() -> None:
    with pytest.raises(typer.BadParameter) as raised:
        ps_watch_cli.validate_watch_output(
            OutputFormat.TABLE,
            RecordingOutput(tty=False),
        )

    assert "TTY" in raised.value.message
    assert raised.value.param_hint == "--watch"


def test_plain_line_writer_writes_and_flushes_each_record() -> None:
    output = RecordingOutput()
    writer = ps_watch_cli.PlainLineWriter(output)

    writer.write('[{"id":"first"}]')
    writer.write("[]")

    assert output.getvalue() == '[{"id":"first"}]\n[]\n'
    assert output.flush_count == 2


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


class SnapshotClient:
    def __init__(self, *snapshots: SessionListResponse) -> None:
        self._snapshots = list(snapshots)
        self.filter_calls: list[tuple[str, ...]] = []

    def sessions(self, filters: tuple[str, ...]) -> SessionListResponse:
        self.filter_calls.append(filters)
        if not self._snapshots:
            raise KeyboardInterrupt
        return self._snapshots.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TimedClient:
    def __init__(self, clock: FakeClock, durations: list[float]) -> None:
        self._clock = clock
        self._durations = durations
        self.poll_starts: list[float] = []

    def sessions(self, filters: tuple[str, ...]) -> SessionListResponse:
        del filters
        self.poll_starts.append(self._clock.now)
        if not self._durations:
            raise KeyboardInterrupt
        self._clock.now += self._durations.pop(0)
        return response()


def test_watch_sessions_uses_terminal_frame_writer_for_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    constructed_with: list[StringIO] = []
    render_calls: list[tuple[OutputFormat, bool, bool]] = []

    class RecordingWriter:
        def __init__(self, output: StringIO) -> None:
            constructed_with.append(output)

        def write(self, frame: str) -> None:
            writes.append(frame)

    def reject_plain_writer(output: StringIO) -> None:
        del output
        raise AssertionError("JSON writer must not be selected")

    def render_sessions(
        snapshot: SessionListResponse,
        output_format: OutputFormat,
        no_trunc: bool,
        compact_json: bool,
    ) -> str:
        del snapshot
        render_calls.append((output_format, no_trunc, compact_json))
        return "table frame"

    monkeypatch.setattr(ps_watch_cli, "TerminalFrameWriter", RecordingWriter)
    monkeypatch.setattr(ps_watch_cli, "PlainLineWriter", reject_plain_writer)
    output = RecordingOutput(tty=True)

    ps_watch_cli.watch_sessions(
        SnapshotClient(response()),
        (),
        OutputFormat.TABLE,
        False,
        render_sessions,
        output=output,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert constructed_with == [output]
    assert writes == ["table frame"]
    assert render_calls == [(OutputFormat.TABLE, False, False)]


def test_watch_sessions_writes_complete_compact_json_snapshots() -> None:
    snapshots = (response(), response())
    client = SnapshotClient(*snapshots)
    output = RecordingOutput()
    render_calls: list[tuple[SessionListResponse, OutputFormat, bool, bool]] = []

    def render_sessions(
        snapshot: SessionListResponse,
        output_format: OutputFormat,
        no_trunc: bool,
        compact_json: bool,
    ) -> str:
        render_calls.append(
            (snapshot, output_format, no_trunc, compact_json)
        )
        index = len(render_calls)
        return f'[{{"snapshot":{index}}}]'

    filters = ["state=active", "model=claude"]
    ps_watch_cli.watch_sessions(
        client,
        filters,
        OutputFormat.JSON,
        True,
        render_sessions,
        output=output,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert output.getvalue() == '[{"snapshot":1}]\n[{"snapshot":2}]\n'
    assert output.flush_count == 2
    assert [call[0] for call in render_calls] == list(snapshots)
    assert all(
        call[1:] == (OutputFormat.JSON, True, True)
        for call in render_calls
    )
    assert client.filter_calls == [
        ("state=active", "model=claude"),
        ("state=active", "model=claude"),
        ("state=active", "model=claude"),
    ]
    assert all(
        filters_call is client.filter_calls[0]
        for filters_call in client.filter_calls
    )


def test_watch_sessions_uses_current_stdout_when_output_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = RecordingOutput()
    monkeypatch.setattr(ps_watch_cli.sys, "stdout", output)

    ps_watch_cli.watch_sessions(
        SnapshotClient(response()),
        (),
        OutputFormat.JSON,
        False,
        lambda *_: "[]",
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert output.getvalue() == "[]\n"
    assert output.flush_count == 1


def test_watch_sessions_starts_immediately_and_keeps_poll_start_cadence() -> None:
    clock = FakeClock()
    client = TimedClient(clock, [0.2, 0.2, 0.2])

    ps_watch_cli.watch_sessions(
        client,
        (),
        OutputFormat.JSON,
        False,
        lambda *_: "[]",
        output=RecordingOutput(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert client.poll_starts == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert clock.sleeps == pytest.approx([0.8, 0.8, 0.8])


def test_watch_sessions_schedules_from_now_after_slow_poll() -> None:
    clock = FakeClock()
    client = TimedClient(clock, [1.5, 1.5])

    ps_watch_cli.watch_sessions(
        client,
        (),
        OutputFormat.JSON,
        False,
        lambda *_: "[]",
        output=RecordingOutput(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert client.poll_starts == pytest.approx([0.0, 2.5, 5.0])
    assert clock.sleeps == pytest.approx([1.0, 1.0])


def test_watch_sessions_returns_normally_on_keyboard_interrupt() -> None:
    class InterruptingClient:
        def sessions(self, filters: tuple[str, ...]) -> SessionListResponse:
            del filters
            raise KeyboardInterrupt

    ps_watch_cli.watch_sessions(
        InterruptingClient(),
        (),
        OutputFormat.JSON,
        False,
        lambda *_: "not reached",
        output=RecordingOutput(),
    )


@pytest.mark.parametrize("failure_source", ["client", "renderer"])
def test_watch_sessions_propagates_failures_unchanged(
    failure_source: str,
) -> None:
    error: Exception
    if failure_source == "client":
        error = ControlError("control failed")

        class FailingClient:
            def sessions(
                self,
                filters: tuple[str, ...],
            ) -> SessionListResponse:
                del filters
                raise error

        client = FailingClient()
        renderer = lambda *_: "not reached"
    else:
        error = RuntimeError("render failed")
        client = SnapshotClient(response())

        def renderer(*_: object) -> str:
            raise error

    with pytest.raises(type(error), match=str(error)) as raised:
        ps_watch_cli.watch_sessions(
            client,
            (),
            OutputFormat.JSON,
            False,
            renderer,
            output=RecordingOutput(),
        )

    assert raised.value is error

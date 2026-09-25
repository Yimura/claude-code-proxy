"""Terminal frame rendering support for ``ps --watch``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import shutil
import sys
import time
from typing import Protocol, TextIO

import typer

from .cli_common import OutputFormat, display_width
from .control.schemas import SessionListResponse

WATCH_INTERVAL_SECONDS = 1.0
RenderSessions = Callable[
    [SessionListResponse, OutputFormat, bool, bool],
    str,
]


class SessionClient(Protocol):
    """Client surface required by the session watch loop."""

    def sessions(self, filters: Sequence[str]) -> SessionListResponse:
        """Return the current session snapshot."""
        ...


def validate_watch_output(
    output_format: OutputFormat,
    output: TextIO,
) -> None:
    """Reject terminal frame output when stdout is not interactive."""
    if output_format is OutputFormat.TABLE and not output.isatty():
        raise typer.BadParameter(
            "table watch output requires a TTY",
            param_hint="--watch",
        )


def _terminal_columns() -> int:
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def physical_rows(frame: str, columns: int) -> int:
    """Return the terminal rows occupied by a rendered frame."""
    if columns <= 0:
        raise ValueError("terminal columns must be positive")

    return sum(
        max(1, math.ceil(display_width(line) / columns))
        for line in frame.split("\n")
    )


class PlainLineWriter:
    """Write append-only records to an output stream."""

    def __init__(self, output: TextIO) -> None:
        self._output = output

    def write(self, line: str) -> None:
        """Write and flush one complete record."""
        self._output.write(line + "\n")
        self._output.flush()


class TerminalFrameWriter:
    """Write terminal frames to an output stream."""

    def __init__(
        self,
        output: TextIO,
        *,
        columns: Callable[[], int] | None = None,
    ) -> None:
        self._output = output
        self._columns = columns if columns is not None else _terminal_columns
        self._previous_frame: str | None = None

    def write(self, frame: str) -> None:
        """Write and flush one frame."""
        previous_frame = self._previous_frame
        if previous_frame is not None:
            self._erase_frame(previous_frame)
        self._output.write(frame + "\n")
        self._output.flush()
        self._previous_frame = frame

    def _erase_frame(self, frame: str) -> None:
        rows = physical_rows(frame, self._columns())
        controls = [f"\x1b[{rows}F"]
        for index in range(rows):
            controls.append("\x1b[2K")
            if index < rows - 1:
                controls.append("\x1b[1E")
        if rows > 1:
            controls.append(f"\x1b[{rows - 1}F")
        self._output.write("".join(controls))


def watch_sessions(
    client: SessionClient,
    filters: Sequence[str],
    output_format: OutputFormat,
    no_trunc: bool,
    render_sessions: RenderSessions,
    *,
    output: TextIO | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll and render session snapshots until interrupted."""
    destination = sys.stdout if output is None else output
    writer: TerminalFrameWriter | PlainLineWriter
    if output_format is OutputFormat.TABLE:
        writer = TerminalFrameWriter(destination)
    else:
        writer = PlainLineWriter(destination)

    normalized_filters = tuple(filters)
    compact_json = output_format is OutputFormat.JSON
    deadline = monotonic() + WATCH_INTERVAL_SECONDS

    try:
        while True:
            snapshot = client.sessions(normalized_filters)
            rendered = render_sessions(
                snapshot,
                output_format,
                no_trunc,
                compact_json,
            )
            writer.write(rendered)

            now = monotonic()
            if now > deadline:
                deadline = now + WATCH_INTERVAL_SECONDS
            sleep(deadline - now)
            deadline += WATCH_INTERVAL_SECONDS
    except KeyboardInterrupt:
        return

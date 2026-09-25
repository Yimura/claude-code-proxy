"""Terminal frame rendering support for ``ps --watch``."""

from __future__ import annotations

from collections.abc import Callable
import math
import shutil
from typing import TextIO

from .cli_common import display_width


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

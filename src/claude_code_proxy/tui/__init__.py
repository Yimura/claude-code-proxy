"""Interactive live performance dashboard.

The package stays lazy so importing the CLI does not import Textual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(socket_path: Path) -> Any:
    """Import and run the Textual application only on command execution."""
    from .app import run_tui

    return run_tui(socket_path)

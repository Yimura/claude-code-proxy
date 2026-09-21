"""Shared command-line validation, rendering, and control reporting."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
import typer
from wcwidth import wcwidth, wcswidth

from .control.client import ControlUnavailable
from .performance_filters import FILTER_FIELDS
from .text_safety import escaped_text_atom, unicode_escape_atom

MAX_FILTERS = 32
MAX_FILTER_LENGTH = 256
SUPPORTED_FILTERS = FILTER_FIELDS
DISPLAY_ATOMS_PER_CELL = 4
ERROR_DISPLAY_LENGTH = 400


class OutputFormat(str, Enum):
    TABLE = "table"
    JSON = "json"


def load_current_directory_environment() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def validated_filters(
    filters: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if len(filters) > MAX_FILTERS:
        raise typer.BadParameter(
            f"at most {MAX_FILTERS} filters may be supplied",
            param_hint="--filter",
        )
    return tuple(validated_filter(entry) for entry in filters)


def validated_filter(entry: str) -> str:
    if len(entry) > MAX_FILTER_LENGTH:
        raise typer.BadParameter(
            f"filter must be at most {MAX_FILTER_LENGTH} characters",
            param_hint="--filter",
        )
    if entry.count("=") != 1:
        raise typer.BadParameter(
            "filter must contain exactly one '='",
            param_hint="--filter",
        )
    raw_key, raw_value = entry.split("=", 1)
    key = raw_key.strip()
    value = raw_value.strip()
    if not key or not value:
        raise typer.BadParameter(
            "filter key and value must not be empty",
            param_hint="--filter",
        )
    if key not in SUPPORTED_FILTERS:
        choices = ", ".join(sorted(SUPPORTED_FILTERS))
        raise typer.BadParameter(
            f"unsupported filter key {key!r}; choose from {choices}",
            param_hint="--filter",
        )
    return f"{key}={value}"


def render_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    widths = [
        max(
            display_width(header),
            *(display_width(row[index]) for row in rows),
        )
        if rows
        else display_width(header)
        for index, header in enumerate(headers)
    ]
    lines = [table_line(headers, widths)]
    lines.extend(table_line(row, widths) for row in rows)
    return "\n".join(lines)


def table_line(values: tuple[str, ...], widths: list[int]) -> str:
    cells = [
        value + " " * (widths[index] - display_width(value))
        for index, value in enumerate(values)
    ]
    return "  ".join(cells).rstrip()


def display_width(value: str) -> int:
    width = wcswidth(value)
    if width < 0:
        raise ValueError("terminal text has indeterminate display width")
    return width


def terminal_text(
    value: object,
    maximum: int | None = None,
    *,
    ellipsis: bool = True,
) -> str:
    atoms = [terminal_atom(character) for character in str(value)]
    text = "".join(atoms)
    if maximum is None:
        return text

    maximum_atoms = maximum * DISPLAY_ATOMS_PER_CELL
    exceeds_cells = display_width(text) > maximum
    exceeds_atoms = len(atoms) > maximum_atoms
    if not exceeds_cells and not exceeds_atoms:
        return text

    marker = "…" if ellipsis else ""
    cell_budget = maximum - display_width(marker)
    atom_budget = maximum_atoms - (1 if marker else 0)
    selected_atoms = []
    selected = ""
    for atom in atoms[:atom_budget]:
        candidate = selected + atom
        if display_width(candidate) > cell_budget:
            break
        selected_atoms.append(atom)
        selected = candidate
    while selected_atoms and display_width(selected + marker) > maximum:
        selected_atoms.pop()
        selected = "".join(selected_atoms)
    return selected + marker


def terminal_atom(character: str) -> str:
    atom = escaped_text_atom(character)
    if atom != character or wcwidth(character) >= 0:
        return atom
    return unicode_escape_atom(character)


def report_unavailable(
    error: ControlUnavailable,
    *,
    command_name: str = "ps",
) -> None:
    control_guidance(str(error), command_name=command_name)


def control_guidance(message: str, *, command_name: str = "ps") -> None:
    typer.echo(f"Error: {bounded_error(message)}", err=True)
    typer.echo("source: start `claude-code-proxy proxy`", err=True)
    typer.echo(
        "Docker: run `docker compose exec proxy "
        f"claude-code-proxy {command_name}`",
        err=True,
    )


def exit_with_error(message: str) -> None:
    typer.echo(f"Error: {bounded_error(message)}", err=True)
    raise typer.Exit(code=1)


def bounded_error(message: str) -> str:
    return terminal_text(message, maximum=ERROR_DISPLAY_LENGTH)

"""Operator commands for private bounded-nesting authorizations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Annotated

import typer

from .cli_common import (
    OutputFormat,
    exit_with_error,
    load_current_directory_environment,
    render_table,
)
from .control.client import ControlClient, ControlError
from .control.schemas import (
    OrchestrationAuthorizationListResponse,
    OrchestrationAuthorizationResponse,
)
from .control.socket import resolve_socket_path

_WARNING = (
    "Warning: recursive Agent delegation can increase fan-out and token use."
)
_DURATION_PATTERN = re.compile(r"^[0-9]+[smh]?$")
_DURATION_MULTIPLIERS = {"s": 1, "m": 60, "h": 3600}
_HEADERS = ("SESSION", "MAX DEPTH", "EXPIRES IN")

orchestration = typer.Typer(
    name="orchestration",
    no_args_is_help=True,
    help="Manage private bounded-nesting authorizations.",
)


def parse_duration(value: str) -> int:
    """Parse an exact positive integer duration from one CLI token."""
    if len(value) > 6 or not _DURATION_PATTERN.fullmatch(value):
        raise ValueError("duration must be a positive integer with optional s, m, or h")
    unit = value[-1] if value[-1] in _DURATION_MULTIPLIERS else "s"
    digits = value[:-1] if value[-1] in _DURATION_MULTIPLIERS else value
    seconds = int(digits) * _DURATION_MULTIPLIERS[unit]
    if not 1 <= seconds <= 86400:
        raise ValueError("duration must be between 1 second and 24 hours")
    return seconds


@orchestration.command("allow-nesting")
def allow_nesting(
    session_id: Annotated[str, typer.Option("--session-id")],
    max_depth: Annotated[int, typer.Option("--max-depth", min=2, max=8)],
    duration: Annotated[str, typer.Option("--for")] = "60m",
    socket: Annotated[Path | None, typer.Option("--socket")] = None,
) -> None:
    """Authorize bounded recursive Agent delegation for one session."""
    typer.echo(_WARNING, err=True)
    try:
        seconds = parse_duration(duration)
        with _client(socket) as client:
            row = client.allow_nesting(
                session_id,
                max_depth=max_depth,
                duration_seconds=seconds,
            )
        typer.echo(_render_rows((row,), OutputFormat.TABLE))
    except (ControlError, ValueError) as error:
        exit_with_error(str(error))
    except Exception:
        exit_with_error("Unable to manage orchestration authorization")


@orchestration.command("revoke-nesting")
def revoke_nesting(
    session_id: Annotated[str, typer.Option("--session-id")],
    socket: Annotated[Path | None, typer.Option("--socket")] = None,
) -> None:
    """Idempotently revoke recursive Agent delegation for one session."""
    try:
        with _client(socket) as client:
            client.revoke_nesting(session_id)
    except ControlError as error:
        exit_with_error(str(error))
    except Exception:
        exit_with_error("Unable to revoke orchestration authorization")


@orchestration.command("authorizations")
def authorizations(
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format."),
    ] = OutputFormat.TABLE,
    socket: Annotated[Path | None, typer.Option("--socket")] = None,
) -> None:
    """List active bounded-nesting authorizations."""
    try:
        with _client(socket) as client:
            result = client.orchestration_authorizations()
        typer.echo(render_authorizations(result, output_format))
    except ControlError as error:
        exit_with_error(str(error))
    except Exception:
        exit_with_error("Unable to list orchestration authorizations")


def _client(socket: Path | None) -> ControlClient:
    load_current_directory_environment()
    return ControlClient(resolve_socket_path(socket))


def render_authorizations(
    result: OrchestrationAuthorizationListResponse,
    output_format: OutputFormat,
) -> str:
    return _render_rows(result.authorizations, output_format)


def _render_rows(
    rows: tuple[OrchestrationAuthorizationResponse, ...],
    output_format: OutputFormat,
) -> str:
    if output_format is OutputFormat.JSON:
        payload = [row.model_dump(mode="json") for row in rows]
        return json.dumps(payload, indent=2, allow_nan=False)
    rendered = [
        (
            row.session_id,
            str(row.max_depth),
            f"{row.remaining_seconds:g}s",
        )
        for row in rows
    ]
    return render_table(_HEADERS, rendered)

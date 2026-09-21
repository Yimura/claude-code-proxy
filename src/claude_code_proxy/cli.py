"""Command-line entrypoint for running and inspecting the proxy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

import typer

from .cli_common import (
    DISPLAY_ATOMS_PER_CELL as _DISPLAY_ATOMS_PER_CELL,
    ERROR_DISPLAY_LENGTH as _ERROR_DISPLAY_LENGTH,
    MAX_FILTER_LENGTH as _MAX_FILTER_LENGTH,
    MAX_FILTERS as _MAX_FILTERS,
    SUPPORTED_FILTERS as _SUPPORTED_FILTERS,
    OutputFormat,
    bounded_error as _bounded_error,
    control_guidance as _report_control_guidance,
    display_width as _display_width,
    exit_with_error as _exit_with_error,
    load_current_directory_environment as _load_current_directory_environment,
    render_table as _render_generic_table,
    report_unavailable as _report_unavailable,
    table_line as _table_line,
    terminal_atom as _terminal_atom,
    terminal_text as _terminal_text,
    validated_filter as _validated_filter,
    validated_filters as _validated_filters,
)
from .config import PerformanceMode, Settings
from .control.client import (
    ControlClient,
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from .control.schemas import AgentResponse, SessionListResponse, SessionResponse
from .control.socket import resolve_socket_path
from .limits import MAX_CONTROL_INTEGER
from .performance_cli import perf as _performance_report

if TYPE_CHECKING:
    from .runtime import RuntimeServices

_MODEL_DISPLAY_LENGTH = 24
_EFFORT_DISPLAY_LENGTH = 12
_HEADERS = (
    "SESSION / AGENT",
    "MODEL",
    "STATE",
    "EFFORT",
    "CONTEXT",
    "ACTIVE",
    "REQUESTS",
    "LAST SEEN",
)

app = typer.Typer(
    name="claude-code-proxy",
    no_args_is_help=False,
    invoke_without_command=True,
    help="Run and inspect the Anthropic-compatible model proxy.",
)

app.command(name="perf", help="Show or watch performance telemetry.")(_performance_report)


def _configure_proxy_logging() -> None:
    """Configure proxy logging without loading provider code for other commands."""
    from .logging import configure_logging

    configure_logging()


def _load_proxy_runtime() -> tuple[
    Callable[[Settings], RuntimeServices],
    Callable[[RuntimeServices, Path], None],
]:
    """Load heavyweight proxy dependencies only when proxy execution begins."""
    from .runtime import create_runtime
    from .server import run_proxy

    return create_runtime, run_proxy


@app.callback()
def _main(context: typer.Context) -> None:
    """Run and inspect the Anthropic-compatible model proxy."""
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())


@app.command()
def proxy(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Public proxy bind host."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Public proxy bind port."),
    ] = None,
    socket: Annotated[
        Path | None,
        typer.Option("--socket", help="Control Unix socket path."),
    ] = None,
    session_limit: Annotated[
        int | None,
        typer.Option(
            "--session-limit",
            min=0,
            max=MAX_CONTROL_INTEGER,
            help="Maximum retained inactive sessions.",
        ),
    ] = None,
    performance: Annotated[
        PerformanceMode,
        typer.Option(
            "--performance",
            help="Performance mode: off, collector, or logging.",
        ),
    ] = PerformanceMode.OFF,
) -> None:
    """Run the public proxy and local control endpoint in the foreground."""
    try:
        configured = Settings.from_environment()
        effective = _apply_proxy_overrides(
            configured,
            host=host,
            port=port,
            socket=socket,
            session_limit=session_limit,
            performance=performance,
        )
        socket_path = resolve_socket_path(effective.control_socket_path)
        _configure_proxy_logging()
        create_runtime, run_proxy = _load_proxy_runtime()
        runtime = create_runtime(effective)
        run_proxy(runtime, socket_path)
    except KeyboardInterrupt:
        return
    except Exception as error:
        _exit_with_error(str(error))


@app.command()
def ps(
    filters: Annotated[
        list[str] | None,
        typer.Option("--filter", help="Session filter in key=value form; repeatable."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format."),
    ] = OutputFormat.TABLE,
    no_trunc: Annotated[
        bool,
        typer.Option("--no-trunc", help="Do not truncate session or model values."),
    ] = False,
    socket: Annotated[
        Path | None,
        typer.Option("--socket", help="Control Unix socket path."),
    ] = None,
) -> None:
    """List sessions observed by a running proxy."""
    normalized = _validated_filters(filters or ())
    try:
        _load_current_directory_environment()
        socket_path = resolve_socket_path(socket)
        with ControlClient(socket_path) as client:
            result = client.sessions(normalized)
        typer.echo(_render_sessions(result, output_format, no_trunc))
    except ControlUnavailable as error:
        _report_unavailable(error, command_name="ps")
        raise typer.Exit(code=1) from None
    except IncompatibleProtocol as error:
        _report_control_guidance(
            f"Incompatible control API at {socket_path}: {error}",
            command_name="ps",
        )
        raise typer.Exit(code=1) from None
    except ControlError as error:
        _exit_with_error(str(error))
    except Exception as error:
        _exit_with_error(str(error))


def _apply_proxy_overrides(
    configured: Settings,
    *,
    host: str | None,
    port: int | None,
    socket: Path | None,
    session_limit: int | None,
    performance: PerformanceMode,
) -> Settings:
    overrides = {
        "proxy_host": host,
        "proxy_port": port,
        "control_socket_path": socket,
        "session_retention_limit": session_limit,
        "performance_mode": performance,
    }
    supplied = {
        name: value
        for name, value in overrides.items()
        if value is not None
    }
    return replace(configured, **supplied)


def _render_sessions(
    result: SessionListResponse,
    output_format: OutputFormat,
    no_trunc: bool,
) -> str:
    try:
        if output_format is OutputFormat.JSON:
            payload = [item.model_dump(mode="json") for item in result.sessions]
            return json.dumps(payload, indent=2, allow_nan=False)
        return _render_table(result, no_trunc)
    except (TypeError, ValueError, RecursionError) as error:
        raise ControlError(
            "Control API returned an invalid sessions response"
        ) from error


def _render_table(result: SessionListResponse, no_trunc: bool) -> str:
    rows: list[tuple[str, ...]] = []
    for session in result.sessions:
        rows.append(
            _activity_row(
                session,
                session.id,
                result.captured_at,
                no_trunc,
            )
        )
        rows.extend(
            _agent_rows(
                session.agents,
                result.captured_at,
                no_trunc,
            )
        )
    return _render_generic_table(_HEADERS, rows)


def _activity_row(
    activity: SessionResponse | AgentResponse,
    identifier: str,
    captured_at: datetime,
    no_trunc: bool,
    prefix: str = "",
) -> tuple[str, ...]:
    if no_trunc:
        rendered_id = _terminal_text(identifier)
        model = _terminal_text(activity.model)
        effort = _terminal_text(activity.effort)
    else:
        rendered_id = _terminal_text(identifier, maximum=12, ellipsis=False)
        model = _terminal_text(activity.model, maximum=_MODEL_DISPLAY_LENGTH)
        effort = _terminal_text(activity.effort, maximum=_EFFORT_DISPLAY_LENGTH)
    return (
        prefix + rendered_id,
        model,
        _terminal_text(activity.state),
        effort,
        _format_context(activity.context_window),
        _format_count(activity.active_requests),
        _format_count(activity.requests),
        _format_relative_time(activity.last_seen, captured_at),
    )


def _ordered_agents(agents: list[AgentResponse]) -> list[AgentResponse]:
    by_id = sorted(agents, key=lambda item: item.id)
    return sorted(by_id, key=lambda item: item.last_seen, reverse=True)


def _agent_tree(
    agents: tuple[AgentResponse, ...],
) -> tuple[list[AgentResponse], dict[str, list[AgentResponse]]]:
    by_id = {agent.id: agent for agent in agents}
    children: dict[str, list[AgentResponse]] = {}
    roots: list[AgentResponse] = []
    for agent in agents:
        parent_id = agent.parent_id
        if parent_id is None or parent_id == agent.id or parent_id not in by_id:
            roots.append(agent)
            continue
        children.setdefault(parent_id, []).append(agent)
    return _ordered_agents(roots), {
        parent_id: _ordered_agents(items)
        for parent_id, items in children.items()
    }


def _agent_roots(
    roots: list[AgentResponse],
    children: dict[str, list[AgentResponse]],
    agents: tuple[AgentResponse, ...],
) -> list[AgentResponse]:
    candidates = list(roots)
    covered: set[str] = set()

    def mark_descendants(agent_id: str) -> None:
        pending = [agent_id]
        while pending:
            current = pending.pop()
            if current in covered:
                continue
            covered.add(current)
            pending.extend(
                child.id for child in children.get(current, [])
            )

    for root in roots:
        mark_descendants(root.id)
    for agent in _ordered_agents(list(agents)):
        if agent.id not in covered:
            candidates.append(agent)
            mark_descendants(agent.id)
    return candidates


def _agent_rows(
    agents: tuple[AgentResponse, ...],
    captured_at: datetime,
    no_trunc: bool,
) -> list[tuple[str, ...]]:
    roots, children = _agent_tree(agents)
    candidates = _agent_roots(roots, children, agents)
    pending = [
        (agent, "", index == len(candidates) - 1)
        for index, agent in reversed(list(enumerate(candidates)))
    ]
    rows: list[tuple[str, ...]] = []
    visited: set[str] = set()
    while pending:
        agent, prefix, is_last = pending.pop()
        if agent.id in visited:
            continue
        visited.add(agent.id)
        connector = "└─ " if is_last else "├─ "
        rows.append(
            _activity_row(
                agent,
                agent.id,
                captured_at,
                no_trunc,
                prefix + connector,
            )
        )
        descendants = children.get(agent.id, [])
        next_prefix = prefix + ("   " if is_last else "│  ")
        pending.extend(
            (child, next_prefix, index == len(descendants) - 1)
            for index, child in reversed(list(enumerate(descendants)))
        )
    return rows


def _format_context(value: int | None) -> str:
    if value is None:
        return "—"
    if type(value) is not int or not 1 <= value <= MAX_CONTROL_INTEGER:
        return "invalid"
    if value >= 1_000_000:
        millions, remainder = divmod(value, 1_000_000)
        return f"{millions}.{remainder // 100_000}m"
    if value >= 1_000:
        thousands, remainder = divmod(value, 1_000)
        if remainder == 0:
            return f"{thousands}k"
        return f"{thousands}.{remainder // 100}k"
    return str(value)


def _format_count(value: int) -> str:
    if type(value) is not int or not 0 <= value <= MAX_CONTROL_INTEGER:
        return "invalid"
    return str(value)


def _format_relative_time(last_seen: datetime, captured_at: datetime) -> str:
    elapsed = max(0, int((captured_at - last_seen).total_seconds()))
    if elapsed == 0:
        return "now"
    if elapsed < 60:
        return f"{elapsed}s"
    if elapsed < 3_600:
        return f"{elapsed // 60}m"
    if elapsed < 86_400:
        return f"{elapsed // 3_600}h"
    return f"{elapsed // 86_400}d"

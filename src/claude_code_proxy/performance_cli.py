"""Performance reporting for the local control API."""

from __future__ import annotations

from collections.abc import Iterator
import json
import math
from pathlib import Path
from typing import Annotated

import typer

from .cli_common import (
    OutputFormat,
    bounded_error,
    exit_with_error,
    load_current_directory_environment,
    render_table,
    terminal_text,
    validated_filters,
)
from .control.client import (
    ControlClient,
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from .control.schemas import (
    MetricAggregateResponse,
    MetricResponse,
    PerformanceListResponse,
    PerformanceStreamEvent,
    SessionPerformanceResponse,
    SessionPerformanceViewResponse,
)
from .control.socket import resolve_socket_path
from .performance_watch_cli import (
    INVALID_PERFORMANCE_STREAM,
    render_watch_event,
    watch_table_header,
)

_MODEL_DISPLAY_LENGTH = 24
_HEADERS = (
    "SESSION",
    "MODEL",
    "STATE",
    "REQS",
    "ACTIVE",
    "LAST",
    "TTFT",
    "TOKENS",
    "CACHE",
    "TOOLS",
    "RETRIES",
    "RESULT",
)
_INVALID_RESPONSE = "Control API returned an invalid performance response"


def perf(
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
    watch: Annotated[
        bool,
        typer.Option("--watch", help="Stream append-only performance events."),
    ] = False,
) -> None:
    """Show or watch performance data from a running proxy."""
    normalized = validated_filters(filters or ())
    if watch:
        _watch_performance(normalized, output_format, no_trunc, socket)
        return
    try:
        load_current_directory_environment()
        socket_path = resolve_socket_path(socket)
        with ControlClient(socket_path) as client:
            result = client.performance(normalized)
        typer.echo(render_performance(result, output_format, no_trunc))
    except ControlUnavailable as error:
        _performance_guidance(str(error))
        raise typer.Exit(code=1) from None
    except IncompatibleProtocol as error:
        _performance_guidance(
            f"Incompatible control API at {socket_path}: {error}"
        )
        raise typer.Exit(code=1) from None
    except ControlError as error:
        exit_with_error(str(error))
    except Exception as error:
        exit_with_error(str(error))


def _watch_performance(
    filters: tuple[str, ...],
    output_format: OutputFormat,
    no_trunc: bool,
    socket: Path | None,
) -> None:
    try:
        load_current_directory_environment()
        socket_path = resolve_socket_path(socket)
        with ControlClient(socket_path) as client:
            with client.performance_events(filters) as stream:
                _emit_watch_events(stream, output_format, no_trunc)
        typer.echo("Performance stream closed", err=True)
    except KeyboardInterrupt:
        return
    except ControlUnavailable as error:
        _performance_guidance(str(error))
        raise typer.Exit(code=1) from None
    except IncompatibleProtocol as error:
        _performance_guidance(
            f"Incompatible control API at {socket_path}: {error}"
        )
        raise typer.Exit(code=1) from None
    except Exception:
        exit_with_error(INVALID_PERFORMANCE_STREAM)


def _emit_watch_events(
    stream: Iterator[PerformanceStreamEvent],
    output_format: OutputFormat,
    no_trunc: bool,
) -> None:
    header_emitted = False
    for event in stream:
        lines = render_watch_event(event, output_format, no_trunc)
        if output_format is OutputFormat.TABLE and lines and not header_emitted:
            typer.echo(watch_table_header())
            header_emitted = True
        for line in lines:
            typer.echo(line)


def render_performance(
    result: PerformanceListResponse,
    output_format: OutputFormat,
    no_trunc: bool,
) -> str:
    """Render one validated performance snapshot without side effects."""
    try:
        if output_format is OutputFormat.JSON:
            return json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                allow_nan=False,
            )
        return _render_performance_table(result, no_trunc)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ControlError(_INVALID_RESPONSE) from error


def _performance_guidance(message: str) -> None:
    typer.echo(f"Error: {bounded_error(message)}", err=True)
    typer.echo(
        "source: start `claude-code-proxy proxy --performance collector`",
        err=True,
    )
    typer.echo(
        "Docker: configure service command "
        "`claude-code-proxy proxy --performance collector`, recreate the "
        "service, then run "
        "`docker compose exec proxy claude-code-proxy perf`",
        err=True,
    )


def _render_performance_table(
    result: PerformanceListResponse,
    no_trunc: bool,
) -> str:
    rows = [_performance_row(view, no_trunc) for view in result.sessions]
    return render_table(_HEADERS, rows)


def _performance_row(
    view: SessionPerformanceViewResponse,
    no_trunc: bool,
) -> tuple[str, ...]:
    activity = view.session
    performance = view.performance
    latest = performance.latest_request
    identifier = _display_identity(activity.id, no_trunc)
    model = _display_model(activity.model, no_trunc)
    return (
        identifier,
        model,
        terminal_text(activity.state),
        str(performance.requests),
        str(performance.current_concurrency),
        _format_latest_metric(None if latest is None else latest.duration),
        _format_latest_metric(None if latest is None else latest.ttft),
        _format_tokens(performance),
        _format_cache_ratio(performance),
        _format_aggregate(performance.tool_calls),
        _format_aggregate(performance.retries),
        "—" if latest is None else terminal_text(latest.outcome),
    )


def _display_identity(value: str, no_trunc: bool) -> str:
    if no_trunc:
        return terminal_text(value)
    return terminal_text(value, maximum=12, ellipsis=False)


def _display_model(value: str, no_trunc: bool) -> str:
    if no_trunc:
        return terminal_text(value)
    return terminal_text(value, maximum=_MODEL_DISPLAY_LENGTH)


def _format_latest_metric(metric: MetricResponse | None) -> str:
    if metric is None or metric.status != "observed":
        return "—"
    value = _finite_non_negative(metric.value)
    return f"{value:.2f}s"


def _format_tokens(performance: SessionPerformanceResponse) -> str:
    input_metrics = (
        performance.input_tokens,
        performance.cache_read_tokens,
        performance.cache_creation_tokens,
    )
    input_total = _format_aggregate_group(input_metrics)
    output_total = _format_aggregate(performance.output_tokens)
    return f"{input_total} / {output_total}"


def _format_aggregate_group(
    metrics: tuple[MetricAggregateResponse, ...],
) -> str:
    observed = sum(item.observed_samples for item in metrics)
    if observed == 0:
        return "—"
    value = sum(item.value for item in metrics)
    partial = any(item.unavailable_samples for item in metrics)
    return _format_number(value) + ("+?" if partial else "")


def _format_aggregate(metric: MetricAggregateResponse) -> str:
    if metric.observed_samples == 0:
        return "—"
    marker = "+?" if metric.unavailable_samples else ""
    return _format_number(metric.value) + marker


def _format_cache_ratio(performance: SessionPerformanceResponse) -> str:
    cache_metrics = (
        performance.cache_read_tokens,
        performance.cache_creation_tokens,
    )
    if any(item.not_applicable_samples for item in cache_metrics):
        return "—"
    metrics = (performance.input_tokens, *cache_metrics)
    observed = sum(item.observed_samples for item in metrics)
    if observed == 0 or performance.cache_read_tokens.observed_samples == 0:
        return "—"
    total = sum(item.value for item in metrics)
    ratio = 0.0 if total == 0 else performance.cache_read_tokens.value / total * 100
    marker = "+?" if any(item.unavailable_samples for item in metrics) else ""
    return _format_percentage(ratio) + marker


def _format_percentage(value: float) -> str:
    rounded = round(_finite_non_negative(value), 1)
    if rounded.is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.1f}%"


def _format_number(value: int | float) -> str:
    safe = _finite_non_negative(value)
    if isinstance(safe, float) and not safe.is_integer():
        return f"{safe:g}"
    return str(int(safe))


def _finite_non_negative(value: object) -> int | float:
    if type(value) not in (int, float):
        raise ValueError("metric is not numeric")
    if value < 0 or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metric is not finite and non-negative")
    return value

"""Pure rendering for append-only performance event streams."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math

from .cli_common import OutputFormat, table_line, terminal_text
from .control.client import ControlError
from .control.schemas import (
    MetricResponse,
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
    RequestPerformanceResponse,
    SessionPerformanceViewResponse,
)

WATCH_HEADERS = (
    "TIME",
    "SEQ",
    "SESSION",
    "REQUEST",
    "OPERATION",
    "EVENT",
    "ELAPSED",
    "TTFT",
    "TOKENS",
    "TOOLS",
    "RETRIES",
)
INVALID_PERFORMANCE_STREAM = (
    "Control API returned an invalid performance event stream"
)
_COLUMN_WIDTHS = (16, 19, 12, 12, 12, 19, 23, 23, 43, 19, 19)


def watch_table_header() -> str:
    """Render the fixed watch table header."""
    return table_line(WATCH_HEADERS, list(_COLUMN_WIDTHS))


def render_watch_event(
    event: PerformanceStreamEvent,
    output_format: OutputFormat,
    no_trunc: bool,
) -> tuple[str, ...]:
    """Render one validated stream event as zero or more append-only lines."""
    try:
        if output_format is OutputFormat.JSON:
            return (_json_line(event),)
        rows = _table_rows(event, no_trunc)
        return tuple(table_line(row, list(_COLUMN_WIDTHS)) for row in rows)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ControlError(INVALID_PERFORMANCE_STREAM) from error


def _json_line(event: PerformanceStreamEvent) -> str:
    return json.dumps(
        event.model_dump(mode="json"),
        separators=(",", ":"),
        allow_nan=False,
    )


def _table_rows(
    event: PerformanceStreamEvent,
    no_trunc: bool,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(event, PerformanceCursorResponse):
        return ()
    if isinstance(event, PerformanceEventResponse):
        return (_ordinary_row(event, no_trunc),)
    if isinstance(event, PerformanceResetResponse):
        return _reset_rows(event, no_trunc)
    raise ValueError("unsupported performance event")


def _ordinary_row(
    event: PerformanceEventResponse,
    no_trunc: bool,
) -> tuple[str, ...]:
    return _request_row(
        occurred_at=event.occurred_at,
        sequence=event.sequence,
        session_id=event.session_id,
        request=event.request,
        event_type=event.type,
        no_trunc=no_trunc,
    )


def _reset_rows(
    event: PerformanceResetResponse,
    no_trunc: bool,
) -> tuple[tuple[str, ...], ...]:
    if not event.snapshot.sessions:
        return (_empty_reset_row(event),)
    return tuple(
        _snapshot_row(event, view, no_trunc)
        for view in event.snapshot.sessions
    )


def _snapshot_row(
    event: PerformanceResetResponse,
    view: SessionPerformanceViewResponse,
    no_trunc: bool,
) -> tuple[str, ...]:
    request = view.performance.latest_request
    if request is None:
        return _snapshot_without_request(event, view, no_trunc)
    return _request_row(
        occurred_at=event.snapshot.captured_at,
        sequence=event.sequence,
        session_id=view.session.id,
        request=request,
        event_type="SNAPSHOT",
        no_trunc=no_trunc,
    )


def _snapshot_without_request(
    event: PerformanceResetResponse,
    view: SessionPerformanceViewResponse,
    no_trunc: bool,
) -> tuple[str, ...]:
    return (
        _format_time(event.snapshot.captured_at),
        str(event.sequence),
        _identity(view.session.id, no_trunc),
        "—",
        "—",
        "SNAPSHOT",
        "—",
        "—",
        "—",
        "—",
        "—",
    )


def _empty_reset_row(event: PerformanceResetResponse) -> tuple[str, ...]:
    return (
        _format_time(event.snapshot.captured_at),
        str(event.sequence),
        "—",
        "—",
        "—",
        "SNAPSHOT",
        "—",
        "—",
        "—",
        "—",
        "—",
    )


def _request_row(
    *,
    occurred_at: datetime,
    sequence: int,
    session_id: str,
    request: RequestPerformanceResponse,
    event_type: str,
    no_trunc: bool,
) -> tuple[str, ...]:
    return (
        _format_time(occurred_at),
        str(sequence),
        _identity(session_id, no_trunc),
        _identity(request.id, no_trunc),
        terminal_text(request.operation),
        terminal_text(event_type),
        _format_duration(request.duration),
        _format_duration(request.ttft),
        _format_request_tokens(request),
        _format_count(request.tool_calls),
        _format_count(request.retries),
    )


def _identity(value: str, no_trunc: bool) -> str:
    if no_trunc:
        return terminal_text(value)
    return terminal_text(value, maximum=12, ellipsis=False)


def _format_time(value: datetime) -> str:
    utc = value.astimezone(UTC)
    if utc.microsecond == 0:
        return utc.strftime("%H:%M:%SZ")
    fraction = f"{utc.microsecond:06d}".rstrip("0")
    return utc.strftime("%H:%M:%S.") + fraction + "Z"


def _format_duration(metric: MetricResponse) -> str:
    value = _observed_value(metric)
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value}.00s"
    rendered = f"{value:.2f}"
    if len(rendered) > 22:
        rendered = f"{value:g}"
    return rendered + "s"


def _format_count(metric: MetricResponse) -> str:
    value = _observed_value(metric)
    if value is None:
        return "—"
    return _number(value)


def _format_request_tokens(request: RequestPerformanceResponse) -> str:
    if request.operation == "count_tokens":
        return f"{_format_count(request.input_tokens)} / —"
    input_metrics = (
        request.input_tokens,
        request.cache_read_tokens,
        request.cache_creation_tokens,
    )
    values = tuple(_observed_value(metric) for metric in input_metrics)
    input_total = (
        "—" if any(value is None for value in values) else _number(sum(values))
    )
    return f"{input_total} / {_format_count(request.output_tokens)}"


def _number(value: int | float) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("derived metric is not finite")
    if isinstance(value, float) and not value.is_integer():
        return f"{value:g}"
    return str(int(value))


def _observed_value(metric: MetricResponse) -> int | float | None:
    if metric.status != "observed":
        return None
    value = metric.value
    if type(value) not in (int, float):
        raise ValueError("metric is not numeric")
    if value < 0 or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metric is not finite and non-negative")
    return value

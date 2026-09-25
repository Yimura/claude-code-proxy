"""Shared performance values and terminal-safe Rich cells."""

from __future__ import annotations

from datetime import UTC, datetime
import math

from rich.text import Text

from ..cli_common import terminal_text
from ..control.schemas import (
    MetricAggregateResponse,
    MetricResponse,
    RequestPerformanceResponse,
    SessionPerformanceResponse,
)


def safe_cell(
    value: object,
    maximum: int | None = None,
    *,
    style: str | None = None,
) -> Text:
    """Build literal terminal-safe text without parsing Rich markup."""
    return Text(
        terminal_text(value, maximum=maximum),
        style=style,
        no_wrap=True,
        overflow="ellipsis",
    )


def format_latest_metric(metric: MetricResponse | None) -> str:
    """Format a latest duration metric for compact tables."""
    if metric is None or metric.status != "observed":
        return "—"
    return f"{finite_non_negative(metric.value):.2f}s"


def format_metric(metric: MetricResponse) -> str:
    """Format one metric, collapsing missing statuses for compact tables."""
    if metric.status != "observed":
        return "—"
    return format_number(finite_non_negative(metric.value))


def metric_detail(metric: MetricResponse) -> str:
    """Format one metric while preserving its missing-value reason."""
    if metric.status == "unavailable":
        return "unavailable"
    if metric.status == "not_applicable":
        return "not applicable"
    return format_number(finite_non_negative(metric.value))


def format_request_elapsed(
    request: RequestPerformanceResponse | None,
    *,
    now: datetime | None = None,
) -> str:
    """Format request duration, updating active requests from wall time."""
    if request is None:
        return "—"
    if request.outcome != "active":
        return format_latest_metric(request.duration)
    observed_now = now or datetime.now(UTC)
    elapsed = max(0.0, (observed_now - request.started_at).total_seconds())
    return f"{elapsed:.1f}s"


def format_session_tokens(performance: SessionPerformanceResponse) -> str:
    """Format total input-side and output token aggregates."""
    input_metrics = (
        performance.input_tokens,
        performance.cache_read_tokens,
        performance.cache_creation_tokens,
    )
    input_total = format_aggregate_group(input_metrics)
    output_total = format_aggregate(performance.output_tokens)
    return f"{input_total} / {output_total}"


def format_aggregate_group(
    metrics: tuple[MetricAggregateResponse, ...],
) -> str:
    """Format a group of additive aggregates with partial-data marker."""
    if sum(item.observed_samples for item in metrics) == 0:
        return "—"
    value = sum(item.value for item in metrics)
    marker = "+?" if any(item.unavailable_samples for item in metrics) else ""
    return format_number(value) + marker


def format_aggregate(metric: MetricAggregateResponse) -> str:
    """Format one aggregate with the existing partial-data marker."""
    if metric.observed_samples == 0:
        return "—"
    marker = "+?" if metric.unavailable_samples else ""
    return format_number(metric.value) + marker


def cache_ratio_value(
    performance: SessionPerformanceResponse,
) -> float | None:
    """Return an observed cache-read percentage, or None when missing."""
    cache_metrics = (
        performance.cache_read_tokens,
        performance.cache_creation_tokens,
    )
    if any(item.not_applicable_samples for item in cache_metrics):
        return None
    metrics = (performance.input_tokens, *cache_metrics)
    if sum(item.observed_samples for item in metrics) == 0:
        return None
    if performance.cache_read_tokens.observed_samples == 0:
        return None
    total = sum(item.value for item in metrics)
    if total == 0:
        return 0.0
    return performance.cache_read_tokens.value / total * 100


def format_cache_ratio(performance: SessionPerformanceResponse) -> str:
    """Format cache-read percentage with partial-data marker."""
    ratio = cache_ratio_value(performance)
    if ratio is None:
        return "—"
    metrics = (
        performance.input_tokens,
        performance.cache_read_tokens,
        performance.cache_creation_tokens,
    )
    marker = "+?" if any(item.unavailable_samples for item in metrics) else ""
    return format_percentage(ratio) + marker


def format_percentage(value: int | float) -> str:
    """Format a validated non-negative percentage to one decimal at most."""
    rounded = round(finite_non_negative(value), 1)
    if isinstance(rounded, int) or rounded.is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.1f}%"


def format_number(value: int | float) -> str:
    """Format a validated non-negative metric without redundant decimals."""
    safe = finite_non_negative(value)
    if isinstance(safe, float) and not safe.is_integer():
        return f"{safe:g}"
    return str(int(safe))


def finite_non_negative(value: object) -> int | float:
    """Validate the numeric contract shared by all public formatters."""
    if type(value) not in (int, float):
        raise ValueError("metric is not numeric")
    if value < 0 or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metric is not finite and non-negative")
    return value

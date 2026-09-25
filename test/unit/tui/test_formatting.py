from datetime import timedelta

import pytest
from rich.text import Text

from claude_code_proxy.control.schemas import (
    MetricAggregateResponse,
    MetricResponse,
    RequestPerformanceResponse,
)
from claude_code_proxy.tui.formatting import (
    cache_ratio_value,
    finite_non_negative,
    format_aggregate,
    format_aggregate_group,
    format_cache_ratio,
    format_latest_metric,
    format_metric,
    format_number,
    format_percentage,
    format_request_elapsed,
    format_session_tokens,
    metric_detail,
    safe_cell,
)
from test.unit.test_performance_cli import (
    aggregate,
    metric,
    performance_response,
    request_payload,
    view_payload,
)


def observed(value: int | float) -> MetricResponse:
    return MetricResponse.model_validate(metric(value=value))


def aggregate_metric(
    value: int | float = 0,
    *,
    observed_samples: int = 1,
    unavailable: int = 0,
    not_applicable: int = 0,
) -> MetricAggregateResponse:
    return MetricAggregateResponse.model_validate(
        aggregate(
            value,
            observed=observed_samples,
            unavailable=unavailable,
            not_applicable=not_applicable,
        )
    )


def test_formatters_preserve_perf_semantics() -> None:
    view = performance_response(
        view_payload(
            aggregates={
                "input_tokens": aggregate(100),
                "cache_read_tokens": aggregate(40),
                "cache_creation_tokens": aggregate(60),
                "output_tokens": aggregate(25, unavailable=1),
                "tool_calls": aggregate(3, unavailable=1),
            }
        )
    ).sessions[0]

    assert format_latest_metric(observed(1.25)) == "1.25s"
    assert format_session_tokens(view.performance) == "200 / 25+?"
    assert format_cache_ratio(view.performance) == "20%"
    assert cache_ratio_value(view.performance) == 20.0
    assert format_aggregate(view.performance.tool_calls) == "3+?"


def test_single_and_detail_metrics_distinguish_detail_status() -> None:
    unavailable = MetricResponse(status="unavailable", value=None)
    not_applicable = MetricResponse(status="not_applicable", value=None)

    assert format_metric(observed(1.5)) == "1.5"
    assert format_metric(unavailable) == "—"
    assert format_metric(not_applicable) == "—"
    assert metric_detail(unavailable) == "unavailable"
    assert metric_detail(not_applicable) == "not applicable"
    assert metric_detail(observed(2)) == "2"
    assert format_latest_metric(None) == "—"


def test_aggregate_group_marks_partial_and_missing_values() -> None:
    assert format_aggregate(
        aggregate_metric(observed_samples=0, unavailable=1)
    ) == "—"
    assert format_aggregate_group(
        (
            aggregate_metric(3),
            aggregate_metric(2, unavailable=1),
        )
    ) == "5+?"
    assert format_aggregate_group(
        (
            aggregate_metric(observed_samples=0, unavailable=1),
            aggregate_metric(observed_samples=0, not_applicable=1),
        )
    ) == "—"


def test_safe_cell_escapes_terminal_and_never_parses_rich_markup() -> None:
    cell = safe_cell("[bold]x[/bold]\x1b\n\ud800", maximum=40)

    assert isinstance(cell, Text)
    assert cell.plain == "[bold]x[/bold]\\x1b\\x0a\\ud800"
    assert len(cell.spans) == 0
    assert cell.no_wrap is True
    cell.plain.encode("utf-8", errors="strict")


def test_active_elapsed_uses_wall_clock_without_changing_snapshot() -> None:
    payload = request_payload(outcome="active")
    payload["finished_at"] = None
    request = RequestPerformanceResponse.model_validate(payload)
    now = request.started_at + timedelta(seconds=3.5)

    assert format_request_elapsed(request, now=now) == "3.5s"
    assert request.duration.value == 1.25


def test_completed_elapsed_uses_duration_metric() -> None:
    request = RequestPerformanceResponse.model_validate(request_payload())

    assert format_request_elapsed(request) == "1.25s"
    assert format_request_elapsed(None) == "—"


@pytest.mark.parametrize(
    ("value", "number", "percentage"),
    [(0, "0", "0%"), (1, "1", "1%"), (1.25, "1.25", "1.2%")],
)
def test_number_and_percentage_formatting(
    value: int | float,
    number: str,
    percentage: str,
) -> None:
    assert format_number(value) == number
    assert format_percentage(value) == percentage


@pytest.mark.parametrize("value", [True, "1", -1, float("inf"), float("nan")])
def test_numeric_formatters_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite and non-negative|numeric"):
        finite_non_negative(value)

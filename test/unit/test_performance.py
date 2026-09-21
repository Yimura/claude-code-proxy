from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta, timezone
import math
import sys

import pytest

from claude_code_proxy.domain.models import (
    CompletionResponse,
    RedactedThinking,
    RedactedThinkingBlock,
    StreamComplete,
    StreamError,
    StreamStart,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolUseBlock,
    ToolUseStart,
)
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.performance import (
    Measurement,
    RequestPerformance,
    RequestPerformanceSnapshot,
)


STARTED = datetime(2026, 9, 21, 10, tzinfo=UTC)


def request(operation: str = "messages") -> RequestPerformance:
    return RequestPerformance(
        request_id="request-1",
        session_id="session-1",
        operation=operation,
        started_at=STARTED,
        started_monotonic=10.0,
        initial_concurrency=1,
    )


def test_measurement_distinguishes_zero_unavailable_and_not_applicable() -> None:
    zero = Measurement.observed(0)

    assert zero.status == "observed"
    assert zero.value == 0
    assert zero.model_value() == {"status": "observed", "value": 0}
    assert Measurement.unavailable().model_value() == {
        "status": "unavailable",
        "value": None,
    }
    assert Measurement.not_applicable().model_value() == {
        "status": "not_applicable",
        "value": None,
    }


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        -0.1,
        math.nan,
        math.inf,
        -math.inf,
        MAX_CONTROL_INTEGER + 1,
        10**1000,
    ],
)
def test_measurement_rejects_invalid_observed_values(value: object) -> None:
    with pytest.raises(ValueError, match="observed measurement"):
        Measurement.observed(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_non_observed_measurement_rejects_values(status: str) -> None:
    with pytest.raises(ValueError, match="must not have a value"):
        Measurement(status, 0)  # type: ignore[arg-type]


def test_measurement_rejects_unknown_status_and_is_frozen() -> None:
    with pytest.raises(ValueError, match="status"):
        Measurement("missing", None)  # type: ignore[arg-type]

    measurement = Measurement.observed(1)
    with pytest.raises(FrozenInstanceError):
        measurement.value = 2  # type: ignore[misc]


def test_snapshot_has_exact_safe_fields_and_is_frozen() -> None:
    expected = {
        "id",
        "session_id",
        "operation",
        "outcome",
        "started_at",
        "finished_at",
        "duration",
        "upstream_duration",
        "ttft",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "tool_calls",
        "retries",
        "peak_concurrency",
        "reasoning_continuation",
        "failure",
    }
    snapshot = request().snapshot(10.0)

    assert {field.name for field in fields(RequestPerformanceSnapshot)} == expected
    with pytest.raises(FrozenInstanceError):
        snapshot.outcome = "completed"  # type: ignore[misc]


def test_messages_begin_with_required_metric_states() -> None:
    snapshot = request().snapshot(10.0)

    assert snapshot.outcome == "active"
    assert snapshot.finished_at is None
    assert snapshot.duration == Measurement.observed(0.0)
    assert snapshot.upstream_duration == Measurement.unavailable()
    assert snapshot.ttft == Measurement.unavailable()
    assert snapshot.input_tokens == Measurement.unavailable()
    assert snapshot.output_tokens == Measurement.unavailable()
    assert snapshot.cache_read_tokens == Measurement.unavailable()
    assert snapshot.cache_creation_tokens == Measurement.unavailable()
    assert snapshot.reasoning_tokens == Measurement.unavailable()
    assert snapshot.tool_calls == Measurement.observed(0)
    assert snapshot.retries == Measurement.unavailable()
    assert snapshot.peak_concurrency == Measurement.observed(1)
    assert snapshot.reasoning_continuation == "unavailable"


def test_count_tokens_has_exact_not_applicable_metrics() -> None:
    performance = request("count_tokens")
    performance.record_usage(TokenUsage(12, 99, 8, 7, thinking_tokens=6))
    snapshot = performance.snapshot(11.0)

    assert snapshot.input_tokens == Measurement.observed(12)
    assert snapshot.ttft == Measurement.not_applicable()
    assert snapshot.output_tokens == Measurement.not_applicable()
    assert snapshot.cache_read_tokens == Measurement.not_applicable()
    assert snapshot.cache_creation_tokens == Measurement.not_applicable()
    assert snapshot.reasoning_tokens == Measurement.not_applicable()
    assert snapshot.tool_calls == Measurement.not_applicable()
    assert snapshot.reasoning_continuation == "not_applicable"


@pytest.mark.parametrize(
    "event",
    [
        StreamStart(),
        TextDelta(""),
        ToolInputDelta("slot-secret", '{"credential":"secret"}'),
    ],
)
def test_non_semantic_stream_events_do_not_set_ttft(event: object) -> None:
    performance = request()

    assert performance.observe_stream_event(event, 11.0) is False  # type: ignore[arg-type]
    assert performance.snapshot(12.0).ttft == Measurement.unavailable()


def test_first_text_sets_ttft_once_and_later_tool_start_only_counts_tool() -> None:
    performance = request()

    assert performance.observe_stream_event(TextDelta("hello"), 11.25) is True
    assert performance.observe_stream_event(
        ToolUseStart("slot-secret", "tool-id-secret", "tool-name-secret"),
        12.0,
    ) is False
    snapshot = performance.snapshot(13.0)

    assert snapshot.ttft == Measurement.observed(1.25)
    assert snapshot.tool_calls == Measurement.observed(1)


@pytest.mark.parametrize(
    "event",
    [
        RedactedThinking("reasoning-secret"),
        ToolUseStart("slot-secret", "tool-id-secret", "tool-name-secret"),
    ],
)
def test_redacted_thinking_and_tool_start_can_be_first_output(event: object) -> None:
    performance = request()

    assert performance.observe_stream_event(event, 10.5) is True  # type: ignore[arg-type]
    assert performance.snapshot(11.0).ttft == Measurement.observed(0.5)


def test_stream_snapshot_never_retains_tool_or_content_details() -> None:
    performance = request()
    performance.observe_stream_event(TextDelta("text-secret"), 10.1)
    performance.observe_stream_event(RedactedThinking("reasoning-secret"), 10.2)
    performance.observe_stream_event(
        ToolUseStart("slot-secret", "tool-id-secret", "tool-name-secret"),
        10.3,
    )
    performance.observe_stream_event(
        ToolInputDelta("slot-secret", '{"password":"credential-secret"}'),
        10.4,
    )

    rendered = repr(performance.snapshot(11.0))

    for secret in (
        "text-secret",
        "reasoning-secret",
        "slot-secret",
        "tool-id-secret",
        "tool-name-secret",
        "credential-secret",
    ):
        assert secret not in rendered


def test_stream_complete_records_only_selectively_observed_usage() -> None:
    performance = request()
    usage = TokenUsage(
        input_tokens=3,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        thinking_tokens=0,
        observed_fields=frozenset({"input_tokens", "output_tokens"}),
    )

    performance.observe_stream_event(StreamComplete("end_turn", usage), 12.0)
    snapshot = performance.snapshot(12.0)

    assert snapshot.input_tokens == Measurement.observed(3)
    assert snapshot.output_tokens == Measurement.observed(0)
    assert snapshot.cache_read_tokens == Measurement.unavailable()
    assert snapshot.cache_creation_tokens == Measurement.unavailable()
    assert snapshot.reasoning_tokens == Measurement.unavailable()


def test_usage_update_is_atomic_when_a_later_metric_is_invalid() -> None:
    performance = request()
    performance.record_usage(TokenUsage(7, 5, 2, 3, thinking_tokens=1))
    before = performance.snapshot(10.0)
    malformed = TokenUsage(
        input_tokens=99,
        output_tokens=MAX_CONTROL_INTEGER + 1,
        observed_fields=frozenset({"input_tokens", "output_tokens"}),
    )

    with pytest.raises(ValueError, match="observed measurement"):
        performance.record_usage(malformed)

    assert performance.snapshot(10.0) == before


def test_count_tokens_usage_rejection_preserves_prior_input() -> None:
    performance = request("count_tokens")
    performance.record_usage(TokenUsage(7, 0))
    before = performance.snapshot(10.0)
    malformed = TokenUsage(
        input_tokens=10**1000,
        output_tokens=0,
        observed_fields=frozenset({"input_tokens"}),
    )

    with pytest.raises(ValueError, match="observed measurement"):
        performance.record_usage(malformed)

    assert performance.snapshot(10.0) == before


def test_stream_error_retains_only_safe_diagnostic() -> None:
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "provider_error",
        provider_code="rate_limit_exceeded",
    )
    performance = request()

    performance.observe_stream_event(
        StreamError(message="api-key-secret", diagnostic=diagnostic),
        11.0,
    )
    snapshot = performance.snapshot(11.0)

    assert snapshot.failure is diagnostic
    assert "api-key-secret" not in repr(snapshot)


def test_response_detects_meaningful_blocks_usage_and_tool_count_safely() -> None:
    response = CompletionResponse(
        id="provider-response-secret",
        model="provider-model-secret",
        content=(
            TextBlock(""),
            TextBlock("response-secret"),
            RedactedThinkingBlock("reasoning-secret"),
            ToolUseBlock(
                "tool-id-secret",
                "tool-name-secret",
                {"password": "tool-input-secret"},
            ),
        ),
        stop_reason="tool_use",
        usage=TokenUsage(7, 5, 2, 3, thinking_tokens=1),
    )
    performance = request()

    assert performance.observe_response(response, 11.0) is True
    snapshot = performance.snapshot(12.0)

    assert snapshot.ttft == Measurement.observed(1.0)
    assert snapshot.input_tokens == Measurement.observed(7)
    assert snapshot.output_tokens == Measurement.observed(5)
    assert snapshot.cache_creation_tokens == Measurement.observed(2)
    assert snapshot.cache_read_tokens == Measurement.observed(3)
    assert snapshot.reasoning_tokens == Measurement.observed(1)
    assert snapshot.tool_calls == Measurement.observed(1)
    rendered = repr(snapshot)
    for secret in (
        "provider-response-secret",
        "provider-model-secret",
        "response-secret",
        "reasoning-secret",
        "tool-id-secret",
        "tool-name-secret",
        "tool-input-secret",
    ):
        assert secret not in rendered


def test_response_without_meaningful_blocks_does_not_set_ttft() -> None:
    performance = request()
    response = CompletionResponse("id", "model", (TextBlock(""),), None, TokenUsage(1, 0))

    assert performance.observe_response(response, 11.0) is False
    assert performance.snapshot(11.0).ttft == Measurement.unavailable()


def test_retry_support_transitions_from_unavailable_to_zero_and_increments() -> None:
    performance = request()

    assert performance.snapshot(10.0).retries == Measurement.unavailable()
    performance.mark_retries_supported()
    assert performance.snapshot(10.0).retries == Measurement.observed(0)
    performance.record_retry()
    performance.record_retry()
    assert performance.snapshot(10.0).retries == Measurement.observed(2)


def test_concurrency_retains_only_the_peak() -> None:
    performance = request()

    performance.set_concurrency(4)
    performance.set_concurrency(2)

    assert performance.snapshot(10.0).peak_concurrency == Measurement.observed(4)


def test_reasoning_continuation_can_be_recorded() -> None:
    performance = request()

    performance.set_reasoning_continuation("restored")

    assert performance.snapshot(10.0).reasoning_continuation == "restored"
    with pytest.raises(ValueError, match="reasoning continuation"):
        performance.set_reasoning_continuation("secret")  # type: ignore[arg-type]


def test_active_and_finished_upstream_and_total_timing() -> None:
    performance = request()
    assert performance.mark_upstream_started(10.5) is True

    active = performance.snapshot(12.0)
    assert active.duration == Measurement.observed(2.0)
    assert active.upstream_duration == Measurement.observed(1.5)

    assert performance.mark_upstream_finished(12.5) is True
    assert performance.finish(
        "completed",
        STARTED + timedelta(seconds=4),
        14.0,
    ) is True
    finished = performance.snapshot(100.0)

    assert finished.duration == Measurement.observed(4.0)
    assert finished.upstream_duration == Measurement.observed(2.0)
    assert finished.finished_at == STARTED + timedelta(seconds=4)


def test_finish_marks_ongoing_upstream_finished_and_clamps_negative_durations() -> None:
    performance = request()
    performance.mark_upstream_started(12.0)

    performance.finish("failed", STARTED, 9.0)
    snapshot = performance.snapshot(100.0)

    assert snapshot.duration == Measurement.observed(0.0)
    assert snapshot.upstream_duration == Measurement.observed(0.0)


def test_finish_is_idempotent_and_freezes_first_terminal_state() -> None:
    first_failure = FailureDiagnostic(
        FailureCategory.INTERNAL,
        FailureStage.ROUTE,
        "first_failure",
    )
    performance = request()

    assert performance.finish("failed", STARTED + timedelta(seconds=1), 11.0, first_failure)
    assert not performance.finish("cancelled", STARTED + timedelta(seconds=2), 12.0)
    snapshot = performance.snapshot(50.0)

    assert snapshot.outcome == "failed"
    assert snapshot.finished_at == STARTED + timedelta(seconds=1)
    assert snapshot.duration == Measurement.observed(1.0)
    assert snapshot.failure is first_failure


def test_completed_rejects_explicit_failure_without_mutation() -> None:
    diagnostic = FailureDiagnostic(
        FailureCategory.INTERNAL,
        FailureStage.ROUTE,
        "unexpected_failure",
    )
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="non-failed outcome"):
        performance.finish("completed", STARTED, 10.0, diagnostic)

    assert performance.snapshot(10.0) == before


def test_completed_rejects_prior_stream_failure_without_mutation() -> None:
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.STREAM,
        "provider_error",
    )
    performance = request()
    performance.observe_stream_event(StreamError(diagnostic=diagnostic), 10.0)
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="non-failed outcome"):
        performance.finish("completed", STARTED, 10.0)

    assert performance.snapshot(10.0) == before


@pytest.mark.parametrize("outcome", ["cancelled", "client_disconnected"])
def test_other_nonfailed_outcomes_reject_diagnostics(outcome: str) -> None:
    diagnostic = FailureDiagnostic(
        FailureCategory.INTERNAL,
        FailureStage.ROUTE,
        "unexpected_failure",
    )
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="non-failed outcome"):
        performance.finish(outcome, STARTED, 10.0, diagnostic)  # type: ignore[arg-type]

    assert performance.snapshot(10.0) == before


@pytest.mark.parametrize("outcome", ["active", "unknown"])
def test_finish_rejects_non_terminal_outcomes(outcome: str) -> None:
    with pytest.raises(ValueError, match="terminal outcome"):
        request().finish(outcome, STARTED, 10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_constructor_rejects_non_finite_monotonic_time(invalid: float) -> None:
    with pytest.raises(ValueError, match="started_monotonic"):
        RequestPerformance("id", "session", "messages", STARTED, invalid, 1)


def test_constructor_rejects_huge_monotonic_integer_as_value_error() -> None:
    with pytest.raises(ValueError, match="started_monotonic"):
        RequestPerformance("id", "session", "messages", STARTED, 10**1000, 1)


@pytest.mark.parametrize("value", [-1e308, 1e308])
def test_constructor_rejects_finite_times_with_unsafe_duration_magnitude(
    value: float,
) -> None:
    assert math.isfinite(value)
    assert abs(value) > sys.float_info.max / 2

    with pytest.raises(ValueError, match="started_monotonic"):
        RequestPerformance("id", "session", "messages", STARTED, value, 1)


def test_finish_rejects_unsafe_finite_time_without_mutation() -> None:
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="finished_monotonic"):
        performance.finish("completed", STARTED, 1e308)

    assert performance.snapshot(10.0) == before


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda item: item.mark_upstream_started(1e308), "upstream start"),
        (
            lambda item: item.observe_stream_event(TextDelta("first"), 1e308),
            "stream event time",
        ),
        (lambda item: item.snapshot(1e308), "snapshot time"),
    ],
)
def test_timing_paths_reject_unsafe_finite_values_before_mutation(
    operation, error: str
) -> None:
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match=error):
        operation(performance)

    assert performance.snapshot(10.0) == before


def test_upstream_finish_rejects_unsafe_finite_time_without_mutation() -> None:
    performance = request()
    performance.mark_upstream_started(10.0)
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="upstream finish"):
        performance.mark_upstream_finished(1e308)

    assert performance.snapshot(10.0) == before


def test_snapshot_rejects_huge_time_as_value_error_without_mutation() -> None:
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="snapshot time"):
        performance.snapshot(10**1000)

    assert performance.snapshot(10.0) == before


def test_finish_rejects_huge_time_as_value_error_without_mutation() -> None:
    performance = request()
    before = performance.snapshot(10.0)

    with pytest.raises(ValueError, match="finished_monotonic"):
        performance.finish("completed", STARTED, 10**1000)

    assert performance.snapshot(10.0) == before


@pytest.mark.parametrize(
    "invalid",
    [datetime(2026, 9, 21, 10), datetime(2026, 9, 21, 10, tzinfo=timezone(timedelta(hours=1)))],
)
def test_constructor_rejects_non_utc_started_at(invalid: datetime) -> None:
    with pytest.raises(ValueError, match="started_at"):
        RequestPerformance("id", "session", "messages", invalid, 10.0, 1)


def test_finish_validates_clocks_without_mutating_active_request() -> None:
    performance = request()

    with pytest.raises(ValueError, match="finished_at"):
        performance.finish("completed", datetime(2026, 9, 21, 10), 11.0)
    with pytest.raises(ValueError, match="finished_monotonic"):
        performance.finish("completed", STARTED, math.nan)

    assert performance.snapshot(11.0).outcome == "active"


def test_constructor_validates_operation_and_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="operation"):
        RequestPerformance("id", "session", "invalid", STARTED, 10.0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_concurrency"):
        RequestPerformance("id", "session", "messages", STARTED, 10.0, 0)
    with pytest.raises(ValueError, match="concurrency"):
        request().set_concurrency(0)


def test_upstream_timing_is_recorded_once_and_requires_start() -> None:
    performance = request()

    with pytest.raises(ValueError, match="before upstream start"):
        performance.mark_upstream_finished(10.5)
    assert performance.mark_upstream_started(11.0) is True
    assert performance.mark_upstream_started(12.0) is False
    assert performance.mark_upstream_finished(13.0) is True
    assert performance.mark_upstream_finished(14.0) is False
    assert performance.snapshot(20.0).upstream_duration == Measurement.observed(2.0)

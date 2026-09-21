from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta
import math

import pytest

from claude_code_proxy.domain.models import TextDelta, TokenUsage, ToolInputDelta
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.performance import (
    Measurement,
    MetricAggregate,
    RequestPerformance,
    SessionPerformance,
    SessionPerformanceSnapshot,
    _Aggregate,
)


STARTED = datetime(2026, 9, 21, 10, tzinfo=UTC)


def session_request(
    index: int,
    *,
    session_id: str = "session-1",
    operation: str = "messages",
    request_id: str | None = None,
    started_offset: int | None = None,
) -> RequestPerformance:
    offset = index if started_offset is None else started_offset
    return RequestPerformance(
        request_id=request_id or f"request-{index}",
        session_id=session_id,
        operation=operation,  # type: ignore[arg-type]
        started_at=STARTED + timedelta(seconds=offset),
        started_monotonic=float(offset),
        initial_concurrency=1,
    )


def finish_request(
    item: RequestPerformance,
    index: int,
    outcome: str = "completed",
) -> None:
    assert item.finish(
        outcome,  # type: ignore[arg-type]
        STARTED + timedelta(seconds=index + 1),
        float(index + 1),
    )


def test_metric_aggregate_validates_values_counts_and_partial_state() -> None:
    complete = MetricAggregate(3, 1, 0, 2)
    partial = MetricAggregate(1.5, 1, 1, 0)

    assert complete.partial is False
    assert partial.partial is True
    with pytest.raises(FrozenInstanceError):
        partial.value = 2  # type: ignore[misc]

    for value in (True, -1, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            MetricAggregate(value, 0, 0, 0)  # type: ignore[arg-type]
    for count in (True, -1, MAX_CONTROL_INTEGER + 1):
        with pytest.raises(ValueError):
            MetricAggregate(0, count, 0, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            MetricAggregate(0, 0, count, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            MetricAggregate(0, 0, 0, count)  # type: ignore[arg-type]

    boundary = MetricAggregate(
        MAX_CONTROL_INTEGER + 1,
        MAX_CONTROL_INTEGER,
        MAX_CONTROL_INTEGER,
        MAX_CONTROL_INTEGER,
    )
    assert boundary.value == MAX_CONTROL_INTEGER + 1


def test_aggregate_add_is_atomic_on_invalid_internal_measurement() -> None:
    aggregate = _Aggregate()
    aggregate.add(Measurement.observed(4))
    before = aggregate.snapshot()
    malformed = object.__new__(Measurement)
    object.__setattr__(malformed, "status", "observed")
    object.__setattr__(malformed, "value", -1)

    with pytest.raises(ValueError):
        aggregate.add(malformed)

    assert aggregate.snapshot() == before


def test_session_retains_latest_twenty_and_lifetime_aggregates() -> None:
    session = SessionPerformance("session-1")

    for index in range(25):
        item = session_request(index)
        session.start(item)
        item.record_usage(TokenUsage(index + 1, 2))
        finish_request(item, index)
        assert session.add_finalized(item) is not None

    snapshot = session.snapshot(30.0)

    assert snapshot.requests == 25
    assert len(snapshot.recent_requests) == 20
    assert [item.id for item in snapshot.recent_requests] == [
        f"request-{index}" for index in range(24, 4, -1)
    ]
    assert snapshot.input_tokens.value == sum(range(1, 26))
    assert snapshot.input_tokens.observed_samples == 25
    assert snapshot.outcomes == {"completed": 25}


def test_session_aggregate_marks_mixed_observed_and_unavailable_as_partial() -> None:
    session = SessionPerformance("session-1")
    observed = session_request(1)
    unavailable = session_request(2)

    session.start(observed)
    observed.record_usage(TokenUsage(7, 2))
    finish_request(observed, 1)
    session.add_finalized(observed)
    session.start(unavailable)
    finish_request(unavailable, 2)
    session.add_finalized(unavailable)

    aggregate = session.snapshot(4.0).input_tokens
    assert aggregate.value == 7
    assert aggregate.observed_samples == 1
    assert aggregate.unavailable_samples == 1
    assert aggregate.not_applicable_samples == 0
    assert aggregate.partial is True


def test_session_counts_unavailable_and_not_applicable_generation_metrics() -> None:
    session = SessionPerformance("session-1")
    message = session_request(1)
    count_tokens = session_request(2, operation="count_tokens")

    session.start(message)
    finish_request(message, 1)
    session.add_finalized(message)
    session.start(count_tokens)
    count_tokens.record_usage(TokenUsage(3, 99))
    finish_request(count_tokens, 2)
    session.add_finalized(count_tokens)

    snapshot = session.snapshot(4.0)
    assert snapshot.input_tokens == MetricAggregate(3, 1, 1, 0)
    assert snapshot.output_tokens == MetricAggregate(0, 0, 1, 1)
    assert snapshot.cache_read_tokens == MetricAggregate(0, 0, 1, 1)
    assert snapshot.cache_creation_tokens == MetricAggregate(0, 0, 1, 1)
    assert snapshot.reasoning_tokens == MetricAggregate(0, 0, 1, 1)
    assert snapshot.tool_calls == MetricAggregate(0, 1, 0, 1)
    assert snapshot.retries == MetricAggregate(0, 0, 2, 0)


def test_overlapping_requests_capture_current_and_peak_concurrency() -> None:
    session = SessionPerformance("session-1")
    first = session_request(1)
    second = session_request(2)

    assert session.start(first) == 1
    assert session.start(second) == 2
    overlap = session.snapshot(2.5)

    assert overlap.current_concurrency == 2
    assert overlap.peak_concurrency == 2
    assert all(
        item.peak_concurrency == Measurement.observed(2)
        for item in overlap.active_requests
    )

    finish_request(second, 2)
    session.add_finalized(second)
    after = session.snapshot(4.0)
    assert after.current_concurrency == 1
    assert after.peak_concurrency == 2
    assert after.active_requests[0].peak_concurrency == Measurement.observed(2)


def test_terminal_pending_requests_are_not_live_or_concurrent() -> None:
    session = SessionPerformance("session-1")
    first = session_request(1)
    second = session_request(2)

    assert session.start(first) == 1
    first.record_usage(TokenUsage(2, 1))
    finish_request(first, 1)

    pending = session.snapshot(2.5)
    assert pending.requests == 1
    assert pending.active_requests == ()
    assert pending.current_concurrency == 0
    assert pending.latest_request is None

    assert session.start(second) == 1
    live = session.snapshot(3.0)
    assert [item.id for item in live.active_requests] == ["request-2"]
    assert live.current_concurrency == 1
    assert live.peak_concurrency == 1
    assert live.active_requests[0].peak_concurrency == Measurement.observed(1)

    second.record_usage(TokenUsage(3, 1))
    finish_request(second, 2)
    assert session.snapshot(4.0).current_concurrency == 0
    assert session.add_finalized(first) is not None
    assert session.add_finalized(second) is not None

    finalized = session.snapshot(4.0)
    assert finalized.current_concurrency == 0
    assert finalized.active_requests == ()
    assert finalized.requests == 2
    assert finalized.outcomes == {"completed": 2}
    assert finalized.input_tokens == MetricAggregate(5, 2, 0, 0)
    assert [item.id for item in finalized.recent_requests] == [
        "request-2",
        "request-1",
    ]


def test_lifetime_aggregate_value_can_exceed_per_request_control_limit() -> None:
    session = SessionPerformance("session-1")
    first = session_request(1)
    second = session_request(2)

    for item, index, input_tokens in (
        (first, 1, MAX_CONTROL_INTEGER),
        (second, 2, 1),
    ):
        session.start(item)
        item.record_usage(TokenUsage(input_tokens, 0))
        finish_request(item, index)
        assert session.add_finalized(item) is not None

    snapshot = session.snapshot(4.0)
    assert snapshot.input_tokens == MetricAggregate(
        MAX_CONTROL_INTEGER + 1,
        2,
        0,
        0,
    )
    assert snapshot.outcomes == {"completed": 2}
    assert snapshot.current_concurrency == 0
    assert len(snapshot.recent_requests) == 2


def test_duplicate_start_is_rejected_without_mutation() -> None:
    session = SessionPerformance("session-1")
    original = session_request(1, request_id="duplicate")
    duplicate = session_request(2, request_id="duplicate")
    session.start(original)
    before = session.snapshot(3.0)

    with pytest.raises(ValueError, match="duplicate"):
        session.start(duplicate)

    assert session.snapshot(3.0) == before


def test_finalization_requires_exact_active_object_and_happens_once() -> None:
    session = SessionPerformance("session-1")
    active = session_request(1, request_id="request-shared")
    different = session_request(1, request_id="request-shared")
    unknown = session_request(2, request_id="request-unknown")
    session.start(active)
    for item, index in ((active, 1), (different, 1), (unknown, 2)):
        item.record_usage(TokenUsage(5, 2))
        finish_request(item, index)

    before = session.snapshot(4.0)
    assert session.add_finalized(different) is None
    assert session.add_finalized(unknown) is None
    assert session.snapshot(4.0) == before

    finalized = session.add_finalized(active)
    assert finalized is not None
    once = session.snapshot(4.0)
    assert once.input_tokens == MetricAggregate(5, 1, 0, 0)
    assert session.add_finalized(active) is None
    assert session.snapshot(4.0) == once


def test_add_finalized_rejects_active_request_without_mutation() -> None:
    session = SessionPerformance("session-1")
    active = session_request(1)
    session.start(active)
    before = session.snapshot(2.0)

    with pytest.raises(ValueError, match="terminal"):
        session.add_finalized(active)

    assert session.snapshot(2.0) == before


def test_ring_eviction_does_not_change_outcomes_or_aggregates() -> None:
    session = SessionPerformance("session-1", history_limit=2)

    for index, outcome in enumerate(("completed", "failed", "completed")):
        item = session_request(index)
        session.start(item)
        item.record_usage(TokenUsage(2, 1))
        finish_request(item, index, outcome)
        session.add_finalized(item)

    snapshot = session.snapshot(5.0)
    assert [item.id for item in snapshot.recent_requests] == [
        "request-2",
        "request-1",
    ]
    assert snapshot.outcomes == {"completed": 2, "failed": 1}
    assert snapshot.input_tokens == MetricAggregate(6, 3, 0, 0)


def test_session_snapshot_has_deterministic_order_and_latest_request() -> None:
    session = SessionPerformance("session-1")
    older = session_request(1, request_id="request-old")
    tied_a = session_request(2, request_id="request-a", started_offset=3)
    tied_b = session_request(3, request_id="request-b", started_offset=3)
    for item in (older, tied_a, tied_b):
        session.start(item)

    active = session.snapshot(4.0)
    assert [item.id for item in active.active_requests] == [
        "request-b",
        "request-a",
        "request-old",
    ]
    assert active.latest_request is active.active_requests[0]

    finish_request(older, 4)
    session.add_finalized(older)
    finish_request(tied_a, 5)
    session.add_finalized(tied_a)
    finalized = session.snapshot(7.0)
    assert [item.id for item in finalized.recent_requests] == [
        "request-a",
        "request-old",
    ]
    assert finalized.latest_request is finalized.recent_requests[0]


def test_session_snapshot_is_deeply_immutable_and_copied() -> None:
    session = SessionPerformance("session-1")
    empty = session.snapshot(0.0)
    item = session_request(1)
    session.start(item)
    finish_request(item, 1)
    session.add_finalized(item)
    populated = session.snapshot(3.0)

    assert isinstance(populated.active_requests, tuple)
    assert isinstance(populated.recent_requests, tuple)
    assert empty.outcomes == {}
    with pytest.raises(TypeError):
        populated.outcomes["completed"] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        populated.requests = 2  # type: ignore[misc]


def test_session_snapshot_has_exact_safe_fields() -> None:
    expected = {
        "session_id",
        "requests",
        "active_requests",
        "recent_requests",
        "outcomes",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "tool_calls",
        "retries",
        "current_concurrency",
        "peak_concurrency",
        "latest_request",
    }

    assert {field.name for field in fields(SessionPerformanceSnapshot)} == expected


@pytest.mark.parametrize("session_id", [None, "", "   ", "\t\n"])
def test_session_rejects_invalid_session_id(session_id: object) -> None:
    with pytest.raises(ValueError, match="session"):
        SessionPerformance(session_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "history_limit",
    [True, False, 0, -1, MAX_CONTROL_INTEGER + 1],
)
def test_session_rejects_invalid_history_limit(history_limit: object) -> None:
    with pytest.raises(ValueError, match="history_limit"):
        SessionPerformance("session-1", history_limit)  # type: ignore[arg-type]


def test_session_accepts_control_integer_history_boundary() -> None:
    session = SessionPerformance("session-1", MAX_CONTROL_INTEGER)

    assert session.snapshot(0.0).requests == 0


def test_session_start_validates_request_state_and_session_atomically() -> None:
    session = SessionPerformance("session-1")
    wrong_session = session_request(1, session_id="session-2")
    terminal = session_request(2)
    finish_request(terminal, 2)

    with pytest.raises(ValueError, match="session"):
        session.start(wrong_session)
    with pytest.raises(ValueError, match="active"):
        session.start(terminal)

    snapshot = session.snapshot(4.0)
    assert snapshot.requests == 0
    assert snapshot.active_requests == ()
    assert snapshot.peak_concurrency == 0


def test_request_coordination_properties_and_final_snapshot_are_read_only() -> None:
    item = session_request(1)

    assert item.request_id == "request-1"
    assert item.session_id == "session-1"
    assert item.is_terminal is False
    with pytest.raises(ValueError, match="terminal"):
        item.final_snapshot()

    item.record_usage(TokenUsage(4, 2))
    finish_request(item, 1)
    assert item.is_terminal is True
    assert item.final_snapshot() == item.snapshot(2.0)
    with pytest.raises(AttributeError):
        item.request_id = "changed"  # type: ignore[misc]


def test_session_snapshot_rejects_invalid_time_without_mutation() -> None:
    session = SessionPerformance("session-1")
    item = session_request(1)
    session.start(item)
    before = session.snapshot(2.0)

    for invalid in (True, math.nan, math.inf, -math.inf, 10**1000):
        with pytest.raises(ValueError, match="snapshot time"):
            session.snapshot(invalid)  # type: ignore[arg-type]

    assert session.snapshot(2.0) == before


def test_session_history_never_retains_sensitive_request_content() -> None:
    session = SessionPerformance("session-1")
    item = session_request(1)
    session.start(item)
    item.observe_stream_event(TextDelta("content-secret"), 1.1)
    item.observe_stream_event(
        ToolInputDelta("slot-secret", '{"credential":"credential-secret"}'),
        1.2,
    )
    finish_request(item, 1)
    session.add_finalized(item)

    rendered = repr(session.snapshot(3.0))
    for secret in ("content-secret", "slot-secret", "credential-secret"):
        assert secret not in rendered


def test_session_request_lookup_returns_only_exact_internal_reducer() -> None:
    session = SessionPerformance("session-1")
    item = session_request(1)
    session.start(item)

    assert session.request("request-1") is item
    assert session.request("missing") is None

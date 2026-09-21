from datetime import UTC, datetime

import pytest

from claude_code_proxy.domain.models import (
    CompletionResponse,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolInputDelta,
    ToolUseStart,
)
from claude_code_proxy.event_journal import EventJournal
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import (
    AmbiguousSessionId,
    ObservationHandle,
    SessionRegistry,
)
from test.unit.observability_test_support import Clock, metadata, registry


def test_observation_handle_preserves_all_legacy_positional_fields() -> None:
    required = ("key", "request", "public", 1.0, False, False)
    handles = [
        ObservationHandle(*required),
        ObservationHandle(*required, "agent-key"),
        ObservationHandle(*required, "agent-key", "agent-public"),
        ObservationHandle(
            *required, "agent-key", "agent-public", "parent-public"
        ),
        ObservationHandle(
            *required,
            "agent-key",
            "agent-public",
            "parent-public",
            True,
        ),
    ]

    assert handles[0].operation == "messages"
    assert handles[1].agent_key == "agent-key"
    assert handles[2].agent_public_id == "agent-public"
    assert handles[3].parent_agent_public_id == "parent-public"
    assert handles[4].agent_is_new is True
    assert all(handle.operation == "messages" for handle in handles)


def test_observation_handle_accepts_explicit_operation_keyword() -> None:
    handle = ObservationHandle(
        "key",
        "request",
        "public",
        1.0,
        False,
        False,
        operation="count_tokens",
    )

    assert handle.operation == "count_tokens"


@pytest.mark.asyncio
async def test_performance_begin_and_finish_publish_safe_shared_snapshots() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)

    handle = sessions.begin(metadata("raw-session-secret"))
    capture = sessions.performance_snapshots()
    replay = events.subscribe(after=0)

    assert capture.cursor == 1
    assert capture.sessions[0].session.id == handle.public_id
    assert capture.sessions[0].performance.session_id == handle.public_id
    assert replay.replay[0].type == "request_started"
    assert replay.replay[0].session_id == handle.public_id
    assert replay.replay[0].request_id == handle.request_id
    assert "raw-session-secret" not in repr(capture)
    assert "raw-session-secret" not in repr(replay.replay)

    terminal = sessions.finish(handle, "completed")
    repeated = sessions.finish(handle, "failed")

    assert terminal is not None
    assert terminal.outcome == "completed"
    assert repeated is None
    assert events.current_sequence == 2
    assert (await replay.receive(1)).type == "completed"
    replay.close()


def test_performance_overlap_keeps_base_and_reducer_concurrency_coherent() -> None:
    clock = Clock()
    sessions = registry(clock)
    first = sessions.begin(metadata())
    second = sessions.begin(metadata())

    active = sessions.performance_snapshots().sessions[0]

    assert active.session.active_requests == 2
    assert active.session.state == "active"
    assert active.performance.current_concurrency == 2
    assert active.performance.peak_concurrency == 2
    assert {item.id for item in active.performance.active_requests} == {
        first.request_id,
        second.request_id,
    }

    sessions.finish(first, "completed")
    one_left = sessions.performance_snapshots().sessions[0]
    assert one_left.session.active_requests == 1
    assert one_left.performance.current_concurrency == 1


@pytest.mark.parametrize(
    ("outcome", "base_result", "base_state"),
    [
        ("completed", "completed", "idle"),
        ("failed", "failed", "failed"),
        ("cancelled", "failed", "failed"),
        ("client_disconnected", "failed", "failed"),
    ],
)
def test_detailed_outcomes_map_to_existing_base_contract(
    outcome: str,
    base_result: str,
    base_state: str,
) -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())

    terminal = sessions.finish(handle, outcome)  # type: ignore[arg-type]
    view = sessions.performance_snapshots().sessions[0]

    assert terminal is not None
    assert terminal.outcome == outcome
    assert view.performance.outcomes == {outcome: 1}
    assert view.session.last_result == base_result
    assert view.session.state == base_state


def test_sequence_exhaustion_keeps_begin_and_finish_state_atomic() -> None:
    clock = Clock()
    begin_events = EventJournal()
    begin_events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]
    begin_sessions = registry(clock, events=begin_events)

    with pytest.raises(ValueError, match="sequence"):
        begin_sessions.begin(metadata())
    assert begin_sessions.snapshots() == []

    finish_events = EventJournal()
    finish_sessions = registry(clock, events=finish_events)
    handle = finish_sessions.begin(metadata())
    before = finish_sessions.performance_snapshots().sessions
    finish_events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        finish_sessions.finish(handle, "completed")

    assert finish_sessions.performance_snapshots().sessions == before
    assert finish_sessions.snapshots()[0].state == "active"


@pytest.mark.asyncio
async def test_progress_is_rate_bounded_but_first_output_is_immediate() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.stream_event(TextDelta("content-secret"))
    for _ in range(100):
        observer.stream_event(ToolInputDelta("slot-secret", "input-secret"))

    assert (await subscription.receive(1)).type == "first_output"
    assert await subscription.receive(0) is None

    clock.advance(0.25)
    observer.stream_event(ToolInputDelta("slot-secret", "input-secret"))
    progress = await subscription.receive(1)

    assert progress.type == "progress"
    assert "content-secret" not in repr(progress)
    assert "slot-secret" not in repr(progress)
    assert "input-secret" not in repr(progress)
    subscription.close()


@pytest.mark.asyncio
async def test_first_tool_start_publishes_first_output_then_tool_use_safely() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.stream_event(
        ToolUseStart("slot-secret", "tool-id-secret", "tool-name-secret")
    )

    published = [
        await subscription.receive(1),
        await subscription.receive(1),
    ]
    assert [item.type for item in published] == ["first_output", "tool_use"]
    rendered = repr(published)
    for secret in ("slot-secret", "tool-id-secret", "tool-name-secret"):
        assert secret not in rendered
    subscription.close()


def test_observer_records_retry_usage_count_tokens_and_reasoning() -> None:
    clock = Clock()
    sessions = registry(clock)
    messages_handle = sessions.begin(metadata())
    messages = sessions.observer(messages_handle)

    messages.upstream_started()
    clock.advance(0.5)
    messages.upstream_finished()
    messages.mark_retries_supported()
    messages.record_retry()
    messages.set_reasoning_continuation("restored")
    messages.response(
        CompletionResponse(
            "response-secret",
            "model-secret",
            (TextBlock("answer-secret"),),
            "end_turn",
            TokenUsage(7, 5),
        )
    )
    sessions.finish(messages_handle, "completed")

    count_handle = sessions.begin(metadata(), operation="count_tokens")
    sessions.observer(count_handle).count_tokens(13)
    sessions.finish(count_handle, "completed")
    snapshot = sessions.performance_snapshots().sessions[0].performance
    count_snapshot, message_snapshot = snapshot.recent_requests

    assert message_snapshot.retries.value == 1
    assert message_snapshot.reasoning_continuation == "restored"
    assert message_snapshot.input_tokens.value == 7
    assert message_snapshot.output_tokens.value == 5
    assert message_snapshot.upstream_duration.value == 0.5
    assert count_snapshot.input_tokens.value == 13
    assert count_snapshot.output_tokens.status == "not_applicable"
    assert "answer-secret" not in repr(snapshot)
    assert "response-secret" not in repr(snapshot)


@pytest.mark.parametrize("value", [True, -1, MAX_CONTROL_INTEGER + 1])
def test_count_tokens_rejects_invalid_values_without_mutation(value: object) -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata(), operation="count_tokens")
    before = sessions.performance_snapshots().sessions

    with pytest.raises(ValueError, match="count_tokens"):
        sessions.observer(handle).count_tokens(value)  # type: ignore[arg-type]

    assert sessions.performance_snapshots().sessions == before


def test_capture_reuses_snapshot_filters_retention_and_latest_twenty() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=1)
    active = sessions.begin(metadata("active"))
    evicted = sessions.begin(metadata("evicted"))
    sessions.finish(evicted, "completed")
    for index in range(25):
        handle = sessions.begin(metadata("history"), operation="count_tokens")
        sessions.observer(handle).count_tokens(index)
        sessions.finish(handle, "completed")

    history = sessions.performance_snapshots({"session_id": ["history"]})
    active_capture = sessions.performance_snapshots({"state": ["ACTIVE"]})

    assert len(history.sessions[0].performance.recent_requests) == 20
    assert history.sessions[0].performance.input_tokens.value == sum(range(25))
    assert active_capture.sessions[0].session.id == active.public_id
    assert sessions.performance_snapshots(
        {"session_id": ["evicted"]}
    ).sessions == ()


@pytest.mark.asyncio
async def test_performance_subscription_capture_replay_reset_and_live_delivery() -> None:
    clock = Clock()
    events = EventJournal(capacity=2)
    sessions = registry(clock, events=events)
    first = sessions.begin(metadata("first"))

    fresh = sessions.subscribe_performance(None, after=None)
    assert fresh.initial is not None
    assert fresh.initial.cursor == 1

    retained = sessions.subscribe_performance(None, after=0)
    assert retained.initial is None
    assert [event.sequence for event in retained.subscription.replay] == [1]

    sessions.finish(first, "completed")
    second = sessions.begin(metadata("second"))
    sessions.finish(second, "completed")
    stale = sessions.subscribe_performance(None, after=1)
    assert stale.subscription.reset_required is True
    assert stale.initial is not None
    assert stale.initial.cursor == 4

    live = sessions.begin(metadata("live"))
    delivered = await fresh.subscription.receive(1)
    assert delivered.request_id == first.request_id
    while delivered.request_id != live.request_id:
        delivered = await fresh.subscription.receive(1)
    assert delivered.type == "request_started"

    for item in (fresh, retained, stale):
        item.subscription.close()


def test_begin_uses_supplied_clocks_and_rejects_invalid_values_atomically() -> None:
    clock = Clock()
    sessions = registry(clock)
    supplied_wall = datetime(2026, 2, 3, tzinfo=UTC)

    handle = sessions.begin(
        metadata("supplied"),
        started_at=supplied_wall,
        started_monotonic=7.5,
    )
    started = sessions.performance_snapshots(
        {"session_id": ["supplied"]}
    ).sessions[0].performance.active_requests[0]

    assert handle.operation == "messages"
    assert handle.started_monotonic == 7.5
    assert started.started_at == supplied_wall
    before = sessions.performance_snapshots().sessions

    with pytest.raises(ValueError, match="started_at"):
        sessions.begin(
            metadata("invalid-wall"),
            started_at=datetime(2026, 2, 3),
        )
    with pytest.raises(ValueError, match="started_monotonic"):
        sessions.begin(
            metadata("invalid-monotonic"),
            started_monotonic=float("nan"),
        )

    assert sessions.performance_snapshots().sessions == before


@pytest.mark.asyncio
async def test_retry_publication_is_immediate_and_safe() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    subscription = events.subscribe(after=1)

    observer.mark_retries_supported()
    assert await subscription.receive(0) is None
    observer.record_retry()
    retry = await subscription.receive(1)

    assert retry.type == "retry"
    assert retry.request.retries.value == 1
    subscription.close()


def test_performance_capture_rejects_ambiguous_public_id_prefix() -> None:
    clock = Clock()
    sessions = registry(clock, inactive_limit=20)
    prefixes: dict[str, str] = {}
    collision = None
    for index in range(17):
        handle = sessions.begin(metadata(f"capture-session-{index}"))
        prefix = handle.public_id[0]
        if prefix in prefixes:
            collision = prefix
            break
        prefixes[prefix] = handle.public_id

    assert collision is not None
    with pytest.raises(AmbiguousSessionId, match=collision):
        sessions.performance_snapshots({"id": [collision.upper()]})


def test_finish_rejects_unsafe_finite_clock_without_registry_mutation() -> None:
    clock = Clock()
    sessions = registry(clock)
    handle = sessions.begin(metadata())
    before = sessions.performance_snapshots().sessions
    clock.monotonic = 1e308

    with pytest.raises(ValueError, match="finished_monotonic"):
        sessions.finish(handle, "completed")

    clock.monotonic = 100.0
    assert sessions.performance_snapshots().sessions == before
    assert sessions.snapshots()[0].state == "active"


@pytest.mark.asyncio
async def test_first_tool_start_commits_two_final_sequences_as_one_batch() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    observer = sessions.observer(sessions.begin(metadata()))
    events._sequence = MAX_CONTROL_INTEGER - 2  # type: ignore[attr-defined]
    subscription = events.subscribe(after=MAX_CONTROL_INTEGER - 2)

    observer.stream_event(ToolUseStart("slot", "tool-id", "tool-name"))

    published = subscription.replay + (
        await subscription.receive(1),
        await subscription.receive(1),
    )
    assert [item.type for item in published] == ["first_output", "tool_use"]
    assert [item.sequence for item in published] == [
        MAX_CONTROL_INTEGER - 1,
        MAX_CONTROL_INTEGER,
    ]
    subscription.close()


def test_first_tool_start_capacity_failure_changes_no_reducer_state() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    before = sessions.performance_snapshots().sessions
    retained_before = tuple(events._events)  # type: ignore[attr-defined]
    events._sequence = MAX_CONTROL_INTEGER - 1  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        observer.stream_event(ToolUseStart("slot", "tool-id", "tool-name"))

    events._sequence = 1  # type: ignore[attr-defined]
    assert sessions.performance_snapshots().sessions == before
    assert tuple(events._events) == retained_before  # type: ignore[attr-defined]


def test_retry_capacity_failure_changes_no_reducer_or_progress_state() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    observer.mark_retries_supported()
    before = sessions.performance_snapshots().sessions
    progress_before = dict(
        sessions._records[handle.key].progress_at  # type: ignore[attr-defined]
    )
    events._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        observer.record_retry()

    events._sequence = 1  # type: ignore[attr-defined]
    assert sessions.performance_snapshots().sessions == before
    assert (  # type: ignore[attr-defined]
        sessions._records[handle.key].progress_at == progress_before
    )


@pytest.mark.asyncio
async def test_failed_initial_capture_unregisters_performance_subscription() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, inactive_limit=20, events=events)
    prefixes: dict[str, str] = {}
    collision = None
    for index in range(17):
        handle = sessions.begin(metadata(f"subscribe-session-{index}"))
        prefix = handle.public_id[0]
        if prefix in prefixes:
            collision = prefix
            break
        prefixes[prefix] = handle.public_id

    assert collision is not None
    before = events.subscriber_count
    with pytest.raises(AmbiguousSessionId, match=collision):
        sessions.subscribe_performance({"id": [collision]}, after=None)

    assert events.subscriber_count == before


def _prepare_observer_clock_case(case, sessions, handle, observer) -> None:
    if case == "upstream_finished":
        observer.upstream_started()
    if case == "record_retry":
        observer.mark_retries_supported()


def _invoke_observer_clock_case(case, observer) -> None:
    if case == "upstream_started":
        observer.upstream_started()
    elif case == "upstream_finished":
        observer.upstream_finished()
    elif case == "stream_event":
        observer.stream_event(TextDelta("clock-secret"))
    elif case == "response":
        observer.response(
            CompletionResponse(
                "response-secret",
                "model-secret",
                (TextBlock("clock-secret"),),
                "end_turn",
                TokenUsage(7, 5),
            )
        )
    elif case == "count_tokens":
        observer.count_tokens(7)
    elif case == "record_retry":
        observer.record_retry()
    else:
        observer.set_reasoning_continuation("restored")


@pytest.mark.parametrize(
    "case",
    [
        "upstream_started",
        "upstream_finished",
        "stream_event",
        "response",
        "count_tokens",
        "record_retry",
        "set_reasoning_continuation",
    ],
)
@pytest.mark.parametrize("invalid_clock", ["monotonic", "wall"])
def test_observer_clock_failure_preserves_all_registry_state(
    case: str, invalid_clock: str
) -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    _prepare_observer_clock_case(case, sessions, handle, observer)
    before = sessions.performance_snapshots().sessions
    cursor_before = events.current_sequence
    progress_before = dict(
        sessions._records[handle.key].progress_at  # type: ignore[attr-defined]
    )
    valid_wall = clock.wall
    valid_monotonic = clock.monotonic
    if invalid_clock == "monotonic":
        clock.monotonic = 1e308
    else:
        clock.wall = datetime(2026, 1, 1)

    with pytest.raises(ValueError):
        _invoke_observer_clock_case(case, observer)

    clock.wall = valid_wall
    clock.monotonic = valid_monotonic
    assert sessions.performance_snapshots().sessions == before
    assert events.current_sequence == cursor_before
    assert (  # type: ignore[attr-defined]
        sessions._records[handle.key].progress_at == progress_before
    )


@pytest.mark.parametrize("case", ["count_tokens", "set_reasoning_continuation"])
def test_coalesced_observer_update_validates_clock_before_mutation(case: str) -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    record = sessions._records[handle.key]  # type: ignore[attr-defined]
    record.progress_at[handle.request_id] = 1e308
    before = sessions.performance_snapshots().sessions
    cursor_before = events.current_sequence
    progress_before = dict(record.progress_at)
    clock.monotonic = 1e308

    with pytest.raises(ValueError):
        _invoke_observer_clock_case(case, observer)

    clock.monotonic = 100.0
    assert sessions.performance_snapshots().sessions == before
    assert events.current_sequence == cursor_before
    assert record.progress_at == progress_before


def test_retry_unsafe_monotonic_preserves_observed_zero_and_cursor() -> None:
    clock = Clock()
    events = EventJournal()
    sessions = registry(clock, events=events)
    handle = sessions.begin(metadata())
    observer = sessions.observer(handle)
    observer.mark_retries_supported()
    before = sessions.performance_snapshots().sessions[0].performance
    cursor_before = events.current_sequence
    assert before.active_requests[0].retries.value == 0
    clock.monotonic = 1e308

    with pytest.raises(ValueError):
        observer.record_retry()

    clock.monotonic = 100.0
    after = sessions.performance_snapshots().sessions[0].performance
    assert after.active_requests[0].retries.value == 0
    assert events.current_sequence == cursor_before


def test_performance_capture_rejects_naive_wall_clock() -> None:
    clock = Clock()
    sessions = registry(clock)
    sessions.begin(metadata())
    clock.wall = datetime(2026, 1, 1)

    with pytest.raises(ValueError, match="wall"):
        sessions.performance_snapshots()


def test_sample_clocks_returns_injected_validated_pair() -> None:
    clock = Clock()
    clock.wall = datetime(2000, 1, 1, tzinfo=UTC)
    clock.monotonic = 10.0
    sessions = registry(clock)

    assert sessions.sample_clocks() == (clock.wall, 10.0)


def test_begin_reuses_supplied_monotonic_without_resampling() -> None:
    wall = datetime(2000, 1, 1, tzinfo=UTC)
    calls = 0

    def monotonic_clock():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("monotonic sampled twice")
        return 10.0

    sessions = SessionRegistry(
        inactive_limit=10,
        secret=b"test-secret",
        wall_clock=lambda: wall,
        monotonic_clock=monotonic_clock,
    )
    started_at, started_monotonic = sessions.sample_clocks()

    handle = sessions.begin(
        metadata(),
        started_at=started_at,
        started_monotonic=started_monotonic,
    )

    assert handle.started_monotonic == 10.0
    assert calls == 1

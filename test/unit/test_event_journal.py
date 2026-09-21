from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
import asyncio
import math
from threading import Event, Thread

import pytest

from claude_code_proxy.event_journal import (
    OVERFLOW,
    EventJournal,
    JournalEvent,
    Subscription,
)
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.performance import (
    RequestPerformance,
    RequestPerformanceSnapshot,
    SessionPerformance,
    SessionPerformanceSnapshot,
)

OCCURRED_AT = datetime(2026, 9, 21, 12, tzinfo=UTC)


def snapshots() -> tuple[RequestPerformanceSnapshot, SessionPerformanceSnapshot]:
    request = RequestPerformance(
        request_id="request-1",
        session_id="session-1",
        operation="messages",
        started_at=OCCURRED_AT,
        started_monotonic=10.0,
        initial_concurrency=1,
    )
    session = SessionPerformance("session-1")
    session.start(request)
    return request.snapshot(10.0), session.snapshot(10.0)


def event(event_type: str = "progress") -> JournalEvent:
    request, session = snapshots()
    return JournalEvent(
        sequence=0,
        occurred_at=OCCURRED_AT,
        type=event_type,  # type: ignore[arg-type]
        session_id="safe-session-1",
        request_id="request-1",
        request=request,
        session=session,
    )


def test_publish_assigns_sequence_without_mutating_frozen_input() -> None:
    journal = EventJournal()
    original = event()

    first = journal.publish(original)
    second = journal.publish(event("tool_use"))

    assert first.sequence == 1
    assert second.sequence == 2
    assert original.sequence == 0
    assert first is not original
    with pytest.raises(FrozenInstanceError):
        first.sequence = 3  # type: ignore[misc]


@pytest.mark.asyncio
async def test_replay_is_strictly_after_cursor_and_current_is_empty() -> None:
    journal = EventJournal()
    published = [journal.publish(event()) for _ in range(3)]

    after_one = journal.subscribe(after=1)
    after_current = journal.subscribe(after=3)

    assert after_one.replay == tuple(published[1:])
    assert after_one.reset_required is False
    assert after_current.replay == ()
    assert after_current.reset_required is False
    after_one.close()
    after_current.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after", [True, False, -1, 1.5, "1"])
async def test_subscribe_rejects_malformed_cursor(after: object) -> None:
    journal = EventJournal()

    with pytest.raises(ValueError, match="after"):
        journal.subscribe(after=after)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_subscribe_rejects_future_cursor() -> None:
    journal = EventJournal()
    journal.publish(event())

    with pytest.raises(ValueError, match="current"):
        journal.subscribe(after=2)


@pytest.mark.asyncio
async def test_rollover_requires_reset_only_before_oldest_cursor() -> None:
    journal = EventJournal(capacity=3)
    published = [journal.publish(event()) for _ in range(5)]

    too_old = journal.subscribe(after=1)
    exact = journal.subscribe(after=2)

    assert too_old.replay == ()
    assert too_old.reset_required is True
    assert exact.replay == tuple(published[-3:])
    assert exact.reset_required is False
    too_old.close()
    exact.close()


@pytest.mark.asyncio
async def test_subscribe_registers_before_immediate_publish() -> None:
    journal = EventJournal()
    first = journal.publish(event())

    subscription = journal.subscribe(after=0)
    live = journal.publish(event("first_output"))

    assert subscription.replay == (first,)
    assert await subscription.receive(1) == live
    subscription.close()


@pytest.mark.asyncio
async def test_concurrent_publish_dispatches_live_events_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = EventJournal()
    subscription = journal.subscribe(after=None)
    first_offering = Event()
    release_first = Event()
    second_offered = Event()
    original_offer = Subscription.offer

    def gated_offer(target: Subscription, item: JournalEvent) -> None:
        if target is subscription and item.sequence == 1:
            first_offering.set()
            release_first.wait(1)
        if target is subscription and item.sequence == 2:
            second_offered.set()
        original_offer(target, item)

    monkeypatch.setattr(Subscription, "offer", gated_offer)
    first = Thread(target=lambda: journal.publish(event("request_started")))
    second = Thread(target=lambda: journal.publish(event("progress")))
    first.start()
    assert first_offering.wait(1)
    second.start()
    overtook = second_offered.wait(0.1)
    release_first.set()
    first.join()
    second.join()

    received = [await subscription.receive(1), await subscription.receive(1)]
    assert overtook is False
    assert [item.sequence for item in received if isinstance(item, JournalEvent)] == [
        1,
        2,
    ]
    subscription.close()


class GatedLoop:
    def __init__(self) -> None:
        self.scheduled: list[tuple[object, tuple[object, ...]]] = []

    def call_soon_threadsafe(self, callback: object, *args: object) -> None:
        self.scheduled.append((callback, args))


@pytest.mark.asyncio
async def test_worker_burst_bounds_ingress_and_schedules_one_wake() -> None:
    journal = EventJournal(subscriber_capacity=4)
    subscription = journal.subscribe(after=None)
    gated_loop = GatedLoop()
    object.__setattr__(subscription, "_loop", gated_loop)

    worker = Thread(target=lambda: [journal.publish(event()) for _ in range(100)])
    worker.start()
    worker.join()

    assert subscription.queue_size == 1
    assert len(gated_loop.scheduled) == 1
    assert journal.subscriber_count == 0
    callback, args = gated_loop.scheduled[0]
    callback(*args)  # type: ignore[operator]
    assert await subscription.receive(0) is OVERFLOW
    subscription.close()


@pytest.mark.asyncio
async def test_slow_subscriber_gets_one_terminal_overflow_episode() -> None:
    journal = EventJournal(subscriber_capacity=1)
    subscription = journal.subscribe(after=None)

    for _ in range(100):
        journal.publish(event())
    await asyncio.sleep(0)

    assert subscription.queue_size <= 1
    assert journal.subscriber_count == 0
    assert await subscription.receive(1) is OVERFLOW
    journal.publish(event())
    await asyncio.sleep(0)
    assert await subscription.receive(0) is None
    subscription.close()
    assert journal.subscriber_count == 0


@pytest.mark.asyncio
async def test_closed_subscription_does_not_prevent_other_delivery() -> None:
    journal = EventJournal()
    closed = journal.subscribe(after=None)
    active = journal.subscribe(after=None)
    closed.close()

    published = journal.publish(event())

    assert await active.receive(1) == published
    assert closed.queue_size == 0
    active.close()


@pytest.mark.asyncio
async def test_closed_event_loop_subscription_does_not_prevent_delivery() -> None:
    journal = EventJournal()
    broken_loop = asyncio.new_event_loop()
    broken = journal.subscribe(after=None)
    object.__setattr__(broken, "_loop", broken_loop)
    broken_loop.close()
    active = journal.subscribe(after=None)

    published = journal.publish(event())

    assert await active.receive(1) == published
    assert journal.subscriber_count == 1
    active.close()


@pytest.mark.asyncio
async def test_worker_thread_publish_reaches_captured_loop() -> None:
    journal = EventJournal()
    subscription = journal.subscribe(after=None)
    result: list[JournalEvent] = []

    worker = Thread(target=lambda: result.append(journal.publish(event())))
    worker.start()
    worker.join()

    assert await subscription.receive(1) == result[0]
    subscription.close()


@pytest.mark.asyncio
async def test_receive_timeout_returns_none() -> None:
    subscription = EventJournal().subscribe(after=None)

    assert await subscription.receive(0) is None
    subscription.close()


@pytest.mark.asyncio
async def test_receive_zero_returns_populated_pending_item_synchronously() -> None:
    journal = EventJournal()
    subscription = journal.subscribe(after=None)
    published = journal.publish(event())

    assert await subscription.receive(0) == published
    subscription.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, False, -1, math.nan, math.inf, -math.inf])
async def test_receive_rejects_invalid_timeout(timeout: object) -> None:
    subscription = EventJournal().subscribe(after=None)

    with pytest.raises(ValueError, match="timeout"):
        await subscription.receive(timeout)  # type: ignore[arg-type]
    subscription.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_publish_after_close_does_not_queue() -> None:
    journal = EventJournal()
    subscription = journal.subscribe(after=None)

    subscription.close()
    subscription.close()
    journal.unsubscribe(subscription)
    journal.publish(event())
    await asyncio.sleep(0)

    assert journal.subscriber_count == 0
    assert subscription.queue_size == 0


@pytest.mark.parametrize(
    ("capacity", "subscriber_capacity"),
    [
        (True, 1),
        (False, 1),
        (0, 1),
        (-1, 1),
        (MAX_CONTROL_INTEGER + 1, 1),
        (1, True),
        (1, False),
        (1, 0),
        (1, -1),
        (1, MAX_CONTROL_INTEGER + 1),
    ],
)
def test_invalid_capacities_are_rejected(
    capacity: object, subscriber_capacity: object
) -> None:
    with pytest.raises(ValueError):
        EventJournal(
            capacity=capacity,  # type: ignore[arg-type]
            subscriber_capacity=subscriber_capacity,  # type: ignore[arg-type]
        )


def test_sequence_exhaustion_does_not_mutate_journal() -> None:
    journal = EventJournal()
    journal._sequence = MAX_CONTROL_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        journal.publish(event())

    assert journal.current_sequence == MAX_CONTROL_INTEGER


def test_publish_rejects_an_already_published_event() -> None:
    journal = EventJournal()

    with pytest.raises(ValueError, match="sequence"):
        journal.publish(replace(event(), sequence=1))

    assert journal.current_sequence == 0


def test_publish_rejects_mutable_event_subclass() -> None:
    class MutableEvent(JournalEvent):
        pass

    original = event()
    mutable = MutableEvent(
        original.sequence,
        original.occurred_at,
        original.type,
        original.session_id,
        original.request_id,
        original.request,
        original.session,
    )
    mutable.extra = []
    journal = EventJournal()

    with pytest.raises(ValueError, match="JournalEvent"):
        journal.publish(mutable)

    assert journal.current_sequence == 0


@pytest.mark.parametrize("sequence", [True, -1, MAX_CONTROL_INTEGER + 1])
def test_unpublished_event_rejects_invalid_sequence(sequence: object) -> None:
    with pytest.raises(ValueError, match="sequence"):
        replace(event(), sequence=sequence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "occurred_at",
    [
        datetime(2026, 9, 21, 12),
        datetime(2026, 9, 21, 12, tzinfo=timezone(timedelta(hours=2))),
        "2026-09-21T12:00:00Z",
    ],
)
def test_event_rejects_non_utc_timestamp(occurred_at: object) -> None:
    with pytest.raises(ValueError, match="occurred_at"):
        replace(event(), occurred_at=occurred_at)  # type: ignore[arg-type]


def test_event_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="type"):
        event("unknown")


@pytest.mark.parametrize("field", ["session_id", "request_id"])
@pytest.mark.parametrize("identifier", [None, "", "   ", "unsafe\nidentifier"])
def test_event_rejects_unsafe_identifiers(field: str, identifier: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(event(), **{field: identifier})


def test_event_payloads_are_exact_immutable_snapshot_types() -> None:
    item = event()

    assert type(item.request) is RequestPerformanceSnapshot
    assert type(item.session) is SessionPerformanceSnapshot
    with pytest.raises(FrozenInstanceError):
        item.request.outcome = "completed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        item.session.outcomes["completed"] = 1  # type: ignore[index]


def test_event_rejects_non_snapshot_payloads() -> None:
    request, session = snapshots()

    with pytest.raises(ValueError, match="request"):
        replace(event(), request={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="session"):
        replace(event(), session={})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_subscribe_without_cursor_has_no_replay_or_reset() -> None:
    journal = EventJournal()
    journal.publish(event())

    subscription = journal.subscribe(after=None)

    assert subscription.replay == ()
    assert subscription.reset_required is False
    subscription.close()

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
import asyncio
import math
from types import SimpleNamespace
from threading import Event, Thread

import pytest

import claude_code_proxy.event_journal as event_journal_module
from claude_code_proxy.event_journal import (
    OVERFLOW,
    EventJournal,
    JournalEvent,
    SessionEventIdentity,
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


def test_session_event_identity_copies_and_freezes_snapshot_fields() -> None:
    identity_type = getattr(event_journal_module, "SessionEventIdentity", None)
    assert identity_type is not None
    snapshot = SimpleNamespace(
        id="safe-session-1",
        state="active",
        active_requests=1,
        requests=2,
        client_model="client-model",
        model="provider-model",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
        first_seen=OCCURRED_AT - timedelta(seconds=2),
        last_seen=OCCURRED_AT,
        elapsed_seconds=2.0,
        last_result="completed",
    )

    identity = identity_type.from_snapshot(snapshot)

    assert identity.id == "safe-session-1"
    assert identity.provider == "openai"
    assert identity.active_requests == 1
    with pytest.raises(FrozenInstanceError):
        identity.provider = "vertex"


def test_session_event_identity_factory_escapes_nonprintable_metadata() -> None:
    source = event_identity()
    snapshot = SimpleNamespace(
        **{
            name: getattr(source, name)
            for name in source.__dataclass_fields__
        }
    )
    snapshot.client_model = "model\nforged\x1b‮"

    identity = SessionEventIdentity.from_snapshot(snapshot)

    assert identity.client_model == "model\\x0aforged\\x1b\\u202e"
    assert identity.client_model.isprintable()


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


def event_identity(session_id: str = "safe-session-1") -> SessionEventIdentity:
    return SessionEventIdentity(
        id=session_id,
        state="active",
        active_requests=1,
        requests=1,
        client_model="client-model",
        model="provider-model",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
        first_seen=OCCURRED_AT,
        last_seen=OCCURRED_AT,
        elapsed_seconds=0.0,
        last_result=None,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", " "),
        ("provider", b"openai"),
        ("active_requests", True),
        ("active_requests", -1),
        ("requests", 0),
        ("context_window", 0),
        ("first_seen", datetime(2026, 9, 21, 13, tzinfo=timezone(timedelta(hours=1)))),
        ("last_seen", OCCURRED_AT - timedelta(seconds=1)),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", MAX_CONTROL_INTEGER + 1),
        ("state", "idle"),
        ("last_result", "unknown"),
    ],
)
def test_session_event_identity_rejects_invalid_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(event_identity(), **{field: value})


def event(event_type: str = "progress") -> JournalEvent:
    request, session = snapshots()
    return JournalEvent(
        sequence=0,
        occurred_at=OCCURRED_AT,
        type=event_type,  # type: ignore[arg-type]
        session_id="safe-session-1",
        request_id="request-1",
        activity=event_identity(),
        request=request,
        session=session,
    )


def test_journal_event_carries_matching_immutable_activity() -> None:
    published = event()

    assert published.activity == event_identity()
    with pytest.raises(ValueError, match="activity identity"):
        replace(published, activity=event_identity("other-session"))


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
async def test_concurrent_receivers_wait_for_distinct_events() -> None:
    journal = EventJournal()
    subscription = journal.subscribe(after=None)
    receivers = {
        asyncio.create_task(subscription.receive(10)),
        asyncio.create_task(subscription.receive(10)),
    }
    await asyncio.sleep(0)

    first_event = journal.publish(event("request_started"))
    completed, pending = await asyncio.wait(
        receivers, timeout=1, return_when=asyncio.FIRST_COMPLETED
    )

    assert len(completed) == 1
    assert len(pending) == 1
    assert completed.pop().result() == first_event
    second_event = journal.publish(event("progress"))
    remaining = pending.pop()
    assert await asyncio.wait_for(remaining, timeout=1) == second_event
    subscription.close()


@pytest.mark.asyncio
async def test_close_wakes_blocked_receiver() -> None:
    subscription = EventJournal().subscribe(after=None)
    receiver = asyncio.create_task(subscription.receive(10))
    await asyncio.sleep(0)

    subscription.close()

    assert await asyncio.wait_for(receiver, timeout=1) is None


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
        original.activity,
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


def test_reserved_two_event_batch_uses_final_two_sequences_atomically() -> None:
    journal = EventJournal()
    journal._sequence = MAX_CONTROL_INTEGER - 2  # type: ignore[attr-defined]

    reservation = journal.reserve(2)
    published = reservation.publish(
        (event("first_output"), event("tool_use"))
    )

    assert [item.sequence for item in published] == [
        MAX_CONTROL_INTEGER - 1,
        MAX_CONTROL_INTEGER,
    ]
    assert journal.current_sequence == MAX_CONTROL_INTEGER


def test_reservation_rejects_insufficient_capacity_without_publication() -> None:
    journal = EventJournal()
    journal._sequence = MAX_CONTROL_INTEGER - 1  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="sequence"):
        journal.reserve(2)

    assert journal.current_sequence == MAX_CONTROL_INTEGER - 1
    assert tuple(journal._events) == ()  # type: ignore[attr-defined]


def test_invalid_reserved_batch_leaves_sequence_and_history_unchanged() -> None:
    journal = EventJournal()
    invalid = replace(event("tool_use"), sequence=1)
    reservation = journal.reserve(2)

    with pytest.raises(ValueError, match="sequence"):
        reservation.publish((event("first_output"), invalid))

    assert journal.current_sequence == 0
    assert tuple(journal._events) == ()  # type: ignore[attr-defined]


def test_reservation_blocks_ordinary_publisher_until_batch_commits() -> None:
    journal = EventJournal()
    reservation = journal.reserve(1)
    attempted = Event()
    result: list[JournalEvent] = []

    def publish_ordinary() -> None:
        attempted.set()
        result.append(journal.publish(event("progress")))

    worker = Thread(target=publish_ordinary)
    worker.start()
    assert attempted.wait(1)
    assert worker.is_alive()
    assert journal.current_sequence == 0

    reserved = reservation.publish((event("retry"),))
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert reserved[0].sequence == 1
    assert result[0].sequence == 2


def test_abandoned_reservation_consumes_nothing_and_is_single_use() -> None:
    journal = EventJournal()

    with journal.reserve(1) as reservation:
        pass

    assert journal.current_sequence == 0
    with pytest.raises(ValueError, match="reservation"):
        reservation.publish((event(),))

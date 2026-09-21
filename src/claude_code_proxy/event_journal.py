"""Bounded, sequenced delivery of immutable performance events."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import math
import threading
from typing import Final, Literal, TypeAlias

from .limits import MAX_CONTROL_INTEGER
from .performance import RequestPerformanceSnapshot, SessionPerformanceSnapshot
from .text_safety import escaped_text_atom

EventType: TypeAlias = Literal[
    "request_started",
    "first_output",
    "progress",
    "tool_use",
    "retry",
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
]

_IDENTITY_TEXT_FIELDS = (
    "id",
    "client_model",
    "model",
    "provider",
    "transport",
    "effort",
)
_EVENT_TYPES = frozenset({
    "request_started",
    "first_output",
    "progress",
    "tool_use",
    "retry",
    "completed",
    "failed",
    "cancelled",
    "client_disconnected",
})


class _Overflow:
    __slots__ = ()


OVERFLOW: Final = _Overflow()


@dataclass(frozen=True, slots=True)
class SessionEventIdentity:
    """Immutable public session metadata captured with one journal event."""

    id: str
    state: Literal["active", "idle", "failed"]
    active_requests: int
    requests: int
    client_model: str
    model: str
    provider: str
    transport: str
    effort: str
    context_window: int | None
    first_seen: datetime
    last_seen: datetime
    elapsed_seconds: float
    last_result: Literal["completed", "failed"] | None

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "SessionEventIdentity":
        values = {
            name: getattr(snapshot, name) for name in cls.__dataclass_fields__
        }
        for name in _IDENTITY_TEXT_FIELDS:
            value = values[name]
            if isinstance(value, str):
                values[name] = "".join(escaped_text_atom(item) for item in value)
        return cls(**values)

    def __post_init__(self) -> None:
        _validate_identity_strings(self)
        _validate_identity_counts(self)
        _validate_identity_timing(self)
        _validate_identity_lifecycle(self)


def _validate_identity_strings(identity: SessionEventIdentity) -> None:
    for name in _IDENTITY_TEXT_FIELDS:
        _require_safe_identifier(name, getattr(identity, name))


def _validate_identity_counts(identity: SessionEventIdentity) -> None:
    _require_control_integer("active_requests", identity.active_requests, minimum=0)
    _require_control_integer("requests", identity.requests, minimum=0)
    if identity.active_requests > identity.requests:
        raise ValueError("active requests must not exceed requests")
    if identity.context_window is not None:
        _require_control_integer(
            "context_window", identity.context_window, minimum=1
        )


def _validate_identity_timing(identity: SessionEventIdentity) -> None:
    _require_utc_datetime("first_seen", identity.first_seen)
    _require_utc_datetime("last_seen", identity.last_seen)
    if identity.first_seen > identity.last_seen:
        raise ValueError("first_seen must not exceed last_seen")
    duration = identity.elapsed_seconds
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration < 0
        or not math.isfinite(duration)
        or isinstance(duration, int) and duration > MAX_CONTROL_INTEGER
    ):
        raise ValueError("elapsed_seconds must be finite and non-negative")


def _validate_identity_lifecycle(identity: SessionEventIdentity) -> None:
    if identity.last_result not in (None, "completed", "failed"):
        raise ValueError("invalid last_result")
    expected_state = "idle"
    if identity.active_requests:
        expected_state = "active"
    elif identity.last_result == "failed":
        expected_state = "failed"
    if identity.state != expected_state:
        raise ValueError("state must match activity and last result")


@dataclass(frozen=True, slots=True)
class JournalEvent:
    """An immutable telemetry event before or after journal sequencing."""

    sequence: int
    occurred_at: datetime
    type: EventType
    session_id: str
    request_id: str
    activity: SessionEventIdentity
    request: RequestPerformanceSnapshot
    session: SessionPerformanceSnapshot

    def __post_init__(self) -> None:
        _require_control_integer("sequence", self.sequence, minimum=0)
        _require_utc_datetime("occurred_at", self.occurred_at)
        if not isinstance(self.type, str) or self.type not in _EVENT_TYPES:
            raise ValueError("invalid event type")
        _require_safe_identifier("session_id", self.session_id)
        _require_safe_identifier("request_id", self.request_id)
        if type(self.activity) is not SessionEventIdentity:
            raise ValueError("activity must be an immutable session identity")
        if self.activity.id != self.session_id:
            raise ValueError("activity identity must match event identity")
        if type(self.request) is not RequestPerformanceSnapshot:
            raise ValueError("request must be an immutable performance snapshot")
        if type(self.session) is not SessionPerformanceSnapshot:
            raise ValueError("session must be an immutable performance snapshot")


QueueItem: TypeAlias = JournalEvent | _Overflow


@dataclass(eq=False, slots=True)
class Subscription:
    """An event-loop-bound view of journal replay and live events."""

    replay: tuple[JournalEvent, ...]
    reset_required: bool
    _journal: EventJournal = field(repr=False)
    _capacity: int = field(repr=False)
    _event: asyncio.Event = field(repr=False)
    _loop: asyncio.AbstractEventLoop = field(repr=False)
    _pending: deque[QueueItem] = field(
        default_factory=deque, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)
    _accepting: bool = field(default=True, init=False, repr=False)
    _notified: bool = field(default=False, init=False, repr=False)
    _wake_scheduled: bool = field(default=False, init=False, repr=False)
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    async def receive(self, timeout: float) -> QueueItem | None:
        """Wait up to timeout seconds for one live event or overflow marker."""
        validated = _require_timeout(timeout)
        item = self._pop_pending()
        if item is not None or validated == 0 or self._is_closed():
            return item
        loop = asyncio.get_running_loop()
        deadline = loop.time() + validated
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self._event.wait(), remaining)
            except TimeoutError:
                return None
            item = self._pop_pending()
            if item is not None or self._is_closed():
                return item

    def offer(self, event: JournalEvent) -> None:
        """Add one event to bounded ingress and schedule at most one wake."""
        schedule_wake = False
        unregister = False
        with self._state_lock:
            if self._closed or not self._accepting:
                return
            if len(self._pending) >= self._capacity:
                self._pending.clear()
                self._pending.append(OVERFLOW)
                self._accepting = False
                unregister = True
            else:
                self._pending.append(event)
            if not self._notified:
                self._notified = True
                if not self._wake_scheduled:
                    self._wake_scheduled = True
                    schedule_wake = True
        if unregister:
            self._journal.unsubscribe(self)
        if schedule_wake:
            self._schedule_wake()

    def _schedule_wake(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            self.close()

    def _wake(self) -> None:
        with self._state_lock:
            self._wake_scheduled = False
            should_wake = self._closed or (
                self._notified and bool(self._pending)
            )
        if should_wake:
            self._event.set()

    def _pop_pending(self) -> QueueItem | None:
        with self._state_lock:
            if not self._pending:
                if not self._closed:
                    self._event.clear()
                return None
            item = self._pending.popleft()
            if not self._pending:
                self._notified = False
                self._event.clear()
            return item

    def _is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def close(self) -> None:
        """Close, wake blocked receivers, and unregister idempotently."""
        schedule_wake = False
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
            self._pending.clear()
            self._notified = False
            if not self._wake_scheduled:
                self._wake_scheduled = True
                schedule_wake = True
        self._journal.unsubscribe(self)
        if schedule_wake:
            self._schedule_wake()

    @property
    def queue_size(self) -> int:
        with self._state_lock:
            return len(self._pending)


@dataclass(eq=False, slots=True)
class EventReservation:
    """Hold ordered journal capacity until one batch commits or is abandoned."""

    _journal: EventJournal = field(repr=False)
    count: int
    _active: bool = field(default=True, init=False, repr=False)
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __enter__(self) -> "EventReservation":
        with self._state_lock:
            if not self._active:
                raise ValueError("event reservation is no longer active")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def publish(
        self, events: tuple[JournalEvent, ...]
    ) -> tuple[JournalEvent, ...]:
        self._claim()
        try:
            return self._journal._publish_reserved(self.count, events)
        finally:
            self._journal._dispatch_lock.release()

    def close(self) -> None:
        with self._state_lock:
            if not self._active:
                return
            self._active = False
        self._journal._dispatch_lock.release()

    def _claim(self) -> None:
        with self._state_lock:
            if not self._active:
                raise ValueError("event reservation is no longer active")
            self._active = False


class EventJournal:
    """Retain bounded history and fan out live events without blocking."""

    __slots__ = (
        "_capacity",
        "_subscriber_capacity",
        "_dispatch_lock",
        "_lock",
        "_events",
        "_sequence",
        "_subscribers",
    )

    def __init__(self, capacity: int = 4096, subscriber_capacity: int = 64) -> None:
        _require_control_integer("capacity", capacity, minimum=1)
        _require_control_integer(
            "subscriber_capacity", subscriber_capacity, minimum=1
        )
        self._capacity = capacity
        self._subscriber_capacity = subscriber_capacity
        self._dispatch_lock = threading.Lock()
        self._lock = threading.Lock()
        self._events: deque[JournalEvent] = deque(maxlen=capacity)
        self._sequence = 0
        self._subscribers: set[Subscription] = set()

    @property
    def current_sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def reserve(self, count: int) -> EventReservation:
        """Reserve consecutive event sequences while serializing dispatch."""
        _require_control_integer("reservation count", count, minimum=1)
        self._dispatch_lock.acquire()
        try:
            with self._lock:
                if self._sequence + count > MAX_CONTROL_INTEGER:
                    raise ValueError("event sequence exceeds the control limit")
            return EventReservation(self, count)
        except BaseException:
            self._dispatch_lock.release()
            raise

    def publish(self, event: JournalEvent) -> JournalEvent:
        """Sequence, retain, and offer one unpublished event."""
        return self.reserve(1).publish((event,))[0]

    def _publish_reserved(
        self, count: int, events: tuple[JournalEvent, ...]
    ) -> tuple[JournalEvent, ...]:
        batch = tuple(events)
        _validate_event_batch(batch, count)
        with self._lock:
            published = tuple(
                replace(event, sequence=self._sequence + index)
                for index, event in enumerate(batch, start=1)
            )
            self._sequence += count
            self._events.extend(published)
            subscribers = tuple(self._subscribers)
        for item in published:
            for subscription in subscribers:
                subscription.offer(item)
        return published

    def subscribe(self, after: int | None) -> Subscription:
        """Atomically capture replay and register for subsequent live events."""
        loop = asyncio.get_running_loop()
        if after is not None:
            _require_control_integer("after", after, minimum=0)
        with self._lock:
            replay, reset_required = self._replay_after(after)
            subscription = Subscription(
                replay,
                reset_required,
                self,
                self._subscriber_capacity,
                asyncio.Event(),
                loop,
            )
            self._subscribers.add(subscription)
        return subscription

    def _replay_after(
        self, after: int | None
    ) -> tuple[tuple[JournalEvent, ...], bool]:
        if after is None:
            return (), False
        if after > self._sequence:
            raise ValueError("after must not exceed the current sequence")
        if self._events and after < self._events[0].sequence - 1:
            return (), True
        return tuple(event for event in self._events if event.sequence > after), False

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription if it is currently registered."""
        with self._lock:
            self._subscribers.discard(subscription)


def _validate_event_batch(
    events: tuple[JournalEvent, ...], count: int
) -> None:
    if len(events) != count:
        raise ValueError("reserved event batch must match reservation count")
    for event in events:
        if type(event) is not JournalEvent:
            raise ValueError("publish requires exact JournalEvent values")
        if event.sequence != 0:
            raise ValueError("publish requires events with sequence 0")


def _require_control_integer(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int or not minimum <= value <= MAX_CONTROL_INTEGER:
        raise ValueError(
            f"{name} must be an integer between {minimum} and "
            f"{MAX_CONTROL_INTEGER}"
        )


def _require_utc_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be an aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")


def _require_safe_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or not value.isprintable():
        raise ValueError(f"{name} must be a nonblank printable string")


def _require_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a finite non-negative number")
    try:
        timeout = float(value)
    except (OverflowError, ValueError):
        raise ValueError("timeout must be a finite non-negative number") from None
    if timeout < 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be a finite non-negative number")
    return timeout

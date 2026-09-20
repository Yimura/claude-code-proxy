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
class JournalEvent:
    """An immutable telemetry event before or after journal sequencing."""

    sequence: int
    occurred_at: datetime
    type: EventType
    session_id: str
    request_id: str
    request: RequestPerformanceSnapshot
    session: SessionPerformanceSnapshot

    def __post_init__(self) -> None:
        _require_control_integer("sequence", self.sequence, minimum=0)
        _require_utc_datetime("occurred_at", self.occurred_at)
        if not isinstance(self.type, str) or self.type not in _EVENT_TYPES:
            raise ValueError("invalid event type")
        _require_safe_identifier("session_id", self.session_id)
        _require_safe_identifier("request_id", self.request_id)
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
    _queue: asyncio.Queue[QueueItem] = field(repr=False)
    _loop: asyncio.AbstractEventLoop = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _overflowed: bool = field(default=False, init=False, repr=False)
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    async def receive(self, timeout: float) -> QueueItem | None:
        """Wait up to timeout seconds for one live event or overflow marker."""
        validated = _require_timeout(timeout)
        try:
            return await asyncio.wait_for(self._queue.get(), validated)
        except TimeoutError:
            return None

    def offer(self, event: JournalEvent) -> None:
        """Schedule a nonblocking queue offer on the captured event loop."""
        with self._state_lock:
            if self._closed:
                return
            loop = self._loop
        try:
            loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:
            self.close()

    def _enqueue(self, event: JournalEvent) -> None:
        with self._state_lock:
            if self._closed:
                return
        if self._overflowed:
            self._restore_overflow_marker()
            return
        if not self._queue.full():
            self._queue.put_nowait(event)
            return
        self._overflowed = True
        self._drain_queue()
        self._queue.put_nowait(OVERFLOW)

    def _restore_overflow_marker(self) -> None:
        if self._queue.empty():
            self._queue.put_nowait(OVERFLOW)

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def close(self) -> None:
        """Close and unregister this subscription idempotently."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._journal.unsubscribe(self)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


class EventJournal:
    """Retain bounded history and fan out live events without blocking."""

    __slots__ = (
        "_capacity",
        "_subscriber_capacity",
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

    def publish(self, event: JournalEvent) -> JournalEvent:
        """Sequence, retain, and offer an unpublished event to subscribers."""
        if not isinstance(event, JournalEvent) or event.sequence != 0:
            raise ValueError("publish requires an event with sequence 0")
        with self._lock:
            if self._sequence >= MAX_CONTROL_INTEGER:
                raise ValueError("event sequence exceeds the control limit")
            published = replace(event, sequence=self._sequence + 1)
            self._sequence = published.sequence
            self._events.append(published)
            subscribers = tuple(self._subscribers)
        for subscription in subscribers:
            subscription.offer(published)
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
                asyncio.Queue(maxsize=self._subscriber_capacity),
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

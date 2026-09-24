"""Framework-independent bounded control-stream lifecycle primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading

from ..control.client import (
    ControlClient,
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from ..control.schemas import (
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
)

_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0)
_INVALID_STREAM = "Control API returned an invalid performance event stream"
_STREAM_CLOSED = "Performance stream closed"
_PROTOCOL_MESSAGE = (
    "TUI requires control protocol v1 with performance and "
    "performance_events capabilities"
)
_UNAVAILABLE_MESSAGE = (
    "Unable to connect to the performance event stream after retries"
)


class PendingEvents:
    """Thread-safe bounded coalescing for complete stream replacements."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset: PerformanceResetResponse | None = None
        self._events: dict[str, PerformanceEventResponse] = {}
        self._cursor: PerformanceCursorResponse | None = None

    def offer(self, event: PerformanceStreamEvent) -> None:
        """Retain only the authoritative reset and newest keyed values."""
        with self._lock:
            if isinstance(event, PerformanceResetResponse):
                self._offer_reset(event)
                return
            if self._is_superseded_by_reset(event):
                return
            if isinstance(event, PerformanceEventResponse):
                self._offer_event(event)
                return
            if isinstance(event, PerformanceCursorResponse):
                self._offer_cursor(event)
                return
            raise ValueError("unsupported performance stream event")

    def drain(self) -> tuple[PerformanceStreamEvent, ...]:
        """Atomically return a sequence-ordered batch and clear pending data."""
        with self._lock:
            items = self._ordered_items()
            self._reset = None
            self._events.clear()
            self._cursor = None
            return items

    def _offer_reset(self, event: PerformanceResetResponse) -> None:
        current = self._reset
        if current is not None and current.process != event.process:
            self._events.clear()
            self._cursor = None
        elif current is not None and event.sequence <= current.sequence:
            return
        else:
            self._discard_items_superseded_by(event)
        self._reset = event

    def _discard_items_superseded_by(
        self,
        reset: PerformanceResetResponse,
    ) -> None:
        self._events = {
            session_id: event
            for session_id, event in self._events.items()
            if event.process == reset.process and event.sequence > reset.sequence
        }
        if self._cursor is not None and (
            self._cursor.process != reset.process
            or self._cursor.sequence <= reset.sequence
        ):
            self._cursor = None

    def _is_superseded_by_reset(
        self,
        event: PerformanceEventResponse | PerformanceCursorResponse,
    ) -> bool:
        return self._reset is not None and (
            event.process != self._reset.process
            or event.sequence <= self._reset.sequence
        )

    def _offer_event(self, event: PerformanceEventResponse) -> None:
        current = self._events.get(event.session_id)
        if current is None or event.sequence > current.sequence:
            self._events[event.session_id] = event

    def _offer_cursor(self, event: PerformanceCursorResponse) -> None:
        if self._cursor is None or event.sequence > self._cursor.sequence:
            self._cursor = event

    def _ordered_items(self) -> tuple[PerformanceStreamEvent, ...]:
        items: list[PerformanceStreamEvent] = []
        if self._reset is not None:
            items.append(self._reset)
        items.extend(sorted(self._events.values(), key=lambda item: item.sequence))
        highest = items[-1].sequence if items else -1
        if self._cursor is not None and self._cursor.sequence > highest:
            items.append(self._cursor)
        return tuple(items)


class ConnectionPhase(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


@dataclass(frozen=True, slots=True)
class ConnectionStatus:
    phase: ConnectionPhase
    attempt: int = 0


@dataclass(frozen=True, slots=True)
class AppResult:
    exit_code: int
    message: str = ""


class ConnectionPump:
    """Run the synchronous control event stream with bounded reconnects."""

    def __init__(
        self,
        socket_path: Path,
        *,
        client_factory: Callable[[Path], ControlClient] = ControlClient,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._client_factory = client_factory
        self._stopped = threading.Event()
        self._wait = wait or self._stopped.wait
        self._lock = threading.Lock()
        self._active_client: ControlClient | None = None

    def run(
        self,
        on_event: Callable[[PerformanceStreamEvent], None],
        on_status: Callable[[ConnectionStatus], None],
    ) -> AppResult:
        """Consume streams until stopped, fatally incompatible, or exhausted."""
        on_status(ConnectionStatus(ConnectionPhase.CONNECTING))
        failures = 0
        while not self._stopped.is_set():
            saw_reset = False

            def mark_reset() -> None:
                nonlocal saw_reset
                saw_reset = True

            try:
                self._consume_once(on_event, mark_reset)
                if self._stopped.is_set():
                    return AppResult(0)
                raise ControlError(_STREAM_CLOSED)
            except IncompatibleProtocol:
                return AppResult(1, _PROTOCOL_MESSAGE)
            except (ControlUnavailable, ControlError):
                if saw_reset:
                    failures = 0
                outcome = self._retry(failures, on_status)
                if isinstance(outcome, AppResult):
                    return outcome
                failures = outcome
        return AppResult(0)

    def stop(self) -> None:
        """Interrupt waits and idempotently close the active client."""
        self._stopped.set()
        with self._lock:
            client = self._active_client
        if client is not None:
            client.close()

    def _retry(
        self,
        failures: int,
        on_status: Callable[[ConnectionStatus], None],
    ) -> int | AppResult:
        on_status(ConnectionStatus(ConnectionPhase.DISCONNECTED, failures))
        if failures >= len(_BACKOFF_SECONDS):
            return AppResult(1, _UNAVAILABLE_MESSAGE)
        delay = _BACKOFF_SECONDS[failures]
        next_failure = failures + 1
        on_status(
            ConnectionStatus(ConnectionPhase.RECONNECTING, next_failure)
        )
        if self._wait(delay):
            return AppResult(0)
        return next_failure

    def _consume_once(
        self,
        on_event: Callable[[PerformanceStreamEvent], None],
        on_reset: Callable[[], None],
    ) -> None:
        client = self._client_factory(self._socket_path)
        with self._lock:
            stopped = self._stopped.is_set()
            if not stopped:
                self._active_client = client
        if stopped:
            client.close()
            return
        try:
            with client:
                with client.performance_events() as stream:
                    self._consume_stream(stream, on_event, on_reset)
        finally:
            with self._lock:
                if self._active_client is client:
                    self._active_client = None

    def _consume_stream(
        self,
        stream: Iterator[PerformanceStreamEvent],
        on_event: Callable[[PerformanceStreamEvent], None],
        on_reset: Callable[[], None],
    ) -> None:
        first = True
        for event in stream:
            if first and not isinstance(event, PerformanceResetResponse):
                raise ControlError(_INVALID_STREAM)
            first = False
            on_event(event)
            if isinstance(event, PerformanceResetResponse):
                on_reset()
            if self._stopped.is_set():
                return

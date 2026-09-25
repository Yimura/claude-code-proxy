from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import threading

from claude_code_proxy.control.client import (
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)
from claude_code_proxy.tui.app import (
    AppResult,
    ConnectionPhase,
    ConnectionPump,
    PendingEvents,
)
from test.unit.tui.support import cursor, event, reset, view

_SOCKET_PATH = Path("/run/control.sock")
_PROTOCOL_MESSAGE = (
    "TUI requires control protocol v1 with performance and "
    "performance_events capabilities"
)
_UNAVAILABLE_MESSAGE = (
    "Unable to connect to the performance event stream after retries"
)


def test_pending_events_keep_latest_complete_event_per_session() -> None:
    pending = PendingEvents()
    first = view("first", state="active")
    second = view("second", state="active")
    pending.offer(event(first, event_type="progress", sequence=8))
    pending.offer(event(second, event_type="progress", sequence=9))
    pending.offer(event(first, event_type="tool_use", sequence=10))

    batch = pending.drain()

    assert [(item.sequence, getattr(item, "session_id", None)) for item in batch] == [
        (9, "second"),
        (10, "first"),
    ]
    assert pending.drain() == ()


def test_reset_supersedes_older_pending_and_keeps_newer_events() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(event(active, event_type="progress", sequence=8))
    pending.offer(reset(active, sequence=20))
    pending.offer(event(active, event_type="retry", sequence=21))
    pending.offer(cursor(22))

    batch = pending.drain()

    assert [item.sequence for item in batch] == [20, 21, 22]


def test_new_process_reset_replaces_higher_sequence_pending_reset() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(reset(active, sequence=100, process_pid=41))
    pending.offer(event(active, event_type="progress", sequence=101, process_pid=41))

    pending.offer(reset(active, sequence=0, process_pid=42))

    batch = pending.drain()
    assert [(item.process.pid, item.sequence) for item in batch] == [(42, 0)]


def test_same_process_reset_keeps_already_buffered_newer_items() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(event(active, event_type="progress", sequence=21))
    pending.offer(cursor(22))

    pending.offer(reset(active, sequence=20))

    assert [item.sequence for item in pending.drain()] == [20, 21, 22]


def test_same_process_stale_reset_does_not_replace_pending_reset() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(reset(active, sequence=20))

    pending.offer(reset(active, sequence=19))

    assert [item.sequence for item in pending.drain()] == [20]


def test_pending_reset_rejects_late_items_from_another_process() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(reset(active, sequence=20, process_pid=42))

    pending.offer(event(active, event_type="progress", sequence=21, process_pid=41))
    pending.offer(cursor(22, process_pid=41))

    assert [item.sequence for item in pending.drain()] == [20]


def test_pending_events_ignore_items_not_newer_than_reset() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(reset(active, sequence=20))
    pending.offer(event(active, event_type="progress", sequence=19))
    pending.offer(cursor(20))

    assert [item.sequence for item in pending.drain()] == [20]


def test_pending_events_keep_only_highest_cursor_after_complete_events() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    pending.offer(cursor(10))
    pending.offer(cursor(12))
    pending.offer(event(active, event_type="progress", sequence=11))

    assert [item.sequence for item in pending.drain()] == [11, 12]


def test_pending_events_are_safe_under_concurrent_offers() -> None:
    pending = PendingEvents()
    active = view("session", state="active")
    threads = [
        threading.Thread(
            target=pending.offer,
            args=(event(active, event_type="progress", sequence=sequence),),
        )
        for sequence in range(8, 28)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [item.sequence for item in pending.drain()] == [27]


class FakeStream(Iterator):
    def __init__(self, events: Iterable) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, step: Exception | Iterable) -> None:
        self.step = step
        self.stream: FakeStream | None = None
        self.closed = False
        self.close_calls = 0
        self.enter_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def performance_events(self):
        if isinstance(self.step, Exception):
            raise self.step
        self.stream = FakeStream(self.step)
        return self.stream

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_calls += 1
        if self.stream is not None:
            self.stream.close()


class SequencedClientFactory:
    def __init__(self, steps: Iterable[Exception | Iterable]) -> None:
        self.steps = list(steps)
        self.clients: list[FakeClient] = []

    def __call__(self, socket_path: Path) -> FakeClient:
        assert socket_path == _SOCKET_PATH
        client = FakeClient(self.steps.pop(0))
        self.clients.append(client)
        return client


def unavailable(reason: str = "missing") -> ControlUnavailable:
    return ControlUnavailable(_SOCKET_PATH, reason)


def test_connection_pump_retries_with_exact_backoff_then_fails() -> None:
    waits: list[float] = []
    statuses = []
    factory = SequencedClientFactory([unavailable()] * 5)
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=factory,
        wait=lambda delay: waits.append(delay) or False,
    )

    result = pump.run(lambda item: None, statuses.append)

    assert waits == [0.5, 1.0, 2.0, 4.0]
    assert statuses[0].phase is ConnectionPhase.CONNECTING
    assert sum(status.phase is ConnectionPhase.RECONNECTING for status in statuses) == 4
    assert all(status.phase is not ConnectionPhase.CONNECTED for status in statuses)
    assert result == AppResult(1, _UNAVAILABLE_MESSAGE)


def test_fresh_reset_resets_retry_failure_budget() -> None:
    waits: list[float] = []
    observed = reset(view("session"))
    steps = [
        unavailable("first"),
        [observed],
        unavailable("second"),
        unavailable("third"),
        unavailable("fourth"),
        unavailable("fifth"),
    ]
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=SequencedClientFactory(steps),
        wait=lambda delay: waits.append(delay) or False,
    )

    result = pump.run(lambda item: None, lambda status: None)

    assert waits == [0.5, 0.5, 1.0, 2.0, 4.0]
    assert result == AppResult(1, _UNAVAILABLE_MESSAGE)


def test_stream_eof_after_reset_is_retryable() -> None:
    waits: list[float] = []
    delivered = []
    factory = SequencedClientFactory(
        [
            [reset(view("session"))],
            IncompatibleProtocol("stop after proving retry"),
        ]
    )
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=factory,
        wait=lambda delay: waits.append(delay) or False,
    )

    result = pump.run(delivered.append, lambda status: None)

    assert waits == [0.5]
    assert len(delivered) == 1
    assert result == AppResult(1, _PROTOCOL_MESSAGE)


def test_control_error_is_retryable_and_first_item_must_be_reset() -> None:
    waits: list[float] = []
    active = view("session", state="active")
    factory = SequencedClientFactory(
        [
            [event(active, event_type="progress", sequence=8)],
            ControlError("retryable"),
            IncompatibleProtocol("fatal"),
        ]
    )
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=factory,
        wait=lambda delay: waits.append(delay) or False,
    )

    result = pump.run(lambda item: None, lambda status: None)

    assert waits == [0.5, 1.0]
    assert result == AppResult(1, _PROTOCOL_MESSAGE)


def test_connection_pump_does_not_retry_protocol_mismatch() -> None:
    waits: list[float] = []
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=SequencedClientFactory(
            [IncompatibleProtocol("missing performance_events")]
        ),
        wait=lambda delay: waits.append(delay) or False,
    )

    result = pump.run(lambda item: None, lambda status: None)

    assert waits == []
    assert result == AppResult(1, _PROTOCOL_MESSAGE)


def test_retry_failure_does_not_retain_sensitive_error_details() -> None:
    marker = "Authorization=secret provider body"
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=SequencedClientFactory([ControlError(marker)] * 5),
        wait=lambda delay: False,
    )

    result = pump.run(lambda item: None, lambda status: None)

    assert result == AppResult(1, _UNAVAILABLE_MESSAGE)
    for sensitive in ("Authorization", "secret", "provider body"):
        assert sensitive not in result.message
        assert sensitive not in repr(result)


class BlockingStream(FakeStream):
    def __init__(self, first) -> None:
        super().__init__(())
        self._first = first
        self._sent_first = False
        self.waiting = threading.Event()
        self._closed = threading.Event()

    def __next__(self):
        if not self._sent_first:
            self._sent_first = True
            return self._first
        self.waiting.set()
        self._closed.wait(timeout=2)
        raise StopIteration

    def close(self) -> None:
        super().close()
        self._closed.set()


class BlockingClient(FakeClient):
    def __init__(self, first) -> None:
        super().__init__(())
        self.stream = BlockingStream(first)

    def performance_events(self):
        return self.stream


def test_stop_closes_active_client_once_and_worker_ends() -> None:
    client = BlockingClient(reset(view("session")))
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=lambda path: client,
    )
    results: list[AppResult] = []
    worker = threading.Thread(
        target=lambda: results.append(
            pump.run(lambda item: None, lambda status: None)
        )
    )
    worker.start()
    assert client.stream.waiting.wait(timeout=1)

    pump.stop()
    pump.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert client.close_calls == 1
    assert results == [AppResult(0)]


def test_stop_event_interrupts_backoff_wait() -> None:
    reconnecting = threading.Event()
    pump = ConnectionPump(
        _SOCKET_PATH,
        client_factory=SequencedClientFactory([unavailable()]),
    )
    results: list[AppResult] = []

    def status_changed(status) -> None:
        if status.phase is ConnectionPhase.RECONNECTING:
            reconnecting.set()

    worker = threading.Thread(
        target=lambda: results.append(pump.run(lambda item: None, status_changed))
    )
    worker.start()
    assert reconnecting.wait(timeout=1)

    pump.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert results == [AppResult(0)]


def test_stop_during_client_factory_closes_without_opening_client() -> None:
    client = FakeClient(())
    pump: ConnectionPump

    def stopping_factory(socket_path: Path) -> FakeClient:
        assert socket_path == _SOCKET_PATH
        pump.stop()
        return client

    pump = ConnectionPump(_SOCKET_PATH, client_factory=stopping_factory)

    result = pump.run(lambda item: None, lambda status: None)

    assert result == AppResult(0)
    assert client.close_calls == 1
    assert client.enter_calls == 0
    assert client.stream is None

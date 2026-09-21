from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import json
from pathlib import Path

import httpx
import pytest

from claude_code_proxy.control import client as client_module
from claude_code_proxy.control.client import (
    ControlClient,
    ControlError,
    IncompatibleProtocol,
)
from claude_code_proxy.control.schemas import (
    PerformanceCursorResponse,
    PerformanceEventResponse,
    PerformanceResetResponse,
    ProcessIdentityResponse,
)
from claude_code_proxy.limits import MAX_CONTROL_INTEGER

SOCKET_PATH = Path("/run/user/1000/claude-code-proxy/control.sock")


def health_payload(
    capabilities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "application_version": "0.1.0",
        "pid": 42,
        "started_at": "2026-01-02T03:00:00Z",
        "uptime_seconds": 10.5,
        "capabilities": capabilities
        if capabilities is not None
        else ["sessions", "agents", "performance", "performance_events"],
        "sessions": {"active": 0, "retained": 1},
        "inactive_limit": 1000,
    }


def process_payload(
    *, pid: int = 42, started_at: str = "2026-01-02T03:00:00Z"
) -> dict[str, object]:
    return {"pid": pid, "started_at": started_at}


def metric_payload(status: str = "observed", value: object = 0) -> dict[str, object]:
    return {"status": status, "value": value}


def aggregate_payload(value: object = 0) -> dict[str, object]:
    return {
        "value": value,
        "observed_samples": 1,
        "unavailable_samples": 0,
        "not_applicable_samples": 0,
    }


_REQUEST_METRICS = (
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
)
_AGGREGATE_METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "tool_calls",
    "retries",
)


def request_payload() -> dict[str, object]:
    return {
        "id": "request-public",
        "session_id": "session-public",
        "operation": "messages",
        "outcome": "completed",
        "started_at": "2026-01-02T03:04:05Z",
        "finished_at": "2026-01-02T03:04:06Z",
        **{name: metric_payload() for name in _REQUEST_METRICS},
        "reasoning_continuation": "not_applicable",
        "failure": None,
    }


def session_performance_payload() -> dict[str, object]:
    return {
        "session_id": "session-public",
        "requests": 1,
        "active_requests": [],
        "recent_requests": [request_payload()],
        "outcomes": {"completed": 1},
        **{name: aggregate_payload() for name in _AGGREGATE_METRICS},
        "current_concurrency": 0,
        "peak_concurrency": 1,
        "latest_request": request_payload(),
    }


def activity_payload() -> dict[str, object]:
    return {
        "id": "session-public",
        "state": "idle",
        "active_requests": 0,
        "requests": 1,
        "client_model": "client-model",
        "model": "provider-model",
        "provider": "provider",
        "transport": "transport",
        "effort": "high",
        "context_window": 1000,
        "first_seen": "2026-01-02T03:04:05Z",
        "last_seen": "2026-01-02T03:04:06Z",
        "elapsed_seconds": 1,
        "last_result": "completed",
    }


def performance_payload(
    *, cursor: int = 1, process: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "process": process or process_payload(),
        "captured_at": "2026-01-02T05:04:06+02:00",
        "cursor": cursor,
        "sessions": [
            {
                "session": activity_payload(),
                "performance": session_performance_payload(),
            }
        ],
    }


class RecordingTransport(httpx.BaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.close_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def close(self) -> None:
        self.close_count += 1


def snapshot_transport(
    response_factory: Callable[[httpx.Request], httpx.Response],
    *, capabilities: list[str] | None = None,
) -> RecordingTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(
                200,
                json=health_payload(capabilities),
                request=request,
            )
        return response_factory(request)

    return RecordingTransport(handler)


def test_performance_queries_repeated_filters_and_validates_snapshot() -> None:
    transport = snapshot_transport(
        lambda request: httpx.Response(
            200,
            json=performance_payload(),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        response = client.performance(("provider=openai", "state=idle"))

    assert response.cursor == 1
    assert response.process.started_at == datetime(2026, 1, 2, 3, tzinfo=UTC)
    assert response.captured_at == datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC)
    assert response.sessions[0].performance.recent_requests[0].finished_at == datetime(
        2026, 1, 2, 3, 4, 6, tzinfo=UTC
    )
    assert [request.url.path for request in transport.requests] == [
        "/v1/health",
        "/v1/performance",
    ]
    assert transport.requests[1].url.params.multi_items() == [
        ("filter", "provider=openai"),
        ("filter", "state=idle"),
    ]


def test_performance_accepts_empty_snapshot_without_query_filters() -> None:
    payload = performance_payload(cursor=0)
    payload["sessions"] = []
    transport = snapshot_transport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        response = client.performance()

    assert response.sessions == ()
    assert transport.requests[1].url.query == b""


@pytest.mark.parametrize(
    "capabilities",
    [[], ["sessions", "agents", "performance_events"]],
)
def test_performance_requires_performance_capability(
    capabilities: list[str],
) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(capabilities),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol) as raised:
            client.performance()

    assert str(raised.value) == (
        "Control API does not advertise the required performance capability"
    )
    assert [request.url.path for request in transport.requests] == ["/v1/health"]


def test_performance_names_capability_when_health_omits_capabilities() -> None:
    payload = health_payload()
    del payload["capabilities"]
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol) as raised:
            client.performance()

    assert str(raised.value) == (
        "Control API does not advertise the required performance capability"
    )


def test_performance_missing_endpoint_is_incompatible() -> None:
    transport = snapshot_transport(
        lambda request: httpx.Response(
            404,
            json={"detail": "secret endpoint detail"},
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol) as raised:
            client.performance()

    message = str(raised.value)
    assert message == "Control API performance endpoint is missing (HTTP 404)"
    assert "secret" not in message


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda request: httpx.Response(
            200,
            content=b"not-json secret-body",
            headers={"content-type": "application/json"},
            request=request,
        ),
        lambda request: httpx.Response(
            200,
            json={"secret": "raw-schema-secret"},
            request=request,
        ),
    ],
)
def test_performance_malformed_response_uses_generic_safe_error(
    response_factory: Callable[[httpx.Request], httpx.Response],
) -> None:
    transport = snapshot_transport(response_factory)

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            client.performance()

    assert str(raised.value) == (
        "Control API returned an invalid performance response"
    )
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


class TrackingStream(httpx.SyncByteStream):
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self.chunks = tuple(chunks)
        self.close_count = 0

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.close_count += 1


class StreamingTransport(httpx.BaseTransport):
    def __init__(
        self,
        chunks: Iterable[bytes],
        *,
        status: int = 200,
        content_type: str = "application/x-ndjson",
        capabilities: list[str] | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.status = status
        self.content_type = content_type
        self.capabilities = capabilities
        self.requests: list[httpx.Request] = []
        self.streams: list[TrackingStream] = []
        self.close_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/health":
            return httpx.Response(
                200,
                json=health_payload(self.capabilities),
                request=request,
            )
        stream = TrackingStream(self.chunks)
        self.streams.append(stream)
        return httpx.Response(
            self.status,
            headers={"content-type": self.content_type},
            stream=stream,
            request=request,
        )

    def close(self) -> None:
        self.close_count += 1


def reset_payload(
    *, sequence: int = 0, process: dict[str, object] | None = None
) -> dict[str, object]:
    identity = process or process_payload()
    snapshot = performance_payload(cursor=sequence, process=identity)
    return {
        "process": identity,
        "sequence": sequence,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "reset",
        "snapshot": snapshot,
    }


def cursor_payload(
    sequence: int, *, process: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "process": process or process_payload(),
        "sequence": sequence,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "cursor",
    }


def ordinary_payload(
    sequence: int, *, process: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "process": process or process_payload(),
        "sequence": sequence,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "completed",
        "session_id": "session-public",
        "activity": activity_payload(),
        "request": request_payload(),
        "session": session_performance_payload(),
    }


def frame(payload: object, *, suffix: bytes = b"\n") -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + suffix


def consume(client: ControlClient, **kwargs: object) -> list[object]:
    return list(client.performance_events(**kwargs))


def test_performance_events_requires_both_capabilities() -> None:
    for capabilities, missing in (
        (["performance_events"], "performance"),
        (["performance"], "performance_events"),
    ):
        transport = StreamingTransport([], capabilities=capabilities)
        with ControlClient(SOCKET_PATH, transport=transport) as client:
            with pytest.raises(IncompatibleProtocol) as raised:
                consume(client)
        assert str(raised.value) == (
            f"Control API does not advertise the required {missing} capability"
        )
        assert [request.url.path for request in transport.requests] == ["/v1/health"]


def test_performance_events_sends_resume_identity_filters_and_stream_timeout() -> None:
    process = ProcessIdentityResponse.model_validate(process_payload())
    transport = StreamingTransport([frame(cursor_payload(6))])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(
            client,
            filters=("provider=openai", "state=idle"),
            after=5,
            process=process,
        )

    assert len(events) == 1
    assert isinstance(events[0], PerformanceCursorResponse)
    request = transport.requests[1]
    assert request.method == "GET"
    assert request.url.path == "/v1/performance/events"
    assert request.url.params.multi_items() == [
        ("filter", "provider=openai"),
        ("filter", "state=idle"),
        ("after", "5"),
        ("pid", "42"),
        ("started_at", "2026-01-02T03:00:00+00:00"),
    ]
    assert request.extensions["timeout"] == {
        "connect": 2.0,
        "read": None,
        "write": 2.0,
        "pool": 2.0,
    }


def test_performance_events_omits_process_without_after() -> None:
    process = ProcessIdentityResponse.model_validate(process_payload())
    transport = StreamingTransport([frame(reset_payload())])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client, process=process)

    assert len(events) == 1
    assert transport.requests[1].url.params.multi_items() == []


@pytest.mark.parametrize("after", [True, -1, MAX_CONTROL_INTEGER + 1, 1.0, "1"])
def test_performance_events_rejects_invalid_after_without_network(after: object) -> None:
    transport = StreamingTransport([])
    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ValueError, match="control protocol range"):
            client.performance_events(after=after)
    assert transport.requests == []


def test_performance_events_missing_endpoint_is_incompatible() -> None:
    transport = StreamingTransport(
        [b'{"detail":"raw endpoint secret"}'],
        status=404,
        content_type="application/json",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol) as raised:
            consume(client)

    assert str(raised.value) == (
        "Control API performance events endpoint is missing (HTTP 404)"
    )
    assert "secret" not in str(raised.value)
    assert transport.streams[0].close_count == 1


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "text/plain", "application/x-ndjsonish"],
)
def test_performance_events_rejects_wrong_content_type(content_type: str) -> None:
    transport = StreamingTransport(
        [frame(reset_payload())],
        content_type=content_type,
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == (
        "Control API returned an invalid performance event stream"
    )
    assert transport.streams[0].close_count == 1


def test_stream_parses_heartbeats_fragmented_utf8_crlf_and_final_line() -> None:
    reset = frame(reset_payload(), suffix=b"\r\n")
    first_cursor = frame(cursor_payload(1))
    ordinary = ordinary_payload(2)
    ordinary["activity"]["provider"] = "prövider"
    ordinary_line = frame(ordinary)
    utf8_boundary = ordinary_line.index("ö".encode()) + 1
    final_cursor = frame(cursor_payload(3), suffix=b"")
    chunks = [
        b"\n\r\n" + reset[:17],
        reset[17:] + first_cursor + ordinary_line[:utf8_boundary],
        ordinary_line[utf8_boundary:] + final_cursor,
    ]
    transport = StreamingTransport(chunks)

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client)

    assert [event.type for event in events] == [
        "reset",
        "cursor",
        "completed",
        "cursor",
    ]
    assert events[2].activity.provider == "prövider"
    assert transport.streams[0].close_count == 1


def padded_frame(payload: object, length: int) -> bytes:
    encoded = frame(payload, suffix=b"")
    assert len(encoded) <= length
    return encoded + b" " * (length - len(encoded))


def test_stream_accepts_line_at_exact_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 512
    monkeypatch.setattr(client_module, "_MAX_NDJSON_LINE_BYTES", limit, raising=False)
    process = ProcessIdentityResponse.model_validate(process_payload())
    transport = StreamingTransport([padded_frame(cursor_payload(1), limit) + b"\n"])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client, after=0, process=process)

    assert [event.sequence for event in events] == [1]


def test_stream_rejects_line_over_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 512
    monkeypatch.setattr(client_module, "_MAX_NDJSON_LINE_BYTES", limit, raising=False)
    process = ProcessIdentityResponse.model_validate(process_payload())
    oversized = padded_frame(cursor_payload(1), limit + 1) + b"\n"
    transport = StreamingTransport([oversized])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client, after=0, process=process)

    assert str(raised.value) == (
        "Control API returned an invalid performance event stream"
    )
    assert transport.streams[0].close_count == 1


@pytest.mark.parametrize(
    "invalid_line",
    [
        b'{"type":"cursor","secret":"invalid-json"\n',
        b'"raw-secret-string"\n',
        b'{"type":"cursor","sequence":NaN,"secret":"constant"}\n',
        b'{"type":"cursor","secret":"bad-utf8-\xff"}\n',
    ],
    ids=["json", "non-object", "nan", "utf8"],
)
def test_stream_rejects_invalid_lines_without_echoing_content(
    invalid_line: bytes,
) -> None:
    transport = StreamingTransport([invalid_line])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == (
        "Control API returned an invalid performance event stream"
    )
    assert "secret" not in str(raised.value)
    assert "NaN" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert transport.streams[0].close_count == 1


def test_stream_requires_initial_reset_without_resume() -> None:
    transport = StreamingTransport([frame(ordinary_payload(1))])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError, match="invalid performance event stream"):
            consume(client)


def test_stream_yields_reset_ordinary_and_filtered_cursor_contiguously() -> None:
    transport = StreamingTransport(
        [
            frame(reset_payload(sequence=5)),
            frame(ordinary_payload(6)),
            frame(cursor_payload(7)),
        ]
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client, filters=("provider=provider",))

    assert [event.sequence for event in events] == [5, 6, 7]
    assert isinstance(events[0], PerformanceResetResponse)
    assert isinstance(events[1], PerformanceEventResponse)
    assert isinstance(events[2], PerformanceCursorResponse)


def test_stream_resumes_from_supplied_process_and_sequence() -> None:
    process = ProcessIdentityResponse.model_validate(process_payload())
    transport = StreamingTransport(
        [frame(ordinary_payload(11)), frame(cursor_payload(12))]
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client, after=10, process=process)

    assert [event.sequence for event in events] == [11, 12]


def test_stream_reset_atomically_replaces_process_and_cursor() -> None:
    old_process = ProcessIdentityResponse.model_validate(process_payload())
    new_process = process_payload(pid=99, started_at="2026-01-03T03:00:00Z")
    transport = StreamingTransport(
        [
            frame(reset_payload(sequence=20, process=new_process)),
            frame(cursor_payload(21, process=new_process)),
        ]
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client, after=10, process=old_process)

    assert [event.process.pid for event in events] == [99, 99]
    assert [event.sequence for event in events] == [20, 21]


def test_stream_ignores_exact_duplicate_sequence() -> None:
    duplicate = cursor_payload(1)
    transport = StreamingTransport(
        [frame(reset_payload()), frame(duplicate), frame(duplicate), frame(cursor_payload(2))]
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client)

    assert [event.sequence for event in events] == [0, 1, 2]


@pytest.mark.parametrize(
    "payloads",
    [
        [reset_payload(), cursor_payload(2)],
        [reset_payload(sequence=2), cursor_payload(1)],
        [
            reset_payload(),
            cursor_payload(
                1,
                process=process_payload(
                    pid=99,
                    started_at="2026-01-03T03:00:00Z",
                ),
            ),
        ],
    ],
    ids=["gap", "out-of-order", "process-mismatch"],
)
def test_stream_rejects_noncontiguous_or_mismatched_events(
    payloads: list[dict[str, object]],
) -> None:
    transport = StreamingTransport([frame(payload) for payload in payloads])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == (
        "Control API returned an invalid performance event stream"
    )


def test_performance_http_error_never_echoes_raw_body() -> None:
    transport = snapshot_transport(
        lambda request: httpx.Response(
            503,
            text="raw snapshot secret stack",
            headers={"content-type": "text/plain"},
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            client.performance()

    assert str(raised.value) == "Control API returned HTTP 503"
    assert "secret" not in str(raised.value)


def test_stream_accepts_ndjson_content_type_with_charset() -> None:
    transport = StreamingTransport(
        [frame(reset_payload())],
        content_type="application/x-ndjson; charset=utf-8",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = consume(client)

    assert [event.type for event in events] == ["reset"]


@pytest.mark.parametrize(
    "process",
    [
        {"pid": True, "started_at": "2026-01-02T03:00:00Z"},
        {"pid": 42, "started_at": "not-a-date"},
    ],
)
def test_stream_validates_process_identity_on_every_frame(
    process: dict[str, object],
) -> None:
    payload = reset_payload()
    payload["process"] = process
    payload["snapshot"]["process"] = process
    transport = StreamingTransport([frame(payload)])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError, match="invalid performance event stream"):
            consume(client)


def test_stream_closes_on_early_generator_close() -> None:
    transport = StreamingTransport(
        [frame(reset_payload()), frame(cursor_payload(1))]
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        events = client.performance_events()
        assert next(events).type == "reset"
        assert transport.streams[0].close_count == 0
        events.close()
        assert transport.streams[0].close_count == 1


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_stream_closes_when_consumer_raises(
    exception_type: type[BaseException],
) -> None:
    transport = StreamingTransport([frame(reset_payload()), frame(cursor_payload(1))])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(exception_type):
            for _event in client.performance_events():
                raise exception_type("consumer stopped")
        assert transport.streams[0].close_count == 1


def test_client_close_closes_active_stream_exactly_once() -> None:
    transport = StreamingTransport([frame(reset_payload()), frame(cursor_payload(1))])
    client = ControlClient(SOCKET_PATH, transport=transport)
    events = client.performance_events()
    assert next(events).type == "reset"

    client.close()
    assert transport.streams[0].close_count == 1
    events.close()
    client.close()
    assert transport.streams[0].close_count == 1


def test_stream_http_error_reads_safe_bounded_json_detail() -> None:
    detail = "bad\nrequest " + "x" * 1000
    transport = StreamingTransport(
        [json.dumps({"detail": detail}).encode()],
        status=422,
        content_type="application/json",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    message = str(raised.value)
    assert message.startswith("Control API returned HTTP 422: bad\\x0arequest")
    assert "\n" not in message
    assert len(message) < 400
    assert transport.streams[0].close_count == 1


def test_stream_http_error_does_not_echo_non_json_body() -> None:
    transport = StreamingTransport(
        [b"raw stream secret stack"],
        status=500,
        content_type="text/plain",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == "Control API returned HTTP 500"
    assert "secret" not in str(raised.value)


def test_stream_http_error_body_read_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_MAX_ERROR_BODY_BYTES", 64, raising=False)
    transport = StreamingTransport(
        [b'{"detail":"' + b"secret" * 100 + b'"}'],
        status=422,
        content_type="application/json",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == "Control API returned HTTP 422"
    assert transport.streams[0].close_count == 1


class ErrorStream(httpx.SyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self.request = request
        self.close_count = 0

    def __iter__(self):
        yield frame(reset_payload())
        raise httpx.ReadError("unsafe\nstream detail", request=self.request)

    def close(self) -> None:
        self.close_count += 1


class FailingStreamingTransport(StreamingTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/health":
            return httpx.Response(200, json=health_payload(), request=request)
        stream = ErrorStream(request)
        self.streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=stream,
            request=request,
        )


def test_stream_request_failure_maps_to_socket_aware_safe_error() -> None:
    transport = FailingStreamingTransport([])

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(client_module.ControlUnavailable) as raised:
            consume(client)

    assert raised.value.socket_path == SOCKET_PATH.resolve()
    assert raised.value.reason == "unsafe\\x0astream detail"
    assert transport.streams[0].close_count == 1


class TimeoutCheckingErrorStream(httpx.SyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self.request = request
        self.observed_timeouts: list[float | None] = []
        self.close_count = 0

    def __iter__(self):
        timeout = self.request.extensions["timeout"]["read"]
        self.observed_timeouts.append(timeout)
        if timeout is None:
            raise AssertionError("error body read remained unbounded")
        raise httpx.ReadTimeout("simulated stalled body", request=self.request)

    def close(self) -> None:
        self.close_count += 1


class TimeoutCheckingTransport(StreamingTransport):
    def __init__(self) -> None:
        super().__init__([], status=422, content_type="application/json")
        self.error_streams: list[TimeoutCheckingErrorStream] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/health":
            return httpx.Response(200, json=health_payload(), request=request)
        stream = TimeoutCheckingErrorStream(request)
        self.error_streams.append(stream)
        return httpx.Response(
            422,
            headers={"content-type": "application/json"},
            stream=stream,
            request=request,
        )


def test_stream_http_error_stall_uses_finite_read_timeout() -> None:
    transport = TimeoutCheckingTransport()

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == "Control API returned HTTP 422"
    observed = transport.error_streams[0].observed_timeouts[0]
    assert observed is not None
    assert 0 < observed <= 2.0
    assert transport.requests[1].extensions["timeout"]["read"] is None
    assert transport.error_streams[0].close_count == 1


def test_stream_http_error_drip_feed_has_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((0.0, 0.5, 2.1))
    monkeypatch.setattr(client_module, "monotonic", lambda: next(times), raising=False)
    transport = StreamingTransport(
        [b'{"detail":"', b'raw secret"}'],
        status=422,
        content_type="application/json",
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            consume(client)

    assert str(raised.value) == "Control API returned HTTP 422"
    assert transport.requests[1].extensions["timeout"]["read"] is None
    assert transport.streams[0].close_count == 1

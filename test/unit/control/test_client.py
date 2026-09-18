from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path

import httpx
import pytest

from claude_code_proxy.control import client as client_module
from claude_code_proxy.control.client import (
    ControlClient,
    ControlError,
    ControlUnavailable,
    IncompatibleProtocol,
)


SOCKET_PATH = Path("/run/user/1000/claude-code-proxy/control.sock")


def health_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": 1,
        "application_version": "0.1.0",
        "pid": 42,
        "started_at": "2026-01-02T03:04:05Z",
        "uptime_seconds": 10.5,
        "capabilities": ["sessions"],
        "sessions": {"active": 1, "retained": 2},
        "inactive_limit": 1000,
    }
    payload.update(overrides)
    return payload


def session_payload(identifier: str = "a" * 64) -> dict[str, object]:
    return {
        "id": identifier,
        "state": "active",
        "active_requests": 1,
        "requests": 3,
        "client_model": "claude-opus",
        "model": "gpt-5.6-sol",
        "provider": "openai",
        "transport": "codex",
        "effort": "high",
        "context_window": 1_000_000,
        "first_seen": "2026-01-02T03:00:00Z",
        "last_seen": "2026-01-02T03:04:00Z",
        "elapsed_seconds": 240.0,
        "last_result": None,
    }


def session_list_payload() -> dict[str, object]:
    return {
        "captured_at": "2026-01-02T03:04:05Z",
        "sessions": [session_payload()],
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


def test_default_transport_uses_given_unix_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=health_payload(), request=request)
    )

    def make_transport(**kwargs: object) -> httpx.BaseTransport:
        created.append(kwargs)
        return transport

    monkeypatch.setattr(client_module.httpx, "HTTPTransport", make_transport)

    client = ControlClient(SOCKET_PATH)
    client.close()

    assert created == [{"uds": str(SOCKET_PATH)}]
    assert transport.close_count == 1


def test_health_parses_response_and_context_manager_closes_transport() -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=health_payload(), request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        response = client.health()
        assert response.protocol_version == 1
        assert response.started_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    assert transport.close_count == 1


@pytest.mark.parametrize("uptime", [float("nan"), float("inf"), float("-inf")])
def test_health_rejects_non_finite_uptime(uptime: float) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            text=json.dumps(
                health_payload(uptime_seconds=uptime),
                allow_nan=True,
            ),
            headers={"content-type": "application/json"},
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError, match="invalid health response"):
            client.health()


@pytest.mark.parametrize(
    "overrides",
    [
        {"pid": 0},
        {"pid": 2**63},
        {"sessions": {"active": -1, "retained": 0}},
        {"sessions": {"active": 2, "retained": 1}},
        {"sessions": {"active": 0, "retained": 2**63}},
        {"inactive_limit": -1},
        {"inactive_limit": 2**63},
        {"uptime_seconds": -1.0},
    ],
)
def test_health_rejects_out_of_range_numeric_fields(
    overrides: dict[str, object],
) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(**overrides),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    assert str(raised.value) == "Control API returned an invalid health response"


def test_health_accepts_signed_64_maximum_integer_fields() -> None:
    maximum = 2**63 - 1
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(
                pid=maximum,
                sessions={"active": maximum, "retained": maximum},
                inactive_limit=maximum,
                uptime_seconds=0.0,
            ),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        result = client.health()

    assert result.pid == maximum
    assert result.sessions.active == maximum
    assert result.sessions.retained == maximum
    assert result.inactive_limit == maximum


@pytest.mark.parametrize(
    ("field", "nested", "value"),
    [
        (field, nested, value)
        for field, nested in (
            ("active", True),
            ("retained", True),
            ("pid", False),
            ("inactive_limit", False),
        )
        for value in (True, "1", 1.0)
    ],
)
def test_health_rejects_coerced_integer_wire_types(
    field: str,
    nested: bool,
    value: object,
) -> None:
    payload = health_payload()
    target = payload["sessions"] if nested else payload
    target[field] = value
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    assert str(raised.value) == "Control API returned an invalid health response"


@pytest.mark.parametrize("value", [True, "1.5"])
def test_health_rejects_coerced_duration_wire_types(value: object) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(uptime_seconds=value),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    assert str(raised.value) == "Control API returned an invalid health response"


@pytest.mark.parametrize("value", [1, 1.5])
def test_health_accepts_integer_and_float_duration_wire_types(
    value: int | float,
) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(uptime_seconds=value),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        result = client.health()

    assert result.uptime_seconds == float(value)
    assert type(result.uptime_seconds) is float


def test_default_timeout_is_finite_for_every_httpx_phase() -> None:
    observed: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.extensions["timeout"])
        return httpx.Response(200, json=health_payload(), request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        client.health()

    assert observed == [
        {
            "connect": 2.0,
            "read": 2.0,
            "write": 2.0,
            "pool": 2.0,
        }
    ]


def test_connect_error_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("not available", request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlUnavailable):
            client.health()

    assert attempts == 1


def test_sessions_sends_repeated_filters_in_order_after_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = health_payload() if request.url.path == "/v1/health" else session_list_payload()
        return httpx.Response(200, json=payload, request=request)

    transport = RecordingTransport(handler)

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        response = client.sessions(["state=active", "model=opus", "state=idle"])

    assert len(response.sessions) == 1
    assert [request.url.path for request in transport.requests] == [
        "/v1/health",
        "/v1/sessions",
    ]
    assert transport.requests[1].url.params.multi_items() == [
        ("filter", "state=active"),
        ("filter", "model=opus"),
        ("filter", "state=idle"),
    ]


@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), float("-inf")])
def test_sessions_rejects_non_finite_elapsed_seconds(elapsed: float) -> None:
    payload = session_list_payload()
    payload["sessions"][0]["elapsed_seconds"] = elapsed

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(
            200,
            text=json.dumps(body, allow_nan=True),
            headers={"content-type": "application/json"},
            request=request,
        )

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError, match="invalid sessions response"):
            client.sessions()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("context_window", 10**999, id="context-huge-positive"),
        pytest.param("context_window", -(10**999), id="context-huge-negative"),
        pytest.param("active_requests", 10**999, id="active-huge-positive"),
        pytest.param("active_requests", -(10**999), id="active-huge-negative"),
        pytest.param("requests", 10**999, id="requests-huge-positive"),
        pytest.param("requests", -(10**999), id="requests-huge-negative"),
        pytest.param("context_window", 2**63, id="context-max-plus-one"),
        pytest.param("active_requests", 2**63, id="active-max-plus-one"),
        pytest.param("requests", 2**63, id="requests-max-plus-one"),
        pytest.param("context_window", 0, id="context-zero"),
        pytest.param("elapsed_seconds", -1.0, id="elapsed-negative"),
    ],
)
def test_sessions_rejects_out_of_range_numeric_fields(
    field: str,
    value: int | float,
) -> None:
    payload = session_list_payload()
    payload["sessions"][0][field] = value

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.sessions()

    assert str(raised.value) == "Control API returned an invalid sessions response"
    assert str(value) not in str(raised.value)


def test_sessions_accepts_signed_64_maximum_integer_fields() -> None:
    maximum = 2**63 - 1
    payload = session_list_payload()
    payload["sessions"][0].update({
        "active_requests": maximum,
        "requests": maximum,
        "context_window": maximum,
        "elapsed_seconds": 0.0,
    })

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.sessions()

    observed = result.sessions[0]
    assert observed.active_requests == maximum
    assert observed.requests == maximum
    assert observed.context_window == maximum


def test_sessions_rejects_active_requests_above_total_requests() -> None:
    payload = session_list_payload()
    payload["sessions"][0].update({"active_requests": 2, "requests": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.sessions()

    assert str(raised.value) == "Control API returned an invalid sessions response"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("active_requests", "requests", "context_window")
        for value in (True, "1", 1.0)
    ],
)
def test_sessions_rejects_coerced_integer_wire_types(
    field: str,
    value: object,
) -> None:
    payload = session_list_payload()
    payload["sessions"][0][field] = value

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.sessions()

    assert str(raised.value) == "Control API returned an invalid sessions response"


@pytest.mark.parametrize("value", [True, "1.5"])
def test_sessions_rejects_coerced_duration_wire_types(value: object) -> None:
    payload = session_list_payload()
    payload["sessions"][0]["elapsed_seconds"] = value

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.sessions()

    assert str(raised.value) == "Control API returned an invalid sessions response"


@pytest.mark.parametrize("value", [1, 1.5])
def test_sessions_accepts_integer_and_float_duration_wire_types(
    value: int | float,
) -> None:
    payload = session_list_payload()
    payload["sessions"][0]["elapsed_seconds"] = value

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.sessions()

    assert result.sessions[0].elapsed_seconds == float(value)
    assert type(result.sessions[0].elapsed_seconds) is float


@pytest.mark.parametrize(
    ("field", "nested"),
    [("captured_at", False), ("first_seen", True), ("last_seen", True)],
)
def test_sessions_rejects_naive_datetimes(field: str, nested: bool) -> None:
    payload = session_list_payload()
    target = payload["sessions"][0] if nested else payload
    target[field] = "2026-01-02T03:04:05"

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError, match="invalid sessions response"):
            client.sessions()


def test_sessions_normalizes_all_datetimes_to_utc() -> None:
    payload = session_list_payload()
    payload["captured_at"] = "2026-01-02T05:04:05+02:00"
    payload["sessions"][0]["first_seen"] = "2026-01-02T04:00:00+01:00"
    payload["sessions"][0]["last_seen"] = "2026-01-01T22:04:00-05:00"

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.sessions()

    assert result.captured_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert result.sessions[0].first_seen == datetime(2026, 1, 2, 3, tzinfo=UTC)
    assert result.sessions[0].last_seen == datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    assert result.captured_at.tzinfo is UTC
    assert result.sessions[0].first_seen.tzinfo is UTC
    assert result.sessions[0].last_seen.tzinfo is UTC


@pytest.mark.parametrize(
    ("field", "nested", "value"),
    [
        ("captured_at", False, "0001-01-01T00:00:00+23:59"),
        ("captured_at", False, "9999-12-31T23:59:59-23:59"),
        ("first_seen", True, "0001-01-01T00:00:00+23:59"),
        ("first_seen", True, "9999-12-31T23:59:59-23:59"),
        ("last_seen", True, "0001-01-01T00:00:00+23:59"),
        ("last_seen", True, "9999-12-31T23:59:59-23:59"),
    ],
)
def test_sessions_maps_utc_conversion_overflow_to_safe_error(
    field: str,
    nested: bool,
    value: str,
) -> None:
    payload = session_list_payload()
    target = payload["sessions"][0] if nested else payload
    target[field] = value

    def handler(request: httpx.Request) -> httpx.Response:
        body = health_payload() if request.url.path == "/v1/health" else payload
        return httpx.Response(200, json=body, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.sessions()

    assert str(raised.value) == "Control API returned an invalid sessions response"
    assert "date value out of range" not in str(raised.value)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_health_rejects_non_exact_protocol_version(version: object) -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(protocol_version=version),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol, match="protocol version"):
            client.health()

    assert [request.url.path for request in transport.requests] == ["/v1/health"]


def test_health_rejects_missing_protocol_version() -> None:
    payload = health_payload()
    del payload["protocol_version"]
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol, match="protocol version missing"):
            client.health()

    assert [request.url.path for request in transport.requests] == ["/v1/health"]


def test_sessions_rejects_missing_sessions_capability_before_query() -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(
            200,
            json=health_payload(capabilities=[]),
            request=request,
        )
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol, match="sessions capability"):
            client.sessions()

    assert [request.url.path for request in transport.requests] == ["/v1/health"]


def test_sessions_rejects_omitted_capabilities_before_query() -> None:
    payload = health_payload()
    del payload["capabilities"]
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )

    with ControlClient(SOCKET_PATH, transport=transport) as client:
        with pytest.raises(IncompatibleProtocol, match="sessions capability"):
            client.sessions()

    assert [request.url.path for request in transport.requests] == ["/v1/health"]


@pytest.mark.parametrize(
    "exception_type",
    [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout],
)
def test_request_failures_map_to_control_unavailable_with_socket_path(
    exception_type: type[httpx.RequestError],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_type("unsafe\ntransport detail", request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlUnavailable) as raised:
            client.health()

    assert raised.value.socket_path == SOCKET_PATH.resolve()
    assert str(SOCKET_PATH.resolve()) in str(raised.value)
    assert raised.value.reason == "unsafe\\x0atransport detail"
    assert "\\x0a" in str(raised.value)
    assert "\ntransport" not in str(raised.value)


def test_unavailable_reason_truncation_never_splits_escape_token() -> None:
    error = ControlUnavailable(
        SOCKET_PATH,
        "a" * 195 + "\x1b" + "tail",
    )

    assert error.reason == "a" * 195 + "..."
    assert len(error.reason) <= 200
    assert "\\x..." not in error.reason
    assert "\\x1..." not in error.reason


def test_unavailable_reason_escapes_surrogate_and_nonprintable_codepoints() -> None:
    error = ControlUnavailable(
        SOCKET_PATH,
        "bad\ud800\U000e0001",
    )

    assert error.reason == "bad\\ud800\\U000e0001"
    error.reason.encode("utf-8", errors="strict")
    str(error).encode("utf-8", errors="strict")


@pytest.mark.parametrize("status", [400, 503])
def test_non_success_json_error_has_bounded_safe_detail(status: int) -> None:
    detail = "bad\nrequest " + "x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": detail}, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    message = str(raised.value)
    assert f"HTTP {status}" in message
    assert "bad\\x0arequest" in message
    assert "\n" not in message
    assert len(message) < 400


def test_non_success_html_error_does_not_dump_body() -> None:
    body = "<html><body>secret internal stack" + "x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=body,
            headers={"content-type": "text/html"},
            request=request,
        )

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    message = str(raised.value)
    assert "HTTP 500" in message
    assert "<html>" not in message
    assert "secret internal stack" not in message
    assert len(message) < 400


@pytest.mark.parametrize(
    "body",
    [
        '{"value":' + "9" * 5000 + "}",
        "[" * 2000 + "0" + "]" * 2000,
    ],
)
def test_success_json_parser_failures_are_safe(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/json"},
            request=request,
        )

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    assert str(raised.value) == "Control API returned invalid JSON"


@pytest.mark.parametrize(
    "body",
    [
        '{"detail":' + "9" * 5000 + "}",
        "[" * 2000 + "0" + "]" * 2000,
    ],
)
def test_http_error_json_parser_failures_fall_back_to_status(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=body,
            headers={"content-type": "application/json"},
            request=request,
        )

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError) as raised:
            client.health()

    assert str(raised.value) == "Control API returned HTTP 500"


@pytest.mark.parametrize(
    ("response_factory", "match"),
    [
        (lambda request: httpx.Response(200, text="not json", request=request), "invalid JSON"),
        (
            lambda request: httpx.Response(
                200,
                json=health_payload(pid="not-an-integer"),
                request=request,
            ),
            "invalid health response",
        ),
    ],
)
def test_health_maps_malformed_responses_to_control_error(
    response_factory: Callable[[httpx.Request], httpx.Response],
    match: str,
) -> None:
    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(response_factory),
    ) as client:
        with pytest.raises(ControlError, match=match):
            client.health()


def test_sessions_maps_schema_failure_to_control_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload: object = health_payload()
        if request.url.path == "/v1/sessions":
            payload = {"captured_at": "not-a-date", "sessions": []}
        return httpx.Response(200, json=payload, request=request)

    with ControlClient(
        SOCKET_PATH,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ControlError, match="invalid sessions response"):
            client.sessions()


def test_close_is_idempotent_and_use_after_close_is_clear() -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json=health_payload(), request=request)
    )
    client = ControlClient(SOCKET_PATH, transport=transport)

    client.close()
    client.close()

    assert transport.close_count == 1
    with pytest.raises(ControlError, match="closed"):
        client.health()

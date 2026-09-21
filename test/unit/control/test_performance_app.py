import asyncio
from datetime import UTC, datetime, timedelta
import json

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import (
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
)
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.event_journal import EventJournal, Subscription
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import SessionMetadata, SessionRegistry


class RegistryClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=UTC)
        self.monotonic = 100.0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self, seconds: float = 1.0) -> None:
        self.wall += timedelta(seconds=seconds)
        self.monotonic += seconds


def metadata(
    client_id: str,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    *,
    client_model: str = "claude-opus",
    upstream_model: str = "openai/gpt-5.6-sol",
    provider: str = "openai",
    transport: str = "codex",
    effort: str = "high",
    context_window: int | None = 1_000_000,
) -> SessionMetadata:
    return SessionMetadata(
        client_identity=ClientIdentity(client_id, agent_id, parent_agent_id),
        client_model=client_model,
        upstream_model=upstream_model,
        provider=provider,
        transport=transport,
        effort=effort,
        context_window=context_window,
    )


def registry(clock: RegistryClock, inactive_limit: int = 10) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
    )


def filtered_registry() -> tuple[SessionRegistry, dict[str, str]]:
    clock = RegistryClock()
    sessions = registry(clock)
    active = sessions.begin(metadata("raw-active"))
    clock.advance()
    idle = sessions.begin(
        metadata(
            "raw-idle",
            client_model="claude-sonnet",
            upstream_model="vertex/claude-sonnet-5",
            provider="vertex",
            transport="litellm",
            effort="medium",
            context_window=None,
        )
    )
    sessions.finish(idle, "completed")
    clock.advance()
    failed = sessions.begin(
        metadata(
            "raw-failed",
            client_model="claude-haiku",
            upstream_model="gemini/gemini-pro",
            provider="gemini",
            transport="litellm",
            effort="low",
            context_window=200_000,
        )
    )
    sessions.finish(failed, "failed")
    return sessions, {
        "active": active.public_id,
        "idle": idle.public_id,
        "failed": failed.public_id,
    }


async def request(app, path: str, params=None):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://control"
    ) as client:
        return await client.get(path, params=params)


async def test_performance_snapshot_wraps_atomic_capture_and_process_identity() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(metadata("raw-performance"))
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        application_version="1.0",
        pid=2468,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["process"] == {
        "pid": 2468,
        "started_at": "2026-01-02T03:04:05Z",
    }
    assert payload["captured_at"] == "2026-01-01T00:00:00Z"
    assert payload["cursor"] == 1
    assert payload["sessions"][0]["session"]["id"] == handle.public_id
    assert payload["sessions"][0]["performance"]["session_id"] == handle.public_id
    PerformanceListResponse.model_validate(payload)


async def test_performance_snapshot_empty_registry_is_valid() -> None:
    clock = RegistryClock()
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    assert response.json()["cursor"] == 0
    assert response.json()["sessions"] == []


async def test_performance_snapshot_reuses_filter_semantics_and_raw_session_resolution() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    by_raw_session = await request(
        app,
        "/v1/performance",
        [("filter", "session_id=raw-idle")],
    )
    repeated_or = await request(
        app,
        "/v1/performance",
        [("filter", "state=idle"), ("filter", "state=failed")],
    )
    different_and = await request(
        app,
        "/v1/performance",
        [("filter", "transport=litellm"), ("filter", "effort=medium")],
    )

    assert [
        item["session"]["id"] for item in by_raw_session.json()["sessions"]
    ] == [ids["idle"]]
    assert {
        item["session"]["id"] for item in repeated_or.json()["sessions"]
    } == {ids["idle"], ids["failed"]}
    assert [
        item["session"]["id"] for item in different_and.json()["sessions"]
    ] == [ids["idle"]]
    assert "raw-idle" not in by_raw_session.text


async def test_performance_snapshot_rejects_invalid_and_ambiguous_filters_generically() -> None:
    clock = RegistryClock()
    sessions = registry(clock, inactive_limit=20)
    prefixes: dict[str, int] = {}
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-performance-{index}")).public_id[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes[prefix] == 2:
            collision = prefix
            break
    assert collision is not None
    app = create_control_app(
        sessions,
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    invalid = await request(
        app,
        "/v1/performance",
        [("filter", "unknown=value")],
    )
    ambiguous = await request(
        app,
        "/v1/performance",
        [("filter", f"id={collision}")],
    )

    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid session filter"}
    assert ambiguous.status_code == 422
    assert ambiguous.json() == {"detail": "Session ID prefix is ambiguous"}
    assert "raw-performance" not in invalid.text + ambiguous.text


async def test_performance_snapshot_excludes_raw_ids_and_provider_content() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    sessions.begin(metadata("raw-private-session"))
    app = create_control_app(
        sessions,
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/performance")

    assert response.status_code == 200
    serialized = response.text
    for marker in (
        "raw-private-session",
        "raw_session_id",
        "provider_payload",
        "prompt",
        "content",
    ):
        assert marker not in serialized


def stream_registry(
    clock: RegistryClock,
    *,
    capacity: int = 4096,
    subscriber_capacity: int = 64,
    inactive_limit: int = 10,
) -> SessionRegistry:
    return SessionRegistry(
        inactive_limit=inactive_limit,
        secret=b"control-test-secret",
        wall_clock=clock.wall_now,
        monotonic_clock=clock.monotonic_now,
        events=EventJournal(capacity, subscriber_capacity),
    )


async def performance_stream_response(
    app,
    *,
    filters: list[str] | None = None,
    after: str | None = None,
    pid: str | None = None,
    started_at: str | None = None,
):
    route = next(route for route in app.routes if route.path == "/v1/performance/events")
    return await route.endpoint(
        filter=filters,
        after=None if after is None else [after],
        pid=None if pid is None else [pid],
        started_at=None if started_at is None else [started_at],
    )


async def next_stream_json(response) -> dict[str, object]:
    frame = await anext(response.body_iterator)
    assert isinstance(frame, str)
    assert frame.endswith("\n")
    assert not frame.endswith("\n\n")
    return json.loads(frame)


def control_app_for_stream(
    sessions: SessionRegistry,
    *,
    heartbeat_interval: float = 15.0,
    pid: int = 42,
    started_at: datetime = datetime(2026, 1, 2, 3, tzinfo=UTC),
):
    return create_control_app(
        sessions,
        started_at=started_at,
        application_version="1.0",
        pid=pid,
        heartbeat_interval=heartbeat_interval,
    )


async def test_performance_stream_initial_reset_precedes_immediate_live_event() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    assert response.media_type == "application/x-ndjson"
    assert sessions.events.subscriber_count == 1

    handle = sessions.begin(metadata("raw-live"))
    reset = await next_stream_json(response)
    event = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == reset["snapshot"]["cursor"] == 0
    assert event["type"] == "request_started"
    assert event["sequence"] == 1
    assert event["session_id"] == handle.public_id
    assert event["request"]["id"] == handle.request_id
    assert event["process"] == reset["process"]
    PerformanceResetResponse.model_validate(reset)
    PerformanceEventResponse.model_validate(event)
    await response.body_iterator.aclose()
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_matching_process_replays_retained_then_live() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    first = sessions.begin(metadata("first"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)
    response = await performance_stream_response(
        app,
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    second = sessions.begin(metadata("second"))

    replay = await next_stream_json(response)
    live = await next_stream_json(response)

    assert replay["type"] == "request_started"
    assert replay["sequence"] == 1
    assert replay["request"]["id"] == first.request_id
    assert live["sequence"] == 2
    assert live["request"]["id"] == second.request_id
    await response.body_iterator.aclose()


async def test_performance_stream_after_current_waits_live_without_reset() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)
    response = await performance_stream_response(
        app,
        after="1",
        pid="42",
        started_at=started.isoformat(),
    )

    live_handle = sessions.begin(metadata("live"))
    live = await next_stream_json(response)

    assert live["type"] == "request_started"
    assert live["sequence"] == 2
    assert live["request"]["id"] == live_handle.request_id
    await response.body_iterator.aclose()


async def test_performance_stream_stale_cursor_gets_current_reset() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock, capacity=1)
    sessions.begin(metadata("first"))
    sessions.begin(metadata("second"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)

    response = await performance_stream_response(
        app,
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 2
    assert reset["snapshot"]["cursor"] == 2
    await response.body_iterator.aclose()


@pytest.mark.parametrize(
    ("pid", "started_at"),
    [
        (None, None),
        ("41", "2026-01-02T03:00:00+00:00"),
        ("42", "2026-01-02T03:00:01+00:00"),
        (None, "2026-01-02T03:00:00+00:00"),
        ("42", None),
    ],
)
async def test_performance_stream_missing_or_mismatched_process_forces_reset(
    pid: str | None,
    started_at: str | None,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        after="0",
        pid=pid,
        started_at=started_at,
    )
    first = await next_stream_json(response)

    assert first["type"] == "reset"
    assert first["sequence"] == 1
    assert first["process"] == {
        "pid": 42,
        "started_at": "2026-01-02T03:00:00Z",
    }
    await response.body_iterator.aclose()


async def test_performance_stream_filters_initial_snapshot_without_raw_id() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    selected = sessions.begin(metadata("raw-selected"))
    sessions.begin(metadata("raw-other"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        filters=["session_id=raw-selected"],
    )
    reset = await next_stream_json(response)
    serialized = json.dumps(reset)

    assert [
        item["session"]["id"] for item in reset["snapshot"]["sessions"]
    ] == [selected.public_id]
    assert "raw-selected" not in serialized
    assert "raw-other" not in serialized
    await response.body_iterator.aclose()


async def test_performance_stream_overflow_resets_and_continues_on_fresh_subscription() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock, subscriber_capacity=1)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    initial = await next_stream_json(response)
    assert initial["type"] == "reset"

    sessions.begin(metadata("first"))
    sessions.begin(metadata("second"))
    assert sessions.events.subscriber_count == 0
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 2
    assert sessions.events.subscriber_count == 1
    continued = sessions.begin(metadata("continued"))
    event = await next_stream_json(response)
    assert event["sequence"] == 3
    assert event["request"]["id"] == continued.request_id
    assert sessions.events.subscriber_count == 1
    await response.body_iterator.aclose()
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_heartbeat_is_blank_and_carries_no_cursor(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)

    async def immediate_timeout(subscription, timeout):
        assert timeout == 15.0
        return None

    monkeypatch.setattr(Subscription, "receive", immediate_timeout)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    await next_stream_json(response)

    heartbeat = await anext(response.body_iterator)

    assert heartbeat == "\n"
    assert all(marker not in heartbeat for marker in ("sequence", "journal", "state"))
    await response.body_iterator.aclose()


def test_performance_stream_rejects_zero_heartbeat_interval() -> None:
    clock = RegistryClock()

    with pytest.raises(ValueError, match="positive"):
        control_app_for_stream(stream_registry(clock), heartbeat_interval=0)


async def test_performance_stream_cancelled_generator_closes_subscription_once(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    close_calls = 0
    original_close = Subscription.close

    def counted_close(subscription) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(subscription)

    monkeypatch.setattr(Subscription, "close", counted_close)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    await next_stream_json(response)
    assert sessions.events.subscriber_count == 1

    with pytest.raises(asyncio.CancelledError):
        await response.body_iterator.athrow(asyncio.CancelledError())

    assert sessions.events.subscriber_count == 0
    assert close_calls == 1


async def test_performance_stream_send_failure_preserves_error_and_closes_once(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    close_calls = 0
    original_close = Subscription.close

    def failing_close(subscription) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(subscription)
        raise RuntimeError("close failed")

    monkeypatch.setattr(Subscription, "close", failing_close)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(app)
    blocked = asyncio.Event()

    async def receive():
        await blocked.wait()
        return {"type": "http.disconnect"}

    async def failing_send(message):
        raise RuntimeError("send failed before body")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/performance/events",
        "raw_path": b"/v1/performance/events",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }

    with pytest.raises(BaseException, match="send failed before body"):
        await response(scope, receive, failing_send)

    assert sessions.events.subscriber_count == 0
    assert close_calls == 1


@pytest.mark.parametrize(
    "after",
    ["true", "false", "-1", "+1", str(MAX_CONTROL_INTEGER + 1)],
)
async def test_performance_stream_rejects_invalid_cursor_with_generic_422(
    after: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(app, "/v1/performance/events", {"after": after})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}
    assert len(response.content) < 100


@pytest.mark.parametrize(
    "started_at",
    ["0", "not-a-date", "2026-01-02T03:00:00"],
)
async def test_performance_stream_rejects_invalid_resume_time_with_generic_422(
    started_at: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(
        app,
        "/v1/performance/events",
        {"after": "0", "pid": "42", "started_at": started_at},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


async def test_performance_stream_rejects_future_cursor_before_subscribing() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions)

    response = await request(
        app,
        "/v1/performance/events",
        {
            "after": "1",
            "pid": "42",
            "started_at": "2026-01-02T03:00:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}
    assert sessions.events.subscriber_count == 0


async def test_performance_stream_rejects_ambiguous_filter_before_subscribing() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    prefixes: dict[str, int] = {}
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-stream-{index}")).public_id[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes[prefix] == 2:
            collision = prefix
            break
    assert collision is not None
    app = control_app_for_stream(sessions)
    before = sessions.events.subscriber_count

    response = await request(
        app,
        "/v1/performance/events",
        [("filter", f"id={collision}")],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Session ID prefix is ambiguous"}
    assert sessions.events.subscriber_count == before


async def test_performance_stream_filters_live_events_server_side() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("raw-selected", provider="vertex"))
    sessions.begin(metadata("raw-other", provider="openai"))
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(
        app,
        filters=["provider=vertex"],
    )
    await next_stream_json(response)

    sessions.begin(metadata("raw-other", provider="openai"))
    selected = sessions.begin(metadata("raw-selected", provider="vertex"))
    cursor = await next_stream_json(response)
    event = await next_stream_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] == 3
    assert set(cursor) == {"process", "sequence", "occurred_at", "type"}
    assert event["session_id"] == selected.public_id
    assert event["request"]["id"] == selected.request_id
    assert "raw-selected" not in json.dumps(event)
    assert "raw-other" not in json.dumps(event)
    await response.body_iterator.aclose()


async def test_performance_stream_filters_queued_events_at_event_time() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    app = control_app_for_stream(sessions)
    response = await performance_stream_response(
        app,
        filters=["provider=vertex"],
    )
    await next_stream_json(response)

    sessions.begin(metadata("transition", provider="openai"))
    matching = sessions.begin(metadata("transition", provider="vertex"))
    cursor = await next_stream_json(response)
    event = await next_stream_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] == 1
    assert event["type"] == "request_started"
    assert event["sequence"] == 2
    assert event["session_id"] == matching.public_id
    assert event["activity"]["provider"] == "vertex"
    await response.body_iterator.aclose()


async def test_performance_stream_replays_filtered_events_after_session_eviction(
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock, inactive_limit=0)
    handle = sessions.begin(metadata("evicted", provider="openai"))
    sessions.finish(handle, "completed")
    assert sessions.snapshots() == []

    async def immediate_timeout(subscription, timeout):
        return None

    monkeypatch.setattr(Subscription, "receive", immediate_timeout)
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)
    response = await performance_stream_response(
        app,
        filters=["session_id=evicted"],
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    started_event = await next_stream_json(response)
    completed_event = await next_stream_json(response)

    assert [started_event["sequence"], completed_event["sequence"]] == [1, 2]
    assert started_event["activity"]["id"] == handle.public_id
    assert completed_event["activity"]["id"] == handle.public_id
    await response.body_iterator.aclose()


async def test_performance_stream_without_after_valid_identity_still_resets() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("existing"))
    app = control_app_for_stream(sessions)

    response = await performance_stream_response(
        app,
        pid="42",
        started_at="2026-01-02T03:00:00Z",
    )
    reset = await next_stream_json(response)

    assert reset["type"] == "reset"
    assert reset["sequence"] == 1
    await response.body_iterator.aclose()


@pytest.mark.parametrize(
    ("pid", "started_at"),
    [
        ("not-an-integer", "2026-01-02T03:00:00Z"),
        ("42", "not-a-date"),
    ],
)
async def test_performance_stream_without_after_rejects_malformed_identity(
    pid: str,
    started_at: str,
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)

    def unexpected_subscription(*args, **kwargs):
        pytest.fail("malformed identity reached subscription")

    monkeypatch.setattr(sessions, "subscribe_performance", unexpected_subscription)
    app = control_app_for_stream(sessions)

    with pytest.raises(HTTPException) as captured:
        await performance_stream_response(
            app,
            pid=pid,
            started_at=started_at,
        )

    assert captured.value.status_code == 422
    assert captured.value.detail == "Invalid performance event request"


@pytest.mark.parametrize(
    "started_at",
    [
        "0001-01-01T00:00:00+14:00",
        "9999-12-31T23:59:59-14:00",
    ],
)
async def test_performance_stream_rejects_utc_overflow_boundaries(
    started_at: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://control") as client:
        response = await client.get(
            "/v1/performance/events",
            params={"after": "0", "pid": "42", "started_at": started_at},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


@pytest.mark.parametrize("pid", ["true", "-1", "0", "+42", "9223372036854775808"])
async def test_performance_stream_rejects_invalid_resume_pid_with_generic_422(
    pid: str,
) -> None:
    clock = RegistryClock()
    app = control_app_for_stream(stream_registry(clock))

    response = await request(
        app,
        "/v1/performance/events",
        {
            "after": "0",
            "pid": pid,
            "started_at": "2026-01-02T03:00:00Z",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


@pytest.mark.parametrize(
    ("key", "first", "second"),
    [
        ("after", "bad", "0"),
        ("after", "0", "bad"),
        ("pid", "bad", "42"),
        ("pid", "42", "bad"),
        ("started_at", "bad", "2026-01-02T03:00:00Z"),
        ("started_at", "2026-01-02T03:00:00Z", "bad"),
    ],
)
async def test_performance_stream_rejects_duplicate_resume_scalars(
    key: str,
    first: str,
    second: str,
    monkeypatch,
) -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)

    def unexpected_subscription(*args, **kwargs):
        pytest.fail("duplicate scalar reached subscription")

    monkeypatch.setattr(sessions, "subscribe_performance", unexpected_subscription)
    app = control_app_for_stream(sessions)
    base = {
        "after": "0",
        "pid": "42",
        "started_at": "2026-01-02T03:00:00Z",
    }
    params = [(name, value) for name, value in base.items() if name != key]
    params.extend(((key, first), (key, second)))

    response = await request(app, "/v1/performance/events", params)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid performance event request"}


async def test_performance_stream_filtered_replay_preserves_global_sequence() -> None:
    clock = RegistryClock()
    sessions = stream_registry(clock)
    sessions.begin(metadata("openai", provider="openai"))
    matching = sessions.begin(metadata("vertex", provider="vertex"))
    started = datetime(2026, 1, 2, 3, tzinfo=UTC)
    app = control_app_for_stream(sessions, started_at=started)

    response = await performance_stream_response(
        app,
        filters=["provider=vertex"],
        after="0",
        pid="42",
        started_at=started.isoformat(),
    )
    cursor = await next_stream_json(response)
    event = await next_stream_json(response)

    assert cursor["type"] == "cursor"
    assert cursor["sequence"] == 1
    assert event["type"] == "request_started"
    assert event["sequence"] == 2
    assert event["session_id"] == matching.public_id
    await response.body_iterator.aclose()

from datetime import UTC, datetime, timedelta, timezone
import json
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
from pydantic import TypeAdapter, ValidationError
import pytest

import claude_code_proxy.control.app as control_app_module
from claude_code_proxy.control.app import create_control_app
from claude_code_proxy.control.schemas import (
    FailureDiagnosticResponse,
    HealthResponse,
    MetricAggregateResponse,
    MetricResponse,
    PerformanceEventResponse,
    PerformanceListResponse,
    PerformanceResetResponse,
    PerformanceStreamEvent,
    ProcessIdentityResponse,
    RequestPerformanceResponse,
    SessionCounts,
    SessionListResponse,
    SessionPerformanceResponse,
    SessionPerformanceViewResponse,
    SessionResponse,
)
from claude_code_proxy.domain.models import ClientIdentity
from claude_code_proxy.failures import (
    FailureCategory,
    FailureDiagnostic,
    FailureStage,
)
from claude_code_proxy.limits import MAX_CONTROL_INTEGER
from claude_code_proxy.observability import (
    SessionMetadata,
    SessionRegistry,
    SessionSnapshot,
)


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
        client_identity=ClientIdentity(
            client_id,
            agent_id,
            parent_agent_id,
        ),
        client_model=client_model,
        upstream_model=upstream_model,
        provider=provider,
        transport=transport,
        effort=effort,
        context_window=context_window,
    )


def registry(
    clock: RegistryClock, inactive_limit: int = 10
) -> SessionRegistry:
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


async def test_session_response_includes_agent_snapshots() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(metadata("session", "agent", "parent"))
    app = create_control_app(
        sessions,
        application_version="1.0",
        pid=123,
    )

    response = await request(app, "/v1/sessions")

    assert response.status_code == 200
    agent = response.json()["sessions"][0]["agents"][0]
    assert agent["id"] == handle.agent_public_id
    assert agent["parent_id"] == handle.parent_agent_public_id
    assert "client_session_id" not in agent
    assert "agent_id" not in agent


async def test_health_counts_roots_not_agent_rows() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    sessions.begin(metadata("session", "first"))
    sessions.begin(metadata("session", "second"))
    app = create_control_app(
        sessions,
        application_version="1.0",
        pid=123,
    )

    response = await request(app, "/v1/health")

    assert response.json()["capabilities"] == ["sessions", "agents"]
    assert response.json()["sessions"] == {"active": 1, "retained": 1}


async def test_health_reports_exact_version_process_time_limit_and_counts() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock, inactive_limit=7)
    sessions.begin(metadata("raw-active"))
    inactive = sessions.begin(metadata("raw-inactive"))
    sessions.finish(inactive, "completed")
    started = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    now = started + timedelta(seconds=12.5)
    app = create_control_app(
        sessions,
        started_at=started,
        application_version="9.8.7",
        pid=4321,
        clock=lambda: now,
    )

    response = await request(app, "/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "protocol_version": 1,
        "application_version": "9.8.7",
        "pid": 4321,
        "started_at": "2026-01-02T03:04:05Z",
        "uptime_seconds": 12.5,
        "capabilities": ["sessions", "agents"],
        "sessions": {"active": 1, "retained": 2},
        "inactive_limit": 7,
    }


async def test_health_clamps_uptime_when_clock_precedes_start() -> None:
    clock = RegistryClock()
    started = datetime(2026, 1, 2, tzinfo=UTC)
    app = create_control_app(
        registry(clock),
        started_at=started,
        application_version="1.0",
        pid=1,
        clock=lambda: started - timedelta(seconds=1),
    )

    response = await request(app, "/v1/health")

    assert response.json()["uptime_seconds"] == 0


async def test_health_uses_installed_distribution_version(monkeypatch) -> None:
    clock = RegistryClock()
    requested_distributions: list[str] = []

    def version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "4.5.6"

    monkeypatch.setattr(control_app_module.metadata, "version", version)
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        pid=1,
        clock=clock.wall_now,
    )

    response = await request(app, "/v1/health")

    assert response.json()["application_version"] == "4.5.6"
    assert requested_distributions == ["anthropic-proxy"]


async def test_control_app_defaults_are_resolved_at_construction(monkeypatch) -> None:
    registry_clock = RegistryClock()
    started = datetime(2026, 4, 5, 6, 7, 8, tzinfo=UTC)
    current = started + timedelta(seconds=3)
    times = iter((started, current))
    requested_distributions: list[str] = []

    monkeypatch.setattr(control_app_module, "_utc_now", lambda: next(times))
    monkeypatch.setattr(control_app_module.os, "getpid", lambda: 2468)

    def version(distribution: str) -> str:
        requested_distributions.append(distribution)
        return "7.8.9"

    monkeypatch.setattr(control_app_module.metadata, "version", version)

    app = create_control_app(registry(registry_clock))
    response = await request(app, "/v1/health")

    assert response.status_code == 200
    assert response.json()["started_at"] == "2026-04-05T06:07:08Z"
    assert response.json()["uptime_seconds"] == 3
    assert response.json()["application_version"] == "7.8.9"
    assert response.json()["pid"] == 2468
    assert requested_distributions == ["anthropic-proxy"]


async def test_sessions_preserve_registry_order_and_one_capture_time() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock)
    first = sessions.begin(metadata("raw-first"))
    registry_clock.advance()
    second = sessions.begin(metadata("raw-second"))
    captured_at = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
    app = create_control_app(
        sessions,
        started_at=captured_at,
        application_version="1.0",
        pid=1,
        clock=lambda: captured_at,
    )

    response = await request(app, "/v1/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["captured_at"] == "2026-02-03T04:05:06Z"
    assert [item["id"] for item in payload["sessions"]] == [
        second.public_id,
        first.public_id,
    ]


@pytest.mark.parametrize(
    ("entry", "expected_name"),
    [
        ("id={id_prefix}", "idle"),
        ("session_id=raw-idle", "idle"),
        ("state=FAILED", "failed"),
        ("provider=VERTEX", "idle"),
        ("transport=CODEX", "active"),
        ("model=CLAUDE-SONNET-5", "idle"),
        ("effort=LOW", "failed"),
    ],
)
async def test_sessions_support_each_filter_key(
    entry: str, expected_name: str
) -> None:
    sessions, ids = filtered_registry()
    value = entry.format(id_prefix=ids["idle"][:12].upper())
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/sessions", [("filter", value)])

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [
        ids[expected_name]
    ]


async def test_session_id_filter_never_returns_raw_id() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    raw_id = "raw-idle"
    response = await request(
        app,
        "/v1/sessions",
        [("filter", f"session_id={raw_id}")],
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [
        ids["idle"]
    ]
    assert raw_id not in response.text


async def test_repeated_filter_key_is_or_and_different_keys_are_and() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    or_response = await request(
        app,
        "/v1/sessions",
        [("filter", "state=idle"), ("filter", "state=failed")],
    )
    and_response = await request(
        app,
        "/v1/sessions",
        [("filter", "transport=litellm"), ("filter", "effort=medium")],
    )

    assert {item["id"] for item in or_response.json()["sessions"]} == {
        ids["idle"],
        ids["failed"],
    }
    assert [item["id"] for item in and_response.json()["sessions"]] == [
        ids["idle"]
    ]


@pytest.mark.parametrize(
    "entry",
    [
        "missing-separator",
        " =value",
        "key=   ",
        "unknown=value",
        "state=idle=extra",
        "state==idle",
    ],
)
async def test_malformed_filter_entries_return_422(entry: str) -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/sessions", [("filter", entry)])

    assert response.status_code == 422


async def test_filter_entry_length_accepts_256_and_rejects_257_characters() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    exactly_256 = "model=" + "x" * 250
    exactly_257 = "model=" + "x" * 251

    accepted = await request(
        app, "/v1/sessions", [("filter", exactly_256)]
    )
    rejected = await request(
        app, "/v1/sessions", [("filter", exactly_257)]
    )

    assert len(exactly_256) == 256
    assert accepted.status_code == 200
    assert accepted.json()["sessions"] == []
    assert len(exactly_257) == 257
    assert rejected.status_code == 422


async def test_invalid_session_id_filter_does_not_echo_raw_value() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    raw_id = "sensitive-session-" + "x" * 240

    response = await request(
        app,
        "/v1/sessions",
        [("filter", f"session_id={raw_id}")],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session filter"}
    assert raw_id not in response.text


async def test_exactly_32_filter_entries_are_accepted() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    entries = [("filter", "state=idle") for _ in range(32)]

    response = await request(app, "/v1/sessions", entries)

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [ids["idle"]]


async def test_filter_whitespace_is_normalized() -> None:
    sessions, ids = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    response = await request(
        app, "/v1/sessions", [("filter", " state = idle ")]
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["sessions"]] == [ids["idle"]]


async def test_more_than_32_filter_entries_returns_422() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )
    entries = [("filter", "state=idle") for _ in range(33)]

    response = await request(app, "/v1/sessions", entries)

    assert response.status_code == 422


async def test_ambiguous_id_prefix_returns_safe_422() -> None:
    registry_clock = RegistryClock()
    sessions = registry(registry_clock, inactive_limit=20)
    seen: set[str] = set()
    collision = None
    for index in range(17):
        prefix = sessions.begin(metadata(f"raw-collision-{index}")).public_id[0]
        if prefix in seen:
            collision = prefix
            break
        seen.add(prefix)
    assert collision is not None
    app = create_control_app(
        sessions,
        started_at=registry_clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(
        app, "/v1/sessions", [("filter", f"id={collision}")]
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Session ID prefix is ambiguous"}
    serialized = response.text
    assert "raw-collision" not in serialized
    assert all(snapshot.id not in serialized for snapshot in sessions.snapshots())


async def test_session_json_excludes_raw_ids_and_internal_fields() -> None:
    sessions, _ = filtered_registry()
    app = create_control_app(
        sessions,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        application_version="1.0",
        pid=1,
    )

    response = await request(app, "/v1/sessions")

    assert response.status_code == 200
    serialized = json.dumps(response.json())
    assert "raw-active" not in serialized
    assert "raw-idle" not in serialized
    assert "raw-failed" not in serialized
    expected_fields = {
        "id",
        "state",
        "active_requests",
        "requests",
        "client_model",
        "model",
        "provider",
        "transport",
        "effort",
        "context_window",
        "first_seen",
        "last_seen",
        "elapsed_seconds",
        "last_result",
        "agents",
    }
    assert all(set(item) == expected_fields for item in response.json()["sessions"])


@pytest.mark.parametrize(
    "path",
    [
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/api/hello",
    ],
)
async def test_control_app_exposes_only_versioned_control_routes(path: str) -> None:
    clock = RegistryClock()
    app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
    )

    response = await request(app, path)

    assert response.status_code == 404


def test_health_schemas_are_frozen_and_protocol_version_is_literal_one() -> None:
    counts = SessionCounts(active=1, retained=2)
    values = {
        "application_version": "1.0",
        "pid": 123,
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "uptime_seconds": 4.0,
        "sessions": counts,
        "inactive_limit": 5,
    }
    health = HealthResponse(**values)

    assert health.protocol_version == 1
    with pytest.raises(ValidationError, match="frozen"):
        counts.active = 2
    with pytest.raises(ValidationError, match="frozen"):
        health.pid = 456
    with pytest.raises(ValidationError, match="Input should be 1"):
        HealthResponse(protocol_version=2, **values)


def test_session_response_validates_snapshot_attributes_and_is_frozen() -> None:
    snapshot = SessionSnapshot(
        id="a" * 64,
        state="idle",
        active_requests=0,
        requests=3,
        client_model="claude-opus",
        model="gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="high",
        context_window=1_000_000,
        first_seen=datetime(2026, 1, 1, tzinfo=UTC),
        last_seen=datetime(2026, 1, 2, tzinfo=UTC),
        elapsed_seconds=2.5,
        last_result="completed",
    )

    response = SessionResponse.model_validate(snapshot)

    assert response.model_dump() == snapshot.__dict__
    assert "client_session_id" not in SessionResponse.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        response.requests = 4


def test_session_list_response_stores_a_frozen_tuple() -> None:
    snapshot = SessionSnapshot(
        id="b" * 64,
        state="active",
        active_requests=1,
        requests=1,
        client_model="claude-sonnet",
        model="gpt-5.6-sol",
        provider="openai",
        transport="codex",
        effort="medium",
        context_window=None,
        first_seen=datetime(2026, 1, 1, tzinfo=UTC),
        last_seen=datetime(2026, 1, 1, tzinfo=UTC),
        elapsed_seconds=1.0,
        last_result=None,
    )
    session = SessionResponse.model_validate(snapshot)

    response = SessionListResponse(
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        sessions=[session],
    )

    assert response.sessions == (session,)
    assert isinstance(response.sessions, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        response.sessions = ()


async def test_control_app_normalizes_aware_datetimes_and_rejects_naive_ones() -> None:
    clock = RegistryClock()
    offset = timezone(timedelta(hours=2))
    app = create_control_app(
        registry(clock),
        started_at=datetime(2026, 1, 1, 2, tzinfo=offset),
        application_version="1.0",
        pid=1,
        clock=lambda: datetime(2026, 1, 1, 3, tzinfo=offset),
    )

    response = await request(app, "/v1/health")

    assert response.json()["started_at"] == "2026-01-01T00:00:00Z"
    assert response.json()["uptime_seconds"] == 3600
    with pytest.raises(ValueError, match="started_at must be timezone-aware"):
        create_control_app(
            registry(clock),
            started_at=datetime(2026, 1, 1),
            application_version="1.0",
            pid=1,
        )

    naive_clock_app = create_control_app(
        registry(clock),
        started_at=clock.wall,
        application_version="1.0",
        pid=1,
        clock=lambda: datetime(2026, 1, 1),
    )
    with pytest.raises(ValueError, match="clock result must be timezone-aware"):
        await request(naive_clock_app, "/v1/health")


def process_payload() -> dict[str, object]:
    return {"pid": 42, "started_at": "2026-01-02T03:00:00Z"}


def failure_payload() -> dict[str, object]:
    return {"category": "internal", "stage": "route", "code": "safe"}


def metric_payload(
    status: str = "observed", value: object = 0
) -> dict[str, object]:
    return {"status": status, "value": value}


def aggregate_payload(value: object = 0) -> dict[str, object]:
    return {
        "value": value,
        "observed_samples": 1,
        "unavailable_samples": 0,
        "not_applicable_samples": 0,
    }


def request_performance_payload() -> dict[str, object]:
    metric = metric_payload()
    return {
        "id": "request-public",
        "session_id": "session-public",
        "operation": "messages",
        "outcome": "completed",
        "started_at": "2026-01-02T03:04:05Z",
        "finished_at": "2026-01-02T03:04:06Z",
        "duration": metric,
        "upstream_duration": metric_payload(),
        "ttft": metric_payload(),
        "input_tokens": metric_payload(),
        "output_tokens": metric_payload(),
        "cache_read_tokens": metric_payload(),
        "cache_creation_tokens": metric_payload(),
        "reasoning_tokens": metric_payload(),
        "tool_calls": metric_payload(),
        "retries": metric_payload(),
        "peak_concurrency": metric_payload(),
        "reasoning_continuation": "not_applicable",
        "failure": None,
    }


def session_performance_payload() -> dict[str, object]:
    request_payload = request_performance_payload()
    aggregate = aggregate_payload()
    return {
        "session_id": "session-public",
        "requests": 1,
        "active_requests": [],
        "recent_requests": [request_payload],
        "outcomes": {"completed": 1},
        "input_tokens": aggregate,
        "output_tokens": aggregate_payload(),
        "cache_read_tokens": aggregate_payload(),
        "cache_creation_tokens": aggregate_payload(),
        "reasoning_tokens": aggregate_payload(),
        "tool_calls": aggregate_payload(),
        "retries": aggregate_payload(),
        "current_concurrency": 0,
        "peak_concurrency": 1,
        "latest_request": request_payload,
    }


def performance_view_payload() -> dict[str, object]:
    return {
        "session": {
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
        },
        "performance": session_performance_payload(),
    }


def performance_list_payload() -> dict[str, object]:
    return {
        "process": process_payload(),
        "captured_at": "2026-01-02T03:04:06Z",
        "cursor": 1,
        "sessions": [performance_view_payload()],
    }


def performance_event_payload() -> dict[str, object]:
    return {
        "process": process_payload(),
        "sequence": 1,
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "completed",
        "session_id": "session-public",
        "request": request_performance_payload(),
        "session": session_performance_payload(),
    }


def performance_reset_payload() -> dict[str, object]:
    snapshot = performance_list_payload()
    return {
        "process": snapshot["process"],
        "sequence": snapshot["cursor"],
        "occurred_at": "2026-01-02T03:04:06Z",
        "type": "reset",
        "snapshot": snapshot,
    }


_PERFORMANCE_SCHEMA_CASES = (
    (ProcessIdentityResponse, process_payload),
    (MetricResponse, metric_payload),
    (MetricAggregateResponse, aggregate_payload),
    (FailureDiagnosticResponse, failure_payload),
    (RequestPerformanceResponse, request_performance_payload),
    (SessionPerformanceResponse, session_performance_payload),
    (SessionPerformanceViewResponse, performance_view_payload),
    (PerformanceListResponse, performance_list_payload),
    (PerformanceEventResponse, performance_event_payload),
    (PerformanceResetResponse, performance_reset_payload),
)


def test_stream_sequences_and_reset_consistency_are_validated() -> None:
    event = performance_event_payload()
    event["sequence"] = 0
    with pytest.raises(ValidationError):
        PerformanceEventResponse.model_validate(event)
    reset = PerformanceResetResponse.model_validate(performance_reset_payload())
    assert reset.sequence == 1
    payload = performance_reset_payload()
    payload["sequence"] = 0
    payload["snapshot"]["cursor"] = 0
    assert PerformanceResetResponse.model_validate(payload).sequence == 0
    for field in ("sequence", "process"):
        payload = performance_reset_payload()
        payload[field] = 2 if field == "sequence" else {
            "pid": 43,
            "started_at": "2026-01-02T03:00:00Z",
        }
        with pytest.raises(ValidationError):
            PerformanceResetResponse.model_validate(payload)


def test_stream_event_alias_discriminates_ordinary_and_reset_events() -> None:
    adapter = TypeAdapter(PerformanceStreamEvent)
    ordinary = adapter.validate_python(performance_event_payload())
    reset = adapter.validate_python(performance_reset_payload())
    assert isinstance(ordinary, PerformanceEventResponse)
    assert isinstance(reset, PerformanceResetResponse)


def test_performance_datetimes_normalize_offsets_and_reject_naive_values() -> None:
    utc = ProcessIdentityResponse.model_validate(
        {"pid": 1, "started_at": "2026-01-02T03:04:05Z"}
    )
    offset = ProcessIdentityResponse.model_validate(
        {"pid": 1, "started_at": "2026-01-02T05:04:05+02:00"}
    )
    assert utc.started_at == offset.started_at
    assert offset.started_at.tzinfo is UTC
    with pytest.raises(ValidationError):
        ProcessIdentityResponse.model_validate(
            {"pid": 1, "started_at": "2026-01-02T03:04:05"}
        )


@pytest.mark.parametrize(
    ("model", "factory"),
    _PERFORMANCE_SCHEMA_CASES,
)
@pytest.mark.parametrize("field", ["prompt", "raw_session_id", "provider_payload"])
def test_performance_schemas_forbid_sensitive_extra_fields(
    model, factory, field: str
) -> None:
    payload = factory()
    payload[field] = "secret"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate(payload)


def test_session_performance_outcomes_are_closed_strict_and_bounded() -> None:
    for outcomes in (
        {"unknown": 1},
        {"completed": True},
        {"completed": "1"},
        {"completed": -1},
        {"completed": MAX_CONTROL_INTEGER + 1},
    ):
        payload = session_performance_payload()
        payload["outcomes"] = outcomes
        with pytest.raises(ValidationError):
            SessionPerformanceResponse.model_validate(payload)


def test_failure_diagnostic_maps_only_safe_structured_fields() -> None:
    diagnostic = FailureDiagnostic(
        FailureCategory.UPSTREAM_HTTP,
        FailureStage.RESPONSE,
        "provider_error",
        provider_code="rate_limit",
        exception_type="ProviderError",
        location="module:function:10",
    )
    response = FailureDiagnosticResponse.model_validate(diagnostic)
    assert response.model_dump(mode="json") == {
        "category": "upstream_http",
        "stage": "response",
        "code": "provider_error",
        "provider_code": "rate_limit",
        "exception_type": "ProviderError",
        "location": "module:function:10",
    }
    assert "message" not in FailureDiagnosticResponse.model_fields


def test_real_registry_capture_converts_without_raw_content() -> None:
    clock = RegistryClock()
    sessions = registry(clock)
    handle = sessions.begin(
        metadata("raw-session-secret"), operation="count_tokens"
    )
    sessions.observer(handle).count_tokens(0)
    sessions.finish(handle, "completed")
    capture = sessions.performance_snapshots()
    wrapped = SimpleNamespace(
        process=SimpleNamespace(pid=42, started_at=clock.wall),
        captured_at=capture.captured_at,
        cursor=capture.cursor,
        sessions=capture.sessions,
    )

    response = PerformanceListResponse.model_validate(wrapped, from_attributes=True)
    dumped = response.model_dump(mode="json")
    serialized = json.dumps(dumped)

    performance = response.sessions[0].performance
    assert performance.recent_requests == (performance.latest_request,)
    recent = dumped["sessions"][0]["performance"]["recent_requests"][0]
    assert recent["input_tokens"] == {"status": "observed", "value": 0}
    input_tokens = dumped["sessions"][0]["performance"]["input_tokens"]
    assert input_tokens["observed_samples"] == 1
    assert handle.public_id in serialized
    secrets = (
        "raw-session-secret",
        "prompt",
        "raw_session_id",
        "provider_payload",
    )
    for secret in secrets:
        assert secret not in serialized


@pytest.mark.parametrize(
    ("model", "factory"),
    _PERFORMANCE_SCHEMA_CASES,
)
def test_performance_schema_models_are_frozen(model, factory) -> None:
    instance = model.model_validate(factory())
    field = next(iter(model.model_fields))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, field, None)
